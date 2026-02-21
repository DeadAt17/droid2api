import { logDebug } from '../logger.js';
import { getSystemPrompt, getModelReasoning, getUserAgent } from '../config.js';

export function transformToOpenAI(openaiRequest) {
  logDebug('Transforming OpenAI request to target OpenAI format');
  
  const targetRequest = {
    model: openaiRequest.model,
    input: [],
    store: false
  };

  // Only add stream parameter if explicitly provided by client
  if (openaiRequest.stream !== undefined) {
    targetRequest.stream = openaiRequest.stream;
  }

  // Transform max_tokens to max_output_tokens
  if (openaiRequest.max_tokens) {
    targetRequest.max_output_tokens = openaiRequest.max_tokens;
  } else if (openaiRequest.max_completion_tokens) {
    targetRequest.max_output_tokens = openaiRequest.max_completion_tokens;
  }

  // Transform messages to input
  if (openaiRequest.messages && Array.isArray(openaiRequest.messages)) {
    for (const msg of openaiRequest.messages) {
      // Responses API only accepts assistant/system/developer/user roles.
      // Convert OpenAI chat tool role into user text payload to preserve context.
      let normalizedRole = msg.role;
      let normalizedContent = msg.content;

      if (normalizedRole === 'tool') {
        normalizedRole = 'user';
        if (typeof normalizedContent !== 'string') {
          try {
            normalizedContent = JSON.stringify(normalizedContent ?? {});
          } catch {
            normalizedContent = String(normalizedContent ?? '');
          }
        }
        normalizedContent = normalizedContent ? `[tool_result]\n${normalizedContent}` : '[tool_result]';
      }

      if (!['assistant', 'system', 'developer', 'user'].includes(normalizedRole)) {
        normalizedRole = 'user';
      }

      const inputMsg = {
        role: normalizedRole,
        content: []
      };

      // Determine content type based on role
      // user/system/developer use input_*, assistant uses output_*
      const textType = normalizedRole === 'assistant' ? 'output_text' : 'input_text';
      const imageType = normalizedRole === 'assistant' ? 'output_image' : 'input_image';

      if (typeof normalizedContent === 'string') {
        if (normalizedContent.trim() !== '') {
          inputMsg.content.push({
            type: textType,
            text: normalizedContent
          });
        }
      } else if (Array.isArray(normalizedContent)) {
        for (const part of normalizedContent) {
          if (part.type === 'text') {
            const txt = part.text || '';
            if (txt.trim() === '') continue;
            inputMsg.content.push({
              type: textType,
              text: txt
            });
          } else if (part.type === 'image_url') {
            const imageObj = typeof part.image_url === 'string'
              ? { url: part.image_url }
              : (part.image_url || {});
            const imageUrl = imageObj.url || '';
            if (imageUrl) {
              inputMsg.content.push({
                type: imageType,
                // Responses API expects image_url as string (not object)
                image_url: imageUrl
              });
            }
          } else {
            // Pass through other types as-is
            inputMsg.content.push(part);
          }
        }
      }

      // Drop empty assistant messages that usually only carried tool calls.
      if (normalizedRole === 'assistant' && inputMsg.content.length === 0) {
        continue;
      }

      if (inputMsg.content.length === 0) {
        inputMsg.content.push({
          type: textType,
          text: '[empty]'
        });
      }

      targetRequest.input.push(inputMsg);
    }
  }

  // Transform tools if present
  // OpenAI Chat format: {type:'function', function:{name,description,parameters,strict}}
  // Responses API format: {type:'function', name, description, parameters, strict}
  if (openaiRequest.tools && Array.isArray(openaiRequest.tools)) {
    targetRequest.tools = openaiRequest.tools.map(tool => {
      if (tool?.type === 'function' && tool.function) {
        return {
          type: 'function',
          name: tool.function.name,
          description: tool.function.description,
          parameters: tool.function.parameters || {},
          strict: tool.function.strict ?? false
        };
      }

      return {
        ...tool,
        strict: tool?.strict ?? false
      };
    });
  }

  // Extract system message as instructions and prepend system prompt
  const systemPrompt = getSystemPrompt();
  const systemMessage = openaiRequest.messages?.find(m => m.role === 'system');
  
  if (systemMessage) {
    let userInstructions = '';
    if (typeof systemMessage.content === 'string') {
      userInstructions = systemMessage.content;
    } else if (Array.isArray(systemMessage.content)) {
      userInstructions = systemMessage.content
        .filter(p => p.type === 'text')
        .map(p => p.text)
        .join('\n');
    }
    targetRequest.instructions = systemPrompt + userInstructions;
    targetRequest.input = targetRequest.input.filter(m => m.role !== 'system');
  } else if (systemPrompt) {
    // If no user-provided system message, just add the system prompt
    targetRequest.instructions = systemPrompt;
  }

  // Handle reasoning field based on model configuration
  const reasoningLevel = getModelReasoning(openaiRequest.model);
  if (reasoningLevel === 'auto') {
    // Auto mode: preserve original request's reasoning field exactly as-is
    if (openaiRequest.reasoning !== undefined) {
      targetRequest.reasoning = openaiRequest.reasoning;
    }
    // If original request has no reasoning field, don't add one
  } else if (reasoningLevel && ['low', 'medium', 'high', 'xhigh'].includes(reasoningLevel)) {
    // Specific level: override with model configuration
    targetRequest.reasoning = {
      effort: reasoningLevel,
      summary: 'auto'
    };
  } else {
    // Off or invalid: explicitly remove reasoning field
    // This ensures any reasoning field from the original request is deleted
    delete targetRequest.reasoning;
  }

  // Pass through other parameters
  if (openaiRequest.temperature !== undefined) {
    targetRequest.temperature = openaiRequest.temperature;
  }
  if (openaiRequest.top_p !== undefined) {
    targetRequest.top_p = openaiRequest.top_p;
  }
  if (openaiRequest.presence_penalty !== undefined) {
    targetRequest.presence_penalty = openaiRequest.presence_penalty;
  }
  if (openaiRequest.frequency_penalty !== undefined) {
    targetRequest.frequency_penalty = openaiRequest.frequency_penalty;
  }
  if (openaiRequest.parallel_tool_calls !== undefined) {
    targetRequest.parallel_tool_calls = openaiRequest.parallel_tool_calls;
  }

  logDebug('Transformed target OpenAI request', targetRequest);
  return targetRequest;
}

export function getOpenAIHeaders(authHeader, clientHeaders = {}, provider = 'openai') {
  // Generate unique IDs if not provided
  const sessionId = clientHeaders['x-session-id'] || generateUUID();
  const messageId = clientHeaders['x-assistant-message-id'] || generateUUID();
  
  const headers = {
    'content-type': 'application/json',
    'authorization': authHeader || '',
    'x-api-provider': provider,
    'x-factory-client': 'cli',
    'x-session-id': sessionId,
    'x-assistant-message-id': messageId,
    'user-agent': getUserAgent(),
    'connection': 'keep-alive'
  };

  // Pass through Stainless SDK headers with defaults
  const stainlessDefaults = {
    'x-stainless-arch': 'x64',
    'x-stainless-lang': 'js',
    'x-stainless-os': 'MacOS',
    'x-stainless-runtime': 'node',
    'x-stainless-retry-count': '0',
    'x-stainless-package-version': '5.23.2',
    'x-stainless-runtime-version': 'v24.3.0'
  };

  // Copy Stainless headers from client or use defaults
  Object.keys(stainlessDefaults).forEach(header => {
    headers[header] = clientHeaders[header] || stainlessDefaults[header];
  });

  return headers;
}

function generateUUID() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0;
    const v = c == 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}
