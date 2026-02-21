import express from 'express';
import fetch from 'node-fetch';
import { createHash } from 'crypto';
import { getConfig, getModelById, getEndpointByType, getSystemPrompt, getModelReasoning, getRedirectedModelId, getModelProvider, getModelCapabilities, getModelFallback, getModelRetryPolicy } from './config.js';
import { logInfo, logDebug, logError, logRequest, logResponse } from './logger.js';
import { transformToAnthropic, getAnthropicHeaders } from './transformers/request-anthropic.js';
import { transformToOpenAI, getOpenAIHeaders } from './transformers/request-openai.js';
import { transformToCommon, getCommonHeaders } from './transformers/request-common.js';
import { AnthropicResponseTransformer } from './transformers/response-anthropic.js';
import { OpenAIResponseTransformer } from './transformers/response-openai.js';
import { getApiKey } from './auth.js';
import { getNextProxyAgent } from './proxy-manager.js';

const router = express.Router();

/**
 * Convert a /v1/responses API result to a /v1/chat/completions-compatible format.
 * Works for non-streaming responses.
 */
function convertResponseToChatCompletion(resp) {
  if (!resp || typeof resp !== 'object') {
    throw new Error('Invalid response object');
  }

  const outputMsg = (resp.output || []).find(o => o.type === 'message');
  const textBlocks = outputMsg?.content?.filter(c => c.type === 'output_text') || [];
  const content = textBlocks.map(c => c.text).join('');

  const chatCompletion = {
    id: resp.id ? resp.id.replace(/^resp_/, 'chatcmpl-') : `chatcmpl-${Date.now()}`,
    object: 'chat.completion',
    created: resp.created_at || Math.floor(Date.now() / 1000),
    model: resp.model || 'unknown-model',
    choices: [
      {
        index: 0,
        message: {
          role: outputMsg?.role || 'assistant',
          content: content || ''
        },
        finish_reason: resp.status === 'completed' ? 'stop' : 'unknown'
      }
    ],
    usage: {
      prompt_tokens: resp.usage?.input_tokens ?? 0,
      completion_tokens: resp.usage?.output_tokens ?? 0,
      total_tokens: resp.usage?.total_tokens ?? 0
    }
  };

  return chatCompletion;
}

function pruneOldImagesInHistory(request) {
  const cleaned = JSON.parse(JSON.stringify(request || {}));
  if (!Array.isArray(cleaned.messages)) return cleaned;

  const isImagePart = (p) => p?.type === 'image_url' || p?.type === 'image' || p?.type === 'input_image' || p?.type === 'output_image';
  const hasImagePart = (msg) => Array.isArray(msg?.content) && msg.content.some(isImagePart);

  let lastUserImageIdx = -1;
  for (let i = cleaned.messages.length - 1; i >= 0; i--) {
    const m = cleaned.messages[i];
    if (m?.role === 'user' && hasImagePart(m)) {
      lastUserImageIdx = i;
      break;
    }
  }

  if (lastUserImageIdx < 0) return cleaned;

  // Keep only the latest user image message and drop all prior message history.
  const latest = cleaned.messages[lastUserImageIdx];
  if (Array.isArray(latest?.content)) {
    latest.content = latest.content.filter(p => {
      if (!isImagePart(p)) return true;
      // Keep only explicit image_url parts; drop ambiguous image wrappers from upstream history.
      return p?.type === 'image_url';
    });
  }

  cleaned.messages = [latest];
  return cleaned;
}

function collectImageFingerprints(request) {
  const fps = [];
  const messages = request?.messages;
  if (!Array.isArray(messages)) return fps;

  for (const msg of messages) {
    if (!Array.isArray(msg?.content)) continue;
    for (const part of msg.content) {
      if (!part || part.type !== 'image_url') continue;
      const imageUrlObj = typeof part.image_url === 'string' ? { url: part.image_url } : (part.image_url || {});
      const url = imageUrlObj.url || '';
      if (url.startsWith('data:')) {
        const b64 = url.split(',')[1] || '';
        try {
          const buf = Buffer.from(b64, 'base64');
          const sha = createHash('sha256').update(buf).digest('hex');
          fps.push({
            source: 'data-url',
            sha256: sha,
            bytes: buf.length
          });
        } catch {
          // ignore malformed data
        }
      } else if (typeof url === 'string' && url) {
        fps.push({
          source: 'url',
          url
        });
      }
    }
  }

  return fps;
}

async function hydrateImageUrlsInRequest(request) {
  const hydrated = JSON.parse(JSON.stringify(request || {}));

  if (!Array.isArray(hydrated.messages)) {
    return hydrated;
  }

  for (const msg of hydrated.messages) {
    if (!Array.isArray(msg?.content)) continue;

    for (const part of msg.content) {
      if (!part || part.type !== 'image_url') continue;

      const imageUrlObj = typeof part.image_url === 'string'
        ? { url: part.image_url }
        : (part.image_url || {});
      const url = imageUrlObj.url;

      if (typeof url !== 'string' || url.startsWith('data:')) {
        part.image_url = imageUrlObj;
        continue;
      }

      if (!/^https?:\/\//i.test(url)) {
        continue;
      }

      try {
        const ctrl = new AbortController();
        const timer = setTimeout(() => ctrl.abort(), 15000);
        const resp = await fetch(url, {
          method: 'GET',
          headers: {
            'user-agent': 'Mozilla/5.0 (compatible; droid2api/1.0)'
          },
          signal: ctrl.signal
        });
        clearTimeout(timer);

        if (!resp.ok) {
          continue;
        }

        const contentType = (resp.headers.get('content-type') || '').split(';')[0].trim();
        if (!contentType.startsWith('image/')) {
          continue;
        }

        const arr = await resp.arrayBuffer();
        const buffer = Buffer.from(arr);
        if (!buffer.length || buffer.length > 8 * 1024 * 1024) {
          continue;
        }

        part.image_url = {
          ...imageUrlObj,
          url: `data:${contentType};base64,${buffer.toString('base64')}`
        };
      } catch (e) {
        logDebug('Image URL hydration skipped', { url, message: e?.message || String(e) });
      }
    }
  }

  return hydrated;
}

function sanitizeForGeminiRetry(request) {
  const cleaned = JSON.parse(JSON.stringify(request || {}));

  if (!Array.isArray(cleaned.messages)) {
    return cleaned;
  }

  const normalizeToolContent = (content) => {
    if (typeof content === 'string') return content;
    if (Array.isArray(content)) return JSON.stringify(content);
    if (content && typeof content === 'object') return JSON.stringify(content);
    return '';
  };

  cleaned.messages = cleaned.messages
    .map(msg => {
      if (!msg || typeof msg !== 'object') return null;
      const next = { ...msg };

      // Keep tool result context by converting tool role into user text.
      if (next.role === 'tool') {
        const toolResultText = normalizeToolContent(next.content);
        next.role = 'user';
        next.content = toolResultText ? `[tool_result]\n${toolResultText}` : '[tool_result]';
        delete next.tool_call_id;
        delete next.name;
      }

      // OpenAI-style tool calls often trigger Gemini thought_signature checks.
      if (next.role === 'assistant' && next.tool_calls) {
        delete next.tool_calls;
      }

      if (Array.isArray(next.content)) {
        next.content = next.content.filter(part => {
          if (!part || typeof part !== 'object') return true;
          const t = String(part.type || '').toLowerCase();

          // Drop function/tool call blocks that do not carry Gemini thought signatures.
          if ((t.includes('function') || t.includes('tool')) && !part.thought_signature) {
            return false;
          }

          // Drop explicit thought/reasoning blocks on retry.
          if (t.includes('thought') || t.includes('reasoning')) {
            return false;
          }

          return true;
        });

        if (next.content.length === 0) {
          next.content = '';
        }
      }

      // Drop empty assistant messages that only carried tool calls.
      if (next.role === 'assistant' && next.content === '') {
        return null;
      }

      return next;
    })
    .filter(Boolean)
    .filter(msg => !(msg.role !== 'user' && msg.content === ''));

  return cleaned;
}

function containsVisionInput(request) {
  const seen = new Set();

  const walk = (node) => {
    if (!node || typeof node !== 'object') return false;
    if (seen.has(node)) return false;
    seen.add(node);

    if (Array.isArray(node)) {
      return node.some(walk);
    }

    const t = String(node.type || '').toLowerCase();
    if (['image', 'image_url', 'input_image', 'output_image'].includes(t)) {
      return true;
    }

    if (node.image_url || node.inlineData || node.inline_data) {
      return true;
    }

    if (node.source && typeof node.source === 'object') {
      const st = String(node.source.type || '').toLowerCase();
      if (st === 'url' || st === 'base64') {
        return true;
      }
      const mediaType = String(node.source.media_type || '').toLowerCase();
      if (mediaType.startsWith('image/')) {
        return true;
      }
    }

    for (const v of Object.values(node)) {
      if (walk(v)) return true;
    }
    return false;
  };

  return walk(request);
}

function isTransientUpstreamError(status, errorText = '') {
  if (status >= 500 && status < 600) return true;
  const t = String(errorText || '').toLowerCase();
  return t.includes('internal_server_error') || t.includes('server had an error');
}

function getRequestPolicy(headers = {}) {
  const retryHeader = String(headers['x-droid-retry'] || '').toLowerCase();
  const fallbackHeader = String(headers['x-droid-fallback'] || '').toLowerCase();
  const normalizeHeader = String(headers['x-droid-error-normalize'] || '').toLowerCase();
  const filterHeader = String(headers['x-droid-filter-placeholder'] || '').toLowerCase();

  return {
    enableRetry: !(retryHeader === 'off' || retryHeader === '0' || retryHeader === 'false'),
    enableFallback: !(fallbackHeader === 'off' || fallbackHeader === '0' || fallbackHeader === 'false'),
    normalizeError: !(normalizeHeader === 'off' || normalizeHeader === '0' || normalizeHeader === 'false'),
    filterPlaceholder: !(filterHeader === 'off' || filterHeader === '0' || filterHeader === 'false')
  };
}

function extractUserText(request) {
  const messages = request?.messages;
  if (!Array.isArray(messages)) return '';
  const chunks = [];
  for (const msg of messages) {
    if (msg?.role !== 'user') continue;
    if (typeof msg.content === 'string') {
      chunks.push(msg.content);
    } else if (Array.isArray(msg.content)) {
      for (const p of msg.content) {
        if (p?.type === 'text' && p.text) chunks.push(p.text);
      }
    }
  }
  return chunks.join('\n');
}

function hasWeatherIntent(request) {
  const t = extractUserText(request).toLowerCase();
  return /(天气|天氣|weather|temperature|气温|氣溫|会不会下雨|會不會下雨|rain)/i.test(t);
}

function inferWeatherLocation(request) {
  const t = extractUserText(request);
  if (!t) return 'Guangzhou';

  if (/广州|廣州/i.test(t)) return 'Guangzhou';
  if (/beijing|北京/i.test(t)) return 'Beijing';
  if (/shanghai|上海/i.test(t)) return 'Shanghai';
  if (/shenzhen|深圳/i.test(t)) return 'Shenzhen';
  if (/hangzhou|杭州/i.test(t)) return 'Hangzhou';

  const cn = t.match(/([\u4e00-\u9fa5]{2,8})(?:天气|天氣)/);
  if (cn?.[1]) return cn[1];
  const en = t.match(/weather\s+(in|for)?\s*([a-zA-Z\-\s]{2,40})/i);
  if (en?.[2]) return en[2].trim();

  return 'Guangzhou';
}

async function fetchWeatherSummary(location = 'Guangzhou') {
  const urls = [
    `https://wttr.in/${encodeURIComponent(location)}?format=%l:+%c+%t+(feels+like+%f),+%w,+%h,+precip:%p`,
    `http://wttr.in/${encodeURIComponent(location)}?format=%l:+%c+%t+(feels+like+%f),+%w,+%h,+precip:%p`
  ];

  for (const url of urls) {
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 12000);
      const resp = await fetch(url, {
        signal: ctrl.signal,
        headers: {
          'user-agent': 'Mozilla/5.0 (compatible; droid2api-weather/1.0)'
        }
      });
      clearTimeout(timer);
      if (!resp.ok) continue;
      const text = (await resp.text()).trim();
      if (text) return text;
    } catch {
      // try next URL
    }
  }

  return null;
}

function isWeatherTransitionText(text = '') {
  const t = String(text || '').trim().toLowerCase();
  if (!t) return false;
  return /^(checking .*(weather|temperature)|let me check.*(weather|temperature)|i('?|’)ll check.*weather|正在查询.*天气|正在查.*天气|我去查一下.*天气|我帮你查一下.*天气|checking guangzhou weather)[.!?。！？]?$/.test(t);
}

function isWeatherNonRealtimeRefusal(text = '') {
  const t = String(text || '').toLowerCase();
  if (!t) return false;
  return /(无法获取实时天气|无法查询实时天气|无法查看实时天气|不能获取实时天气|没有联网查询实时|无法获取实时日期信息|i can't access real-time weather|cannot access real-time weather|no real-time weather)/i.test(t);
}

function getAssistantTextFromPayload(payload) {
  if (!payload || typeof payload !== 'object') return '';

  if (Array.isArray(payload?.choices) && payload.choices[0]?.message) {
    return String(payload.choices[0].message.content || '');
  }

  if (payload.type === 'message' && Array.isArray(payload.content)) {
    return payload.content
      .filter(c => c?.type === 'text')
      .map(c => c.text || '')
      .join('');
  }

  if (Array.isArray(payload.content) && payload.content[0]?.text) {
    return payload.content
      .filter(c => c?.type === 'text')
      .map(c => c.text || '')
      .join('');
  }

  return '';
}

function setAssistantTextInPayload(payload, text) {
  if (!payload || typeof payload !== 'object') return payload;

  if (Array.isArray(payload?.choices) && payload.choices[0]?.message) {
    payload.choices[0].message.content = text;
    return payload;
  }

  if (payload.type === 'message' && Array.isArray(payload.content)) {
    const firstText = payload.content.find(c => c?.type === 'text');
    if (firstText) {
      firstText.text = text;
    } else {
      payload.content.push({ type: 'text', text });
    }
    return payload;
  }

  if (Array.isArray(payload.content)) {
    const firstText = payload.content.find(c => c?.type === 'text');
    if (firstText) {
      firstText.text = text;
    } else {
      payload.content.push({ type: 'text', text });
    }
  }

  return payload;
}

function buildNormalizedError(status, errorText = '', extra = {}) {
  const text = String(errorText || '');
  const low = text.toLowerCase();

  let errorCode = 'UPSTREAM_ERROR';
  if (low.includes('thought_signature')) errorCode = 'GEMINI_THOUGHT_SIGNATURE';
  else if (low.includes('does not support image inputs')) errorCode = 'NO_VISION';
  else if (low.includes('missing required parameter: \"tools[0].name\"')) errorCode = 'TOOLS_SCHEMA_MISMATCH';
  else if (low.includes('invalid value: \"tool\"')) errorCode = 'ROLE_TOOL_NOT_SUPPORTED';
  else if (low.includes('image.source.type: field required')) errorCode = 'IMAGE_SCHEMA_MISMATCH';
  else if (status >= 500) errorCode = 'UPSTREAM_TRANSIENT';

  return {
    error: `Endpoint returned ${status}`,
    details: text,
    error_code: errorCode,
    ...extra
  };
}

function applyPolicyForErrorResponse(res, status, errorText, policy, extra = {}) {
  const body = policy.normalizeError
    ? buildNormalizedError(status, errorText, extra)
    : {
        error: `Endpoint returned ${status}`,
        details: errorText
      };
  return res.status(status).json(body);
}

async function fetchWithPolicy({
  response,
  endpoint,
  fetchOptions,
  transformedRequest,
  model,
  modelId,
  hasVisionInput,
  policy,
  authHeader,
  clientHeaders,
  openaiRequest,
  proxyAgentInfo,
  req,
  res
}) {
  if (response.ok) {
    return { response, ok: true };
  }

  let errorText = await response.text();

  // Retry #1: Gemini thought signature incompatibility
  const shouldGeminiRetry = model.type === 'common'
    && response.status === 400
    && /thought_signature|INVALID_ARGUMENT/i.test(errorText);

  if (shouldGeminiRetry) {
    logInfo('Gemini thought_signature 400 detected, retrying with sanitized request');

    const retriedRequest = sanitizeForGeminiRetry(transformedRequest);
    const retryOptions = {
      ...fetchOptions,
      body: JSON.stringify(retriedRequest)
    };

    response = await fetch(endpoint.base_url, retryOptions);
    logInfo(`Retry response status: ${response.status}`);

    if (response.ok) {
      return { response, ok: true };
    }
    errorText = await response.text();
  }

  // Retry #2: transient upstream 5xx (CPA-like resilience)
  const shouldRetry5xx = policy.enableRetry && isTransientUpstreamError(response.status, errorText);
  if (!response.ok && shouldRetry5xx) {
    const retryPolicy = getModelRetryPolicy(modelId);
    const maxRetries = retryPolicy.retry_count;
    const retryDelays = retryPolicy.delays_ms.length > 0 ? retryPolicy.delays_ms : [600, 1200];

    for (let i = 0; i < maxRetries; i++) {
      const d = retryDelays[Math.min(i, retryDelays.length - 1)];
      logInfo(`Upstream transient error detected, retrying #${i + 1}/${maxRetries} after ${d}ms`);
      await new Promise(r => setTimeout(r, d));
      response = await fetch(endpoint.base_url, fetchOptions);
      logInfo(`Transient retry #${i + 1} status: ${response.status}`);
      if (response.ok) {
        break;
      }
      errorText = await response.text();
    }

    if (response.ok) {
      return { response, ok: true };
    }
  }

  // Retry #3: vision fallback model
  if (!response.ok && hasVisionInput && policy.enableFallback) {
    const fallbackModelId = getModelFallback(modelId);
    if (fallbackModelId && fallbackModelId !== modelId) {
      const fallbackModel = getModelById(fallbackModelId);
      const fallbackEndpoint = fallbackModel ? getEndpointByType(fallbackModel.type) : null;
      if (fallbackModel && fallbackEndpoint) {
        logInfo(`Vision fallback activated: ${modelId} -> ${fallbackModelId}`);

        const fallbackRequestRaw = { ...openaiRequest, model: fallbackModelId };
        let fallbackTransformed;
        let fallbackHeaders;
        const fallbackProvider = getModelProvider(fallbackModelId);

        if (fallbackModel.type === 'anthropic') {
          fallbackTransformed = transformToAnthropic(fallbackRequestRaw);
          const isStreaming = openaiRequest.stream === true;
          fallbackHeaders = getAnthropicHeaders(authHeader, clientHeaders, isStreaming, fallbackModelId, fallbackProvider);
        } else if (fallbackModel.type === 'openai') {
          fallbackTransformed = transformToOpenAI(fallbackRequestRaw);
          fallbackHeaders = getOpenAIHeaders(authHeader, clientHeaders, fallbackProvider);
        } else {
          fallbackTransformed = transformToCommon(fallbackRequestRaw);
          fallbackHeaders = getCommonHeaders(authHeader, clientHeaders, fallbackProvider);
        }

        const fallbackOptions = {
          method: 'POST',
          headers: fallbackHeaders,
          body: JSON.stringify(fallbackTransformed)
        };
        if (proxyAgentInfo?.agent) {
          fallbackOptions.agent = proxyAgentInfo.agent;
        }

        const fallbackResp = await fetch(fallbackEndpoint.base_url, fallbackOptions);
        logInfo(`Vision fallback response status: ${fallbackResp.status}`);
        if (fallbackResp.ok) {
          return { response: fallbackResp, ok: true, usedFallback: fallbackModelId };
        }
        errorText = await fallbackResp.text();
      }
    }
  }

  logError(`Endpoint error after retry: ${response.status}`, new Error(errorText));
  applyPolicyForErrorResponse(res, response.status, errorText, policy, {
    model_id: modelId,
    request_policy: {
      retry: policy.enableRetry,
      fallback: policy.enableFallback
    }
  });
  return { response, ok: false };
}

router.get('/v1/models', (req, res) => {
  logInfo('GET /v1/models');
  
  try {
    const config = getConfig();
    const models = config.models.map(model => ({
      id: model.id,
      object: 'model',
      created: Date.now(),
      owned_by: model.type,
      permission: [],
      root: model.id,
      parent: null,
      capabilities: model.capabilities || {},
      fallback_model: model.fallback_model || null,
      retry_policy: model.retry_policy || null
    }));

    const response = {
      object: 'list',
      data: models
    };

    logResponse(200, null, response);
    res.json(response);
  } catch (error) {
    logError('Error in GET /v1/models', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// 标准 OpenAI 聊天补全处理函数（带格式转换）
async function handleChatCompletions(req, res) {
  logInfo('POST /v1/chat/completions');

  try {
    const openaiRequestRaw = req.body;
    const prunedRequest = pruneOldImagesInHistory(openaiRequestRaw);
    const openaiRequest = await hydrateImageUrlsInRequest(prunedRequest);

    const imageFingerprints = collectImageFingerprints(openaiRequest);
    if (imageFingerprints.length > 0) {
      logInfo(`Image fingerprint count=${imageFingerprints.length}`);
      logDebug('Image fingerprints', imageFingerprints);
    }

    const modelId = getRedirectedModelId(openaiRequest.model);

    if (!modelId) {
      return res.status(400).json({ error: 'model is required' });
    }

    const model = getModelById(modelId);
    if (!model) {
      return res.status(404).json({ error: `Model ${modelId} not found` });
    }

    const modelCapabilities = getModelCapabilities(modelId);
    const hasVisionInput = containsVisionInput(openaiRequest);
    if (hasVisionInput && modelCapabilities.vision === false) {
      return res.status(400).json({
        error: 'Model capability mismatch',
        message: `Model ${modelId} does not support image inputs`,
        code: 'NO_VISION'
      });
    }

    const endpoint = getEndpointByType(model.type);
    if (!endpoint) {
      return res.status(500).json({ error: `Endpoint type ${model.type} not found` });
    }

    logInfo(`Routing to ${model.type} endpoint: ${endpoint.base_url}`);

    // Get API key (will auto-refresh if needed)
    let authHeader;
    try {
      authHeader = await getApiKey(req.headers.authorization);
    } catch (error) {
      logError('Failed to get API key', error);
      return res.status(500).json({ 
        error: 'API key not available',
        message: 'Failed to get or refresh API key. Please check server logs.'
      });
    }

    let transformedRequest;
    let headers;
    const clientHeaders = req.headers;

    // Log received client headers for debugging
    logDebug('Client headers received', {
      'x-factory-client': clientHeaders['x-factory-client'],
      'x-session-id': clientHeaders['x-session-id'],
      'x-assistant-message-id': clientHeaders['x-assistant-message-id'],
      'user-agent': clientHeaders['user-agent']
    });

    // Update request body with redirected model ID before transformation
    const requestWithRedirectedModel = { ...openaiRequest, model: modelId };

    const policy = getRequestPolicy(req.headers || {});

    // Get provider from model config
    const provider = getModelProvider(modelId);

    if (model.type === 'anthropic') {
      transformedRequest = transformToAnthropic(requestWithRedirectedModel);
      const isStreaming = openaiRequest.stream === true;
      headers = getAnthropicHeaders(authHeader, clientHeaders, isStreaming, modelId, provider);
    } else if (model.type === 'openai') {
      transformedRequest = transformToOpenAI(requestWithRedirectedModel);
      headers = getOpenAIHeaders(authHeader, clientHeaders, provider);
    } else if (model.type === 'common') {
      transformedRequest = transformToCommon(requestWithRedirectedModel);
      headers = getCommonHeaders(authHeader, clientHeaders, provider);
    } else {
      return res.status(500).json({ error: `Unknown endpoint type: ${model.type}` });
    }

    logRequest('POST', endpoint.base_url, headers, transformedRequest);

    const proxyAgentInfo = getNextProxyAgent(endpoint.base_url);
    const fetchOptions = {
      method: 'POST',
      headers,
      body: JSON.stringify(transformedRequest)
    };

    if (proxyAgentInfo?.agent) {
      fetchOptions.agent = proxyAgentInfo.agent;
    }

    let response = await fetch(endpoint.base_url, fetchOptions);

    logInfo(`Response status: ${response.status}`);

    const policyResult = await fetchWithPolicy({
      response,
      endpoint,
      fetchOptions,
      transformedRequest,
      model,
      modelId,
      hasVisionInput,
      policy,
      authHeader,
      clientHeaders,
      openaiRequest,
      proxyAgentInfo,
      req,
      res
    });

    if (!policyResult.ok) {
      return;
    }

    response = policyResult.response;

    const isStreaming = transformedRequest.stream === true;

    if (isStreaming) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      // common 类型直接转发，不使用 transformer
      if (model.type === 'common') {
        try {
          let sseBuffer = '';
          const shouldFilterPlaceholder = policy.filterPlaceholder;

          for await (const chunk of response.body) {
            const textChunk = chunk.toString();
            if (!shouldFilterPlaceholder) {
              res.write(chunk);
              continue;
            }

            sseBuffer += textChunk;
            const parts = sseBuffer.split('\n\n');
            sseBuffer = parts.pop() || '';

            for (const evt of parts) {
              const lines = evt.split('\n');
              let dataLine = lines.find(l => l.startsWith('data:'));
              if (!dataLine) {
                res.write(evt + '\n\n');
                continue;
              }

              const raw = dataLine.slice(5).trim();
              if (raw === '[DONE]') {
                res.write(evt + '\n\n');
                continue;
              }

              try {
                const obj = JSON.parse(raw);
                const delta = obj?.choices?.[0]?.delta?.content;
                if (typeof delta === 'string' && /reading the attached image/i.test(delta.trim())) {
                  continue;
                }
              } catch (_) {
                // pass through malformed/non-json chunks
              }

              res.write(evt + '\n\n');
            }
          }

          if (sseBuffer) {
            res.write(sseBuffer);
          }

          res.end();
          logInfo('Stream forwarded (common type)');
        } catch (streamError) {
          logError('Stream error', streamError);
          res.end();
        }
      } else {
        // anthropic 和 openai 类型使用 transformer
        let transformer;
        if (model.type === 'anthropic') {
          transformer = new AnthropicResponseTransformer(modelId, `chatcmpl-${Date.now()}`);
        } else if (model.type === 'openai') {
          transformer = new OpenAIResponseTransformer(modelId, `chatcmpl-${Date.now()}`);
        }

        try {
          for await (const chunk of transformer.transformStream(response.body)) {
            res.write(chunk);
          }
          res.end();
          logInfo('Stream completed');
        } catch (streamError) {
          logError('Stream error', streamError);
          res.end();
        }
      }
    } else {
      const data = await response.json();
      if (model.type === 'openai') {
        try {
          const converted = convertResponseToChatCompletion(data);

          // Auto-heal weather transition text leakage for droid path.
          const maybeText = getAssistantTextFromPayload(converted);
          if (hasWeatherIntent(openaiRequest) && (isWeatherTransitionText(maybeText) || isWeatherNonRealtimeRefusal(maybeText))) {
            const loc = inferWeatherLocation(openaiRequest);
            const weather = await fetchWeatherSummary(loc);
            if (weather) {
              setAssistantTextInPayload(converted, weather);
            }
          }

          logResponse(200, null, converted);
          res.json(converted);
        } catch (e) {
          // 如果转换失败，回退为原始数据
          logResponse(200, null, data);
          res.json(data);
        }
      } else {
        // anthropic/common: 保持现有逻辑，直接转发
        const maybeText = getAssistantTextFromPayload(data);
        if (hasWeatherIntent(openaiRequest) && (isWeatherTransitionText(maybeText) || isWeatherNonRealtimeRefusal(maybeText))) {
          const loc = inferWeatherLocation(openaiRequest);
          const weather = await fetchWeatherSummary(loc);
          if (weather) {
            setAssistantTextInPayload(data, weather);
          }
        }

        logResponse(200, null, data);
        res.json(data);
      }
    }

  } catch (error) {
    logError('Error in /v1/chat/completions', error);
    res.status(500).json({ 
      error: 'Internal server error',
      message: error.message 
    });
  }
}

// 直接转发 OpenAI 请求（不做格式转换）
async function handleDirectResponses(req, res) {
  logInfo('POST /v1/responses');

  try {
    const openaiRequest = req.body;
    const modelId = getRedirectedModelId(openaiRequest.model);

    if (!modelId) {
      return res.status(400).json({ error: 'model is required' });
    }

    const model = getModelById(modelId);
    if (!model) {
      return res.status(404).json({ error: `Model ${modelId} not found` });
    }

    const policy = getRequestPolicy(req.headers || {});

    // 只允许 openai 类型端点
    if (model.type !== 'openai') {
      return applyPolicyForErrorResponse(
        res,
        400,
        `/v1/responses 接口只支持 openai 类型端点，当前模型 ${modelId} 是 ${model.type} 类型`,
        policy,
        {
          error_code: 'ENDPOINT_MODEL_TYPE_MISMATCH',
          endpoint: '/v1/responses',
          model_id: modelId,
          expected_type: 'openai',
          actual_type: model.type
        }
      );
    }

    const endpoint = getEndpointByType(model.type);
    if (!endpoint) {
      return res.status(500).json({ error: `Endpoint type ${model.type} not found` });
    }

    logInfo(`Direct forwarding to ${model.type} endpoint: ${endpoint.base_url}`);

    // Get API key - support client x-api-key for anthropic endpoint
    let authHeader;
    try {
      const clientAuthFromXApiKey = req.headers['x-api-key']
        ? `Bearer ${req.headers['x-api-key']}`
        : null;
      authHeader = await getApiKey(req.headers.authorization || clientAuthFromXApiKey);
    } catch (error) {
      logError('Failed to get API key', error);
      return res.status(500).json({ 
        error: 'API key not available',
        message: 'Failed to get or refresh API key. Please check server logs.'
      });
    }

    const clientHeaders = req.headers;
    
    // Get provider from model config
    const provider = getModelProvider(modelId);
    
    // 获取 headers
    const headers = getOpenAIHeaders(authHeader, clientHeaders, provider);

    // 注入系统提示到 instructions 字段，并更新重定向后的模型ID
    const systemPrompt = getSystemPrompt();
    const modifiedRequest = { ...openaiRequest, model: modelId };
    if (systemPrompt) {
      // 如果已有 instructions，则在前面添加系统提示
      if (modifiedRequest.instructions) {
        modifiedRequest.instructions = systemPrompt + modifiedRequest.instructions;
      } else {
        // 否则直接设置系统提示
        modifiedRequest.instructions = systemPrompt;
      }
    }

    // 处理reasoning字段
    const reasoningLevel = getModelReasoning(modelId);
    if (reasoningLevel === 'auto') {
      // Auto模式：保持原始请求的reasoning字段不变
      // 如果原始请求有reasoning字段就保留，没有就不添加
    } else if (reasoningLevel && ['low', 'medium', 'high', 'xhigh'].includes(reasoningLevel)) {
      modifiedRequest.reasoning = {
        effort: reasoningLevel,
        summary: 'auto'
      };
    } else {
      // 如果配置是off或无效，移除reasoning字段
      delete modifiedRequest.reasoning;
    }

    logRequest('POST', endpoint.base_url, headers, modifiedRequest);

    // 转发修改后的请求
    const proxyAgentInfo = getNextProxyAgent(endpoint.base_url);
    const fetchOptions = {
      method: 'POST',
      headers,
      body: JSON.stringify(modifiedRequest)
    };

    if (proxyAgentInfo?.agent) {
      fetchOptions.agent = proxyAgentInfo.agent;
    }

    let response = await fetch(endpoint.base_url, fetchOptions);

    logInfo(`Response status: ${response.status}`);

    if (!response.ok) {
      let errorText = await response.text();

      const shouldRetry5xx = policy.enableRetry && isTransientUpstreamError(response.status, errorText);
      if (shouldRetry5xx) {
        const retryPolicy = getModelRetryPolicy(modelId);
        const maxRetries = retryPolicy.retry_count;
        const retryDelays = retryPolicy.delays_ms.length > 0 ? retryPolicy.delays_ms : [600, 1200];

        for (let i = 0; i < maxRetries; i++) {
          const d = retryDelays[Math.min(i, retryDelays.length - 1)];
          await new Promise(r => setTimeout(r, d));
          response = await fetch(endpoint.base_url, fetchOptions);
          logInfo(`Direct responses retry #${i + 1} status: ${response.status}`);
          if (response.ok) break;
          errorText = await response.text();
        }
      }

      if (!response.ok) {
        logError(`Endpoint error: ${response.status}`, new Error(errorText));
        return applyPolicyForErrorResponse(res, response.status, errorText, policy, {
          model_id: modelId,
          endpoint: '/v1/responses'
        });
      }
    }

    const isStreaming = openaiRequest.stream === true;

    if (isStreaming) {
      // 直接转发流式响应，不做任何转换
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      try {
        // 直接将原始响应流转发给客户端
        for await (const chunk of response.body) {
          res.write(chunk);
        }
        res.end();
        logInfo('Stream forwarded successfully');
      } catch (streamError) {
        logError('Stream error', streamError);
        res.end();
      }
    } else {
      // 直接转发非流式响应，不做任何转换
      const data = await response.json();
      logResponse(200, null, data);
      res.json(data);
    }

  } catch (error) {
    logError('Error in /v1/responses', error);
    res.status(500).json({ 
      error: 'Internal server error',
      message: error.message 
    });
  }
}

// 直接转发 Anthropic 请求（不做格式转换）
async function handleDirectMessages(req, res) {
  logInfo('POST /v1/messages');

  try {
    const anthropicRequest = req.body;
    const modelId = getRedirectedModelId(anthropicRequest.model);

    if (!modelId) {
      return res.status(400).json({ error: 'model is required' });
    }

    const model = getModelById(modelId);
    if (!model) {
      return res.status(404).json({ error: `Model ${modelId} not found` });
    }

    const policy = getRequestPolicy(req.headers || {});

    // 只允许 anthropic 类型端点
    if (model.type !== 'anthropic') {
      return applyPolicyForErrorResponse(
        res,
        400,
        `/v1/messages 接口只支持 anthropic 类型端点，当前模型 ${modelId} 是 ${model.type} 类型`,
        policy,
        {
          error_code: 'ENDPOINT_MODEL_TYPE_MISMATCH',
          endpoint: '/v1/messages',
          model_id: modelId,
          expected_type: 'anthropic',
          actual_type: model.type
        }
      );
    }

    const endpoint = getEndpointByType(model.type);
    if (!endpoint) {
      return res.status(500).json({ error: `Endpoint type ${model.type} not found` });
    }

    logInfo(`Direct forwarding to ${model.type} endpoint: ${endpoint.base_url}`);

    // Get API key - support client x-api-key for anthropic endpoint
    let authHeader;
    try {
      const clientAuthFromXApiKey = req.headers['x-api-key']
        ? `Bearer ${req.headers['x-api-key']}`
        : null;
      authHeader = await getApiKey(req.headers.authorization || clientAuthFromXApiKey);
    } catch (error) {
      logError('Failed to get API key', error);
      return res.status(500).json({ 
        error: 'API key not available',
        message: 'Failed to get or refresh API key. Please check server logs.'
      });
    }

    const clientHeaders = req.headers;
    
    // Get provider from model config
    const provider = getModelProvider(modelId);
    
    // 获取 headers
    const isStreaming = anthropicRequest.stream === true;
    const headers = getAnthropicHeaders(authHeader, clientHeaders, isStreaming, modelId, provider);

    // 注入系统提示到 system 字段，并更新重定向后的模型ID
    const systemPrompt = getSystemPrompt();
    const modifiedRequest = { ...anthropicRequest, model: modelId };
    if (systemPrompt) {
      if (modifiedRequest.system && Array.isArray(modifiedRequest.system)) {
        // 如果已有 system 数组，则在最前面插入系统提示
        modifiedRequest.system = [
          { type: 'text', text: systemPrompt },
          ...modifiedRequest.system
        ];
      } else {
        // 否则创建新的 system 数组
        modifiedRequest.system = [
          { type: 'text', text: systemPrompt }
        ];
      }
    }

    // 处理thinking字段
    const reasoningLevel = getModelReasoning(modelId);
    if (reasoningLevel === 'auto') {
      // Auto模式：保持原始请求的thinking字段不变
      // 如果原始请求有thinking字段就保留，没有就不添加
    } else if (reasoningLevel && ['low', 'medium', 'high', 'xhigh'].includes(reasoningLevel)) {
      const budgetTokens = {
        'low': 4096,
        'medium': 12288,
        'high': 24576,
        'xhigh': 40960
      };
      
      modifiedRequest.thinking = {
        type: 'enabled',
        budget_tokens: budgetTokens[reasoningLevel]
      };
    } else {
      // 如果配置是off或无效，移除thinking字段
      delete modifiedRequest.thinking;
    }

    logRequest('POST', endpoint.base_url, headers, modifiedRequest);

    // 转发修改后的请求
    const proxyAgentInfo = getNextProxyAgent(endpoint.base_url);
    const fetchOptions = {
      method: 'POST',
      headers,
      body: JSON.stringify(modifiedRequest)
    };

    if (proxyAgentInfo?.agent) {
      fetchOptions.agent = proxyAgentInfo.agent;
    }

    let response = await fetch(endpoint.base_url, fetchOptions);

    logInfo(`Response status: ${response.status}`);

    if (!response.ok) {
      let errorText = await response.text();

      const shouldRetry5xx = policy.enableRetry && isTransientUpstreamError(response.status, errorText);
      if (shouldRetry5xx) {
        const retryPolicy = getModelRetryPolicy(modelId);
        const maxRetries = retryPolicy.retry_count;
        const retryDelays = retryPolicy.delays_ms.length > 0 ? retryPolicy.delays_ms : [600, 1200];

        for (let i = 0; i < maxRetries; i++) {
          const d = retryDelays[Math.min(i, retryDelays.length - 1)];
          await new Promise(r => setTimeout(r, d));
          response = await fetch(endpoint.base_url, fetchOptions);
          logInfo(`Direct messages retry #${i + 1} status: ${response.status}`);
          if (response.ok) break;
          errorText = await response.text();
        }
      }

      if (!response.ok) {
        logError(`Endpoint error: ${response.status}`, new Error(errorText));
        return applyPolicyForErrorResponse(res, response.status, errorText, policy, {
          model_id: modelId,
          endpoint: '/v1/messages'
        });
      }
    }

    if (isStreaming) {
      // 直接转发流式响应，不做任何转换
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      try {
        // 直接将原始响应流转发给客户端
        for await (const chunk of response.body) {
          res.write(chunk);
        }
        res.end();
        logInfo('Stream forwarded successfully');
      } catch (streamError) {
        logError('Stream error', streamError);
        res.end();
      }
    } else {
      // 直接转发非流式响应，不做任何转换
      const data = await response.json();
      logResponse(200, null, data);
      res.json(data);
    }

  } catch (error) {
    logError('Error in /v1/messages', error);
    res.status(500).json({
      error: 'Internal server error',
      message: error.message
    });
  }
}

// 处理 Anthropic count_tokens 请求
async function handleCountTokens(req, res) {
  logInfo('POST /v1/messages/count_tokens');

  try {
    const anthropicRequest = req.body;
    const modelId = getRedirectedModelId(anthropicRequest.model);

    if (!modelId) {
      return res.status(400).json({ error: 'model is required' });
    }

    const model = getModelById(modelId);
    if (!model) {
      return res.status(404).json({ error: `Model ${modelId} not found` });
    }

    // 只允许 anthropic 类型端点
    if (model.type !== 'anthropic') {
      return res.status(400).json({
        error: 'Invalid endpoint type',
        message: `/v1/messages/count_tokens 接口只支持 anthropic 类型端点，当前模型 ${modelId} 是 ${model.type} 类型`
      });
    }

    const endpoint = getEndpointByType('anthropic');
    if (!endpoint) {
      return res.status(500).json({ error: 'Endpoint type anthropic not found' });
    }

    // Get API key
    let authHeader;
    try {
      const clientAuthFromXApiKey = req.headers['x-api-key']
        ? `Bearer ${req.headers['x-api-key']}`
        : null;
      authHeader = await getApiKey(req.headers.authorization || clientAuthFromXApiKey);
    } catch (error) {
      logError('Failed to get API key', error);
      return res.status(500).json({
        error: 'API key not available',
        message: 'Failed to get or refresh API key. Please check server logs.'
      });
    }

    const clientHeaders = req.headers;
    
    // Get provider from model config
    const provider = getModelProvider(modelId);
    
    const headers = getAnthropicHeaders(authHeader, clientHeaders, false, modelId, provider);

    // 构建 count_tokens 端点 URL
    const countTokensUrl = endpoint.base_url.replace('/v1/messages', '/v1/messages/count_tokens');

    // 更新请求体中的模型ID为重定向后的ID
    const modifiedRequest = { ...anthropicRequest, model: modelId };

    logInfo(`Forwarding to count_tokens endpoint: ${countTokensUrl}`);
    logRequest('POST', countTokensUrl, headers, modifiedRequest);

    const proxyAgentInfo = getNextProxyAgent(countTokensUrl);
    const fetchOptions = {
      method: 'POST',
      headers,
      body: JSON.stringify(modifiedRequest)
    };

    if (proxyAgentInfo?.agent) {
      fetchOptions.agent = proxyAgentInfo.agent;
    }

    const response = await fetch(countTokensUrl, fetchOptions);

    logInfo(`Response status: ${response.status}`);

    if (!response.ok) {
      const errorText = await response.text();
      logError(`Count tokens error: ${response.status}`, new Error(errorText));
      return res.status(response.status).json({
        error: `Endpoint returned ${response.status}`,
        details: errorText
      });
    }

    const data = await response.json();
    logResponse(200, null, data);
    res.json(data);

  } catch (error) {
    logError('Error in /v1/messages/count_tokens', error);
    res.status(500).json({
      error: 'Internal server error',
      message: error.message
    });
  }
}

// 注册路由
router.post('/v1/chat/completions', handleChatCompletions);
router.post('/v1/responses', handleDirectResponses);
router.post('/v1/messages', handleDirectMessages);
router.post('/v1/messages/count_tokens', handleCountTokens);

export default router;
