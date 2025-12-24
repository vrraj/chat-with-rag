// Minimal embedded chat client for /chat
// - No sidebar or metrics
// - Preset params via URL query string
// - Uses same ChatRequest schema as main chat UI

(function () {
  function qs(sel) {
    return document.querySelector(sel);
  }

  const historyEl = qs('#embed_chat_history');
  const inputEl = qs('#embed_chat_input');
  const sendBtn = qs('#embed_send_button');

  if (!historyEl || !inputEl || !sendBtn) {
    console.error('Embedded chat: required DOM elements not found');
    return;
  }

  // --- Querystring config ---
  function parseConfigFromQuery() {
    const sp = new URLSearchParams(window.location.search || '');
    const cfg = {};
    sp.forEach((v, k) => {
      cfg[k] = v;
    });
    return cfg;
  }

  const rawConfig = parseConfigFromQuery();

  function asBool(val, def) {
    if (val == null) return def;
    const s = String(val).toLowerCase();
    if (s === 'true' || s === '1' || s === 'yes') return true;
    if (s === 'false' || s === '0' || s === 'no') return false;
    return def;
  }

  function asNum(val) {
    if (val == null) return null;
    const n = Number(val);
    return Number.isFinite(n) ? n : null;
  }

  const embedConfig = (function buildEmbedConfig() {
    const c = rawConfig;
    const out = {};

    // Retrieval
    out.top_k = asNum(c.top_k);
    out.score_threshold = asNum(c.score_threshold);

    // Summarizer / history
    out.chat_history_window_turns = asNum(c.chat_history_window_turns);
    out.raw_tail_turns = asNum(c.raw_tail_turns);
    out.summarizer_max_input_tokens = asNum(c.summarizer_max_input_tokens);
    out.summarizer_max_output_tokens = asNum(c.summarizer_max_output_tokens);

    // Inference
    out.temperature = asNum(c.temperature);
    out.top_p = asNum(c.top_p);
    out.max_output_tokens = asNum(c.max_output_tokens);

    // Query rewrite
    if (c.enable_query_rewrite != null) {
      out.enable_query_rewrite = asBool(c.enable_query_rewrite, null);
    }
    out.rewrite_confidence_threshold = asNum(c.rewrite_confidence_threshold);
    out.rewrite_tail_turns = asNum(c.rewrite_tail_turns);

    // Tools
    if (c.use_tools != null) {
      out.use_tools = asBool(c.use_tools, false);
    }

    // Provider/model overrides
    ['inference', 'rewrite', 'summary', 'rerank'].forEach((stage) => {
      const pKey = stage + '_provider';
      const mKey = stage + '_model';
      if (c[pKey] != null) out[pKey] = String(c[pKey]);
      if (c[mKey] != null) out[mKey] = String(c[mKey]);
    });

    // Show processing steps: default false for embed, can be overridden
    out.show_processing_steps = asBool(c.show_processing_steps, false);

    // Explicit conversation id / namespace
    out.conversation_id = c.conversation_id || c.namespace || null;

    // Optional mode tag for observability
    out.mode = c.mode || 'embed';

    return out;
  })();

  // --- ID helpers ---
  function generateId8() {
    try {
      if (window.crypto && typeof window.crypto.randomUUID === 'function') {
        return window.crypto.randomUUID().slice(0, 8);
      }
      if (window.crypto && window.crypto.getRandomValues) {
        const arr = new Uint32Array(2);
        window.crypto.getRandomValues(arr);
        return (arr[0].toString(16) + arr[1].toString(16)).slice(0, 8);
      }
    } catch (_) {}
    return Math.random().toString(36).slice(2, 10);
  }

  let conversationId = embedConfig.conversation_id || null;
  if (!conversationId) {
    try {
      const key = 'conversation_id_embed';
      conversationId = sessionStorage.getItem(key);
      if (!conversationId) {
        conversationId = generateId8();
        sessionStorage.setItem(key, conversationId);
      }
    } catch (e) {
      conversationId = generateId8();
    }
  }

  // --- Rendering helpers ---
  function appendMessage(role, text) {
    const wrapper = document.createElement('div');
    wrapper.className = 'msg ' + role;

    const badge = document.createElement('span');
    badge.className = 'badge';
    badge.textContent = role === 'user' ? 'User' : 'Assistant';

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;

    wrapper.appendChild(badge);
    wrapper.appendChild(bubble);
    historyEl.appendChild(wrapper);
    historyEl.scrollTop = historyEl.scrollHeight;

    return bubble;
  }

  function collectHistory() {
    const out = [];
    historyEl.querySelectorAll('.msg').forEach((node) => {
      const role = node.classList.contains('user') ? 'user' : 'assistant';
      const bubble = node.querySelector('.bubble');
      if (bubble) {
        out.push({ role: role, content: bubble.textContent || '' });
      }
    });
    return out;
  }

  function buildParams(queryId) {
    const p = {};

    // Copy only non-null fields from embedConfig
    Object.keys(embedConfig).forEach((k) => {
      const v = embedConfig[k];
      if (v !== null && v !== undefined) {
        p[k] = v;
      }
    });

    p.query_id = queryId;
    p.conversation_id = conversationId;

    return p;
  }

  // --- Streaming visualizer (optional) ---
  // Lightweight SSE consumer for processing stages. This is enabled only when
  // embedConfig.show_processing_steps is true. It reuses the assistant bubble
  // as a compact visualizer, similar to setupStageStreaming in chat.js.
  window.__STAGE_STREAMS = window.__STAGE_STREAMS || new Map();

  function setupEmbedStageStreaming(queryId, bubbleEl) {
    if (!queryId || !embedConfig.show_processing_steps) return;

    try {
      // Close any existing stream for this queryId (safety)
      const prev = window.__STAGE_STREAMS.get(queryId);
      if (prev && typeof prev.close === 'function') prev.close();
      window.__STAGE_STREAMS.delete(queryId);
    } catch (_) {}

    try {
      const es = new EventSource(`/chat/stream/stages?query_id=${encodeURIComponent(queryId)}`);
      window.__STAGE_STREAMS.set(queryId, es);

      const resolveBubble = () => {
        if (bubbleEl && document.body.contains(bubbleEl)) return bubbleEl;
        const nodes = document.querySelectorAll('#embed_chat_history .msg.assistant .bubble');
        return nodes[nodes.length - 1] || null;
      };

      let bubble = resolveBubble();
      let currentText = (bubble?.textContent || 'Processing');
      let finished = false;
      const seenStages = new Set();

      const closeAndForget = () => {
        if (finished) return;
        finished = true;
        try { es.close(); } catch (_) {}
        try { window.__STAGE_STREAMS.delete(queryId); } catch (_) {}
      };

      es.onmessage = (e) => {
        if (finished) return;
        try {
          const payload = JSON.parse(e.data);
          if (payload && (payload.type === 'final' || payload.stage === 'Final Answer')) {
            bubble = resolveBubble();
            if (bubble) {
              const finalContent = payload.text || payload.finalContent || payload.response || payload.answer || '';
              if (finalContent) bubble.textContent = finalContent;
            }
            closeAndForget();
            return;
          }
          if (!payload || payload.type !== 'stage') return;

          bubble = resolveBubble();
          if (payload.final === true || payload.stage === 'Done') {
            if (bubble && typeof payload.finalContent === 'string' && payload.finalContent.length > 0) {
              bubble.textContent = payload.finalContent;
            }
            closeAndForget();
            return;
          }

          const stage = payload.stage;
          if (!stage || !bubble) return;
          if (seenStages.has(stage)) return;
          seenStages.add(stage);
          currentText = `${currentText} --> ${stage}`;
          bubble.textContent = currentText;
        } catch (_) {
          // ignore parse errors
        }
      };

      es.onerror = () => {
        if (!finished) {
          try { es.close(); } catch (_) {}
          try { window.__STAGE_STREAMS.delete(queryId); } catch (_) {}
        }
      };
    } catch (_) {
      // ignore streaming errors; embed UI should still function
    }
  }

  // --- Send logic ---
  async function sendMessage() {
    const text = (inputEl.value || '').trim();
    if (!text) return;

    const queryId = generateId8();

    // User bubble
    appendMessage('user', text);

    // Assistant placeholder
    const placeholderBubble = appendMessage('assistant', 'Processing');

    // Start streaming visualizer (if enabled via config)
    setupEmbedStageStreaming(queryId, placeholderBubble);

    // Build payload
    const payload = {
      message: text,
      use_web_search: false,
      history: collectHistory(),
      params: buildParams(queryId),
    };

    // Disable input while request is in-flight
    inputEl.value = '';
    inputEl.disabled = true;
    sendBtn.disabled = true;
    inputEl.setAttribute('aria-busy', 'true');

    try {
      const resp = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        throw new Error('HTTP ' + resp.status);
      }

      const data = await resp.json();
      const answer = data.answer || data.response || '[No answer field in response]';
      placeholderBubble.textContent = answer;
    } catch (err) {
      console.error('Embedded chat: request failed', err);
      placeholderBubble.textContent = 'Error: unable to fetch answer.';
    } finally {
      inputEl.disabled = false;
      sendBtn.disabled = false;
      inputEl.removeAttribute('aria-busy');
      inputEl.focus();
    }
  }

  // --- Wiring ---
  sendBtn.addEventListener('click', (e) => {
    e.preventDefault();
    sendMessage();
  });

  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
})();
