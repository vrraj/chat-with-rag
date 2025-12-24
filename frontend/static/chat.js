// Chat UI logic
// - Enter/Shift+Enter send behavior
// - Collects params/history and calls backend /chat
// - Replaces placeholder with backend answer; updates metrics
// - Disables input during request; simple error toast on failure

(function () {
  const qs = (s) => document.querySelector(s);
  const chatHistory = qs('#chat_history');
  const input = qs('#chat_input');
  const sendBtn = qs('#send_button');
  const helpLink = qs('#tools_help_link');
  const helpPopup = qs('#tools_help_popup');
  const subsectionHelpLink = qs('#rewrite_subsection_help_link');
  const subsectionPopup = qs('#rewrite_subsection_popup');
  const rewriteHelpLink = qs('#rewrite_help_link');
  const rewriteHelpPopup = qs('#rewrite_help_popup');

  // --- Model configuration (providers/models per stage) ---
  const changeModelsBtn = qs('#change_models_btn');
  const modelsModal = document.getElementById('models_modal');
  const modelsModalClose = document.getElementById('models_modal_close');
  const modelsModalSave = document.getElementById('models_modal_save');
  const modelsModalCancel = document.getElementById('models_modal_cancel');

  const infProvSel = document.getElementById('inference_provider_select');
  const infModelSel = document.getElementById('inference_model_select');
  const rwProvSel = document.getElementById('rewrite_provider_select');
  const rwModelSel = document.getElementById('rewrite_model_select');
  const sumProvSel = document.getElementById('summary_provider_select');
  const sumModelSel = document.getElementById('summary_model_select');
  const rrProvSel = document.getElementById('rerank_provider_select');
  const rrModelSel = document.getElementById('rerank_model_select');

  // Providers and models are kept client-side for now (can be moved to backend later).
  // Restrict to OpenAI and Gemini for now.
  const PROVIDERS = ['openai', 'gemini'];

  const MODELS_BY_STAGE = {
    inference: {
      openai: ['gpt-4o-mini', 'gpt-5-nano'],
      gemini: ['models/gemini-2.5-flash-lite', 'gemini-2.5-flash', 'gemini-3-flash-preview'],
    },
    rewrite: {
      openai: ['gpt-4o-mini', 'gpt-5-nano'],
      gemini: ['models/gemini-2.5-flash-lite', 'gemini-2.5-flash', 'gemini-3-flash-preview'],
    },
    summary: {
      openai: ['gpt-4o-mini', 'gpt-5-nano'],
      gemini: ['models/gemini-2.5-flash-lite', 'gemini-2.5-flash', 'gemini-3-flash-preview'],
    },
    rerank: {
      openai: ['gpt-4o-mini', 'gpt-5-nano'],
      gemini: ['models/gemini-2.5-flash-lite', 'gemini-2.5-flash', 'gemini-3-flash-preview'],
    },
  };

  // Current selections (defaults will be reconciled with backend config / labels).
  const stageModelConfig = {
    inference: { provider: 'openai', model: 'gpt-4o' },
    rewrite:   { provider: 'openai', model: 'gpt-4o' },
    summary:   { provider: 'openai', model: 'gpt-4o' },
    rerank:    { provider: 'openai', model: 'gpt-4o-mini' },
  };

  function _populateProviderSelect(sel) {
    if (!sel) return;
    sel.innerHTML = '';
    PROVIDERS.forEach((p) => {
      const opt = document.createElement('option');
      opt.value = p;
      opt.textContent = p;
      sel.appendChild(opt);
    });
  }

  function _populateModelSelect(stage, provider, sel) {
    if (!sel) return;
    sel.innerHTML = '';
    const byProv = MODELS_BY_STAGE[stage] || {};
    const models = byProv[provider] || [];
    models.forEach((m) => {
      const opt = document.createElement('option');
      opt.value = m;
      opt.textContent = m;
      sel.appendChild(opt);
    });
  }

  function initModelsModalFromConfig() {
    try {
      // Seed from visible labels if present (keeps backend /api/config authoritative).
      const labInf = document.getElementById('model_inference');
      const labRw = document.getElementById('model_query_rewrite');
      const labSum = document.getElementById('model_summarizer');
      const labRr = document.getElementById('model_reranker');

      if (labInf && labInf.textContent && labInf.textContent !== 'Not Found') {
        stageModelConfig.inference.model = labInf.textContent.trim();
      }
      if (labRw && labRw.textContent && labRw.textContent !== 'Not Found') {
        stageModelConfig.rewrite.model = labRw.textContent.trim();
      }
      if (labSum && labSum.textContent && labSum.textContent !== 'Not Found') {
        stageModelConfig.summary.model = labSum.textContent.trim();
      }
      if (labRr && labRr.textContent && labRr.textContent !== 'Not Found') {
        stageModelConfig.rerank.model = labRr.textContent.trim();
      }

      // Populate provider selects
      [infProvSel, rwProvSel, sumProvSel, rrProvSel].forEach(_populateProviderSelect);

      // Set initial provider selections
      if (infProvSel) infProvSel.value = stageModelConfig.inference.provider;
      if (rwProvSel) rwProvSel.value = stageModelConfig.rewrite.provider;
      if (sumProvSel) sumProvSel.value = stageModelConfig.summary.provider;
      if (rrProvSel) rrProvSel.value = stageModelConfig.rerank.provider;

      // Populate model selects based on current provider+stage
      _populateModelSelect('inference', stageModelConfig.inference.provider, infModelSel);
      _populateModelSelect('rewrite', stageModelConfig.rewrite.provider, rwModelSel);
      _populateModelSelect('summary', stageModelConfig.summary.provider, sumModelSel);
      _populateModelSelect('rerank', stageModelConfig.rerank.provider, rrModelSel);

      // Try to select current model if present; else fall back to first option
      if (infModelSel) {
        infModelSel.value = stageModelConfig.inference.model;
        if (!infModelSel.value && infModelSel.options.length) {
          stageModelConfig.inference.model = infModelSel.options[0].value;
          infModelSel.value = stageModelConfig.inference.model;
        }
      }
      if (rwModelSel) {
        rwModelSel.value = stageModelConfig.rewrite.model;
        if (!rwModelSel.value && rwModelSel.options.length) {
          stageModelConfig.rewrite.model = rwModelSel.options[0].value;
          rwModelSel.value = stageModelConfig.rewrite.model;
        }
      }
      if (sumModelSel) {
        sumModelSel.value = stageModelConfig.summary.model;
        if (!sumModelSel.value && sumModelSel.options.length) {
          stageModelConfig.summary.model = sumModelSel.options[0].value;
          sumModelSel.value = stageModelConfig.summary.model;
        }
      }
      if (rrModelSel) {
        rrModelSel.value = stageModelConfig.rerank.model;
        if (!rrModelSel.value && rrModelSel.options.length) {
          stageModelConfig.rerank.model = rrModelSel.options[0].value;
          rrModelSel.value = stageModelConfig.rerank.model;
        }
      }
    } catch (e) {
      console.debug('Failed to initialize models modal', e);
    }
  }

  function openModelsModal() {
    if (!modelsModal) return;
    if (!modelsModal.style.display || modelsModal.style.display === 'none') {
      initModelsModalFromConfig();
    }
    modelsModal.style.display = 'block';
  }

  // Expose as a best-effort global so HTML can call it directly if needed.
  try {
    window.__openModelsModal = openModelsModal;
  } catch (_) {}

  function closeModelsModal() {
    if (!modelsModal) return;
    modelsModal.style.display = 'none';
  }

  function wireModelsModalEvents() {
    if (changeModelsBtn) {
      changeModelsBtn.addEventListener('click', (e) => {
        e.preventDefault();
        openModelsModal();
      });
    }
    if (modelsModalClose) {
      modelsModalClose.addEventListener('click', (e) => {
        e.preventDefault();
        closeModelsModal();
      });
    }
    if (modelsModalCancel) {
      modelsModalCancel.addEventListener('click', (e) => {
        e.preventDefault();
        closeModelsModal();
      });
    }

    // Provider change -> refresh models
    if (infProvSel) {
      infProvSel.addEventListener('change', () => {
        const prov = infProvSel.value || 'openai';
        stageModelConfig.inference.provider = prov;
        _populateModelSelect('inference', prov, infModelSel);
      });
    }
    if (rwProvSel) {
      rwProvSel.addEventListener('change', () => {
        const prov = rwProvSel.value || 'openai';
        stageModelConfig.rewrite.provider = prov;
        _populateModelSelect('rewrite', prov, rwModelSel);
      });
    }
    if (sumProvSel) {
      sumProvSel.addEventListener('change', () => {
        const prov = sumProvSel.value || 'openai';
        stageModelConfig.summary.provider = prov;
        _populateModelSelect('summary', prov, sumModelSel);
      });
    }
    if (rrProvSel) {
      rrProvSel.addEventListener('change', () => {
        const prov = rrProvSel.value || 'openai';
        stageModelConfig.rerank.provider = prov;
        _populateModelSelect('rerank', prov, rrModelSel);
      });
    }

    if (modelsModalSave) {
      modelsModalSave.addEventListener('click', (e) => {
        e.preventDefault();
        try {
          // Snapshot selections back into config
          if (infProvSel) stageModelConfig.inference.provider = infProvSel.value || 'openai';
          if (infModelSel) stageModelConfig.inference.model = infModelSel.value || stageModelConfig.inference.model;
          if (rwProvSel) stageModelConfig.rewrite.provider = rwProvSel.value || 'openai';
          if (rwModelSel) stageModelConfig.rewrite.model = rwModelSel.value || stageModelConfig.rewrite.model;
          if (sumProvSel) stageModelConfig.summary.provider = sumProvSel.value || 'openai';
          if (sumModelSel) stageModelConfig.summary.model = sumModelSel.value || stageModelConfig.summary.model;
          if (rrProvSel) stageModelConfig.rerank.provider = rrProvSel.value || 'openai';
          if (rrModelSel) stageModelConfig.rerank.model = rrModelSel.value || stageModelConfig.rerank.model;

          // Update visible labels under "Models Used"
          const labInf = document.getElementById('model_inference');
          const labRw = document.getElementById('model_query_rewrite');
          const labSum = document.getElementById('model_summarizer');
          const labRr = document.getElementById('model_reranker');
          if (labInf) labInf.textContent = stageModelConfig.inference.model;
          if (labRw) labRw.textContent = stageModelConfig.rewrite.model;
          if (labSum) labSum.textContent = stageModelConfig.summary.model;
          if (labRr) labRr.textContent = stageModelConfig.rerank.model;
        } catch (err) {
          console.debug('Failed to apply model config', err);
        }
        closeModelsModal();
      });
    }

    // Close modal on ESC
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') closeModelsModal();
    });
  }

  // Static tool help metadata for UI only (not sent to backend/model)
  const TOOL_HELP = [
    {
      name: 'get_nearby_airports',
      description: 'Find the closest/nearest airport(s) to a place or coordinates. Defaults to commercial airports.',
      examples: [
        'closest airport to Mount Whitney',
        'airports near Kilimanjaro',
      ],
    },
    {
      name: 'get_weather',
      description: 'Fetch current, high/low, and average temperatures for a location.',
      examples: [
        'forecast for Boston',
        'get weather for Bangalore, India',
      ],
    },
    {
      name: 'web_search',
      description: 'Search the web to gather additional context.',
      examples: [
        'search web for Mount Whitney elevation',
        'search web for top mountains and peaks',
      ],
    },
  ];

  function truncate(text, max = 140) {
    if (!text) return '';
    return text.length > max ? text.slice(0, max - 1) + '…' : text;
  }

  // --- Conversation ID helpers (stateless namespace) ---
  function _generateId8() {
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

  function getConversationId() {
    try {
      let id = sessionStorage.getItem('conversation_id');
      if (!id) {
        id = _generateId8();
        sessionStorage.setItem('conversation_id', id);
        console.log('Generated conversation_id:', id);
      }
      return id;
    } catch (_) {
      // Fallback to window-scoped variable if sessionStorage not available
      if (!window.__CONVERSATION_ID) {
        window.__CONVERSATION_ID = _generateId8();
        console.log('Generated conversation_id (window):', window.__CONVERSATION_ID);
      }
      return window.__CONVERSATION_ID;
    }
  }

  function regenerateConversationId() {
    try {
      const id = _generateId8();
      try { sessionStorage.setItem('conversation_id', id); } catch (_) {}
      window.__CONVERSATION_ID = id;
      console.log('Regenerated conversation_id:', id);
      return id;
    } catch (_) {
      return getConversationId();
    }
  }

  function renderToolsHelp() {
    if (!helpPopup) return;
    const parts = [];
    parts.push('<div class="title">Available tools</div>');
    TOOL_HELP.forEach(t => {
      parts.push('<div class="tool">');
      parts.push(`<div class="tool-name">${t.name}</div>`);
      parts.push(`<div class="tool-desc">${truncate(t.description)}</div>`);
      if (Array.isArray(t.examples) && t.examples.length) {
        parts.push('<div class="tool-examples">');
        t.examples.slice(0, 2).forEach(ex => {
          parts.push(`<div><span class="label" style="color:var(--muted);font-size:12px;">Example:</span> <code>${ex}</code></div>`);
        });
        parts.push('</div>');
      }
      parts.push('</div>');
    });
    helpPopup.innerHTML = parts.join('');
  }

  function showToolsHelp() {
    if (!helpPopup || !helpLink) return;
    renderToolsHelp();
    helpPopup.style.display = 'block';
    helpPopup.setAttribute('aria-hidden', 'false');
  }

  function hideToolsHelp() {
    if (!helpPopup) return;
    helpPopup.style.display = 'none';
    helpPopup.setAttribute('aria-hidden', 'true');
  }

  function toggleToolsHelp(evt) {
    evt && evt.preventDefault();
    if (!helpPopup) return;
    const isHidden = helpPopup.getAttribute('aria-hidden') !== 'false' && helpPopup.style.display !== 'block';
    if (isHidden) showToolsHelp(); else hideToolsHelp();
  }

  // Dismiss popover on outside click or Escape
  document.addEventListener('click', (e) => {
    if (!helpPopup || helpPopup.style.display !== 'block') return;
    const within = helpPopup.contains(e.target) || (helpLink && helpLink.contains(e.target));
    if (!within) hideToolsHelp();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') hideToolsHelp();
  });
  if (helpLink) {
    helpLink.addEventListener('click', toggleToolsHelp);
  }
  if (subsectionHelpLink) {
    subsectionHelpLink.addEventListener('click', (e) => {
      e.preventDefault();
      if (!subsectionPopup) return;
      const isHidden = subsectionPopup.style.display !== 'block';
      // Toggle visibility
      subsectionPopup.style.display = isHidden ? 'block' : 'none';
      subsectionPopup.setAttribute('aria-hidden', isHidden ? 'false' : 'true');
    });
  }

  // Outside-click close for rewrite popup
  document.addEventListener('mousedown', (ev) => {
    if (!subsectionPopup || subsectionPopup.style.display !== 'block') return;
    const withinPopup = subsectionPopup.contains(ev.target);
    const withinLink = subsectionHelpLink && subsectionHelpLink.contains(ev.target);
    if (!withinPopup && !withinLink) {
      subsectionPopup.style.display = 'none';
      subsectionPopup.setAttribute('aria-hidden', 'true');
    }
  });
  // Additional: Rewrite Help toggle (non-breaking, similar behavior)
  if (rewriteHelpLink) {
    rewriteHelpLink.addEventListener('click', (e) => {
      e.preventDefault();
      if (!rewriteHelpPopup) return;
      const isHidden = rewriteHelpPopup.style.display !== 'block';
      rewriteHelpPopup.style.display = isHidden ? 'block' : 'none';
      rewriteHelpPopup.setAttribute('aria-hidden', isHidden ? 'false' : 'true');
    });
  }

  // Helpers for data-bound metrics
  function deepGet(obj, path, def = 0) {
    if (!obj || !path) return def;
    return path.split('.').reduce((o, k) => (o && o[k] !== undefined ? o[k] : undefined), obj);
  }
  function toNumber(v, d = 0) {
    const n = Number(v);
    return Number.isFinite(n) ? n : d;
  }
  function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  }
  // Format very small costs (avoid scientific notation like 1.2e-7)
  function formatCost(value, digits = 8) {
    const num = Number(value);
    if (!Number.isFinite(num) || num === 0) return '0';
    return new Intl.NumberFormat('en-US', {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
      notation: 'standard',
      useGrouping: false,
    }).format(num);
  }
  function isCostPath(path) {
    return typeof path === 'string' && /(^|\.)cost(\.|_|$)/i.test(path);
  }

  function resetMetricsBarToZero() {
    document.querySelectorAll('#metrics_bar [data-metric], #metrics_bar [data-sum]').forEach(el => {
      el.textContent = '0';
    });
  }
  function updateMetrics(payload = {}) {
    const data = payload || {};

    // Fill direct bindings
    document.querySelectorAll('#metrics_bar [data-metric]').forEach(el => {
      const path = el.getAttribute('data-metric');
      const val = deepGet(data, path, 0);
      el.textContent = isCostPath(path) ? formatCost(val) : (val == null ? 0 : val);
    });

    // Fill sum bindings
    document.querySelectorAll('#metrics_bar [data-sum]').forEach(el => {
      const paths = (el.getAttribute('data-sum') || '').split(',').map(s => s.trim()).filter(Boolean);
      const total = paths.reduce((acc, p) => acc + toNumber(deepGet(data, p, 0), 0), 0);
      el.textContent = total;
    });

    // Backward-compat: if only legacy metrics were returned, map what we can.
    if (!data.turn_metrics && data.metrics) {
      const m = data.metrics;
      setText('prompt_tokens', toNumber(m.prompt_tokens, 0));
      setText('inference_cached_tokens', toNumber(m.prompt_cached_tokens || 0, 0));
      setText('completion_tokens', toNumber(m.completion_tokens, 0));
      setText('total_tokens', toNumber(m.total_tokens, 0));
      setText('prompt_cost', formatCost(m.prompt_cost));
      setText('completion_cost', formatCost(m.completion_cost));
      setText('total_cost', formatCost(m.total_cost));
      setText('rerank_tokens_in', toNumber(m.rerank_input_tokens || m.rerank_tokens || 0, 0));
      setText('rerank_tokens_out', toNumber(m.rerank_output_tokens || 0, 0));
      setText('rerank_cost', formatCost(m.rerank_cost || 0));
    }
  }

  // Render the tiny "Query rewrite" pill under the metrics bar.
  function renderRewritePill(display) {
    try {
      const bar = document.getElementById('metrics_bar');
      if (!bar) return;
      // Find or create the pill container
      let pill = document.getElementById('rewrite_pill');
      const removePill = () => { if (pill && pill.parentNode) pill.parentNode.removeChild(pill); };

      // Guard: show only when rewrite ran AND was accepted
      const d = display || {};
      const enabled = !!d.enabled;
      const triggered = !!d.triggered;
      const accepted = !!d.accepted;
      if (!enabled || !triggered || !accepted) {
        removePill();
        return;
      }

      const rewritten = (d.rewritten || d.candidate || '').toString();
      const conf = (typeof d.confidence === 'number') ? d.confidence : Number(d.confidence || 0);
      const confText = Number.isFinite(conf) ? conf.toFixed(2) : (d.confidence == null ? '' : String(d.confidence));

      if (!pill) {
        pill = document.createElement('div');
        pill.id = 'rewrite_pill';
        // Lightweight inline styling so we don't rely on chat.css updates
        pill.style.cssText = 'margin:6px 0 10px; display:flex; align-items:center; gap:8px; font-size:12px; background:#eef2ff; color:#1e293b; border:1px solid #c7d2fe; border-radius:12px; padding:6px 10px;';
        // Insert just after the metrics bar
        bar.insertAdjacentElement('afterend', pill);
      }

      // Build content
      pill.innerHTML = '';
      const label = document.createElement('span');
      label.textContent = 'Rewritten Query:';
      label.style.fontWeight = '600';

      const text = document.createElement('span');
      text.textContent = `"${truncate(rewritten, 120)}"`;
      text.title = `original: ${d.original || ''}`;

      const meta = document.createElement('span');
      meta.textContent = (confText ? `• conf=${confText}` : '') + (d.threshold != null ? ` (≥ ${Number(d.threshold).toFixed ? Number(d.threshold).toFixed(2) : d.threshold})` : '');
      meta.style.opacity = '0.8';

      pill.appendChild(label);
      pill.appendChild(text);
      pill.appendChild(meta);
    } catch (e) {
      console.debug('Failed to render rewrite pill', e);
    }
  }

  // Render a single-line rewrite status into a fixed placeholder (if present in HTML)
  function renderRewriteLine(display) {
    try {
      const el = document.getElementById('rewrite_line');
      if (!el) return; // only update if the HTML added a placeholder
      const d = display || {};
      if (d && d.accepted && d.rewritten) {
        const conf = (typeof d.confidence === 'number') ? d.confidence : Number(d.confidence || 0);
        const confText = Number.isFinite(conf) ? conf.toFixed(2) : (d.confidence == null ? '' : String(d.confidence));
        const thr = d.threshold;
        const thrText = (thr != null) ? (Number.isFinite(Number(thr)) && Number(thr).toFixed ? Number(thr).toFixed(2) : String(thr)) : '';
        const parts = [];
        parts.push(`Rewritten: "${truncate(String(d.rewritten), 160)}"`);
        if (confText) parts.push(`• conf=${confText}`);
        if (thrText) parts.push(`(≥ ${thrText})`);
        el.textContent = parts.join(' ');
        el.title = d.original ? `Original: ${d.original}` : '';
      } else {
        el.textContent = '';
        el.title = '';
      }
    } catch (e) {
      console.debug('Failed to render rewrite line', e);
    }
  }

  // Append a message bubble with role badge.
  function appendMessage(role, text) {
    const wrapper = document.createElement('div');
    wrapper.className = `msg ${role}`;

    const badge = document.createElement('span');
    badge.className = 'badge';
    badge.textContent = role === 'user' ? 'User' : 'Assistant';

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;

    // Order badge then bubble for clear role labeling.
    wrapper.appendChild(badge);
    wrapper.appendChild(bubble);
    chatHistory.appendChild(wrapper);

    // Ensure the latest message is visible.
    chatHistory.scrollTop = chatHistory.scrollHeight;
  }

  // Collect sidebar params as numbers.
  function collectParams() {
    const getNum = (id) => {
      const el = qs('#' + id);
      const v = el ? Number(el.value) : null;
      return Number.isFinite(v) ? v : null;
    };
    const useToolsEl = qs('#use_tools');
    const use_tools = useToolsEl ? !!useToolsEl.checked : false;

    const base = {
      top_k: getNum('top_k'),
      score_threshold: getNum('score_threshold'),
      summarizer_max_input_tokens: getNum('summarizer_max_input_tokens'),
      summarizer_max_output_tokens: getNum('summarizer_max_output_tokens'),
      temperature: getNum('temperature'),
      max_output_tokens: getNum('max_output_tokens'),
      top_p: getNum('top_p'),
      chat_history_window_turns: getNum('chat_history_window_turns'),
      raw_tail_turns: getNum('raw_tail_turns'),
      enable_query_rewrite: (qs('#enable_query_rewrite') ? !!qs('#enable_query_rewrite').checked : null),
      rewrite_confidence_threshold: getNum('rewrite_confidence_threshold'),
      rewrite_tail_turns: getNum('rewrite_tail_turns'),
      use_tools,
    };

    // Attach model/provider overrides per stage (if any). These map to backend resolve_stage_specs.
    try {
      base.inference_provider = stageModelConfig.inference.provider;
      base.inference_model = stageModelConfig.inference.model;
      base.rewrite_provider = stageModelConfig.rewrite.provider;
      base.rewrite_model = stageModelConfig.rewrite.model;
      base.summary_provider = stageModelConfig.summary.provider;
      base.summary_model = stageModelConfig.summary.model;
      base.rerank_provider = stageModelConfig.rerank.provider;
      base.rerank_model = stageModelConfig.rerank.model;
    } catch (e) {
      console.debug('Failed to attach model overrides to params', e);
    }

    return base;
  }

  // Collect chat history from DOM bubbles.
  function collectHistory() {
    const out = [];
    chatHistory.querySelectorAll('.msg').forEach((node) => {
      const role = node.classList.contains('user') ? 'user' : 'assistant';
      const bubble = node.querySelector('.bubble');
      if (bubble) out.push({ role, content: bubble.textContent || '' });
    });
    return out;
  }

  // Simple ephemeral toast for errors.
  function toast(msg) {
    const t = document.createElement('div');
    t.textContent = msg;
    t.style.cssText = 'position:fixed;bottom:16px;right:16px;background:#fee2e2;color:#991b1b;border:1px solid #fecaca;padding:8px 10px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.08);font-size:12px;z-index:9999;';
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 4000);
  }

  // Send the current input value if non-empty.
  async function sendMessage() {
    const text = (input.value || '').trim();
    if (!text) return;

    // 1) Generate a query_id for the request (secure-context-safe)
    let queryId;
    try {
      if (window.crypto && typeof window.crypto.randomUUID === 'function') {
        queryId = window.crypto.randomUUID().slice(0, 8);
      } else if (window.crypto && window.crypto.getRandomValues) {
        // Fallback using getRandomValues
        const arr = new Uint32Array(2);
        window.crypto.getRandomValues(arr);
        queryId = (arr[0].toString(16) + arr[1].toString(16)).slice(0, 8);
      } else {
        // Last-resort fallback
        queryId = Math.random().toString(36).slice(2, 10);
      }
    } catch (_) {
      queryId = Math.random().toString(36).slice(2, 10);
    }
    console.log('Prepared queryId for streaming:', queryId);

    // Ensure a conversation_id exists (generate on first use if needed)
    const conversationId = getConversationId();

    // Append user message first
    appendMessage('user', text);

    // 3) Add assistant placeholder to replace later (before starting SSE)
    const wrapper = document.createElement('div');
    wrapper.className = 'msg assistant';
    const badge = document.createElement('span');
    badge.className = 'badge';
    badge.textContent = 'Assistant';
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
   //bubble.textContent = 'Thinking…';
    bubble.textContent = 'Processing';
    wrapper.appendChild(badge);
    wrapper.appendChild(bubble);
    chatHistory.appendChild(wrapper);
    chatHistory.scrollTop = chatHistory.scrollHeight;

    // Close any previously open stage streams before starting a new turn
    try {
      if (window.__STAGE_STREAMS && window.__STAGE_STREAMS.size) {
        window.__STAGE_STREAMS.forEach((es) => { try { es.close(); } catch (_) {} });
        window.__STAGE_STREAMS.clear();
      }
    } catch (_) {}
    // Start stage streaming now that the assistant bubble exists
    setupStageStreaming(queryId, bubble);
    // Once the conversation has started, lock the history-window controls
    try {
      const lockIds = ['chat_history_window_turns', 'raw_tail_turns'];
      lockIds.forEach(id => {
        const el = qs('#' + id);
        if (el && !el.disabled) {
          el.disabled = true;
          el.setAttribute('data-locked-by-convo', 'true');
          // add a small helper hint next to the control to explain lock
          const hintId = id + '_locked_hint';
          if (!qs('#' + hintId)) {
            // create a small '*' help link and popup similar to tools_help
            const link = document.createElement('a');
            link.id = id + '_locked_link';
            link.className = 'help-link';
            link.href = '#';
            link.textContent = '*';
            link.style.cssText = 'margin-left:6px;font-weight:700;text-decoration:underline;color:#0366d6;cursor:pointer;';

            const popup = document.createElement('div');
            popup.id = id + '_locked_popup';
            popup.className = 'popover';
            popup.role = 'dialog';
            popup.setAttribute('aria-hidden', 'true');
            popup.style.display = 'none';
            popup.style.position = 'absolute';
            popup.style.zIndex = 9999;
            popup.style.maxWidth = '260px';
            popup.style.padding = '8px';
            popup.style.background = 'white';
            popup.style.border = '1px solid rgba(0,0,0,0.12)';
            popup.style.boxShadow = '0 6px 18px rgba(0,0,0,0.12)';
            popup.textContent = 'Locked for this conversation. Clear chat to change this setting.';

            // insert link into label (so it's inline) but append popup to body for reliable positioning
            const label = document.querySelector('label[for="' + id + '"]');
            if (label) {
              label.appendChild(link);
            } else {
              el.insertAdjacentElement('afterend', link);
            }
            document.body.appendChild(popup);

            // Toggle popup and position it near the link
            const toggle = (ev) => {
              ev && ev.preventDefault();
              const rect = link.getBoundingClientRect();
              const top = window.scrollY + rect.bottom + 6; // small gap
              const left = window.scrollX + rect.left;
              popup.style.left = left + 'px';
              popup.style.top = top + 'px';
              const isHidden = popup.getAttribute('aria-hidden') !== 'false' || popup.style.display !== 'block';
              if (isHidden) {
                popup.style.display = 'block';
                popup.setAttribute('aria-hidden', 'false');
              } else {
                popup.style.display = 'none';
                popup.setAttribute('aria-hidden', 'true');
              }
            };
            link.addEventListener('click', toggle);
            // click outside to dismiss
            const outsideClick = (e) => {
              if (popup.style.display !== 'block') return;
              if (popup.contains(e.target) || link.contains(e.target) || el.contains(e.target)) return;
              popup.style.display = 'none';
              popup.setAttribute('aria-hidden', 'true');
            };
            document.addEventListener('click', outsideClick);
            // store references for cleanup
            link.__locked_popup_handler = outsideClick;
            link.__locked_popup_elem = popup;
          }
          el.setAttribute('title', 'Locked for this conversation. Click the * for details.');
        }
      });
    } catch (e) {
      console.debug('Failed to lock history controls', e);
    }

    // 4) Prepare payload with the same query_id
    const payload = {
      message: text,
      params: {
        ...collectParams(),
        query_id: queryId,  // Using the same query_id as above
        conversation_id: conversationId
      },
      history: collectHistory(),
    };

    // 5) Clear input and disable controls during request
    input.value = '';
    autoResize();
    input.disabled = true;
    sendBtn.disabled = true;
    input.setAttribute('aria-busy', 'true');

    try {
      const resp = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      let data;
      try {
        data = await resp.json();
      } catch (_) {
        data = null;
      }

      // If backend surfaced a structured error, prefer that message.
      if (!resp.ok || (data && data.error)) {
        const err = (data && data.error) || {};
        const msg = err.message || 'Request failed. Please try again.';
        bubble.textContent = msg;
        toast(msg);
        return;
      }

      // Replace placeholder bubble with answer
      const answerText = (
        (data && (data.response ?? data.answer ?? data.output ?? data.message))
      ) || '(no answer)';
      // Strip any inline "Tools Used:" tail if present; we render it separately
      const toolsLineRe = /\n\nTools Used:.*$/s;
      let displayText = answerText;
      if (toolsLineRe.test(displayText)) {
        displayText = displayText.replace(toolsLineRe, '');
      }
      bubble.textContent = displayText;
      // Render tools-used dim line, if provided
      if (data && Array.isArray(data.tools_used) && data.tools_used.length > 0) {
        const toolsDiv = document.createElement('div');
        toolsDiv.className = 'tools-used';
        toolsDiv.textContent = 'Tools Used: ' + data.tools_used.join(', ');
        bubble.appendChild(document.createElement('br'));
        bubble.appendChild(toolsDiv);
      }
      // Update metrics
      updateMetrics(data || {});
      // Render (or remove) the query rewrite pill
      try { renderRewritePill(data && data.rewrite_display); } catch (_) {}
      // Update single-line rewrite placeholder if present in HTML
      try { renderRewriteLine(data && data.rewrite_display); } catch (_) {}
    } catch (e) {
      const msg = 'Request failed. Please try again.';
      bubble.textContent = msg;
      toast(msg);
    } finally {
      input.disabled = false;
      sendBtn.disabled = false;
      input.removeAttribute('aria-busy');
      input.focus();
    }
  }

  // Auto-resize textarea up to ~4 lines.
  function autoResize() {
    // Reset height to measure scrollHeight accurately.
    input.style.height = 'auto';
    const cs = window.getComputedStyle(input);
    const lineHeight = parseFloat(cs.lineHeight) || 18;
    const max = lineHeight * 4 + 8; // small buffer
    const next = Math.min(input.scrollHeight, max);
    input.style.height = next + 'px';
  }

  // Key handling: Enter to send, Shift+Enter for newline.
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  input.addEventListener('input', autoResize);
  sendBtn.addEventListener('click', () => { sendMessage(); });

  // First render: ensure input has proper height.
  autoResize();

  // Set fixed label column width
  (function setLabelColumnWidth() {
    const form = qs('#params_form');
    if (!form) return;
    function applyWidth() {
      form.style.setProperty('--label-col', '90px');
    }
    applyWidth();
    window.addEventListener('resize', applyWidth);
  })();

  // Wire up Clear Chat: clear history and metrics, retain params.
  const clearChatBtn = qs('#clear_chat_btn');
  if (clearChatBtn) {
    clearChatBtn.addEventListener('click', () => {
      chatHistory.innerHTML = '';
      resetMetricsBarToZero();
      // Remove rewrite pill if present
      try {
        const rp = document.getElementById('rewrite_pill');
        if (rp && rp.parentNode) rp.parentNode.removeChild(rp);
      } catch (_) {}
      // Clear rewrite one-line placeholder if present
      try {
        const rl = document.getElementById('rewrite_line');
        if (rl) { rl.textContent = ''; rl.title = ''; }
      } catch (_) {}
      // Unlock any convo-locked controls so the user can change them for next conversation
      try {
        ['chat_history_window_turns', 'raw_tail_turns'].forEach(id => {
          const el = qs('#' + id);
          // Also remove the locked_link and its popup if present
          try {
            const link = qs('#' + id + '_locked_link');
            if (link) {
              const handler = link.__locked_popup_handler;
              if (handler) document.removeEventListener('click', handler);
              const popup = link.__locked_popup_elem || qs('#' + id + '_locked_popup');
              if (popup) popup.remove();
              link.remove();
            }
          } catch (e) {
            console.debug('Failed cleanup locked link', e);
          }
          if (el && el.getAttribute('data-locked-by-convo') === 'true') {
            el.disabled = false;
            el.removeAttribute('data-locked-by-convo');
            el.removeAttribute('title');
          }
        });
      } catch (e) {
        console.debug('Failed to unlock history controls', e);
      }
      // Also close/clear any open stage streams for the old conversation
      try {
        if (window.__STAGE_STREAMS && window.__STAGE_STREAMS.size) {
          window.__STAGE_STREAMS.forEach((es) => { try { es.close(); } catch (_) {} });
          window.__STAGE_STREAMS.clear();
        }
      } catch (_) {}

      // Clear server-side summaries for this conversation and rotate a new conversation_id
      try {
        const currentId = getConversationId();
        // Best-effort: call backend to clear this conversation's summaries (non-breaking if endpoint missing)
        fetch('/chat/clear', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ conversation_id: currentId })
        }).catch(() => {});
      } catch (_) {}

      // Generate a new conversation id for the next conversation
      try { regenerateConversationId(); } catch (_) {}
    });
  }

  // Reset Parameters: re-fetch defaults from backend and apply to the form
  const resetParamsBtn = qs('#reset_params_btn');
  if (resetParamsBtn) {
    resetParamsBtn.addEventListener('click', async () => {
      const original = resetParamsBtn.textContent;
      resetParamsBtn.disabled = true;
      resetParamsBtn.textContent = 'Resetting…';
      try {
        await loadModelConfig();
        resetParamsBtn.textContent = 'Parameters Reset';
      } catch (e) {
        console.error('Failed to reset parameters', e);
        resetParamsBtn.textContent = 'Reset failed';
      } finally {
        setTimeout(() => { resetParamsBtn.textContent = original; resetParamsBtn.disabled = false; }, 1000);
      }
    });
  }

  // Fetch and display model configuration from backend
  async function loadModelConfig() {
    try {
      // Initialize with all values as null - will be updated from API
      const modelConfig = {
        embedding_model: null,
        re_ranker_model: null,
        query_rewrite_model: null,
        summarizer_model: null,
        inference_model: null
      };
      
      // Try to fetch from API
      try {
        const response = await fetch('/api/config');
        if (response.ok) {
          const config = await response.json();
          // Only update with values that exist in the response
          Object.keys(modelConfig).forEach(key => {
            if (config[key] !== undefined && config[key] !== null) {
              modelConfig[key] = config[key];
            }
          });
        }
      } catch (error) {
        console.error('Error loading model config:', error);
      }
      
      // Update the UI with the model information
      const modelElements = {
        'model_embedding': modelConfig.embedding_model,
        'model_reranker': modelConfig.re_ranker_model,
        'model_query_rewrite': modelConfig.query_rewrite_model,
        'model_summarizer': modelConfig.summarizer_model,
        'model_inference': modelConfig.inference_model
      };
      
      // Update the UI, showing 'Not Found' for any null/undefined values
      Object.entries(modelElements).forEach(([id, value]) => {
        const element = document.getElementById(id);
        if (element) {
          element.textContent = value !== null && value !== undefined ? value : 'Not Found';
        }
      });

      // Also try to populate parameter defaults for the form fields if provided
      try {
        const response = await fetch('/api/config');
        if (response.ok) {
          const cfg = await response.json();
          // Map config keys to form element IDs
          const mapping = {
            score_threshold: 'score_threshold',
            top_k: 'top_k',
            summarizer_max_input_tokens: 'summarizer_max_input_tokens',
            summarizer_max_output_tokens: 'summarizer_max_output_tokens',
            inference_temperature: 'temperature',
            inference_top_p: 'top_p',
            inference_context_rows: 'inference_context_rows',
            chat_history_window_turns: 'chat_history_window_turns',
            raw_tail_turns: 'raw_tail_turns',
            max_inference_output_tokens: 'max_output_tokens',
            enable_tools: 'use_tools',
            enable_query_rewrite: 'enable_query_rewrite',
            rewrite_confidence_threshold: 'rewrite_confidence_threshold',
            rewrite_tail_turns: 'rewrite_tail_turns'
          };
          Object.entries(mapping).forEach(([cfgKey, elId]) => {
            if (cfg[cfgKey] === undefined) return;
            const el = document.getElementById(elId);
            if (!el) return;
            if (el.type === 'checkbox') {
              el.checked = !!cfg[cfgKey];
            } else {
              el.value = String(cfg[cfgKey]);
            }
  }
  // Also close/clear any open stage streams for the old conversation
  try {
    if (window.__STAGE_STREAMS && window.__STAGE_STREAMS.size) {
      window.__STAGE_STREAMS.forEach((es) => { try { es.close(); } catch (_) {} });
      window.__STAGE_STREAMS.clear();
    }
  } catch (_) {}

  // Clear server-side summaries for this conversation and rotate a new conversation_id
  try {
    const currentId = getConversationId();
    // Best-effort: call backend to clear this conversation's summaries (non-breaking if endpoint missing)
    fetch('/chat/clear', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ conversation_id: currentId })
    }).catch(() => {});
  } catch (_) {}

  // Generate a new conversation id for the next conversation
  try { regenerateConversationId(); } catch (_) {}
});

}

// Reset Parameters: re-fetch defaults from backend and apply to the form
const resetParamsBtn = qs('#reset_params_btn');
if (resetParamsBtn) {
resetParamsBtn.addEventListener('click', async () => {
  const original = resetParamsBtn.textContent;
  resetParamsBtn.disabled = true;
  resetParamsBtn.textContent = 'Resetting…';
  try {
    await loadModelConfig();
    resetParamsBtn.textContent = 'Parameters Reset';
  } catch (e) {
    console.error('Failed to reset parameters', e);
    resetParamsBtn.textContent = 'Reset failed';
  } finally {
    setTimeout(() => { resetParamsBtn.textContent = original; resetParamsBtn.disabled = false; }, 1000);
  }
});
}

// Fetch and display model configuration from backend
async function loadModelConfig() {
try {
  // Initialize with all values as null - will be updated from API
  const modelConfig = {
    embedding_model: null,
    re_ranker_model: null,
    query_rewrite_model: null,
    summarizer_model: null,
    inference_model: null
  };
  
  // Try to fetch from API
  try {
    const response = await fetch('/api/config');
    if (response.ok) {
      const config = await response.json();
      // Only update with values that exist in the response
      Object.keys(modelConfig).forEach(key => {
        if (config[key] !== undefined && config[key] !== null) {
          modelConfig[key] = config[key];
        }
      });
    }
  } catch (error) {
    console.error('Error loading model config:', error);
  }
  
  // Update the UI with the model information
  const modelElements = {
    'model_embedding': modelConfig.embedding_model,
    'model_reranker': modelConfig.re_ranker_model,
    'model_query_rewrite': modelConfig.query_rewrite_model,
    'model_summarizer': modelConfig.summarizer_model,
    'model_inference': modelConfig.inference_model
  };
  
  // Update the UI, showing 'Not Found' for any null/undefined values
  Object.entries(modelElements).forEach(([id, value]) => {
    const element = document.getElementById(id);
    if (element) {
      element.textContent = value !== null && value !== undefined ? value : 'Not Found';
    }
  });

  // Also try to populate parameter defaults for the form fields if provided
  try {
    const response = await fetch('/api/config');
    if (response.ok) {
      const cfg = await response.json();
      // Map config keys to form element IDs
      const mapping = {
        score_threshold: 'score_threshold',
        top_k: 'top_k',
        summarizer_max_input_tokens: 'summarizer_max_input_tokens',
        summarizer_max_output_tokens: 'summarizer_max_output_tokens',
        inference_temperature: 'temperature',
        inference_top_p: 'top_p',
        inference_context_rows: 'inference_context_rows',
        chat_history_window_turns: 'chat_history_window_turns',
        raw_tail_turns: 'raw_tail_turns',
        max_inference_output_tokens: 'max_output_tokens',
        enable_tools: 'use_tools',
        enable_query_rewrite: 'enable_query_rewrite',
        rewrite_confidence_threshold: 'rewrite_confidence_threshold',
        rewrite_tail_turns: 'rewrite_tail_turns'
      };
      Object.entries(mapping).forEach(([cfgKey, elId]) => {
        if (cfg[cfgKey] === undefined) return;
        const el = document.getElementById(elId);
        if (!el) return;
        if (el.type === 'checkbox') {
          el.checked = !!cfg[cfgKey];
        } else {
          el.value = String(cfg[cfgKey]);
        }
      });
    }
  } catch (err) {
    // silently ignore if /api/config isn't available or fails
    console.debug('No runtime config values available:', err && err.message);
  }
} catch (error) {
  console.error('Error in loadModelConfig:', error);
}
}

// Load model configuration when the page loads
document.addEventListener('DOMContentLoaded', async () => {
try {
  await loadModelConfig();
} finally {
  // After backend config/labels are loaded, wire and init the models modal.
  try { wireModelsModalEvents(); } catch (e) { console.debug('Failed to wire models modal', e); }
}
});

// Ensure a conversation_id exists on load
document.addEventListener('DOMContentLoaded', () => { try { getConversationId(); } catch (_) {} });

// Best-effort: on page reload/close, clear server-side summaries for current conversation.
function _clearServerSummariesOnUnload() {
try {
  const cid = getConversationId();
  if (!cid) return;
  const data = JSON.stringify({ conversation_id: cid });
  if (navigator.sendBeacon) {
    const blob = new Blob([data], { type: 'application/json' });
    navigator.sendBeacon('/chat/clear', blob);
  } else {
    // Fallback: non-blocking fetch with keepalive
    fetch('/chat/clear', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: data,
      keepalive: true,
    }).catch(() => {});
  }
} catch (_) {}
}

// Minor visual tweak: metrics subsection titles
document.addEventListener('DOMContentLoaded', () => {
try {
  document.querySelectorAll('.metrics-subsection-title').forEach(el => {
    el.style.marginTop = '1px';
  });
} catch (e) {
  console.debug('Failed to apply metrics-subsection-title styles', e);
}
});

// Initialize collapsible parameter groups in the sidebar
document.addEventListener('DOMContentLoaded', () => {
try {
  const groups = document.querySelectorAll('.collapsible-group');
  if (!groups || !groups.length) return;

  groups.forEach(group => {
    const header = group.querySelector('.collapsible-header');
    if (!header) return;

    const isButtonHeader = header.tagName === 'BUTTON';
    const toggle = (evt) => {
      if (evt) {
        const target = evt.target;
        // Do not intercept clicks on nested interactive controls like Change Models or help links
        if (target && (target.closest('.link-button') || target.closest('a.help-link'))) {
          return;
        }
      }
      const collapsed = group.classList.toggle('is-collapsed');
      const expanded = !collapsed;
      header.setAttribute('aria-expanded', expanded ? 'true' : 'false');
    };

    if (isButtonHeader) {
      header.addEventListener('click', toggle);
    } else {
      // For non-button headers (e.g., Models Used), make them clickable and keyboard-accessible
      header.addEventListener('click', toggle);
      header.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          toggle(e);
        }
      });
    }
  });
} catch (e) {
  console.debug('Failed to initialize collapsible parameter groups', e);
}
});

// Best-effort: close any active streams when navigating away
window.addEventListener('beforeunload', () => {
try {
  if (window.__STAGE_STREAMS && window.__STAGE_STREAMS.size) {
    window.__STAGE_STREAMS.forEach((es) => { try { es.close(); } catch (_) {} });
    window.__STAGE_STREAMS.clear();
  }
} catch (_) {}
});

// Global map of active stage streams (per queryId)
window.__STAGE_STREAMS = window.__STAGE_STREAMS || new Map();
// Lightweight stage streaming integration (additive UI) using the existing Thinking bubble
function setupStageStreaming(queryId, bubbleEl) {
if (!queryId) return;
// If a stream already exists for this query, close it first (safety)
try {
  const prev = window.__STAGE_STREAMS.get(queryId);
  if (prev && typeof prev.close === 'function') prev.close();
  window.__STAGE_STREAMS.delete(queryId);
} catch (_) {}
try {
  const es = new EventSource(`/chat/stream/stages?query_id=${encodeURIComponent(queryId)}`);
  // Track this stream instance by queryId
  window.__STAGE_STREAMS.set(queryId, es);
  try {
    const prev = window.__STAGE_STREAMS.get(queryId);
    if (prev && typeof prev.close === 'function') prev.close();
    window.__STAGE_STREAMS.delete(queryId);
  } catch (_) {}
  try {
    const es = new EventSource(`/chat/stream/stages?query_id=${encodeURIComponent(queryId)}`);
    // Track this stream instance by queryId
    window.__STAGE_STREAMS.set(queryId, es);

    // Resolve the target bubble: prefer the one passed in; otherwise pick the latest assistant bubble
    const resolveBubble = () => {
      if (bubbleEl && document.body.contains(bubbleEl)) return bubbleEl;
      const nodes = document.querySelectorAll('#chat_history .msg.assistant .bubble');
      return nodes[nodes.length - 1] || null;
    };

    let bubble = resolveBubble();
    let currentText = (bubble?.textContent || 'Thinking');
    // Finalization and de-dupe state
    let finished = false;
    const closeAndForget = () => {
      if (finished) return;
      finished = true;
      try { es.close(); } catch (_) {}
      try { window.__STAGE_STREAMS.delete(queryId); } catch (_) {}
    };
    const seenStages = new Set();

    es.onmessage = (e) => {
      if (finished) return;
      try {
        const payload = JSON.parse(e.data);
        // Handle explicit final messages too (if backend chooses to send them)
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

        // Re-resolve in case DOM changed since last event
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
        if (seenStages.has(stage)) return;   // ignore duplicate stage events
        seenStages.add(stage);
        currentText = `${currentText} --> ${stage}`;
        bubble.textContent = currentText;
      } catch (err) {
        // ignore parse errors
      }
    };

    es.onerror = () => {
      if (!finished) {
        try { es.close(); } catch (_) {}
        try { window.__STAGE_STREAMS.delete(queryId); } catch (_) {}
      }
    };
  } catch (err) {
    // ignore streaming errors; UI should still function
  }
}

// Auto-connect if a queryId is present on the body
document.addEventListener('DOMContentLoaded', () => {
  const qid = document.body?.dataset?.queryId || window.__QUERY_ID__;
  if (qid) setupStageStreaming(qid);
});
