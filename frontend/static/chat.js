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
  const summarizerOutputHelpLink = qs('#summarizer_output_help_link');
  const summarizerOutputHelpPopup = qs('#summarizer_output_help_popup');

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

  // Providers and models were originally kept client-side; now default to a
  // minimal built-in set but prefer live data from the backend registry.
  const PROVIDERS = ['openai', 'gemini'];

  // Model registry with display names (defaults; overridden by /api/models).
  let MODEL_REGISTRY = {
    'openai:gpt-4o-mini': {
      provider: 'openai',
      model: 'gpt-4o-mini',
      display: 'openai:gpt-4o-mini (openai/gpt-4o-mini)'
    },
    'openai:gpt-5-nano': {
      provider: 'openai',
      model: 'gpt-5-nano',
      display: 'openai:gpt-5-nano (openai/gpt-5-nano)'
    },
    'gemini:flash-lite': {
      provider: 'gemini',
      model: 'models/gemini-2.5-flash-lite',
      display: 'gemini:flash-lite (gemini/gemini-2.5-flash-lite)'
    },
    'gemini:flash': {
      provider: 'gemini',
      model: 'gemini-2.5-flash',
      display: 'gemini:flash (gemini/gemini-2.5-flash)'
    },
    'gemini:preview': {
      provider: 'gemini',
      model: 'gemini-3-flash-preview',
      display: 'gemini:preview (gemini/gemini-3-flash-preview)'
    },
    'openai:gpt-4o': {
      provider: 'openai',
      model: 'gpt-4o',
      display: 'openai:gpt-4o (openai/gpt-4o)'
    },
    'openai:text-embedding-3-small': {
      provider: 'openai',
      model: 'text-embedding-3-small',
      display: 'openai:text-embedding-3-small (openai/text-embedding-3-small)'
    }
  };

  // Available models by stage (defaults; overridden by /api/models).
  let MODELS_BY_STAGE = {
    inference: ['openai:gpt-4o-mini', 'openai:gpt-5-nano', 'gemini:flash-lite', 'gemini:flash', 'gemini:preview'],
    rewrite: ['openai:gpt-4o-mini', 'openai:gpt-5-nano', 'gemini:flash-lite', 'gemini:flash', 'gemini:preview'],
    summary: ['openai:gpt-4o-mini', 'openai:gpt-5-nano', 'gemini:flash-lite', 'gemini:flash', 'gemini:preview'],
    rerank: ['openai:gpt-4o-mini', 'openai:gpt-5-nano', 'gemini:flash-lite', 'gemini:flash', 'gemini:preview']
  };

  // Helper function to get model info
  function getModelInfo(key) {
    return MODEL_REGISTRY[key] || { provider: '', model: key, display: key };
  }

  // Update the "Models Used" labels in the main chat UI from the current stageModelConfig.
  function updateModelLabels() {
    const updateLabel = (stage, elementId) => {
      const el = document.getElementById(elementId);
      if (!el) return;
      const modelKey = stageModelConfig[stage] && stageModelConfig[stage].model_key;
      if (!modelKey) {
        el.textContent = 'Not Found';
        return;
      }
      const info = getModelInfo(modelKey);
      el.textContent = info.display || modelKey || 'Not Found';
    };

    updateLabel('embedding', 'model_embedding');
    updateLabel('inference', 'model_inference');
    updateLabel('rewrite', 'model_query_rewrite');
    updateLabel('summary', 'model_summarizer');
    updateLabel('rerank', 'model_reranker');
  }

  // Fetch live model registry from backend and hydrate MODEL_REGISTRY / MODELS_BY_STAGE
  async function fetchModelRegistry() {
    try {
      const resp = await fetch('/api/models?merge_custom_registry=true');
      if (!resp.ok) return; // keep defaults
      const data = await resp.json();

      const registry = {};
      const stages = {
        inference: [],
        rewrite: [],
        summary: [],
        rerank: [],
      };

      Object.values(data).forEach((m) => {
        if (!m || !m.key) return;
        const key = m.key;
        registry[key] = {
          provider: m.provider,
          model: m.model,
          endpoint: m.endpoint,
          display: `${key} → ${m.model} (${m.provider}, ${m.endpoint})`,
          capabilities: m.capabilities || {},
        };

        // Heuristic: any non-embedding model is eligible for all chat stages.
        if (m.endpoint && m.endpoint !== 'embeddings') {
          stages.inference.push(key);
          stages.rewrite.push(key);
          stages.summary.push(key);
          stages.rerank.push(key);
        }
      });

      // Only overwrite if we actually parsed something.
      if (Object.keys(registry).length) {
        MODEL_REGISTRY = registry;
        MODELS_BY_STAGE = stages;

        // Refresh modal selects if the modal is currently open.
        if (modelsModal && modelsModal.style.display === 'block') {
          initModelsModalFromConfig();
        } else {
          // Also refresh the labels so the main UI shows nice display strings.
          if (typeof updateModelLabels === 'function') {
            updateModelLabels();
          }
        }
      }
    } catch (e) {
      console.debug('Failed to fetch model registry from backend; using defaults', e);
    }
  }

  // Current selections (defaults are hydrated from backend /api/config model_key fields).
  const stageModelConfig = {
    embedding: { model_key: null },
    inference: { model_key: null },
    rewrite:   { model_key: null },
    summary:   { model_key: null },
    rerank:    { model_key: null },
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

  function _populateModelSelect(stage, modelKey, sel) {
    if (!sel) return;
    sel.innerHTML = '';
    
    // Get the model keys for this stage
    const modelKeys = MODELS_BY_STAGE[stage] || [];
    
    // Add options for each model key, excluding embedding models
    modelKeys.forEach(key => {
      // Skip models with "embed" in the key (e.g., embedding models)
      if (key.toLowerCase().includes('embed')) {
        return;
      }
      
      const modelInfo = getModelInfo(key);
      const opt = document.createElement('option');
      opt.value = key;
      opt.textContent = modelInfo.display || key;
      opt.selected = (key === modelKey);
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

      // Set default model keys based on labels if available
      const setModelKeyFromLabel = (label, stage) => {
        if (label && label.textContent && label.textContent !== 'Not Found') {
          const modelName = label.textContent.trim();
          // Find the model key that matches the model name
          const matchingKey = Object.keys(MODEL_REGISTRY).find(key => {
            const info = MODEL_REGISTRY[key];
            return info && info.model === modelName;
          });
          if (matchingKey) {
            stageModelConfig[stage].model_key = matchingKey;
          }
        }
      };

      // Update model keys from labels if available
      setModelKeyFromLabel(labInf, 'inference');
      setModelKeyFromLabel(labRw, 'rewrite');
      setModelKeyFromLabel(labSum, 'summary');
      setModelKeyFromLabel(labRr, 'rerank');

      // Populate model selects with current selections
      _populateModelSelect('inference', stageModelConfig.inference.model_key, infModelSel);
      _populateModelSelect('rewrite', stageModelConfig.rewrite.model_key, rwModelSel);
      _populateModelSelect('summary', stageModelConfig.summary.model_key, sumModelSel);
      _populateModelSelect('rerank', stageModelConfig.rerank.model_key, rrModelSel);

      // Update model labels in the UI
      updateModelLabels();
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
    if (!changeModelsBtn) return;

    // Open modal
    changeModelsBtn.addEventListener('click', openModelsModal);

    // Close modal
    if (modelsModalClose) modelsModalClose.addEventListener('click', closeModelsModal);
    if (modelsModalCancel) modelsModalCancel.addEventListener('click', closeModelsModal);

    // Close on outside click
    window.addEventListener('click', (e) => {
      if (e.target === modelsModal) closeModelsModal();
    });

    // Close on Escape
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && modelsModal.style.display === 'block') {
        closeModelsModal();
      }
    });

    // Update model config when a model is selected
    const updateModelConfig = (selectElement, stage) => {
      if (!selectElement) return;
      selectElement.addEventListener('change', () => {
        const modelKey = selectElement.value;
        if (modelKey && stageModelConfig[stage]) {
          stageModelConfig[stage].model_key = modelKey;
          updateModelLabels();
        }
      });
    };

    // Set up model selection change handlers
    updateModelConfig(infModelSel, 'inference');
    updateModelConfig(rwModelSel, 'rewrite');
    updateModelConfig(sumModelSel, 'summary');
    updateModelConfig(rrModelSel, 'rerank');

    if (modelsModalSave) {
      modelsModalSave.addEventListener('click', (e) => {
        e.preventDefault();
        try {
          // Update model keys from the dropdown selections
          if (infModelSel) stageModelConfig.inference.model_key = infModelSel.value;
          if (rwModelSel) stageModelConfig.rewrite.model_key = rwModelSel.value;
          if (sumModelSel) stageModelConfig.summary.model_key = sumModelSel.value;
          if (rrModelSel) stageModelConfig.rerank.model_key = rrModelSel.value;

          // Update the UI to reflect the selected models
          updateModelLabels();
          closeModelsModal();
        } catch (e) {
          console.debug('Failed to save model selections', e);
        }
      });
    }
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

  function renderSummarizerOutputHelp() {
    if (!summarizerOutputHelpPopup) return;
    const parts = [];
    parts.push('<div class="title">Summarizer Max Output Tokens</div>');
    parts.push('<div class="tool-desc">This setting controls the maximum length of generated summaries and applies to both:</div>');
    parts.push('<div class="tool-examples">');
    parts.push('<div><span class="label" style="color:var(--muted);font-size:12px;">Rewrite Pre-Summarization:</span> Limits output when summarizing older conversation turns before query rewrite.</div>');
    parts.push('<div><span class="label" style="color:var(--muted);font-size:12px;">Context Window (Chunked History):</span> Limits output when updating accumulated conversation summaries in chunked mode.</div>');
    parts.push('</div>');
    parts.push('<div class="tool-desc"><strong>Raw Tail Turns:</strong> When the conversation reaches this limit, older turns are summarized into a rolling summary that maintains the full conversation context. This summary grows incrementally as the conversation continues, ensuring no context is lost while keeping the active window manageable.</div>');
    parts.push('<div class="tool-desc">Note: Input token limiting (summarizer_max_input_tokens) only applies to rewrite pre-summarization, not chunked history.</div>');
    summarizerOutputHelpPopup.innerHTML = parts.join('');
  }

  function showSummarizerOutputHelp() {
    if (!summarizerOutputHelpPopup || !summarizerOutputHelpLink) return;
    renderSummarizerOutputHelp();
    summarizerOutputHelpPopup.style.display = 'block';
    summarizerOutputHelpPopup.setAttribute('aria-hidden', 'false');
  }

  function hideSummarizerOutputHelp() {
    if (!summarizerOutputHelpPopup) return;
    summarizerOutputHelpPopup.style.display = 'none';
    summarizerOutputHelpPopup.setAttribute('aria-hidden', 'true');
  }

  function toggleSummarizerOutputHelp(evt) {
    evt && evt.preventDefault();
    if (!summarizerOutputHelpPopup) return;
    const isHidden = summarizerOutputHelpPopup.getAttribute('aria-hidden') !== 'false' && summarizerOutputHelpPopup.style.display !== 'block';
    if (isHidden) showSummarizerOutputHelp(); else hideSummarizerOutputHelp();
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
    if (e.key === 'Escape') {
      hideToolsHelp();
      hideSummarizerOutputHelp();
    }
  });
  if (helpLink) {
    helpLink.addEventListener('click', toggleToolsHelp);
  }
  if (summarizerOutputHelpLink) {
    summarizerOutputHelpLink.addEventListener('click', toggleSummarizerOutputHelp);
  }
  // Dismiss summarizer output help on outside click
  document.addEventListener('click', (e) => {
    if (!summarizerOutputHelpPopup || summarizerOutputHelpPopup.style.display !== 'block') return;
    const within = summarizerOutputHelpPopup.contains(e.target) || (summarizerOutputHelpLink && summarizerOutputHelpLink.contains(e.target));
    if (!within) hideSummarizerOutputHelp();
  });
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

    // Optionally hide Inference #2 (Tool Synthesis) block when all its metrics are zero.
    try {
      const block = document.getElementById('inference2_block');
      if (block) {
        const paths = [
          'turn_metrics.inference_tools_synth.input_tokens',
          'turn_metrics.inference_tools_synth.cached_tokens',
          'turn_metrics.inference_tools_synth.output_tokens',
          'turn_metrics.inference_tools_synth.cost_input',
          'turn_metrics.inference_tools_synth.cost_output',
          'turn_metrics.inference_tools_synth.cost_total',
        ];
        const anyNonZero = paths.some(p => {
          const v = deepGet(data, p, 0);
          return toNumber(v, 0) !== 0;
        });
        block.style.display = anyNonZero ? '' : 'none';
      }
    } catch (e) {
      console.debug('Failed to toggle inference2_block visibility', e);
    }

    // Backward-compat: if only legacy metrics were returned, map what we can.
    if (!data.turn_metrics && data.metrics) {
      const m = data.metrics;
      setText('prompt_tokens', toNumber(m.input_tokens || m.prompt_tokens, 0));
      setText('inference_cached_tokens', toNumber(m.cached_tokens || m.prompt_cached_tokens || 0, 0));
      setText('completion_tokens', toNumber(m.output_tokens || m.completion_tokens, 0));
      setText('reasoning_tokens', toNumber(m.reasoning_tokens || 0, 0));
      setText('total_tokens', toNumber(m.total_tokens, 0));
      setText('prompt_cost', formatCost(m.cost_input || m.prompt_cost));
      setText('completion_cost', formatCost(m.cost_output || m.completion_cost));
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
    wrapper.className = 'msg ' + role;

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

  function setAssistantBubbleHtml(bubble, html) {
    try {
      if (!bubble) return;
      bubble.classList.add('markdown');
      bubble.innerHTML = (html == null ? '' : String(html));
    } catch (e) {
      try {
        if (bubble) bubble.textContent = (html == null ? '' : String(html));
      } catch (_) {}
    }
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
      prompt_domain: (qs('#prompt_domain') ? String(qs('#prompt_domain').value || '') : ''),
      active_domain: (qs('#prompt_domain') ? String(qs('#prompt_domain').value || '') : ''),
      temperature: getNum('temperature'),
      max_output_tokens: getNum('max_output_tokens'),
      top_p: getNum('top_p'),
      raw_tail_turns: getNum('raw_tail_turns'),
      enable_query_rewrite: (qs('#enable_query_rewrite') ? !!qs('#enable_query_rewrite').checked : null),
      rewrite_confidence_threshold: getNum('rewrite_confidence_threshold'),
      rewrite_tail_turns: getNum('rewrite_tail_turns'),
      rewrite_summary_turns: getNum('rewrite_summary_turns'),
      use_tools,
    };

    try {
      const selectedDomain = String(base.active_domain || '').trim();
      if (selectedDomain) {
        localStorage.setItem('active_domain', selectedDomain);
      }
    } catch (e) {
      console.debug('Failed to persist active_domain', e);
    }

    // Feature flag (Option A): request backend-rendered HTML (additive).
    base.render_html = true;

    try {
      const showProcEl = qs('#show_processing_steps');
      if (showProcEl) {
        base.show_processing_steps = !!showProcEl.checked;
      }
    } catch (e) {
      console.debug('Failed to read show_processing_steps checkbox', e);
    }

    // Attach model keys per stage. These map to backend model registry.
    try {
      base.model_keys = {
        inference: stageModelConfig.inference.model_key,
        rewrite: stageModelConfig.rewrite.model_key,
        summary: stageModelConfig.summary.model_key,
        rerank: stageModelConfig.rerank.model_key
      };
      
      // Keep the old format for backward compatibility
      const getModelInfo = (key) => {
        const modelInfo = MODEL_REGISTRY[key] || {};
        return {
          provider: modelInfo.provider || '',
          model: modelInfo.model || key
        };
      };
      
      // Add legacy provider/model fields for backward compatibility
      const infInfo = getModelInfo(stageModelConfig.inference.model_key);
      const rwInfo = getModelInfo(stageModelConfig.rewrite.model_key);
      const sumInfo = getModelInfo(stageModelConfig.summary.model_key);
      const rrInfo = getModelInfo(stageModelConfig.rerank.model_key);
      
      base.inference_provider = infInfo.provider;
      base.inference_model = infInfo.model;
      base.rewrite_provider = rwInfo.provider;
      base.rewrite_model = rwInfo.model;
      base.summary_provider = sumInfo.provider;
      base.summary_model = sumInfo.model;
      base.rerank_provider = rrInfo.provider;
      base.rerank_model = rrInfo.model;
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
      const lockIds = ['raw_tail_turns'];
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

      // Clear bubble and render main answer text
      try {
        const answerHtml = data && (data.answer_html ?? data.answerHtml ?? data.response_html ?? data.responseHtml);
        if (answerHtml) setAssistantBubbleHtml(bubble, answerHtml);
        else bubble.textContent = displayText;
      } catch (e) {
        bubble.textContent = displayText;
      }

      // Optional: collapsible reasoning panel when backend provides it.
      if (data && data.reasoning) {
        const toggle = document.createElement('button');
        toggle.type = 'button';
        toggle.className = 'reasoning-toggle';
        toggle.textContent = 'Show reasoning ▾';

        const panel = document.createElement('div');
        panel.className = 'reasoning-panel';
        panel.textContent = data.reasoning;
        panel.style.display = 'none';

        toggle.addEventListener('click', () => {
          const isHidden = panel.style.display === 'none';
          panel.style.display = isHidden ? 'block' : 'none';
          toggle.textContent = isHidden ? 'Hide reasoning ▴' : 'Show reasoning ▾';
        });

        bubble.appendChild(document.createElement('br'));
        bubble.appendChild(toggle);
        bubble.appendChild(panel);
      }

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
        ['raw_tail_turns'].forEach(id => {
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
        inference_model: null,
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

          // Display active collection information
          try {
            const activeCollectionEl = document.getElementById('active_collection');
            if (activeCollectionEl && config.collection_name) {
              // Map collection names to user-friendly names
              const collectionNames = {
                'document_index': 'OpenAI (document_index)',
                'document_index_gemini': 'Gemini (document_index_gemini)'
              };
              const displayName = collectionNames[config.collection_name] || config.collection_name;
              activeCollectionEl.textContent = displayName;
              
              // Also show the active domain if available
              if (config.active_domain) {
                activeCollectionEl.textContent += ` [domain: ${config.active_domain}]`;
              }
            }
          } catch (e) {
            console.debug('Could not display active collection:', e);
          }

          // Hydrate stageModelConfig from *_model_key fields when present.
          // These keys are the source of truth for default model selection.
          try {
            if (config.embedding_model_key) {
              stageModelConfig.embedding.model_key = config.embedding_model_key;
            }
            if (config.inference_model_key) {
              stageModelConfig.inference.model_key = config.inference_model_key;
            }
            if (config.rewrite_model_key) {
              stageModelConfig.rewrite.model_key = config.rewrite_model_key;
            }
            if (config.summarizer_model_key) {
              stageModelConfig.summary.model_key = config.summarizer_model_key;
            }
            if (config.rerank_model_key) {
              stageModelConfig.rerank.model_key = config.rerank_model_key;
            }
          } catch (e) {
            console.debug('Failed to hydrate stageModelConfig from model_key fields', e);
          }
        }
      } catch (error) {
        console.error('Error loading model config:', error);
      }
      
      // Update the UI with the model information (excluding embedding which is handled by updateModelLabels)
      const modelElements = {
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
            raw_tail_turns: 'raw_tail_turns',
            max_inference_output_tokens: 'max_output_tokens',
            enable_tools: 'use_tools',
            enable_query_rewrite: 'enable_query_rewrite',
            rewrite_confidence_threshold: 'rewrite_confidence_threshold',
            rewrite_tail_turns: 'rewrite_tail_turns',
            rewrite_summary_turns: 'rewrite_summary_turns'
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

      // After config and keys are loaded, refresh the visible labels using
      // the hydrated stageModelConfig together with MODEL_REGISTRY.
      try {
        if (typeof updateModelLabels === 'function') {
          updateModelLabels();
        }
      } catch (e) {
        console.debug('Failed to update model labels after loading config', e);
      }
    } catch (error) {
      console.error('Error in loadModelConfig:', error);
    }
  }

  // Load model configuration and live model registry when the page loads
  document.addEventListener('DOMContentLoaded', async () => {
    try {
      // Prefer live registry (overwrites defaults in MODEL_REGISTRY / MODELS_BY_STAGE)
      await fetchModelRegistry();
    } catch (e) {
      console.error('Failed to fetch model registry', e);
    }
    try {
      await loadModelConfig();
    } catch (e) {
      console.error('Failed to load model configuration', e);
    }
    // After backend config/labels are loaded, wire and init the models modal.
    try { wireModelsModalEvents(); } catch (e) { console.debug('Failed to wire models modal', e); }
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

  // Trigger on unload paths
  window.addEventListener('beforeunload', _clearServerSummariesOnUnload);
  document.addEventListener('visibilitychange', () => {
    try {
      if (document.visibilityState === 'hidden') _clearServerSummariesOnUnload();
    } catch (_) {}
  });

  // Add a minimal metrics-subsection-title override (only margin-top)
  document.addEventListener('DOMContentLoaded', () => {
    try {
      document.querySelectorAll('.metrics-subsection-title').forEach(el => {
        el.style.marginTop = '1px';
      });
    } catch (e) {
      console.debug('Failed to apply metrics-subsection-title styles', e);
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

})();

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
            try {
              const finalHtml = payload.finalHtml || payload.final_html || payload.html || '';
              if (finalHtml) setAssistantBubbleHtml(bubble, finalHtml);
              else if (finalContent) bubble.textContent = finalContent;
            } catch (e) {
              if (finalContent) bubble.textContent = finalContent;
            }
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
