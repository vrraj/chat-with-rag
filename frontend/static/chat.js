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
  const metricIds = {
    prompt_tokens: 'input_tokens',
    completion_tokens: 'inference_tokens',
    total_tokens: null, // computed
    prompt_cost: 'input_cost',
    completion_cost: 'inference_cost',
    total_cost: 'total_cost',
  };

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
    return {
      top_k: getNum('top_k'),
      score_threshold: getNum('score_threshold'),
      temperature: getNum('temperature'),
      max_output_tokens: getNum('max_output_tokens'),
      presence_penalty: getNum('presence_penalty'),
      frequency_penalty: getNum('frequency_penalty'),
      top_p: getNum('top_p'),
    };
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

  // Update metrics spans from backend payload; fallback to '—'.
  function updateMetrics(m = {}) {
    const set = (id, value) => {
      const el = qs('#' + id);
      if (!el) return;
      const v = value;
      el.textContent = (v == null || Number.isNaN(v)) ? '—' : String(v);
    };
    set('prompt_tokens', m.prompt_tokens);
    set('completion_tokens', m.completion_tokens);
    set('total_tokens', m.total_tokens);
    set('prompt_cost', m.prompt_cost);
    set('completion_cost', m.completion_cost);
    set('total_cost', m.total_cost);
    set('rerank_tokens', m.rerank_tokens);
    set('rerank_cost', m.rerank_cost);
    set('vectors_retrieved', m.vectors_retrieved);
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

    // 1) Append user message
    appendMessage('user', text);

    // 2) Prepare payload before adding placeholder assistant
    const payload = {
      message: text,
      params: collectParams(),
      history: collectHistory(),
    };

    // 3) Add assistant placeholder to replace later
    const wrapper = document.createElement('div');
    wrapper.className = 'msg assistant';
    const badge = document.createElement('span');
    badge.className = 'badge';
    badge.textContent = 'Assistant';
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = 'Thinking…';
    wrapper.appendChild(badge);
    wrapper.appendChild(bubble);
    chatHistory.appendChild(wrapper);
    chatHistory.scrollTop = chatHistory.scrollHeight;

    // 4) Clear input and disable controls during request
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
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      const data = await resp.json();

      // Replace placeholder bubble with answer
      bubble.textContent = (data && data.answer) ? data.answer : '(no answer)';
      // Update metrics
      updateMetrics(data && data.metrics ? data.metrics : {});
    } catch (e) {
      bubble.textContent = 'Error. Try again.';
      toast('Request failed. Please try again.');
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
})();
