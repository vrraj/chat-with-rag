document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('retrievalEvalForm');
  const activeDomain = document.getElementById('activeDomain');
  const payloadPreview = document.getElementById('payloadPreview');
  const retrievalMeta = document.getElementById('retrievalMeta');
  const retrievalResults = document.getElementById('retrievalResults');
  const colbertMeta = document.getElementById('colbertMeta');
  const colbertResults = document.getElementById('colbertResults');
  const rerankMeta = document.getElementById('rerankMeta');
  const rerankResults = document.getElementById('rerankResults');

  function textFromItem(item) {
    const payload = (item && item.payload) || {};
    return payload.text || payload.snippet || payload.content || '';
  }

  function card(item, scoreLabel, scoreValue) {
    const payload = (item && item.payload) || {};
    const scoreText = scoreValue === null || scoreValue === undefined ? '' : `${scoreLabel}: ${Number(scoreValue).toFixed(4)}`;
    return `
      <div class="border rounded p-3 bg-gray-50">
        <div class="text-xs text-gray-600 mb-1">
          <span>${payload.url || 'unknown-url'}</span>
          <span> • ${payload.section || 'N/A'}</span>
          <span> • ${payload.subsection || 'N/A'}</span>
          <span> • chunk ${payload.chunk_index ?? 'N/A'}</span>
          ${scoreText ? `<span> • ${scoreText}</span>` : ''}
        </div>
        <div class="text-sm text-gray-900 whitespace-pre-wrap">${textFromItem(item)}</div>
      </div>
    `;
  }

  function setDomainOptions(domains) {
    const preferred = localStorage.getItem('active_domain') || '';
    activeDomain.innerHTML = `<option value="">(default)</option>${domains.map((d) => `<option value="${d}">${d}</option>`).join('')}`;
    if (preferred && domains.includes(preferred)) {
      activeDomain.value = preferred;
    }
  }

  async function loadDomains() {
    try {
      const res = await fetch('/api/domains');
      if (!res.ok) return;
      const data = await res.json();
      setDomainOptions(Array.isArray(data.domains) ? data.domains : []);
    } catch (_) {
      setDomainOptions(['default', 'mountains', 'oceans', 'finance']);
    }
  }

  activeDomain.addEventListener('change', () => {
    const val = String(activeDomain.value || '').trim();
    if (val) {
      localStorage.setItem('active_domain', val);
    } else {
      localStorage.removeItem('active_domain');
    }
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const query = String(document.getElementById('query').value || '').trim();
    if (!query) {
      alert('Query is required');
      return;
    }

    const urlFilter = String(document.getElementById('urlFilter').value || '').trim();
    const payload = {
      query,
      active_domain: String(activeDomain.value || '').trim() || undefined,
      search_mode: String(document.getElementById('searchMode').value || 'dense').trim(),
      top_k: Number(document.getElementById('topK').value || 8),
      score_threshold: Number(document.getElementById('scoreThreshold').value || 0.35),
      query_filter: urlFilter ? { url: urlFilter } : null,
      with_payload: !!document.getElementById('withPayload').checked,
      exact: !!document.getElementById('exact').checked,
      use_colbert: !!document.getElementById('useColbert').checked,
      colbert_top_n: Number(document.getElementById('colbertTopN').value || 8),
      enable_cross_encoder_rerank: !!document.getElementById('enableCrossEncoderRerank').checked,
      cross_encoder_top_n: Number(document.getElementById('crossEncoderTopN').value || 5),
    };

    payloadPreview.textContent = JSON.stringify(payload, null, 2);
    retrievalResults.innerHTML = '<div class="text-sm text-gray-500">Running retrieval...</div>';
    colbertResults.innerHTML = '';
    rerankResults.innerHTML = '';

    try {
      const res = await fetch('/retrieval-evals/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Pipeline failed');
      }

      const retrieval = data.retrieval || {};
      const retrievalItems = Array.isArray(retrieval.results) ? retrieval.results : [];
      retrievalMeta.textContent = `Requested mode: ${retrieval.requested_search_mode || 'n/a'} | Effective mode: ${retrieval.effective_search_mode || 'n/a'} | Returned: ${retrievalItems.length}`;
      retrievalResults.innerHTML = retrievalItems.map((item) => card(item, 'retrieval score', item.score)).join('') || '<div class="text-sm text-gray-500">No retrieval results</div>';

      const colbert = data.colbert;
      if (colbert) {
        const rows = Array.isArray(colbert.all_scored) ? colbert.all_scored : [];
        colbertMeta.textContent = `Model: ${colbert.model || 'n/a'} | Top-N: ${colbert.top_n ?? 'n/a'} | Returned: ${colbert.count_after_top_n ?? rows.length}`;
        colbertResults.innerHTML = rows.map((row) => card(row.item, 'colbert score', row.colbert_score)).join('') || '<div class="text-sm text-gray-500">No ColBERT results</div>';
      } else {
        colbertMeta.textContent = 'ColBERT disabled';
        colbertResults.innerHTML = '<div class="text-sm text-gray-500">Enable ColBERT to view section.</div>';
      }

      const reranked = data.reranked || {};
      const rerankRows = Array.isArray(reranked.items) ? reranked.items : [];
      rerankMeta.textContent = `Model: ${reranked.model || 'n/a'} | Returned: ${rerankRows.length}`;
      rerankResults.innerHTML = rerankRows.map((row) => card(row.item, 'cross-encoder score', row.cross_encoder_score)).join('') || '<div class="text-sm text-gray-500">No reranked results</div>';
    } catch (err) {
      retrievalResults.innerHTML = `<div class="text-sm text-red-600">Error: ${err.message}</div>`;
      colbertMeta.textContent = '';
      rerankMeta.textContent = '';
    }
  });

  loadDomains();
});
