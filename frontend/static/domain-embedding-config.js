let configData = null;

const els = {
  editor: document.getElementById('editor'),
  registryMeta: document.getElementById('registryMeta'),
  activeDomainMeta: document.getElementById('activeDomainMeta'),
  activeDomainSelect: document.getElementById('activeDomainSelect'),
  applyActiveDomainBtn: document.getElementById('applyActiveDomainBtn'),
  status: document.getElementById('status'),
  saveBtn: document.getElementById('saveBtn'),
  reloadBtn: document.getElementById('reloadBtn'),
};

function populateActiveDomainSelector(domains, activeDomain) {
  if (!els.activeDomainSelect) return;
  const options = Array.isArray(domains) ? domains : [];
  els.activeDomainSelect.innerHTML = options
    .map((d) => `<option value="${d}">${d}</option>`)
    .join('');
  if (activeDomain && options.includes(activeDomain)) {
    els.activeDomainSelect.value = activeDomain;
  }
}

function setStatus(message, kind = 'info') {
  const color = {
    info: 'text-gray-700',
    success: 'text-green-700',
    error: 'text-red-700',
    warning: 'text-yellow-700',
  }[kind] || 'text-gray-700';
  els.status.className = `text-sm ${color}`;
  els.status.textContent = message || '';
}

async function loadConfig() {
  setStatus('Loading domain embedding config...');
  const resp = await fetch('/api/domain-embedding-config');
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(text || `HTTP ${resp.status}`);
  }

  configData = await resp.json();
  const registry = configData.registry || {};
  els.editor.value = JSON.stringify(registry, null, 2);
  els.registryMeta.textContent = `Registry: ${configData.registry_path || '(unknown path)'}`;
  const domainList = Array.isArray(configData.domains) ? configData.domains : [];
  const domains = domainList.join(', ');
  populateActiveDomainSelector(domainList, configData.active_domain);
  els.activeDomainMeta.textContent = `Active domain: ${configData.active_domain || '(not set)'}${domains ? ` | Domains: ${domains}` : ''}`;
  setStatus('Domain embedding config loaded.', 'success');
}

async function applyActiveDomain() {
  const domain = String(els.activeDomainSelect?.value || '').trim();
  if (!domain) {
    throw new Error('Please select an active domain');
  }

  setStatus(`Applying active domain: ${domain} ...`);
  const resp = await fetch('/api/domain-embedding-config/active-domain', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ active_domain: domain }),
  });

  if (!resp.ok) {
    let detail = '';
    try {
      const data = await resp.json();
      detail = data.detail || JSON.stringify(data);
    } catch (_) {
      detail = await resp.text();
    }
    throw new Error(detail || `HTTP ${resp.status}`);
  }

  const data = await resp.json();
  setStatus(data.message || `Active domain set to ${domain}.`, 'success');
  await loadConfig();
}

async function saveConfig() {
  let parsed;
  try {
    parsed = JSON.parse(els.editor.value || '{}');
  } catch (err) {
    throw new Error(`Invalid JSON: ${err.message || String(err)}`);
  }

  setStatus('Saving domain embedding config...');
  const resp = await fetch('/api/domain-embedding-config', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ registry: parsed }),
  });

  if (!resp.ok) {
    let detail = '';
    try {
      const data = await resp.json();
      detail = data.detail || JSON.stringify(data);
    } catch (_) {
      detail = await resp.text();
    }
    throw new Error(detail || `HTTP ${resp.status}`);
  }

  const data = await resp.json();
  const backup = data && data.backup_path ? ` Backup: ${data.backup_path}` : '';
  setStatus(`Saved successfully.${backup} Applying active domain...`.trim(), 'success');

  const selectedDomain = String(els.activeDomainSelect?.value || '').trim();
  if (selectedDomain) {
    const applyResp = await fetch('/api/domain-embedding-config/active-domain', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ active_domain: selectedDomain }),
    });

    if (!applyResp.ok) {
      let detail = '';
      try {
        const applyData = await applyResp.json();
        detail = applyData.detail || JSON.stringify(applyData);
      } catch (_) {
        detail = await applyResp.text();
      }
      throw new Error(`Config saved but failed to apply active domain '${selectedDomain}': ${detail || `HTTP ${applyResp.status}`}`);
    }

    await applyResp.json();
  }

  await loadConfig();
  setStatus(`Saved and applied active domain${selectedDomain ? ` '${selectedDomain}'` : ''}.${backup}`.trim(), 'success');
}

els.saveBtn.addEventListener('click', async () => {
  try {
    await saveConfig();
  } catch (err) {
    setStatus(`Save failed: ${err.message || String(err)}`, 'error');
  }
});

els.reloadBtn.addEventListener('click', async () => {
  try {
    await loadConfig();
  } catch (err) {
    setStatus(`Reload failed: ${err.message || String(err)}`, 'error');
  }
});

els.applyActiveDomainBtn.addEventListener('click', async () => {
  try {
    await applyActiveDomain();
  } catch (err) {
    setStatus(`Failed to apply active domain: ${err.message || String(err)}`, 'error');
  }
});

window.addEventListener('DOMContentLoaded', async () => {
  try {
    await loadConfig();
  } catch (err) {
    setStatus(`Failed to load config: ${err.message || String(err)}`, 'error');
  }
});
