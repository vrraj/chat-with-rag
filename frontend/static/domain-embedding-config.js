let configData = null;

const els = {
  editor: document.getElementById('editor'),
  registryMeta: document.getElementById('registryMeta'),
  activeDomainMeta: document.getElementById('activeDomainMeta'),
  status: document.getElementById('status'),
  saveBtn: document.getElementById('saveBtn'),
  reloadBtn: document.getElementById('reloadBtn'),
};

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
  const domains = Array.isArray(configData.domains) ? configData.domains.join(', ') : '';
  els.activeDomainMeta.textContent = `Active domain: ${configData.active_domain || '(not set)'}${domains ? ` | Domains: ${domains}` : ''}`;
  setStatus('Domain embedding config loaded.', 'success');
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
  setStatus(`Saved successfully.${backup} ${data.message || ''}`.trim(), 'success');
  await loadConfig();
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

window.addEventListener('DOMContentLoaded', async () => {
  try {
    await loadConfig();
  } catch (err) {
    setStatus(`Failed to load config: ${err.message || String(err)}`, 'error');
  }
});
