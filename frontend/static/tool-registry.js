let registryData = null;

const els = {
  artifactInjectionEnabled: document.getElementById('artifactInjectionEnabled'),
  artifactAllowedTools: document.getElementById('artifactAllowedTools'),
  artifactMaxChars: document.getElementById('artifactMaxChars'),
  artifactEnforcePlaceholder: document.getElementById('artifactEnforcePlaceholder'),
  artifactAllowedTypes: document.getElementById('artifactAllowedTypes'),
  artifactAllowedModes: document.getElementById('artifactAllowedModes'),
  toolSelect: document.getElementById('toolSelect'),
  toolEnabled: document.getElementById('toolEnabled'),
  producesArtifact: document.getElementById('producesArtifact'),
  artifactType: document.getElementById('artifactType'),
  artifactKey: document.getElementById('artifactKey'),
  injectionMode: document.getElementById('injectionMode'),
  placeholder: document.getElementById('placeholder'),
  rawPreview: document.getElementById('rawPreview'),
  registryMeta: document.getElementById('registryMeta'),
  status: document.getElementById('status'),
  saveBtn: document.getElementById('saveBtn'),
  reloadBtn: document.getElementById('reloadBtn'),
  reloadCacheBtn: document.getElementById('reloadCacheBtn'),
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

function getTools() {
  const tools = registryData?.registry?.tools;
  return Array.isArray(tools) ? tools : [];
}

function ensureArtifactInjectionConfig() {
  if (!registryData || !registryData.registry || typeof registryData.registry !== 'object') {
    return { enabled: true, allowed_tools: [] };
  }
  if (!registryData.registry.artifact_injection || typeof registryData.registry.artifact_injection !== 'object') {
    registryData.registry.artifact_injection = {
      enabled: true,
      allowed_tools: [],
      security: {
        max_artifact_chars: 120000,
        allowed_artifact_types: ['svg'],
        allowed_injection_modes: ['verbatim'],
        enforce_placeholder_format: true,
      },
    };
  }
  const cfg = registryData.registry.artifact_injection;
  if (!Array.isArray(cfg.allowed_tools)) {
    cfg.allowed_tools = [];
  }
  if (typeof cfg.enabled !== 'boolean') {
    cfg.enabled = true;
  }
  if (!cfg.security || typeof cfg.security !== 'object') {
    cfg.security = {};
  }
  if (!Number.isInteger(cfg.security.max_artifact_chars) || cfg.security.max_artifact_chars <= 0) {
    cfg.security.max_artifact_chars = 120000;
  }
  if (!Array.isArray(cfg.security.allowed_artifact_types) || cfg.security.allowed_artifact_types.length === 0) {
    cfg.security.allowed_artifact_types = ['svg'];
  }
  if (!Array.isArray(cfg.security.allowed_injection_modes) || cfg.security.allowed_injection_modes.length === 0) {
    cfg.security.allowed_injection_modes = ['verbatim'];
  }
  if (typeof cfg.security.enforce_placeholder_format !== 'boolean') {
    cfg.security.enforce_placeholder_format = true;
  }
  return cfg;
}

function getSelectedTool() {
  const name = els.toolSelect.value;
  return getTools().find((t) => t && t.name === name) || null;
}

function ensureArtifact(tool) {
  if (!tool.artifact || typeof tool.artifact !== 'object') {
    tool.artifact = {
      produces_artifact: false,
      artifact_type: '',
      artifact_key: '',
      injection_mode: '',
      placeholder: '',
    };
  }
  return tool.artifact;
}

function refreshPreview() {
  els.rawPreview.value = JSON.stringify(registryData?.registry || {}, null, 2);
}

function refreshEditors() {
  const policy = ensureArtifactInjectionConfig();
  const security = policy.security || {};
  els.artifactInjectionEnabled.checked = !!policy.enabled;
  els.artifactAllowedTools.value = (policy.allowed_tools || []).join(', ');
  els.artifactMaxChars.value = Number.isInteger(security.max_artifact_chars) ? String(security.max_artifact_chars) : '120000';
  els.artifactEnforcePlaceholder.checked = !!security.enforce_placeholder_format;
  els.artifactAllowedTypes.value = (security.allowed_artifact_types || []).join(', ');
  els.artifactAllowedModes.value = (security.allowed_injection_modes || []).join(', ');

  const tool = getSelectedTool();
  if (!tool) {
    els.toolEnabled.checked = false;
    els.producesArtifact.checked = false;
    els.artifactType.value = '';
    els.artifactKey.value = '';
    els.injectionMode.value = '';
    els.placeholder.value = '';
    refreshPreview();
    return;
  }

  const artifact = ensureArtifact(tool);
  tool.enabled = tool.enabled !== false;

  els.toolEnabled.checked = !!tool.enabled;
  els.producesArtifact.checked = !!artifact.produces_artifact;
  els.artifactType.value = artifact.artifact_type || '';
  els.artifactKey.value = artifact.artifact_key || '';
  els.injectionMode.value = artifact.injection_mode || '';
  els.placeholder.value = artifact.placeholder || '';

  const disabled = !artifact.produces_artifact;
  els.artifactType.disabled = disabled;
  els.artifactKey.disabled = disabled;
  els.injectionMode.disabled = disabled;
  els.placeholder.disabled = disabled;

  refreshPreview();
}

function persistCurrentTool() {
  const policy = ensureArtifactInjectionConfig();
  policy.enabled = !!els.artifactInjectionEnabled.checked;
  policy.allowed_tools = (els.artifactAllowedTools.value || '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  if (!policy.security || typeof policy.security !== 'object') {
    policy.security = {};
  }
  const parsedMax = parseInt(els.artifactMaxChars.value || '120000', 10);
  policy.security.max_artifact_chars = Number.isInteger(parsedMax) && parsedMax > 0 ? parsedMax : 120000;
  policy.security.enforce_placeholder_format = !!els.artifactEnforcePlaceholder.checked;
  policy.security.allowed_artifact_types = (els.artifactAllowedTypes.value || '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  if (policy.security.allowed_artifact_types.length === 0) {
    policy.security.allowed_artifact_types = ['svg'];
  }
  policy.security.allowed_injection_modes = (els.artifactAllowedModes.value || '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  if (policy.security.allowed_injection_modes.length === 0) {
    policy.security.allowed_injection_modes = ['verbatim'];
  }

  const tool = getSelectedTool();
  if (!tool) return;

  tool.enabled = !!els.toolEnabled.checked;
  const artifact = ensureArtifact(tool);
  artifact.produces_artifact = !!els.producesArtifact.checked;
  artifact.artifact_type = (els.artifactType.value || '').trim();
  artifact.artifact_key = (els.artifactKey.value || '').trim();
  artifact.injection_mode = (els.injectionMode.value || '').trim();
  artifact.placeholder = (els.placeholder.value || '').trim();
}

function populateTools() {
  const toolNames = getTools().map((t) => t.name).filter(Boolean);
  els.toolSelect.innerHTML = toolNames.map((name) => `<option value="${name}">${name}</option>`).join('');
  if (toolNames.length > 0) {
    els.toolSelect.value = toolNames[0];
  }
}

async function loadRegistry() {
  setStatus('Loading tool registry...');
  const resp = await fetch('/api/tool-registry');
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(text || `HTTP ${resp.status}`);
  }
  registryData = await resp.json();
  populateTools();
  refreshEditors();
  els.registryMeta.textContent = `Registry: ${registryData.registry_path || '(unknown path)'}`;
  setStatus('Tool registry loaded.', 'success');
}

async function saveRegistry() {
  persistCurrentTool();

  setStatus('Saving tool registry...');
  const resp = await fetch('/api/tool-registry', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ registry: registryData.registry }),
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
  const cleared = Number(data?.cache_entries_cleared ?? 0);
  setStatus(`Saved successfully.${backup} Cache entries cleared: ${cleared}.`, 'success');
}

async function reloadToolCache() {
  setStatus('Reloading tool registry cache...');
  const resp = await fetch('/api/tool-registry/reload-cache', {
    method: 'POST',
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
  const cleared = Number(data?.cache_entries_cleared ?? 0);
  setStatus(`Tool registry cache reloaded. Cache entries cleared: ${cleared}.`, 'success');
}

function onAnyInputChanged() {
  persistCurrentTool();
  refreshEditors();
}

els.toolSelect.addEventListener('change', refreshEditors);
els.artifactInjectionEnabled.addEventListener('change', onAnyInputChanged);
els.artifactAllowedTools.addEventListener('input', onAnyInputChanged);
els.artifactMaxChars.addEventListener('input', onAnyInputChanged);
els.artifactEnforcePlaceholder.addEventListener('change', onAnyInputChanged);
els.artifactAllowedTypes.addEventListener('input', onAnyInputChanged);
els.artifactAllowedModes.addEventListener('input', onAnyInputChanged);
els.toolEnabled.addEventListener('change', onAnyInputChanged);
els.producesArtifact.addEventListener('change', onAnyInputChanged);
els.artifactType.addEventListener('input', onAnyInputChanged);
els.artifactKey.addEventListener('input', onAnyInputChanged);
els.injectionMode.addEventListener('input', onAnyInputChanged);
els.placeholder.addEventListener('input', onAnyInputChanged);

els.saveBtn.addEventListener('click', async () => {
  try {
    await saveRegistry();
  } catch (err) {
    setStatus(`Save failed: ${err.message || String(err)}`, 'error');
  }
});

els.reloadBtn.addEventListener('click', async () => {
  try {
    await loadRegistry();
  } catch (err) {
    setStatus(`Reload failed: ${err.message || String(err)}`, 'error');
  }
});

els.reloadCacheBtn.addEventListener('click', async () => {
  try {
    await reloadToolCache();
  } catch (err) {
    setStatus(`Reload cache failed: ${err.message || String(err)}`, 'error');
  }
});

window.addEventListener('DOMContentLoaded', async () => {
  try {
    await loadRegistry();
  } catch (err) {
    setStatus(`Failed to load registry: ${err.message || String(err)}`, 'error');
  }
});
