let registryData = null;

const els = {
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
  setStatus(`Saved successfully.${backup}`, 'success');
}

function onAnyInputChanged() {
  persistCurrentTool();
  refreshEditors();
}

els.toolSelect.addEventListener('change', refreshEditors);
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

window.addEventListener('DOMContentLoaded', async () => {
  try {
    await loadRegistry();
  } catch (err) {
    setStatus(`Failed to load registry: ${err.message || String(err)}`, 'error');
  }
});
