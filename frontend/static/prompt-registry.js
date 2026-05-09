let registryData = null;

const els = {
  domainSelect: document.getElementById('domainSelect'),
  stageSelect: document.getElementById('stageSelect'),
  sectionSelect: document.getElementById('sectionSelect'),
  globalEditor: document.getElementById('globalEditor'),
  domainEditor: document.getElementById('domainEditor'),
  effectiveEditor: document.getElementById('effectiveEditor'),
  registryMeta: document.getElementById('registryMeta'),
  status: document.getElementById('status'),
  saveBtn: document.getElementById('saveBtn'),
  reloadBtn: document.getElementById('reloadBtn'),
  resetOverrideBtn: document.getElementById('resetOverrideBtn'),
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

function getFullPayloadTemplate(stageObj) {
  if (!stageObj || !Array.isArray(stageObj.user_messages)) {
    return '';
  }
  const fullPayload = stageObj.user_messages.find((x) => x && x.name === 'full_payload');
  return fullPayload && typeof fullPayload.template === 'string' ? fullPayload.template : '';
}

function setFullPayloadTemplate(stageObj, templateText) {
  if (!stageObj.user_messages || !Array.isArray(stageObj.user_messages)) {
    stageObj.user_messages = [];
  }
  const idx = stageObj.user_messages.findIndex((x) => x && x.name === 'full_payload');
  if (idx >= 0) {
    stageObj.user_messages[idx].template = templateText;
  } else {
    stageObj.user_messages.push({
      name: 'full_payload',
      enabled_if: true,
      template: templateText,
    });
  }
}

function getGlobalStage(stage) {
  return ((registryData || {}).registry || {}).global_defaults?.[stage] || {};
}

function getDomainStage(domain, stage, ensure = false) {
  if (!registryData || !registryData.registry) {
    return {};
  }
  if (!registryData.registry.domains) {
    registryData.registry.domains = {};
  }
  if (!registryData.registry.domains[domain]) {
    if (!ensure) {
      return {};
    }
    registryData.registry.domains[domain] = {};
  }
  if (!registryData.registry.domains[domain][stage]) {
    if (!ensure) {
      return {};
    }
    registryData.registry.domains[domain][stage] = {};
  }
  return registryData.registry.domains[domain][stage] || {};
}

function getCurrentSelection() {
  return {
    domain: els.domainSelect.value,
    stage: els.stageSelect.value,
    section: els.sectionSelect.value,
  };
}

function refreshEditors() {
  if (!registryData) return;

  const { domain, stage, section } = getCurrentSelection();
  const globalStage = getGlobalStage(stage);
  const domainStage = domain === '__global__' ? {} : getDomainStage(domain, stage, false);

  let globalText = '';
  let domainText = '';

  if (section === 'system_instruction') {
    globalText = globalStage.system_instruction || '';
    domainText = domainStage.system_instruction || '';
  } else {
    globalText = getFullPayloadTemplate(globalStage);
    domainText = getFullPayloadTemplate(domainStage);
  }

  els.globalEditor.value = globalText;
  els.domainEditor.value = domain === '__global__' ? '' : domainText;
  els.domainEditor.disabled = domain === '__global__';
  els.resetOverrideBtn.disabled = domain === '__global__';

  const effective = (domainText || '').trim() ? domainText : globalText;
  els.effectiveEditor.value = effective;
}

function persistCurrentDomainEdit() {
  if (!registryData) return;

  const { domain, stage, section } = getCurrentSelection();
  if (domain === '__global__') {
    return;
  }

  const stageObj = getDomainStage(domain, stage, true);
  const nextValue = els.domainEditor.value || '';

  if (section === 'system_instruction') {
    stageObj.system_instruction = nextValue;
  } else {
    setFullPayloadTemplate(stageObj, nextValue);
  }
}

function populateSelectors() {
  const stages = (registryData.stages || []).filter((s) => typeof s === 'string');
  const domains = ['__global__', ...(registryData.domains || [])];

  els.stageSelect.innerHTML = stages.map((s) => `<option value="${s}">${s}</option>`).join('');
  els.domainSelect.innerHTML = domains
    .map((d) => `<option value="${d}">${d === '__global__' ? 'global_defaults (no override)' : d}</option>`)
    .join('');

  const preferredStage = stages.includes('inference') ? 'inference' : stages[0];
  if (preferredStage) {
    els.stageSelect.value = preferredStage;
  }
  els.domainSelect.value = '__global__';
}

async function loadRegistry() {
  setStatus('Loading prompt registry...');
  const resp = await fetch('/api/prompt-registry');
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(text || `HTTP ${resp.status}`);
  }
  registryData = await resp.json();
  populateSelectors();
  refreshEditors();
  els.registryMeta.textContent = `Registry: ${registryData.registry_path || '(unknown path)'}`;
  setStatus('Prompt registry loaded.', 'success');
}

async function saveRegistry() {
  persistCurrentDomainEdit();

  setStatus('Saving prompt registry...');
  const resp = await fetch('/api/prompt-registry', {
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

function resetCurrentOverride() {
  const { domain, stage, section } = getCurrentSelection();
  if (domain === '__global__') {
    return;
  }

  const stageObj = getDomainStage(domain, stage, true);
  if (section === 'system_instruction') {
    stageObj.system_instruction = '';
  } else if (Array.isArray(stageObj.user_messages)) {
    stageObj.user_messages = stageObj.user_messages.filter((x) => !(x && x.name === 'full_payload'));
  }
  refreshEditors();
  setStatus(`Cleared override for ${domain}.${stage}.${section}`, 'warning');
}

els.domainEditor.addEventListener('input', () => {
  persistCurrentDomainEdit();
  refreshEditors();
});

els.domainSelect.addEventListener('change', () => {
  refreshEditors();
});

els.stageSelect.addEventListener('change', () => {
  refreshEditors();
});

els.sectionSelect.addEventListener('change', () => {
  refreshEditors();
});

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

els.resetOverrideBtn.addEventListener('click', () => {
  resetCurrentOverride();
});

window.addEventListener('DOMContentLoaded', async () => {
  try {
    await loadRegistry();
  } catch (err) {
    setStatus(`Failed to load registry: ${err.message || String(err)}`, 'error');
  }
});
