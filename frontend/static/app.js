// Default values (used as fallback if API is unavailable)
const DEFAULT_CONFIG = {
    html: {
        maxChunks: 0,
        skipSections: ["References", "External links", "See also", "Further reading"],
        estimate: true,
        forceDelete: false
    },
    pdf: {
        maxChunks: 0,
        skipSections: ["References", "External links", "Further reading", "Notes"],
        estimate: true,
        forceDelete: false
    },
    mediawiki: {
        maxChunks: 0,
        skipSections: ["References", "External links", "See also", "Further reading"],
        estimate: true,
        forceDelete: false,
        apiUrl: "https://en.wikipedia.org/w/api.php",
        userAgent: "WebsiteChatAgent/0.1 (contact@example.com)"
    }
};

// Current config (starts with defaults, updated from API)
let appConfig = JSON.parse(JSON.stringify(DEFAULT_CONFIG));

// Form element references
let pdfUrlInput, pdfFileInput, pdfMaxChunksInput, pdfSkipSectionsInput, pdfEstimateToggle, pdfForceDelete, pdfIndexBtn, pdfProgress;
let mwUrlInput, mwMaxChunksInput, mwSkipSectionsInput, mwApiUrlInput, mwUAInput, mwEstimateToggle, mwForceDelete, mwIndexBtn, mwProgress;
let htmlUrlInput, htmlMaxChunksInput, htmlSkipSectionsInput, htmlEstimateToggle, htmlForceDelete, htmlIndexBtn, htmlProgress;

// Initialize DOM references
function initializeElements() {
    // PDF elements
    pdfUrlInput = document.getElementById('pdfUrl');
    pdfFileInput = document.getElementById('pdfFile');
    pdfMaxChunksInput = document.getElementById('pdfMaxChunks');
    pdfSkipSectionsInput = document.getElementById('pdfSkipSections');
    pdfEstimateToggle = document.getElementById('pdfEstimateToggle');
    pdfForceDelete = document.getElementById('pdfForceDelete');
    pdfIndexBtn = document.getElementById('pdfIndexBtn');
    pdfProgress = document.getElementById('pdfProgress');

    // MediaWiki elements
    mwUrlInput = document.getElementById('mwUrl');
    mwMaxChunksInput = document.getElementById('mwMaxChunks');
    mwSkipSectionsInput = document.getElementById('mwSkipSections');
    mwApiUrlInput = document.getElementById('mwApiUrl');
    mwUAInput = document.getElementById('mwUA');
    mwEstimateToggle = document.getElementById('mwEstimateToggle');
    mwForceDelete = document.getElementById('mwForceDelete');
    mwIndexBtn = document.getElementById('mwIndexBtn');
    mwProgress = document.getElementById('mwProgress');

    // HTML elements
    htmlUrlInput = document.getElementById('htmlUrl');
    htmlMaxChunksInput = document.getElementById('htmlMaxChunks');
    htmlSkipSectionsInput = document.getElementById('htmlSkipSections');
    htmlEstimateToggle = document.getElementById('htmlEstimateToggle');
    htmlForceDelete = document.getElementById('htmlForceDelete');
    htmlIndexBtn = document.getElementById('htmlIndexBtn');
    htmlProgress = document.getElementById('htmlProgress');
}

// Apply configuration to form fields
function applyConfigToUI() {
    try {
        console.log('Applying config to UI:', appConfig);
        
        // HTML Form
        if (htmlMaxChunksInput) htmlMaxChunksInput.value = appConfig.html.maxChunks;
        if (htmlSkipSectionsInput) htmlSkipSectionsInput.value = appConfig.html.skipSections.join(', ');
        if (htmlEstimateToggle) htmlEstimateToggle.checked = appConfig.html.estimate;
        if (htmlForceDelete) htmlForceDelete.checked = appConfig.html.forceDelete;
        
        // PDF Form
        if (pdfMaxChunksInput) pdfMaxChunksInput.value = appConfig.pdf.maxChunks;
        if (pdfSkipSectionsInput && Array.isArray(appConfig.pdf.skipSections)) {
            pdfSkipSectionsInput.value = appConfig.pdf.skipSections.join(', ');
        }
        if (pdfEstimateToggle) pdfEstimateToggle.checked = appConfig.pdf.estimate;
        if (pdfForceDelete) pdfForceDelete.checked = appConfig.pdf.forceDelete;
        
        // MediaWiki Form
        if (mwMaxChunksInput) mwMaxChunksInput.value = appConfig.mediawiki.maxChunks;
        if (mwSkipSectionsInput) mwSkipSectionsInput.value = appConfig.mediawiki.skipSections.join(', ');
        if (mwEstimateToggle) mwEstimateToggle.checked = appConfig.mediawiki.estimate;
        if (mwForceDelete) mwForceDelete.checked = appConfig.mediawiki.forceDelete;
        if (mwApiUrlInput) mwApiUrlInput.value = appConfig.mediawiki.apiUrl;
        if (mwUAInput) mwUAInput.value = appConfig.mediawiki.userAgent;
    } catch (error) {
        console.error('Error applying config to UI:', error);
    }
}

// Fetch configuration from API
async function fetchConfig() {
    try {
        const response = await fetch('/api/config/api-defaults');
        if (response.ok) {
            const data = await response.json();
            // Merge with defaults to ensure all fields exist
            appConfig = {
                html: { ...DEFAULT_CONFIG.html, ...data.html },
                pdf: { ...DEFAULT_CONFIG.pdf, ...data.pdf },
                mediawiki: { ...DEFAULT_CONFIG.mediawiki, ...data.mediawiki }
            };
            console.log('Loaded config from API:', appConfig);
            return true;
        }
        throw new Error(`HTTP error! status: ${response.status}`);
    } catch (error) {
        console.warn('Could not load config from API, using defaults:', error);
        appConfig = JSON.parse(JSON.stringify(DEFAULT_CONFIG));
        return false;
    }
}

document.addEventListener('DOMContentLoaded', async () => {
    console.log('DOM fully loaded, initializing app...');
    
    // Initialize all DOM elements
    initializeElements();
    
    // Quick link buttons
    const openSearchBtn = document.getElementById('openSearchBtn');
    const openDebugBtn = document.getElementById('openDebugBtn');
    const openListDocsBtn = document.getElementById('openListDocsBtn');
    const openChatBtn = document.getElementById('openChatBtn');
    const openDeleteIndexBtn = document.getElementById('openDeleteIndexBtn');
    const openPromptRegistryBtn = document.getElementById('openPromptRegistryBtn');
    const openProcessBatchBtn = document.getElementById('openProcessBatchBtn');
    
    try {
        // Initialize the app
        console.log('Fetching config from API...');
        await fetchConfig();
        applyConfigToUI();
        console.log('App initialization complete');
    } catch (error) {
        console.error('Error during app initialization:', error);
    }

    // PDF indexing handler
    async function indexPdf() {
        try {
            if (pdfIndexBtn) {
                pdfIndexBtn.disabled = true;
                pdfIndexBtn.textContent = 'Indexing...';
            }
            if (pdfProgress) pdfProgress.textContent = 'Reading document...';
            
            const url = (pdfUrlInput?.value || '').trim();
            const file = pdfFileInput?.files?.[0];
            const maxChunks = parseInt(pdfMaxChunksInput?.value || '0', 10) || 0;
            const forceDelete = !!(pdfForceDelete && pdfForceDelete.checked);
            const estimate = !!(pdfEstimateToggle && pdfEstimateToggle.checked);
            const skipRaw = (pdfSkipSectionsInput?.value || '').trim();
            const skipSections = skipRaw ? skipRaw.split(',').map(s => s.trim()).filter(Boolean) : undefined;

            if (!file && !url) {
                alert('Provide either a PDF URL or upload a file.');
                return;
            }
           // if (file) {
           //     alert(`PDF indexing started for file: ${file.name}`);
           // }   else if (url) {
           //      alert(`PDF indexing started for URL: ${url}`);
           // }
           
            // For file uploads, read as base64
            let fileBase64;
            if (file) {
                fileBase64 = await new Promise((resolve, reject) => {
                    const reader = new FileReader();
                    reader.onload = () => resolve(reader.result.split(',')[1]);
                    reader.onerror = error => reject(error);
                    reader.readAsDataURL(file);
                });
            }

            const requestBody = {
                url: url || undefined,
                file: fileBase64,
                filename: file ? file.name : undefined,
                max_chunks: maxChunks,
                force_delete: forceDelete,
                estimate: estimate,
                skip_sections: skipSections
            };

            const resp = await fetch('/pdf', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody),
            });
            
            // Handle non-OK responses
            if (!resp.ok) {
                // Check if it's a 409 Conflict (already indexed)
                if (resp.status === 409) {
                    const data = await resp.json();
                    const warningHtml = `
                        <div class="p-3 mb-4 rounded-md bg-yellow-50 border-l-4 border-yellow-400">
                            <div class="flex">
                                <div class="flex-shrink-0">
                                    <svg class="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                                        <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
                                    </svg>
                                </div>
                                <div class="ml-3 flex items-center space-x-2">
                                    <p class="text-sm text-yellow-700">${data.message || 'This document has already been indexed'}</p>
                                    <span class="text-sm text-yellow-600">•</span>
                                    <p class="text-sm text-yellow-600">${data.hint || 'Use "Force delete existing" to reindex'}</p>
                                </div>
                            </div>
                        </div>
                    `;
                    if (pdfProgress) {
                        pdfProgress.insertAdjacentHTML('afterbegin', warningHtml);
                    }
                    return;
                }
                // For other errors, throw as before
                const t = await resp.text();
                throw new Error(t || 'Failed to index PDF');
            }
            
            const data = await resp.json();
            
            // Check for already_indexed flag in successful response (if backend returns it)
            if (data.already_indexed) {
                const warningHtml = `
                    <div class="p-3 mb-4 rounded-md bg-yellow-50 border-l-4 border-yellow-400">
                        <div class="flex">
                            <div class="flex-shrink-0">
                                <svg class="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                                    <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
                                </svg>
                            </div>
                            <div class="ml-3 flex items-center space-x-2">
                                <p class="text-sm text-yellow-700">${data.message || 'This document has already been indexed'}</p>
                                <span class="text-sm text-yellow-600">•</span>
                                <p class="text-sm text-yellow-600">${data.hint || 'Use "Force delete existing" to reindex'}</p>
                            </div>
                        </div>
                    </div>
                `;
                if (pdfProgress) {
                    pdfProgress.insertAdjacentHTML('afterbegin', warningHtml);
                }
                return;
            }
            
            if (estimate) {
                if (pdfProgress) {
                    const chunks = data.chunks_planned ?? 0;
                    const tokens = data.tokens_used ?? 0;
                    const cost = (tokens * 0.0000001).toFixed(8); // Using the same rate as MediaWiki
                    
                    pdfProgress.innerHTML = `
                        <div class="font-semibold text-gray-900">
                            Estimated chunks: ${chunks} | Tokens: ${tokens} | Estimated cost: $${cost}
                        </div>
                    `;
                }
            } else {
                const cost = (data.embedding_cost ?? 0);
                if (pdfProgress) {
                    pdfProgress.innerHTML = `
                        <div class="font-semibold text-gray-900">
                            Done. Vectors: ${data.vectors_indexed ?? 0} | Tokens: ${data.tokens_used ?? 0} | Cost: $${Number(cost).toFixed(6)}
                        </div>
                    `;
                }
            }
            if (pdfFileInput) pdfFileInput.value = '';
            if (pdfUrlInput) pdfUrlInput.value = '';
        } catch (err) {
            alert(err.message || String(err));
        } finally {
            if (pdfIndexBtn) {
                pdfIndexBtn.disabled = false;
                pdfIndexBtn.textContent = 'Index PDF';
            }
        }
    }
    pdfIndexBtn && pdfIndexBtn.addEventListener('click', indexPdf);

    // MediaWiki indexing handlers (if UI is present)
    async function indexMediaWiki() {
        try {
            if (mwIndexBtn) {
                mwIndexBtn.disabled = true;
                mwIndexBtn.textContent = 'Indexing...';
            }
            // Update progress directly on mwProgress element
            const url = (mwUrlInput?.value || '').trim();
            if (!url) { alert('Enter a MediaWiki URL'); return; }
            const maxChunks = parseInt(mwMaxChunksInput?.value || '0', 10) || 0;
            const skipRaw = (mwSkipSectionsInput?.value || '').trim();
            const skipSections = skipRaw ? skipRaw.split(',').map(s => s.trim()).filter(Boolean) : undefined;
            const apiUrl = (mwApiUrlInput?.value || '').trim();
            const ua = (mwUAInput?.value || '').trim();
            const estimate = !!(mwEstimateToggle && mwEstimateToggle.checked);
            const forceDelete = !!(mwForceDelete && mwForceDelete.checked);

            if (mwProgress) mwProgress.textContent = 'Reading document...';
            const requestBody = {
                url,
                max_chunks: maxChunks,
                skip_sections: skipSections,
                force_delete: forceDelete,
                api_url: apiUrl || undefined,
                user_agent: ua || undefined,
                estimate: estimate || undefined
            };
            const resp = await fetch('/mediawiki/url', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody),
            });
            
            // First, handle non-OK responses
            if (!resp.ok) {
                // Check if it's a 409 Conflict (already indexed)
                if (resp.status === 409) {
                    const data = await resp.json();
                    const warningHtml = `
                        <div class="p-3 mb-4 rounded-md bg-yellow-50 border-l-4 border-yellow-400">
                            <div class="flex">
                                <div class="flex-shrink-0">
                                    <svg class="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                                        <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
                                    </svg>
                                </div>
                                <div class="ml-3 flex items-center space-x-2">
                                    <p class="text-sm text-yellow-700">${data.message || 'This URL has already been indexed'}</p>
                                    <span class="text-sm text-yellow-600">•</span>
                                    <p class="text-sm text-yellow-600">${data.hint || 'Use "Force delete existing" to reindex'}</p>
                                </div>
                            </div>
                        </div>
                    `;
                    if (mwProgress) {
                        mwProgress.insertAdjacentHTML('afterbegin', warningHtml);
                    }
                    return;
                }
                // For other errors, throw as before
                const t = await resp.text();
                throw new Error(t || 'Failed to index MediaWiki');
            }

            // Handle successful responses
            const data = await resp.json();
            
            // Check for already_indexed flag in successful response (if backend returns it)
            if (data.already_indexed) {
                const warningHtml = `
                    <div class="p-3 mb-4 rounded-md bg-yellow-50 border-l-4 border-yellow-400">
                        <div class="flex">
                            <div class="flex-shrink-0">
                                <svg class="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                                    <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
                                </svg>
                            </div>
                            <div class="ml-3 flex items-center space-x-2">
                                <p class="text-sm text-yellow-700">${data.message || 'This URL has already been indexed'}</p>
                                <span class="text-sm text-yellow-600">•</span>
                                <p class="text-sm text-yellow-600">${data.hint || 'Use "Force delete existing" to reindex'}</p>
                            </div>
                        </div>
                    </div>
                `;
                if (mwProgress) {
                    mwProgress.insertAdjacentHTML('afterbegin', warningHtml);
                }
                return;
            }

            // Success handling with cost estimation
            if (estimate) {
                const embeddingCostPerMillion = 0.02; // $0.02 per 1M tokens
                const estimatedCost = ((data.tokens_used || 0) / 1000000) * embeddingCostPerMillion;
                if (mwProgress) {
                    mwProgress.innerHTML = `
                        <strong>
                            Estimated: ${data.chunks_planned ?? 0} chunks | 
                            ~${(data.tokens_used || 0).toLocaleString()} tokens | 
                            Cost: ~$${estimatedCost.toFixed(6)}
                        </strong>`;
                }
            } else {
                const cost = (data.embedding_cost ?? 0);
                if (mwProgress) {
                    mwProgress.innerHTML = `
                        <strong>
                            Done. Vectors: ${data.vectors_indexed ?? 0} | 
                            Tokens: ${data.tokens_used ?? 0} | 
                            Cost: $${Number(cost).toFixed(6)}
                        </strong>`;
                }
            }
        } catch (err) {
            alert(err.message || String(err));
        } finally {
            if (mwIndexBtn) {
                mwIndexBtn.disabled = false;
                mwIndexBtn.textContent = 'Index MediaWiki';
            }
        }
    }
    mwIndexBtn && mwIndexBtn.addEventListener('click', indexMediaWiki);
    openSearchBtn && openSearchBtn.addEventListener('click', () => window.open('/search', '_blank'));
    openDebugBtn && openDebugBtn.addEventListener('click', () => window.open('/debug_index', '_blank'));
    openDeleteIndexBtn && openDeleteIndexBtn.addEventListener('click', () => window.open('/delete_index', '_blank'));
    openPromptRegistryBtn && openPromptRegistryBtn.addEventListener('click', () => window.open('/prompt-registry', '_blank'));
    openListDocsBtn && openListDocsBtn.addEventListener('click', () => window.open('/list-docs.html', '_blank'));
    openChatBtn && openChatBtn.addEventListener('click', () => window.open('/chat.html', '_blank'));
    
    // Add event listener for batch upload button
    const openBatchUploadBtn = document.getElementById('openBatchUploadBtn');
    openBatchUploadBtn && openBatchUploadBtn.addEventListener('click', () => window.open('/process-batch-docs.html', '_blank'));
    openProcessBatchBtn && openProcessBatchBtn.addEventListener('click', () => window.open('/process-batch-docs.html', '_blank'));
    // HTML indexing (single URL)
    const htmlUrlInput = document.getElementById('htmlUrl');
    const htmlMaxChunksInput = document.getElementById('htmlMaxChunks');
    const htmlEstimateToggle = document.getElementById('htmlEstimateToggle');
    const htmlForceDelete = document.getElementById('htmlForceDelete');
    const htmlIndexBtn = document.getElementById('htmlIndexBtn');
    const htmlProgress = document.getElementById('htmlProgress');
    const htmlSkipSectionsInput = document.getElementById('htmlSkipSections');

    async function indexHtml() {
        try {
            htmlIndexBtn && (htmlIndexBtn.disabled = true, htmlIndexBtn.textContent = 'Indexing...');
            if (htmlProgress) htmlProgress.textContent = 'Reading document...';
            const url = (htmlUrlInput?.value || '').trim();
            if (!url) { alert('Enter a page URL'); return; }
            const maxChunks = parseInt(htmlMaxChunksInput?.value || '0', 10) || 0;
            const forceDelete = !!(htmlForceDelete && htmlForceDelete.checked);
            const estimate = !!(htmlEstimateToggle && htmlEstimateToggle.checked);

            // Get skip sections as comma-separated list and split into array
            const skipSections = (htmlSkipSectionsInput?.value || '')
                .split(',')
                .map(s => s.trim())
                .filter(Boolean);

            const resp = await fetch('/index', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    urls: [url],
                    doc_type: "HTML",
                    max_chunks: maxChunks,
                    force_delete: forceDelete,
                    estimate: estimate,
                    skip_sections: skipSections
                })
            });
            
            // Handle non-OK responses
            if (!resp.ok) {
                // Check if it's a 409 Conflict (already indexed)
                if (resp.status === 409) {
                    const data = await resp.json();
                    const warningHtml = `
                        <div class="p-3 mb-4 rounded-md bg-yellow-50 border-l-4 border-yellow-400">
                            <div class="flex">
                                <div class="flex-shrink-0">
                                    <svg class="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                                        <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
                                    </svg>
                                </div>
                                <div class="ml-3 flex items-center space-x-2">
                                    <p class="text-sm text-yellow-700">${data.message || 'This URL has already been indexed'}</p>
                                    <span class="text-sm text-yellow-600">•</span>
                                    <p class="text-sm text-yellow-600">${data.hint || 'Use "Force delete existing" to reindex'}</p>
                                </div>
                            </div>
                        </div>
                    `;
                    if (htmlProgress) {
                        htmlProgress.insertAdjacentHTML('afterbegin', warningHtml);
                    }
                    return;
                }
                // For other errors, throw as before
                const t = await resp.text();
                throw new Error(t || 'Failed to index HTML');
            }
            
            const data = await resp.json();
            
            // Check for already_indexed flag in successful response (if backend returns it)
            if (data.already_indexed) {
                const warningHtml = `
                    <div class="p-3 mb-4 rounded-md bg-yellow-50 border-l-4 border-yellow-400">
                        <div class="flex">
                            <div class="flex-shrink-0">
                                <svg class="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                                    <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
                                </svg>
                            </div>
                            <div class="ml-3 flex items-center space-x-2">
                                <p class="text-sm text-yellow-700">${data.message || 'This URL has already been indexed'}</p>
                                <span class="text-sm text-yellow-600">•</span>
                                <p class="text-sm text-yellow-600">${data.hint || 'Use "Force delete existing" to reindex'}</p>
                            </div>
                        </div>
                    </div>
                `;
                if (htmlProgress) {
                    htmlProgress.insertAdjacentHTML('afterbegin', warningHtml);
                }
                return;
            }
            
            if (estimate) {
                if (htmlProgress) {
                    const chunks = data.chunks_planned ?? 0;
                    const tokens = data.tokens_used ?? 0;
                    const estimatedCost = (tokens * 0.0000001).toFixed(8); // $0.10 per 1M tokens

                    // Build optional error details from backend (e.g., crawler HTTP errors)
                    let errorHtml = '';
                    if (Array.isArray(data.errors) && data.errors.length > 0) {
                        const items = data.errors.map(err => {
                            const url = err.url || 'Unknown URL';
                            const status = (err.status !== undefined && err.status !== null) ? `HTTP ${err.status}` : 'Error';
                            const msg = err.message || '';
                            return `<li><span class="font-medium">${url}</span> \u2192 <span class="text-gray-800">${status}</span>${msg ? ` \u2014 ${msg}` : ''}</li>`;
                        }).join('');
                        errorHtml = `
                            <div class="mt-2 text-sm text-red-700">
                                <div class="font-semibold">Some sources failed during indexing:</div>
                                <ul class="list-disc ml-5 mt-1">${items}</ul>
                            </div>
                        `;
                    }

                    htmlProgress.innerHTML = `
                        <div class="font-semibold text-gray-900">
                            Estimated: ${chunks} chunks | 
                            ~${tokens.toLocaleString()} tokens | 
                            Cost: ~$${estimatedCost}
                        </div>
                        ${errorHtml}
                    `;
                }
            } else {
                const cost = (data.embedding_cost ?? 0);
                if (htmlProgress) {
                    // Build optional error details from backend (e.g., crawler HTTP errors)
                    let errorHtml = '';
                    if (Array.isArray(data.errors) && data.errors.length > 0) {
                        const items = data.errors.map(err => {
                            const url = err.url || 'Unknown URL';
                            const status = (err.status !== undefined && err.status !== null) ? `HTTP ${err.status}` : 'Error';
                            const msg = err.message || '';
                            return `<li><span class="font-medium">${url}</span> \u2192 <span class="text-gray-800">${status}</span>${msg ? ` \u2014 ${msg}` : ''}</li>`;
                        }).join('');
                        errorHtml = `
                            <div class="mt-2 text-sm text-red-700">
                                <div class="font-semibold">Some sources failed during indexing:</div>
                                <ul class="list-disc ml-5 mt-1">${items}</ul>
                            </div>
                        `;
                    }

                    htmlProgress.innerHTML = `
                        <div class="font-semibold text-gray-900">
                            Done. Vectors: ${data.vectors_indexed ?? 0} | Tokens: ${data.tokens_used ?? 0} | Cost: $${Number(cost).toFixed(6)}
                        </div>
                        ${errorHtml}
                    `;
                }
            }
        } catch (err) {
            alert(err.message || String(err));
        } finally {
            if (htmlIndexBtn) {
                htmlIndexBtn.disabled = false;
                htmlIndexBtn.textContent = 'Index HTML';
            }
        }
    }
    htmlIndexBtn && htmlIndexBtn.addEventListener('click', indexHtml);

    // PDF progress handled directly in indexPdf
});
