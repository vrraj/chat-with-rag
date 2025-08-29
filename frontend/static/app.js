document.addEventListener('DOMContentLoaded', () => {
    const chatContainer = document.getElementById('chatContainer');
    const messageInput = document.getElementById('messageInput');
    const chatForm = document.getElementById('chatForm');
    const urlForm = document.getElementById('urlForm');
    const webSearchToggle = document.getElementById('webSearchToggle');
    // PDF elements (optional; only present on index.html we ship)
    const pdfUrlInput = document.getElementById('pdfUrl');
    const pdfFileInput = document.getElementById('pdfFile');
    const pdfMaxChunksInput = document.getElementById('pdfMaxChunks');
    const pdfEstimateToggle = document.getElementById('pdfEstimateToggle');
    const pdfForceDelete = document.getElementById('pdfForceDelete');
    const pdfIndexBtn = document.getElementById('pdfIndexBtn');
    const pdfProgress = document.getElementById('pdfProgress');

    // MediaWiki elements (for Index MediaWiki section)
    const mwUrlInput = document.getElementById('mwUrl');
    const mwMaxChunksInput = document.getElementById('mwMaxChunks');
    const mwSkipSectionsInput = document.getElementById('mwSkipSections');
    const mwApiUrlInput = document.getElementById('mwApiUrl');
    const mwUAInput = document.getElementById('mwUA');
    const mwEstimateToggle = document.getElementById('mwEstimateToggle');
    const mwForceDelete = document.getElementById('mwForceDelete');
    const mwIndexBtn = document.getElementById('mwIndexBtn');
    const mwProgress = document.getElementById('mwProgress');

    // Quick link buttons
    const openSearchBtn = document.getElementById('openSearchBtn');
    const openDebugBtn = document.getElementById('openDebugBtn');

    // Chat state
    let chatHistory = [];
    let currentContext = [];

    // Add message to chat
    function addMessage(message, isUser = true) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user-message' : 'assistant-message'} ${isUser ? 'ml-auto' : 'mr-auto'}`;
        messageDiv.textContent = message;
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    // Add sources to message
    function addSources(sources) {
        if (!sources || sources.length === 0) return;

        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'mt-2';

        sources.forEach((source, index) => {
            const sourceDiv = document.createElement('div');
            sourceDiv.className = 'text-sm text-gray-600';
            
            if (source.url) {
                sourceDiv.innerHTML = `
                    <span class="font-medium">Source ${index + 1}:</span>
                    <a href="${source.url}" target="_blank" class="source-link">${source.title || 'Click here'}</a>
                `;
            } else {
                sourceDiv.textContent = `Source ${index + 1}: ${source.text}`;
            }
            sourcesDiv.appendChild(sourceDiv);
        });

        chatContainer.appendChild(sourcesDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    // Show loading indicator
    function showLoading(isLoading) {
        const button = chatForm.querySelector('button');
        if (isLoading) {
            button.innerHTML = '<div class="loading">Sending...</div>';
            button.disabled = true;
        } else {
            button.innerHTML = 'Send';
            button.disabled = false;
        }
    }

    // Send chat message with streaming support
    async function sendMessage(message) {
        showLoading(true);
        
        try {
            const response = await fetch('http://localhost:8000/chat_with_content', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message,
                    context: currentContext,
                    use_web_search: webSearchToggle.checked
                })
            });

            // Create message container
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message assistant-message ml-auto';
            chatContainer.appendChild(messageDiv);
            
            // Create text node for streaming content
            const textNode = document.createTextNode('');
            messageDiv.appendChild(textNode);

            // Handle streaming response
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value, { stream: true });
                textNode.appendData(chunk);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }

            // Get full response with sources
            const fullResponse = await fetch('http://localhost:8000/chat_with_content_full', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message,
                    context: currentContext,
                    use_web_search: webSearchToggle.checked
                })
            });
            const data = await fullResponse.json();
            
            // Update chat history
            chatHistory.push({
                role: 'user',
                content: message
            });
            chatHistory.push({
                role: 'assistant',
                content: data.response
            });

            // Update current context
            currentContext = data.sources;

            return data;
        } catch (error) {
            console.error('Error:', error);
            throw error;
        } finally {
            showLoading(false);
        }
    }

    // Handle chat form submission
    chatForm && chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const message = messageInput.value.trim();
        if (!message) return;

        addMessage(message, true);
        messageInput.value = '';

        try {
            const response = await sendMessage(message);
            addMessage(response.response, false);
            addSources(response.sources);
        } catch (error) {
            addMessage('Sorry, there was an error processing your request.', false);
        }
    });

    // Handle URL form submission
    urlForm && urlForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const url = document.getElementById('websiteUrl').value;
        const depth = parseInt(document.getElementById('depth').value);

        try {
            const response = await fetch('http://localhost:8000/index', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    url,
                    depth
                })
            });

            const data = await response.json();
            addMessage(`Successfully indexed website: ${url}`, false);
        } catch (error) {
            console.error('Error:', error);
            addMessage('Error indexing website. Please try again.', false);
        }
    });

    // Simple PDF indexing handler if PDF UI is present
    async function indexPdf() {
        try {
            if (pdfIndexBtn) {
                pdfIndexBtn.disabled = true;
                pdfIndexBtn.textContent = 'Indexing...';
            }
            const fd = new FormData();
            const url = (pdfUrlInput?.value || '').trim();
            const file = pdfFileInput?.files?.[0];
            if (!file && !url) {
                alert('Provide either a PDF URL or upload a file.');
                return;
            }
            if (file) fd.append('file', file);
            if (url) fd.append('url', url);
            const maxChunksVal = parseInt(pdfMaxChunksInput?.value || '0', 10) || 0;
            fd.append('max_chunks', String(maxChunksVal));
            fd.append('force_delete', String(!!(pdfForceDelete && pdfForceDelete.checked)));
            const estimate = !!(pdfEstimateToggle && pdfEstimateToggle.checked);
            const resp = await fetch(`http://localhost:8000/pdf?estimate=${estimate}`, {
                method: 'POST',
                body: fd,
            });
            if (!resp.ok) {
                const t = await resp.text();
                throw new Error(t || 'Failed to index PDF');
            }
            const data = await resp.json();
            if (estimate) {
                alert(`PDF estimate: ${data.chunks_planned ?? 0} chunk(s).`);
                if (pdfProgress) pdfProgress.textContent = `Estimated chunks: ${data.chunks_planned ?? 0}`;
            } else {
                alert(`PDF indexed successfully${data.source ? ` from ${data.source}` : ''}.`);
                if (pdfProgress) {
                    const cost = (data.embedding_cost ?? 0);
                    pdfProgress.textContent = `Done. Vectors: ${data.vectors_indexed ?? 0} | Tokens: ${data.tokens_used ?? 0} | Cost: $${Number(cost).toFixed(6)}`;
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
            const prog = (t) => { const el = document.getElementById('mwProgress'); if (el) el.textContent = t; };
            const url = (mwUrlInput?.value || '').trim();
            if (!url) { alert('Enter a MediaWiki URL'); return; }
            const maxChunks = parseInt(mwMaxChunksInput?.value || '0', 10) || 0;
            const skipRaw = (mwSkipSectionsInput?.value || '').trim();
            const skipSections = skipRaw ? skipRaw.split(',').map(s => s.trim()).filter(Boolean) : undefined;
            const apiUrl = (mwApiUrlInput?.value || '').trim();
            const ua = (mwUAInput?.value || '').trim();
            const estimate = !!(mwEstimateToggle && mwEstimateToggle.checked);
            const force_delete = !!(mwForceDelete && mwForceDelete.checked);

            const qs = new URLSearchParams();
            if (apiUrl) qs.set('api_url', apiUrl);
            if (ua) qs.set('ua', ua);
            if (estimate) qs.set('estimate', 'true');
            prog('Reading document...');
            const resp = await fetch(`http://localhost:8000/mediawiki/url${qs.toString() ? '?' + qs.toString() : ''}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url, max_chunks: maxChunks, skip_sections: skipSections, force_delete }),
            });
            if (!resp.ok) {
                const t = await resp.text();
                throw new Error(t || 'Failed to index MediaWiki');
            }
            const data = await resp.json();
            if (estimate) {
                alert(`MediaWiki estimate: ${data.chunks_planned ?? 0} chunk(s).`);
                prog(`Estimated chunks: ${data.chunks_planned ?? 0}`);
            } else {
                alert(`MediaWiki indexed successfully. Chunks: ${data.chunks_indexed ?? 'n/a'}`);
                const cost = (data.embedding_cost ?? 0);
                prog(`Done. Vectors: ${data.vectors_indexed ?? 0} | Tokens: ${data.tokens_used ?? 0} | Cost: $${Number(cost).toFixed(6)}`);
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
    // HTML indexing (single URL)
    const htmlUrlInput = document.getElementById('htmlUrl');
    const htmlMaxChunksInput = document.getElementById('htmlMaxChunks');
    const htmlEstimateToggle = document.getElementById('htmlEstimateToggle');
    const htmlForceDelete = document.getElementById('htmlForceDelete');
    const htmlIndexBtn = document.getElementById('htmlIndexBtn');
    const htmlProgress = document.getElementById('htmlProgress');

    async function indexHtml() {
        try {
            htmlIndexBtn && (htmlIndexBtn.disabled = true, htmlIndexBtn.textContent = 'Indexing...');
            if (htmlProgress) htmlProgress.textContent = 'Reading document...';
            const url = (htmlUrlInput?.value || '').trim();
            if (!url) { alert('Enter a page URL'); return; }
            const maxChunks = parseInt(htmlMaxChunksInput?.value || '0', 10) || 0;
            const force_delete = !!(htmlForceDelete && htmlForceDelete.checked);
            const estimate = !!(htmlEstimateToggle && htmlEstimateToggle.checked);
            if (htmlProgress) htmlProgress.textContent = estimate ? 'Chunking document...' : 'Embedding document...';
            const resp = await fetch(`/index?estimate=${estimate}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ urls: [url], doc_type: 'HTML', max_chunks: maxChunks, force_delete })
            });
            if (!resp.ok) {
                const t = await resp.text();
                throw new Error(t || 'Failed to index HTML');
            }
            const data = await resp.json();
            if (estimate) {
                alert(`HTML estimate: ${data.chunks_planned ?? 0} chunk(s).`);
                if (htmlProgress) htmlProgress.textContent = `Estimated chunks: ${data.chunks_planned ?? 0}`;
            } else {
                alert('HTML indexed successfully.');
                if (htmlProgress) {
                    const cost = (data.embedding_cost ?? 0);
                    htmlProgress.textContent = `Done. Vectors: ${data.vectors_indexed ?? 0} | Tokens: ${data.tokens_used ?? 0} | Cost: $${Number(cost).toFixed(6)}`;
                }
            }
        } catch (err) {
            alert(err.message || String(err));
        } finally {
            htmlIndexBtn && (htmlIndexBtn.disabled = false, htmlIndexBtn.textContent = 'Index HTML');
        }
    }
    htmlIndexBtn && htmlIndexBtn.addEventListener('click', indexHtml);

    // Enhance PDF progress
    if (pdfIndexBtn) {
        const origIndexPdf = indexPdf;
        indexPdf = async function() {
            try {
                if (pdfProgress) pdfProgress.textContent = 'Reading document...';
            } catch {}
            await origIndexPdf();
        };
    }
});
