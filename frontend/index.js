import SearchSection from './src/components/SearchSection';
import ChatSection from './src/components/ChatSection';

document.addEventListener('DOMContentLoaded', () => {
    let urls = [];
    const MAX_URLS = 10;

    // Elements
    const urlList = document.getElementById('urlList');
    const newUrlInput = document.getElementById('newUrl');
    const docTypeSelect = document.getElementById('docType');
    const addUrlBtn = document.getElementById('addUrlBtn');
    const indexBtn = document.getElementById('indexBtn');
    const maxChunksInput = document.getElementById('maxChunksInput');
    const estimateToggle = document.getElementById('estimateToggle');
    // PDF UI elements
    const pdfUrlInput = document.getElementById('pdfUrl');
    const pdfFileInput = document.getElementById('pdfFile');
    const pdfMaxChunksInput = document.getElementById('pdfMaxChunks');
    const pdfEstimateToggle = document.getElementById('pdfEstimateToggle');
    const pdfForceDelete = document.getElementById('pdfForceDelete');
    const pdfIndexBtn = document.getElementById('pdfIndexBtn');
    const container = document.querySelector('.container');

    // Add Search Section
    const searchSection = document.createElement('div');
    searchSection.innerHTML = SearchSection();
    container.insertBefore(searchSection, document.querySelector('.bg-white.rounded-lg.shadow-md.p-6.mb-8'));

    // Add Chat Section
    const chatSection = document.createElement('div');
    chatSection.innerHTML = ChatSection();
    container.insertBefore(chatSection, document.querySelector('.bg-white.rounded-lg.shadow-md.p-6.mb-8'));

    // Event Listeners
    addUrlBtn.addEventListener('click', addUrl);
    indexBtn.addEventListener('click', indexContent);
    if (pdfIndexBtn) pdfIndexBtn.addEventListener('click', indexPdf);

    // URL Management
    function addUrl() {
        const url = newUrlInput.value.trim();
        const docType = docTypeSelect.value;

        if (!url) {
            alert('Please enter a URL');
            return;
        }

        if (urls.length >= MAX_URLS) {
            alert('Maximum 10 URLs allowed');
            return;
        }

        // Validate URL format
        try {
            new URL(url);
        } catch (error) {
            alert('Please enter a valid URL');
            return;
        }

        // Add URL to list
        urls.push({ url, type: docType });
        updateUrlList();
        newUrlInput.value = '';
    }

    function removeUrl(index) {
        urls.splice(index, 1);
        updateUrlList();
    }

    function updateUrlList() {
        urlList.innerHTML = '';
        urls.forEach((urlObj, index) => {
            const urlItem = document.createElement('div');
            urlItem.className = 'flex items-center justify-between p-4 bg-gray-50 rounded-lg';
            
            const urlText = document.createElement('div');
            urlText.className = 'flex-1';
            urlText.innerHTML = `
                <h3 class="text-sm font-medium text-gray-900">${urlObj.url}</h3>
                <p class="text-xs text-gray-500">Type: ${urlObj.type}</p>
            `;

            const removeBtn = document.createElement('button');
            removeBtn.className = 'text-red-500 hover:text-red-700';
            removeBtn.innerHTML = 'Remove';
            removeBtn.onclick = () => removeUrl(index);

            urlItem.appendChild(urlText);
            urlItem.appendChild(removeBtn);
            urlList.appendChild(urlItem);
        });
    }

    // Indexing
    async function indexContent() {
        if (urls.length === 0) {
            alert('Please add at least one URL');
            return;
        }

        try {
            indexBtn.disabled = true;
            indexBtn.textContent = 'Indexing...';

            const estimate = !!(estimateToggle && estimateToggle.checked);
            const response = await fetch(`http://localhost:8000/index?estimate=${estimate}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    urls: urls.map(urlObj => urlObj.url),
                    types: urls.map(urlObj => urlObj.type),
                    max_chunks: parseInt(maxChunksInput?.value || '0', 10) || 0
                }),
            });

            if (!response.ok) {
                throw new Error('Failed to index content');
            }

            const data = await response.json();
            if (estimate) {
                alert(`Estimate: ${data.chunks_planned ?? 0} chunk(s) would be indexed.`);
            } else {
                alert('Content indexed successfully!');
            }
            urls = [];
            updateUrlList();
        } catch (error) {
            alert(error.message);
        } finally {
            indexBtn.disabled = false;
            indexBtn.textContent = 'Index Content';
        }
    }

    // Index structured PDF via /pdf (URL or Upload)
    async function indexPdf() {
        try {
            pdfIndexBtn.disabled = true;
            pdfIndexBtn.textContent = 'Indexing...';

            const fd = new FormData();
            const pdfUrl = (pdfUrlInput?.value || '').trim();
            const file = pdfFileInput?.files?.[0];
            if (!file && !pdfUrl) {
                alert('Provide either a PDF URL or upload a file.');
                return;
            }
            if (file) fd.append('file', file);
            if (pdfUrl) fd.append('url', pdfUrl);
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
            } else {
                alert(`PDF indexed successfully${data.source ? ` from ${data.source}` : ''}.`);
            }
            // Reset inputs
            if (pdfFileInput) pdfFileInput.value = '';
            if (pdfUrlInput) pdfUrlInput.value = '';
        } catch (err) {
            alert(err.message || String(err));
        } finally {
            pdfIndexBtn.disabled = false;
            pdfIndexBtn.textContent = 'Index PDF';
        }
    }
});
