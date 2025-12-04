document.addEventListener('DOMContentLoaded', () => {
    const searchForm = document.getElementById('searchForm');
    const searchResults = document.getElementById('searchResults');

    console.log('Search form initialized');
    console.log('Form elements:', {
        searchForm: !!searchForm,
        searchResults: !!searchResults,
        searchQuery: document.getElementById('searchQuery'),
        searchLimit: document.getElementById('searchLimit'),
        searchUrlFilter: document.getElementById('searchUrlFilter')
    });

    searchForm.addEventListener('submit', async (e) => {
        console.log('Search button clicked');
        e.preventDefault();
        
        const query = document.getElementById('searchQuery').value;
        const limit = parseInt(document.getElementById('searchLimit').value);
        const urlFilter = document.getElementById('searchUrlFilter').value;
        const scoreThresholdRaw = document.getElementById('scoreThreshold')?.value;
        const scoreThreshold = scoreThresholdRaw !== undefined && scoreThresholdRaw !== null
            ? parseFloat(scoreThresholdRaw)
            : undefined;

        console.log('Form submission triggered');
        console.log('Sending request with payload:', {
            query: query.trim() || null,
            query_filter: urlFilter ? { url: urlFilter } : null,
            limit
        });
        console.log('Form values:', { query, limit, urlFilter });

        if (!query.trim()) {
            alert('Please enter a search query');
            return;
        }

        try {
            searchResults.innerHTML = '<div class="text-center py-4">Searching...</div>';
            
            // Build request payload (include score_threshold only if valid number)
            const payload = {
                query: query.trim() || null,
                query_filter: urlFilter ? { url: urlFilter } : null,
                limit
            };
            if (Number.isFinite(scoreThreshold)) {
                payload.score_threshold = scoreThreshold;
            }

            const response = await fetch('/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });

            console.log('Request sent to /search');
            console.log('Response status:', response.status);
            console.log('Response ok:', response.ok);

            const responseData = await response.json();
            console.log('Response data:', responseData);

            const data = responseData;

            if (!response.ok) {
                throw new Error(data.detail || 'Search failed');
            }

            displaySearchResults(data.results);
        } catch (error) {
            searchResults.innerHTML = `
                <div class="text-center py-4 text-red-600">
                    Error: ${error.message}
                </div>
            `;
        }
    });

    function displaySearchResults(results) {
        if (!results || results.length === 0) {
            searchResults.innerHTML = `
                <div class="text-center py-4 text-gray-500">
                    No results found
                </div>
            `;
            return;
        }

        const resultsHtml = results.map(result => `
            <div class="search-result">
                <div class="metadata">
                    <span class="url">${result.payload.url}</span>
                    <span class="section">• ${result.payload.section}</span>
                    <span class="subsection">• ${result.payload.subsection}</span>
                    <span class="chunk-index">• Chunk: ${result.payload.chunk_index}</span>
                    <span class="score">• Score: ${(result.score * 100).toFixed(2)}%</span>
                </div>
                <div class="chunk-content">
                    ${result.payload.text}
                </div>
            </div>
        `).join('');

        searchResults.innerHTML = resultsHtml;
    }
});
