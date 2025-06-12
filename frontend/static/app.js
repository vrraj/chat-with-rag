document.addEventListener('DOMContentLoaded', () => {
    const chatContainer = document.getElementById('chatContainer');
    const messageInput = document.getElementById('messageInput');
    const chatForm = document.getElementById('chatForm');
    const urlForm = document.getElementById('urlForm');
    const webSearchToggle = document.getElementById('webSearchToggle');

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

    // Send chat message
    async function sendMessage(message) {
        showLoading(true);
        
        try {
            const response = await fetch('http://localhost:8000/chat', {
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

            const data = await response.json();
            
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
    chatForm.addEventListener('submit', async (e) => {
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
    urlForm.addEventListener('submit', async (e) => {
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
});
