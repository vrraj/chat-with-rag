function ChatSection() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [session, setSession] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [webSearchEnabled, setWebSearchEnabled] = useState(false);
    const chatContainer = useRef(null);

    useEffect(() => {
        // Create new chat session on component mount
        createChatSession();
    }, []);

    useEffect(() => {
        // Scroll to bottom when messages update
        if (chatContainer.current) {
            chatContainer.current.scrollTop = chatContainer.current.scrollHeight;
        }
    }, [messages]);

    const createChatSession = async () => {
        try {
            const response = await fetch('/chat/session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
            });

            if (!response.ok) {
                throw new Error('Failed to create chat session');
            }

            const data = await response.json();
            setSession(data.session_id);
        } catch (err) {
            setError('Failed to create chat session');
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!input.trim()) return;

        setLoading(true);
        setError('');

        try {
            // Add user message to messages
            setMessages(prev => [...prev, {
                role: 'user',
                content: input.trim(),
                sources: []
            }]);

            // Send message to chat endpoint
            const response = await fetch(`/chat/${session}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: input.trim(),
                    use_web_search: webSearchEnabled
                }),
            });

            if (!response.ok) {
                throw new Error('Failed to get response');
            }

            const data = await response.json();
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: data.response,
                sources: data.sources
            }]);

            setInput('');
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleWebSearchToggle = () => {
        setWebSearchEnabled(!webSearchEnabled);
    };

    return (
        <div className="space-y-4">
            {/* Chat Messages */}
            <div ref={chatContainer} className="h-[600px] overflow-y-auto p-4 border rounded-lg">
                {messages.map((message, index) => (
                    <div key={index} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} mb-4`}>
                        <div className={`max-w-[80%] p-3 rounded-lg ${
                            message.role === 'user' ? 'bg-indigo-500 text-white' : 'bg-gray-100'
                        }`}>
                            {message.content}
                            {message.sources && message.sources.length > 0 && (
                                <div className="mt-2 text-sm text-gray-600">
                                    {message.sources.map((source, i) => (
                                        <p key={i}>
                                            Source {i + 1}:{' '}
                                            <a href={source.url} target="_blank" className="text-indigo-600 hover:text-indigo-500">
                                                {source.title || 'Click here'}
                                            </a>
                                        </p>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                ))}
            </div>

            {/* Web Search Toggle */}
            <label className="relative inline-flex items-center cursor-pointer mb-4">
                <input 
                    type="checkbox" 
                    checked={webSearchEnabled}
                    onChange={handleWebSearchToggle}
                    className="sr-only peer"
                />
                <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                <span className="ml-3 text-sm font-medium text-gray-900">Enable Web Search</span>
            </label>

            {/* Message Input */}
            <form onSubmit={handleSubmit} className="flex gap-4">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Type your message..."
                    className="flex-1 p-2 rounded-lg border border-gray-300 focus:border-indigo-500 focus:ring-indigo-500"
                    disabled={loading}
                />
                <button
                    type="submit"
                    disabled={loading || !input.trim()}
                    className="px-4 py-2 rounded-lg text-white bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    {loading ? 'Sending...' : 'Send'}
                </button>
            </form>

            {error && (
                <div className="text-red-600 text-sm mt-2">
                    {error}
                </div>
            )}
        </div>
    );
}

export default ChatSection;
