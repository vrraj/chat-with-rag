# Website Chat Agent

A modular system for crawling, embedding, and chatting with website content using AI.

## Features

- Web crawling and content extraction
- Semantic search using embeddings
- Chat interface with GPT
- Real-time web search capabilities
- FastAPI backend
- Modern web interface

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your API keys:
```env
OPENAI_API_KEY=your_openai_key
```

3. Run the development server:
```bash
uvicorn backend.main:app --reload
```

## Project Structure

```
backend/
├── core/          # Core configuration and settings
├── crawler/       # URL crawling functionality
├── extractor/     # Content extraction from HTML
├── embeddings/    # Embedding generation and storage
├── db/           # Database operations
├── chat/         # Chat functionality with GPT
└── web/          # Web interface components
```

## Usage

1. Submit URLs to be crawled and indexed
2. Chat with the system to query the indexed content
3. Use web search fallback for additional context
