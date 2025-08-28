# Website and PDF Document Chat Agent

An intelligent chat agent that enables natural language interaction with both website content and PDF documents using advanced AI technologies.

## Features

- 📚 Document Processing
  - Extract and process content from both websites and PDF documents
  - Preserve document structure and formatting
  - Handle both structured and unstructured content

- 🧠 Intelligent Search
  - Semantic search using OpenAI embeddings
  - Vector database storage with Qdrant
  - Smart text chunking for better search results

- 💬 Natural Language Chat
  - Chat with documents using GPT-4.1-mini
  - Context-aware responses
  - Real-time search across multiple documents

## Usage

1. Start the application:
```bash
python run.py
```

2. Access the web interface at `http://localhost:8000`

3. Key Features:
   - Upload and process PDF documents
   - Submit website URLs for crawling and processing
   - Chat with the system using natural language
   - Get context-aware responses about your documents
   - Search across multiple documents simultaneously

## Prerequisites

- Python 3.10+
- Docker and Docker Compose (for production)
- OpenAI API Key
- Qdrant vector database

### API: Index content

- MediaWiki: `POST /mediawiki/url`
  - Body: `{ "url": "https://en.wikipedia.org/wiki/...", "max_chunks": 0, "force_delete": true }`
  - Notes: `max_chunks > 0` limits chunks to that number; `0` or omitted means no user limit. A hard cap (`MAX_CHUNKS_PER_DOC`) is always enforced.
  - Optional: `?estimate=true` query param to return planned chunk count without indexing.

- Generic URLs/PDFs: `POST /index`
  - Body: `{ "urls": ["https://..."], "doc_type": "HTML" | "PDF", "max_chunks": 0, "force_delete": true, "force_crawl": true }`
  - Behavior: standardize on chunk caps; character-based limits are removed.
  - Optional: `?estimate=true` query param to return planned chunk count without indexing.

- Structured PDF (keep sections/headings like MediaWiki):
  - Single endpoint: `POST /pdf` as multipart form with fields:
    - `file` (UploadFile, optional) or `url` (string, optional)
    - `max_chunks` (int, default 0), `force_delete` (bool, default true)
    - Optional query: `?estimate=true` to return planned chunk count only

Examples:
```bash
curl -X POST http://localhost:8000/mediawiki/url \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://en.wikipedia.org/wiki/OpenAI","max_chunks":50,"force_delete":true}'

curl -X POST http://localhost:8000/index \
  -H 'Content-Type: application/json' \
  -d '{"urls":["https://openai.com"],"doc_type":"HTML","max_chunks":100,"force_delete":true}'

# Estimate only examples
curl -X POST 'http://localhost:8000/mediawiki/url?estimate=true' \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://en.wikipedia.org/wiki/OpenAI","max_chunks":0}'

curl -X POST 'http://localhost:8000/index?estimate=true' \
  -H 'Content-Type: application/json' \
  -d '{"urls":["https://openai.com"],"doc_type":"HTML","max_chunks":0}'

# Structured PDF examples
# Upload a local PDF
curl -X POST 'http://localhost:8000/pdf?estimate=false' \
  -F 'file=@/path/to/file.pdf' \
  -F 'max_chunks=100' \
  -F 'force_delete=true'

# Use a PDF URL, estimate only
curl -X POST 'http://localhost:8000/pdf?estimate=true' \
  -F 'url=https://example.com/file.pdf' \
  -F 'max_chunks=0'
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/vrraj/website_pdf_chat.git
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Configure environment variables:
```env
OPENAI_API_KEY=your_openai_key

# Note: Qdrant settings are configured in settings.py with defaults:
# qdrant_host=localhost
# qdrant_port=6333

# Additional configuration
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4.1-mini-2025-04-14
MAX_HISTORY_TOKENS=4000
COLLECTION_NAME=website_collection

# Embedding safety limits (prevent runaway loops)
MAX_CHUNKS_PER_DOC=500
EMBEDDINGS_MAX_RETRIES=5
EMBEDDINGS_INITIAL_BACKOFF_SECS=1.0
EMBEDDINGS_MAX_CONSECUTIVE_FAILURES_PER_DOC=20
EMBEDDINGS_TOTAL_TIME_LIMIT_SECS=300
EMBEDDINGS_CALL_DELAY_SECS=0.0
```

4. Run the development server:
```bash
python run.py
```

## Project Structure

```
backend/
├── core/          # Core configuration and settings
├── chat/          # Chat management and OpenAI integration
├── crawler/       # URL crawling functionality
├── extractor/     # Content extraction from HTML and PDFs
├── embeddings/    # Embedding generation and Qdrant integration
└── test/          # Test files and configurations

frontend/
├── app.js         # Main application logic
├── index.html     # HTML template
└── styles.css     # Styling
```
