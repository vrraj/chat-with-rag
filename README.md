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
