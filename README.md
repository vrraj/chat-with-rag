# Website PDF Chat Agent

A powerful AI-powered system for interacting with website and PDF content using modern AI technologies.

## Features

- 📚 PDF and Website Content Processing
  - Extract text from both websites and PDF documents
  - Handle structured and unstructured content
  - Preserve document hierarchy and formatting

- 🧠 Semantic Search and Embeddings
  - Generate embeddings using OpenAI's text-embedding-3-small model
  - Store and search embeddings using Qdrant vector database
  - Smart chunking of text for optimal search performance

- 💬 AI-Powered Chat Interface
  - Chat with content using GPT-4.1-mini
  - Context-aware responses
  - Real-time web search integration

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
