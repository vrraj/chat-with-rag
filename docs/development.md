# Development Guide

This guide covers development workflows, testing, and contribution guidelines for the chat-with-rag project.

## Table of Contents

- [Development Setup](#development-setup)
- [Code Organization](#code-organization)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Debugging](#debugging)
- [Adding Features](#adding-features)
- [Contribution Guidelines](#contribution-guidelines)

---

## Development Setup

### Prerequisites

- **Python 3.10+**
- **Docker & Docker Compose**
- **Git**
- **Code Editor** (VS Code recommended)

### Local Development Environment

#### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/vrraj/chat-with-rag.git
cd chat-with-rag

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If exists

# Setup environment
cp .env.example .env
# Edit .env with your API keys
```

#### 2. Install Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

#### 3. Development Docker Setup

Create `docker-compose.dev.yml`:

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEBUG_VERBOSE=true
      - SHOW_PROCESSING_STEPS=true
    env_file:
      - .env
    volumes:
      - .:/app  # Mount source code
      - /app/venv  # Exclude venv
    depends_on:
      - qdrant
    command: python run.py  # Use run.py for hot reload

  qdrant:
    image: qdrant/qdrant:v1.14.1
    ports:
      - "6333:6333"
    volumes:
      - qdrant_dev_data:/qdrant/storage

volumes:
  qdrant_dev_data:
```

#### 4. Start Development Environment

```bash
# Start services with hot reload
docker-compose -f docker-compose.dev.yml up

# Or run locally
make start
python run.py  # Hot reload enabled
```

### IDE Setup

#### VS Code Configuration

Create `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "100"],
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    ".pytest_cache": true,
    ".coverage": true
  }
}
```

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: FastAPI",
      "type": "python",
      "request": "launch",
      "program": "start.py",
      "console": "integratedTerminal",
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      }
    },
    {
      "name": "Python: Tests",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": ["tests/", "-v"],
      "console": "integratedTerminal"
    }
  ]
}
```

---

## Code Organization

### Project Structure

```
chat-with-rag/
├── backend/
│   ├── api/                 # API endpoints
│   ├── chat/               # Chat orchestration
│   ├── core/               # Core configuration and schemas
│   ├── crawler/            # Web crawling
│   ├── embeddings/         # Embedding management
│   ├── llm/                # LLM client abstraction
│   ├── scripts/            # Utility scripts
│   └── main.py             # FastAPI application
├── frontend/
│   ├── static/             # JavaScript, CSS
│   └── *.html              # HTML pages
├── docs/                   # Documentation
├── tests/                  # Test suite
├── scripts/                # Shell scripts
├── prompts/                # Prompt registry
└── docker-compose.yml      # Docker configuration
```

### Key Components

#### Backend Architecture

```python
# Core components
backend/
├── core/
│   ├── config.py          # Configuration management
│   ├── schemas.py         # Pydantic models
│   └── file_utils.py      # File utilities
├── chat/
│   ├── chat_manager.py    # Main chat orchestration
│   └── chunked_history_manager.py  # Context management
├── embeddings/
│   └── embeddings_manager.py  # Embedding operations
├── llm/
│   └── llm_client.py      # LLM abstraction layer
└── main.py                # FastAPI routes
```

#### Frontend Architecture

```javascript
// Frontend structure
frontend/
├── static/
│   ├── chat.js           # Main chat functionality
│   ├── chat-embed.js     # Embeddable chat
│   ├── embed-loader.js   # Embed loader
│   └── styles.css        # Styles
├── chat.html             # Main chat interface
├── chat-embed.html       # Embeddable chat
└── index.html            # Landing page
```

---

## Development Workflow

### 1. Feature Development

#### Create Feature Branch

```bash
# Create and switch to feature branch
git checkout -b feature/new-feature-name

# Or for bug fixes
git checkout -b fix/bug-description
```

#### Development Process

```bash
# 1. Make changes
# Edit files...

# 2. Run tests
pytest tests/

# 3. Lint and format
black backend/ frontend/
pylint backend/

# 4. Type checking
mypy backend/

# 5. Test manually
python start.py
# Test in browser or via API

# 6. Commit changes
git add .
git commit -m "feat: add new feature description"
```

#### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new feature
fix: resolve bug in chat pipeline  
docs: update API documentation
style: format code with black
refactor: improve chat manager structure
test: add tests for embedding manager
chore: update dependencies
```

### 2. Testing Strategy

#### Unit Tests

```python
# tests/unit/test_embeddings_manager.py
import pytest
from backend.embeddings.embeddings_manager import EmbeddingsManager

class TestEmbeddingsManager:
    def setup_method(self):
        self.manager = EmbeddingsManager()
    
    def test_embed_text(self):
        """Test text embedding"""
        text = "Test document"
        embedding = self.manager.embed_text(text)
        
        assert len(embedding) == 1536  # OpenAI dimensions
        assert all(isinstance(x, float) for x in embedding)
    
    def test_batch_embed(self):
        """Test batch embedding"""
        texts = ["Doc 1", "Doc 2", "Doc 3"]
        embeddings = self.manager.embed_batch(texts)
        
        assert len(embeddings) == 3
        assert all(len(e) == 1536 for e in embeddings)
```

#### Integration Tests

```python
# tests/integration/test_chat_pipeline.py
import pytest
from backend.chat.chat_manager import ChatManager

class TestChatPipeline:
    def setup_method(self):
        self.chat_manager = ChatManager()
    
    @pytest.mark.asyncio
    async def test_chat_request(self):
        """Test end-to-end chat request"""
        request = {
            "message": "What is the capital of France?",
            "history": [],
            "params": {
                "top_k": 5,
                "temperature": 0.7
            }
        }
        
        response = self.chat_manager.handle_chat(request)
        
        assert "answer" in response
        assert "metrics" in response
        assert len(response["answer"]) > 0
```

#### API Tests

```python
# tests/integration/test_api.py
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_check():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_chat_endpoint():
    """Test chat API"""
    payload = {
        "message": "Test message",
        "history": [],
        "params": {"top_k": 5}
    }
    
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    assert "answer" in response.json()
```

### 3. Running Tests

#### Local Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=backend --cov-report=html

# Run specific test file
pytest tests/unit/test_embeddings_manager.py

# Run with verbose output
pytest -v

# Run integration tests only
pytest tests/integration/
```

#### Test Configuration

Create `pytest.ini`:

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow running tests
```

---

## Debugging

### 1. Debug Configuration

#### VS Code Debugging

```json
// .vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug FastAPI",
      "type": "python",
      "request": "launch",
      "program": "start.py",
      "console": "integratedTerminal",
      "env": {
        "DEBUG_VERBOSE": "true",
        "PYTHONPATH": "${workspaceFolder}"
      },
      "args": [],
      "justMyCode": false
    }
  ]
}
```

#### Python Debugging

```python
# Add debug prints
import logging
logging.basicConfig(level=logging.DEBUG)

# Use pdb for debugging
import pdb; pdb.set_trace()

# Or use ipdb (better interface)
import ipdb; ipdb.set_trace()
```

### 2. Logging Configuration

#### Development Logging

```python
# backend/core/logging.py
import logging
import sys

def setup_logging():
    """Setup development logging"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('debug.log')
        ]
    )

# Use in modules
logger = logging.getLogger(__name__)
logger.info("Processing chat request")
logger.debug(f"Request data: {request_data}")
```

#### Structured Logging

```python
import structlog

logger = structlog.get_logger()

def process_chat(request):
    logger.info("chat_started", 
                user_id=request.user_id,
                message_length=len(request.message))
    
    try:
        result = chat_pipeline(request)
        logger.info("chat_completed",
                    user_id=request.user_id,
                    response_length=len(result.answer))
        return result
    except Exception as e:
        logger.error("chat_failed",
                    user_id=request.user_id,
                    error=str(e),
                    exc_info=True)
        raise
```

### 3. Common Debugging Scenarios

#### API Issues

```python
# Debug API requests
import requests
import logging

# Enable request logging
logging.basicConfig(level=logging.DEBUG)
requests_log = logging.getLogger("requests.packages.urllib3")
requests_log.setLevel(logging.DEBUG)
requests_log.propagate = True

# Test API call
response = requests.post(url, json=payload)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")
```

#### Database Issues

```python
# Debug Qdrant operations
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter

def debug_qdrant():
    client = QdrantClient(host="localhost", port=6333)
    
    # Check collections
    collections = client.get_collections()
    print(f"Collections: {collections}")
    
    # Check collection info
    collection_info = client.get_collection_info("document_index")
    print(f"Collection info: {collection_info}")
    
    # Test search
    try:
        results = client.search(
            collection_name="document_index",
            query_vector=[0.1] * 1536,
            limit=5
        )
        print(f"Search results: {len(results)}")
    except Exception as e:
        print(f"Search error: {e}")
```

#### Embedding Issues

```python
# Debug embedding generation
from backend.llm.llm_client import LLMClient

def debug_embeddings():
    client = LLMClient()
    
    # Test single embedding
    text = "Test document for debugging"
    try:
        embedding = client.embed(text)
        print(f"Embedding shape: {len(embedding)}")
        print(f"Sample values: {embedding[:5]}")
    except Exception as e:
        print(f"Embedding error: {e}")
    
    # Test batch embedding
    texts = ["Doc 1", "Doc 2", "Doc 3"]
    try:
        embeddings = client.embed_batch(texts)
        print(f"Batch embeddings: {len(embeddings)} x {len(embeddings[0])}")
    except Exception as e:
        print(f"Batch embedding error: {e}")
```

---

## Adding Features

### 1. Adding New API Endpoints

#### Step 1: Define Pydantic Models

```python
# backend/core/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class NewFeatureRequest(BaseModel):
    input_data: str
    options: Optional[dict] = None

class NewFeatureResponse(BaseModel):
    result: str
    metadata: Optional[dict] = None
```

#### Step 2: Implement Business Logic

```python
# backend/api/endpoints/new_feature.py
from fastapi import APIRouter, Depends
from backend.core.schemas import NewFeatureRequest, NewFeatureResponse

router = APIRouter()

@router.post("/new-feature", response_model=NewFeatureResponse)
async def handle_new_feature(request: NewFeatureRequest):
    """Handle new feature request"""
    # Implement logic here
    result = process_new_feature(request.input_data, request.options)
    
    return NewFeatureResponse(
        result=result["output"],
        metadata=result.get("metadata")
    )
```

#### Step 3: Register Route

```python
# backend/main.py
from backend.api.endpoints.new_feature import router as new_feature_router

app.include_router(new_feature_router, tags=["New Feature"])
```

#### Step 4: Add Tests

```python
# tests/unit/test_new_feature.py
import pytest
from backend.api.endpoints.new_feature import handle_new_feature
from backend.core.schemas import NewFeatureRequest

def test_new_feature():
    request = NewFeatureRequest(
        input_data="test input",
        options={"option1": "value1"}
    )
    
    response = await handle_new_feature(request)
    
    assert response.result is not None
    assert isinstance(response.metadata, dict)
```

### 2. Adding New LLM Providers

#### Step 1: Create Provider Class

```python
# backend/llm/providers/new_provider.py
from backend.llm.llm_client import LLMProvider

class NewLLMProvider(LLMProvider):
    def __init__(self, api_key: str, base_url: str = None):
        self.api_key = api_key
        self.base_url = base_url or "https://api.newprovider.com/v1"
    
    async def chat_completion(self, messages, **kwargs):
        """Implement chat completion"""
        # Provider-specific implementation
        pass
    
    async def embed(self, texts, **kwargs):
        """Implement embedding"""
        # Provider-specific implementation
        pass
```

#### Step 2: Register Provider

```python
# backend/llm/llm_client.py
from backend.llm.providers.new_provider import NewLLMProvider

class LLMClient:
    def __init__(self):
        self.providers = {
            "openai": OpenAIProvider,
            "gemini": GeminiProvider,
            "new_provider": NewLLMProvider  # Add new provider
        }
```

#### Step 3: Update Model Registry

```python
# Add to model registry configuration
"new_provider:model-name": {
    "provider": "new_provider",
    "model": "model-name",
    "api_type": "chat_completions",
    "max_tokens": 4096,
    "input_cost_per_1k": 0.001,
    "output_cost_per_1k": 0.002,
    "supports_tools": True,
    "supports_reasoning": False
}
```

### 3. Adding New Tools

#### Step 1: Define Tool Function

```python
# backend/chat/tools/new_tool.py
from typing import Dict, Any

async def new_tool_function(param1: str, param2: int) -> Dict[str, Any]:
    """New tool implementation"""
    # Tool logic here
    result = {
        "output": f"Processed {param1} with {param2}",
        "success": True
    }
    return result
```

#### Step 2: Register Tool

```python
# backend/chat/tool_manager.py
from backend.chat.tools.new_tool import new_tool_function

class ToolManager:
    def __init__(self):
        self.tools = {
            "get_weather": get_weather,
            "get_airports": get_airports,
            "new_tool": new_tool_function  # Add new tool
        }
```

#### Step 3: Update Tool Schema

```python
# Update tool schema for LLM
NEW_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "new_tool",
        "description": "Description of what the tool does",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "First parameter"},
                "param2": {"type": "integer", "description": "Second parameter"}
            },
            "required": ["param1", "param2"]
        }
    }
}
```

---

## Contribution Guidelines

### 1. Code Style

#### Python Style Guide

Follow PEP 8 and use these tools:

```bash
# Format code
black backend/ frontend/

# Lint code
pylint backend/
flake8 backend/

# Type checking
mypy backend/

# Import sorting
isort backend/
```

#### JavaScript Style Guide

```bash
# Format JavaScript
npx prettier --write frontend/static/*.js

# Lint JavaScript
npx eslint frontend/static/*.js
```

### 2. Documentation Standards

#### Code Documentation

```python
def process_chat_request(request: ChatRequest) -> ChatResponse:
    """
    Process a chat request and generate response.
    
    Args:
        request: Chat request containing message and parameters
        
    Returns:
        Chat response with answer and metadata
        
    Raises:
        ValueError: If request is invalid
        LLMError: If LLM processing fails
        
    Example:
        >>> request = ChatRequest(message="Hello")
        >>> response = process_chat_request(request)
        >>> print(response.answer)
    """
    pass
```

#### API Documentation

```python
@app.post("/chat", 
          tags=["Chat"],
          summary="Process chat message",
          response_model=ChatResponse,
          responses={404: {"model": ErrorResponse}})
async def chat_endpoint(request: ChatRequest):
    """
    Process a chat message using the RAG pipeline.
    
    - **message**: User message to process
    - **params**: Optional parameters for customization
    """
    pass
```

### 3. Testing Requirements

#### Coverage Requirements

- Unit tests: >80% coverage
- Integration tests: Core workflows covered
- API tests: All endpoints tested

#### Test Categories

```python
# Mark tests appropriately
@pytest.mark.unit  # Unit tests
@pytest.mark.integration  # Integration tests
@pytest.mark.slow  # Slow tests (run separately)
```

### 4. Pull Request Process

#### PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass with appropriate coverage
- [ ] Documentation updated
- [ ] Commit messages follow convention
- [ ] No breaking changes (or clearly documented)

#### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
```

### 5. Release Process

#### Version Management

Use semantic versioning (semver):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

#### Release Steps

```bash
# 1. Update version
# Update version in pyproject.toml and __init__.py

# 2. Update changelog
# Add changes to CHANGELOG.md

# 3. Create release tag
git tag -a v1.2.3 -m "Release version 1.2.3"
git push origin v1.2.3

# 4. Create GitHub release
# Use tag and changelog
```

---

## Development Tips

### 1. Productivity Tips

- Use hot reload during development (`python run.py`)
- Run tests frequently to catch issues early
- Use Git branches for feature isolation
- Keep commits small and focused
- Use IDE integrations for better productivity

### 2. Debugging Tips

- Use structured logging for better debugging
- Add breakpoints for complex logic
- Test with small datasets first
- Use estimate mode before large operations
- Monitor API usage during development

### 3. Performance Tips

- Profile code before optimizing
- Use caching for expensive operations
- Batch API calls where possible
- Monitor memory usage
- Test with realistic data sizes

### 4. Common Pitfalls

- Don't commit API keys or secrets
- Don't ignore test failures
- Don't merge without review
- Don't break backward compatibility without notice
- Don't forget to update documentation

This development guide provides comprehensive coverage for contributing to the chat-with-rag project effectively.
