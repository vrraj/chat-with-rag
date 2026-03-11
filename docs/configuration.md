# Configuration Reference

> **About this document**
>
> This page provides a **comprehensive configuration reference** for the *Chat-with-RAG* system, including environment variables, model settings, and runtime parameters.
>
> If you landed here directly (for example from documentation hosting or search), start with the repository **[README](../README.md)** to see how to run the system locally and try the interactive demo.

## Table of Contents

- [Environment Variables](#environment-variables)
- [Backend Configuration](#backend-configuration)
- [Frontend Configuration](#frontend-configuration)
- [Model Registry](#model-registry)
- [Domain Configuration](#domain-configuration)
- [Embedding Configuration](#embedding-configuration)
- [Chat Pipeline Configuration](#chat-pipeline-configuration)

---

## Environment Variables

Create a `.env` file in the project root with these variables:

### Core Settings

```bash
# API Keys (required)
OPENAI_API_KEY=sk-your-openai-key-here
GEMINI_API_KEY=your-gemini-key-here

# Server Configuration
HOST=0.0.0.0
PORT=8000
ALLOWED_ORIGINS=http://localhost:8000,http://127.0.0.1:8000

# Debug Settings
DEBUG_VERBOSE=false
DEBUG_LOG_KEYS=false
DEBUG_LOG_TRUNCATE_CHARS=200
SHOW_PROCESSING_STEPS=true
```

### Database Settings

```bash
# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=  # Leave empty for local Qdrant

# Collection Settings
DEFAULT_COLLECTION=document_index
```

### LLM Provider Settings

```bash
# OpenAI Configuration
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_ORG_ID=  # Optional

# Gemini Configuration  
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1
```

### Prompt Registry

```bash
# Prompt Debugging
PROMPT_REGISTRY_LOG_FULL=0  # Set to 1 to log full resolved prompts
```

---

## Backend Configuration

### Main Settings Class

Located in `backend/core/config.py`:

```python
class Settings(BaseSettings):
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    allowed_origins: List[str] = ["http://localhost:8000"]
    
    # Database
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None
    
    # Default Models
    embedding_model: str = "openai:embed_small"
    inference_model: str = "openai:gpt-4o"
    rewrite_model: str = "openai:gpt-4o-mini"
    summary_model: str = "openai:gpt-4o-mini"
    rerank_model: str = "openai:gpt-4o-mini"
    tools_synth_model: str = "openai:gpt-4o-mini"
    
    # Processing Settings
    top_k: int = 8
    score_threshold: float = 0.35
    max_inference_output_tokens: int = 500
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Chat Settings
    raw_tail_turns: int = 10
    summarizer_max_input_tokens: int = 4000
    summarizer_max_output_tokens: int = 128
    summarizer_temperature: float = 0.3
    
    # Query Rewrite
    enable_query_rewrite: bool = True
    rewrite_confidence_threshold: float = 0.6
    rewrite_tail_turns: int = 1
    rewrite_summary_turns: int = 3
    rewrite_cache_ttl_s: int = 300
    
    # Tools
    use_tools: bool = True
    use_web_search: bool = False
    max_tool_passes: int = 2
    
    # Embedding Settings
    embedding_batch_size: int = 100
    default_chunk_size: int = 800
    default_chunk_overlap: int = 100
    max_chunks_per_doc: int = 0
    
    # Reasoning
    inference_reasoning_effort: str = "low"
    inference_reasoning_model: bool = False
    debug_thoughts: bool = True
    
    # Debug
    debug_verbose: bool = False
    debug_log_keys: bool = False
    debug_log_truncate_chars: int = 200
    show_processing_steps: bool = True
```

### Content Processing Configuration

```python
class MediaWikiConfig(BaseModel):
    api_url: str = "https://en.wikipedia.org/w/api.php"
    user_agent: str = "WebsiteChatAgent/0.1 (contact@example.com)"
    max_chunks: int = 0  # 0 = no limit
    skip_sections: List[str] = [
        "References", "External links", "See also", "Further reading"
    ]
    estimate: bool = True
    force_delete: bool = False

class HTMLConfig(BaseModel):
    max_chunks: int = 0
    skip_sections: List[str] = [
        "References", "External links", "See also", "Further reading"
    ]
    estimate: bool = True
    force_delete: bool = False

class PDFConfig(BaseModel):
    max_chunks: int = 0
    skip_sections: List[str] = [
        "References", "External links", "Further reading", 
        "Notes", "See Also", "Acknowledgements"
    ]
    estimate: bool = True
    force_delete: bool = False
```

---

## Model Registry

The model registry defines all available LLM providers and models for the chat-with-rag system. For complete model details, pricing, and capabilities, see the **[full Model Registry documentation](https://vrraj.github.io/llm-adapter/model-registry.html)**.

### Available Models

#### OpenAI Models
- **`openai:embed_small`** - text-embedding-3-small (1536 dimensions)
- **`openai:embed_large`** - text-embedding-3-large (3072 dimensions)
- **`openai:gpt-4o-mini`** - Fast inference model
- **`openai:gpt-4o`** - Standard inference model
- **`openai:chat_gpt-4o-mini`** - Chat completions endpoint
- **`openai:chat_gpt-4o`** - Chat completions endpoint
- **`openai:reasoning_o3-mini`** - Reasoning model (o3-mini)
- **`openai:reasoning_gpt-5-mini`** - Reasoning model (gpt-5-mini)

#### Gemini Models
- **`gemini:native-embed`** - gemini-embedding-001 (1536 dimensions)
- **`gemini:openai-2.5-flash-lite`** - Fast inference via OpenAI adapter
- **`gemini:openai-3-flash-preview`** - Latest preview model
- **`gemini:native-sdk-3-flash-preview`** - Native SDK endpoint
- **`gemini:openai-reasoning-2.5-flash`** - Reasoning via OpenAI adapter
- **`gemini:native-sdk-reasoning-2.5-flash`** - Reasoning via native SDK

### Model Categories

| Category | Models | Use Case |
|----------|--------|----------|
| **Embeddings** | `openai:embed_*`, `gemini:native-embed` | Vector search and retrieval |
| **Fast Inference** | `openai:gpt-4o-mini`, `gemini:*-flash*` | Chat responses, query rewriting |
| **Standard Inference** | `openai:gpt-4o` | Complex tasks, summarization |
| **Reasoning** | `openai:reasoning_*`, `gemini:*-reasoning*` | Complex problem solving |

### Default Configuration

The system uses these default models:
- **Embedding**: `openai:embed_small` (mountains domain) / `gemini:native-embed` (oceans domain)
- **Inference**: `openai:gpt-4o`
- **Query Rewrite**: `openai:gpt-4o`
- **Summarization**: `openai:gpt-4o`

> **For detailed model specifications, pricing, and advanced configuration options, see the complete [Model Registry documentation](https://vrraj.github.io/llm-adapter/model-registry.html).**

---

## Domain Configuration

Domain-based configuration allows multiple isolated knowledge bases:

```python
DOMAIN_EMBEDDING_CONFIG = {
    "default": {
        "collection_name": "document_index",
        "embedding_model_key": "openai:embed_small"
    },
    "mountains": {
        "collection_name": "document_index",
        "embedding_model_key": "openai:embed_small" 
    },
    "oceans": {
        "collection_name": "document_index_gemini",
        "embedding_model_key": "gemini:native-embed"
    }
}

# Active domain (change this to switch domains)
active_domain: str = "mountains"
```

### Using Different Domains

```python
# In backend/core/config.py, change:
active_domain = "oceans"  # Switch to oceans domain

# Or override via environment variable
# ACTIVE_DOMAIN=oceans python start.py
```

---

## Embedding Configuration

### Chunking Parameters

```python
# Text chunking settings
default_chunk_size: int = 800          # Characters per chunk
default_chunk_overlap: int = 100       # Overlap between chunks
embedding_batch_size: int = 100  # Chunks per embedding API call
max_chunks_per_doc: int = 0        # 0 = no limit
```

### Provider-Specific Limits

| Provider | Max Inputs | Max Tokens per Input | Batch API |
|----------|------------|---------------------|-----------|
| OpenAI   | 2,048      | 8,191               | Yes       |
| Gemini   | 250        | 2,048               | No        |

### Recommended Settings

**OpenAI text-embedding-3-small:**
```python
chunk_size = 800
embedding_batch_size = 100
```

**Gemini gemini-embedding-001:**
```python  
chunk_size = 600
embedding_batch_size = 50
```

---

## Chat Pipeline Configuration

### Retrieval Settings

```python
# Vector search parameters
top_k: int = 8                    # Number of documents to retrieve
score_threshold: float = 0.35     # Minimum similarity score
namespace: str = "default"        # Collection/domain isolation
```

### Inference Settings

```python
# LLM generation parameters
temperature: float = 0.7          # Randomness (0.0-1.0)
top_p: float = 0.9                # Nucleus sampling
max_inference_output_tokens: int = 500     # Response length limit
reasoning_effort: str = "low"    # For reasoning models
inference_reasoning_model: bool = False  # Enable reasoning model
```

### Context Management

```python
# Conversation memory
raw_tail_turns: int = 10                    # Verbatim recent turns
summarizer_max_input_tokens: int = 4000      # Summary input limit  
summarizer_max_output_tokens: int = 128     # Summary output limit
summarizer_temperature: float = 0.3         # Summarization randomness
```

### Query Rewrite Configuration

```python
enable_query_rewrite: bool = True
rewrite_confidence_threshold: float = 0.6    # Minimum confidence to accept rewrite
rewrite_tail_turns: int = 1                   # Recent turns for context
rewrite_summary_turns: int = 3                # How many summary turns to consider
rewrite_cache_ttl_s: int = 300                # Cache duration in seconds
```

### Tool Configuration

```python
use_tools: bool = True
use_web_search: bool = False
max_tool_passes: int = 2                     # Maximum tool loops per turn

# Available tools
# - get_weather: Weather information
# - get_airports: Airport lookup  
# - web_search: DuckDuckGo search (if enabled)
```

### Processing Visibility

```python
show_processing_steps: bool = True  # Show intermediate pipeline stages
show_sources: bool = True           # Show source citations
```

---

## Runtime Parameter Override

All configuration can be overridden at runtime via the `params` object in API calls:

### Example Override

```python
params = {
    "top_k": 12,                    # Override default top_k
    "temperature": 0.3,             # Override default temperature
    "model_keys": {                 # New format for model overrides
        "inference": "openai:gpt-4o-mini"
    },
    "enable_query_rewrite": False,  # Disable query rewrite
    "show_processing_steps": False  # Hide processing steps
}
```

### Per-Stage Model Override

```python
params = {
    "model_keys": {
        "inference": "openai:gpt-4o",              # Main inference
        "rewrite": "openai:gpt-4o-mini",           # Query rewrite  
        "summary": "openai:gpt-4o-mini",           # Summarization
        "rerank": "openai:gpt-4o-mini",            # Reranking
        "tools_synth": "gemini:openai-2.5-flash-lite"  # Tool synthesis
    }
}
```

### Reasoning Model Override

```python
params = {
    "model_keys": {
        "inference": "openai:reasoning_o3-mini",    # OpenAI reasoning model
        "reasoning_effort": "medium"                 # Reasoning intensity
    }
}
```

### Gemini Reasoning Model Override

```python
params = {
    "model_keys": {
        "inference": "gemini:openai-3-flash-preview",  # Gemini reasoning model
        "thinking_level": "low"                      # Gemini reasoning parameter
    }
}
```

---

## Configuration Validation

### Validate Configuration

```python
from backend.core.config import settings

# Check settings
print(f"Embedding model: {settings.embedding_model}")
print(f"Collection: {settings.collection_name}")
print(f"Top K: {settings.top_k}")
```

### Test Connectivity

```bash
# Test API connections
python scripts/api_smoke_test_openai.py
python scripts/api_smoke_test_gemini.py

# Test embedding generation
python scripts/embedding_compare.py

# Test Qdrant connection
python scripts/qdrant_scripts/qdrant_ops.py --list-collections
```

---

## Best Practices

### Prompt Registry

#### Registry file

- **Path:** `prompts/prompt_registry.yaml`
- **Role:** Source of truth for stage prompt text and templates.
- **Current coverage:** Inference and query rewrite are registry-driven; rerank and summarization use the registry for their fixed instructions/templates.

#### Prompt domains (`params.prompt_domain`)

You can select a prompt domain per request using `params.prompt_domain`.

- If `prompt_domain` is empty or omitted, the system uses `global_defaults`.
- If `prompt_domain` is set (example: `mountains`), the system applies domain-specific overrides (currently by appending additional domain system instructions).

In the UI (`frontend/chat.html`), the **Prompt Domain** dropdown under **Inference** controls the value sent on every chat request.

#### Debug logging (safe by default)

The backend logs:

- Which domain was resolved for inference.
- A short tail snippet of the resolved system instruction.

To log the full resolved prompt/template for debugging, set:

```bash
PROMPT_REGISTRY_LOG_FULL=1
```

### Performance Optimization

1. **Use appropriate model tiers:**
   - Fast models for rewrite/rerank/summary
   - Capable models for main inference

2. **Configure batch sizes:**
   - Larger batches for embedding (within provider limits)
   - Smaller chunks for better relevance

3. **Set appropriate limits:**
   - `top_k`: 5-15 for most use cases
   - `max_output_tokens`: Based on expected response length

### Cost Management

1. **Enable estimate mode** for large indexing operations
2. **Use faster models** for non-critical stages
3. **Monitor usage** with conversation totals
4. **Set appropriate token limits**

### Security

1. **Never commit API keys** to version control
2. **Use environment variables** for sensitive configuration
3. **Restrict allowed origins** in production
4. **Monitor API usage** and costs

---

## Troubleshooting Configuration

### Common Issues

1. **Dimension mismatch:** Ensure embedding model matches collection
2. **API key errors:** Verify keys in `.env` file
3. **Connection refused:** Check Qdrant is running
4. **CORS errors:** Verify allowed origins configuration

### Debug Configuration

```bash
# Enable verbose logging
DEBUG_VERBOSE=true
DEBUG_LOG_KEYS=true

# Log full prompts (for debugging)
PROMPT_REGISTRY_LOG_FULL=1

# Check current configuration
python -c "from backend.core.config import settings; print(settings.dict())"
```

### Reset Configuration

#### Reset Environment Configuration

```bash
# Reset environment variables to defaults
cp .env.example .env
# Edit .env with your API keys and restart application
```

#### Reset Qdrant Database

```bash
# Clear data but keep collection structure
python scripts/qdrant_scripts/qdrant_ops.py truncate --collection document_index
python scripts/qdrant_scripts/qdrant_ops.py truncate --collection document_index_gemini

# Delete entire collection and re-seed
python scripts/qdrant_scripts/qdrant_ops.py delete --collection document_index
python scripts/qdrant_scripts/qdrant_ops.py delete --collection document_index_gemini
make seed
```

#### Full System Reset

```bash
# Complete reset to factory defaults
cp .env.example .env
make stop
python scripts/qdrant_scripts/qdrant_ops.py delete --collection document_index
python scripts/qdrant_scripts/qdrant_ops.py delete --collection document_index_gemini
make seed
make start
```
