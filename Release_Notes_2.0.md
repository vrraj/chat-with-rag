# Release Notes: v1.0.1 → v2.0.0

## 🎯 Executive Summary

This release transforms the application from a single-provider system into a **multi-provider, enterprise-ready platform** with a revolutionary model registry architecture. The introduction of centralized model management eliminates configuration sprawl while enabling seamless provider mixing, advanced embedding strategies, and production-ready security features.

**Key Achievements:**
- 🏗️ **Model Registry Architecture** - Single source of truth for all model configurations
- 🔄 **Multi-Provider Support** - OpenAI and Gemini with unified abstraction
- 🧩 **Prompt Registry (YAML)** - Centralized prompt definitions in `prompts/prompt_registry.yaml` with domain-based overrides selected via `params.prompt_domain`
- ⚡ **Performance Optimization** - Batch processing and rate limit management
- 🔒 **Security Enhancements** - Embeddable chat with domain controls + API security
- 📊 **Cost Management** - Provider-specific pricing and budget tracking
- 🌐 **Embeddable Chat Widget** - Configurable chat for any website
- 🧪 **Postprocessing (Markdown → HTML)** - Backend rendering with sanitized HTML, sources formatting, and scoped UI styling

---

## 🚀 Major Features

### **🏗️ Central Model Registry Architecture**
- **NEW `model_registry.py`** - Single source of truth for all model configurations
- **Complete abstraction** of model nuances, capabilities, and costs
- **Centralized model management** - eliminated scattered model configurations
- **Massive code reduction** through registry-based approach
- **Removed ALL model configurations** from `config.py` - now uses registry keys only

### **🌐 Embeddable Chat Widget**
- **Production-ready embeddable chat** for any website
- **Configurable via data attributes**:
  - `data-model_key` - Preselected model from registry
  - `data-temperature` - Response creativity control
  - `data-top_k` - Retrieval configuration
  - `data-show_processing_steps` - Enable streaming visualization
  - `data-show_citations` - Source citation display
  - `data-namespace` - Custom knowledge base isolation
- **Easy integration** - Single script tag deployment
- **Responsive iframe design** with isolated styling
- **Real-time streaming** support for processing visualization

### **🔒 Comprehensive Security Framework**
- **Domain-based access controls** for all FastAPI routes
- **Origin and host allowlist** enforcement:
  - `allowed_origins`: Full URL validation (e.g., "https://mysite.com")
  - `allowed_hosts`: Hostname validation (e.g., "mysite.com:8000")
- **Automatic security enforcement** on all critical endpoints:
  - `/chat`, `/index`, `/embed`, `/upload`, `/batch`
  - Ingestion, chat, and embedding endpoints
- **CORS protection** with configurable domain whitelisting
- **Best-effort security** - graceful fallback when not configured

### **Multiple LLM Provider Support**
- **Unified LLM handler** leveraging the model registry for provider abstraction
- **Seamless provider switching** through registry key configuration
- **Capability-based routing** - handler automatically selects correct endpoint
- **Provider-specific optimizations** handled transparently

### **Advanced Embedding Support**
- **Gemini embeddings integration** with L2 normalization
- **Batch processing capabilities** with configurable batch sizes
- **Magnitude metadata inclusion** for normalized embeddings
- **Registry-driven embedding model selection**

---

## 🔧 Technical Architecture

### **Configuration Simplification**
- **Clean separation**: Registry handles models, config handles application settings
- **Registry key references** in config instead of model details:
  - `embedding_model_key: "gemini:embed"`
  - `inference_model_key: "openai:fast"`
  - `rerank_model_key: "openai:fast"`
- **Dramatically simplified config** - removed complex model parameter objects

### **Mixed Provider Pipeline Architecture**
- **Stage-specific provider selection** using registry keys:
  - Reranking: OpenAI for cost-effective processing
  - Inference: Gemini for advanced reasoning
  - Summarization: Configurable per use case
- **Flexible provider mixing** - optimize cost/performance per stage
- **Registry-driven model resolution**

### **Batch Processing & Rate Limit Management**
- **Configurable batch embedding sizes** to work around rate limits:
  - `embedding_batch_size_default`: 30 chunks per batch
  - `embedding_batch_size_openai`: 30 chunks per batch  
  - `embedding_batch_size_gemini`: 30 chunks per batch
- **Performance optimization** through controlled batch processing

### **Extensible LLM Handler Architecture**
- **Complete API abstraction** - no model APIs in the application code
- **Registry-backed model resolution** in `LLMHandler` class
- **Provider-agnostic interface** through registry abstraction
- **Extensible design** - new providers added via registry entries

### **🧪 Postprocessing: Markdown → HTML Rendering**
- **NEW backend rendering** (`backend/markdown_render.py`): Converts Markdown to sanitized HTML using `markdown-it-py`/`Markdown` with `bleach`. Wraps tables in scrollable containers and hardens links.
- **Sources formatting**: Detects and splits any `Sources:` block so it starts on a new line, with each source on its own line, and makes the heading bold (`<strong>Sources:</strong>`).
- **Feature flag**: Enabled when `params.render_html=true` (sent by the frontend). Returns `answer_html` in `/chat` response and `finalHtml` in SSE final stage.
- **Frontend rendering**: Conditional `innerHTML` vs `textContent` with scoped CSS for tight spacing and table styling. Graceful fallback to plain text.
- **Safety**: Server-side sanitization prevents malformed or malicious HTML from reaching the browser.
- **Additive only**: No breaking changes; works entirely behind the feature flag.

---

## 🔍 Detailed Technical Features

### **Model Registry Deep Dive**

**Registry Structure:**
```python
@dataclass(frozen=True)
class ModelInfo:
    key: str                    # Stable alias (e.g., "openai:fast")
    provider: Provider          # "openai" or "gemini"
    model: str                  # Provider-native model ID
    endpoint: Endpoint          # API shape to use
    pricing: Optional[Pricing]  # Cost information
    capabilities: Dict[str, Any]  # Feature support
    thinking_tax: Dict[str, Any]  # Reasoning model rules
```

**Available Models:**
- **OpenAI**: `embed_small`, `fast`, `best`, `chat_fast`, `chat_best`
- **Gemini**: `embed`, `native-embed`, `fast`, `fast-test`, `fast-3-flash`, `native-sdk-fast-3-flash`, `best`

**Capability Mapping:**
- Tools, streaming, reasoning, temperature support
- Parameter standardization across providers
- Pricing transparency for budget management

### **Embeddable Chat Configuration**

**Simple Integration:**
```html
<div id="support-chat"></div>
<script src="https://your-app.com/static/embed-loader.js"
        data-target="#support-chat"
        data-model_key="gemini:fast"
        data-temperature="0.4"
        data-show_processing_steps="true"
        data-show_citations="true"
        data-namespace="docs-help">
</script>
```

**Advanced Features:**
- **Model selection** via registry keys
- **Streaming visualization** of processing steps
- **Citation display** for source transparency
- **Namespace isolation** for multi-tenant deployments
- **Responsive design** for mobile compatibility

### **Security Implementation**

**Domain Controls:**
```python
# Configuration in config.py
allowed_origins: "https://mysite.com,https://partner.com"
allowed_hosts: "mysite.com:8000,partner.com:443"

# Automatic enforcement on all routes
def enforce_origin_host(request: Request) -> None:
    # Origin and host validation
    # 403 response for unauthorized domains
```

**Protected Endpoints:**
- All `/chat/*` routes
- All `/index/*` routes  
- All `/embed` routes
- All ingestion endpoints
- File upload and batch processing

### **Gemini Asymmetric Embedding Contracts**
- **Task-specific embedding optimization**:
  - **Ingestion**: `"RETRIEVAL_DOCUMENT"` - Optimizes for storage and 'find-ability'
  - **Retrieval**: `"RETRIEVAL_QUERY"` - Optimizes for 'seeker' intent
- **Configurable task types** in `config.py`:
  - `gemini_embed_type_documents`: "RETRIEVAL_DOCUMENT"
  - `gemini_embed_type_query`: "RETRIEVAL_QUERY"

---

## 📊 Architecture Benefits

### **Code Reduction & Maintainability**
- **Eliminated model configuration sprawl** across multiple files
- **Single source of truth** for model information
- **Prompt Registry (YAML)** - Centralized prompt definitions in `prompts/prompt_registry.yaml` with domain-based overrides via `params.prompt_domain`
- **Dramatically reduced boilerplate** code
- **Centralized capability management**

### **Provider Flexibility**
- **Zero-code provider switching** - just change registry keys
- **Capability-aware routing** - automatic endpoint selection
- **Cost transparency** through registry pricing
- **Unified parameter interface** across providers

### **Security & Compliance**
- **Production-ready access controls** out of the box
- **Domain-based whitelisting** for embed security
- **API endpoint protection** for all critical operations
- **Configurable security policies** per deployment

### **Extensibility**
- **New models added via registry entries** - no code changes needed
- **New providers supported** through handler implementation
- **Capability evolution** handled through registry updates
- **Backward compatibility** maintained through stable keys

---

## 📈 Performance & Scalability

### **Batch Processing**
- **Batch embedding support** for improved throughput
- **Optimized token usage** tracking and cost management
- **Enhanced Qdrant integration** with better metadata handling

### **Cost Management**
- **Registry-based pricing** models
- **Token usage tracking** per operation
- **Budget controls** and limits configuration

---

## 🧪 Testing & Documentation

### **Comprehensive Test Suite**
- Gemini embeddings testing (`test_gemini_embeddings.py`)
- OpenAI handler tests (`test_openai_handler.py`)
- Token counting and normalization tests
- API smoke tests for both providers
- Magnitude metadata validation

### **New Documentation**
- `README_LLM_HANDLER.md` with comprehensive handler documentation
- Enhanced API integration specs
- Registry usage examples
- Embed integration guide

---

## 🔄 Migration Guide

### **Configuration Changes**
- **Model configurations moved** from `config.py` to `model_registry.py`
- **Registry key references** replace direct model configurations
- **Simplified config structure** with cleaner separation of concerns
- **Security configuration** added for domain controls

### **Breaking Changes**
- Updated LLM handler call signatures
- Refactored configuration structure
- Removed OpenAI client dependencies from core chat
- **Security enforcement** now active on all endpoints

### **New Dependencies**
- Google Gemini SDK integration
- Enhanced tokenization support
- Additional testing frameworks

---

## 📊 Impact Metrics

- **68 files changed**
- **14,968 insertions, 876 deletions**
- **20+ new test files**
- **🏗️ 1 critical new component** (Model Registry)
- **2 major new components** (LLM handler, Model registry)
- **3 new frontend embed files**
- **🔒 Security hardening** across all API endpoints

---

## 🎯 Key Architectural Achievement

This release establishes a **production-ready, enterprise-grade platform** through:

✅ **Centralized model management** - Single source of truth  
✅ **Massive code reduction** - Eliminated configuration duplication  
✅ **Provider abstraction** - Clean separation from application logic  
✅ **Capability transparency** - Clear model capabilities and limits  
✅ **Cost visibility** - Built-in pricing information  
✅ **Extensibility** - New models/providers via configuration only  
✅ **🌐 Embeddable chat** - Deploy to any website with full configuration  
✅ **🔒 Security framework** - Domain-based access controls for all APIs  

The revolutionary model registry pattern combined with comprehensive security and embeddable chat capabilities transforms this into a scalable, multi-provider AI platform ready for enterprise deployment.

---

## 🚀 What's Next

### **Planned Enhancements**
- Additional provider support (Anthropic, Cohere)
- Advanced caching strategies
- Enhanced monitoring and analytics
- Multi-tenant isolation improvements
- Advanced rate limiting and quota management

### **Community Contributions**
- Extended test coverage
- Additional embedding models
- Performance optimizations
- Security enhancements
- Documentation improvements

---

*This release represents a significant architectural evolution from v1.0.1, establishing the foundation for scalable, multi-provider AI operations with enterprise-grade security and extensibility.*
