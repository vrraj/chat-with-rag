# Release Notes: v2.0.0

## 🎯 Executive Summary

This release evolves the Chat-with-RAG application from a single‑provider prototype into a **multi‑provider Tool-Assisted RAG Platform** with a centralized model registry architecture. Model configuration, provider abstraction, prompt management, and security controls are now standardized across the system, enabling flexible provider mixing, improved cost visibility, and cleaner separation between application logic and model integrations.

**Key Achievements:**
- 🏗️ **Model Registry Architecture** - Single source of truth for all model configurations
- 🔄 **Multi-Provider Support** - OpenAI and Gemini with unified abstraction
- 🧩 **Prompt Registry (YAML)** - Centralized prompt definitions in `prompts/prompt_registry.yaml` with domain-based overrides selected via `params.prompt_domain` 
- ⚡ **Performance Optimization** - Batch processing and rate limit management
- 🔒 **Security Enhancements** - Embeddable chat with domain controls + API security
- 📊 **Cost Management** - Provider-specific pricing and budget tracking
- 🌐 **Embeddable Chat Widget** - Configurable chat for any website
- 🔄 **Dual Chat Architecture** - Both stateless (`/chat`) and stateful (`/chat/{session_id}`) endpoints for different use cases
- 🧪 **Postprocessing (Markdown → HTML)** - Backend rendering with sanitized HTML, sources formatting, and scoped UI styling

---

## 🚀 Major Features

- **External `llm-adapter` package** – Provides the centralized model registry and unified provider abstraction used by the application
- **Complete abstraction** of model nuances, capabilities, and costs
- **Centralized model management** - eliminated scattered model configurations
- **Simplified application code** through registry-based model configuration
- **Application configuration now references registry keys** instead of provider-specific model parameters

### **🌐 Embeddable Chat Widget**
- **Embeddable chat widget** designed for integration into websites
- **Simple integration** – Single script tag deployment
- **Configurable via HTML attributes** (model selection, prompt domain, retrieval settings, citations, namespace)
- **Responsive iframe design** with isolated styling
- **Pipeline event streaming** for real-time stage visualization
- **Same pipeline configuration options** as the main application

### **🔒 Comprehensive Security Framework**
- **Domain-based access control** for all FastAPI routes
- **Origin and host allowlists** via `allowed_origins` (full URL) and `allowed_hosts` (hostname)
- **Automatic protection of critical endpoints** (`/chat`, `/index`, `/embed`, `/upload`, `/batch`)
- **Configurable CORS enforcement** for trusted domains
- **Graceful fallback behavior** when security configuration is not provided

### **🧠 Advanced Context & Memory Management**
- **Cache-optimized conversation history** with rolling summaries and bounded context windows
- **Chunked conversation strategy** maintaining accumulated summary + verbatim recent tail
- **Configurable chunk size** via `raw_tail_turns` parameter
- **Automatic context window management** preventing token limit overflows
- **Efficient caching behavior** through stable context patterns
- **Long-running conversation support** with semantic continuity preservation

### **⚙️ Per-Stage Model Configuration**
- **Runtime model selection** per pipeline stage (UI or API)
- **Independent configuration** for query rewrite, rerank, summarization, and inference
- **Dynamic model selection UI** for mid-conversation changes
- **Provider mixing** (e.g., OpenAI for some stages, Gemini for others)
- **API control** via FastAPI endpoints
- **Stage-level cost/performance tuning**

### **Multi-Provider LLM Support**
- **Unified LLM handler** leveraging the model registry for provider abstraction
- **Seamless provider switching** through registry key configuration
- **Provider-specific optimizations** handled transparently

### **🏗️ Domain-Aware Vector Collections**
- **Domain-based collection isolation** with automatic dimension management
- **Flexible provider assignment** - Different domains can use different embedding providers
- **Configuration-driven switching** via `active_domain` setting

### **📊 Performance & Cost Controls**
- **Batch processing** with configurable sizes and rate-limit management
- **Per-stage cost tracking** with detailed token usage metrics
- **Real-time performance monitoring** and budget visibility

### **Registry-Driven Model Configuration**
- **Gemini embeddings integration** with L2 normalization and batch processing
- **Unified model selection** using stable registry keys across all pipeline stages
- **Application now references stable registry keys** instead of provider-specific details. e.g `embedding_model_key: "openai:embed_small"`, `inference_model_key: "openai:gpt-4o-mini"` etc
- **Dramatically simplified config** - removed complex model parameter objects

For complete model registry documentation, see: https://vrraj.github.io/llm-adapter/model-registry.html


---

## What's Next

**Retrieval Enhancement:** Implementing Query Expansion (Multi-query generation) to capture broader semantic intent.

**Hybrid Search:** Augmenting vector-based retrieval with text-based search (BM25) to improve keyword accuracy.

**Advanced Reranking:** Integration of cross-encoders for high-precision result filtering.

**Identity Management:** Adding user authentication and management to enhance existing multi-user session isolation.

---

*This release represents a significant architectural evolution from v1.0.1, establishing the foundation for scalable, multi-provider AI operations with enterprise-grade security and extensibility.*
