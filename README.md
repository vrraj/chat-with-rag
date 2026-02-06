# Chat with Your Docs: End-to-End RAG Pipeline

![CI Status](https://github.com/vrraj/chat-with-rag/actions/workflows/python-ci.yml/badge.svg)

A modular RAG framework that turns unstructured data into reliable answers using query rewriting, explicit retrieval, reranking, observability, and tool-augmented inference.

This system goes beyond basic vector search by implementing a multi-stage LLM orchestration layer. It ingests complex formats (MediaWiki, PDFs, HTML), preserves document structure, and provides a fully verifiable chat experience with live-streamed **pipeline execution stages** and direct **source citations**.

## 🆕 What's New in v2.0

- **Multi-Provider LLM Handler** - Unified abstraction supporting OpenAI, Gemini, and future providers
- **Model Registry Architecture** - Centralized model management with capability definitions, pricing, and provider abstraction
- **Prompt Registry (YAML)** - Externalized prompts with global and domain-specific customization
- **Advanced Context Management** - Cache-optimized conversation history with summary injection
- **Stage-Specific Model Configuration** - Flexible runtime model selection per pipeline stage via frontend/API parameters
- **Domain-Based Collection Management** - Application domain configuration with tightly coupled collections and embedding models with automatic dimension derivation
- **Gemini Native Embeddings** - Direct SDK integration with token estimation fallback
- **Performance Optimizations** - Batch processing, rate limit management, and cost tracking
- **Embeddable Chat Widget** - Deploy to any website with domain controls and real-time streaming
- **Domain-Based Access Controls** - Domain-based access controls for all API endpoints



**Project Scope & Intent**

This project explores end-to-end RAG system design, prioritizing transparency and modularity over abstraction-heavy frameworks to make pipeline behavior explicit and observable.

## Table of Contents

- [High-Level RAG Pipeline Overview](#-high-level-rag-pipeline-overview)
- [Features](#-features)
- [🚀 Getting Started](#-getting-started)
  - [1. Prerequisites](#-1-prerequisites)
  - [⚡ 2.0 One-command setup](#-20-one-command-setup-macoslinux)
  - [2.1 Manual setup](#-21-manual-setup-step-by-step)
  - [ 2.2 Running & Managing the Application](#-22-running--managing-the-application)
- [Knowledge Base and Sample Data](#-knowledge-base-and-sample-data)
  - [Data Attribution](#-data-attribution)
  - [Explore the Data](#-explore-the-data)
  - [Managing Your Collections](#-managing-your-collections)
- [Example Queries](#example-queries)
- [Batch Ingestion](#batch-ingestion)
- [Technical Overview](#technical-overview)
- [Project Structure](#project-structure)
- [License & Usage](#license--usage)


## 🧠 High-Level RAG Pipeline Overview

  The system operates through two primary parallel workflows: an **Ingestion Pipeline** for knowledge base construction and a **Chat Pipeline** for real-time retrieval and response generation.

| **Ingestion Pipeline** (Data → Vector) | **Chat Pipeline** (Prompt → Answer) |
| :--- | :--- |
| 1. **Documents** (Single or Batch) | 1. **User Prompt** |
| 2. **Document Loading** | 2. **Query Rewrite** (Optimization) |
| 3. **Extraction** (PDF, HTML, Wiki) | 3. **Document Retrieval** (Qdrant Search) |
| 4. **Processing & Normalization** | 4. **Relevance Reranking** |
| 5. **Metadata Augmentation** | 5. **Context Construction** (History + reranked chunks) |
| 6. **Embedding Generation** (OpenAI) | 6. **Inference Context Assembly** (System + Domain + History + Chunks + Web + User) |
| 7. **Vector Storage** (Qdrant) | 7. **LLM Inference** (GPT-4o-mini) |
| | 8. **Tool Execution** (e.g., Weather, Maps) |
| | 9. **Postprocessing** (Markdown → HTML, Sources formatting) |
| | 10. **Final Response** (with Citations) |

The screenshot below illustrates how these **inference pipeline stages** work together to generate a complete response in action, showing multi-turn conversation with  rewritten query, context maintenance across turns (_compare with ..._), tool calling capabilities, **multi-model** capabilities (openai, gemini), and LLM response postprocessing (HTML-formatted responses) with citations.

<p align="center">
  <a href="images/chat-pipeline-rewrite-context-tools-inference.png">
    <img
      src="images/chat-pipeline-rewrite-context-tools-inference.png"
      style="max-width: 100%; height: auto;"
      alt="Chat pipeline UI showing query rewrite, multi-turn context, pipeline stages, tool calls, and citations"
    />
  </a>
</p>

*Chat pipeline UI showing query rewriting, multi-turn context handling, explicit pipeline stages, tool invocation, and cited responses.*

This workspace is the **main entry point to the application**, combining navigation, ingestion, and operational tooling into a single interface. It provides access to chat configuration, embeddable experiences, vector store inspection, document management, and batch ingestion, making the full lifecycle of data and retrieval behavior visible and controllable from one place.

<p align="center">
  <a href="images/content-ingestion-primary-actions.png">
    <img
      src="images/content-ingestion-primary-actions.png"
      style="max-width: 100%; height: auto;"
      alt="Content ingestion UI with primary actions and indexing tools for PDFs, HTML, and MediaWiki"
    />
  </a>
</p>

*Content ingestion UI showing primary actions and indexing tools (batch upload, PDF/HTML/MediaWiki), estimation mode, and metadata controls.*
> **Auth & Security Note**

This application includes a **domain-based access control framework** with built-in security for API endpoints and embedded widgets. Key features include domain isolation, collection separation, widget lockdown, and configuration-driven security. See the **Security & Deployment** section below for comprehensive implementation details and additional deployment recommendations.

## ✨ Features

An end-to-end modular RAG ecosystem that orchestrates advanced LLM workflows to synthesize raw documents into structured intelligence, featuring a high-fidelity ingestion engine and live observability for verifiable, context-grounded insights.

### 📥 High-Fidelity Ingestion
* **Multi-Source Extraction**: Native support for high-fidelity parsing of **PDFs**, **MediaWiki**, and **HTML**.
* **Intelligent Processing**: 
    * **Smart Chunking**: Configurable strategies to preserve semantic context across fragments.
    * **Structure Preservation**: Maintains the integrity of complex tables and structured layouts.
    * **Noise Filtering**: Automated removal of headers, footers, and irrelevant boilerplate for cleaner context.
* **Batch & Scale**: Process local directories (`file://`) or remote URLs with built-in **token and cost estimation** before committing to storage.

### 🌐 Embeddable Chat Widget
- **Embeddable chat** for any website with configurable data attributes and advanced controls.

See the **Embeddable Widget Configuration** section below for detailed implementation examples and all available options.

### 🧠 Advanced Chat Orchestration

**Advanced Chat Orchestration is the control plane of the system — it governs which models run, in what order, with which prompts, context, providers, and output formats for every request.**

---

#### 1️⃣ Pipeline Control & Execution Flow  
*What runs, when, and how each stage is configured*

- **Multi-Stage LLM Pipeline Orchestration**  
  Granular control over each stage: Query Rewrite → Retrieval → Rerank → Summarization → Inference → Tools → Post-processing.

- **Stage-Specific Model Configuration**  
  Independent model selection per stage (rewrite, rerank, inference, summarization), configurable at runtime via UI or API.

- **Dynamic Model Selection (UI)**  
  “Configure Models” modal enables per-stage model changes mid-conversation, including mixing providers (OpenAI + Gemini).

- **API-Level Control**  
  Full pipeline configuration is available programmatically via FastAPI endpoints for automation, integrations, and experimentation.

- **Multiple LLM Provider Support Framework**  
  A unified execution framework that abstracts multiple provider surfaces behind a single contract:
  - OpenAI Chat Completions API  
  - OpenAI Responses API  
  - Gemini OpenAI-compatible adapter  
  - Gemini native SDK  
  Provider-specific differences are normalized so the pipeline remains provider-agnostic.

- **Model Registry Centralization**  
  All model definitions are centralized in a registry that abstracts provider- and model-specific nuances (capabilities, pricing, parameters).  
  These definitions are consumed uniformly by the LLM handler and the application to drive routing, validation, and cost tracking.

- **Prompt Registry (YAML-Driven Control)**  
  All prompts for rewrite, rerank, summarization, and inference are centralized in a prompt registry, fully decoupled from code.  
  The registry supports:
  - Global default prompts
  - Domain-specific augmentation (e.g., finance, geography)
  - Instruction and example layering  
  Runtime context (history, retrieved chunks, web results, user constraints) is injected by the application, not hardcoded in prompts.

- **Post-processing & Output Transformation**  
  Final model outputs pass through a dedicated post-processing stage that formats responses into structured HTML (tables, highlights, links) for the UI.  
  This module is extensible and intentionally separated from inference, allowing future output formats beyond HTML.

---

#### 2️⃣ Context & Memory Management

Conversation history is managed using a bounded, cache-aware strategy that preserves semantic continuity without exceeding context limits. The system maintains an accumulated conversation summary alongside a verbatim recent tail of turns. When the tail reaches a configurable size, it is summarized and folded into the accumulated context while the tail resets. This ensures stable context size, efficient caching behavior, and continuity across long-running conversations.

---

#### 3️⃣ Query Intelligence & Rewrite Optimization

Before retrieval, user queries may be expanded or clarified by a rewrite stage that improves recall and precision. Rewrite behavior is confidence-gated, ensuring rewritten queries only replace the original when confidence exceeds a threshold. The rewrite stage can draw from recent turns verbatim, summarized history, or be disabled entirely, allowing precise control over how user intent is refined.

---

#### 4️⃣ Retrieval, Inference & Tool Augmentation  
*How answers are synthesized and verified*

- **Retrieval Optimization**
  - Vector search via Qdrant with configurable Top-K and distance thresholds
  - Secondary semantic reranking for relevance

- **Inference Context Assembly**  
  Final LLM prompts are composed from:
  - System instructions (prompt registry)
  - Domain-specific augmentations
  - Conversation summary and verbatim tail
  - Retrieved and reranked document chunks
  - Optional web search context
  - User instructions and query

- **Tool Execution**
  - Native function/tool calling (weather, airports, web search, etc.)
  - Tool outputs are merged into the final synthesis stage

- **Verified Citations**
  - All final answers include deep-linked citations to source documents
* **Real-Time Observability**: Live **SSE (Server-Sent Events)** stream providing a window into the "thoughts" and progress of the RAG flow as it happens.
* **Granular Cost Tracking**: Instant transparency with per-stage token usage and dollar-cost metrics for every request.
* **Extensible Tooling**: Built-in support for function calling (e.g., weather, local APIs) to augment responses with live, real-time data.

#### 🧪 Postprocessing (Markdown → HTML)
After inference, the system optionally postprocesses the assistant’s text to render rich Markdown content in the chat UI:
- **Backend rendering** (`backend/markdown_render.py`): Converts Markdown to sanitized HTML using `markdown-it-py`/`Markdown` with `bleach`. Wraps tables in scrollable containers and hardens links.
- **Sources formatting**: Detects and splits any `Sources:` block so it starts on a new line, with each source on its own line, and makes the heading bold.
- **Frontend rendering**: When `params.render_html=true`, the backend returns `answer_html` (and `finalHtml` via SSE). The frontend uses `innerHTML` to display rendered content, with scoped CSS for tighter spacing and table styling.
- **Fallback**: If HTML rendering is disabled or fails, the frontend gracefully falls back to plain text (`textContent`).

Web search is supported via an optional automatic web context stage and via an LLM tool call. See the Web Search section below for details.


##  Getting Started

Get the system running in minutes using the provided `Makefile`. This setup uses Docker for the core infrastructure while maintaining a developer-friendly local environment through volume mounting.

### 📋 1. Prerequisites
Ensure your environment meets these requirements before proceeding:
- **OS:** macOS or Linux (Windows supported via Docker).
- **Git** – required to clone the repository. Install: https://git-scm.com/downloads
- **Docker & Docker Compose:** Required for the Qdrant v1.14.1 database and the web app container. [Get Docker here](https://docs.docker.com/get-started/)
- **Python 3.10+:** Required for local development, IDE support, and ingestion scripts.
- **OpenAI API Key:** Required for embeddings and chat pipeline. [Get one here](https://platform.openai.com/api-keys)



### ⚡ 2.0 One-command setup (macOS/Linux)

If you prefer fewer copy/paste steps, you can **run the setup in one go**.

This script will:
- Create `.env` (if missing) and prompt you for `OPENAI_API_KEY`
- Start Docker services (`make start`)
- Create a Python venv, install dependencies, and seed sample data (`make seed`)

> [!IMPORTANT]
> The API key is written to your local `.env` file. Treat it like a password (don't commit it).
>
> Before running the setup, create an **OpenAI API Platform** key and set a **hard usage limit** (budget + alerts) in your OpenAI Dashboard.
> See: [2.1.3 Configure OpenAI API & Costs](#213-configure-openai-api--costs)

Paste the commands below into your terminal.

**Step 1 — Clone the repo**

```bash
git clone https://github.com/vrraj/chat-with-rag.git
cd chat-with-rag


```

> **Note:** You will be prompted to enter your OpenAI API key when you run the setup script. The API key is stored in `.env`. Treat it like a password — never commit it to Git.

**Step 2 — Run the setup**

```bash
bash scripts/rag_setup.sh

```

> **Troubleshooting (macOS):** If `make smoke_api` or `python3 scripts/api_smoke_test.py` fails with `SSL: CERTIFICATE_VERIFY_FAILED`, your Python install may be missing trusted root certificates.
>
> Run:
```bash
open "/Applications/Python 3.12/Install Certificates.command"
```


### 🛠️ 2.1 Manual setup (step-by-step)

> If you ran the **2.0 One-command setup** above, you can skip this entire section.

**2.1.1) Verify Docker Installation**
```bash
# Verify Docker
 docker --version

# Verify Docker Compose (v2 or v1)
 docker compose version || docker-compose --version
```
 You should see version numbers if Docker is installed correctly.

> **Note for Linux Users:** If you get "permission denied," add your user to the docker group: `sudo usermod -aG docker $USER` and then log out/in.

**2.1.2) Clone the Repository**
``` bash
git clone https://github.com/vrraj/chat-with-rag.git
cd chat-with-rag

```

#### 2.1.3 Choose Your AI Provider(s)

This application supports **multiple AI providers** with different capabilities:

| Provider | Default Use | Key Features | Requirements |
|----------|---------------|---------------|---------------|
| **OpenAI** | ✅ Default provider | Standard models, reasoning models, native API | OpenAI Platform account |
| **Gemini** | ✅ Optional provider | Thinking models, OpenAI-compatible API | Google AI Platform account |

> **Note**: You can use **either** provider or **both** for different pipeline stages.

### **Key Features by Provider:**

#### **OpenAI Models**
- **Standard Models**: gpt-4o, gpt-4o-mini
- **Reasoning Models**: o1-mini, o3-mini
- **Embeddings**: text-embedding-3-small, text-embedding-3-large
- **Query Rewrite**: gpt-4o-mini (recommended for optimal performance)

#### **Gemini Models**
- **Standard Models**: gemini-2.5-flash-lite, gemini-2.5-pro
- **Thinking Models**: gemini-2.5-flash-lite (thinking_budget), gemini-2.5-pro (thinking_level)
- **Embeddings**: gemini-embedding-001

### **Configuration Options:**
- **OpenAI-only**: Use all features with default setup
- **Gemini-only**: Requires embedding model change (see 2.1.4)
- **Mixed**: Use different providers for different stages

#### 2.1.4 Configure API Keys & Costs

Choose your provider(s) and set up appropriate API keys:

##### Option A: OpenAI (Recommended for Getting Started)
> [!IMPORTANT]
> Required for **default configuration** and sample data testing.

| Recommendation | Action | Rationale |
| :--- | :--- | :--- |
| **Budget** | Set a limit of **$5–$10**. | Establishes a safety ceiling for testing. |
| **Dedicated Key** | Name it `chat-with-rag`. | Isolates usage tracking for this specific project. |
| **Alerts** | Set a 50% notification. | Provides proactive cost control. |

##### Option B: Gemini (Alternative Provider)
> [!IMPORTANT]
> Optional alternative to OpenAI. Requires configuration changes for full compatibility.

| Recommendation | Action | Rationale |
| :--- | :--- | :--- |
| **Quota** | Set a **daily quota limit** based on your budget. | Prevents unexpected cost overruns. |
| **Dedicated Key** | Name it `chat-with-rag-gemini`. | Isolates usage tracking for this project. |
| **Monitoring** | Enable **usage alerts** in Google Cloud Console. | Provides proactive cost visibility. |

> **Note:** Gemini uses quota-based limits instead of hard dollar limits. Configure quotas in Google AI Studio or Google Cloud Console.

##### Option C: Both Providers (Advanced)
> Use different providers for different pipeline stages:
> - OpenAI for embeddings (default sample data compatibility)
> - Gemini for inference models (thinking capabilities)

### 2.1.6 LLM Providers, Models, and Endpoints

The application uses a centralized **model registry** (`backend/llm/model_registry.py`) and **LLM handler** (`backend/llm/llm_handler.py`) to route requests to the correct provider, model, and API surface. You typically configure models via registry keys (e.g. `openai:fast`, `gemini:fast`, `gemini:embed`).

#### OpenAI Provider

Routed through the native OpenAI Python client using the **Responses API**:

- **Chat / Completions**
  - Endpoint: `responses.create` (internally `_openai_call` in `LLMHandler`).
  - Example registry profiles:
    - `openai:fast` → `gpt-4o-mini` (chat/inference, tools, streaming).
    - `openai:best` → `gpt-4o` (higher-quality chat/inference).
    - `openai:reasoning` → `o3-mini` (reasoning tasks when enabled).

- **Embeddings**
  - Endpoint: `client.embeddings.create(model=..., input=...)`.
  - Example registry profiles:
    - `openai:embed_small` → `text-embedding-3-small`.
    - `openai:embed_large` → `text-embedding-3-large`.

#### Gemini Provider (OpenAI-Compatible Adapter)

Routed through an OpenAI-compatible Gemini endpoint (e.g. `GEMINI_OPENAI_BASE_URL`), but still surfaced via OpenAI-style clients in `LLMHandler`.

- **Chat / Completions**
  - Endpoint: `chat.completions.create` on the Gemini adapter client.
  - Registry profiles:
    - `gemini:fast` → `models/gemini-2.5-flash-lite`, endpoint=`"chat_completions"`.
      - Capabilities: tools, streaming, temperature, top_p, etc.

- **Embeddings (Adapter Path)**
  - Endpoint: `embeddings.create(model=..., input=..., dimensions=...)` on the Gemini adapter client.
  - Registry profiles:
    - `gemini:embed` → `gemini-embedding-001`, endpoint=`"embeddings"`.
      - Capabilities: `dimensions=1536`, `normalize_embedding=True`.
  - `LLMHandler` wraps this in `_gemini_embedding_call`, adds a usage shim when missing, and can optionally L2-normalize vectors based on config.

#### Gemini Provider (Native SDK)

For some experimental and advanced use cases, the app can talk directly to Gemini via the native `google-genai` SDK, using a **separate endpoint type**.

- **Chat / Generative Content**
  - Endpoint: `client.models.generate_content(...)` wrapped by `_gemini_sdk_call` and `_GeminiSDKResponsesWrapper`.
  - Exposes a Responses-like surface with `output_text`, `output`, and normalized `usage`.

- **Embeddings (Native SDK Path)**
  - Endpoint: `client.models.embed_content(model=..., contents=..., config=EmbedContentConfig(...))`.
  - Registry profile:
    - `gemini:native-embed` → `gemini-embedding-001`, endpoint=`"gemini_sdk"`.
      - Capabilities:
        - `dimensions`: 1536
        - `task_type`: `RETRIEVAL_DOCUMENT`
        - `output_dimensionality`: 1536
        - `normalize_embedding`: True
  - `LLMHandler._gemini_native_embedding_call` uses these capabilities to build `EmbedContentConfig` and returns an OpenAI-style embeddings response (`data[].embedding`, `usage`).

#### Model Registry as Source of Truth

All of the above profiles live in `backend/llm/model_registry.py` and are referenced from config via stable keys (e.g. `embedding_model_key`, `rewrite_model_key`, `inference_model_key`). The registry defines, for each key:

- `provider`: `"openai"` or `"gemini"`.
- `model`: provider-native model id (`gpt-4o-mini`, `models/gemini-2.5-flash-lite`, `gemini-embedding-001`, etc.).
- `endpoint`: which code path `LLMHandler` should use (`"responses"`, `"chat_completions"`, `"embeddings"`, or `"gemini_sdk"`).
- `pricing`: input/output token rates used for per-stage cost calculation.
- `capabilities`: feature flags (tools, streaming, temperature, reasoning_effort, dimensions, normalize_embedding, etc.).

When you change a registry profile or pick a different key in `backend/core/config.py`, the LLM handler and chat manager automatically route to the correct provider, model, and endpoint while keeping cost accounting and parameter handling consistent.

#### 2.1.5 Set up local environment variables

Copy the example environment file and add your API key(s).

> **Note:** Optional: Advanced users may instead set API keys as OS environment variables.  
> If set, they will take precedence over the values in `.env`.
> If you used the one-command setup script above, it will prompt for OpenAI key and write it into your local .env automatically.

```bash
cp .env.example .env
# IMPORTANT: Open .env and add your API keys
vi .env   # or use 'nano .env' / your preferred text editor

# Add one or both keys:
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

#### 2.1.6 Provider-Specific Configuration

##### Using Gemini as Primary Provider
If you prefer Gemini over OpenAI:

1. **Update Embedding Model** (in `backend/core/config.py`):
   ```python
   embedding_model = "gemini:embed"  # Change from "openai:embed_small"
   ```

2. **Re-index Sample Data**:
   ```bash
   # Export current data
   # Visit frontend/list-docs.html → Export as JSON
   
   # Process with new embeddings
   # Visit frontend/process-batch-docs.html → Import JSON
   ```

##### Mixed Provider Setup
Use OpenAI for embeddings (sample data compatibility) + Gemini for inference:

```python
# In config - no embedding change needed
embedding_model = "openai:embed_small"  # Keep default

# In UI or API calls - use Gemini models
provider="gemini"
model="models/gemini-2.5-flash-lite"
```

#### 2.1.7 Start Infrastructure

```bash
make start

```
> **Note for macOS Users:**
> `make start` will automatically attempt to launch Docker Desktop if it isn't running. The script will pause briefly while the daemon initializes.


#### 2.1.8 Initialize environment and seed data

To see the RAG system in action immediately, load the sample dataset (~50 outdoor-themed Wikipedia pages). This requires a local Python environment.

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate
# Install dependencies and seed Qdrant
pip install -r requirements.txt 
make seed
deactivate

```

#### 2.1.7) Access the interface
Once the seeding is complete, open your browser and start chatting: 👉 http://localhost:8000


### ▶️ 2.3 Running & Managing the Application

Use the following Make targets to manage the application lifecycle:

- **Start the application stack (Qdrant + web app):**
  ```bash
  make start

  ```

- **Stop the application stack:**
  ```bash
  make stop

  ```

These commands are the recommended way to run and shut down the system locally.

For additional Make targets (logs, reset, reseed, maintenance utilities), refer to:
- the `Makefile` in the project root
- **TECHNICAL_OVERVIEW.md**

For details on the stateless chat API (`POST /chat`) used by `frontend/chat.html`, including request/response shape and parameter contract, see:

👉 **[README_CHAT_API.md](README_CHAT_API.md)**

---

## 🧩 Prompt Registry (YAML)

This repo uses a YAML-based prompt registry to keep prompts centralized and avoid drift between code paths.

### Registry file

- **Path:** `prompts/prompt_registry.yaml`
- **Role:** Source of truth for stage prompt text and templates.
- **Current coverage:** Inference and query rewrite are registry-driven; rerank and summarization use the registry for their fixed instructions/templates.

### Prompt domains (`params.prompt_domain`)

You can select a prompt domain per request using `params.prompt_domain`.

- If `prompt_domain` is empty or omitted, the system uses `global_defaults`.
- If `prompt_domain` is set (example: `mountains`), the system applies domain-specific overrides (currently by appending additional domain system instructions).

In the UI (`frontend/chat.html`), the **Prompt Domain** dropdown under **Inference** controls the value sent on every chat request.

### Debug logging (safe by default)

The backend logs:

- Which domain was resolved for inference.
- A short tail snippet of the resolved system instruction.

To log the full resolved prompt/template for debugging, set:

- `PROMPT_REGISTRY_LOG_FULL=1`

---

---

## 🌐 Embeddable Chat Widget

Embeddable chat for any website with comprehensive configuration options.

### **Basic Integration**
```html
<script src="https://your-server.com/chat-embed.js"></script>
<div id="chat-embed" 
     data-api-url="https://your-server.com"
     data-namespace="oceans">
</div>
```

### **Configurable via Data Attributes**
- **`data-model_key`** - Preselected model from registry (e.g., "openai:gpt-4o-mini")
- **`data-temperature`** - Response creativity control (0.0-1.0)
- **`data-top_k`** - Retrieval configuration (number of documents to retrieve)
- **`data-show_processing_steps`** - Enable streaming visualization of pipeline stages
- **`data-show_citations`** - Source citation display toggle
- **`data-namespace`** - Custom knowledge base isolation

### **Advanced Controls**
Full access to all pipeline tuning parameters:
- **Rewrite context configuration** (tail turns, summary turns, confidence threshold)
- **Inference context management** (history chunking, conversation limits)
- **Model selection per stage** (rewrite, rerank, inference, summarization)
- **Retrieval parameters** (top_k, distance thresholds, filters)

### **Features**
- **Easy integration** - Single script tag deployment
- **Responsive iframe design** with isolated styling
- **Real-time streaming** support for processing visualization
- **Domain-based security** - Widget only works on authorized domains

### **Example Configurations**

#### **Simple Chat Widget**
```html
<div id="chat-embed" 
     data-api-url="https://your-server.com"
     data-top_k="5"
     data-show_citations="true">
</div>
```

#### **Advanced Configuration**
```html
<div id="chat-embed" 
     data-api-url="https://your-server.com"
     data-model_key="openai:gpt-4o-mini"
     data-temperature="0.7"
     data-top_k="10"
     data-show_processing_steps="true"
     data-show_citations="true"
     data-namespace="oceans">
</div>
```

---

## 🌐 Web Search (Two Paths)

The system supports web search (DuckDuckGo Instant Answer API) in two distinct ways:

1. **Automatic Web Context (`web_context`)**
   - Enabled by:
     - `backend/core/config.py`: `use_web_search` (default toggle; default is `False`)
     - Request override: `POST /chat` payload field `use_web_search` (when provided)
   - Behavior:
     - Runs as part of the chat pipeline stage **Establish Web Context**.
     - Adds a `WEB SEARCH RESULTS:` block to the inference prompt.
     - Web results are cited as `[web-1]`, `[web-2]`, ... and can appear in the final `Sources:` block.

2. **LLM Tool Call (`web_search` tool)**
   - Enabled when tools are enabled and the model chooses to call the tool.
   - Behavior:
     - Returns a formatted text block of results.
     - Tool outputs are provided to the synthesis stage as `[SOURCE: TOOL - web_search] ...`.

Both paths use the same underlying DuckDuckGo Instant Answer extraction logic (see `backend/chat/web_search.py`), but they are injected into the LLM context differently.

### 🧪 2.4 Developer Mode (Optional)

To enable **hot-reload** (Uvicorn reload) for active development:

1.  **Open `docker-compose.yml`**.
2.  **Change the command** from `python start.py` to `python run.py`.
3.  **Restart the container** (run `make start` again).

> [!TIP]
> Because the webapp is a **volume mount**, any changes you make to your local `.py` files will reflect instantly inside the container. `run.py` detects these changes and triggers an automatic Uvicorn restart.

---


## 📚 Knowledge Base and Sample Data

When you run `make seed`, the system populates Qdrant with a high-quality sample dataset of approximately **50 Wikipedia pages**. This focus on world-renowned mountains, national parks, and trails provides a rich environment to test the RAG pipeline's accuracy.

### 📄 Data Attribution
To demonstrate multi-source RAG capabilities, this project includes a sample knowledge base derived from Wikipedia.
* **Source:** 55 curated Wikipedia articles processed via a custom high-fidelity MediaWiki extraction pipeline.
* **Integrity:** Source URLs and author metadata are preserved within the vector payloads to enable **verified citations**.
* **License:** Distributed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
* **Full Credits:** Detailed source links and compliance information can be found in [ATTRIBUTIONS.md](./ATTRIBUTIONS.md).

### 🔍 Explore the Data

You can verify the indexed documents through the web interface or the command line:

| Method | Action |
| :--- | :--- |
| **Frontend UI** | Navigate to the **"View Documents"** page to see titles, URLs, and metadata. |
| **Terminal (CLI)** | Run the following to list the first 100 document titles: |

```bash
source venv/bin/activate
python qdrant_scripts/qdrant_ops.py --list-titles --limit 100

```

### 🔄 Managing Your Collections

The system supports **domain-based collection management** where each domain is tightly coupled with its embedding model to prevent dimension drift and ensure consistency. Selecting a collection automatically sets the compatible embedding model and vector dimensions.

#### **Domain-Based Configuration (Recommended)**

The new approach uses a single `active_domain` setting that automatically configures both the collection name and embedding model:

```python
# In backend/core/config.py
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

# Active domain selection (single change point)
active_domain: str = "oceans"
```

**Benefits:**
- 🎯 **Single Change Point**: Only change `active_domain` to switch both collection and model
- 🔗 **Automatic Linking**: Collection and embedding model are always correctly paired
- 📏 **Dynamic Vector Size**: Automatically computed from the embedding model's dimensions
- 🌐 **Domain-Specific**: Each domain can use different providers (OpenAI, Gemini)
- 🛡️ **Dimension Safety**: Prevents model/collection dimension mismatches

**Usage Examples:**
```python
# Switch to oceans domain (Gemini embeddings)
active_domain: str = "oceans"
# → collection_name = "document_index_gemini"
# → embedding_model_key = "gemini:native-embed" 
# → vector_size = 1536 (auto-derived from model registry)

# Switch to mountains domain (OpenAI embeddings)
active_domain: str = "mountains"
# → collection_name = "document_index"
# → embedding_model_key = "openai:embed_small"
# → vector_size = 1536 (auto-derived from model registry)
```

#### **Creating New Collections**

When creating a new collection in the UI or via API:
1. **Choose Domain**: Select or create a domain configuration
2. **Model Auto-Selected**: The compatible embedding model is automatically assigned
3. **Dimensions Set**: Vector size is derived from the model registry
4. **Ready to Use**: Collection is immediately ready for ingestion with the correct model

This tight coupling ensures that you can never accidentally use incompatible embedding models with existing collections, preventing data corruption and search failures.

#### **Legacy Manual Configuration**

If you prefer manual configuration, you can still manage collections directly:

1.  **Open `backend/core/config.py`**.
2.  **Create a new domain entry** in `DOMAIN_EMBEDDING_CONFIG`:
    ```python
    "my_domain": {
        "collection_name": "my_custom_knowledge_base",
        "embedding_model_key": "openai:embed_small"
    }
    ```
3.  **Set the active domain**:
    ```python
    active_domain: str = "my_domain"
    ```
4.  **Restart the app**. The system will automatically detect the missing collection and create a fresh one in Qdrant.

> [!TIP]
> This approach allows you to maintain multiple "knowledge bases" on the same server. You can swap between domains at any time just by changing the `active_domain` variable.

#### **Collection Management Options**

**Option A: Create Fresh Collections (Recommended)**
Each domain automatically gets its own collection when first used. No manual setup required.

**Option B: Clear Existing Collection**
Use this if you want to completely clear a collection but keep using the same name.

> [!WARNING]
> This action will permanently delete the collection and all vectors within it. This cannot be undone.

```bash
# Activate your environment
source .venv/bin/activate

# Delete specific collection
python qdrant_scripts/qdrant_ops.py --delete-collection document_index_gemini

# Or delete the current active collection
python qdrant_scripts/qdrant_ops.py --delete-collection $(python -c "from backend.core.config import settings; print(settings.collection_name)")
```


### 💬 Example Queries
The following examples are based on the seed data.

1. #### Retrieval-based
	•	“Where is Mount Kilimanjaro located?”

	•	“Compare Kilimanjaro with Mount Whitney.”

  • "Elevation of mont blanc in the alps , the current weather and the closest airport"
  
  •	“Which mountain is taller — Everest or Whitney?”

	•	“Where is Mount Whitney located? What is the closest airport to it?”
  
	•	“Tell me about hiking in Patagonia. What is the weather forecast for the area?”

2. ####  Multi-Turn RAG + Tool Calling Demo (Conversation Chain)

The following is a real conversation demonstrating:

- Multi-turn context retention  
- Retrieval + rewrite + rerank  
- Automatic tool invocation (`get_nearby_airports`)  
- Weather queries over prior tool results  
- Final synthesis combining retrieval + tool calls  
- Per-stage cost and token metrics  

USER: where is mount kilimanjaro ?
ASSISTANT:Mount Kilimanjaro is located in Tanzania....

USER: compare kilimanjaro with Mount whitney

ASSISTANT: Mount Kilimanjaro and Mount Whitney are both prominent peaks, but differ in height, location....

USER: what are the closest airports to Mount Kilimanjaro

ASSISTANT:The nearest airport to the coordinates (-3.0674, 37.3556) is **Amboseli Airport (ASV • HKAM)**, ~48 km away.
Tools Used: get_nearby_airports

USER: and for Mount Whitney

ASSISTANT:Closest airports to Mount Whitney (36.5785, -118.2919)....
Tools Used: get_nearby_airports

USER: what is the weather forecast for mount whitney

ASSISTANT:Mount Whitney, California — 28 Oct  High: 7.2 °C (45 °F)....
Tools Used: get_weather


<p align="center">
<img src="images/multi-turn-conversation-with-tools.png" width="100%" style="max-width: 100%; height: auto;" alt="Multi-turn conversation with tool calls and citations. Preserves context and history across turns." />
</p>

*Multi-turn conversation with tool calls and citations. Preserves context and history across turns.*

---

## 📦 Batch Ingestion

> **Note:** Changing the embedding model requires re-embedding and rebuilding the vector index. See **TECHNICAL_OVERVIEW.md** for the recommended re-ingestion workflow.

### 📊 Embedding Provider Limits

When configuring chunk sizes and batch processing, be aware of provider-specific limits:

| Feature | OpenAI (text-embedding-3-small/large) | Gemini (gemini-embedding-001) |
|---------|----------------------------------------|-------------------------------|
| **Max Inputs per Request** | 2,048 texts | 250 texts |
| **Max Tokens per Request** | Variable (often restrict ed by Tier) | 20,000 tokens |
| **Max Tokens per Text** | 8,191 tokens | 2,048 (or 8,000 on newer models) |
| **Truncation Behavior** | Manual (must be handled by user) | Silent (automatic) by default |
| **Batch API Support** | Yes (up to 50,000 requests/file) | No (synchronous only via API) |

> **Note**: These limits affect how you should configure `chunk_size` and `embedding_batch_size` in `backend/core/config.py`. Always check current provider documentation for the latest limits.

The system supports batch processing of multiple documents (PDFs, web pages, MediaWiki) with the following directory structure and requirements:

### Directory Structure for Local PDFs
When using local file paths in batch processing, the backend expects the following structure:

```
chat-with-rag/
├── data/
│   └── pdf-files-for-upload/  # Recommended directory for PDFs
│       ├── document1.pdf
│       ├── document2.pdf
│       └── document3.pdf
```

### Path Handling for batch processing 
- **Relative Paths (Recommended)**: Use paths relative to the project root
  - Example: For batch processing documents (Front End: /process-batch-docs.html) from a folder: /app/data/pdf-files-for-upload 
  - Example direct file reference: `./data/pdf-files-for-upload/document1.pdf`

- **Absolute Paths**: Must be accessible within the Docker container
  - Example: `/app/data/pdf-files-for-upload/document1.pdf`

### Batch Processing Features
- Process multiple PDFs, web pages, or MediaWiki articles in a single operation
- Skip common sections (References, External links, etc.)
- Set global or per-document chunking and processing options
- Preview and edit configuration before processing

### Embedding Batch Indexing

To reduce latency and API overhead, the ingestion pipeline batches **multiple chunks** into a single embeddings call wherever possible:

- **Pre-chunked ingestion** (`/mediawiki/url`, `/index`, `frontend/process-batch-docs.html`)
  - Uses `EmbeddingsManager.index_chunks`, which groups chunks into batches and calls `llm_handler.embeddings.create` once per batch.
- **Raw document ingestion** (HTML/PDF via `/index` and `/pdf`)
  - Uses `EmbeddingsManager.process_document`, which also batches chunk texts before calling the embedding provider.

Batch size is provider-aware and configurable in `backend/core/config.py`:

```python
embedding_batch_size_default: int = 25
embedding_batch_size_openai: int = 25
embedding_batch_size_gemini: int = 25
```

- For **OpenAI embeddings**, the system uses `embedding_batch_size_openai`.
- For **Gemini embeddings**, the system uses `embedding_batch_size_gemini`.
- Any future providers fall back to `embedding_batch_size_default`.

The effective behavior is roughly:

- `num_chunks = 40`, `embedding_batch_size_gemini = 25` → 2 embedding calls (25 + 15 chunks).
- Token usage and cost accounting remain accurate because each batched call returns aggregate usage, which is tracked per document.

### Example Batch Configuration
```json
{
  "items": [
    {
      "url": "file:///app/data/pdf-files-for-upload/document1.pdf",
      "doc_type": "pdf",
      "skip_sections": ["References", "External links"]
    },
    {
      "url": "https://en.wikipedia.org/wiki/Example",
      "doc_type": "mediawiki"
    }
  ],
  "max_chunks": 100,
  "estimate": true,
  "force_delete": false
}
```

### Best Practices
1. Place all PDFs in the `data/pdf-files-for-upload` directory
2. Use relative paths when possible for better portability
3. Start with `"estimate": true` to preview processing before actual ingestion
4. Check the web interface's "View Documents" page to verify successful ingestion

---

## 🧠 Reasoning vs Non-Reasoning Models

This system supports both traditional LLMs and advanced reasoning models, with different behaviors and optimizations for each type.

### Model Categories

#### **Non-Reasoning Models (Traditional LLMs)**
- **Examples**: `gpt-4o-mini`, `gpt-4`, `claude-3-haiku`, `gemini-1.5-flash`
- **Characteristics**: Fast, direct response generation
- **Use Case**: General Q&A, quick responses, cost-effective queries

#### **Reasoning Models (Advanced LLMs)**
- **Examples**: `gpt-5-mini`, `o3-mini`, `o3`, `gpt-5`
- **Characteristics**: Slower, deeper analysis, step-by-step reasoning
- **Use Case**: Complex queries, multi-step problems, analytical tasks

### Key Differences

#### **Query Rewrite**
- **Non-Reasoning**: Uses reasoning models for query optimization
- **Reasoning**: **Do NOT use reasoning models for query rewrite** - they're reserved for the final response generation

#### **Tool Calls**
- **Non-Reasoning**: Standard tool execution via ChatCompletions API
- **Reasoning**: **Uses OpenAI Responses API format** with special handling:
  ```python
  # Non-Reasoning (ChatCompletions):
  choices[0].message.tool_calls = [{"function": {...}}]
  
  # Reasoning (Responses API):
  output = [
      ResponseReasoningItem(type="reasoning", ...),
      ResponseOutputMessage(type="message", content=[
          ResponseOutputText(...),
          ResponseToolCall(name="get_weather", arguments={...})
      ])
  ]
  ```

#### **Prompt Engineering**
- **Non-Reasoning**: Standard RAG prompt with tool encouragement
- **Reasoning**: **Modified prompts to encourage tool usage**:
  ```python
  # Before: "If context insufficient, reply exactly with: I couldn't find..."
  # After:  "If context insufficient, USE THE AVAILABLE TOOLS to gather information"
  ```

#### **Response Format**
- **Non-Reasoning**: Direct text response with citations
- **Reasoning**: Structured response with reasoning steps:
  ```python
  output = [
      ResponseReasoningItem(type="reasoning", summary=[...]),
      ResponseOutputMessage(type="message", content=[...])
  ]
  ```

### Performance Considerations

#### **Speed**
- **Non-Reasoning**: 1-3 seconds typical
- **Reasoning**: 5-15 seconds typical (due to reasoning process)

#### **Cost**
- **Non-Reasoning**: Lower cost per query
- **Reasoning**: Higher cost but higher accuracy for complex tasks

#### **Token Usage**
- **Non-Reasoning**: Input + Output tokens
- **Reasoning**: Input + Output + **Reasoning tokens** (additional cost for thinking process)

---

## 📊 Metrics and Costs

### Token Accounting

The system tracks and costs tokens based on provider/model reporting. Current token accounting includes:

#### **Tracked Token Types**
- **Prompt Tokens** (Input): Tokens sent to the model (user message + context)
- **Cached Tokens**: Cached prompt tokens (lower cost)
- **Completion Tokens** (Output): Tokens in the model's response to the user

#### **Reasoning Tokens Handling**
- **Provider Reporting**: Most providers (OpenAI, Gemini) include reasoning/internal thinking tokens as part of the completion tokens count
- **Costing**: Reasoning tokens are billed at the same rate as regular completion tokens
- **No Separate Tracking**: Currently, reasoning tokens are not tracked separately from completion tokens in the metrics

### Token Limit Configuration

#### **max_completion_tokens Considerations**
When setting `max_completion_tokens`, consider the following for reasoning models:

- **Gemini Models**: `thinking_level` affects internal reasoning token allocation
  - `thinking_level="minimal"`: More tokens available for actual response
  - `thinking_level="low"`: Moderate reasoning, balanced response length
  - `thinking_level="medium"`: More reasoning, shorter responses
  - **Recommendation**: Increase `max_completion_tokens` for higher `thinking_level` values

- **OpenAI Reasoning Models**: `reasoning_effort` affects internal token usage
  - `reasoning_effort="low"`: Minimal reasoning, maximum response tokens
  - `reasoning_effort="medium"`: Balanced reasoning and response
  - `reasoning_effort="high"`: Maximum reasoning, shorter responses
  - **Recommendation**: Adjust `max_completion_tokens` based on desired reasoning vs response balance

#### **Configuration Examples**
```python
# Gemini with minimal thinking (max response tokens)
max_completion_tokens = 800
thinking_level = "minimal"

# Gemini with medium thinking (balance reasoning and response)
max_completion_tokens = 1200
thinking_level = "medium"

# OpenAI with high reasoning effort
max_completion_tokens = 1000
reasoning_effort = "high"
```

### Cost Calculation

Costs are calculated per-stage using the following formula:
```
Total Cost = (Prompt Tokens × Input Rate) + 
             (Completion Tokens × Output Rate) + 
             (Cached Tokens × Cached Rate)
```

#### **Rate Sources**
- **Model Registry** (`backend/llm/model_registry.py`): Provider-specific pricing per model
- **Fallback Rates**: Default rates when model-specific pricing unavailable
- **Currency**: All costs calculated in USD

#### **Stage-Based Costing**
Costs are tracked separately for each pipeline stage:
- **Embedding**: Vector generation costs
- **Rewrite**: Query rewrite processing
- **Inference**: Primary response generation
- **Tools Synthesis**: Tool call planning and execution

### Monitoring

The system provides real-time token usage and cost tracking through:
- **Per-Turn Metrics**: Token counts and costs for each chat turn
- **Conversation Totals**: Cumulative usage across entire conversation
- **Stage Breakdown**: Detailed breakdown by pipeline stage
- **Cost Attribution**: Clear cost attribution per model and provider

### Configuration

#### **Model Selection**
```python
# Non-Reasoning (fast, cost-effective)
inf_spec = {"model": "openai:gpt-4o-mini"}

# Reasoning (complex queries)
inf_spec = {"model": "openai:gpt-5-mini"}
```

#### **Tool Integration**
Both model types support the same tools:
- `get_weather` - Weather information
- `web_search` - Web search
- `get_nearby_airports` - Airport finder

#### **Best Practices**

1. **Use Non-Reasoning for**:
   - Simple factual questions
   - Quick lookups
   - Cost-sensitive applications
   - High-volume queries

2. **Use Reasoning for**:
   - Complex analytical tasks
   - Multi-step problems
   - Queries requiring deep reasoning
   - When accuracy is more important than speed

3. **Avoid Reasoning for**:
   - Query rewrite stage (use non-reasoning models)
   - Simple retrieval tasks
   - Real-time applications requiring sub-second responses

---

## 🤖 LLM Handler Architecture

This system features a **unified LLM handler** that provides a consistent interface for multiple AI providers through a centralized architecture.

### Core Components

#### **LLM Handler** (`backend/llm/llm_handler.py`)
- **Unified Interface**: Single entry point for all LLM calls across providers
- **Automatic Parameter Mapping**: Handles provider-specific parameter differences
- **Capability Filtering**: Automatically filters unsupported parameters per model
- **Error Handling**: Structured error responses with provider-specific context

#### **Model Registry** (`backend/llm/model_registry.py`)
- **Centralized Metadata**: All model configurations in one place
- **Provider Support**: Currently supports **OpenAI** and **Gemini** APIs
- **Extensible Design**: Easy to add new providers and models
- **Capability Definitions**: Feature support flags per model (tools, streaming, reasoning)

### Supported Providers

#### **OpenAI Integration**
- **Native API Support**: Direct integration with OpenAI's Responses API
- **Model Coverage**: GPT-4o, GPT-4o-mini, o1, o3, reasoning models
- **Full Feature Set**: Tools, streaming, temperature control, reasoning parameters

#### **Gemini Integration**
- **OpenAI-Compatible API**: Uses OpenAI adapter for Gemini models
- **Current Approach**: Standardized interface via OpenAI client library
- **Future Support**: Architecture ready for native Gemini SDK integration
- **Model Coverage**: Gemini 2.5 Flash, Gemini 2.5 Pro, embedding models

### Key Benefits

✅ **Provider Agnostic**: Same code works across OpenAI and Gemini  
✅ **Automatic Adaptation**: Parameter differences handled automatically  
✅ **Future Proof**: Easy to extend to additional providers  
✅ **Type Safety**: Structured responses and error handling  
✅ **Performance**: Optimized routing and capability caching  

### Usage Example

```python
from backend.llm.llm_handler import llm_handler

# Works across providers with same interface
response = llm_handler.create(
    provider="openai",     # or "gemini"
    model="gpt-4o-mini",   # or "models/gemini-2.5-flash-lite"
    input="Explain quantum computing",
    temperature=0.7,
    max_output_tokens=1000
)
```

### Detailed Documentation

For comprehensive documentation on the LLM handler architecture, provider-specific features, model compatibility, and extension guidelines, see:

👉 **[README_LLM_HANDLER.md](README_LLM_HANDLER.md)**

---

## 🏗️ Technical Overview

Technical details about the system architecture, pipelines, design decisions, and engineering approach are available here:

👉 **[TECHNICAL_OVERVIEW.md](TECHNICAL_OVERVIEW.md)**

This overview covers module structure, extraction pipeline, embedding flow, Qdrant indexing, batch ingestion (local PDFs + URLs with optional cost estimation), chat orchestration, SSE streaming, and frontend–backend integration.

---

## 🗂️ Project Structure

```text
chat-with-rag/
├── backend/               # Server-side application
│   ├── api/              # HTTP routes (chat, ingestion)
│   ├── chat/             # Chat orchestration, tools, SSE stages
│   ├── core/             # Settings, logging, shared schemas
│   ├── db/               # Qdrant client + vector store layer
│   ├── embeddings/       # Embedding manager + model abstraction
│   ├── extractor/        # HTML/MediaWiki/PDF extractors + splitters
│   ├── crawler/          # URL & PDF fetch utilities
│   └── utils/            # Shared helpers and admin scripts
├── frontend/             # Browser UI
│   ├── static/           # JS/CSS assets
│   ├── index.html        # Landing page
│   └── chat.html         # Chat interface
├── scripts/              # Maintenance + ingestion scripts
├── qdrant_scripts/       # Qdrant maintenance scripts
├── data/                 # Seed / demo datasets
├── images/               # Images for system use
├── logs/                 # Rotating runtime logs
└── qdrant_storage/       # Local Qdrant data volume
```



---

## � Security & Deployment

This application includes a **domain-based access control framework** that provides built-in security for API endpoints and embedded widgets.

### **Implemented Security Features**

#### **Domain-Based Access Controls**
- **API Endpoint Protection**: All `/chat` and embedding endpoints enforce domain-based access controls
- **Embeddable Widget Security**: `chat-embed.html` can only be embedded on authorized domains via `data-domain` attribute
- **Collection Isolation**: Each domain has isolated collections and embedding models to prevent data cross-contamination
- **Configuration-Based**: Security enforced through `DOMAIN_EMBEDDING_CONFIG` in `backend/core/config.py`

#### **Domain Configuration Example**
```python
# In backend/core/config.py
DOMAIN_EMBEDDING_CONFIG = {
    "default": {
        "collection_name": "document_index",
        "embedding_model_key": "openai:embed_small"
    },
    "oceans": {
        "collection_name": "document_index_gemini", 
        "embedding_model_key": "gemini:native-embed"
    }
}

# Active domain selection
active_domain: str = "oceans"  # Switches both collection and model
```

#### **Embeddable Widget Security**
```html
<!-- Only works on authorized domains -->
<div id="chat-embed" 
     data-domain="oceans" 
     data-api-url="https://your-server.com">
</div>
```

### **Additional Security Recommendations**

#### **Deployment Considerations**
When deploying beyond local/dev environments, consider these additional protections:

#### **Network Layer Security**
- **Reverse Proxy**: Use nginx/Apache with SSL termination
- **Firewall Rules**: Restrict access to API endpoints by IP/CIDR
- **VPN/Private Networks**: Deploy within private networks when possible

#### **Authentication & Authorization**
- **API Keys**: Implement API key authentication for external access
- **OAuth/JWT**: Add user authentication for multi-tenant scenarios  
- **Role-Based Access**: Different permissions for different user types

#### **Rate Limiting & Abuse Prevention**
- **Request Rate Limiting**: Per-IP and per-user rate limits
- **Request Size Limits**: Prevent large payload attacks
- **Captcha Integration**: For public-facing deployments

#### **Monitoring & Auditing**
- **Access Logs**: Log all API access with timestamps and user identifiers
- **Anomaly Detection**: Monitor for unusual usage patterns
- **Security Headers**: Implement proper CORS, CSP, and security headers

#### **Data Protection**
- **Encryption at Rest**: Encrypt database storage
- **Encryption in Transit**: Enforce HTTPS/TLS for all communications
- **Data Retention Policies**: Automatic cleanup of old conversation data

### **Security Architecture Benefits**

✅ **Domain Isolation**: Each domain operates in its own security context  
✅ **Collection Separation**: Data cannot leak between domains  
✅ **Widget Lockdown**: Embedded widgets only work on authorized domains  
✅ **Configuration-Driven**: Security enforced through config, not code changes  
✅ **Scalable**: Easy to add new domains without security reconfiguration  

This section serves as the canonical overview of auth/security for the application; feature-specific docs (such as `README_CHAT_EMBED.md`) refer back here.

---

## 📡 API Usage Example

### **Complete Chat API Call**

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main differences between Atlantic and Pacific oceans?",
    "namespace": "oceans",
    "render_html": true,
    "show_processing_steps": true,
    "stage_specs": {
      "rewrite": {"model_key": "openai:gpt-4o-mini"},
      "rerank": {"model_key": "openai:gpt-4o-mini"},
      "inference": {"model_key": "openai:gpt-4o-mini"},
      "summarization": {"model_key": "openai:gpt-4o-mini"}
    },
    "top_k": 5,
    "show_citations": true
  }' | jq '.'
```

### **Sample API Response**

```json
{
  "answer": "The Atlantic Ocean is generally warmer and saltier than the Pacific Ocean...",
  "response": "The Atlantic Ocean is generally warmer and saltier than the Pacific Ocean...",
  "answer_html": "<p>The Atlantic Ocean is generally warmer and saltier than the Pacific Ocean...</p>",
  "sources": [
    {
      "title": "Ocean Comparison Study",
      "url": "https://example.com/ocean-study",
      "snippet": "Atlantic waters average 22°C while Pacific averages 17°C...",
      "citation": "[1]"
    }
  ],
  "metrics": {
    "vectors_retrieved": 5,
    "tokens_used": 1250,
    "cost_estimate": 0.0042
  },
  "turn_metrics": {
    "rewrite_confidence": 0.85,
    "reranking_score": 0.92
  },
  "conversation_totals": {
    "total_turns": 3,
    "total_tokens": 3800
  },
  "tools_used": ["web_search"],
  "rewrite_display": {
    "original": "differences between Atlantic and Pacific",
    "rewritten": "What are the main differences between Atlantic and Pacific oceans?",
    "confidence": 0.85
  }
}
```

### **Key Response Fields**

- **`answer`**: Complete response text with citations
- **`answer_html`**: Formatted HTML response (when `render_html=true`)
- **`sources`**: Source documents with citations
- **`metrics`**: Token usage, costs, retrieval counts
- **`turn_metrics`**: Current turn performance data
- **`conversation_totals`**: Session-level statistics
- **`tools_used`**: Tools invoked during processing
- **`rewrite_display`**: Query rewrite information

---

## �� License & Usage

This project is **source-available** for **personal, educational, and evaluation purposes**.  
It is permitted to **run, modify, and fork** the code for non-commercial use.

**Redistribution, sublicensing, or commercial use** of this project or derivative works **requires explicit written permission** from the author.
© 2025 Rajkumar Velliavitil — All Rights Reserved
