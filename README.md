# Chat with Your Docs: End-to-End RAG Pipeline

![CI Status](https://github.com/vrraj/chat-with-rag/actions/workflows/python-ci.yml/badge.svg)


A modular **Retrieval-Augmented Generation (RAG) framework** for building AI applications that generate **grounded answers with citations** from unstructured documents.

The system implements an explicit, multi-stage orchestration pipeline covering **high-fidelity ingestion, retrieval, reasoning, tool execution, and response synthesis**, with **multi-LLM support (OpenAI + Gemini)**.

Unlike simple vector-search demos, this project exposes each stage of the RAG pipeline as a **configurable and observable component**, enabling experimentation with retrieval strategies, prompt design, model selection, and cost control.

All LLM interactions are handled through the standalone Python library  
**[vrraj-llm-adapter](https://pypi.org/project/vrraj-llm-adapter/)** — a registry-driven adapter that normalizes requests, responses, tool calls, and usage accounting across providers.

➡️ **Quick Start:** See [Getting Started](#-getting-started) to run the system locally.

## Why This Project Exists

Many RAG implementations start simple but quickly become difficult to manage as capabilities expand.

Typical challenges include:

- managing multiple LLM providers
- handling provider-specific API differences
- controlling prompt behavior across stages
- maintaining stable context windows
- tracking token usage and costs
- keeping ingestion pipelines reliable

This project explores a **structured approach to RAG architecture**, where each stage of the pipeline is explicit, configurable, and observable.

### 🆕 What's New in v2.0

- **Multi-Provider LLM Framework**  
Unified abstraction supporting OpenAI, Gemini, and extensible to additional providers.

- **Centralized Model Registry**  
  Single source of truth for model capabilities, pricing, API, parameter normalization and provider-specific nuances.

- **Prompt Registry (YAML-Driven)**  
  Centralized prompt control layer that decouples prompts from code. Supports per-request `prompt_domain` switching for **A/B testing** without code changes.

- **Advanced Context Window Management**  
  Hybrid strategy combining summarized conversation history with recent verbatim turns to maintain context while controlling token usage.

- **Per-Stage Model Configuration**  
  Runtime model selection per pipeline stage via UI or API.

- **Performance & Cost Controls**  
  Configurable controls for all pipeline stages with cost tracking.

- **Postprocessing LLM Response**  
  Currently Markdown to scoped HTML conversion - can be extended for additional actions to support workflows.

- **Embeddable Chat Widget**  
  Drop-in widget with comprehensive configuration via API params.

- **Domain-Based Access Controls**  
  Isolation and authorization enforced consistently across APIs and embedded clients.

**For additional details, see the [Release Notes 2.0](Release_Notes_2.0.md).**



> **Auth & Security Note**  
This app enforces **domain-based access controls** across APIs and embedded widgets (domain isolation, collection separation, widget lockdown). See **[Security & Deployment](#-security--deployment)** below for details and deployment guidance.



## 🧠 High-Level RAG Pipeline Overview

The system runs through two parallel workflows: an **Ingestion Pipeline** (build the knowledge base) and a **Chat Orchestration Pipeline** (retrieve + answer).

**Ingestion Pipeline (Data → Enriched Vectors)**
> Documents (single or batch) ⟹ Load ⟹ Extract (PDF / HTML / Wiki) ⟹ Process & normalize (chunking) ⟹ Metadata augmentation ⟹ Embeddings ⟹ Vector storage

**Chat Orchestration Pipeline (Prompt → Answer)**
> User prompt ⟹ Query rewrite ⟹ Retrieval ⟹ Rerank ⟹ Summarization ⟹ Context assembly ⟹ Inference prompt assembly ⟹ LLM inference ⟹ Tool execution (if needed) ⟹ Post-processing ⟹ Final response


### Inference Pipeline in Action 

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

### Application Workspace
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

## Documentation

Additional technical documentation is available in the `/docs` directory:

- **Technical Architecture:** `docs/technical-overview.md`
- **API Reference:** `docs/api-reference.md`
- **Data Attribution:** `docs/attributions.md`

### Table of Contents

- [High-Level RAG Pipeline Overview](#-high-level-rag-pipeline-overview)
- [Features](#-features)
  - [Advanced Chat Orchestration](#-advanced-chat-orchestration)
  - [Prompt Registry](#-prompt-registry-yaml)
- [Getting Started](#-getting-started)
- [Embeddable Chat Widget](#-embeddable-chat-widget)
- [Knowledge Base and Sample Data](#-knowledge-base-and-sample-data)
- [Batch Ingestion](#-batch-ingestion)
- [API Usage Examples](#-api-usage-examples)
- [Security & Deployment](#-security--deployment)
- [License & Usage](#-license--usage)

---
## ✨ Features

An end-to-end modular RAG system that transforms raw documents into grounded answers using a configurable multi-stage pipeline with high-fidelity ingestion and live observability.

### 📥 High-Fidelity Ingestion
* **Multi-Source Extraction**: Native support for high-fidelity parsing of **PDFs**, **MediaWiki**, and **HTML**.
* **Intelligent Processing**: 
    * **Smart Chunking**: Preserves semantic context across chunks.
    * **Structure Preservation**: Maintains the integrity of tables and structured layouts.
    * **Noise Filtering**: Configurable noise filtering to remove headers, footers, and selected sections for cleaner context.
> **Batch & Scale**: Process local directories (`file://`) or remote URLs with built-in **token and cost estimation** before committing to storage.


### 🧠 Advanced Chat Orchestration

Advanced Chat Orchestration is the system’s control plane for intelligent retrieval and response generation. It coordinates pipeline execution, prompt and model selection, context and memory handling, retrieval and tool augmentation, observability, and output rendering into a single deterministic, observable flow—making complex multi-stage LLM behavior explicit, configurable, and testable.

#### 🔹 1. Pipeline Control & Execution Flow
*Defines how models, prompts, providers, tools, and post-processing stages are orchestrated for each request.*

- **Multi-Stage LLM Pipeline Orchestration**  
  Granular control over each stage: Query Rewrite → Retrieval → Rerank → Summarization → Inference → Tools → Post-processing.

- **Stage-Specific Model Configuration**  
  Independent model selection per stage, configurable at runtime via UI or API.

- **Dynamic Model Selection (UI)**  
  Per-stage model selection mid-conversation, including mixing providers (OpenAI + Gemini) for  capabilities and cost-control.

- **API-Level Control**  
  Pipeline configuration is available programmatically via FastAPI endpoints for automation, integrations, and workflows.

  - **Multiple LLM Provider Support**  
  A unified **LLM Adapter** framework that abstracts multiple provider surfaces behind a single contract. Currently supports
  - **OpenAI** (Chat Completions API, Responses API)  
  - **Gemini** (OpenAI-compatible Adapter, native Gemini SDK)  

  >**Provider-specific differences are normalized so the pipeline remains provider-agnostic.**

- **Model Registry Centralization**  
  Models are selected via stable **registry keys** (e.g. `openai:fast`, `gemini:embed`) that abstract provider and model differences. The registry is the single source of truth for **capabilities, pricing, parameter semantics, and normalization**, enabling per-stage provider mixing, consistent routing, validation, and cost tracking without application changes.

- **Prompt Registry (YAML-Driven Control)**  
  A centralized prompt control layer that decouples prompts from code, supports default system and user prompts, with  domain-specific augmentation. Injects application-derived context dynamically at runtime. 
  >A `prompt_domain` parameter can be passed per request to alter pipeline behavior without code changes, enabling parallel chat sessions and **A/B testing** of prompt strategies.



#### 🧠 2. Context & Memory Management
*Maintains long-running conversational continuity while keeping context size bounded and cache-efficient.*

Long-running conversations remain coherent and performant without exceeding context limits by combining a persistent conversation summary with a short, verbatim recent history. As the conversation grows, older turns are automatically **summarized and merged into the active context**, preserving continuity while maintaining stable context size and cache efficiency.


#### ✏️ 3. Query Intelligence & Rewrite
Improves retrieval accuracy by selectively refining user intent before search. Rewrites are confidence-gated, context-aware (verbatim turns or summaries), and fully configurable or disable-able per request.


#### 🔍 4. Retrieval, Inference & Tool Augmentation
*Synthesizes grounded answers by combining retrieved knowledge, tools, and model inference.*

- **Retrieval Optimization**
  - Vector search via Qdrant with configurable Top-K and score thresholds

- **Inference Context Assembly**  
  Final LLM prompts are composed from:
  - System and user instructions (prompt registry)
  - Domain-specific prompt augmentations
  - Conversation summary and verbatim tail
  - Retrieved and reranked document chunks
  - Optional web search context
  - User query

- **Tool Execution**
  - Native function/tool calling (currently get_weather and get_airports)
  - Tool outputs are merged into the final synthesis stage


  > Web search is supported via an optional automatic web context stage and via an LLM tool call.

- **Verified Citations**
  - All final answers include citations (URL and source document section / sub-section reference)


#### 📊 5. Observability & Cost Management
*Provides real-time visibility into pipeline execution, token usage, and per-stage costs.*

- **Real-Time Observability**
Live **SSE (Server-Sent Events)** stream providing a window into the "thoughts" and progress of the RAG flow as it happens

- **Granular Cost Tracking**
Instant transparency with per-stage token usage and dollar-cost metrics for every turn

#### 🎨 6. Postprocessing & Output Rendering
*Transforms raw LLM output into presentation-ready responses without affecting core inference behavior.*

After inference, responses can be post-processed to deliver a richer, presentation-ready experience in the chat UI.

> This post-processing stage is isolated from inference, ensuring output formatting can evolve independently without affecting core model behavior. In addition post processing can be extended to support custom workflows and transformations.

### 🌐 Embeddable Chat Widget
**Embeddable chat** for any website with full access to pipeline configuration controls.

See the **[Embeddable Widget Configuration](#-embeddable-chat-widget)** section below for detailed implementation examples and all available options.




---

## 🚀 Getting Started

Get the system running in minutes using the provided `Makefile`. This setup uses Docker for the core infrastructure while maintaining a developer-friendly local environment through volume mounting.

> **LLMProvider Note:** The system supports multiple LLM providers, including **OpenAI** and **Gemini**, and is designed to be extensible to additional providers. Providers can be switched or mixed **per pipeline stage** after setup. The tested models are defined centrally in the **Model Registry**


### 📋 1. Prerequisites
Ensure your environment meets these requirements before proceeding:
- **OS:** macOS or Linux (Windows supported via Docker).
- **Git** – required to clone the repository. Install: https://git-scm.com/downloads
- **Docker & Docker Compose:** Required for the Qdrant v1.14.1 database and the web app container. [Get Docker here](https://docs.docker.com/get-started/)
- **Python 3.10+:** Required for local development, IDE support, and ingestion scripts.
- **LLM Provider API Keys:** Required for embeddings and chat pipeline. Default setup uses OpenAI. [Get one here](https://platform.openai.com/api-keys)



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

> **Note**: You can use **either** provider or mix **the two** for different pipeline stages.

The following models are available for use in the model registry:
**OpenAI**: gpt-4o, gpt-4o-mini, o1-mini, o3-mini, text-embedding-3-small/large
**Gemini**: gemini-2.5-flash-lite, gemini-2.5-pro, gemini-embedding-001

#### 2.1.4 Configure API Keys & Costs

Set up your LLM Provider, OpenAI and / or Gemini (API Keys and Budget Limits).
>The setup instructions in this section uses OpenAI for reference. You may follow the same steps for Gemini. 

##### Option A: OpenAI (Getting Started)

| Recommendation | Action | Rationale |
| :--- | :--- | :--- |
| **Budget** | Set a limit of **$5–$10**. | Establishes a safety ceiling for testing. |
| **Dedicated Key** | Name it `chat-with-rag`. | Isolates usage tracking for this specific project. |
| **Alerts** | Set a 50% notification. | Provides proactive cost control. |

##### Option B: Gemini 


| Recommendation | Action | Rationale |
| :--- | :--- | :--- |
| **Quota** | Set a **daily quota limit** based on your budget. | Prevents unexpected cost overruns. |
| **Dedicated Key** | Name it `chat-with-rag-gemini`. | Isolates usage tracking for this project. |
| **Monitoring** | Enable **usage alerts** in Google Cloud Console. | Provides proactive cost visibility. |

> **Note:** Gemini uses quota-based limits instead of hard dollar limits. Configure quotas in Google AI Studio or Google Cloud Console.

##### Option C: Both Providers (Advanced)
Use different providers for different pipeline stages via UI or API
> Sample data collection uses OpenAI embeddings (text-embedding-3-small) with 1536 dimensions

### 2.1.6 LLM Providers, Models, and Endpoints (Overview)

The system uses a centralized **Model Registry** to define and manage all supported LLM providers, models, and API surfaces. The registry serves as the **single source of truth** for provider routing, model capabilities, and cost tracking, and is consumed uniformly by the application and LLM handler.

Currently the system supports these LLM *endpoints** :

- **Open AI:** 
  - chat-completions
  - responses API
  - embeddings

- **Gemini:**
  - chat-completions (OpenAI compatible adapter)
  - gemini-sdk
  - embeddings



> Detailed provider behavior, endpoint mappings, and capability flags are documented in **[docs/technical-overview.md](docs/technical-overview.md)** and the model registry itself.

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
- **[docs/technical-overview.md](docs/technical-overview.md)**

For details on the stateless chat API (`POST /chat`) used by `frontend/chat.html`, including request/response shape and parameter contract, see:

👉 **[docs/api-reference.md](docs/api-reference.md)**

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

## 🌐 Embeddable Chat Widget

A lightweight, secure widget that embeds the **full RAG pipeline** into any website.

It exposes the same multi-stage orchestration used by the main app—**retrieval, reranking, context management, tool calling, and post-processing**—while staying easy to deploy and easy to tune.
> Supports application domain isolation for prompt selection.

### **How you configure it**
You can configure the widget via **URL query parameters** (direct iframe) or **HTML `data-*` attributes** (embed loader). Supports full parameter API contract and includes `top_k`, `score_threshold`, 'inference_model', `max_output_tokens`, `show_processing_steps`, `show_sources` etc

Screenshot of Inline embed vs iFrame option:

<p align="center">
  <a href="images/chat-embedding-options.png">
    <img
      src="images/chat-embedding-options.png"
      style="max-width: 100%; height: auto;"
      alt="Chat embedding options iframe and inline page"
    />
  </a>
</p>

*Embeddable chat widget options (inline page or iFrame ).*

### **Sample Configurations**

#### **Simple Chat Widget (Direct iframe)**
```html
<iframe 
  src="https://your-server.com/chat-embed.html?top_k=5&show_citations=true&namespace=simple-chat"
  width="100%" 
  height="400px"
  style="border: 0; border-radius: 8px;"
  title="Embedded Chat">
</iframe>
```

#### **Advanced Configuration (Embed Loader Script)**
```html
<!-- 1. Add target container -->
<div id="chat-embed" 
     data-api-url="https://your-server.com"
     data-model_key="openai:gpt-4o-mini"
     data-temperature="0.7"
     data-top_k="10"
     data-show_processing_steps="true"
     data-show_citations="true"
     data-namespace="oceans">
</div>

<!-- 2. Add embed loader script (REQUIRED!) -->
<script src="https://your-server.com/static/embed-loader.js"
        data-target="#chat-embed"
        data-api-url="https://your-server.com"
        data-model_key="openai:gpt-4o-mini"
        data-temperature="0.7"
        data-top_k="10"
        data-show_processing_steps="true"
        data-show_citations="true"
        data-namespace="oceans">
</script>
```

---

## 🌐 Web Search 

The system supports web search (DuckDuckGo Instant Answer API) in two distinct modes, **Automatic Web Context** and **LLM Tool Call**:

1. **Automatic Web Context**
   - Enabled in configuration with `use_web_search`
   - Behavior:
     - Runs as part of the chat pipeline stage **Establish Web Context**. and adds <web_search>web search results</web_search> to the inference prompt.
     
2. **LLM Tool Call (`web_search` tool)**
   - Enabled when tools are enabled and the model chooses to call the tool.
   - Returns as Tool Call output to the synthesis stage as `[SOURCE: TOOL - web_search] ...`.


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

> **Note:** Sample data is created in the `document_index` collection in qDrant and uses the `openai:embed_small` embedding key (text-embedding-3-small with 1536 dimensions).

### 📄 Data Attribution
To demonstrate multi-source RAG capabilities, this project includes a sample knowledge base derived from Wikipedia.
* **Source:** 55 curated Wikipedia articles processed via a custom high-fidelity MediaWiki extraction pipeline.
* **Integrity:** Source URLs and author metadata are preserved within the vector payloads to enable **verified citations**.
* **License:** Distributed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
* **Full Credits:** Detailed source links and compliance information can be found in [docs/attributions.md](docs/attributions.md).

### 🔍 Explore the Data

You can verify the indexed documents through the web interface or the command line:

| Method | Action |
| :--- | :--- |
| **Frontend UI** | Navigate to the **"View Documents"** page to see titles, URLs, and metadata. |
| **Terminal (CLI)** | Run the following to list the first 100 document titles: |

```bash
source venv/bin/activate
python scripts/qdrant_scripts/qdrant_ops.py --list-titles --limit 100

```

### 🔄 Managing Your Collections

The system supports **domain-based collection management** where each domain is tightly coupled with its embedding model to prevent dimension drift and ensure consistency. Selecting a collection automatically sets the compatible embedding model and vector dimensions.

#### **Domain-Based Configuration (Recommended)**

A single `active_domain` setting configures both the collection name and embedding model. This helps prevent dimension drift and ensures consistency.
The system comes with a default configuration for the `default` domain and two additional domains: `mountains` and `oceans`. You may modify /add to the configuration in `backend/core/config.py`.
> *Only one domain can be active at a time, and that defines the Qdrant collection and embedding model.*

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
active_domain: str = "mountains"
```

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
python scripts/qdrant_scripts/qdrant_ops.py --delete-collection $(python -c "from backend.core.config import settings; print(settings.collection_name)")
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

> **Note:** Changing the embedding model requires re-embedding and rebuilding the vector index. See **[docs/technical-overview.md](docs/technical-overview.md)** for the recommended re-ingestion workflow.

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
  - Uses `EmbeddingsManager.index_chunks`, which groups chunks into batches and calls `llm_client.embed` once per batch.
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

## 🧠 Reasoning Models Overview

The system supports both reasoning and non-reasoning models with provider-specific behaviors.

1. **Reasoning Control**
The `reasoning_effort` parameter from the inference stage controls reasoning level. These are mapped to provider-specific parameters OpenAI (reasoning.effort) and Gemini (thinking_level/thinking_budget) :

2. **Provider Differences**
- **OpenAI**: Reasoning tokens are **hidden** from user display
- **Gemini**: Reasoning shown as `<thought>` tags and **displayed** in frontend. Max completion tokens includes reasoning token and requires padding of "max_inference_token parameter" to account for reasoning tokens. THe padding is calculated from the configurations in the model registry.

3. **System Resolution**
- **LLM Adapter** (`llm-adapter` package): Defines capabilities, parameters, and tool handling
- **LLM Client** (`llm_client.py`): Clean interface to llm-adapter package
- **Frontend** (`chat.html`): Displays `<thought>` tags for Gemini, hides OpenAI reasoning tokens

---

## 📊 Metrics and Costs

1. **Token Accounting**
The system tracks and costs tokens based on provider/model usage reporting. Current token accounting includes:
  - **Prompt Tokens** (Input): Tokens sent to the model (user message + context)
- **Cached Tokens**: Cached prompt tokens (lower cost)
- **Completion Tokens** (Output): Tokens in the model's response to the user
- **Reasoning Tokens**: provided by OpenAI and calculated for Gemini.


2 **Rate Sources**
- **Model Registry** (`llm-adapter` package): Provider-specific pricing per model
- **Currency**: All costs calculated in USD

3 **Stage-Based Costing**
Costs are tracked separately for each pipeline stage:
- **Embedding**: Vector generation costs
- **Rewrite**: Query rewrite processing
- **ReRanking**: Document re-ranking
- **Summary**: Context Window summary processing
- **Inference**: Primary response generation
- **Inference with Tools**: Final response if tools are used

---

## 🤖 LLM Integration

This system features a **unified LLM client** that provides a consistent interface for multiple AI providers and models through the llm-adapter package.

### Core Components

1. **LLM Client** (`backend/llm/llm_client.py`)
- **Unified Interface**: Single entry point for all LLM calls across providers
- **Automatic Parameter Mapping**: Handles provider-specific parameter differences
- **Capability Filtering**: Automatically filters unsupported parameters per model
- **Error Handling**: Structured error responses with provider-specific context

2. **LLM Adapter** (`llm-adapter` package)
- **Centralized Metadata**: All model configurations in one place
- **Provider Support**: Currently supports **OpenAI** and **Gemini** APIs
- **Extensible Design**: Easy to add new providers and models
- **Capability Definitions**: Feature support flags per model (tools, streaming, reasoning)
- **Tool Sanitization**: Automatic tool format conversion for all providers

### Tested Providers and Models

- **OpenAI**: Responses API and Chat Completions API
  - **Model Coverage**: 
    -  **Pipeline stages**: `gpt-4o-mini`, `gpt-4o` `o3-mini`, `gpt-5-mini`
    - **Embeddings**: `text-embedding-3-small`, `text-embedding-3-large`

- **Gemini**: OpenAI-Compatible API and Gemini SDK
  - **Model Coverage**:
    - **Pipeline stages**: `gemini-2.5-flash-lite`, `gemini-3-flash-preview`, `gemini-2.5-flash`, `gemini-3-flash`
    - **Embeddings**: `gemini-embedding-001`

### Key Benefits

✅ **Provider Agnostic**: Same code works across OpenAI and Gemini  
✅ **Parameter Adaptation**: Parameter differences handled automatically  
✅ **Custom Registry Support**: Extend with your own models and configurations  
✅ **Cost Tracking**: Built-in pricing metadata for all models  

### Custom Model Registry

The chat-with-rag application supports custom model registries that extend or override the default `llm-adapter` registry. This allows you to add custom models, override pricing, and configure provider-specific parameters.

### How It Works

The system uses the `llm-adapter` package's built-in registry merging. When you provide a custom registry:

```python
# LLMAdapter automatically merges: defaults + custom_registry
merged_registry = {**dict(defaults), **dict(custom_registry)}
```

- **Default models**: All standard `llm-adapter` models remain available
- **Custom models**: Your models are added to the registry
- **Overrides**: Custom models can override default models by using the same key
- **Smart routing**: Chat pipeline automatically uses the appropriate adapter

### Implementation

1. **Create custom registry** at `examples/custom_registry.py`:

```python
from llm_adapter.model_registry import ModelInfo, Pricing

REGISTRY = {
    # Override existing model with custom pricing
    "openai:gpt-4o": ModelInfo(
        provider="openai",
        model="gpt-4o",
        endpoint="chat_completions",  # or "responses"
        pricing=Pricing(
            input_per_mm=0.005,  # Custom pricing
            output_per_mm=0.015,
            cached_input_per_mm=0.0025
        ),
        capabilities={"reasoning": True, "tools": True},
        param_policy={
            "allowed": {"max_output_tokens", "temperature", "top_p", "tools", "tool_choice"},
            "disabled": set()
        }
    ),
    
    # Add completely new custom model
    "custom:experimental": ModelInfo(
        provider="openai",
        model="gpt-4o-mini",
        endpoint="chat_completions",
        pricing=Pricing(
            input_per_mm=0.01,
            output_per_mm=0.03
        ),
        capabilities={
            "experimental": True,
            "max_tokens": 4096
        },
        param_policy={
            "allowed": {"max_output_tokens", "temperature", "top_p"},
            "disabled": {"tools"}  # Disable tools for this model
        }
    ),
    
    # Add Gemini model with custom reasoning
    "gemini:custom-reasoning": ModelInfo(
        provider="gemini",
        model="models/gemini-3-flash-preview",
        endpoint="gemini_sdk",
        pricing=Pricing(
            input_per_mm=0.001,
            output_per_mm=0.004,
            cached_input_per_mm=0.0005
        ),
        reasoning_policy={
            "mode": "gemini_level",
            "effort_map": {
                "minimal": {"thinking_level": 1},
                "low": {"thinking_level": 2},
                "medium": {"thinking_level": 3},
                "high": {"thinking_level": 4}
            }
        }
    ),
    # Add more custom models...
}
```

2. **Configure environment** (optional):
```bash
export CUSTOM_REGISTRY_PATH=/path/to/your/custom_registry.py
```

3. **Use in UI** - The "Change Models" dialog will automatically include your custom models.

4. **API Integration** - Custom models are available via `/api/models?merge_custom_registry=true`

5. **Hot Reload** - Changes to `custom_registry.py` are automatically picked up. No server restart needed.

**Manual Reload URL**: 
```
GET http://localhost:8000/api/models?merge_custom_registry=true
```

**Features**:
- Override existing model configurations
- Add new provider/model combinations  
- Custom pricing and capability flags
- Provider-specific parameter policies
- Automatic UI integration
- **Hot reload** for instant updates  
✅ **Future Proof**: Easy to extend to additional providers  
✅ **Type Safety**: Structured responses and error handling  
✅ **Performance**: Optimized routing and capability caching  

### Technical Architecture

#### Registry Merging Process

The custom registry system leverages the `llm-adapter` package's built-in merging capabilities:

```python
# In LLMAdapter.__init__ (llm_adapter/llm_adapter.py:184-185)
defaults = getattr(_model_registry, "REGISTRY", {})
self.model_registry = {**dict(defaults), **dict(model_registry)}
```

**Merge Semantics**:
- **Base registry**: All default `llm-adapter` models (14+ models)
- **Custom registry**: Your `examples/custom_registry.py` models
- **Result**: Combined registry with custom models overriding defaults on key collision

#### Adapter Selection Logic

The chat pipeline uses smart adapter selection in `backend/llm/llm_client.py`:

```python
def _get_adapter_for_model(model_key: str) -> LLMAdapter:
    # Try custom adapter first (with merged registry)
    try:
        custom_adapter = _get_adapter(merge_custom_registry=True)
        if custom_adapter._lookup_model_info_from_registry(model_key) is not None:
            return custom_adapter  # Custom model found
    except Exception:
        pass
    
    # Fall back to default adapter
    return llm_adapter  # Default model
```

**Decision Flow**:
1. **Model lookup** in custom registry → Use custom adapter
2. **Model not found** in custom registry → Use default adapter
3. **Custom registry fails** → Use default adapter (fallback)

#### Hot Reload Mechanism

Hot reload is implemented in `backend/api/endpoints/model_keys.py`:

```python
# Module reload on each API call
registry_module = __import__(module_name)
import importlib
importlib.reload(registry_module)  # Re-executes custom_registry.py
USER_REGISTRY = getattr(registry_module, 'REGISTRY')
custom_adapter = LLMAdapter(model_registry=USER_REGISTRY)
```

**Reload Process**:
1. **API call** to `/api/models?merge_custom_registry=true`
2. **Module reload**: `importlib.reload()` re-executes `custom_registry.py`
3. **Registry rebuild**: New `LLMAdapter` with latest custom models
4. **Cache update**: Fresh registry data returned to client

#### Provider Resolution

The LLM adapter resolves providers using the lookup hierarchy:

```python
# In LLMAdapter.create() (llm_adapter/llm_adapter.py:1916-1923)
if not provider:
    try:
        mi = self._lookup_model_info_from_registry(model)
        inferred = getattr(mi, "provider", None) if mi is not None else None
        if inferred:
            provider = str(inferred).strip().lower()
    except Exception:
        provider = ""  # Empty string triggers "Provider '' not supported" error
```

**Resolution Order**:
1. **Explicit provider** passed to `create()` method
2. **Registry lookup** via `_lookup_model_info_from_registry()`
3. **Provider extraction** from `ModelInfo.provider` attribute
4. **Fallback** to empty string (error case)

#### Endpoint Routing

Different endpoints trigger different API calls:

```python
# OpenAI endpoints in LLMAdapter._openai_call()
if endpoint == self.ENDPOINT_RESPONSES:
    # New OpenAI API format
    resp = client.responses.create(...)
elif endpoint == self.ENDPOINT_CHAT_COMPLETIONS:
    # Legacy OpenAI API format  
    resp = client.chat.completions.create(...)
```

**Supported Endpoints**:
- **`responses`**: New OpenAI API (recommended)
- **`chat_completions`**: Legacy OpenAI API (widely supported)
- **`gemini_sdk`**: Gemini native SDK
- **`embeddings`**: OpenAI embedding models
- **`embed_content`**: Gemini embedding models

#### Error Handling

The system implements multi-level error handling:

1. **Registry Loading**: Falls back to default adapter if custom registry fails
2. **Model Lookup**: Falls back to default adapter if model not found in custom registry
3. **Provider Resolution**: Raises `LLMError` if provider cannot be determined
4. **API Calls**: Provider-specific error handling with detailed error messages

#### Performance Considerations

- **Registry Size**: Merged registry typically contains 17+ models (14 default + 3+ custom)
- **Lookup Performance**: O(1) dictionary lookup for model resolution
- **Hot Reload Overhead**: ~1-2ms per API call (negligible compared to LLM calls)
- **Memory Usage**: Each adapter instance maintains its own registry copy

#### Configuration Options

**Environment Variables**:
```bash
# Custom registry path (optional)
CUSTOM_REGISTRY_PATH=/path/to/custom_registry.py

# Model allowlist (optional)
LLM_ADAPTER_ALLOWED_MODELS=openai:gpt-4o,custom:experimental
```

**Registry Structure**:
```python
REGISTRY = {
    "provider:model_key": ModelInfo(
        provider="openai|gemini",           # Required
        model="provider-model-name",         # Required  
        endpoint="responses|chat_completions|gemini_sdk|embeddings|embed_content",
        pricing=Pricing(...),                # Optional
        capabilities={...},                  # Optional
        param_policy={...},                  # Optional
        reasoning_policy={...},              # Optional
        limits={...},                        # Optional
    )
}
```

```python
from backend.llm.llm_client import generate

# Works across providers with same interface
response = generate(
    model_key="openai:gpt-4o-mini",     # or "gemini:openai-2.5-flash-lite"
    input="Explain quantum computing",
    temperature=0.7,
    max_output_tokens=1000
)
```

---

## 🏗️ Technical Overview

Technical details about the system architecture, pipelines, design decisions, and engineering approach are available here:

👉 **[docs/technical-overview.md](docs/technical-overview.md)**

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
│   ├── llm/              # LLM handler and model registry
│   ├── tools/            # Tool implementations (weather, web search)
│   ├── crawler/          # URL & PDF fetch utilities
│   └── utils/            # Shared helpers and admin scripts
├── frontend/             # Browser UI
│   ├── static/           # JS/CSS assets (embed-loader.js, chat-embed.js)
│   ├── index.html        # Landing page
│   ├── chat.html         # Chat interface
│   ├── chat-embed.html   # Embeddable chat widget
│   └── chat-embed-example.html  # Integration examples
├── scripts/              # Maintenance + ingestion scripts
│   ├── qdrant_scripts/   # Qdrant maintenance scripts
│   └── batch/            # Batch processing scripts
├── prompts/              # Prompt registry (YAML-driven control)
├── data/                 # Seed / demo datasets
├── images/               # Images for documentation
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
