# Chat with Your Docs: End-to-End RAG Pipeline

![CI Status](https://github.com/vrraj/chat-with-rag/actions/workflows/python-ci.yml/badge.svg)

A modular RAG framework that transforms unstructured data into **actionable intelligence** through sophisticated retrieval-reranking pipelines, real-time observability, and tool-augmented reasoning.

This system goes beyond basic vector search by implementing a multi-stage LLM orchestration layer. It ingests complex formats (MediaWiki, PDFs, HTML), preserves document structure, and provides a fully verifiable chat experience with live-streamed **pipeline execution stages** and direct **source citations**.

##  Table of Contents

- [High-Level RAG Pipeline Overview](#-high-level-rag-pipeline-overview)
- [Features](#-features)
- [Getting Started with Chat with RAG](#getting-started-with-chat-with-rag)
- [Knowledge Base and Sample Data](#knowledge-base-and-sample-data)
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
| 2. **Extraction** (PDF, HTML, Wiki) | 2. **Query Rewrite** (Optimization) |
| 3. **Processing & Normalization** | 3. **Document Retrieval** (Qdrant Search) |
| 4. **Metadata Augmentation** | 4. **Relevance Reranking** |
| 5. **Embedding Generation** (OpenAI) | 5. **Context Construction** (History + reranked chunks) |
| 6. **Vector Storage** (Qdrant) | 6. **LLM Inference** (GPT-4o-mini) |
| | 7. **Tool Execution** (e.g., Weather, Maps) |
| | 8. **Final Response** (with Citations) |


## ✨ Features

An end-to-end modular RAG ecosystem that orchestrates advanced LLM workflows to synthesize raw documents into structured intelligence, featuring a high-fidelity ingestion engine and live observability for verifiable, context-grounded insights.

### 📥 High-Fidelity Ingestion
* **Multi-Source Extraction**: Native support for high-fidelity parsing of **PDFs**, **MediaWiki**, and **HTML**.
* **Intelligent Processing**: 
    * **Smart Chunking**: Configurable strategies to preserve semantic context across fragments.
    * **Structure Preservation**: Maintains the integrity of complex tables and structured layouts.
    * **Noise Filtering**: Automated removal of headers, footers, and irrelevant boilerplate for cleaner context.
* **Batch & Scale**: Process local directories (`file://`) or remote URLs with built-in **token and cost estimation** before committing to storage.

### 🧠 Advanced Chat Orchestration
* **Multi-Stage LLM Pipeline**: Granular control with independent model configuration for every stage: *Query Rewrite, Reranking, Summarization, and Final Inference.*
* **Dynamic Context Control**: Fine-tune conversation history using a hybrid approach of **raw tail-turns** and **summary turns** to perfectly balance memory depth and token efficiency.
* **Retrieval Optimization**:
    * **Vector Search**: Powered by **Qdrant** with configurable Top-K and distance thresholds.
    * **Semantic Reranking**: Secondary relevance scoring applied to retrieved candidates to eliminate "hallucination noise."
    * **Query Rewriting**: Intelligent expansion of user prompts with confidence-based filtering for better search hits.
* **Verified Citations**: Final answers include direct deep-linked citations across multiple source documents.

### 🛠️ Developer & Ops Experience
* **Real-Time Observability**: Live **SSE (Server-Sent Events)** stream providing a window into the "thoughts" and progress of the RAG flow as it happens.
* **Granular Cost Tracking**: Instant transparency with per-stage token usage and dollar-cost metrics for every request.
* **Extensible Tooling**: Built-in support for function calling (e.g., weather, local APIs) to augment responses with live, real-time data.


## 🚀 Getting Started with Chat with RAG

Get the system running in minutes using the provided `Makefile`. This setup uses Docker for the core infrastructure while maintaining a developer-friendly local environment through volume mounting.

### 📋 1. Prerequisites
Ensure your environment meets these requirements before proceeding:
* **OS:** macOS or Linux (Windows supported via Docker).
* **Docker & Docker Compose:** Required for the Qdrant v1.14.1 database and the web app container. [Get Docker here](https://docs.docker.com/get-started/)
* **Python 3.10+:** Required for local development, IDE support, and ingestion scripts.
* **OpenAI API Key:** Required for embeddings and chat pipeline. [Get one here](https://platform.openai.com/api-keys)


### 2. Setup and Launch Application

**2.1) Verify Docker Installation**
```bash
   # Verify Docker Installation
   docker --version
   docker-compose --version

   ```
 You should see version numbers if Docker is installed correctly.

> **Note for Linux Users:** If you get "permission denied," add your user to the docker group: `sudo usermod -aG docker $USER` and then log out/in.

**2.2) Clone the Repository**
``` bash
git clone https://github.com/vrraj/chat-with-rag.git
cd chat-with-rag

```

**2.3) Configure OpenAI API & Costs**

> [!IMPORTANT]
> This application requires an **OpenAI API Platform** account (different from a ChatGPT Plus subscription). It is strongly recommended to set a **hard usage limit** in your [OpenAI Dashboard](https://platform.openai.com/api-keys) to stay within your desired budget.

| Recommendation | Action | Rationale |
| :--- | :--- | :--- |
| **Budget** | Set a limit of **$5–$10**. | Establishes a safety ceiling for testing. |
| **Dedicated Key** | Name it `chat-with-rag`. | Isolates usage tracking for this specific project. |
| **Alerts** | Set a 50% notification. | Provides proactive cost control. |


**2.4) Set up local environment variables**

Copy the example environment file and add your API key.

> **Note:** Optional: Advanced users may instead set `OPENAI_API_KEY` as an OS environment variable.  
> If set, it will take precedence over the value in `.env`.

```bash
cp .env.example .env
# IMPORTANT: Open .env and add your OPENAI_API_KEY
vi .env   # or use 'nano .env' / your preferred text editor

```

### 🚀 3. Launch and Populate Seed Data

**3.1) Start Infrastructure** This launches the Qdrant vector database and the FastAPI web application.

```bash
make start

```
**Note for macOS Users:**
 `will automatically attempt to launch Docker Desktop if it isn't running. The script will pause briefly while the daemon initializes.


**3.2) Initialize environment and seed data**

To see the RAG system in action immediately, load the sample dataset (~50 outdoor-themed Wikipedia pages). This requires a local Python environment.

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate
# Install dependencies and seed Qdrant
pip install -r requirements.txt 
make seed

```

**3.3) Access the interface**: Once the seeding is complete, open your browser and start chatting: 👉 http://localhost:8000

### 🧪 4. Developer Mode (Optional)

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

The default collection is named `document_index` (defined in `backend/core/config.py`). If you want to move beyond the sample data, choose one of the following paths:

#### Option A: Create a Fresh Collection (Recommended)
This is the cleanest way to experiment with your own data (PDFs, URLs, etc.) without losing the original seed data.

1.  **Open `backend/core/config.py`**.
2.  **Update the `collection_name` variable**:
    ```python
    collection_name = "my_custom_knowledge_base"
    ```
3.  **Restart the app**. The system will automatically detect the missing collection and create a fresh, empty one in Qdrant.

> [!TIP]
> This approach allows you to maintain multiple "knowledge bases" on the same server. You can swap back to the seed data at any time just by changing this variable back to `document_index`.

#### Option B: Delete and Purge (Destructive)
Use this if you want to completely clear the sample data but keep using the `document_index` name for your own knowledge base.

> [!WARNING]
> This action will permanently delete the collection and all vectors within it. This cannot be undone.

1.  **Activate your environment**:
    ```bash
    source venv/bin/activate
    
    ```
2.  **Run the deletion script**:
    ```bash
    python qdrant_scripts/qdrant_ops.py --delete-collection document_index

    ```
3.  **Verify or Re-seed**: 
    If you visit the UI now, the collection will be gone. You can either start fresh by uploading your own files via the interface or run `make seed` to repopulate it from scratch.


### 💬 Example Queries
The following examples are based on the seed data.

1. #### Retrieval-based
	•	“Where is Mount Kilimanjaro located?”

	•	“Compare Kilimanjaro with Mount Whitney.”

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
<img src="images/multi-turn-conversation-with-tools.png" width="95%" />
</p>

---

## 📦 Batch Ingestion

> **Note:** Changing the embedding model requires re-embedding and rebuilding the vector index. See **TECHNICAL_OVERVIEW.md** for the recommended re-ingestion workflow.

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


## 🏗️ Technical Overview

Technical details about the system architecture, pipelines, design decisions, and engineering approach are available here:

👉 **[TECHNICAL_OVERVIEW.md](TECHNICAL_OVERVIEW.md)**

This overview covers module structure, extraction pipeline, embedding flow, Qdrant indexing, batch ingestion (local PDFs + URLs with optional cost estimation), chat orchestration, SSE streaming, and frontend–backend integration.

## 🗂️ Project Structure



```text
backend/                     # Server-side application
├── api/                    # HTTP routes (chat, ingestion)
├── chat/                   # Chat orchestration, tools, SSE stages
├── core/                   # Settings, logging, shared schemas
├── db/                     # Qdrant client + vector store layer
├── embeddings/             # Embedding manager + model abstraction
├── extractor/              # HTML / MediaWiki / PDF extractors + splitters
├── crawler/                # URL & PDF fetch utilities
└── utils/                  # Shared helpers and admin scripts

frontend/                    # Browser UI
├── static/                 # JS/CSS assets
├── index.html              # Landing page
└── chat.html               # Chat interface

scripts/                     # Maintenance + ingestion scripts
qdrant_scripts/              # Qdrant maintenance scripts
data/                        # Seed / demo datasets
images/                      # Images for system use
logs/                        # Rotating runtime logs
qdrant_storage/              # Local Qdrant data volume
```


© 2025 Rajkumar Velliavitil — All Rights Reserved

## 📜 License & Usage

This project is **source-available** for **personal, educational, and evaluation purposes**.  
It is permitted to **run, modify, and fork** the code for non-commercial use.

**Redistribution, sublicensing, or commercial use** of this project or derivative works **requires explicit written permission** from the author.
