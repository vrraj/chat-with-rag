# Chat with Your Docs: End-to-End RAG Pipeline

![CI Status](https://github.com/vrraj/chat-with-rag/actions/workflows/python-ci.yml/badge.svg)


A full end‑to‑end RAG system that ingests MediaWiki pages, HTML and PDFs, converts them into searchable embeddings, and lets you chat with your knowledge base using real‑time retrieval. The platform also supports tool integration and extensible API‑driven connectors, enabling live data augmentation from external systems or custom enterprise workflows.

##  Table of Contents

- [High-Level RAG Pipeline Overview](#-high-level-rag-pipeline-overview)
- [Features](#-features)
- [System Requirements](#system-requirements)
- [Quick Start – Setup & Run](#quick-start---setup--run)
- [Sample (Seed) Data Overview](#sample-seed-data-overview)
- [Example Queries](#example-queries)
- [Batch Ingestion](#batch-ingestion)
- [Technical Overview](#technical-overview)
- [Project Structure](#project-structure)
- [License & Usage](#license--usage)


## 🧠 High-Level RAG Pipeline Overview

**Ingestion Pipeline**
```
Documents (single or batch)
→ Extraction
→ Processing & Normalization
→ Metadata Augmentation
→ Embedding Generation
→ Vector Storage (Qdrant)
```

**Chat & Query Pipeline**
```
User Prompt
→ Query Rewrite (optional)
→ Document Retrieval
→ Relevance Reranking (if needed)
→ Context Construction (raw tail turns + summarized history)
→ Prompt Assembly
→ LLM Inference
→ Tool Execution (if needed)
→ Final Response
```

## ✨ Features

The system offers a complete pipeline for document-to-chat capabilities:

- **Intelligent Document Ingestion**
  - Extracts, parses, and processes content from PDFs, Mediawiki pages, and HTML documents via a high-fidelity pipeline that includes:
    - **Smart Chunking:** Configurable text chunking strategies to optimize retrieval context.
    - **Semantic Indexing:** Generates vector embeddings and stores them along with rich metadata in the Qdrant vector database.
    - **Noise Filtering:** Configurable rules to ignore irrelevant sections (e.g., headers/footers, references, etc.) for cleaner context.
  - **Batch Ingestion:** Process multiple local PDFs (`file://`) and remote URLs with optional token/cost estimation.
  - Preserve document structure and handle both structured and unstructured content.


- **Configurable Chat Orchestration**
  - **LLM Configuration :** Supports OpenAI models, with *separate* configurable models for each stage: Query Rewrite, Summarizer, Reranker, and Final Inference.
  - **Retrieval Optimization:** Fine-tune retrieval with configurable parameters:
    - Top-K, Distance Thresholds for Qdrant results.
    - Query Rewrite: Configurable step for refining search queries with a confidence factor.
    - Re-Ranking: Integrated Re-ranking stage applied to Qdrant results for improved relevance.
  - **Context Control & Cost Management:** Conversation chain context is highly configurable:
    - Set the number of **raw tail turns** and **summary turns** and **Token limits** to be included in the context. This allows users to strike an optimal balance between conversational history and token cost management.
  - **Real-Time Observability (SSE):** Real-time Server-Sent Events (SSE) showing the progress of the entire RAG flow.
- **Extensible Tool Calling:** Includes example tools (`get_nearby_airports`, `get_weather`) and supports adding custom API-driven tools for live data..
  - **Per-Stage Cost Metrics:** Provides **tokens and cost for every stage** of the RAG pipeline.
  - **Final Response:** Final responses across multiple documents with citations. 

## ⚙️ System Requirements

The application requires the following to run. Follow the Quick Start instructions to get started:

- **macOS/Linux** . Not tested on Windows although it should work with docker
- **Qdrant (Vector Database)** v1.14.1
- **Docker** (required to run Qdrant)
- **Python** 3.9 or higher
- **OpenAI API Key** (for using OpenAI models)
- **Git** (for cloning the repository)


## 🚀 Quick Start - Setup & Run

There are two supported ways to get the application running. Choose the path that best fits your use case.
>**Note:** The provided `Makefile` simplifies setup and management. It contains a number of helpful targets to get started and manage the application.

**Method 1 — Deployment Quick Start (Docker Compose)**  
*Best for users who want to run the application quickly with minimal setup.*

```
Setup OpenAI account → Install Docker → Start Application → Seed Data (optional) → Launch application
```

**Method 2 — Developer Setup (Hybrid Mode)**  
*Best for developers who want to modify or extend the backend code.*

```
Setup OpenAI account → Install Docker → Create Python Environment → Start Application → Seed Data (optional) → Launch application
```


### 1. Environment Setup (for both methods)

**1.1) Clone the Repository**
``` bash
git clone https://github.com/vrraj/chat-with-rag.git
cd chat-with-rag

```

**1.2) Install Docker and Verify Installation**

If you don't have Docker installed, follow the instructions for your platform:
   - Mac / Windows: [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)
   - Linux: Setup Docker Engine and Docker Compose

   ```bash
   # Verify Docker Installation
   docker --version
   docker-compose --version

   ```
   You should see version numbers for both commands if Docker is installed correctly.

> **Note:** Linux Users: If you encounter a "permission denied" error when running Docker commands, it means your user is not in the docker group. To fix this, please follow the post-installation steps for Linux, which typically involve adding your user to the group: `sudo usermod -aG docker $USER` and then logging out and back in to apply the change.



### 2. OpenAI API Access & Cost Management
> **Note:** The core API features of this application require access to the **OpenAI API Platform** (not just ChatGPT access) for ingesting data, querying documents, and chatting with your knowledge base.

>Users are responsible for managing and monitoring their own API usage and associated costs.To keep usage predictable, it’s strongly recommended to create a **dedicated project, API key and budget** for this application. 

| Recommendation | Action | Rationale |
| :--- | :--- | :--- |
| **Budget** | An initial budget of **$5–$10** is generally plenty for testing. | Establishes a **safety ceiling** to prevent unexpected costs. |
| **API Key** | Create a **new, dedicated API Key** named something like `chat-with-rag` in your [OpenAI API Key Dashboard](https://platform.openai.com/api-keys). | Allows you to track all usage specifically for this application on your usage dashboard. |
| **Limits** | Set a **hard usage limit** or a **notification threshold** on your account or a dedicated Project to receive an email alert when you approach your set budget. | Provides proactive cost control. |


### 3. Set up the Application

Use one of the following methods to set up the application. 
#### Method 1: Deployment Quick Start

1.1 **Set OpenAI API Key in .env**
> **Optional:** Advanced users may instead set `OPENAI_API_KEY` as an OS environment variable.  
> If set, it will take precedence over the value in `.env`.

Copy the example .env file to .env and set your OpenAI API key
OPENAI_API_KEY=your-key-here
```bash

cp .env.example .env
vi .env

```
1.2 **Start the Application**:

Start Qdrant and the web app. Pulls **qdrant image** from dockerhub.
```bash
make start

```
**Note for macOS Users:**
The make start command is configured to automatically launch the Docker Desktop application if it is not already running. The script will pause briefly while waiting for the Docker daemon to initialize before starting the containers.


1.3  **Seed sample data**
After seeding data open the **View Documents** page in the frontend UI — it displays the complete dataset loaded into Qdrant.
You will need a python environment to run this command.
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt 
make seed

```

1.4 **Access the web interface at [http://localhost:8000](http://localhost:8000)**

#### Method 2: Developer Quick Start - Hybrid (Qdrant in Docker, Python app locally)

2.1 **Set OpenAI API Key in .env**
> **Optional:** Advanced users may instead set `OPENAI_API_KEY` as an OS environment variable.  
> If set, it will take precedence over the value in `.env`.

Copy the example .env file to .env and set your OpenAI API key
OPENAI_API_KEY=your-key-here
```bash

cp .env.example .env
vi .env

```
2.2 **Set up Python environment**
```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

```

2.3 **Start the Application**
 Starts the application in hybrid mode - Qdrant in docker and the web app locally. Pulls qdrant image from dockerhub.

```bash
make start-hybrid

```

2.4 **Seed sample data** (Optional)
You will need a python environment to run this command.
After seeding data open the **View Documents** page in the frontend UI — it displays the complete dataset loaded into Qdrant.
```bash
python3 -m venv venv
source venv/bin/activate   
pip install -r requirements.txt 
make seed

```

2.5 **Access the web interface at [http://localhost:8000](http://localhost:8000)**

---

## 🌱 Sample (Seed) Data Overview

When `make seed` is run, the application loads approximately 50 Wikipedia pages into the Qdrant `document_index` collection. These cover well-known mountains, parks, trails, and outdoor destinations worldwide.

### Explore Sample Data

To view **all** seed indexed documents (with titles, URLs, and metadata), open the  
**View Documents** page in the frontend UI — it displays the complete dataset loaded into Qdrant.

Alternatively, run in your unix shell (Terminal) from the project root directory:
```bash
source venv/bin/activate
python qdrant_scripts/qdrant_ops.py --list-titles --limit 100
```
 

### Resetting or Removing Seed Data

When `make seed` is run, all sample documents are loaded into the default Qdrant collection (config.py):

```
collection_name=document_index
```

**NOTE:** To clear the seed data and start fresh with custom documents, the simplest option is to switch to a **new collection name**.

1. #### Option A — Create a Fresh Collection (Recommended)

Edit `backend/core/config.py` file and update the `collection_name` to use a different collection name, for example:

```
collection_name=my_new_collection
```

On the next startup, the system will automatically create this new collection in Qdrant (if it does not already exist). From that point on, any documents ingested (PDFs, URLs, MediaWiki pages, etc.) will be stored in this new collection, completely separate from the original seed data. This provides a clean environment for custom data without modifying or deleting the original `document_index` collection.


2. #### Option B — Delete the document_index Collection

To delete the seed data collection, run the following command:

```bash
source venv/bin/activate
python qdrant_scripts/qdrant_ops.py --delete-collection document_index

```
This deletes the document_index collection from Qdrant. Afterward, run make seed to load new sample data into the document_index collection.

---

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

## 📦 Batch Ingestion

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
