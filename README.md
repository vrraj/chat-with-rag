# Chat with Your Docs: End-to-End RAG Pipeline

A full end‑to‑end RAG system that ingests PDFs, webpages, and internal documentation, converts them into searchable embeddings, and lets you chat with your knowledge base using real‑time retrieval. The platform also supports tool integration and extensible API‑driven connectors, enabling live data augmentation from external systems, internal services, or custom enterprise workflows.

## ✨ Features

The system offers a complete pipeline for document-to-chat capabilities:

- 📚 **Intelligent Document Ingestion**
  - Extracts, parses, and processes content from PDFs, Mediawiki pages, and HTML documents via a high-fidelity pipeline that includes:
    - **Smart Chunking:** Configurable text chunking strategies to optimize retrieval context.
    - **Semantic Indexing:** Generates vector embeddings and stores them along with rich metadata in the Qdrant vector database.
    - **Noise Filtering:** Configurable rules to ignore irrelevant sections (e.g., headers/footers) for cleaner context.
  - **Batch Ingestion:** Process multiple local PDFs (`file://`) and remote URLs with optional token/cost estimation.
  - Preserve document structure and handle both structured and unstructured content.


- ## 💬 **Configurable Chat Orchestration**
  - **LLM Configuration :** Supports OpenAI models, with *separate* configurable models for each stage: Query Rewrite, Summarizer, Reranker, and Final Inference.
  - **Retrieval Optimization:** Fine-tune retrieval with configurable parameters:
    - Top-K Tuning: Configurable `top-k` results fetched from Qdrant.
    - Query Rewrite: Configurable step for refining search queries with a confidence factor.
    - Re-Ranking: Integrated Re-ranking stage applied to Qdrant results for improved relevance.
  - **Context Control & Cost Management:** Conversation chain context is highly configurable:
    - Set the number of **raw tail turns** and **summary turns** and **Token limits** to be included in the context. This allows users to strike an optimal balance between conversational history and token cost management.
  - **Real-Time Observability (SSE):** The UI provides real-time Server-Sent Events (SSE) showing the progress of the entire RAG flow (Query Rewrite, Retrieval, Reranking, Summarizer, Tool calling, Inference) for transparency.
  - **Per-Stage Cost Metrics:** Calculates **tokens and cost for every component** of the RAG pipeline and provides **total conversation metrics** for granular budget tracking.
  - **Final Response:** Final responses across multiple documents with citations. 

## System Requirements

The application requires the following to run. Follow the Quick Start instructions to get started:

- **macOS/Linux** . Not tested on Windows although it should work with docker
- **Qdrant (Vector Database)** v1.14.1
- **Docker** (required to run Qdrant)
- **Python** 3.9 or higher
- **OpenAI API Key** (for using OpenAI models)
- **Git** (for cloning the repository)


## Quick Start - Setup & Run

There are two methods for getting the application running:

1.  **Deployment Quick Start - Method 1:** Use Docker Compose for a quick, single-command deployment of both the Web app and Qdrant. *(Recommended for users who just want to run the application.)*
2.  **Developer Setup - Method 2:** Run Qdrant in Docker while running the Python web application locally within a `venv`. *(Recommended for developers who need to modify the backend code.)*

**Note:** The `Makefile` contains a number of helpful targets to get started and manage the application.


### Environment Setup (for both methods)

**1) Clone the Repository**
``` bash
git clone https://github.com/vrraj/chat-with-rag.git
cd chat-with-rag

```

**2) Configure API Key**
Set your OpenAI API key in .env file. 
Copy the example .env file to .env and set your OpenAI API key
OPENAI_API_KEY=your-key-here
```bash

cp .env.example .env
vi .env

```

**3) Install Docker and Verify Installation**
1. Install Docker (if not already installed):
   - Mac / Windows: [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)
   - Linux: Setup Docker Engine and Docker Compose

2. Verify Docker Installation:
   ```bash
   docker --version
   docker-compose --version

   ```
   You should see version numbers for both commands if Docker is installed correctly.

### Method 1: 🚀 Deployment Quick Start)

1.1. **Start the Application**:

Start Qdrant and the web app. Pulls **qdrant image** from dockerhub if not already downloaded.
```bash
make start

```

1.2 (Optional, last) **Seed sample data**
After seeding data open the **View Documents** page in the frontend UI — it displays the complete dataset loaded into Qdrant.
You will need a python environment to run this command.
```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt 
make seed

```

1.3 **Access the web interface at [http://localhost:8000](http://localhost:8000)**

### Method 2: 🚀 Developer Quick Start - Hybrid (Qdrant in Docker, Python app locally)

2.1) **Set up Python environment**
```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

```

2.2) **Start the Application**
 Starts the application in hybrid mode - Qdrant in docker and the web app locally. Pulls qdrant image from dockerhub if not already downloaded.

```bash
make start-hybrid

```

2.3) **Seed sample data** (Optional)
You will need a python environment to run this command.
After seeding data open the **View Documents** page in the frontend UI — it displays the complete dataset loaded into Qdrant.
```bash
python3 -m venv venv
source venv/bin/activate   
pip install -r requirements.txt 
make seed

```

2.4 **Access the web interface at [http://localhost:8000](http://localhost:8000)**

---

## Sample (Seed) Data Overview

When you run `make seed`, the application loads approximately 50 Wikipedia pages into the Qdrant `document_index` collection. These cover well-known mountains, parks, trails, and outdoor destinations worldwide.

### Explore Sample Data

To view **all** seed indexed documents (with titles, URLs, and metadata), open the  
**View Documents** page in the frontend UI — it displays the complete dataset loaded into Qdrant.

Alternatively, run in your unix shell (Terminal) from the project root directory:
```bash
source venv/bin/activate
python qdrant_scripts/qdrant_ops.py --list-titles --limit 100
```
 

### Resetting or Removing Seed Data

When you run `make seed`, all sample documents are loaded into the default Qdrant collection (in config.py):

```
collection_name=document_index
```

If you want to clear the seed data and start fresh with your own documents, the simplest option is to switch to a **new collection name**.

#### Option A — Create a Fresh Collection (Recommended)

Edit `backend/core/config.py` file and update the `collection_name` to use a different collection name, for example:

```
collection_name=my_new_collection
```

On the next startup, the system will automatically create this new collection in Qdrant (if it does not already exist) and start indexing into it. From that point on, any documents you ingest (PDFs, URLs, MediaWiki pages, etc.) will be stored in this new collection, completely separate from the original seed data. This gives you a clean environment for your own data without modifying or deleting the original `document_index` collection.




#### Option B — Delete the document_index Collection

If you want to delete the seed data collection, you can do so by running the following command:

```bash
source venv/bin/activate
python qdrant_scripts/qdrant_ops.py --delete-collection document_index

```

This will delete the `document_index` collection from Qdrant. You can then run `make seed` to load new sample data into the `document_index` collection.
---

### Example Queries You Can Try
This is based on the seed data.

#### Retrieval-based
	•	“Where is Mount Kilimanjaro located?”
	•	“Compare Kilimanjaro with Mount Whitney.”
	•	“Which mountain is taller — Everest or Whitney?”
	•	“List all mountains and peaks in the Alps. Which one is the tallest and what is tthe closest airport to it?”
	•	“Tell me about hiking in Patagonia. What is the weather forecast for the area?”

#### Conversation chain based examples (remembers context)

USER: where is mount kilimanjaro
ASSISTANT:Mount Kilimanjaro is located in Tanzania....

USER: compare kilimanjaro with Mount whitney

ASSISTANT: Mount Kilimanjaro and Mount Whitney are both prominent peaks, but differ in height, location....

USER: what is the closest airport to Kilimanjaro

ASSISTANT:The nearest airport to the coordinates (-3.0674, 37.3556) is **Amboseli Airport (ASV • HKAM)**, ~48 km away.
Tools Used: get_nearby_airports

USER: and for whitney

ASSISTANT:Closest airports to Mount Whitney (36.5785, -118.2919)....
Tools Used: get_nearby_airports

USER: what is the weather forecast for mount whitney

ASSISTANT:Mount Whitney, California — 28 Oct  High: 7.2 °C (45 °F)....
Tools Used: get_weather

## Batch Ingestion

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


## Technical Overview

A detailed technical explanation of the system architecture, pipelines, design decisions, and engineering approach is available here:

👉 **[TECHNICAL_OVERVIEW.md](TECHNICAL_OVERVIEW.md)**

This overview covers module structure, extraction pipeline, embedding flow, Qdrant indexing, batch ingestion (local PDFs + URLs with optional cost estimation), chat orchestration, SSE streaming, and frontend–backend integration.

## Project Structure



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
logs/                        # Rotating runtime logs
qdrant_storage/              # Local Qdrant data volume
```


© 2025 Rajkumar Velliavitil — All Rights Reserved