

# 🧠 Technical Overview

## 📚 Table of Contents

- [High‑Level Architecture Diagram](#-highlevel-architecture-diagram)
- [Purpose and Scope](#-purpose-and-scope)
- [System Overview](#-system-overview)
- [Runtime & Deployment Model](#-runtime--deployment-model)
- [Ingestion Pipeline](#-ingestion-pipeline)
  - [2a. Batch Ingestion](#-2a-batch-ingestion)
  - [3. High-Level Flow](#-3-high-level-flow)
  - [4. Content Source Selection](#-4-content-source-selection)
  - [5. Extraction](#-5-extraction)
    - [5.1 Source-Specific Extraction Behavior](#-51-source-specific-extraction-behavior)
    - [5.2 Tables and Structured Data (High-Level)](#-52-tables-and-structured-data-high-level)
  - [6. Chunking & Metadata](#-6-chunking--metadata)
  - [7. Embedding](#-7-embedding)
  - [8. Index Storage (Qdrant)](#-8-index-storage-qdrant)
  - [9. Re-indexing and Maintenance](#-9-re-indexing-and-maintenance)
- [Embedding Flow](#-embedding-flow)
- [Chat Orchestration](#-chat-orchestration)
- [Retrieval and Ranking](#-retrieval-and-ranking)
- [SSE Streaming](#-sse-streaming)
- [Frontend–Backend Integration](#-frontendbackend-integration)
- [Metrics and Observability](#-metrics-and-observability)
- [Configuration and Settings](#-configuration-and-settings)
- [Error Handling and Stability Guarantees](#-error-handling-and-stability-guarantees)
- [Extensibility and Organization-Specific Customization](#-extensibility-and-organization-specific-customization)
- [Architecture Summary](#-architecture-summary)
- [Repository Structure (High-Level)](#-repository-structure-high-level)
- [Developer & Operator Utilities (Makefile)](#-developer--operator-utilities-makefile)
- [Qdrant Operations CLI](#-qdrant-operations-cli)
- [Automated Quality Checks (CI Workflow)](#-automated-quality-checks-ci-workflow)
- [Browser Compatibility: Secure Context Requirement](#-browser-compatibility-secure-context-requirement)
- [API Examples (Advanced)](#-api-examples-advanced)

---

## 🗺️ High‑Level Architecture Diagram

A simplified conceptual overview of the system’s flow:

**Ingestion Flow**

```
        +-------------------+
        |   Source Docs     |
        | (HTML/Wiki/PDF)   |
        +---------+---------+
                  |
         Extraction & Processing
                  |
          Chunking & Metadata
                  |
     Estimation / Embedding Generation
                  |
             Qdrant Index
                
```

**Retrieval Flow**

```
        +-------------------+
        |   User Query      |
        +---------+---------+
                  |
  Query Processing (Rewrite / Clarify) 
                  |
              Retrieval
                  |
              Reranking
                  |
      Context Assembly (Context Window)
                  |
            Prompt Building
                  | 
        LLM Reasoning & Tool Calls
                  |
            Final User Response
            
```

.

## 🎯 Purpose and Scope

The RAG Pipeline Chat system is designed to help organizations convert their internal
knowledge—such as technical documentation, operational manuals, process descriptions,
wikis, and PDF repositories—into an interactive AI‑powered conversational interface.
By ingesting large **heterogeneous content sets** and indexing them with semantic search,
the system enables employees, customers, or support agents to query their knowledge
base through grounded, auditable, context‑aware chat interactions.

This document is intended for system architects, AI engineers, and collaborators who want to understand the system's architecture, including ingestion, embedding, retrieval, chat orchestration, and real-time streaming.

> **Note:** While this repository includes example tools and sample datasets, it is designed as a
general‑purpose reference architecture rather than a domain‑specific product. You are expected to integrate your own content sources, internal tools, APIs, and policies to adapt the platform to your specific workflows and requirements.

### What This Is Not

This repository is not a turnkey enterprise product or a drop‑in replacement for
organization‑specific knowledge platforms. Instead, it serves as a modular,
extensible reference implementation that teams can adapt, extend, and integrate
with their own tools, data sources, workflows, and compliance requirements.


## 🧩 System Overview

The RAG Pipeline Chat application integrates document ingestion, vector indexing, semantic retrieval, and Large Language Model (LLM)-based reasoning into a unified end-to-end architecture. It is designed to support context-grounded chat interactions over heterogeneous content sets.

- **Ingestion Pipeline** – Extracts content from documents, chunks text, embeds it, and stores structured vectors in Qdrant.
- **Embedding Flow** – Generates embeddings from text chunks with full metadata preservation and cost-estimation capabilities.
- **Retrieval & Ranking** – Performs semantic search and applies heuristic or LLM‑based reranking to produce high‑quality context windows.
- **Chat Orchestration** – Manages the multi‑stage pipeline of rewriting, retrieval, reranking, context assembly, LLM inference, tool calls, and final synthesis.
- **Metrics & Observability** – Provides detailed telemetry, per-stage token/cost metrics, and real-time SSE streaming.
- **SSE Streaming** – Streams pipeline stages and incremental model output to the frontend in real‑time.
- **Frontend–Backend Integration** – Coordinates chat requests, SSE connections, UI state, and multi‑turn interactions.
- **Configuration & Settings** – Centralizes model choices, thresholds, safeguards, and feature flags.
- **Error Handling & Stability** – Ensures the system fails gracefully and avoids runaway computation.

Together, these components form a modular, scalable architecture that supports reliable RAG‑augmented conversational experiences.

## 🚀 Runtime & Deployment Model

For local development, the RAG Pipeline Chat application runs as a small containerized stack managed by Docker Compose:

- **Webapp container** – FastAPI backend, SSE streaming, and the browser-based UI
- **Qdrant container** – the vector database used for all document embeddings

The `docker-compose.yml` file in the repository root defines these services, their ports, and the shared storage volume (`qdrant_storage/`). Starting the system with Docker Compose launches both services and ensures the backend is automatically connected to Qdrant through configuration settings.

In production environments, teams commonly:

- point the backend at an external or managed Qdrant instance  
- run the webapp container behind a reverse proxy (e.g., nginx, Caddy) with HTTPS  
- integrate the backend into internal APIs or enterprise tools via the tools layer  

The ingestion, retrieval, and chat pipelines remain fully decoupled from where Qdrant is hosted; only configuration variables need to be adjusted.


## 📥 Ingestion Pipeline

The ingestion pipeline is responsible for converting raw documents (HTML, MediaWiki, PDF, etc.) into structured vector entries stored in Qdrant. This section outlines the high-level flow and major components without going into low-level implementation details.

### ✅ 1. Goals

- Provide a repeatable, scriptable way to ingest content into the system
- Preserve document structure (sections, headings, source URLs) for better retrieval
- Normalize different source types (MediaWiki, HTML, PDF) into a common internal representation
- Produce rich metadata to support filtering, reranking, and traceability in chat responses

- MediaWiki pages
- Generic HTML pages
- PDF documents (structured where possible)
- Uploaded PDF files provided directly through the chat UI
- Local sample/seed documents used for demos and testing


### 📦 2a. Batch Ingestion

The Ingestion Pipeline provides a **batch ingestion mode** driven by a JSON specification. A single batch file can describe a heterogeneous set of sources—local files and remote URLs—that are processed in one run.

Each batch definition contains:

- `items`: a list of documents, where each entry specifies:
  - `url`: either a `file://` URI pointing at a local PDF or an `http(s)://` URL
  - `doc_type`: `pdf`, `mediawiki`, or `html`, which selects the appropriate extractor
  - optional `skip_sections`: headings to drop (e.g., `References`, `External links`, `See also`, `Further reading`) to reduce noise and cost
- global options such as:
  - `max_chunks`: a safety cap on the number of chunks per document
  - `estimate`: a toggle to run in estimation‑only mode or perform full embedding and indexing
  - `force_delete`: a flag indicating whether existing content for the target collection should be dropped and rebuilt

When executed, the batch runner orchestrates extraction, chunking, and (optionally) embedding for each item in sequence, emitting per‑document statistics (chunk counts, token usage, and estimated embedding cost) as well as a final aggregate summary for the batch. This enables teams to quickly onboard corporate PDF repositories, internal wiki pages, or mixed documentation sets into a single Qdrant collection through a repeatable, scriptable workflow.


> **Note:** The Ingestion Pipeline supports an **estimation mode** that runs extraction and chunking steps without invoking the embedding model or writing to the index. This mode is useful for:
>- Quickly previewing how many chunks each document will produce
>- Validating extraction rules or section skipping logic
>- Estimating embedding costs before committing to full ingestion

> Estimation mode can be triggered via CLI flags or configuration settings.

### 🔄 3. High-Level Flow

At a high level, the Ingestion Pipeline follows this sequence:

1. **Content Source Selection** – Identify which documents or URLs should be ingested.
2. **Extraction** – Use specialized extractors to pull clean text and structure from each source type.
3. **Chunking & Metadata Construction** – Split documents into logical chunks and attach metadata.
4. **Embedding** – Convert chunks into vector embeddings using the configured embedding model.
5. **Index Storage (Qdrant)** – Upsert embeddings and metadata into the Qdrant collection.

Each of these stages is implemented as a separate component so that they can evolve independently.

### 🧭 4. Content Source Selection

- Determines **which content sources** (URLs, MediaWiki pages, PDFs, or local seed files) should be ingested.
- Serves as the entry point for ingestion scripts and Make targets.
- Does not perform parsing; it only defines the set of inputs that flow into extraction.
- Can be extended in the future to support automated discovery such as crawling, category-based wiki selection, or file-system monitoring.


### 🧹 5. Extraction

- Uses specialized extractors per source type (MediaWiki, HTML, PDF)
- Normalizes output into a common internal structure (text + structural metadata)
- Preserves important layout information where possible (headings, sections, paragraphs)

#### 🧬 5.1 Source-Specific Extraction Behavior

Although all extractors normalize into the same internal representation, each source type has additional behavior to preserve as much context as possible:

- **MediaWiki pages** – When a Parsoid endpoint is available, the extractor uses it first and falls back to the Action API only when necessary. Ingestion preserves the lead section and heading hierarchy (H1/H2/H3+ mapped into `section` / `subsection` fields). Infobox content is retained. The extractor also supports table-aware ingestion for list-style pages: table content can be indexed in a structured form (preserving column context across chunks) rather than as flattened prose.

- **PDF documents** – The PDF extractor is layout-aware and can leverage `pymupdf4llm` to reconstruct headings, sections, tables, and sidebar “infobox”-style panels (such as summary boxes with elevation, prominence, or key parameters). These are mapped into the same `section` / `subsection` schema used for MediaWiki, so mixed corpora of wiki pages and PDFs behave consistently at retrieval time. Table content can be indexed separately from prose to avoid duplication, but extraction quality still depends on the PDF having recognizable table structure.

- **HTML pages** – The HTML extractor uses headings and container structure to infer sections and subsections and makes a best effort to capture key information such as hero text, infobox-style side panels, and tables. The quality of this structure depends on the source HTML following basic best practices (semantic headings, real table markup for tabular data, actual text instead of text baked into images, and minimal reliance on JavaScript-only rendering). Poorly structured pages are still ingested, but may appear as flatter “Lead-only” documents with less granular section metadata.

#### 🧱 5.2 Tables and Structured Data (High-Level)

Tabular content is common in knowledge bases (inventory lists, spec sheets, comparison tables, wiki lists). The ingestion pipeline supports table-aware extraction across source types, but the quality of results depends on the underlying document being structured:

- **Best-case:** tables are present as real tables (HTML `<table>`, MediaWiki wikitables, or PDFs with consistent cell geometry). Column context can be preserved so each chunk remains interpretable.
- **Degraded-case:** if the source is poorly structured (tables rendered as images, irregular layout, or heavy JS rendering), extraction may fall back to flattened “table-like” text. This is still indexable but may lose column semantics.

As a result, table-aware ingestion is additive and can be enabled without changing the baseline prose indexing, but it benefits significantly from sources that follow standard formatting conventions.

### ✂️ 6. Chunking & Metadata

- Splits extracted text into semantically meaningful chunks rather than arbitrary fixed-size splits
- Attaches metadata such as:
  - section/heading hierarchy
  - source URL or file path
  - document identifiers and chunk IDs
  - optional character offsets or spans
- Produces a stream of `(chunk_text, metadata)` pairs that are ready for embedding

### 🧲 7. Embedding

- Takes `(chunk_text, metadata)` pairs and calls the configured embedding model
- Produces vector embeddings while preserving the metadata alongside the vector
- Handles batching and model configuration where applicable


### 🗄️ 8. Index Storage (Qdrant)

- Upserts embeddings and metadata into a Qdrant collection dedicated to this project
- Ensures consistency of:
  - collection name and schema
  - payload fields (e.g., source, section, chunk ID)
- Provides a foundation for semantic search, reranking, and retrieval in the chat pipeline

#### 🧰 Collection Management

The system uses a single active Qdrant collection at any given time, with the
name controlled by the `collection_name` setting in:

```
backend/core/config.py
```

The default is:

```
collection_name = "document_index"
```

Changing this value points the ingestion and retrieval layers to a different
logical dataset. When a new collection name is provided, Qdrant will
automatically create the collection on first write, using the configured
embedding dimensionality and payload schema.

This mechanism enables:

- isolating seed/demo data from user‑specific datasets  
- maintaining multiple datasets side‑by‑side in the same Qdrant instance  
- switching datasets by configuration instead of manual DB operations  


All ingestion pipelines (HTML, MediaWiki, PDF, batch) and all retrieval flows
always operate against the currently configured collection.

### 🌱 Seed Data and Demo Collection

For local development and exploration, the repository includes a small demo dataset
that can be ingested into the default Qdrant collection (`document_index`) using the
standard Makefile targets or ingestion scripts.

- The raw seed data lives under `data/` (for example `docs-index-seed.jsonl` and related files).
- Running the seeding workflow embeds these documents and loads them into the active
  collection defined in `backend/core/config.py`.
- The frontend provides a **List Documents** view that displays all indexed documents
  along with their titles, source URLs, and metadata, making it easy to inspect the
  current contents of the collection.

In production or enterprise deployments, teams typically ingest their own internal
documentation repositories and may disable or replace the demo dataset entirely.

### 🧹 9. Re-indexing and Maintenance

- Supports clearing and rebuilding the index when seed data or extraction logic changes
- Supports targeted deletion by URL, removing all indexed chunks associated with a specific source URL so individual documents can be re-ingested or retracted without affecting other data.


## 🧮 Embedding Flow

The embedding flow transforms text chunks produced by the chunking stage into high‑dimensional vector embeddings suitable for semantic retrieval. The embedding layer is provider‑extensible at the code level, but changing the embedding model requires a full re‑embedding and re‑indexing of the corpus, as vectors produced by different models are not directly comparable. The system is designed to be efficient and fully metadata‑preserving.

> **Note (Re-embedding workflow):** If you want to experiment with a different OpenAI embedding model (e.g., `text-embedding-3-large`), export your existing document URLs, update the embedding model in `backend/core/config.py`, and then re-ingest the same URLs using the batch ingestion mode. A JSON file ready for batch import can be exported directly from the UI via **List Documents → Download JSON**.


### 1. Input and Output

- **Input:** `(chunk_text, metadata)` pairs produced by the chunking stage
- **Output:** `(embedding_vector, metadata)` objects ready for Qdrant indexing

This strict separation ensures that metadata flows through the system unchanged.

### 2. Model Abstraction Layer

The embedding component wraps the embedding model behind a dedicated interface. This allows:

- Selecting a model per stage without modifying pipeline logic
- Configuring model parameters in one place
- Adding retries, batching, and API‑related safeguards

> **Note:** At this time, stage-level model selection is limited to OpenAI models (embedding, rerank, and inference). The abstraction keeps the ingestion pipeline decoupled from model implementation details and leaves room for additional providers later.

### 3. Token and Cost Estimation

During estimation mode, the embedding stage computes:

- approximate token counts
- projected embedding costs
- total chunks and estimated compute usage

The system performs these calculations **without generating any vectors**, allowing users to preview ingestion costs before without having to encounter costs associated with the embedding model.

### 4. Batching and Throughput

The embedding flow processes chunks in batches to improve performance and reduce API overhead. Batching ensures:

- consistent embedding dimensionality
- efficient parallelization
- predictable request boundaries

The pipeline maintains ordering so metadata remains aligned with each embedding.

### 5. Metadata Preservation

All metadata generated in earlier stages (e.g., section hierarchy, source URL, chunk ID) is preserved verbatim during embedding. This enables:

- structured filtering and reranking during retrieval
- provenance tracing in chat responses
- improved interpretability of retrieved chunks

Metadata a.k.a payload is extensible and can be configured in the embedding component.

### 6. Error Handling and Logging

The embedding layer includes:

- cost and token‑usage logging
- exception handling for extraction/formatting anomalies
- retry logic for transient API issues
- safeguards to ensure failed embeddings do not corrupt the index

These controls enhance reliability during large‑scale ingestion.

---


## 🧠 Chat Orchestration

This section will describe how user queries flow through the multi-stage chat pipeline, including retrieval, reranking, tool calls, and final LLM response construction. It will outline the orchestration sequence and the role of intermediate stages.

### 1. Query Reception and Validation

- Receives user input queries
- Validates and sanitizes input for security and correctness

### 2. Key Stages

At a high level, the chat orchestration pipeline processes queries as follows:

1. **Query Reception** – Accept the user query from the frontend.
2. **Preprocessing** – Normalization, cleaning, lowercasing, and optional heuristics.
   This stage also includes an optional **user-query rewrite pass**, where the system corrects spelling mistakes, expands abbreviations, and normalizes phrasing to improve retrieval quality. The rewritten query is used internally but the user's original query is preserved for context. In ambiguous cases, this stage may trigger a clarification question instead of proceeding directly to retrieval, and the pipeline returns early after emitting a `Clarification Needed` stage.
3. **Retrieval** – Perform semantic search on the indexed vectors in Qdrant. Gather top-K chunks and prepare prompt context
4. **Reranking** – Apply metadata filters and score-based reranking.
5. **Context Assembly** – Context is assembled by with configurable raw tail turns, summary turns, and retrieved context.
6. **Context Augmentation** – The context is augmented with the system prompt and user query to generate the final context for final LLM Inference.
7. **LLM Reasoning** – The context, LLM Inference and Tool Calls (if needed) are combined to generate the final response.
8. **SSE Stage Emission** – Stream intermediate results to the frontend.
9. **Response Delivery** – Send the final answer back to the user.

### 3. Retrieval

- Queries the vector store with the processed user query
- Applies metadata filters and score thresholds
- Returns top-K relevant chunks for downstream reasoning

### 4. Reranking

- Uses additional heuristics or secondary models to reorder chunks
- Supports user or system preferences for filtering


### 5. Context Assembly

- Concatenates retrieved chunks into a coherent context window
- Inserts system prompts and user query information

Context assembly builds the inference context using two inputs: **raw tail turns** (a configurable number of the most recent user/assistant turns) and **summarized history** (a condensed representation of earlier conversation state). These controls are configurable so deployments can tune for cost, context window pressure, and answer quality.

### 6. LLM Reasoning

- Invokes the language model with assembled context
- Supports multi-turn interactions and state management

### 7. Tool Calls During Inference

The orchestration layer supports LLM-initiated **tool calls** during the reasoning phase when tools are enabled. Tool calls are intended for **external data augmentation** (outside the vector store) that complements retrieved document context—for example, live lookups, structured datasets, or organization-specific APIs. Vector-store retrieval is handled by the retrieval stage and does not typically require tool calls.

Tools are used for enriching the final answer with operations such as:

- retrieving live or dynamic data (e.g., weather, nearby airports, schedules)
- calling internal or enterprise APIs (e.g., ticketing systems, inventory, CRM)
- performing structured computations or validation steps
- running project-specific utilities that add value to the final answer

When the LLM requests a tool call, the orchestrator:

1. extracts tool calls from the model output
2. validates tool names and arguments
3. executes tools through a registry of executors
4. aggregates tool outputs
5. performs a final synthesis pass to merge retrieved context and tool results into the final response

### 8. SSE Stage Emission

For each major step in the pipeline, the orchestrator emits a human-readable SSE stage label so the frontend can display live progress. Typical stages include:

- `Query Rewrite`
- `Clarification Needed` (when ambiguity is detected and a follow-up question is asked)
- `Retrieve Vectors`
- `Rerank Retrieval Results`
- `Summarize Chat History`
- `Establish Web Context` (when web search is enabled)
- `Inference Prompt Build`
- `Generating Response`
- `Tool Calls`

The detailed wiring of SSE connections and reconnection behavior is described separately in the **SSE Streaming** section, but the orchestration logic here defines *what* gets emitted and *when*.

## 🔎 Retrieval and Ranking

The retrieval and ranking subsystem identifies the most semantically relevant document chunks to support the LLM's answer generation. It operates in two phases: vector retrieval and optional LLM-based reranking.

### 1. Query Embedding
A single embedding is generated for the rewritten user query to be used to retrieve the most relevant document chunks from Qdrant.

### 2. Vector Search (Qdrant)
The system executes a similarity search with:
- cosine (or dot-product) similarity
- configurable `top_k` limits
- optional payload filters (URL, doc_type, section)

> **Tuning note (Top‑K vs cost):** Retrieval quality is sensitive to the `top_k` candidate set. For noisy datasets or ambiguous queries, increasing `top_k` can improve recall, but it may increase downstream reranking cost (when enabled) and can place additional pressure on the inference context budget. This trade‑off is intentional and configurable.

Each Qdrant result includes:
- embedding similarity score
- chunk text
- full metadata payload

### 3. Filtering
Before reranking, results may be filtered by (not implemented yet):
- document type
- URL/domain
- section name or headings
- minimum similarity threshold

### 4. Heuristic Reranking
The system applies a lightweight heuristic layer to improve relevance:
- exact-match boost
- clear-winner detection

### 5. LLM Reranking (optional based on retrieved context)
For ambiguous retrieval sets, the query and top candidates are passed to a rerank model. This produces refined relevance scores and a reduced top-K set.

### 6. Final Selection and Context Packaging
The number of retrieved chunks included in the inference prompt is bounded by a configurable inference input limit (i.e., how many context rows/chunks are allowed to be sent to the model), with retrieval and optional reranking providing the candidate set. The inference prompt is then assembled from **retrieved chunks**, **raw tail turns**, **summarized history**, and **tool outputs (when applicable)** to build the final context for LLM inference.

## 📡 SSE Streaming

The SSE (Server-Sent Events) subsystem delivers real-time streaming updates from the backend to the browser. It enables the UI to reflect pipeline progress and LLM output incrementally.

### 1. Endpoint Structure
Each chat request receives a unique `stream_id`. The frontend connects to:

```
/stream/{stream_id}
```

using an `EventSource` client.

### 2. Event Format
The server emits UTF‑8 encoded events of the form:
```
event: message
data: { ... JSON payload ... }
```
Each message corresponds to a pipeline stage or LLM token.

### 3. Stage Emission
The orchestrator uses a shared `emit_stage()` helper to push structured stage updates. Stages are human-readable and reflect the exact progress in chat orchestration.

### 4. Token Streaming
During the LLM call, partial tokens are streamed as incremental `data:` messages. The frontend appends these to the visible response.

### 5. Disconnect Handling
`sse_starlette` automatically detects client disconnects. The backend:
- terminates the streaming loop
- unregisters consumer handlers
- cleans up stream registry state


## 🔗 Frontend–Backend Integration

The frontend interacts with the backend through two channels:
1. REST POST requests for submitting user messages
2. SSE streams for receiving staged updates and model output

### 1. Request Lifecycle
- User submits a message
- Frontend sends POST `/chat` with the message payload
- Backend generates a `stream_id` and begins orchestration
- Frontend immediately opens `EventSource(/stream/{stream_id})`

### 2. Handling SSE Messages
The UI:
- updates progress indicators based on stage labels
- appends partial tokens to the chat window
- displays clarification prompts when emitted
- finalizes messages when `Done` is received

### 3. Error Handling
Frontend reacts to:
- malformed SSE messages
- dropped connections (EventSource auto-reconnect)
- explicit error stages from backend

### 4. State & History
The UI maintains:
- multi-turn conversation state
- chat history for context
- per-message streaming buffers

## 📊 Metrics and Observability

The system tracks detailed metrics across two phases: **ingestion-time estimation** and **chat-time execution**.

### 1. Ingestion-Time Estimation Metrics (Embedding Cost Preview)
During ingestion, the system can run in an estimation-only mode that performs extraction and chunking without generating vectors. In this mode, metrics focus on predicting embedding cost before indexing:

- estimated token usage per chunk and per document
- estimated total embedding tokens and cost for the ingestion run
- per-document chunk counts and safety-cap outcomes (e.g., max chunk limits)

### 2. Chat-Time Per-Turn Metrics (Runtime Costs)
During chat execution, the UI displays **per-turn** metrics that break down token usage and cost by stage:

- **Query Embedding:** tokens and cost to embed the user query
- **Query Rewrite (optional):** input/output tokens and cost
- **Retrieval:** Qdrant response timing and top-k outcomes
- **Rerank (optional):** tokens and cost for reranking calls
- **Summarizer (optional):** tokens and cost for history summarization
- **Inference:** prompt/completion tokens and cost for the final LLM call

### 3. Running Conversation Totals
In addition to per-turn metrics, the system maintains running totals across the conversation during chat sessions:

- cumulative input/output tokens
- cumulative cost (per stage and overall)
- total turns and aggregate usage

### 4. Logging and Diagnostics
Logs are structured with per-stage prefixes and include:

- SSE stage emission and streaming traces
- ingestion and chunk-processing traces
- model invocation summaries (tokens/cost where available)
- error traces for transient API failures and pipeline fallbacks

The centralized logging configuration in `backend/core/logging.py` also configures:

- **Rotating server logs:** `logs/server.log` is capped at a fixed size per file with a limited number of backups to prevent unbounded disk usage.
- **Rotating error logs:** `logs/error.log` follows the same rotation strategy, retaining only the most recent error history needed for debugging.



## ⚙️ Configuration and Settings

Configuration is centralized in `backend/core/config.py`, which acts as the control plane for model selection, pipeline behavior, and safety limits. The system is designed so most behavior can be tuned through configuration without requiring code changes.

At a high level, configuration covers the following control categories:

- **Model selection (per stage):** embedding, reranking, inference, and tool-synthesis models used across ingestion and chat.
- **Extraction behavior (by source type):** source-specific controls for HTML, MediaWiki, and PDF extraction, including table-aware extraction and preservation of structured content where supported.
- **Chunking controls:** chunk size and overlap, section-aware chunking behavior, and per-document safety caps (maximum chunks and token budgets).
- **Table and structured data handling:** enable/disable table indexing, structured versus flattened table strategies, and safeguards to avoid duplicate or oversized table-derived chunks.
- **Retrieval and ranking behavior:** top-k retrieval limits, similarity thresholds, and optional reranking toggles.
- **Context assembly knobs:** configuration for how conversation history is incorporated (raw tail turns versus summarized history) and limits that prevent context-window overflow.
- **Stability and safeguards:** retry policies, timeouts, and failure caps applied across ingestion and chat execution.
- **Feature flags:** enable or disable optional capabilities such as tools, web search, estimation-only ingestion mode, and related experimental features.

Configuration values are loaded at application startup and can be overridden via environment variables for different deployment environments. Defaults are chosen to be safe for local development while remaining suitable for production tuning.

## 🛡️ Error Handling and Stability Guarantees

The system employs multiple layers of protection to prevent runaway computation and ensure graceful degradation.

### 1. Embedding Safeguards
- per-document token budget
- per-document chunk budget
- max-failure caps
- time-limit caps
- retry with exponential backoff

### 2. Chat Pipeline Safeguards
- early exit on clarification prompts
- fallback when retrieval fails
- safe handling of malformed tool arguments
- defensive JSON parsing
- per-session summary cache with idle TTL-based eviction, ensuring in-memory chat summaries are periodically cleared for idle sessions without affecting correctness (summaries are recomputed on demand when needed)

### 3. SSE Safeguards
- automatic cleanup on disconnect
- safe stream termination
- consumer registry tracking

## 🧰 Extensibility and Organization-Specific Customization

Although this project includes working examples—such as weather and nearby‑airports tools,
sample batch ingestion, and seed datasets—it is primarily intended as a modular,
general‑purpose RAG architecture that organizations can extend to meet their unique
operational needs.

Common customization areas include:

- **Tools Layer (`backend/tools/`):** replacing example tools with integrations into
  internal systems such as ticketing platforms, analytics engines, knowledge bases,
  incident‑management systems, or proprietary APIs.
- **Content Ingestion:** pointing the indexing pipeline at internal documentation
  repositories (wikis, runbooks, process docs, PDFs) instead of the public demo
  documents included with the repository.
- **Retrieval and Guardrails:** adapting prompt templates, retrieval filters,
  section‑skip rules, tool‑calling behavior, and safety heuristics to match internal
  compliance, privacy, or quality‑of‑service constraints.
- **Frontend Integration:** embedding the chat UI into an existing application (e.g., a website, desktop application, or mobile application) or
  customer‑facing application, or wiring the backend into a different user interface.


## 🧾 Architecture Summary

The RAG Pipeline Chat system is composed of modular, loosely coupled stages:

```
User Query → Rewrite → Retrieval → Rerank → Context Assembly → LLM Reasoning → Tool Calls → Final Synthesis
```

Ingestion flows independently:
```
Source → Extraction → Chunking → Embedding → Qdrant
```

This separation ensures maintainability, extensibility, and clear reasoning paths throughout the system.


## 🗂️ Repository Structure (High-Level)

At a high level, the repository is organized into the following areas:

- **Root**
  - `start.py` – entry point used by the Docker container for production (no uvicorn reload flags)
  - `run.py` – entry point for running the app in development mode (uses uvicorn with `--reload`)
  - `docker-compose.yml` – orchestrates the Webapp and Qdrant containers
  - `requirements.txt` – Python dependencies
  - `Makefile` – convenience targets for running, seeding, and tooling
  - `README.md`, `TECHNICAL_OVERVIEW.md`, `SSE-stream.md`, `ATTRIBUTIONS.md`
  - `.env`, `.env.example`, `.python-version`, `.gitignore`

- **backend/** – core server-side application
  - `main.py` – FastAPI app wiring and startup
  - `api/` – HTTP endpoints (e.g., chat processing)
  - `chat/` – chat orchestration, web search integration, and overview helpers
  - `core/` – configuration, logging, schemas, and shared utilities
  - `embeddings/` – embedding manager, collection management, and related utilities
  - `extractor/` – HTML, MediaWiki, and PDF extractors plus text splitters
  - `crawler/` – URL and PDF crawling utilities
  - `db/` – Qdrant client and vector database abstraction
  - `tools/` – tool-call implementations (weather, nearby airports, web search)
  - `utils/` – helper scripts such as Qdrant collection creation and prompt utilities
  - `scripts/` – backend-side ingestion helpers (e.g., URL processing)
  - `qdrant_scripts/` – Qdrant management scripts (e.g., collection creation, deletion, listing)
  - `stream_registry.py`, `stream_stages.py`, `stream_emit.py` – SSE stream coordination
  - `dump_vector.py` – debugging and inspection of stored vectors

- **frontend/** – browser UI and static assets
  - `index.html`, `chat.html`, `search.html`, `list-docs.html`, `debug-index.html`, `process-batch-docs.html`
  - `static/` – CSS and JS bundles (`chat.css`, `search.css`, `styles.css`, `chat.js`, `search.js`, `app.js`, etc.)
  - `src/components/` – React-style components for search and chat sections
  

- **scripts/** – standalone maintenance and batch utilities
  - `qdrant_query_url.py`, `qdrant_clone_collection.py`, `qdrant_create_payload_indexes.py`
  - `seed_qdrant.py`, `embedding_compare.py`
  - `batch/process_docs.py` and sample batch input under `batch/input/`

- **qdrant_scripts/** – additional Qdrant administrative operations

- **data/** – seed data and sample datasets
  - `docs-index.seed.jsonl` – initial documents for indexing
  - `pins.json` – example pin or marker data
  - `ourairports/airports.csv` – airport dataset used by tools

- **logs/**, **qdrant_storage/** – runtime artifacts for local development

- **deprecated/** – legacy or experimental code retained for reference

This structure keeps ingestion, retrieval, chat orchestration, and frontend concerns clearly separated while providing dedicated spaces for scripts, tools, and operational data.



## 🧑‍💻 Developer & Operator Utilities (Makefile)

The `Makefile` includes specialized targets essential for debugging, maintenance, and system administration, particularly for the Qdrant vector store. These commands simplify operational tasks by abstracting complex Docker commands and API calls.

### Application Start/Stop

| Target | Description | Usage |
| :--- | :--- | :--- |
| `make start` | **Starts the full Docker Compose stack** (Webapp + Qdrant). (Recommended for general deployment.) | `make start` |
| `make start-hybrid` | Starts the Qdrant container, then runs the FastAPI application in a local **Python virtual environment (venv)**. (Recommended for local development/debugging.) | `make start-hybrid` |
| `make stop` | Stops and removes the full Docker Compose stack. | `make stop` |
| `make stop-hybrid` | Stops the web app and Qdrant container and resources. | `make stop-hybrid` |


### Core Operations

| Target | Description | Usage |
| :--- | :--- | :--- |
| `make seed` | **Ingests sample data** into the current Qdrant collection. Requires the local `venv` to be active. | `make seed` |
| `make smoke_api` | Runs an **OpenAI API smoke test** to verify `OPENAI_API_KEY` authentication and connectivity. | `make smoke_api` |
| `make start-qdrant` | Starts only the **Qdrant vector database** container in detached mode. | `make start-qdrant` |
| `make stop-qdrant` | Stops and removes the Qdrant container and resources. | `make stop-qdrant` |
| `make stop-uvicorn` | Gracefully kills the local running FastAPI application process (SIGTERM) without affecting Qdrant. | `make stop-uvicorn` |

### Qdrant Debugging & Inspection

These targets automatically connect to Qdrant using the configured `QDRANT_HOST` and `QDRANT_PORT` settings, falling back to `localhost:6333` if not specified.

| Target | Description | Usage Example |
| :--- | :--- | :--- |
| `make qdrant-collections` | Lists all collections currently running in Qdrant. | `make qdrant-collections` |
| `make qdrant-info` | Shows concise info (status, dimensions, vector count) for a specific collection. | `make qdrant-info COLLECTION=document_index` |
| `make qdrant-indexes` | Shows field indexes (payload schema) for a collection, useful for checking filters. | `make qdrant-indexes COLLECTION=my_data` |
| `make qdrant-logs` | Streams the logs from the Qdrant container live (`docker compose logs -f qdrant`). | `make qdrant-logs` |


### Maintenance & Data Management

| Target | Description |
| :--- | :--- |


| `make qdrant-backup` | Creates a compressed archive (`.tar.gz`) of the local `qdrant_storage/` bind mount directory. |
| `make my-ip` | Utility to retrieve the current machine's local IP address, useful for connecting to the application from other devices on the same network. |


## 🧱 Qdrant Operations CLI

In addition to the Makefile targets, the repository includes a Python-based Qdrant operations CLI located at `qdrant_scripts/qdrant_ops.py`. This utility provides a simple administrative surface over the active collection and is useful for inspection, backup, and safe maintenance.

Supported operations include:

- **Inspect points and payloads** using filters (e.g., by `source` or `base_url`).
- **List fields and titles** to understand the payload schema and document coverage.
- **Count chunks** for a given base URL to see how many segments a document produced.
- **Export a collection** to JSONL for backup or seeding into another environment.
- **Truncate a collection** while preserving its configuration (distance, vector size, payload schema).
- **Delete points** by id or by payload filter, with interactive confirmation for destructive actions.

Example invocations:

```bash
# List distinct payload fields
python qdrant_scripts/qdrant_ops.py list-fields

# List document titles (with an optional limit)
python qdrant_scripts/qdrant_ops.py list-titles --limit 50

# Count chunks for a specific base URL
python qdrant_scripts/qdrant_ops.py count-chunks --base-url "https://en.wikipedia.org/wiki/Mont_Blanc"

# Export the active collection to a JSONL file under data/
python qdrant_scripts/qdrant_ops.py export -f docs-index-export.jsonl

# Safely truncate the active collection (interactive confirmation)
python qdrant_scripts/qdrant_ops.py truncate
```

This CLI complements the Makefile targets by providing more granular and scriptable control over the Qdrant collection, and it can be extended with additional commands as operational needs evolve.



## ✅ Automated Quality Checks (CI Workflow)

The repository includes a lightweight Continuous Integration (CI) workflow to provide fast feedback on code health without pulling in the full Docker/Qdrant stack.

- **Workflow location:** `.github/workflows/python-ci.yml`
- **Triggers:** Runs on every `push` and `pull_request` to the repository.
- **Environment:** Uses `ubuntu-latest` with Python 3.10.
- **Dependency caching:** Caches the pip directory based on the hash of `requirements.txt` to speed up repeated runs.
- **Checks performed:**
  - Installs dependencies via `pip install -r requirements.txt`.
  - Runs `python -m compileall backend scripts qdrant_scripts` to perform a syntax-level compile of all project Python code.

This CI workflow is intentionally minimal: it validates that dependencies install and that all Python modules compile successfully, while keeping runs fast and avoiding the need to start Docker, Qdrant, or external services. It serves as a basic quality gate and a foundation that teams can extend with additional tests, type checking, or linting as needed.


## 🌐 Browser Compatibility: Secure Context Requirement

If you access the application from **another machine** using an IP address (e.g., `http://192.168.1.10:8000`) certain browsers — especially **Safari 15–16.1** — treat this as a **non‑secure context**.

Some Web APIs such as `crypto.randomUUID()` are available **only in secure contexts** (`https://` or `http://localhost`). When the frontend attempted to generate a `query_id` using:

```js
crypto.randomUUID().slice(0, 8)
```

this caused Safari to throw an error on non-secure IP-based pages, leading to symptoms like:

- The **Send button doing nothing**
- No network calls being triggered
- No error messages appearing

### Fix Implemented
Replaced the direct `crypto.randomUUID()` call with a compatibility-safe fallback:

```js
let queryId;
try {
  if (window.crypto && typeof window.crypto.randomUUID === 'function') {
    queryId = window.crypto.randomUUID().slice(0, 8);
  } else if (window.crypto && window.crypto.getRandomValues) {
    const arr = new Uint32Array(2);
    window.crypto.getRandomValues(arr);
    queryId = (arr[0].toString(16) + arr[1].toString(16)).slice(0, 8);
  } else {
    queryId = Math.random().toString(36).slice(2, 10);
  }
} catch (_) {
  queryId = Math.random().toString(36).slice(2, 10);
}
```

This ensures the chat works on:
- Safari over IP
- Older browsers
- Any environment that is not considered a secure context

### Recommendation for Production
To avoid similar issues for end-users:
- Prefer serving the frontend via **HTTPS**
- Or use a reverse proxy (e.g., nginx, Caddy) with a local certificate

This ensures maximum compatibility of browser APIs.

---

## 🧪 API Examples (Advanced)

<details>
<summary>Click to expand API ingestion examples</summary>

- MediaWiki: `POST /mediawiki/url`
  - Body: `{ "url": "https://en.wikipedia.org/wiki/...", "max_chunks": 0, "force_delete": true }`
  - Notes: `max_chunks > 0` limits chunks to that number; `0` or omitted means no user limit. A hard cap (`MAX_CHUNKS_PER_DOC`) is always enforced.
  - Optional: `?estimate=true` query param to return planned chunk count without indexing.

- Generic URLs/PDFs: `POST /index`
  - Body: `{ "urls": ["https://..."], "doc_type": "HTML" | "PDF", "max_chunks": 0, "force_delete": true, "force_crawl": true }`
  - Behavior: standardize on chunk caps; character-based limits are removed.
  - Optional: `?estimate=true` query param to return planned chunk count without indexing.

- Structured PDF (keep sections/headings like MediaWiki):
  - Single endpoint: `POST /pdf` as multipart form with fields:
    - `file` (UploadFile, optional) or `url` (string, optional)
    - `max_chunks` (int, default 0), `force_delete` (bool, default true)
    - Optional query: `?estimate=true` to return planned chunk count only

Examples:
```bash
curl -X POST http://localhost:8000/mediawiki/url \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://en.wikipedia.org/wiki/OpenAI","max_chunks":50,"force_delete":true}'

curl -X POST http://localhost:8000/index \
  -H 'Content-Type: application/json' \
  -d '{"urls":["https://openai.com"],"doc_type":"HTML","max_chunks":100,"force_delete":true}'

# Estimate only examples
curl -X POST 'http://localhost:8000/mediawiki/url?estimate=true' \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://en.wikipedia.org/wiki/OpenAI","max_chunks":0}'

curl -X POST 'http://localhost:8000/index?estimate=true' \
  -H 'Content-Type: application/json' \
  -d '{"urls":["https://openai.com"],"doc_type":"HTML","max_chunks":0}'

# Structured PDF examples
# Upload a local PDF
curl -X POST 'http://localhost:8000/pdf?estimate=false' \
  -F 'file=@/path/to/file.pdf' \
  -F 'max_chunks=100' \
  -F 'force_delete=true'

# Use a PDF URL, estimate only
curl -X POST 'http://localhost:8000/pdf?estimate=true' \
  -F 'url=https://example.com/file.pdf' \
  -F 'max_chunks=0'
```

</details>

© 2025 Rajkumar Velliavitil — All Rights Reserved

## 📜 License & Usage

This project is **source-available** for **personal, educational, and evaluation purposes**.  
It is permitted to **run, modify, and fork** the code for non-commercial use.

**Redistribution, sublicensing, or commercial use** of this project or derivative works **requires explicit written permission** from the author.

