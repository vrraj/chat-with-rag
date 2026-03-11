---
layout: default
title: "Chat with RAG | Tool-Assisted Multi-Provider RAG Framework"
description: "Chat with RAG is a modular, tool-assisted Retrieval-Augmented Generation framework with multi-provider LLM support, configurable pipelines, embeddable chat, pipeline observability events, domain-aware prompts, and experiment‑oriented observability."
---

# Chat with RAG

<p align="left">
  <a href="https://github.com/vrraj/chat-with-rag">
    <img src="https://img.shields.io/github/stars/vrraj/chat-with-rag?style=social" alt="GitHub Stars">
  </a>
  <a href="https://github.com/vrraj/chat-with-rag/releases">
    <img src="https://img.shields.io/github/v/release/vrraj/chat-with-rag?label=github%20release&color=orange&logo=github" alt="GitHub Release">
  </a>
  <a href="https://github.com/vrraj/chat-with-rag/actions">
    <img src="https://img.shields.io/github/actions/workflow/status/vrraj/chat-with-rag/python-app.yml?label=CI&logo=github" alt="CI Status">
  </a>
</p>

A modular **Python framework for building Retrieval-Augmented Generation (RAG) systems**.
Chat with RAG is a modular framework and experimentation environment for building, evaluating, and running configurable RAG systems.

The project goes beyond a basic vector-search demo. It provides a **multi-stage RAG pipeline** with configurable retrieval, prompts, models, tools, streaming, and observability—designed for experimentation and API-driven workflows.


## Key Capabilities

Chat with RAG is a reference framework for exploring how configurable, modular RAG systems can be built.

- **Multi‑Provider LLM Support** – Works with OpenAI, Gemini, and additional providers through the `vrraj-llm-adapter`, enabling unified model calls and provider abstraction.
- **Configurable Multi‑Stage RAG Pipeline** – Query rewrite → retrieval → rerank → context assembly → inference → tool execution → response synthesis → post‑processing.
- **Document Ingestion Pipeline** – Import documents or URLs, parse and normalize content, chunk text, generate embeddings, and store vectors in the vector database for retrieval.
- **Registry‑Driven Model and Prompt Management** – Central registries control model selection, parameters, and domain‑specific prompt templates.
- **Embeddable Chat Interface** – Drop‑in web chat widget that can be embedded in external sites with domain routing and configuration.
- **Stateful and Stateless Chat Modes** – Supports both persistent chat sessions and API‑driven request workflows.
- **Advanced Context Management** – Combines summarized history with recent turns to maintain long conversations without exceeding model context limits.
- **Tool‑Assisted Generation** – Optional tool execution during inference for agent‑style workflows.
- **Observability and Cost Awareness** – Pipeline stage events and usage tracking provide visibility into how queries move through the RAG pipeline.
- **Secure Domain‑Based Access Controls** – API and widget access can be restricted to configured domains and hosts.

These capabilities make the project useful for experimentation, learning, and prototyping RAG-based applications.

## High-Level Pipelines

The system is organized around two primary pipelines: **document ingestion** and **chat orchestration**.

| Pipeline | Flow |
|---|---|
| **Ingestion** | `Documents / URLs` → `Load Sources` → `Extract & Parse` → `Chunk & Normalize` → `Metadata Augmentation` → `Embeddings` → `Vector Storage` |
| **Chat** | `User Prompt` → `Query Rewrite` → `Retrieval` → `Rerank` → `Context Assembly` → `LLM Inference` → `Tool Execution` → `Response Synthesis` → `Post-Processing` → `Final Response` |


## Quick Start

Clone the repository and run the setup script:

```bash
git clone https://github.com/vrraj/chat-with-rag.git
cd chat-with-rag
bash scripts/rag_setup.sh
```

Add your **OpenAI** or **Gemini** API key to the `.env` file and start the application.

👉 http://localhost:8000

For the complete setup and configuration steps, see **Getting Started in the README**:

[Getting Started](../README.md#-getting-started)

## Use Cases

Chat with RAG can support several AI application patterns:

- **Knowledge Assistants** – Answer questions using internal documents and curated knowledge bases.
- **Document‑Grounded Support** – Retrieve information from product docs, policies, or technical documentation.
- **Embeddable Website Assistants** – Add contextual chat to websites or documentation portals.
- **Internal Knowledge Search** – Conversational access to engineering docs, playbooks, or operational procedures.
- **Research and Experimentation** – Compare models, prompts, and retrieval strategies.
- **API‑Driven RAG Services** – Integrate retrieval‑augmented responses into other applications.
- **Tool‑Augmented Assistants** – Combine RAG responses with external tools or APIs.

## Application Interfaces

Chat with RAG provides three primary interfaces for different use cases:

- **Web Application** – Interactive interface for exploring and testing the RAG pipeline
- **Embeddable Chat Interface** – Popup or iframe widget for external websites
- **API Access** – Stateful and stateless endpoints for programmatic integration

Through the web UI you can:

- Experiment with **different LLM models** used by the pipeline
- Compare responses across **different prompt templates and model configurations**
- Observe how **retrieval settings** (for example *top‑k results* and similarity scores) affect answers
- Understand how **generation parameters** such as *temperature*, *top‑p*, and *token limits* influence responses
- Explore how the **retrieval → rerank → context assembly → inference pipeline** produces grounded responses with citations

The interface can also be **deployed on a server and accessed by multiple users**, making it useful for experimentation and collaborative testing.

## Explore the Project

- [GitHub Repository](https://github.com/vrraj/chat-with-rag)
- [Getting Started in README](../README.md#-getting-started)
- [Release Notes 2.0](../Release_Notes_2.0.md)

## Documentation

- [Full Documentation (README)](../README.md) - Complete project overview and setup guide
- [API Reference](api-reference.md) - REST API documentation and integration examples
- [Configuration Reference](configuration.md) - Configuration options and settings
- [Technical Overview](technical-overview.md) - Architecture, pipelines, and design patterns
- [Development Guide](development.md) - Contributing, local setup, and workflows
- [Deployment Guide](deployment.md) - Production deployment strategies and best practices
- [Embedded Chat Guide](embedded-chat.md) - Embeddable chat UI configuration and usage
- [Server-Sent Events](server-sent-events.md) - Real-time streaming implementation
- [Troubleshooting Guide](troubleshooting.md) - Common issues and solutions
- [Attributions](attributions.md) - Credits, licenses, and third-party components
