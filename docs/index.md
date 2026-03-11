---
layout: default
title: "Chat with RAG"
description: "Advanced RAG system with multi-provider LLM support and streaming capabilities"
---

# Chat with RAG

 This page serves as the **main documentation index** for the *Chat-with-RAG* system, providing an overview and navigation to all available documentation.


<p align="left">
  <a href="https://github.com/vrraj/chat-with-rag">
    <img src="https://img.shields.io/github/stars/vrraj/chat-with-rag?style=social" alt="GitHub Stars">
  </a>
  <a href="https://github.com/vrraj/chat-with-rag/releases">
    <img src="https://img.shields.io/github/v/release/vrraj/chat-with-rag?label=github%20release&color=orange&logo=github" alt="GitHub Release">
  </a>
  <a href="https://github.com/vrraj/chat-with-rag/actions">
    <img src="https://github.com/vrraj/chat-with-rag/actions/workflows/ci.yml/badge.svg" alt="CI Status">
  </a>
</p>

An advanced Retrieval-Augmented Generation (RAG) system with multi-provider LLM support, streaming capabilities, and comprehensive documentation.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/vrraj/chat-with-rag.git
cd chat-with-rag

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Start the application
python start.py
```

## Documentation

- **[Main README](../README.md)** - Project overview, features, and complete documentation guide
- **[API Reference](api-reference.md)** - Complete API documentation
- **[Configuration Reference](configuration.md)** - All configuration options and settings
- **[Troubleshooting Guide](troubleshooting.md)** - Common issues and solutions
- **[Development Guide](development.md)** - Development setup and workflows
- **[Deployment Guide](deployment.md)** - Production deployment strategies
- **[Embedded Chat Guide](embedded-chat.md)** - Embeddable chat UI configuration and usage
- **[Server-Sent Events](server-sent-events.md)** - Real-time streaming implementation
- **[Technical Overview](technical-overview.md)** - Architecture and design
- **[Attributions](attributions.md)** - Credits and licenses

## Features

- **Multi-Provider Support**: OpenAI, Gemini, and custom LLM providers
- **Streaming Responses**: Real-time response streaming with Server-Sent Events
- **Advanced RAG**: Context-aware retrieval and generation
- **Flexible Embeddings**: Multiple embedding providers and configurations
- **Comprehensive API**: RESTful API with full documentation
- **Docker Support**: Containerized deployment options

## Architecture

The system consists of:
- **Backend**: FastAPI-based REST API with streaming support
- **Frontend**: React-based web interface
- **Vector Store**: Qdrant for efficient similarity search
- **LLM Integration**: Multi-provider abstraction layer

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [Attributions](attributions.md) for details.
