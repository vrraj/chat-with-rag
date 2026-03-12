[← Back to Chat with RAG Home](https://vrraj.github.io/chat-with-rag)

# Deployment Guide

> **About this document**
>
> This page covers **tested deployment scenarios** for the *Chat-with-RAG* system, including Docker setup, production considerations, and security best practices.
>
> **Note:** If you landed here directly (for example from documentation hosting or search), start with the repository **[README](https://github.com/vrraj/chat-with-rag/blob/main/README.md)** to see how to run the system locally and try the interactive demo.

## Table of Contents

- [Deployment Overview](#deployment-overview)
- [Docker Deployment (Supported)](#docker-deployment-supported)
- [Production Considerations](#production-considerations)
- [Security Best Practices](#security-best-practices)
- [Backup and Maintenance](#backup-and-maintenance)

---

## Deployment Overview

The chat-with-rag system consists of several components:

### Core Components

- **Web Application** (FastAPI backend + frontend)
- **Vector Database** (Qdrant)
- **LLM Providers** (OpenAI, Gemini, or custom)
- **Optional External Services** (weather, airports, web search)

### Currently Supported Deployment Scenarios

1. **Local Development** - Docker Compose on single machine ✅
2. **Single-Node Docker** - All services on one host ✅
3. **Basic Production** - Docker with environment variables ✅

### Not Yet Tested/Documented

- **Multi-Node deployments** (separate app/database servers)
- **Cloud platforms** (AWS, Google Cloud, Azure)
- **Load balancers** (nginx, HAProxy)
- **Container orchestration** (Kubernetes)
- **Managed cloud services**

> **Note**: The system is designed to be cloud-compatible, but specific cloud deployment guides are not yet tested or documented. Contributions welcome!

---

## Docker Deployment (Supported)

### Quick Start (Development)

```bash
# Clone and setup
git clone https://github.com/vrraj/chat-with-rag.git
cd chat-with-rag

# Run the bootstrap script
bash scripts/rag_setup.sh
```

### Manual Docker Setup

```bash
# Create environment file
cp .env.example .env
# Edit .env with your API keys

# Start services
make start

# Seed sample data
make seed
```

### Production Docker Compose

For production, create a `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
    depends_on:
      - qdrant
    restart: unless-stopped
    volumes:
      - ./logs:/app/logs

  qdrant:
    image: qdrant/qdrant:v1.14.1
    ports:
      - "6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage
    restart: unless-stopped

volumes:
  qdrant_storage:
```

---

## Production Considerations

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here

# Optional
QDRANT_HOST=localhost
QDRANT_PORT=6333
DEBUG=false
LOG_LEVEL=INFO
```

### Resource Requirements

**Minimum (Development/Small Dataset):**
- CPU: 2 cores
- Memory: 4GB RAM
- Storage: 1GB (for Qdrant data + models)

**Recommended (Production/Large Dataset):**
- CPU: 4 cores
- Memory: 8GB RAM
- Storage: 10GB SSD (for large knowledge bases)

**Storage Scaling:**
- Small datasets (~100 documents): ~100MB storage
- Medium datasets (~1K documents): ~500MB storage  
- Large datasets (~10K documents): ~2-5GB storage

### Health Checks

```bash
# Check application health
curl http://localhost:8000/health

# Check Qdrant health
curl http://localhost:6333/health
```

---

## Security Best Practices

### API Key Management

```bash
# Use environment variables, never commit keys
echo ".env" >> .gitignore

# In production, use secret management:
# - Docker secrets
# - Environment variable injection
# - Cloud secret managers (when supported)
```

### Network Security

```bash
# Use Docker networks for internal communication
docker network create app-network

# Don't expose Qdrant to public internet
# Only expose the web application (port 8000)
```

### SSL/TLS

For production, consider:
- Reverse proxy with SSL termination
- Let's Encrypt certificates
- Internal network encryption

---

## Backup and Maintenance

### Data Backup

```bash
# Backup Qdrant data
docker exec qdrant-server tar -czf /tmp/backup.tar.gz -C /qdrant/storage .
docker cp qdrant-server:/tmp/backup.tar.gz ./backups/

# Backup configuration
tar -czf config_backup.tar.gz .env docker-compose.prod.yml
```

### Log Management

```bash
# Configure logging in production
LOG_LEVEL=INFO
LOG_FILE=/app/logs/app.log

# Rotate logs
logrotate -f /etc/logrotate.d/chat-with-rag
```

### Updates

```bash
# Update application
git pull origin main
docker compose down
docker compose build
docker compose up -d

# Re-seed if needed (note: overwrites document_index and document_index_gemini collections)
make seed
```

---

## Future Deployment Options

The following deployment scenarios are architecturally possible but not yet documented:

- **Cloud Platforms**: AWS ECS, Google Cloud Run, Azure Container Instances
- **Load Balancing**: nginx, HAProxy, cloud load balancers
- **Container Orchestration**: Kubernetes, Docker Swarm
- **Managed Services**: Qdrant Cloud, managed databases

> **Contributions**: If you successfully deploy to any of these platforms, please consider contributing deployment documentation!

---

## Troubleshooting

### Common Issues

1. **Qdrant connection failed**
   - Check if Qdrant is running: `docker ps | grep qdrant`
   - Verify network connectivity

2. **API key errors**
   - Ensure keys are properly set in environment
   - Check for typos in key names

3. **Memory issues**
   - Increase available RAM
   - Monitor Qdrant memory usage

### Debug Commands

```bash
# Check container logs
docker logs chat-with-rag_app_1
docker logs chat-with-rag_qdrant_1

# Check resource usage
docker stats

# Test API endpoints
curl http://localhost:8000/api/config
```
