# Deployment Guide

This guide covers deployment strategies for the chat-with-rag system in various environments.

## Table of Contents

- [Deployment Overview](#deployment-overview)
- [Docker Deployment](#docker-deployment)
- [Production Deployment](#production-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Security Considerations](#security-considerations)
- [Monitoring & Maintenance](#monitoring--maintenance)
- [Scaling Strategies](#scaling-strategies)

---

## Deployment Overview

The chat-with-rag system consists of several components:

### Core Components

- **Web Application** (FastAPI backend + frontend)
- **Vector Database** (Qdrant)
- **LLM Providers** (OpenAI, Gemini, or custom)
- **Optional External Services** (weather, airports, web search)

### Deployment Architectures

1. **Single-Node Docker** - All services on one host
2. **Multi-Node** - Separate application and database servers  
3. **Cloud-Native** - Container orchestration with managed services
4. **Hybrid** - On-premise app with cloud LLM services

---

## Docker Deployment

### Quick Start (Development)

```bash
# Clone and setup
git clone https://github.com/vrraj/chat-with-rag.git
cd chat-with-rag

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start services
make start

# Seed sample data
make seed
```

### Production Docker Setup

#### 1. Docker Compose Production

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - HOST=0.0.0.0
      - PORT=8000
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
    env_file:
      - .env.prod
    depends_on:
      - qdrant
    restart: unless-stopped
    volumes:
      - ./logs:/app/logs
    networks:
      - app-network

  qdrant:
    image: qdrant/qdrant:v1.14.1
    ports:
      - "6333:6333"
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
    volumes:
      - qdrant_data:/qdrant/storage
    restart: unless-stopped
    networks:
      - app-network

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    restart: unless-stopped
    networks:
      - app-network

volumes:
  qdrant_data:

networks:
  app-network:
    driver: bridge
```

#### 2. Production Environment

Create `.env.prod`:

```bash
# Server Configuration
HOST=0.0.0.0
PORT=8000
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com

# Database
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# API Keys (use secure storage in production)
OPENAI_API_KEY=${OPENAI_API_KEY}
GEMINI_API_KEY=${GEMINI_API_KEY}

# Production Settings
DEBUG_VERBOSE=false
SHOW_PROCESSING_STEPS=false

# Logging
LOG_LEVEL=INFO
LOG_FILE=/app/logs/app.log
```

#### 3. Nginx Configuration

Create `nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream app {
        server app:8000;
    }

    # HTTP redirect to HTTPS
    server {
        listen 80;
        server_name yourdomain.com www.yourdomain.com;
        return 301 https://$server_name$request_uri;
    }

    # HTTPS server
    server {
        listen 443 ssl http2;
        server_name yourdomain.com www.yourdomain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";

        location / {
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # SSE support
            proxy_buffering off;
            proxy_cache off;
            proxy_set_header Connection '';
            proxy_http_version 1.1;
            chunked_transfer_encoding off;
        }

        # Static file caching
        location /static/ {
            proxy_pass http://app;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
}
```

#### 4. Deploy Commands

```bash
# Build and start production services
docker-compose -f docker-compose.prod.yml up -d

# Check status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f app

# Stop services
docker-compose -f docker-compose.prod.yml down
```

---

## Production Deployment

### System Requirements

#### Minimum Specifications

- **CPU:** 4 cores
- **Memory:** 8GB RAM
- **Storage:** 50GB SSD
- **Network:** 100 Mbps

#### Recommended Specifications

- **CPU:** 8 cores  
- **Memory:** 16GB RAM
- **Storage:** 100GB SSD
- **Network:** 1 Gbps

### Operating System

- **Linux:** Ubuntu 20.04+ / CentOS 8+ / RHEL 8+
- **Docker:** 20.10+
- **Docker Compose:** 2.0+

### Database Sizing

#### Qdrant Storage Requirements

```bash
# Estimate storage needs
# ~1KB per chunk + metadata
# 1000 documents × 50 chunks = 50MB base
# Vector data: 4 bytes per dimension × 1536 dimensions = 6KB per chunk
# Total: ~350MB per 1000 documents

# Plan for 3-5x growth for safety
```

#### Memory Requirements

- **Qdrant:** 2-4GB for 100K vectors
- **Application:** 1-2GB base + processing
- **OS & Cache:** 2-4GB

---

## Cloud Deployment

### AWS Deployment

#### 1. ECS (Elastic Container Service)

```yaml
# ecs-task-definition.json
{
  "family": "chat-with-rag",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "app",
      "image": "your-account.dkr.ecr.region.amazonaws.com/chat-with-rag:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "QDRANT_HOST",
          "value": "qdrant-cluster.region.amazonaws.com"
        }
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:openai-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/chat-with-rag",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### 2. Qdrant Cloud

```bash
# Using Qdrant Cloud managed service
# Set environment variables:
QDRANT_HOST=your-cluster.qdrant.io
QDRANT_PORT=6333
QDRANT_API_KEY=your-api-key
```

#### 3. Load Balancer Setup

```bash
# Application Load Balancer
aws elbv2 create-load-balancer \
  --name chat-with-rag-alb \
  --subnets subnet-12345 subnet-67890 \
  --security-groups sg-12345

# Target group
aws elbv2 create-target-group \
  --name chat-with-rag-tg \
  --protocol HTTP \
  --port 8000 \
  --target-type ip \
  --vpc-id vpc-12345

# Listener
aws elbv2 create-listener \
  --load-balancer-arn lb-arn \
  --protocol HTTP \
  --port 80 \
  --default-actions Type=forward,TargetGroupArn=tg-arn
```

### Google Cloud Platform

#### 1. Cloud Run Deployment

```bash
# Build and push image
gcloud builds submit --tag gcr.io/project-id/chat-with-rag

# Deploy to Cloud Run
gcloud run deploy chat-with-rag \
  --image gcr.io/project-id/chat-with-rag \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars QDRANT_HOST=qdrant-service \
  --set-secrets OPENAI_API_KEY=openai-key:latest
```

#### 2. Cloud Memorystore for Qdrant

```bash
# Or use managed Qdrant service
# Deploy Qdrant on GKE
kubectl apply -f qdrant-deployment.yaml
```

### Azure Deployment

#### 1. Container Instances

```bash
# Create container group
az container create \
  --resource-group chat-with-rag-rg \
  --name chat-with-rag \
  --image your-registry/chat-with-rag:latest \
  --dns-name-label chat-with-rag-unique \
  --ports 8000 \
  --environment-variables QDRANT_HOST=qdrant-service \
  --secure-environment-variables OPENAI_API_KEY=$OPENAI_API_KEY
```

---

## Security Considerations

### API Key Management

#### Environment Variables

```bash
# Use secret management in production
# AWS: Parameter Store / Secrets Manager
# GCP: Secret Manager  
# Azure: Key Vault

# Never commit API keys to git
echo ".env" >> .gitignore
echo "*.key" >> .gitignore
```

#### Docker Secrets

```yaml
# docker-compose.prod.yml
services:
  app:
    secrets:
      - openai_api_key
      - gemini_api_key
    environment:
      - OPENAI_API_KEY_FILE=/run/secrets/openai_api_key

secrets:
  openai_api_key:
    file: ./secrets/openai.key
  gemini_api_key:
    file: ./secrets/gemini.key
```

### Network Security

#### Firewall Rules

```bash
# Only allow necessary ports
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw enable
```

#### SSL/TLS Configuration

```bash
# Use Let's Encrypt for SSL
certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Auto-renewal
crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### Application Security

#### CORS Configuration

```python
# In .env
ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com

# Restrict to specific domains in production
```

#### Rate Limiting

```python
# Implement rate limiting (example with slowapi)
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/chat")
@limiter.limit("10/minute")  # 10 requests per minute per IP
async def chat_endpoint(request: Request, chat_request: ChatRequest):
    # ... endpoint logic
```

#### Input Validation

```python
# Validate all inputs
from pydantic import BaseModel, validator

class ChatRequest(BaseModel):
    message: str
    history: List[Dict] = []
    
    @validator('message')
    def validate_message(cls, v):
        if len(v) > 10000:  # Reasonable limit
            raise ValueError('Message too long')
        return v
```

---

## Monitoring & Maintenance

### Health Checks

#### Application Health

```python
# Add to backend/main.py
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Qdrant connection
        qdrant_client.get_collections()
        return {"status": "healthy", "timestamp": datetime.utcnow()}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

#### Docker Health Check

```dockerfile
# Dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

### Logging

#### Structured Logging

```python
import structlog

logger = structlog.get_logger()

@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest):
    logger.info("chat_request", 
                user_id=get_user_id(),
                message_length=len(chat_request.message))
    # ... process request
```

#### Log Aggregation

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  app:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
  
  # Add log aggregation
  loki:
    image: grafana/loki:latest
    ports:
      - "3100:3100"
    volumes:
      - ./loki-config.yml:/etc/loki/local-config.yaml

  promtail:
    image: grafana/promtail:latest
    volumes:
      - /var/log:/var/log
      - ./promtail-config.yml:/etc/promtail/config.yml
```

### Monitoring Metrics

#### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, generate_latest

# Define metrics
chat_requests = Counter('chat_requests_total', 'Total chat requests')
chat_duration = Histogram('chat_duration_seconds', 'Chat processing time')

@app.post("/chat")
@chat_duration.time()
async def chat_endpoint(chat_request: ChatRequest):
    chat_requests.inc()
    # ... process request
```

#### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Chat with RAG Metrics",
    "panels": [
      {
        "title": "Chat Requests",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(chat_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(chat_duration_seconds_bucket[5m]))"
          }
        ]
      }
    ]
  }
}
```

### Backup Strategy

#### Qdrant Backup

```bash
#!/bin/bash
# backup-qdrant.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/qdrant"

# Create backup
docker exec qdrant_container \
  qdrant-snapshot --snapshot-path /qdrant/snapshots/$DATE

# Copy to backup location
docker cp qdrant_container:/qdrant/snapshots/$DATE $BACKUP_DIR/

# Clean old backups (keep 7 days)
find $BACKUP_DIR -name "*" -type d -mtime +7 -exec rm -rf {} \;
```

#### Configuration Backup

```bash
#!/bin/bash
# backup-config.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/config"

# Backup configuration
tar -czf $BACKUP_DIR/config_$DATE.tar.gz \
  .env \
  docker-compose.prod.yml \
  nginx.conf \
  prompts/

# Backup database schema
python scripts/export_collection.py --output $BACKUP_DIR/schema_$DATE.json
```

---

## Scaling Strategies

### Horizontal Scaling

#### Load Balancer Setup

```nginx
# nginx.conf upstream
upstream app_servers {
    server app1:8000 weight=1;
    server app2:8000 weight=1;
    server app3:8000 weight=1;
}
```

#### Session Management

```python
# Use Redis for session storage
import redis

redis_client = redis.Redis(host='redis', port=6379, db=0)

class RedisSessionManager:
    def __init__(self):
        self.redis = redis_client
    
    def get_session(self, session_id):
        data = self.redis.get(f"session:{session_id}")
        return json.loads(data) if data else None
    
    def set_session(self, session_id, data):
        self.redis.setex(f"session:{session_id}", 3600, json.dumps(data))
```

### Database Scaling

#### Qdrant Clustering

```yaml
# qdrant-cluster.yml
version: '3.8'
services:
  qdrant-node1:
    image: qdrant/qdrant:v1.14.1
    environment:
      - QDRANT__SERVICE__URI=http://qdrant-node1:6333
      - QDRANT__CLUSTER__ENABLED=true
      - QDRANT__CLUSTER__PEER_ADDRESS=qdrant-node1:6334
    ports:
      - "6333:6333"
      - "6334:6334"
  
  qdrant-node2:
    image: qdrant/qdrant:v1.14.1
    environment:
      - QDRANT__SERVICE__URI=http://qdrant-node2:6333
      - QDRANT__CLUSTER__ENABLED=true
      - QDRANT__CLUSTER__PEER_ADDRESS=qdrant-node2:6334
      - QDRANT__CLUSTER__URI=http://qdrant-node1:6334
    ports:
      - "6335:6333"
      - "6336:6334"
```

### Caching Strategy

#### Redis Caching

```python
import redis
from functools import wraps

redis_client = redis.Redis(host='redis', port=6379, db=0)

def cache_result(expiration=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try cache first
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            redis_client.setex(cache_key, expiration, json.dumps(result))
            return result
        return wrapper
    return decorator

@cache_result(expiration=1800)  # 30 minutes
def search_similar(query, top_k=8):
    # ... search logic
    pass
```

### Auto-scaling

#### Kubernetes HPA

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: chat-with-rag-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: chat-with-rag
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## Deployment Checklist

### Pre-deployment

- [ ] Environment variables configured
- [ ] SSL certificates obtained
- [ ] DNS records configured
- [ ] Firewall rules set
- [ ] Backup strategy in place
- [ ] Monitoring configured
- [ ] Load testing completed

### Post-deployment

- [ ] Health checks passing
- [ ] SSL certificates valid
- [ ] Monitoring alerts configured
- [ ] Backup jobs scheduled
- [ ] Log rotation configured
- [ ] Performance baselines established

### Ongoing Maintenance

- [ ] Regular security updates
- [ ] Monitor API usage and costs
- [ ] Review and optimize queries
- [ ] Update documentation
- [ ] Test disaster recovery procedures

---

## Troubleshooting Deployment

### Common Issues

1. **Container won't start**
   - Check environment variables
   - Verify Docker logs
   - Ensure ports are available

2. **Database connection failed**
   - Check Qdrant service status
   - Verify network connectivity
   - Check authentication

3. **High memory usage**
   - Monitor container resources
   - Check for memory leaks
   - Optimize batch sizes

4. **Slow responses**
   - Check resource utilization
   - Monitor network latency
   - Review query performance

### Debug Commands

```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs -f app

# Check resource usage
docker stats

# Test connectivity
curl -f http://localhost:8000/health

# Check database
curl http://localhost:6333/collections
```

This deployment guide provides comprehensive coverage for deploying chat-with-rag in various environments, from development to production scale.
