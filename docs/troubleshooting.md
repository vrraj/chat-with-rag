# Troubleshooting Guide

This guide covers common issues, errors, and solutions for the chat-with-rag system.

## Table of Contents

- [Environment Setup Issues](#environment-setup-issues)
- [Docker & Infrastructure Issues](#docker--infrastructure-issues)
- [API & Connection Issues](#api--connection-issues)
- [Embedding & Indexing Issues](#embedding--indexing-issues)
- [Chat & Retrieval Issues](#chat--retrieval-issues)
- [Performance Issues](#performance-issues)
- [Configuration Issues](#configuration-issues)

---

## Environment Setup Issues

### Python Environment Issues

**Issue:** `ModuleNotFoundError` or `ImportError` after installation

**Solutions:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt

# Verify installation
python -c "import backend.main; print('Import successful')"
```

**Issue:** SSL Certificate errors on macOS

**Solution:**
```bash
# Install certificates for Python
open "/Applications/Python 3.12/Install Certificates.command"
```

### API Key Issues

**Issue:** `401 Unauthorized` or `Invalid API key` errors

**Solutions:**
1. Verify API key in `.env` file:
   ```bash
   cat .env | grep API_KEY
   ```

2. Check API key format and permissions:
   - OpenAI: Ensure key has appropriate permissions and budget limits
   - Gemini: Verify quota limits in Google Cloud Console

3. Test API key manually:
   ```bash
   # OpenAI test
   curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models
   
   # Gemini test  
   curl -H "x-goog-api-key: $GEMINI_API_KEY" https://generativelanguage.googleapis.com/v1/models
   ```

---

## Docker & Infrastructure Issues

### Docker Desktop Not Running

**Issue:** `Cannot connect to the Docker daemon`

**Solutions:**
1. Start Docker Desktop manually
2. Verify Docker status:
   ```bash
   docker --version
   docker compose version
   ```

3. On Linux, add user to docker group:
   ```bash
   sudo usermod -aG docker $USER
   # Log out and back in
   ```

### Qdrant Connection Issues

**Issue:** `Connection refused` to Qdrant

**Solutions:**
1. Check if Qdrant container is running:
   ```bash
   docker ps | grep qdrant
   ```

2. Restart services:
   ```bash
   make stop
   make start
   ```

3. Check Qdrant logs:
   ```bash
   docker logs qdrant_container_name
   ```

### Port Conflicts

**Issue:** Port 8000 or 6333 already in use

**Solutions:**
1. Find process using the port:
   ```bash
   lsof -i :8000  # or :6333 for Qdrant
   ```

2. Kill the process or change ports in `docker-compose.yml`

---

## API & Connection Issues

### CORS Issues

**Issue:** `CORS policy` errors in browser

**Solutions:**
1. Verify `ALLOWED_ORIGINS` in `.env`:
   ```bash
   echo "ALLOWED_ORIGINS=http://localhost:8000,http://127.0.0.1:8000" >> .env
   ```

2. Check origin enforcement in backend logs

### SSE Streaming Issues

**Issue:** Streaming not working or connection drops

**Solutions:**
1. Check if `show_processing_steps` is enabled
2. Verify browser supports SSE (most modern browsers do)
3. Check network tab in browser dev tools for connection status
4. Ensure no proxy/firewall blocking SSE connections

### Timeout Issues

**Issue:** Requests timing out during processing

**Solutions:**
1. Increase timeout in client:
   ```python
   requests.post(url, json=payload, timeout=120)  # Increase from default 30s
   ```

2. Check if processing is actually slow (check logs)
3. Consider reducing `top_k` or enabling `estimate` mode first

---

## Embedding & Indexing Issues

### Embedding Dimension Mismatch

**Issue:** `Dimension mismatch` errors during indexing

**Solutions:**
1. Check embedding model configuration in `backend/core/config.py`
2. Ensure all documents use same embedding model
3. Clear and re-index if changing models:
   ```bash
   python scripts/qdrant_scripts/qdrant_ops.py --delete-collection document_index
   ```

### Rate Limiting

**Issue:** `Rate limit exceeded` from embedding providers

**Solutions:**
1. Reduce `embedding_batch_size` in config
2. Add delays between batches
3. Check provider rate limits and upgrade plan if needed

### Memory Issues During Indexing

**Issue:** Out of memory errors with large documents

**Solutions:**
1. Reduce `chunk_size` in configuration
2. Process documents in smaller batches
3. Enable `estimate` mode first to preview costs

---

## Chat & Retrieval Issues

### No Results Found

**Issue:** Empty results from retrieval

**Solutions:**
1. Lower `score_threshold` (try 0.2-0.3)
2. Increase `top_k` (try 10-20)
3. Check if documents are actually indexed:
   ```bash
   python scripts/qdrant_scripts/qdrant_ops.py --list-titles --limit 10
   ```

4. Verify collection name matches indexed data

### Poor Quality Results

**Issue:** Retrieved documents not relevant

**Solutions:**
1. Enable query rewrite: `"enable_query_rewrite": true`
2. Adjust `rewrite_confidence_threshold` (try 0.6-0.8)
3. Check document chunking strategy
4. Consider re-indexing with different chunk size

### Tool Calling Issues

**Issue:** Tools not being called or failing

**Solutions:**
1. Ensure `"use_tools": true` in params
2. Check tool configurations in backend
3. Verify API keys for external services (weather, airports)
4. Check tool execution logs

---

## Performance Issues

### Slow Response Times

**Issue:** Chat responses taking too long

**Solutions:**
1. Enable `estimate` mode to preview costs/time
2. Reduce `top_k` and `max_output_tokens`
3. Use faster models for non-critical stages
4. Enable caching where possible

### High Memory Usage

**Issue:** System running out of memory

**Solutions:**
1. Reduce batch sizes in configuration
2. Process documents sequentially instead of in parallel
3. Monitor memory usage during indexing
4. Consider using smaller models

---

## Configuration Issues

### Invalid Configuration

**Issue:** Application fails to start with config errors

**Solutions:**
1. Check configuration syntax:
   ```python
   python -c "from backend.core.config import settings; print('Config OK')"
   ```

2. Validate environment variables in `.env`
3. Check for typos in configuration keys

### Domain/Collection Issues

**Issue:** Wrong collection being used

**Solutions:**
1. Verify `active_domain` in `backend/core/config.py`
2. Check `DOMAIN_EMBEDDING_CONFIG` mappings
3. Ensure collection exists and has correct dimensions

---

## Getting Help

### Debug Mode

Enable debug logging for more detailed information:

```bash
# Add to .env
DEBUG_VERBOSE=true
DEBUG_LOG_KEYS=true
PROMPT_REGISTRY_LOG_FULL=1
```

### Log Locations

- Application logs: Console output or configured log file
- Docker logs: `docker logs <container_name>`
- Qdrant logs: `docker logs qdrant_container`

### Common Debug Commands

```bash
# Check indexed documents
python scripts/qdrant_scripts/qdrant_ops.py --list-titles --limit 10

# Test API connection
python scripts/api_smoke_test_openai.py

# Check collections
python scripts/qdrant_scripts/qdrant_ops.py --list-collections

# Test embedding generation
python scripts/embedding_compare.py
```

### When to Ask for Help

If you've tried the above solutions and still have issues:

1. Check the [GitHub Issues](https://github.com/vrraj/chat-with-rag/issues) for similar problems
2. Create a new issue with:
   - Error messages (full stack traces)
   - Configuration details (remove sensitive info)
   - Steps to reproduce
   - System information (OS, Docker version, etc.)

---

## Prevention Tips

1. **Always test with estimate mode first** before large indexing operations
2. **Monitor API usage and costs** regularly
3. **Keep backups of important configurations**
4. **Document custom configurations** for your specific use case
5. **Regular updates** to stay current with fixes and improvements
