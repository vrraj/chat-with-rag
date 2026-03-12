[← Back to Chat with RAG Home](https://vrraj.github.io/chat-with-rag)

# Troubleshooting Guide

> **About this document**
>
> This page provides **troubleshooting guidance** for the *Chat-with-RAG* system, including common issues, error solutions, and debugging steps.
>
> **Note:** If you landed here directly (for example from documentation hosting or search), start with the repository **[README](https://github.com/vrraj/chat-with-rag/blob/main/README.md)** to see how to run the system locally and try the interactive demo.

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

### Container Updates After Code Changes

**Issue:** Services not reflecting recent code changes after git pull

**Solution:**
If you've pulled new changes, rebuild the containers to pick up service updates:

```bash
docker compose up --build
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

**Issue:** `Dimension mismatch` errors during indexing or retrieval

**Solutions:**
1. Check embedding model configuration in `backend/core/config.py`
2. Ensure all documents use same embedding model
3. Verify domain configuration matches collection:
   ```python
   # Check active domain and corresponding model
   from backend.core.config import settings
   print(f"Domain: {settings.active_domain}")
   print(f"Model: {settings.embedding_model_key}")
   print(f"Collection: {settings.collection_name}")
   ```
4. Clear and re-index if changing models:
   ```bash
   python scripts/qdrant_scripts/qdrant_ops.py --delete-collection document_index
   make seed
   ```

### Gemini Embedding Normalization

**Issue:** Poor retrieval quality with Gemini embeddings

**Solutions:**
1. Ensure `gemini_embedding_normalize=true` in configuration
2. Test normalization consistency:
   ```bash
   python scripts/test_gemini_normalization.py
   ```
3. Compare OpenAI vs Gemini embeddings:
   ```bash
   python scripts/test_gemini_embed_retrieval.py
   ```
4. Consider using OpenAI embeddings if issues persist

### Rate Limiting

**Issue:** `Rate limit exceeded` from embedding providers

**Solutions:**
1. Reduce `embedding_batch_size` in config
2. Add delays between batches
3. Check provider rate limits and upgrade plan if needed

### Memory Issues During Indexing

**Issue:** Out of memory errors with large documents

**Solutions:**
1. Reduce `default_chunk_size` in configuration (try 300-500)
2. Process documents in smaller batches
3. Enable `estimate` mode first to preview costs
4. Reduce `embedding_batch_size` if processing many documents

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
5. Check `active_domain` configuration - ensure you're using the correct domain/collection

### Poor Quality Results

**Issue:** Retrieved documents not relevant

**Solutions:**
1. Enable query rewrite: `"enable_query_rewrite": true`
2. Adjust `rewrite_confidence_threshold` (try 0.6-0.8)
3. Check document chunking strategy:
   - Reduce `default_chunk_size` (try 300-500)
   - Adjust `default_chunk_overlap` (try 50-100)
4. Consider re-indexing with different chunk size
5. Enable reranking if not already enabled

### Tool Calling Issues

**Issue:** Tools not being called or failing

**Solutions:**
1. Ensure `"use_tools": true` in params
2. Check tool configurations in backend
3. Verify API keys for external services (weather, airports)
4. Check tool execution logs
5. Verify `max_tool_passes` allows sufficient tool loops

### Reasoning Model Issues

**Issue:** Reasoning models not working or showing reasoning

**Solutions:**
1. Use correct model keys:
   - OpenAI: `openai:reasoning_o3-mini` or `openai:reasoning_gpt-5-mini`
   - Gemini: `gemini:openai-3-flash-preview` or `gemini:openai-reasoning-2.5-flash`
2. Set correct reasoning parameters:
   - OpenAI: `"reasoning_effort": "low"|"medium"|"high"`
   - Gemini: `"thinking_level": "minimal"|"low"|"medium"|"high"`
3. Enable `debug_thoughts=true` to see reasoning output
4. Check model registry for reasoning support

### Query Rewrite Issues

**Issue:** Query rewrite not working or always rejected

**Solutions:**
1. Ensure `"enable_query_rewrite": true` in params
2. Check `rewrite_confidence_threshold` (default 0.7, try 0.6)
3. Verify conversation history exists (rewrite needs context)
4. Check `rewrite_tail_turns` and `rewrite_summary_turns` settings
5. Look for rewrite display in response to see why rejected

### Session Management Issues

**Issue:** Session-based chat not maintaining context

**Solutions:**
1. Ensure consistent `session_id` across requests
2. Check session manager logs for session creation/deletion
3. Verify session TTL settings (`chunk_manager_idle_ttl_seconds`)
4. Test with simple session first to isolate the issue

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
1. Reduce batch sizes in configuration:
   - `embedding_batch_size`: try 50-100
   - `default_chunk_size`: try 300-500
2. Process documents sequentially instead of in parallel
3. Monitor memory usage during indexing
4. Consider using smaller models for non-critical stages

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

**Issue:** No search results or poor results due to wrong domain/collection

**Common Scenarios:**
- Indexed data with OpenAI but searching with Gemini (or vice versa)
- `active_domain` doesn't match the collection that contains your data
- Collection exists but has wrong embedding dimensions

**Solutions:**
1. Check current domain configuration:
   ```python
   from backend.core.config import settings
   print(f"Active domain: {settings.active_domain}")
   print(f"Collection: {settings.collection_name}")
   print(f"Embedding model: {settings.embedding_model_key}")
   ```

2. Verify domain mappings:
   ```python
   from backend.core.config import settings
   for domain, config in settings.DOMAIN_EMBEDDING_CONFIG.items():
       print(f"{domain}: {config['collection_name']} -> {config['embedding_model_key']}")
   ```

3. Check what's actually indexed:
   ```bash
   python scripts/qdrant_scripts/qdrant_ops.py --list-collections
   python scripts/qdrant_scripts/qdrant_ops.py --count-chunks --collection document_index
   python scripts/qdrant_scripts/qdrant_ops.py --count-chunks --collection document_index_gemini
   ```

4. Switch domain if needed:
   - **Mountains domain**: Uses OpenAI embeddings (`document_index`)
   - **Oceans domain**: Uses Gemini embeddings (`document_index_gemini`)
   - Change `active_domain` in `backend/core/config.py` or set `ACTIVE_DOMAIN` environment variable

5. Re-index with correct model if domain/collection mismatch:
   ```bash
   # Switch to correct domain first, then re-seed
   export ACTIVE_DOMAIN=oceans  # or mountains
   make seed
   ```

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

# Test API connections
python scripts/api_smoke_test_openai.py
python scripts/api_smoke_test_gemini.py

# Check collections
python scripts/qdrant_scripts/qdrant_ops.py --list-collections

# Test embedding generation
python scripts/embedding_compare.py

# Test Gemini embeddings specifically
python scripts/test_gemini_client_embeddings.py

# Test reasoning models
python scripts/test_gemini_reasoning.py

# Test chunked history manager
python scripts/test_chunked_history_manager.py

# Check Qdrant collection info
python scripts/qdrant_scripts/qdrant_ops.py --collection-info document_index
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
