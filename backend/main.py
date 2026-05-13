import os
from dotenv import load_dotenv
# Load environment variables from .env (if present) without overriding OS env vars.
# Precedence: OS env vars win; .env is a fallback.
load_dotenv(override=False)

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uvicorn
import re
import requests
import yaml
from urllib.parse import urlsplit, urlunsplit, urlparse
from qdrant_client.http import models
from backend.core import settings
from backend.embeddings.embeddings_manager import EmbeddingsManager
from backend.crawler.crawler import WebCrawler
#from backend.crawler.pdf_crawler import PDFCrawler
from backend.extractor import HTMLExtractor, MediaWikiExtractor, PDFExtractor
from backend.core import (
    EmbeddingRequest,
    SearchRequest,
    SearchResponse,
    ChatRequest,
    ChatResponse,
    MediaWikiURLInput,
    PDFInput,
    URLInput,
    PayloadUpdateRequest,
)
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from pydantic import BaseModel, Field
from backend.db import QdrantDB
from backend.chat.chat_manager import ChatManager
from backend.chat.prompt_registry import clear_prompt_registry_cache
from backend.chat.chat_manager import clear_tool_registry_cache
from pydantic import BaseModel
from backend.api.endpoints import model_keys as model_keys_endpoint
from backend.llm.llm_client import generate, embed, get_pricing_for_model


class ClearChatRequest(BaseModel):
    conversation_id: str | None = None
    user_id: str | None = None

import logging.config
from backend.core.logging import LOGGING_CONFIG

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


# --- Origin / host allowlist helper (initial protection for critical routes) ---

def _parse_allowed_list(raw: str | None) -> set[str]:
    """Return a set of stripped items from a comma-separated config string."""
    try:
        if not raw:
            return set()
        return {item.strip() for item in str(raw).split(',') if item and item.strip()}
    except Exception:
        return set()


_ALLOWED_ORIGINS = _parse_allowed_list(getattr(settings, "allowed_origins", None))
_ALLOWED_HOSTS = _parse_allowed_list(getattr(settings, "allowed_hosts", None))


def _get_embedding_rate_per_mm_tokens() -> float:
    """Return the embedding cost (USD) per 1M tokens for the active embedding model.

    Preference order (non-breaking):
      1) Use pricing.input_per_mm from model_registry for settings.embedding_model_key
      2) Fall back to 0.0 when registry/pricing is unavailable or misconfigured.
    """
    # Safe default when no pricing is available
    default_rate = 0.0

    try:
        embedding_key = str(getattr(settings, "embedding_model_key", "")).strip()
        if not embedding_key:
            return default_rate

        pricing = get_pricing_for_model(model_key=embedding_key)
        if pricing is None:
            return default_rate

        rate = getattr(pricing, "input_per_mm", None)
        if rate is None:
            return default_rate
        return float(rate)
    except Exception:
        # Any lookup or config issue should silently fall back to the prior behavior.
        return default_rate


def _resolve_domain_config(active_domain: Optional[str]) -> Dict[str, str]:
    available_domains = getattr(settings, "DOMAIN_EMBEDDING_CONFIG", {}) or {}
    configured_default_domain = str(getattr(settings, "active_domain", "") or "").strip() or "default"
    requested_domain = str(active_domain or configured_default_domain).strip()
    effective_domain = requested_domain if requested_domain in available_domains else configured_default_domain
    cfg = available_domains.get(effective_domain) or available_domains.get(configured_default_domain) or {}
    collection_name = str(cfg.get("collection_name") or settings.collection_name)
    embedding_model_key = str(cfg.get("embedding_model_key") or settings.embedding_model_key)
    return {
        "requested_domain": requested_domain,
        "effective_domain": effective_domain,
        "collection_name": collection_name,
        "embedding_model_key": embedding_model_key,
    }


def _build_domain_qdrant(active_domain: Optional[str]) -> QdrantDB:
    domain_cfg = _resolve_domain_config(active_domain)
    return QdrantDB(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=domain_cfg["collection_name"],
        embedding_model_key=domain_cfg["embedding_model_key"],
    )


def _get_embedding_spec_for_domain(active_domain: Optional[str]) -> Dict[str, Any]:
    """Get embedding spec from retrieval config for the given domain."""
    from backend.retrieval.config import resolve_retrieval_specs
    
    domain_cfg = _resolve_domain_config(active_domain)
    retrieval_specs = resolve_retrieval_specs(
        domain=domain_cfg["effective_domain"],
        config_path=str(getattr(settings, "retrieval_config_path", "prompts/retrieval_registry.yaml") or "").strip() or None,
    )
    
    emb_cfg = (retrieval_specs or {}).get("embedding") or {}
    
    # Fall back to legacy config if retrieval config is empty
    if not emb_cfg:
        from backend.embeddings.specs import resolve_embedding_spec
        legacy_spec = resolve_embedding_spec(settings) or {}
        return {
            "runtime": "hosted",
            "provider": legacy_spec.get("provider", "openai"),
            "model": domain_cfg["embedding_model_key"],
            "dimensions": legacy_spec.get("dimensions"),
            "normalize": True,
            "batch_size": 32,
            "device": None,
            "extra": {},
        }
    
    return {
        "runtime": emb_cfg.get("runtime", "hosted"),
        "provider": emb_cfg.get("provider", "openai"),
        "model": emb_cfg.get("model", domain_cfg["embedding_model_key"]),
        "dimensions": emb_cfg.get("dimensions"),
        "normalize": emb_cfg.get("normalize", True),
        "batch_size": emb_cfg.get("batch_size", 32),
        "device": emb_cfg.get("device"),
        "extra": emb_cfg.get("extra") if isinstance(emb_cfg.get("extra"), dict) else {},
    }


def _generate_embeddings_with_retrieval(texts: List[str], active_domain: Optional[str]) -> List[List[float]]:
    """Generate embeddings using the retrieval module."""
    from backend.retrieval.embedding_router import EmbeddingRouter
    from backend.retrieval.schemas import EmbeddingSpec
    
    spec_dict = _get_embedding_spec_for_domain(active_domain)
    spec = EmbeddingSpec(
        task="embedding",
        runtime=spec_dict["runtime"],
        provider=spec_dict["provider"],
        model=spec_dict["model"],
        dimensions=spec_dict["dimensions"],
        normalize=spec_dict["normalize"],
        batch_size=spec_dict["batch_size"],
        device=spec_dict["device"],
        extra=spec_dict["extra"],
    )
    
    router = EmbeddingRouter()
    result = router.embed(texts, spec)
    return result.vectors


def _index_chunks_with_retrieval(
    chunks: List[Dict[str, Any]],
    active_domain: Optional[str],
    force_delete: bool = True,
    max_chunks: Optional[int] = None,
) -> Dict[str, int]:
    """Index chunks using the retrieval module instead of EmbeddingsManager."""
    import uuid
    import tiktoken
    from qdrant_client import models
    
    domain_cfg = _resolve_domain_config(active_domain)
    qdrant = _build_domain_qdrant(active_domain)
    collection_name = domain_cfg["collection_name"]
    
    # Ensure collection exists
    try:
        qdrant.client.get_collection(collection_name)
    except Exception:
        logger.debug("Collection not found, creating it now...")
        qdrant.create_collection()

    # Detect whether collection expects named vectors (e.g., "dense")
    try:
        collection_info = qdrant.client.get_collection(collection_name)
        vectors_cfg = collection_info.config.params.vectors
        sparse_cfg = collection_info.config.params.sparse_vectors
        has_named_dense_vector = isinstance(vectors_cfg, dict) and "dense" in vectors_cfg
        has_sparse_vector = isinstance(sparse_cfg, dict) and "sparse" in sparse_cfg
    except Exception:
        has_named_dense_vector = False
        has_sparse_vector = False
    
    # Cap chunks if needed
    effective_cap = int(getattr(settings, "max_chunks_per_doc", 500))
    if max_chunks is not None:
        try:
            user_cap = int(max_chunks)
            if user_cap > 0:
                effective_cap = min(effective_cap, user_cap)
        except Exception:
            pass
    if len(chunks) > effective_cap:
        chunks = chunks[:effective_cap]
    
    # Get embedding spec for metadata
    spec_dict = _get_embedding_spec_for_domain(active_domain)
    
    # Process in batches
    batch_size = spec_dict["batch_size"]
    points = []
    tokens_used = 0
    
    # Token encoder for estimation
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        encoding = None
    
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start:batch_start + batch_size]
        batch_texts = []
        for chunk in batch:
            text = chunk.get("text", "") if isinstance(chunk, dict) else chunk
            batch_texts.append(text)
        
        # Generate embeddings
        embeddings = _generate_embeddings_with_retrieval(batch_texts, active_domain)
        
        # Create points
        for offset, chunk in enumerate(batch):
            idx = batch_start + offset
            text = chunk.get("text", "") if isinstance(chunk, dict) else chunk
            embedding = embeddings[offset]
            
            # Estimate tokens
            if encoding:
                try:
                    tokens_used += len(encoding.encode(text, disallowed_special=()))
                except Exception:
                    pass
            
            # Build payload
            chunk_url = chunk.get("url", "")
            chunk_base_url = chunk.get("base_url", _strip_fragment_url(chunk_url))
            payload = {
                "text": text,
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "url": chunk_url,
                "url_lower": (chunk_url or "").lower(),
                "base_url": chunk_base_url,
                "base_url_lower": (chunk_base_url or "").lower(),
                "document_type": chunk.get("document_type", "mediawiki"),
                "source": chunk_url,
                "title": chunk.get("title", ""),
                "section": chunk.get("section", "Lead"),
                "subsection": chunk.get("subsection"),
                "embedding_model": spec_dict["model"],
                "embedding_provider": spec_dict["provider"],
                "embedding_runtime": spec_dict["runtime"],
            }
            
            point_id = str(uuid.uuid4())

            sparse_vector = None
            if has_sparse_vector:
                try:
                    sparse_emb = qdrant.generate_sparse_embeddings(text)
                    sparse_indices = sparse_emb.get("indices") or []
                    sparse_values = sparse_emb.get("values") or []
                    sparse_vector = models.SparseVector(indices=sparse_indices, values=sparse_values)
                except Exception:
                    sparse_vector = models.SparseVector(indices=[], values=[])

            if has_named_dense_vector and sparse_vector is not None:
                vector_payload = {"dense": embedding, "sparse": sparse_vector}
            elif has_named_dense_vector:
                vector_payload = {"dense": embedding}
            elif sparse_vector is not None:
                vector_payload = {"sparse": sparse_vector}
            else:
                vector_payload = embedding

            points.append(models.PointStruct(
                id=point_id,
                vector=vector_payload,
                payload=payload,
            ))
    
    # Delete existing if force_delete
    if force_delete and chunks:
        url = chunks[0].get("url", "")
        if url:
            qdrant.delete_by_url(url)
    
    # Insert points
    qdrant.client.upsert(
        collection_name=collection_name,
        points=points,
    )
    
    return {
        "vectors_indexed": len(points),
        "tokens_used": tokens_used,
    }


def enforce_origin_host(request: Request) -> None:
    """Best-effort origin/host check for critical FastAPI routes.

    When no allowlists are configured, this is a no-op. Otherwise, it will allow
    a request if either the full Origin header value matches an entry in
    settings.allowed_origins OR the parsed origin host / Host header matches an
    entry in settings.allowed_hosts.
    """

    # Fast path: no configuration => no additional enforcement
    
    if not _ALLOWED_ORIGINS and not _ALLOWED_HOSTS:
        logger.warning("SECURITY: Origin/Host Blocked; ALLOWLIST origins=%s hosts=%s; got origin=%s host=%s",
               _ALLOWED_ORIGINS, _ALLOWED_HOSTS, origin, host_hdr)
        return

    try:
        origin = request.headers.get("origin") or request.headers.get("referer") or ""
        host_hdr = request.headers.get("host") or ""

        origin_ok = False
        host_ok = False

        # Direct Origin match (scheme + host[:port])
        if origin and _ALLOWED_ORIGINS:
            origin_ok = origin in _ALLOWED_ORIGINS

        # Host-based checks
        if _ALLOWED_HOSTS:
            origin_host = ""
            if origin:
                try:
                    parsed = urlparse(origin)
                    if parsed.hostname:
                        origin_host = parsed.hostname
                        if parsed.port:
                            origin_host = f"{parsed.hostname}:{parsed.port}"
                except Exception:
                    origin_host = ""

            if origin_host and origin_host in _ALLOWED_HOSTS:
                host_ok = True
            elif host_hdr and host_hdr in _ALLOWED_HOSTS:
                host_ok = True

        if not (origin_ok or host_ok):
            logger.warning("SECURITY: origin not ok or host not ok Origin/Host Blocked; ALLOWLIST origins=%s hosts=%s; got origin=%s host=%s",
               _ALLOWED_ORIGINS, _ALLOWED_HOSTS, origin, host_hdr)
            raise HTTPException(status_code=403, detail="Origin/host not allowed")
    except HTTPException:
        raise
    except Exception:
        # On any unexpected error in enforcement, fail open to avoid breaking dev.
        return


# --- Swagger/OpenAPI tags & UI ordering ---
tags_metadata = [
    {"name": "1. UI Pages", "description": "Frontend pages served by FastAPI (HTML)."},
    {"name": "2. Ingest", "description": "Bring content in: MediaWiki, URLs, PDFs; embed documents; delete by URL; batch processing."},
    {"name": "3. Search & Chat", "description": "Vector search and conversational endpoints over indexed content."},
    {"name": "4. Index Admin", "description": "Index maintenance utilities (payload updates, etc.)."},
    {"name": "5. Debug", "description": "Developer diagnostics and inspection utilities."},
]


app = FastAPI(
    title="RAG Pipeline Chat API",
    openapi_tags=tags_metadata,
    swagger_ui_parameters={
        "tagsSorter": "alpha",
        "operationsSorter": "alpha",
    },
)

# Attempt to mount SSE streaming routes by default. This is best-effort and guarded
# to avoid breaking startup if the SSE module isn't ready yet.
try:
    from backend.stream_stages import router as stream_stages_router
    app.include_router(stream_stages_router, prefix="/chat")
except Exception as _e:
    logger.warning("SSE stream routes could not be registered: %s", _e)

# Expose backend model registry for frontend consumption
app.include_router(model_keys_endpoint.router, tags=["3. Search & Chat"])

# Configure static file serving
# This allows the frontend to be served from the same server as the API
# Static files will be served from the frontend directory
from pathlib import Path

static_dir = Path(__file__).resolve().parent.parent / "frontend" / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# Root route to serve the main index.html file
@app.get(
    "/",
    tags=["1. UI Pages"],
    summary="1. Home (index.html)",
    response_class=HTMLResponse,
    responses={200: {"content": {"text/html": {"example": "<!DOCTYPE html><html><body>Home Page</body></html>"}}}}
)
async def root():
    """Serve the main HTML file at the root path"""
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    return FileResponse(os.path.join(frontend_dir, "index.html"))

# Debug route to list registered routes
@app.get("/debug/routes", tags=["5. Debug"], summary="List registered routes")
async def debug_routes():
    return {"paths": sorted([r.path for r in app.router.routes])}

# Search page route
@app.get(
    "/search",
    tags=["1. UI Pages"],
    summary="2. Search page (HTML)",
    response_class=HTMLResponse,
    responses={200: {"content": {"text/html": {"example": "<!DOCTYPE html><html><body>Search Page</body></html>"}}}}
)
async def search_page():
    """Serve the dedicated search interface page"""
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    return FileResponse(os.path.join(frontend_dir, "search.html"))

# Debug index HTML page (separate from API JSON at /debug-index)
@app.get(
    "/debug_index",
    tags=["1. UI Pages"],
    summary="3. Debug Index (HTML)",
    response_class=HTMLResponse,
    responses={200: {"content": {"text/html": {"example": "<!DOCTYPE html><html><body>Debug Index</body></html>"}}}}
)
async def debug_index_page():
    """Serve the debug-index HTML page"""
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    return FileResponse(os.path.join(frontend_dir, "debug-index.html"))


@app.get(
    "/api/domains",
    tags=["1. UI Pages"],
    summary="Get available domains",
    responses={200: {"content": {"application/json": {"example": "{\"domains\": [\"default\", \"mountains\", \"finance\"]}"}}}},
)
async def get_available_domains():
    """Return a list of available domains from the retrieval config and legacy config."""
    from backend.retrieval.config import _read_yaml
    from pathlib import Path
    
    domains_set = set()
    
    # Add domains from legacy DOMAIN_EMBEDDING_CONFIG
    try:
        legacy_domains = getattr(settings, "DOMAIN_EMBEDDING_CONFIG", {}) or {}
        if isinstance(legacy_domains, dict):
            domains_set.update(legacy_domains.keys())
    except Exception:
        pass
    
    # Add domains from retrieval config YAML
    try:
        config_path = str(getattr(settings, "retrieval_config_path", "prompts/retrieval_registry.yaml") or "").strip()
        if config_path:
            cfg = _read_yaml(config_path)
            if isinstance(cfg, dict):
                retrieval_cfg = cfg.get("retrieval") if isinstance(cfg.get("retrieval"), dict) else {}
                domains_cfg = retrieval_cfg.get("domains") if isinstance(retrieval_cfg.get("domains"), dict) else {}
                if isinstance(domains_cfg, dict):
                    domains_set.update(domains_cfg.keys())
    except Exception:
        pass
    
    # Ensure "default" is always available
    domains_set.add("default")
    
    # Sort domains for consistent display
    domains = sorted(list(domains_set))
    
    return {"domains": domains}


@app.get(
    "/list-docs-data",
    tags=["1. UI Pages"],
    summary="4. List Documents Data (JSON)",
    responses={200: {"content": {"application/json": {"example": "{\"documents\": []}"}}}},
)
async def list_docs_data(limit: int = None, url: str = None, active_domain: Optional[str] = None):
    """Return a JSON payload containing documents for display, optionally filtered by URL.

    Each item includes: base_url, title, total_chunks, updated_at.
    The frontend displays one item per line.

    Args:
        limit: Maximum number of documents to return (optional)
        url: If provided, only return documents matching this URL (case-insensitive)
    """
    # Try to fetch real data from Qdrant using the configured collection
    data = {"documents": []}
    try:
        from backend.db import QdrantDB
        from backend.core.config import settings

        from qdrant_client import models

        # Initialize DB handle
        domain_cfg = _resolve_domain_config(active_domain)
        qd = QdrantDB(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=domain_cfg["collection_name"],
            embedding_model_key=domain_cfg["embedding_model_key"],
        )
        
        # Create filter if URL is provided
        scroll_filter = None
        if url:
            url_lower = url.lower()
            scroll_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="url_lower",
                        match=models.MatchValue(value=url_lower)
                    )
                ]
            )

        # Attempt a simple scroll to fetch documents; best-effort approach
        offset = None
        while True:
            try:
                # The underlying client.scroll returns (points, next_offset)
                resp = qd.client.scroll(
                    collection_name=qd.collection_name,
                    offset=offset,
                    limit=100,
                    scroll_filter=scroll_filter
                )
                if isinstance(resp, tuple) and len(resp) == 2:
                    points, offset = resp
                else:
                    points = resp
                    offset = None
            except TypeError:
                # Fallback if the client returns a different structure
                points = []
                offset = None

            if not isinstance(points, list):
                points = list(points)

            for p in points:
                payload = getattr(p, 'payload', {}) or {}
                base_url = payload.get('base_url') or payload.get('url')
                title = payload.get('title') or ''
                # total_chunks: prefer explicit field, else infer from a chunks list
                total_chunks = payload.get('total_chunks')
                if total_chunks is None:
                    chunks = payload.get('chunks') or []
                    total_chunks = len(chunks)
                updated_at = payload.get('updated_at', '') or ''
                if base_url:
                    # Only add if URL matches (case-insensitive) or no URL filter
                    if not url or (isinstance(base_url, str) and url.lower() in base_url.lower()):
                        data['documents'].append({
                            'base_url': base_url,
                            'title': title,
                            'total_chunks': int(total_chunks) if total_chunks is not None else 0,
                            'updated_at': updated_at,
                        })
            if not offset:
                break
        # Respect the limit at display time; the UI will slice, and the download will provide full data
        if limit is not None and limit > 0:
            data['documents'] = data['documents'][:limit]
    except Exception as e:
        logger.exception("Error wiring real data for list-docs-data: %s", e)
        # Fall back to a best-effort empty payload to avoid breaking the UI
        data = {"documents": []}

    from fastapi.responses import JSONResponse
    return JSONResponse(content=data)

@app.get(
    "/list-docs.html",
    include_in_schema=False  # Keep this alias for the UI; not part of API docs
)
async def list_docs_html_page():
    """Serve the list-docs HTML page (UI)"""
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    return FileResponse(os.path.join(frontend_dir, "list-docs.html"))

@app.get(
    "/process-batch-docs",
    tags=["1. UI Pages"],
    summary="5. Batch Process Documents (HTML)",
    response_class=HTMLResponse,
    responses={200: {"content": {"text/html": {"example": "<!DOCTYPE html><html><body>Batch Process Page</body></html>"}}}}
)
@app.get(
    "/process-batch-docs.html",
    include_in_schema=False  # Hide this from the API docs since it's just an alias
)
async def process_batch_docs_page():
    """Serve the process-batch-docs HTML page"""
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    return FileResponse(os.path.join(frontend_dir, "process-batch-docs.html"))

# Chat page route (HTML)
@app.get(
    "/chat.html",
    tags=["1. UI Pages"],
    summary="4. Chat page (HTML)",
    response_class=HTMLResponse,
    responses={200: {"content": {"text/html": {"example": "<!DOCTYPE html><html><body>Chat Page</body></html>"}}}}
)
async def chat_page():
    """Serve the standalone chat page"""
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    return FileResponse(os.path.join(frontend_dir, "chat.html"))


@app.get(
    "/chat-embed.html",
    tags=["1. UI Pages"],
    summary="Embedded chat page (HTML)",
    response_class=HTMLResponse,
    responses={200: {"content": {"text/html": {"example": "<!DOCTYPE html><html><body>Embedded Chat Page</body></html>"}}}},
)
async def chat_embed_page():
    """Serve the minimal embeddable chat page (chat-embed.html)."""
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    return FileResponse(os.path.join(frontend_dir, "chat-embed.html"))


@app.get(
    "/chat-embed-example.html",
    tags=["1. UI Pages"],
    summary="Embedded chat example page (HTML)",
    response_class=HTMLResponse,
    responses={200: {"content": {"text/html": {"example": "<!DOCTYPE html><html><body>Chat Embed Example</body></html>"}}}},
)
async def chat_embed_example_page():
    """Serve the example page demonstrating the embeddable chat popup."""
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    return FileResponse(os.path.join(frontend_dir, "chat-embed-example.html"))

from typing import Optional
@app.post("/mediawiki/url", tags=["2. Ingest"], summary="1. Index MediaWiki URL")
async def index_mediawiki_url(
    mediawiki_input: MediaWikiURLInput,
    request: Request,
):
    """
    Index content from MediaWiki wikitext, optionally limiting number of chunks indexed by max_chunks.
    You can override the target MediaWiki API with `api_url` and the User-Agent with `ua` per request.
    """
    # Enforce origin/host allowlist for this ingestion endpoint.
    enforce_origin_host(request)

    domain_cfg = _resolve_domain_config(mediawiki_input.active_domain)
    qdrant_for_check = _build_domain_qdrant(domain_cfg["effective_domain"])
    try:
        #logger.info("TRACE MW: Before MediaWikiExtractor init")
        extractor = MediaWikiExtractor(
            chunk_size=settings.html_chunk_size,
            chunk_overlap=settings.html_chunk_overlap,
            api_url=mediawiki_input.api_url,
            user_agent=mediawiki_input.user_agent,
            wiki_mode=settings.wiki_mode, # "parsoid" or "action_api". Parsoid is recommended for better accuracy.
            wiki_index_tables=settings.wiki_index_tables,
            wiki_table_rows_per_chunk=settings.wiki_table_rows_per_chunk,
            wiki_drop_tables_from_prose=settings.wiki_drop_tables_from_prose,
        )
        #logger.info("TRACE MW: MediaWikiExtractor init done")
        url = mediawiki_input.url
        max_chunks = mediawiki_input.max_chunks
        skip_sections = mediawiki_input.skip_sections
        force_delete = mediawiki_input.force_delete
        estimate = mediawiki_input.estimate
        #logger.info(f"TRACE MW: Read URL: {url}")
        # Optional pre-check: if the document is already indexed in Qdrant, prompt for confirmation
        if bool(settings.check_document_indexed):
            try:
                existing_count = qdrant_for_check.count_points_by_url(url)
                logger.info("Existing vectors found for URL: %s", existing_count)
                #logger.info("Confirm reindex: %s", force_delete)
            except Exception as e:
                # If counting fails, proceed without blocking indexing
                existing_count = 0
                logger.warning("Failed checking existing index for %s: %s", url, e)
            if existing_count > 0 and not estimate and not bool(force_delete):
                # logger.info("Confirmation required to reindex; aborting until confirmed")
                return {
                    "message": "URL already indexed",
                    "url": url,
                    "already_indexed": True,
                    "vectors_found": int(existing_count),
                    "confirmation_required": True,
                    "hint": "Resubmit with force_delete=true to proceed",
                }
            elif existing_count > 0 and bool(force_delete):
                logger.info("force_delete=true; proceeding with reindex")
        # Use extractor.parse_from_url directly, passing skip_sections
        #logger.info(f"TRACE MW: Extraction started for URL: {url}")
        chunks = extractor.parse_from_url(url, skip_sections=skip_sections)
        #logger.info(f"TRACE MW: Extraction done. Number of chunks created: {len(chunks)}")
        # Limit the number of chunks indexed if max_chunks is a positive value
        if max_chunks is not None and max_chunks > 0:
            chunks = chunks[:max_chunks]
        #logger.info(f"TRACE MW: Chunks after capping (if any): {len(chunks)}")
        if mediawiki_input.estimate:
            # Calculate estimated tokens
            tokens_used = 0
            import tiktoken
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                for chunk in chunks:
                    text = chunk.get("text", "") if isinstance(chunk, dict) else chunk
                    tokens_used += len(encoding.encode(text, disallowed_special=()))
            except Exception:
                pass

            # logger.info("Estimate only; planned chunks: %d, tokens: %d", len(chunks), tokens_used)
            # Calculate estimated embedding cost using provider/model-aware rate
            rate_per_mm = _get_embedding_rate_per_mm_tokens()
            estimated_cost = (tokens_used * rate_per_mm) / 1_000_000.0
            return {
                "message": "Estimate only", 
                "chunks_planned": len(chunks),
                "tokens_used": tokens_used,
                "embedding_cost": round(estimated_cost, 8)
            }
        #logger.info(f"TRACE MW: Sending {len(chunks)} chunks for embedding")
        result = _index_chunks_with_retrieval(
            chunks,
            active_domain=mediawiki_input.active_domain,
            force_delete=force_delete,
            max_chunks=max_chunks,
        )
        logger.info(
            "Embedding finished. vectors_indexed=%s, tokens_used=%s",
            result.get('vectors_indexed', 0),
            result.get('tokens_used', 0),
        )
        rate_per_mm = _get_embedding_rate_per_mm_tokens()
        embedding_cost = (result.get("tokens_used", 0) * rate_per_mm) / 1_000_000.0
        return {
            "message": "MediaWiki content indexed successfully",
            "chunks_indexed": len(chunks),
            "vectors_indexed": result.get("vectors_indexed", 0),
            "tokens_used": result.get("tokens_used", 0),
            "embedding_cost": round(embedding_cost, 8),
            "url": url,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/clear", tags=["3. Search & Chat"], summary="6. Clear conversation summaries (stateless)")
async def clear_chat_summaries(payload: ClearChatRequest, request: Request):
    """Clear server-side summary cache for a conversation namespace.

    Namespace rule (Option A):
    - If both user_id and conversation_id: f"{user_id}:{conversation_id}"
    - Else: conversation_id only
    - If conversation_id missing/empty: no-op
    """
    enforce_origin_host(request)

    try:
        from backend.chat import chat_manager as cm
        cid = (payload.conversation_id or "").strip()
        uid = (payload.user_id or "").strip()
        if not cid and not uid:
            return JSONResponse({"removed": 0, "remaining": 0, "reclaimed_bytes": 0})
        ns = f"{uid}:{cid}" if uid and cid else (cid or "")
        stats = cm.clear_summaries_for_namespace(ns)
        # Also clear chunked history state for this namespace so the conversation truly resets.
        try:
            chunk_stats = cm.clear_chunk_manager_for_namespace(ns)
            if isinstance(stats, dict) and isinstance(chunk_stats, dict):
                stats["chunked_history"] = chunk_stats
        except Exception:
            pass
        # Also reset per-conversation totals for this namespace so a new conversation_id starts at zero
        try:
            totals_stats = cm.clear_convo_totals_for_namespace(ns)
            if isinstance(stats, dict) and isinstance(totals_stats, dict):
                stats["totals"] = totals_stats
        except Exception:
            pass
        return JSONResponse(stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pdf", tags=["2. Ingest"], summary="1c. Index PDF (upload or URL)")
async def index_pdf(
    pdf_input: PDFInput,
    request: Request,
):
    """Index a PDF either by uploaded file or by URL."""
    enforce_origin_host(request)

    domain_cfg = _resolve_domain_config(pdf_input.active_domain)
    qdrant_for_check = _build_domain_qdrant(domain_cfg["effective_domain"])

    logger.info(
        "PDF DEBUG: url=%s has_file=%s max_chunks=%s force_delete=%s estimate=%s pdfinputfilename=%s",
        pdf_input.url,
        bool(pdf_input.file),
        pdf_input.max_chunks,
        pdf_input.force_delete,
        pdf_input.estimate,
        pdf_input.filename
    )
    if pdf_input.file:
        logger.info("PDF DEBUG: pdf_input.file length=%s", len(pdf_input.file))
    
    if not pdf_input.file and not pdf_input.url:
        raise HTTPException(status_code=400, detail="Provide either a PDF file or a URL")
    
    try:
        source = pdf_input.url if pdf_input.url else "uploaded://file.pdf"
        skip_sections = pdf_input.skip_sections
        
        # For URL-based PDFs, check if already indexed
        if pdf_input.url and bool(settings.check_document_indexed):
            try:
                existing_count = qdrant_for_check.count_points_by_url(pdf_input.url)
                logger.info("Existing vectors found for PDF URL: %s", existing_count)
            except Exception as e:
                # If counting fails, proceed without blocking indexing
                existing_count = 0
                logger.warning("Failed checking existing index for %s: %s", pdf_input.url, e)
            
            if existing_count > 0 and not pdf_input.estimate and not bool(pdf_input.force_delete):
                return {
                    "message": "PDF URL already indexed",
                    "url": pdf_input.url,
                    "already_indexed": True,
                    "vectors_found": int(existing_count),
                    "confirmation_required": True,
                    "hint": "Resubmit with force_delete=true to proceed",
                }
            elif existing_count > 0 and bool(pdf_input.force_delete):
                logger.info("force_delete=true; proceeding with reindex")
        
        extractor = PDFExtractor(
            chunk_size=settings.html_chunk_size,
            chunk_overlap=settings.html_chunk_overlap,
            skip_sections=skip_sections,
            header_footer_filter=getattr(settings, 'header_footer_filter', True),
            pdf_index_tables=getattr(settings, 'pdf_index_tables', True),
            pdf_table_rows_per_chunk=getattr(settings, 'pdf_table_rows_per_chunk', 12),
            pdf_repeat_table_header=getattr(settings, 'pdf_repeat_table_header', True),
            pdf_table_min_rows=getattr(settings, 'pdf_table_min_rows', 1),
            pdf_drop_tables_from_prose=getattr(settings, 'pdf_drop_tables_from_prose', False),   
        )
        # for PDF file upload
        if pdf_input.file:
            # Handle base64 encoded file
            import base64
            import hashlib
            try:
                file_data = base64.b64decode(pdf_input.file)
                
                # Generate a unique hash of the file content for duplicate checking
                file_hash = hashlib.sha256(file_data).hexdigest()
                #source = f"uploaded://{file_hash}"
                # Prefer the original URL (including file:///app/...) if present.
                # Only fall back to filename or hash when no URL is available.
                if pdf_input.url:
                    source = pdf_input.url
                elif pdf_input.filename:
                    source = f"file://{pdf_input.filename}"
                else:
                    source = f"uploaded://{file_hash}"
                #source = f"file://{pdf_input.filename}"
                
                logger.info("PDF DEBUG filename : source=%s", source)
                # Check for existing indexed content with the same hash
                if bool(settings.check_document_indexed):
                    try:
                        existing_count = qdrant_for_check.count_points_by_url(source)
                        logger.info("Existing vectors found for PDF hash: %s", existing_count)
                        
                        if existing_count > 0 and not pdf_input.estimate and not bool(pdf_input.force_delete):
                            return {
                                "message": "This PDF has already been indexed",
                                "url": source,
                                "already_indexed": True,
                                "vectors_found": int(existing_count),
                                "confirmation_required": True,
                                "hint": "Resubmit with 'Force delete existing' checked to proceed",
                            }
                        elif existing_count > 0 and bool(pdf_input.force_delete):
                            logger.info("force_delete=true; proceeding with reindex")
                            
                    except Exception as e:
                        # If counting fails, proceed without blocking indexing
                        logger.warning("Failed checking existing index for PDF hash: %s", e)
                
                chunks = extractor.parse_from_bytes(file_data, source)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid file data: {str(e)}")
        else:
            # Handle URL
            chunks = extractor.parse_from_url(pdf_input.url)
            
        if pdf_input.max_chunks is not None and pdf_input.max_chunks > 0:
            chunks = chunks[:pdf_input.max_chunks]
        logger.debug("PDF: Final chunks for embedding=%d", len(chunks))  
        if pdf_input.estimate:
            # Calculate estimated tokens
            tokens_used = 0
            import tiktoken
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                for chunk in chunks:
                    text = chunk.get("text", "") if isinstance(chunk, dict) else chunk
                    tokens_used += len(encoding.encode(text, disallowed_special=()))
            except Exception:
                pass

            # Calculate estimated embedding cost using provider/model-aware rate
            rate_per_mm = _get_embedding_rate_per_mm_tokens()
            estimated_cost = (tokens_used * rate_per_mm) / 1_000_000.0
            return {
                "message": "Estimate only",
                "chunks_planned": len(chunks),
                "tokens_used": tokens_used,
                "embedding_cost": round(estimated_cost, 8)
            }

        result = _index_chunks_with_retrieval(
            chunks,
            active_domain=pdf_input.active_domain,
            force_delete=pdf_input.force_delete,
            max_chunks=pdf_input.max_chunks,
        )
        
        rate_per_mm = _get_embedding_rate_per_mm_tokens()
        embedding_cost = (result.get("tokens_used", 0) * rate_per_mm) / 1_000_000.0
        return {
            "message": "PDF content indexed successfully",
            "chunks_indexed": len(chunks),
            "vectors_indexed": result.get("vectors_indexed", 0),
            "tokens_used": result.get("tokens_used", 0),
            "embedding_cost": round(embedding_cost, 8),
            "source": source,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

## Consolidated structured PDF indexing handled by /pdf endpoint below

from backend.chat.chat_manager import ChatManager
from fastapi import Depends, HTTPException
from pydantic import BaseModel
from backend.core.config import Settings
import uuid
import time

# Initialize managers as None; instantiate lazily in routes
embeddings_manager = None
chat_manager = None
qdrant_db = None

# Chat session management
class ChatSessionManager:
    def __init__(self):
        self.sessions = {}
        self.settings = Settings()

    def create_session(self) -> str:
        """Create a new chat session"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "messages": [],
            "context": [],
            "last_access": time.time()
        }
        return session_id

    def get_session(self, session_id: str) -> dict:
        """Get a chat session by ID"""
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail="Chat session not found")
        return self.sessions[session_id]

    def update_session(self, session_id: str, message: dict) -> None:
        """Update chat session with new message"""
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        # Add new message to session
        self.sessions[session_id]["messages"].append(message)
        self.sessions[session_id]["last_access"] = time.time()

    def get_context(self, session_id: str) -> list:
        """Get relevant context for chat session"""
        session = self.get_session(session_id)
        messages = session["messages"]
        
        # Get recent messages within token limit
        total_tokens = 0
        context = []
        
        # Start from most recent message and work backwards
        for msg in reversed(messages):
            msg_tokens = len(msg["content"].split())
            if total_tokens + msg_tokens <= self.settings.max_history_tokens:
                context.append(msg)
                total_tokens += msg_tokens
            else:
                break
        
        return list(reversed(context))

    def reset_session(self, session_id: str) -> None:
        """Clear messages and context for a session."""
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail="Chat session not found")
        self.sessions[session_id]["messages"] = []
        self.sessions[session_id]["context"] = []
        self.sessions[session_id]["last_access"] = time.time()

chat_session_manager = ChatSessionManager()

@app.post("/chat/session", tags=["3. Search & Chat"], summary="1. Create chat session")
async def create_chat_session(request: Request):
    """Create a new chat session"""
    session_id = chat_session_manager.create_session()
    return {"session_id": session_id}

# --- Reset chat totals/state for frontend ---
@app.post(
    "/chat/reset",
    tags=["3. Search & Chat"],
    summary="Reset chat totals/state",
    status_code=200
)
async def reset_chat_totals(request: Request):
    """
    Reset conversation-level metrics/state for the stateless chat path used by chat.html.
    This resets the chat history, parameters, and clears the conversation window.
    """
    enforce_origin_host(request)

    global chat_manager
    if chat_manager is None:
        chat_manager = ChatManager()
    try:
        reset_fn = getattr(chat_manager, "reset_metrics", None) or getattr(chat_manager, "reset", None)
        if callable(reset_fn):
            reset_fn()
        return {"ok": True}
    except Exception as e:
        # logger.error("Error resetting chat totals: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/{session_id}", tags=["3. Search & Chat"], summary="2. Chat (session)")
async def chat_endpoint(session_id: str, chat_request: ChatRequest, request: Request):
    """Process a chat message with context"""
    enforce_origin_host(request)
    
    # Initialize chat_manager if needed
    global chat_manager
    if chat_manager is None:
        chat_manager = ChatManager()
    
    try:
        # Get session context
        session = chat_session_manager.get_session(session_id)
        
        # Get relevant context from chat history
        context = chat_session_manager.get_context(session_id)
        
        # Process chat message with context (wrap synchronous call in thread)
        # Add session_id to params for namespace/token accounting
        chat_params = chat_request.params or {}
        chat_params["session_id"] = session_id

        # Domain-aware path: route through stateless handle_chat orchestration when
        # a domain selector is present, so retrieval/embedding config is per-request.
        requested_domain = str(chat_params.get("active_domain") or chat_params.get("prompt_domain") or "").strip()
        if requested_domain:
            payload = {
                "message": chat_request.message,
                "params": chat_params,
                "history": context,
                "use_web_search": chat_request.use_web_search,
            }
            handler = getattr(chat_manager, "handle_chat", None)
            if handler is None:
                from backend.chat import chat_manager as cm_module

                handler = getattr(cm_module, "handle_chat")
            response = await asyncio.to_thread(handler, payload)
        else:
            response = await asyncio.to_thread(
                chat_manager.chat,
                message=chat_request.message,
                context=context,
                use_web_search=chat_request.use_web_search,
                params=chat_params,
            )
        
        # Update session with new message and response
        chat_session_manager.update_session(session_id, {
            "role": "user",
            "content": chat_request.message,
            "sources": []
        })
        chat_session_manager.update_session(session_id, {
            "role": "assistant",
            "content": response["response"],
            "sources": response["sources"],
            "tools_used": response.get("tools_used", [])
        })
        
        return ChatResponse(
            response=response["response"],
            sources=response["sources"],
            tools_used=response.get("tools_used", []),
            answer=response.get("answer"),
            metrics=response.get("metrics", {}),
            turn_metrics=response.get("turn_metrics", {}),
            conversation_totals=response.get("conversation_totals", {}),
            rewrite_display=response.get("rewrite_display", {})
        )
    except Exception as e:
        # logger.error("Error processing chat: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/{session_id}/history", tags=["3. Search & Chat"], summary="3. Get chat history")
async def get_chat_history(session_id: str):
    """Get chat history for session"""
    try:
        session = chat_session_manager.get_session(session_id)
        return {
            "messages": session["messages"],
            "context": session["context"]
        }
    except Exception as e:
        # logger.error("Error getting chat history: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/{session_id}/reset", tags=["3. Search & Chat"], summary="Reset chat session state")
async def reset_chat_session(session_id: str, request: Request):
    """Reset server-side chat session messages/context for the given session."""
    enforce_origin_host(request)
    try:
        chat_session_manager.reset_session(session_id)
        return {"ok": True}
    except Exception as e:
        # logger.error("Error resetting chat session: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index", tags=["2. Ingest"], summary="2. Index URLs / PDFs")
async def index_content(url_input: URLInput, request: Request):
    """Index content from URLs (HTML or PDF)"""
    enforce_origin_host(request)

    domain_cfg = _resolve_domain_config(url_input.active_domain)
    qdrant_for_check = _build_domain_qdrant(domain_cfg["effective_domain"])
    try:
        #print("DEBUG: Entered index_content route")
        #print("Received input:", url_input.dict())  # Debugging line
        # Process each URL
        planned_chunks_total = 0
        tokens_total = 0
        all_errors: List[Dict[str, Any]] = []  # Aggregate non-fatal errors per-URL (e.g., crawler HTTP errors)
        for url in url_input.urls:
            if url_input.doc_type.lower() == "html":
                crawler = WebCrawler(
                    url,
                    force_crawl=url_input.force_crawl
                )
                pages = await crawler.crawl(url, 1)  # Depth 1 for now
                # Best-effort: if the crawler exposes a collected errors list, merge it into all_errors
                crawl_errors = getattr(crawler, "errors", None)
                if isinstance(crawl_errors, list) and crawl_errors:
                    all_errors.extend(crawl_errors)
                # logger.debug("Crawled %d page(s) from %s", len(pages), url)

                for page in pages:
                    url = page['url']
                    # Optional pre-check: if the document is already indexed in Qdrant, prompt for confirmation
                    if bool(settings.check_document_indexed):
                        try:
                            existing_count = qdrant_for_check.count_points_by_url(url)
                            logger.info("Existing vectors found for URL: %s", existing_count)
                        except Exception as e:
                            # If counting fails, proceed without blocking indexing
                            existing_count = 0
                            logger.warning("Failed checking existing index for %s: %s", url, e)
                        if existing_count > 0 and not url_input.estimate and not bool(url_input.force_delete):
                            return {
                                "message": "URL already indexed",
                                "url": url,
                                "already_indexed": True,
                                "vectors_found": int(existing_count),
                                "confirmation_required": True,
                                "hint": "Resubmit with force_delete=true to proceed",
                            }
                        elif existing_count > 0 and bool(url_input.force_delete):
                            logger.info("force_delete=true; proceeding with reindex")

                    # logger.debug("Extracting content from page: %s", page['url'])
                    # logger.debug("Raw page content length: %d", len(page['content']))

                    # Use section-aware HTML extractor (DOM mode) for parity with PDF/MediaWiki
                    extractor = HTMLExtractor(
                        chunk_size=settings.html_chunk_size,
                        chunk_overlap=settings.html_chunk_overlap,
                        html_mode="dom",
                        html_index_tables=True,
                        html_table_rows_per_chunk=12,
                        html_drop_tables_from_prose=True,
                        skip_sections=url_input.skip_sections,
                    )

                    # If provided content lacks real heading tags, refetch raw HTML to ensure DOM headings are available
                    html_input = page['content'] or ''
                    if not re.search(r'<h[2-4][^>]*>', html_input, flags=re.IGNORECASE):
                        try:
                            # logger.debug("No <h2-4> tags in crawler content; refetching raw HTML: %s", page['url'])
                            resp = requests.get(page['url'], timeout=20)
                            resp.raise_for_status()
                            html_input = resp.text
                        except Exception as e:
                            logger.debug("Refetch failed; using crawler content. Error: %s", e)

                    chunks = extractor.parse(
                        html_input,
                        page['url'],
                        skip_sections=url_input.skip_sections,
                    )

                    # Debug: show detected sections
                    try:
                        found_sections = sorted({ (c.get('section') or 'Lead') for c in chunks })
                        if settings.debug_verbose:
                            logger.debug("Sections detected: %s", found_sections)
                    except Exception:
                        pass

                    # Sanity: show first chunk's key fields before indexing
                    try:
                        if chunks and settings.debug_verbose:
                            for sample in chunks:
                                keys_of_interest = [
                                    "section", "subsection", "subsubsection",
                                    "section_index", "subsection_index",
                                    "chunk_index", "total_chunks",
                                    "url", "document_type", "title", "description",
                                ]
                                view = {k: sample.get(k) for k in keys_of_interest}
                                missing = [k for k in keys_of_interest if k not in sample]
                                #logger.debug("Sample chunk payload %s (pre-index): %s", sample.get('chunk_index'), view)
                                if missing:
                                    logger.debug("Missing keys in sample chunk: %s", missing)
                    except Exception as e:
                        logger.debug("Failed to print sample chunk payload: %s", e, exc_info=True)

                    # Standardize on chunk-based limiting for /index
                    user_max_chunks = url_input.max_chunks if (url_input.max_chunks and url_input.max_chunks > 0) else None

                    if url_input.estimate:
                        effective_cap = int(settings.max_chunks_per_doc)
                        if user_max_chunks is not None:
                            effective_cap = min(effective_cap, int(user_max_chunks))
                        planned_chunks = min(len(chunks), effective_cap)
                        planned_chunks_total += planned_chunks
                        # Calculate tokens for the planned chunks
                        import tiktoken
                        try:
                            encoding = tiktoken.get_encoding("cl100k_base")
                            for chunk in chunks[:planned_chunks]:
                                text = chunk.get("text", "") if isinstance(chunk, dict) else chunk
                                tokens_total += len(encoding.encode(text, disallowed_special=()))
                        except Exception:
                            pass
                    else:
                        #logger.debug("Indexing HTML chunks for: %s", page['url'])
                        result = _index_chunks_with_retrieval(
                            chunks,
                            active_domain=url_input.active_domain,
                            force_delete=url_input.force_delete,
                            max_chunks=user_max_chunks,
                        )
                        planned_chunks_total += result.get("vectors_indexed", 0)
                        tokens_total += result.get("tokens_used", 0)
            
            # elif url_input.doc_type == "PDF":
            #     pdf_crawler = PDFCrawler()
            #     pdf_data = pdf_crawler.crawl(url)
            #     if settings.debug_verbose:
            #         from pprint import pformat
            #         logger.debug("PDF data retrieved: %s", pformat(pdf_data)[:settings.debug_log_truncate_chars])
                
            #     if pdf_data:
            #         for section in pdf_data['sections']:
            #             document = {
            #                 'url': url,
            #                 'text': section['content'],
            #                 'doc_type': 'PDF',
            #                 'title': pdf_data['title'],
            #                 'page_number': section['page_number']
            #             }
            #             # Standardize on chunk-based limiting for /index (no max_chars support)
            #             user_max_chunks = url_input.max_chunks if (url_input.max_chunks and url_input.max_chunks > 0) else None
            #             if url_input.estimate:
            #                 from backend.extractor.splitters import TextSplitter
            #                 # Use PDF chunk settings for PDFs
            #                 chunk_size = settings.pdf_chunk_size or len(document['text'])
            #                 splitter = TextSplitter(
            #                     chunk_size=chunk_size,
            #                     chunk_overlap=settings.pdf_chunk_overlap,
            #                     use_manual_splitter=False,
            #                 )
            #                 chunks = splitter.split_text(document.get('text', '') or '')
            #                 effective_cap = int(settings.max_chunks_per_doc)
            #                 if user_max_chunks is not None:
            #                     effective_cap = min(effective_cap, int(user_max_chunks))
            #                 planned_chunks = min(len(chunks), effective_cap)
            #                 planned_chunks_total += planned_chunks
            #                 # Calculate tokens for the planned chunks
            #                 for chunk in chunks[:planned_chunks]:
            #                     tokens_total += embeddings_manager.estimate_tokens(chunk)
            #             else:
            #                 result = embeddings_manager.index_document(
            #                     document,
            #                     force_delete=url_input.force_delete,
            #                     max_chunks=user_max_chunks
            #                 )
            #                 planned_chunks_total += result.get("vectors_indexed", 0)
            #                 tokens_total += result.get("tokens_used", 0)
        rate_per_mm = _get_embedding_rate_per_mm_tokens()
        if url_input.estimate:
            estimated_cost = (tokens_total * rate_per_mm) / 1_000_000.0
            return {
                "message": "Estimate only", 
                "chunks_planned": planned_chunks_total,
                "tokens_used": tokens_total,
                "embedding_cost": round(estimated_cost, 8),
                "errors": all_errors,
            }
        embedding_cost = (tokens_total * rate_per_mm) / 1_000_000.0
        return {
            "message": "Content indexed successfully",
            "vectors_indexed": planned_chunks_total,
            "tokens_used": tokens_total,
            "embedding_cost": round(embedding_cost, 8),
            "errors": all_errors,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", tags=["3. Search & Chat"], summary="4. Vector search")
async def search_content(search_request: SearchRequest):
    """Search indexed content"""
    qdrant_db = _build_domain_qdrant(search_request.active_domain)
    try:
        if not search_request.query:
            if search_request.query_filter and "url" in search_request.query_filter:
                results = qdrant_db.get_chunks_by_url(
                    url=search_request.query_filter["url"],
                    limit=search_request.limit
                )
                return SearchResponse(results=results, total=len(results))
            else:
                raise HTTPException(status_code=400, detail="URL must be provided in query_filter if no query is given.")
        else:
            qdrant_filter = None
            if search_request.query_filter:
                filter_conditions = []
                for key, value in search_request.query_filter.items():
                    qdrant_key = "url_lower" if key == "url" else key
                    match_value = value.lower() if isinstance(value, str) else value
                    filter_conditions.append(
                        models.FieldCondition(
                            key=qdrant_key,
                            match=models.MatchValue(value=match_value)
                        )
                    )
                qdrant_filter = models.Filter(must=filter_conditions) if filter_conditions else None
            
            logger.debug("Search: query=%s limit=%s filter=%s", search_request.query, search_request.limit, qdrant_filter)
            
            # Extract optional parameters (direct attribute access)
            score_threshold = search_request.score_threshold
            exact = search_request.exact
            with_payload = search_request.with_payload
            search_mode = str(search_request.search_mode or "dense").strip().lower()
            if search_mode not in {"dense", "hybrid", "sparse"}:
                raise HTTPException(status_code=400, detail="search_mode must be one of: dense, hybrid, sparse")
            effective_score_threshold = score_threshold if search_mode == "dense" else None
            
            # Use QdrantDB directly for search mode selection.
            if search_mode == "hybrid":
                results = qdrant_db.search_similar_hybrid(
                    query=search_request.query,
                    limit=search_request.limit,
                    query_filter=qdrant_filter,
                    score_threshold=effective_score_threshold,
                    exact=exact,
                    with_payload=with_payload,
                )
            elif search_mode == "sparse":
                results = qdrant_db.search_similar_sparse(
                    query=search_request.query,
                    limit=search_request.limit,
                    query_filter=qdrant_filter,
                    score_threshold=effective_score_threshold,
                    exact=exact,
                    with_payload=with_payload,
                )
            else:
                results = qdrant_db.search_similar(
                    query=search_request.query,
                    limit=search_request.limit,
                    query_filter=qdrant_filter,
                    score_threshold=effective_score_threshold,
                    exact=exact,
                    with_payload=with_payload
                )
            logger.debug("Search results count: %d", len(results))
            return SearchResponse(results=results, total=len(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", tags=["3. Search & Chat"], summary="5. Chat (stateless)")
async def chat_with_content(chat_request: ChatRequest, request: Request):
    """Chat with the indexed content"""
    enforce_origin_host(request)

    global chat_manager
    if chat_manager is None:
        chat_manager = ChatManager()
    try:
        # Thin handler: delegate to handle_chat(payload) and return as-is
        # First try to get the handler from the instance (chat_manager/handle_chat)
        handler = getattr(chat_manager, "handle_chat", None)
        if handler is None:
            # Fallback: import module-level handler(backend/chat/chat_manager.py)(if not bound to instance)
            from backend.chat import chat_manager as cm_module
            handler = getattr(cm_module, "handle_chat")
        # Offload blocking work to a background thread so the event loop can stream SSE
        result = await asyncio.to_thread(handler, chat_request.model_dump())
        return result
    except Exception as e:
        logger.error("Error in chat endpoint: %s", e)
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/embed", tags=["2. Ingest"], summary="3. Embed single document")
async def generate_embedding(embedding_request: EmbeddingRequest, request: Request):
    """Generate embedding for a specific document"""
    enforce_origin_host(request)

    domain_cfg = _resolve_domain_config(embedding_request.domain)
    embeddings_manager = EmbeddingsManager(active_domain=domain_cfg["effective_domain"])
    try:
        document = {
            'url': embedding_request.url,
            'text': embedding_request.content,
            'title': embedding_request.title,
            'description': embedding_request.description,
            'domain': embedding_request.domain,
            'document_type': embedding_request.document_type,
            'date': embedding_request.date,
            'chunk_index': embedding_request.chunk_index,
            'total_chunks': embedding_request.total_chunks
        }
        embeddings_manager.index_document(document)
        return {"message": "Document embedded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete/{url}", tags=["2. Ingest"], summary="4. Delete by URL")
async def delete_document(url: str, active_domain: Optional[str] = None):
    """Delete embeddings for a specific document"""
    domain_cfg = _resolve_domain_config(active_domain)
    embeddings_manager = EmbeddingsManager(active_domain=domain_cfg["effective_domain"])
    try:
        embeddings_manager.delete_document(url)
        return {"message": "Document deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug-index", tags=["5. Debug"], summary="1. Inspect Qdrant collections")
def debug_index(url: Optional[str] = None, active_domain: Optional[str] = None):
    """List current indexed collections and their first 10 documents, optionally filtered by URL"""
    domain_cfg = _resolve_domain_config(active_domain)
    embeddings_manager = EmbeddingsManager(active_domain=domain_cfg["effective_domain"])
    try:
        collections_info = embeddings_manager.qdrant.client.get_collections()
        collection_names = [col.name for col in collections_info.collections]
        debug_info = {}
        for name in collection_names:
            try:
                #logger.info("Fetching documents from collection: %s", name)
                all_docs = []
                next_offset = None
                if url:
                    url_lower = url.lower()
                    #logger.debug("Filtering for URL (lowercase): %s", url_lower)
                    qdrant_filter = models.Filter(
                        must=[
                            models.FieldCondition(
                                key="base_url_lower",
                                match=models.MatchValue(value=url_lower)
                            )
                        ]
                    )
                    # Debug: Check if any documents match this filter
                    count_resp = embeddings_manager.qdrant.client.count(
                        collection_name=name,
                        count_filter=qdrant_filter
                    )
                    #logger.debug("Found %d documents matching URL filter", count_resp.count)
                    
                    # Debug: Check the actual field names in the first few documents
                    sample_docs, _ = embeddings_manager.qdrant.client.scroll(
                        collection_name=name,
                        limit=3,
                        with_payload=True,
                        scroll_filter=qdrant_filter
                    )
                    if settings.debug_verbose:
                        for i, doc in enumerate(sample_docs):
                            logger.debug("Sample doc %d fields: %s", i, list(doc.payload.keys()))
                            if 'url' in doc.payload or 'url_lower' in doc.payload:
                                logger.debug("Sample doc %d URL fields - url: %s, url_lower: %s", i, doc.payload.get('url'), doc.payload.get('url_lower'))
                else:
                    qdrant_filter = None
                    logger.debug("No URL filter applied")
                while True:
                    docs, next_offset = embeddings_manager.qdrant.client.scroll(
                        collection_name=name,
                        limit=100,  # Batch size per scroll request (not total limit). Adjust for performance/memory.
                        with_payload=True,
                        offset=next_offset,
                        scroll_filter=qdrant_filter
                    )
                    all_docs.extend(docs)
                    if next_offset is None:
                        break
                if url and not docs:
                    continue
                # Get collection info to verify total count
                collection_info = embeddings_manager.qdrant.client.get_collection(collection_name=name)
                if settings.debug_verbose:
                    logger.debug("Qdrant collection %s info: %s", name, collection_info)
                logger.debug("Total points in collection: %s", getattr(collection_info, 'points_count', 'n/a'))
                
                #logger.info("Scroll response for %s: %d docs retrieved", name, len(all_docs))
                if all_docs:
                    #logger.debug("Found %d total documents in collection %s", len(all_docs), name)
                    def get_sort_key(d):
                        return (
                            int(d.payload.get("section_index") or 0),
                            int(d.payload.get("subsection_index") or 0),
                            int(d.payload.get("chunk_index") or 0)
                        )
                    
                    sorted_docs = sorted(all_docs, key=get_sort_key)
                    #logger.debug("After sorting, have %d documents", len(sorted_docs))
                    if settings.debug_verbose:
                        for d in sorted_docs:
                            logger.debug(
                                "URL: %s | Section: %s | Subsection: %s | SectionIdx: %s | SubsectionIdx: %s | ChunkIdx: %s",
                                d.payload.get('url', ''), d.payload.get('section', ''), d.payload.get('subsection', ''),
                                d.payload.get('section_index'), d.payload.get('subsection_index'), d.payload.get('chunk_index')
                            )
                    debug_info[name] = sorted_docs
                else:
                    debug_info[name] = f"No documents found in collection {name}"
            except Exception as inner_e:
                logger.error("Error retrieving documents from %s: %s", name, inner_e)
                debug_info[name] = f"Error retrieving documents: {str(inner_e)}"
        if settings.debug_verbose:
            from pprint import pformat
            logger.debug("Debug info to return: %s", pformat(debug_info)[:settings.debug_log_truncate_chars])
        return {"collections": debug_info}
    except Exception as e:
        logger.exception("Top-level exception in /debug-index")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list-docs-debug", tags=["5. Debug"], summary="List all documents in Qdrant (debug)")
def list_docs(url: Optional[str] = None, active_domain: Optional[str] = None):
    """Return a JSON list of all documents in Qdrant with selected fields.

    The returned structure is: {"documents": [ {document_title, url, total_chunks, updated_at, collection} ]}
    """
    domain_cfg = _resolve_domain_config(active_domain)
    embeddings_manager = EmbeddingsManager(active_domain=domain_cfg["effective_domain"])
    try:
        collection_docs = []
        # Restrict to the configured collection name only
        configured = domain_cfg.get("collection_name") or getattr(settings, 'collection_name', None)
        if not configured:
            raise HTTPException(status_code=500, detail='No collection_name configured in settings')
        collection_names = [configured]
        for name in collection_names:
            try:
                all_docs = []
                next_offset = None
                if url:
                    url_lower = url.lower()
                    qdrant_filter = models.Filter(
                        must=[
                            models.FieldCondition(
                                key="base_url_lower",
                                match=models.MatchValue(value=url_lower)
                            )
                        ]
                    )
                else:
                    qdrant_filter = None
                while True:
                    docs, next_offset = embeddings_manager.qdrant.client.scroll(
                        collection_name=name,
                        limit=100,
                        with_payload=True,
                        offset=next_offset,
                        scroll_filter=qdrant_filter
                    )
                    all_docs.extend(docs)
                    if next_offset is None:
                        break
                # Aggregate by base_url: count chunks per base_url, keep title and most recent updated_at
                agg = {}
                for d in all_docs:
                    p = d.payload or {}
                    base = p.get('base_url') or p.get('url') or p.get('url_lower')
                    if not base:
                        continue
                    # If a url filter was provided, ensure match
                    if url and url.lower() not in str(base).lower():
                        continue
                    title = p.get('title')
                    updated_at = p.get('updated_at') or p.get('date')
                    entry = agg.get(base)
                    if not entry:
                        agg[base] = {
                            'document_title': title,
                            'url': base,
                            'total_chunks': 1,
                            'updated_at': updated_at,
                            'collection': name,
                        }
                    else:
                        entry['total_chunks'] = entry.get('total_chunks', 0) + 1
                        # choose the most recent updated_at if present (lexicographic fallback)
                        if updated_at:
                            prev = entry.get('updated_at')
                            if not prev or str(updated_at) > str(prev):
                                entry['updated_at'] = updated_at
                # extend aggregated results
                collection_docs.extend(list(agg.values()))
            except Exception as inner_e:
                logger.error("Error retrieving documents from %s: %s", name, inner_e)
                # skip collection on error
                continue
        return {"documents": collection_docs}
    except Exception as e:
        logger.exception("Top-level exception in /list-docs")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/delete_index",
    tags=["1. UI Pages"],
    summary="6. Delete-by-URL admin page (HTML)",
    response_class=HTMLResponse,
)
async def delete_index_page():
    """Serve the delete-index HTML page used for delete-by-URL admin operations."""
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    return FileResponse(os.path.join(frontend_dir, "delete-index.html"))


@app.get(
    "/prompt-registry",
    tags=["1. UI Pages"],
    summary="7. Prompt Registry editor (HTML)",
    response_class=HTMLResponse,
)
async def prompt_registry_page():
    """Serve the prompt-registry HTML page for editing YAML prompts by domain/stage."""
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    return FileResponse(os.path.join(frontend_dir, "prompt-registry.html"))


class DeleteByBaseURLRequest(BaseModel):
    """Request body for deleting Qdrant points grouped by base URL.

    Either `url` or `base_url` may be provided; if `base_url` is omitted,
    it will be derived from `url` by stripping any fragment.
    """

    url: Optional[str] = None
    base_url: Optional[str] = None
    active_domain: Optional[str] = None


def _strip_fragment_url(u: str) -> str:
    if not u:
        return u
    p = urlsplit(u)
    return urlunsplit((p.scheme, p.netloc, p.path, p.query, ""))


@app.get(
    "/admin/delete-preview",
    tags=["4. Index Admin"],
    summary="Preview Qdrant chunks matched by base_url_lower before delete",
)
def delete_preview(
    url: str = Query(..., description="Full URL of the document to delete"),
    sample_limit: int = Query(5, ge=1, le=500),
    active_domain: Optional[str] = Query(None, description="Optional domain for selecting collection/model"),
):
    """Return a count and sample of chunks that would be deleted for the given URL's base_url.

    - Normalizes the provided URL to a base URL (no fragment) and lowercases it.
    - Counts all matching chunks via `base_url_lower`.
    - Returns up to `sample_limit` example chunks for review.
    """

    qdrant_db = _build_domain_qdrant(active_domain)
    try:
        base = _strip_fragment_url(url)
        total = qdrant_db.count_points_by_base_url(base)
        sample_chunks = qdrant_db.get_chunks_by_base_url(base, limit=sample_limit)
        # Optionally, keep sample payloads smaller by trimming long text fields
        max_text_chars = int(getattr(settings, "debug_log_truncate_chars", 500))
        for ch in sample_chunks:
            payload = ch.get("payload") or {}
            txt = payload.get("text")
            if isinstance(txt, str) and len(txt) > max_text_chars:
                payload["text"] = txt[:max_text_chars] + "…"
            ch["payload"] = payload

        return {
            "input_url": url,
            "base_url": base,
            "base_url_lower": (base or "").lower(),
            "total_chunks": int(total),
            "sample_limit": int(sample_limit),
            "sample_chunks": sample_chunks,
        }
    except Exception as e:
        logger.exception("Error in /admin/delete-preview: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/admin/delete-by-base-url",
    tags=["4. Index Admin"],
    summary="Delete Qdrant chunks by base_url_lower",
)
def delete_by_base_url(request: DeleteByBaseURLRequest, http_request: Request):
    """Delete all Qdrant chunks whose `base_url_lower` matches the derived base URL.

    You can provide either:
    - `url`: a full URL, from which base_url will be derived, or
    - `base_url`: used directly.
    """
    qdrant_db = _build_domain_qdrant(request.active_domain)
    try:
        base = request.base_url or _strip_fragment_url(request.url or "")
        if not base:
            raise HTTPException(status_code=400, detail="Either url or base_url must be provided")

        deleted = qdrant_db.delete_by_base_url(base)
        return {
            "base_url": base,
            "base_url_lower": (base or "").lower(),
            "deleted_points": int(deleted),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /admin/delete-by-base-url: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

class ModelConfig(BaseModel):
    """Model configuration response model."""
    # Concrete model names (for debug / compatibility)
    embedding_model: Optional[str] = None
    re_ranker_model: Optional[str] = None
    query_rewrite_model: Optional[str] = None
    summarizer_model: Optional[str] = None
    inference_model: Optional[str] = None

    # Stable model profile keys from Settings (source of truth for defaults)
    embedding_model_key: Optional[str] = None
    rewrite_model_key: Optional[str] = None
    rerank_model_key: Optional[str] = None
    summarizer_model_key: Optional[str] = None
    inference_model_key: Optional[str] = None

    # Collection and domain information
    collection_name: Optional[str] = None
    active_domain: Optional[str] = None

    # Runtime defaults exposed for frontend controls
    score_threshold: Optional[float] = None
    top_k: Optional[int] = None
    summarizer_max_input_tokens: Optional[int] = None
    summarizer_max_output_tokens: Optional[int] = None
    inference_temperature: Optional[float] = None
    inference_top_p: Optional[float] = None
    raw_tail_turns: Optional[int] = None
    max_inference_output_tokens: Optional[int] = None
    enable_tools: Optional[bool] = None
    enable_query_rewrite: Optional[bool] = None
    rewrite_confidence_threshold: Optional[float] = None
    rewrite_tail_turns: Optional[int] = None
    rewrite_summary_turns: Optional[int] = None
    inference_context_rows: Optional[int] = None


class PromptRegistryUpdateRequest(BaseModel):
    """Request model for writing prompt registry YAML."""

    registry: Dict[str, Any]


class ToolRegistryUpdateRequest(BaseModel):
    """Request model for writing tool registry YAML."""

    registry: Dict[str, Any]


def _prompt_registry_path() -> Path:
    """Resolve configured prompt registry path to absolute path."""
    raw_path = str(getattr(settings, "inference_prompt_registry_path", "") or "").strip() or "prompts/prompt_registry.yaml"
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / candidate


def _tool_registry_path() -> Path:
    """Resolve tool registry path to absolute path."""
    raw_path = str(getattr(settings, "tool_registry_path", "") or "").strip() or "prompts/tool_registry.yaml"
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / candidate


def _validate_prompt_registry_shape(registry: Dict[str, Any]) -> None:
    """Basic schema validation for prompt registry editor writes."""
    if not isinstance(registry, dict):
        raise HTTPException(status_code=400, detail="Registry must be a JSON object")

    global_defaults = registry.get("global_defaults")
    if not isinstance(global_defaults, dict):
        raise HTTPException(status_code=400, detail="Missing required key: global_defaults")

    required_stages = ["inference", "rewrite", "rerank", "summary", "tools_synth"]
    for stage in required_stages:
        stage_cfg = global_defaults.get(stage)
        if not isinstance(stage_cfg, dict):
            raise HTTPException(status_code=400, detail=f"Missing required stage config: global_defaults.{stage}")
        system_instruction = stage_cfg.get("system_instruction")
        if not isinstance(system_instruction, str) or not system_instruction.strip():
            raise HTTPException(status_code=400, detail=f"Missing required system_instruction for stage: {stage}")

    stages_requiring_full_payload = ["inference", "rewrite", "rerank"]
    for stage in stages_requiring_full_payload:
        stage_cfg = global_defaults.get(stage) or {}
        user_messages = stage_cfg.get("user_messages")
        if not isinstance(user_messages, list):
            raise HTTPException(status_code=400, detail=f"Missing user_messages list for stage: {stage}")

        has_full_payload = False
        for item in user_messages:
            if not isinstance(item, dict):
                continue
            if str(item.get("name") or "") == "full_payload":
                template = item.get("template")
                if isinstance(template, str) and template.strip():
                    has_full_payload = True
                    break
        if not has_full_payload:
            raise HTTPException(status_code=400, detail=f"Missing full_payload template for stage: {stage}")


def _validate_tool_registry_shape(registry: Dict[str, Any]) -> None:
    """Basic schema validation for tool registry editor writes."""
    if not isinstance(registry, dict):
        raise HTTPException(status_code=400, detail="Registry must be a JSON object")

    artifact_injection = registry.get("artifact_injection")
    if not isinstance(artifact_injection, dict):
        raise HTTPException(status_code=400, detail="Missing required key: artifact_injection")

    enabled = artifact_injection.get("enabled")
    if not isinstance(enabled, bool):
        raise HTTPException(status_code=400, detail="artifact_injection.enabled must be boolean")

    allowed_tools = artifact_injection.get("allowed_tools")
    if not isinstance(allowed_tools, list):
        raise HTTPException(status_code=400, detail="artifact_injection.allowed_tools must be a list")
    for i, name in enumerate(allowed_tools):
        if not isinstance(name, str) or not name.strip():
            raise HTTPException(status_code=400, detail=f"artifact_injection.allowed_tools[{i}] must be a non-empty string")

    security = artifact_injection.get("security")
    if not isinstance(security, dict):
        raise HTTPException(status_code=400, detail="artifact_injection.security must be an object")

    max_artifact_chars = security.get("max_artifact_chars")
    if not isinstance(max_artifact_chars, int) or max_artifact_chars <= 0:
        raise HTTPException(status_code=400, detail="artifact_injection.security.max_artifact_chars must be a positive integer")

    allowed_artifact_types = security.get("allowed_artifact_types")
    if not isinstance(allowed_artifact_types, list) or not allowed_artifact_types:
        raise HTTPException(status_code=400, detail="artifact_injection.security.allowed_artifact_types must be a non-empty list")
    for i, value in enumerate(allowed_artifact_types):
        if not isinstance(value, str) or not value.strip():
            raise HTTPException(status_code=400, detail=f"artifact_injection.security.allowed_artifact_types[{i}] must be a non-empty string")

    allowed_injection_modes = security.get("allowed_injection_modes")
    if not isinstance(allowed_injection_modes, list) or not allowed_injection_modes:
        raise HTTPException(status_code=400, detail="artifact_injection.security.allowed_injection_modes must be a non-empty list")
    for i, value in enumerate(allowed_injection_modes):
        if not isinstance(value, str) or not value.strip():
            raise HTTPException(status_code=400, detail=f"artifact_injection.security.allowed_injection_modes[{i}] must be a non-empty string")

    enforce_placeholder_format = security.get("enforce_placeholder_format")
    if not isinstance(enforce_placeholder_format, bool):
        raise HTTPException(status_code=400, detail="artifact_injection.security.enforce_placeholder_format must be boolean")

    tools = registry.get("tools")
    if not isinstance(tools, list) or not tools:
        raise HTTPException(status_code=400, detail="Missing required non-empty key: tools")

    seen_names: set[str] = set()
    for idx, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise HTTPException(status_code=400, detail=f"tools[{idx}] must be an object")

        name = str(tool.get("name") or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail=f"tools[{idx}].name is required")
        if name in seen_names:
            raise HTTPException(status_code=400, detail=f"Duplicate tool name in registry: {name}")
        seen_names.add(name)

        artifact = tool.get("artifact")
        if not isinstance(artifact, dict):
            raise HTTPException(status_code=400, detail=f"tools[{idx}].artifact must be an object")

        produces_artifact = artifact.get("produces_artifact")
        if not isinstance(produces_artifact, bool):
            raise HTTPException(status_code=400, detail=f"tools[{idx}].artifact.produces_artifact must be boolean")

        if produces_artifact:
            for key in ("artifact_type", "artifact_key", "injection_mode", "placeholder"):
                value = str(artifact.get(key) or "").strip()
                if not value:
                    raise HTTPException(
                        status_code=400,
                        detail=f"tools[{idx}].artifact.{key} is required when produces_artifact=true",
                    )


@app.get(
    "/api/prompt-registry",
    tags=["5. Debug"],
    summary="Get prompt registry YAML",
)
async def get_prompt_registry():
    """Return parsed prompt registry YAML for UI editing."""
    path = _prompt_registry_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Prompt registry not found at {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            registry = yaml.safe_load(f) or {}

        global_defaults = registry.get("global_defaults") if isinstance(registry, dict) else {}
        stages = list(global_defaults.keys()) if isinstance(global_defaults, dict) else []
        domains = sorted(list((registry.get("domains") or {}).keys())) if isinstance(registry, dict) else []
        return {
            "registry_path": str(path),
            "registry": registry,
            "stages": stages,
            "domains": domains,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read prompt registry: {e}")


@app.get(
    "/api/tool-registry",
    tags=["5. Debug"],
    summary="Get tool registry YAML",
)
async def get_tool_registry():
    """Return parsed tool registry YAML for UI editing."""
    path = _tool_registry_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Tool registry not found at {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            registry = yaml.safe_load(f) or {}

        tools = registry.get("tools") if isinstance(registry, dict) else []
        tool_names = [str(t.get("name") or "") for t in tools if isinstance(t, dict)]
        return {
            "registry_path": str(path),
            "registry": registry,
            "tools": tool_names,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read tool registry: {e}")


@app.put(
    "/api/prompt-registry",
    tags=["5. Debug"],
    summary="Update prompt registry YAML",
)
async def update_prompt_registry(payload: PromptRegistryUpdateRequest, request: Request):
    """Validate and write prompt registry YAML from editor UI."""
    enforce_origin_host(request)
    registry = payload.registry or {}
    _validate_prompt_registry_shape(registry)

    path = _prompt_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = path.with_suffix(path.suffix + ".bak")

    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                previous = f.read()
            with open(backup_path, "w", encoding="utf-8") as bf:
                bf.write(previous)

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(registry, f, sort_keys=False, allow_unicode=True)

        removed = clear_prompt_registry_cache(str(path))

        return {
            "ok": True,
            "registry_path": str(path),
            "backup_path": str(backup_path) if backup_path.exists() else None,
            "cache_entries_cleared": int(removed),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write prompt registry: {e}")


@app.put(
    "/api/tool-registry",
    tags=["5. Debug"],
    summary="Update tool registry YAML",
)
async def update_tool_registry(payload: ToolRegistryUpdateRequest, request: Request):
    """Validate and write tool registry YAML from editor UI."""
    enforce_origin_host(request)
    registry = payload.registry or {}
    _validate_tool_registry_shape(registry)

    path = _tool_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = path.with_suffix(path.suffix + ".bak")

    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                previous = f.read()
            with open(backup_path, "w", encoding="utf-8") as bf:
                bf.write(previous)

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(registry, f, sort_keys=False, allow_unicode=True)

        removed = clear_tool_registry_cache(str(path))

        return {
            "ok": True,
            "registry_path": str(path),
            "backup_path": str(backup_path) if backup_path.exists() else None,
            "cache_entries_cleared": int(removed),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write tool registry: {e}")


@app.post(
    "/api/prompt-registry/reload-cache",
    tags=["5. Debug"],
    summary="Reload prompt registry cache",
)
async def reload_prompt_registry_cache(request: Request):
    """Clear prompt registry cache so latest YAML is used on next prompt resolution."""
    enforce_origin_host(request)
    path = _prompt_registry_path()
    removed = clear_prompt_registry_cache(str(path))
    return {
        "ok": True,
        "registry_path": str(path),
        "cache_entries_cleared": int(removed),
        "message": "Prompt registry cache cleared. New prompts will be used on the next request.",
    }


@app.post(
    "/api/tool-registry/reload-cache",
    tags=["5. Debug"],
    summary="Reload tool registry cache",
)
async def reload_tool_registry_cache(request: Request):
    """Clear tool registry cache so latest YAML is used on next tool synthesis."""
    enforce_origin_host(request)
    path = _tool_registry_path()
    removed = clear_tool_registry_cache(str(path))
    return {
        "ok": True,
        "registry_path": str(path),
        "cache_entries_cleared": int(removed),
        "message": "Tool registry cache cleared. New tool artifact policy will be used on the next request.",
    }


@app.get(
    "/tool-registry",
    tags=["1. UI Pages"],
    summary="8. Tool Registry editor (HTML)",
    response_class=HTMLResponse,
)
async def tool_registry_page():
    """Serve the tool-registry HTML page for editing tool artifact metadata."""
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    return FileResponse(os.path.join(frontend_dir, "tool-registry.html"))

@app.get("/api/config", response_model=ModelConfig, tags=["5. Debug"])
async def get_model_config():
    """
    Get the current model configuration.
    This endpoint returns the models used for different tasks in the application.
    Returns None for any model that is not configured.
    """
    def get_model_setting(name):
        value = getattr(settings, name, None)
        return value if value is not None else None
    
    # Resolve embedding model from provider key to actual model name
    embedding_model_display = None
    try:
        from backend.embeddings.specs import resolve_embedding_spec
        emb_spec = resolve_embedding_spec(settings)
        if emb_spec and emb_spec.get("model"):
            embedding_model_display = emb_spec["model"]
    except Exception as e:
        logger.debug("Failed to resolve embedding spec for display: %s", e)
        # Fallback to model key if resolution fails
        embedding_model_display = get_model_setting("embedding_model_key")
    
    return ModelConfig(
        # Concrete model names
        embedding_model=embedding_model_display,
        re_ranker_model=get_model_setting("re_ranker_model"),
        query_rewrite_model=get_model_setting("rewrite_model"),
        summarizer_model=get_model_setting("summarizer_model"),
        inference_model=get_model_setting("inference_model"),

        # Stable model profile keys from Settings
        embedding_model_key=get_model_setting("embedding_model_key"),
        rewrite_model_key=get_model_setting("rewrite_model_key"),
        rerank_model_key=get_model_setting("rerank_model_key"),
        summarizer_model_key=get_model_setting("summarizer_model_key"),
        inference_model_key=get_model_setting("inference_model_key"),

        # Collection and domain information
        collection_name=get_model_setting("collection_name"),
        active_domain=get_model_setting("active_domain"),

        # Runtime defaults so the frontend can pre-populate form fields
        score_threshold=get_model_setting("score_threshold"),
        top_k=get_model_setting("top_k"),
        summarizer_max_input_tokens=get_model_setting("summarizer_max_input_tokens"),
        summarizer_max_output_tokens=get_model_setting("summarizer_max_output_tokens"),
        inference_temperature=get_model_setting("inference_temperature"),
        inference_top_p=get_model_setting("inference_top_p"),
        raw_tail_turns=get_model_setting("raw_tail_turns"),
        max_inference_output_tokens=get_model_setting("max_inference_output_tokens"),
        enable_tools=get_model_setting("enable_tools"),
        enable_query_rewrite=get_model_setting("enable_query_rewrite"),
        rewrite_confidence_threshold=get_model_setting("rewrite_confidence_threshold"),
        rewrite_tail_turns=get_model_setting("rewrite_tail_turns"),
        rewrite_summary_turns=get_model_setting("rewrite_summary_turns"),
        inference_context_rows=get_model_setting("inference_context_rows"),
    )

@app.get(
    "/api/config/api-defaults",
    tags=["5. Debug"],
    summary="Get frontend configuration defaults",
    response_description="A JSON object containing default values for frontend forms",
    responses={
        200: {
            "description": "Successfully retrieved frontend configuration defaults",
            "content": {
                "application/json": {
                    "example": {
                        "html": {
                            "maxChunks": 0,
                            "skipSections": ["References", "External links"],
                            "estimate": False,
                            "forceDelete": False
                        },
                        "pdf": {
                            "maxChunks": 0,
                            "estimate": False,
                            "forceDelete": False
                        },
                        "mediawiki": {
                            "maxChunks": 0,
                            "skipSections": ["References", "External links"],
                            "estimate": False,
                            "forceDelete": False,
                            "apiUrl": "https://en.wikipedia.org/w/api.php",
                            "userAgent": "WebsiteChatAgent/0.1"
                        }
                    }
                }
            }
        },
        500: {"description": "Internal server error while retrieving configuration"}
    }
)
async def get_api_defaults():
    """
    Retrieve the default configuration values for frontend forms.
    
    This endpoint provides the centralized configuration that defines the default
    values used in the frontend's HTML, PDF, and MediaWiki indexing forms. The values
    are defined in `backend/core/config.py` and are converted to camelCase for
    frontend compatibility.
    
    Returns:
        dict: A dictionary containing default configuration values for all frontend forms.
        The structure matches the frontend's expected format with camelCase keys.
    """
    try:
        # Get the frontend config and add the collection name
        config = settings.frontend.to_dict()
        config['collection_name'] = settings.collection_name
        return config
    except Exception as e:
        print(f"Error in get_api_defaults: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class BatchURLItem(BaseModel):
    """Individual URL to process in a batch"""
    url: str
    doc_type: str  # 'html', 'pdf', or 'mediawiki'
    skip_sections: List[str] = Field(
        default_factory=lambda: ["References", "External links", "See also", "Further reading"]
    )
    active_domain: Optional[str] = None
    user_agent: Optional[str] = None
    api_url: Optional[str] = None  # For MediaWiki API URL override

class BatchRequest(BaseModel):
    """Request model for batch processing documents"""
    items: List[BatchURLItem]
    max_chunks: Optional[int] = None
    estimate: bool = True  # Default to True for safety
    force_delete: bool = False
    filename: Optional[str] = None
    active_domain: Optional[str] = None

from fastapi.responses import StreamingResponse, JSONResponse
import json
import asyncio
import os
from typing import List, Dict
import aiohttp
from fastapi import HTTPException

async def process_item(item, request, settings):
    try:
        # Handle local file paths for PDFs
        logger.info(f"Processing item: {item.url}")
        if item.doc_type.lower() == 'pdf' and item.url.startswith('file://'):
            # Turn "file:///app/data/..." into "/app/data/..."
            parsed = urlsplit(item.url)          # scheme=file, path=/app/...
            file_path = parsed.path             # "/app/data/…/file.pdf"
            logger.info (f"file_path {file_path}")
            logger.info (f"original file name {os.path.basename(file_path)}")
            try:
                # Create a file-like object from the local file
                with open(file_path, 'rb') as f:
                    files = {'file': (os.path.basename(file_path), f, 'application/pdf')}
                    form_data = {
                        'estimate': str(request.estimate).lower(),
                        'force_delete': str(request.force_delete).lower()
                    }
                    if request.max_chunks is not None:
                        form_data['max_chunks'] = str(request.max_chunks)
                    
                    # Make an internal request to the /pdf endpoint
                    async with aiohttp.ClientSession() as session:
                        # Read the file and encode as base64
                        import base64
                        with open(file_path, 'rb') as f:
                            file_content = f.read()
                            file_b64 = base64.b64encode(file_content).decode('utf-8')
                        
                        # Create JSON payload with base64-encoded file
                        payload = {
                            'file': file_b64,
                            'estimate': request.estimate,
                            'force_delete': request.force_delete,
                            'active_domain': item.active_domain or request.active_domain,
                            'url': item.url,
                            'filename': os.path.basename(file_path)
                        }
                        if request.max_chunks is not None:
                            payload['max_chunks'] = request.max_chunks
                            
                        # Using default FastAPI port for local development
                        headers = {'Content-Type': 'application/json'}
                        async with session.post(
                            'http://localhost:8000/pdf',
                            json=payload,
                            headers=headers
                        ) as response:
                            response.raise_for_status()
                            result = await response.json()
                            return {
                                "url": item.url,
                                "doc_type": item.doc_type,
                                "status": "success",
                                "result": result
                            }
                            
            except Exception as e:
                logger.error(f"Error processing local file {file_path}: {str(e)}")
                return {
                    "url": item.url,
                    "doc_type": item.doc_type,
                    "status": "error",
                    "error": f"Failed to process local file {file_path}: {str(e)}"
                }
        
        # Route based on doc_type for non-file URLs
        if item.doc_type.lower() == 'pdf':
            input_data = PDFInput(
                url=item.url,
                max_chunks=request.max_chunks,
                force_delete=request.force_delete,
                active_domain=item.active_domain or request.active_domain,
                estimate=request.estimate
            )
            handler = index_pdf
            
        elif item.doc_type.lower() == 'mediawiki':
            input_data = MediaWikiURLInput(
                url=item.url,
                max_chunks=request.max_chunks,
                force_delete=request.force_delete,
                active_domain=item.active_domain or request.active_domain,
                estimate=request.estimate
            )
            handler = index_mediawiki_url
            
        else:  # Default to HTML
            input_data = URLInput(
                urls=[item.url],
                doc_type='html',
                max_chunks=request.max_chunks,
                force_delete=request.force_delete,
                active_domain=item.active_domain or request.active_domain,
                estimate=request.estimate,
                skip_sections=item.skip_sections,
                user_agent=item.user_agent or settings.default_user_agent
            )
            handler = index_content
        
        # Process the document
        response = await handler(input_data)
        
        return {
            "url": item.url,
            "doc_type": item.doc_type,
            "status": "success",
            "result": response
        }
        
    except HTTPException as e:
        return {
            "url": item.url,
            "doc_type": item.doc_type,
            "status": "error",
            "error": e.detail,
            "status_code": e.status_code
        }
    except Exception as e:
        return {
            "url": item.url,
            "doc_type": item.doc_type,
            "status": "error",
            "error": str(e)
        }

@app.post(
    "/batch/process_docs",
    tags=["2. Ingest"],
    summary="Batch process multiple documents with streaming updates",
    responses={
        200: {"description": "Batch processing completed with streaming updates"},
        500: {"description": "Internal server error"}
    }
)
async def batch_process_docs(batch_request: BatchRequest, request: Request):
    """
    Process multiple documents (PDFs, web pages, or MediaWiki) with streaming updates.
    
    - Supports mixed document types in one batch
    - Global settings for max_chunks, estimate, and force_delete
    - Per-document skip_sections customization
    - Streams progress updates in real-time
    """
    enforce_origin_host(request)

    async def stream_results():
        total_items = len(batch_request.items)
        successful = 0
        errors = 0
        total_chunks = 0
        total_tokens = 0
        total_cost = 0.0
        
        # Process each item one by one
        for i, item in enumerate(batch_request.items):
            # Send progress update
            progress = (i / total_items) * 100
            yield json.dumps({
                "progress": progress,
                "message": f"Processing {i+1} of {total_items}: {item.url}"
            }) + "\n"
            
            # Process the item
            result = await process_item(item, request, settings)
            
            # Update totals
            if result.get('status') == 'success' and 'result' in result:
                res = result['result']
                if isinstance(res, dict):
                    if 'chunks_planned' in res:
                        total_chunks += res['chunks_planned']
                    elif 'chunks_indexed' in res:
                        total_chunks += res['chunks_indexed']
                    elif 'vectors_indexed' in res:
                        total_chunks += res['vectors_indexed']

                    if 'tokens_used' in res:
                        total_tokens += res['tokens_used']
                    if 'embedding_cost' in res:
                        total_cost += res['embedding_cost']
                successful += 1
            else:
                errors += 1
            
            # Send the result
            yield json.dumps(result) + "\n"
            
            # Small delay to allow the client to process the update
            await asyncio.sleep(0.1)
        
        # Send final summary
        summary = {
            "summary": {
                "type": "estimate" if batch_request.estimate else "index",
                "total_items": total_items,
                "successful_items": successful,
                "failed_items": errors,
                "total_chunks": total_chunks,
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "progress": 100,
                "message": "Processing complete"
            }
        }
        yield json.dumps(summary) + "\n"
    
    return StreamingResponse(stream_results(), media_type="text/event-stream")

@app.post("/batch/list-pdfs", tags=["2. Ingest"])
async def list_pdfs(request: Request, folder_path: str = Form(...)):
    """
    List all PDF files in the specified directory.
    
    Args:
        folder_path: The absolute path to the directory to scan for PDF files
        
    Returns:
        List of PDF files with their paths and metadata
    """
    enforce_origin_host(request)

    if not os.path.isdir(folder_path):
        raise HTTPException(status_code=400, detail="Invalid directory path")
    
    try:
        pdf_files = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.pdf'):
                full_path = os.path.join(folder_path, filename)
                pdf_files.append({
                    "url": f"file://{os.path.abspath(full_path)}",
                    "filename": filename,
                    "size": os.path.getsize(full_path)
                })
        
        return {"pdf_files": pdf_files}
    except Exception as e:
        logger.error(f"Error listing PDFs in {folder_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error scanning directory: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


# Endpoint to update a specific payload field for all chunks matching the given URL.
from fastapi import HTTPException

@app.post("/update_payload", tags=["4. Index Admin"], summary="1. Bulk update payload by URL")
async def update_payload_field(update_request: PayloadUpdateRequest, request: Request):
    """
    Update a specific payload field for all chunks matching the given URL.
    """
    enforce_origin_host(request)

    qdrant_db = _build_domain_qdrant(update_request.active_domain)
    try:
        updated = qdrant_db.update_payload_by_url(update_request)
        return {
            "message": f"Updated payloads with key '{update_request.meta_key}' for URL '{update_request.url}'",
            "updated": updated
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
