import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
from qdrant_client.http import models
from backend.core.config import settings
from backend.embeddings.embeddings_manager import EmbeddingsManager
from backend.crawler.crawler import WebCrawler
from backend.crawler.pdf_crawler import PDFCrawler
from backend.extractor.extractor import ContentExtractor
from backend.embeddings.schemas import EmbeddingRequest, SearchRequest, SearchResponse, ChatRequest, ChatResponse, MediaWikiURLInput, URLInput, PayloadUpdateRequest
from backend.db.qdrant_db import QdrantDB
from backend.extractor.mediawiki_extractor import MediaWikiExtractor
from backend.chat.chat_manager import ChatManager

# Configure logging
import logging.config

# Configure logging to work with uvicorn
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            '()': 'uvicorn.logging.DefaultFormatter',
            'fmt': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
    },
    'handlers': {
        'default': {
            'formatter': 'default',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stderr',
        },
    },
    'loggers': {
        'trafilatura': {
            'level': 'WARNING',
            'handlers': ['default'],
            'propagate': False
        },
        '': {
            'handlers': ['default'],
            'level': 'DEBUG',
        },
        'backend': {
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': False,
        },
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# --- Swagger/OpenAPI tags & UI ordering ---
tags_metadata = [
    {"name": "1. UI Pages", "description": "Frontend pages served by FastAPI (HTML)."},
    {"name": "2. Ingest", "description": "Bring content in: MediaWiki, URLs, PDFs; embed documents; delete by URL."},
    {"name": "3. Search & Chat", "description": "Vector search and conversational endpoints over indexed content."},
    {"name": "4. Index Admin", "description": "Index maintenance utilities (payload updates, etc.)."},
    {"name": "5. Debug", "description": "Developer diagnostics and inspection utilities."},
]

app = FastAPI(
    title="Website Chat Agent API",
    openapi_tags=tags_metadata,
    swagger_ui_parameters={
        "tagsSorter": "alpha",
        "operationsSorter": "alpha",
    },
)

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

from typing import Optional
@app.post("/mediawiki/url", tags=["2. Ingest"], summary="1. Index MediaWiki URL")
async def index_mediawiki_url(
    mediawiki_input: MediaWikiURLInput,
    api_url: Optional[str] = Query(None, description="Override MediaWiki API endpoint (defaults to settings)"),
    ua: Optional[str] = Query(None, description="Override User-Agent for MediaWiki requests"),
):
    """
    Index content from MediaWiki wikitext, optionally limiting number of chunks indexed by max_chunks.
    You can override the target MediaWiki API with `api_url` and the User-Agent with `ua` per request.
    """
    global embeddings_manager
    if embeddings_manager is None:
        logger.info("Initializing embeddings manager")
        embeddings_manager = EmbeddingsManager()
        logger.info("Embeddings manager initialized")
    try:
        extractor = MediaWikiExtractor(api_url=api_url, user_agent=ua)
        url = mediawiki_input.url
        max_chunks = mediawiki_input.max_chunks
        skip_sections = mediawiki_input.skip_sections
        print(f"DEBUG: Received MediaWiki URL: {url}")
        # Use extractor.parse_from_url directly, passing skip_sections
        chunks = extractor.parse_from_url(url, skip_sections=skip_sections)
        print(f"DEBUG: Parsed {len(chunks)} chunks from url")
        print(f"DEBUG: Total chunks returned by extractor: {len(chunks)}")
        # Limit the number of chunks indexed if max_chunks is set
        if max_chunks is not None:
            chunks = chunks[:max_chunks]
        print(f"DEBUG: Indexing {len(chunks)} chunks")
        embeddings_manager.index_chunks(chunks, force_delete=mediawiki_input.force_delete)
        return {"message": "MediaWiki content indexed successfully", "chunks_indexed": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from backend.chat.chat_manager import ChatManager
from fastapi import Depends, HTTPException
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

chat_session_manager = ChatSessionManager()

@app.post("/chat/session", tags=["3. Search & Chat"], summary="1. Create chat session")
async def create_chat_session():
    """Create a new chat session"""
    session_id = chat_session_manager.create_session()
    return {"session_id": session_id}

@app.post("/chat/{session_id}", tags=["3. Search & Chat"], summary="2. Chat (session)")
async def chat_endpoint(session_id: str, chat_request: ChatRequest):
    """Process a chat message with context"""
    try:
        # Get session context
        session = chat_session_manager.get_session(session_id)
        
        # Get relevant context from chat history
        context = chat_session_manager.get_context(session_id)
        
        # Process chat message with context
        response = await chat_manager.chat(
            message=chat_request.message,
            context=context,
            use_web_search=chat_request.use_web_search
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
            "sources": response["sources"]
        })
        
        return ChatResponse(
            response=response["response"],
            sources=response["sources"]
        )
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
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
        logger.error(f"Error getting chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index", tags=["2. Ingest"], summary="2. Index URLs / PDFs")
async def index_content(url_input: URLInput):
    """Index content from URLs (HTML or PDF)"""
    global embeddings_manager
    if embeddings_manager is None:
        embeddings_manager = EmbeddingsManager()
    try:
        #print("DEBUG: Entered index_content route")
        #print("Received input:", url_input.dict())  # Debugging line
        # Process each URL
        for url in url_input.urls:
            if url_input.doc_type == "HTML":
                crawler = WebCrawler(
                    url,
                    force_crawl=url_input.force_crawl
                )
                pages = await crawler.crawl(url, 1)  # Depth 1 for now
                print(f"DEBUG: Crawled {len(pages)} page(s) from {url}")
                
                for page in pages:
                    print("DEBUG: Extracting content from page:", page['url'])
                    print("Raw page content length:", len(page['content']))
                    content = ContentExtractor.extract_content(page['content'], url=page['url'])
                    if content:
                        if url_input.max_chars:
                            content["max_chars"] = url_input.max_chars
                        document = content
                        #print("DEBUG: Full document to index:", document)
                        print("DEBUG: Document prepared for indexing:", document['url'])
                        embeddings_manager.index_document(document, force_delete=url_input.force_delete)
                    
            elif url_input.doc_type == "PDF":
                pdf_crawler = PDFCrawler()
                pdf_data = pdf_crawler.crawl(url)
                print("PDF data retrieved:", pdf_data)
                
                if pdf_data:
                    for section in pdf_data['sections']:
                        document = {
                            'url': url,
                            'text': section['content'],
                            'doc_type': 'PDF',
                            'title': pdf_data['title'],
                            'page_number': section['page_number']
                        }
                        if url_input.max_chars:
                            document["max_chars"] = url_input.max_chars
                        embeddings_manager.index_document(document, force_delete=url_input.force_delete)
        
        return {"message": "Content indexed successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", tags=["3. Search & Chat"], summary="4. Vector search")
async def search_content(search_request: SearchRequest):
    """Search indexed content"""
    global qdrant_db
    if qdrant_db is None:
        qdrant_db = QdrantDB(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.collection_name
        )
    try:
        if not search_request.query:
            if search_request.query_filter and "url" in search_request.query_filter:
                #logger.debug(f"Performing metadata-only search for URL: {search_request.query_filter['url']}, limit: {search_request.limit}")
                results = qdrant_db.get_chunks_by_url(
                    url=search_request.query_filter["url"],
                    limit=search_request.limit
                )
                #logger.debug(f"Found {len(results)} results for metadata-only search")
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
            
            print(f"[DEBUG] Performing vector search with query: {search_request.query}, limit: {search_request.limit}, filter: {qdrant_filter}")
            
            # Extract optional parameters (direct attribute access)
            score_threshold = search_request.score_threshold
            exact = search_request.exact
            with_payload = search_request.with_payload
            
            # Use QdrantDB directly for the search with query string
            results = qdrant_db.search_similar(
                query=search_request.query,
                limit=search_request.limit,
                query_filter=qdrant_filter,
                score_threshold=score_threshold,
                exact=exact,
                with_payload=with_payload
            )
            print(f"[DEBUG] Search results count: {len(results)}")
            return SearchResponse(results=results, total=len(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", tags=["3. Search & Chat"], summary="5. Chat (stateless)")
async def chat_with_content(chat_request: ChatRequest):
    """Chat with the indexed content"""
    global chat_manager
    if chat_manager is None:
        chat_manager = ChatManager()
    try:
        response = await chat_manager.chat(
            chat_request.message,
            chat_request.context,
            chat_request.use_web_search
        )
        if response is None:
            raise HTTPException(status_code=500, detail="Chat response is None")
        return ChatResponse(**response)
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed", tags=["2. Ingest"], summary="3. Embed single document")
async def generate_embedding(embedding_request: EmbeddingRequest):
    """Generate embedding for a specific document"""
    global embeddings_manager
    if embeddings_manager is None:
        embeddings_manager = EmbeddingsManager()
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
async def delete_document(url: str):
    """Delete embeddings for a specific document"""
    global embeddings_manager
    if embeddings_manager is None:
        embeddings_manager = EmbeddingsManager()
    try:
        embeddings_manager.delete_document(url)
        return {"message": "Document deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug-index", tags=["5. Debug"], summary="1. Inspect Qdrant collections")
def debug_index(url: Optional[str] = None):
    """List current indexed collections and their first 10 documents, optionally filtered by URL"""
    global embeddings_manager
    if embeddings_manager is None:
        embeddings_manager = EmbeddingsManager()
    try:
        logger.info("Starting /debug-index route")
        collections_info = embeddings_manager.qdrant.client.get_collections()
        logger.info(f"Collections info: {collections_info}")
        collection_names = [col.name for col in collections_info.collections]
        logger.info(f"Collection names: {collection_names}")
        debug_info = {}
        for name in collection_names:
            try:
                logger.info(f"Fetching documents from collection: {name}")
                all_docs = []
                next_offset = None
                if url:
                    url_lower = url.lower()
                    qdrant_filter = models.Filter(
                        must=[
                            models.FieldCondition(
                                key="url_lower",
                                match=models.MatchValue(value=url_lower)
                            )
                        ]
                    )
                else:
                    qdrant_filter = None
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
                logger.info(f"Scroll response for {name}: {len(all_docs)} docs retrieved")
                if all_docs:
                    sorted_docs = sorted(
                        all_docs,
                        key=lambda d: (
                            d.payload.get("url", ""),
                            int(d.payload.get("section_index")) if d.payload.get("section_index") is not None else 0,
                            int(d.payload.get("subsection_index")) if d.payload.get("subsection_index") is not None else -1,
                            int(d.payload.get("chunk_index")) if d.payload.get("chunk_index") is not None else 0
                        )
                    )
                    # [DEBUG] loop after sorting
                    for d in sorted_docs:
                        print(
                            f"[DEBUG] URL: {d.payload.get('url', '')} | Section: {d.payload.get('section', '')} | Subsection: {d.payload.get('subsection', '')} | Section Index: {d.payload.get('section_index')} | Subsection Index: {d.payload.get('subsection_index')} | Chunk Index: {d.payload.get('chunk_index')}"
                        )
                    debug_info[name] = sorted_docs
                else:
                    debug_info[name] = f"No documents found in collection {name}"
            except Exception as inner_e:
                logger.error(f"Error retrieving documents from {name}: {str(inner_e)}")
                debug_info[name] = f"Error retrieving documents: {str(inner_e)}"
        logger.info(f"Debug info to return: {debug_info}")
        return {"collections": debug_info}
    except Exception as e:
        print("DEBUG-INDEX ERROR:", e)
        logger.error(f"Top-level exception in /debug-index: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


# Endpoint to update a specific payload field for all chunks matching the given URL.
from fastapi import HTTPException

@app.post("/update_payload", tags=["4. Index Admin"], summary="1. Bulk update payload by URL")
async def update_payload_field(update_request: PayloadUpdateRequest):
    """
    Update a specific payload field for all chunks matching the given URL.
    """
    global qdrant_db
    if qdrant_db is None:
        qdrant_db = QdrantDB(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.collection_name
        )
    try:
        updated = qdrant_db.update_payload_by_url(update_request)
        return {
            "message": f"Updated payloads with key '{update_request.meta_key}' for URL '{update_request.url}'",
            "updated": updated
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
