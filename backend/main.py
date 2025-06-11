from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import logging
from qdrant_client.http import models
logger = logging.getLogger(__name__)
from backend.core.config import settings
from backend.embeddings.embeddings_manager import EmbeddingsManager
from backend.crawler.crawler import WebCrawler
from backend.crawler.pdf_crawler import PDFCrawler
from backend.extractor.extractor import ContentExtractor
from backend.embeddings.schemas import EmbeddingRequest, SearchRequest, SearchResponse, ChatRequest, ChatResponse, MediaWikiURLInput
from backend.extractor.mediawiki_extractor import MediaWikiExtractor

app = FastAPI(title="Website Chat Agent API")

from typing import Optional

class MediaWikiURLInput(BaseModel):
    url: str
    max_chunks: Optional[int] = None
    skip_sections: Optional[List[str]] = None
    force_delete: Optional[bool] = True

@app.post("/mediawiki/url")
async def index_mediawiki_url(mediawiki_input: MediaWikiURLInput):
    """
    Index content from MediaWiki wikitext, optionally limiting number of chunks indexed by max_chunks.
    """
    try:
        extractor = MediaWikiExtractor()
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

class URLInput(BaseModel):
    urls: List[str]
    doc_type: str = "HTML"  # 'HTML' or 'PDF'
    force_crawl: Optional[bool] = True
    max_chars: Optional[int] = 1000  # Limit characters to to embed for a web page (Testing)
    force_delete: Optional[bool] = True

class ChatMessage(BaseModel):
    message: str
    context: List[Dict] = []

# Initialize managers
embeddings_manager = EmbeddingsManager()
chat_manager = ChatManager()

@app.post("/index")
async def index_content(url_input: URLInput):
    """Index content from URLs (HTML or PDF)"""
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
                        print("DEBUG: Full document to index:", document)
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

@app.post("/search")
async def search_content(search_request: SearchRequest):
    """Search indexed content"""
    try:
        if not search_request.query:
            if search_request.query_filter and "url" in search_request.query_filter:
                print(f"[DEBUG] Performing metadata-only search for URL: {search_request.query_filter['url']}, limit: {search_request.limit}")
                results = embeddings_manager.get_chunks_by_url(
                    url=search_request.query_filter["url"],
                    limit=search_request.limit
                )
                print(f"[DEBUG] Search results count: {len(results)}")
                return SearchResponse(results=results, total=len(results))
            else:
                raise HTTPException(status_code=400, detail="URL must be provided if no query is given.")
        else:
            qdrant_filter = None
            if search_request.query_filter and "url" in search_request.query_filter:
                qdrant_filter = {
                    "must": [
                        {
                            "key": "url",
                            "match": {
                                "value": search_request.query_filter["url"]
                            }
                        }
                    ]
                }
            print(f"[DEBUG] Performing vector search with query: {search_request.query}, limit: {search_request.limit}, filter: {qdrant_filter}")
            results = embeddings_manager.search_similar(
                query=search_request.query,
                limit=search_request.limit,
                query_filter=qdrant_filter
            )
            print(f"[DEBUG] Search results count: {len(results)}")
            return SearchResponse(results=results, total=len(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_with_content(chat_request: ChatRequest):
    """Chat with the indexed content"""
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

@app.post("/embed")
async def generate_embedding(embedding_request: EmbeddingRequest):
    """Generate embedding for a specific document"""
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

@app.delete("/delete/{url}")
async def delete_document(url: str):
    """Delete embeddings for a specific document"""
    try:
        embeddings_manager.delete_document(url)
        return {"message": "Document deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug-index")
async def debug_index():
    """List current indexed collections and their first 10 documents"""
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
                while True:
                    docs, next_offset = embeddings_manager.qdrant.client.scroll(
                        collection_name=name,
                        limit=100,  # Batch size per scroll request (not total limit). Adjust for performance/memory.
                        with_payload=True,
                        offset=next_offset
                    )
                    all_docs.extend(docs)
                    if next_offset is None:
                        break
                logger.info(f"Scroll response for {name}: {len(all_docs)} docs retrieved")
                if all_docs:
                    sorted_docs = sorted(
                        all_docs,
                        key=lambda d: (
                            d.payload.get("url", ""),
                            d.payload.get("section_index", 0),
                            d.payload.get("subsection_index", 0),
                            d.payload.get("chunk_index", 0)
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
