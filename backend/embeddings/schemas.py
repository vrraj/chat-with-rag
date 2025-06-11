from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class EmbeddingRequest(BaseModel):
    url: str
    content: str
    title: Optional[str] = ""
    description: Optional[str] = ""
    domain: Optional[str] = ""
    document_type: Optional[str] = "HTML"
    date: Optional[str] = ""
    chunk_index: Optional[int] = 0
    total_chunks: Optional[int] = 1

class SearchRequest(BaseModel):
    query: str
    limit: int = 5
    query_filter: Optional[Dict] = None

class SearchResponse(BaseModel):
    results: List[Dict]
    total: int

class ChatRequest(BaseModel):
    message: str
    context: List[Dict] = []
    use_web_search: bool = False

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict] = []

class MediaWikiURLInput(BaseModel):
    url: str
    max_chunks: Optional[int] = None
    skip_sections: Optional[List[str]] = None
    force_delete: Optional[bool] = True
