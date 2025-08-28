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
    limit: Optional[int] = 8
    query_filter: Optional[Dict] = None  # For flexible filtering (e.g., {"url": "example.com"})
    score_threshold: Optional[float] = 0.35  # Default threshold for Cosine similarity is 0.35
    exact: Optional[bool] = False
    with_payload: Optional[bool] = True

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
    skip_sections: Optional[List[str]] = Field(
        default=["References", "External Links", "Further reading", "Notes"],
        example=["References", "External Links", "Further reading", "Notes"],
        description="List of section titles to skip when parsing the wiki page"
    )
    force_delete: Optional[bool] = True

class URLInput(BaseModel):
    urls: List[str]
    doc_type: str = "HTML"  # 'HTML' or 'PDF'
    force_crawl: Optional[bool] = True
    max_chunks: Optional[int] = 0  # 0 or less means no user limit; hard cap still applies
    force_delete: Optional[bool] = True


## Consolidated PDF schema handled via /pdf endpoint with form fields; URL-only schema removed


# Payload update request schema
class PayloadUpdateRequest(BaseModel):
    url: str
    meta_key: str
    meta_value: str
