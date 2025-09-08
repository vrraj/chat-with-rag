from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

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
    # Pass-through of UI parameters and chat bubbles history (stateless UI)
    params: Optional[Dict[str, Any]] = Field(default_factory=dict)
    history: Optional[List[Dict[str, str]]] = Field(default_factory=list)

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict] = []

class MediaWikiURLInput(BaseModel):
    url: str
    max_chunks: Optional[int] = None
    skip_sections: Optional[List[str]] = Field(
        default_factory=lambda: ["References", "External Links", "Further reading", "Notes"],
        description="List of section titles to skip when parsing the wiki page"
    )
    force_delete: Optional[bool] = False
    api_url: Optional[str] = Field(
        None,
        description="Override MediaWiki API endpoint (defaults to settings)"
    )
    user_agent: Optional[str] = Field(
        None,
        description="Override User-Agent for MediaWiki requests"
    )
    estimate: Optional[bool] = Field(
        False,
        description="If true, return planned chunk count without indexing"
    )

class URLInput(BaseModel):
    urls: List[str]
    doc_type: str = "HTML"  # 'HTML' or 'PDF'
    force_crawl: Optional[bool] = True
    max_chunks: Optional[int] = 0  # 0 or less means no user limit; hard cap still applies
    force_delete: Optional[bool] = True


class PDFInput(BaseModel):
    file: Optional[bytes] = Field(
        None,
        description="PDF file content (base64 encoded) for direct upload"
    )
    url: Optional[str] = Field(
        None,
        description="URL of the PDF to download and process"
    )
    max_chunks: Optional[int] = Field(
        0,
        description="Maximum number of chunks to process (0 = no limit)"
    )
    force_delete: Optional[bool] = Field(
        False,
        description="Force re-indexing if document already exists"
    )
    estimate: Optional[bool] = Field(
        False,
        description="If true, return planned chunk count without indexing"
    )

## Consolidated PDF schema handled via /pdf endpoint with form fields; URL-only schema removed


# Payload update request schema
class PayloadUpdateRequest(BaseModel):
    url: str
    meta_key: str
    meta_value: str
