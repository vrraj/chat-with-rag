"""Pydantic models for URL processing."""
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, HttpUrl, Field

class ContentType(str, Enum):
    HTML = "html"
    PDF = "pdf"
    MEDIAWIKI = "mediawiki"

class ActionType(str, Enum):
    ESTIMATE = "estimate"
    INDEX = "index"

class ProcessingConfig(BaseModel):
    """Configuration for processing a URL."""
    type: ContentType = ContentType.HTML
    max_chunks: int = 0  # 0 means no limit
    force_delete: bool = False
    skip_sections: List[str] = ["References", "External links", "See also", "Further reading"]
    custom_params: Dict[str, Any] = Field(default_factory=dict, description="Additional type-specific parameters")

class URLItem(BaseModel):
    """Single URL item to be processed."""
    url: HttpUrl
    action: ActionType = ActionType.ESTIMATE
    config: Optional[ProcessingConfig] = None

class ProcessingResult(BaseModel):
    """Result of processing a URL."""
    status: str  # "success", "error"
    url: str
    action: str
    config_used: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str
    duration_seconds: Optional[float] = None
