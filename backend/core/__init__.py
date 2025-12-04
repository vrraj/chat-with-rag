"""Core package public API.

Exports foundational config and schemas used across the backend.
This file is lightweight and side-effect free.
"""

from .config import settings, Settings
from .schemas import (
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

__all__ = [
    "settings",
    "Settings",
    "EmbeddingRequest",
    "SearchRequest",
    "SearchResponse",
    "ChatRequest",
    "ChatResponse",
    "MediaWikiURLInput",
    "PDFInput",
    "URLInput",
    "PayloadUpdateRequest",
]

