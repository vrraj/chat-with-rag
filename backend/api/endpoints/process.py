"""API endpoints for URL processing."""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from backend.scripts.schemas import URLItem, ProcessingResult, ActionType
from backend.scripts.process_urls import URLProcessor

router = APIRouter()

class ProcessRequest(BaseModel):
    """Request model for processing URLs."""
    urls: List[URLItem]
    default_config: Optional[Dict[str, Any]] = None

@router.post("/process", response_model=List[ProcessingResult])
async def process_urls(request: ProcessRequest):
    """
    Process one or more URLs with the given configurations.
    
    Example request body:
    {
        "urls": [
            {
                "url": "https://example.com",
                "action": "estimate",
                "config": {
                    "type": "html",
                    "max_chunks": 100
                }
            }
        ],
        "default_config": {
            "force_delete": false
        }
    }
    """
    processor = URLProcessor(default_config=request.default_config or {})
    
    results = []
    for url_item in request.urls:
        result = await processor.process_url(url_item)
        results.append(result)
    
    return results

# Add the router to FastAPI in main.py
# from backend.api.endpoints import process
# app.include_router(process.router, prefix="/api/v1")
