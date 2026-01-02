from typing import Dict, Any

from fastapi import APIRouter

from backend.llm.model_registry import REGISTRY

router = APIRouter()


@router.get("/api/models", response_model=Dict[str, Dict[str, Any]])
async def get_models() -> Dict[str, Dict[str, Any]]:
    """Expose the backend model registry for the frontend.

    Returns a JSON mapping of model_key -> lightweight model info
    suitable for building the frontend MODEL_REGISTRY.
    """
    payload: Dict[str, Dict[str, Any]] = {}
    for key, m in REGISTRY.items():
        payload[key] = {
            "key": m.key,
            "provider": m.provider,
            "model": m.model,
            "endpoint": m.endpoint,
            "capabilities": m.capabilities or {},
        }
    return payload
