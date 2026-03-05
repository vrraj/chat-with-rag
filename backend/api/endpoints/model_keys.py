import os
from typing import Dict, Any, Optional
from pathlib import Path
import sys

from fastapi import APIRouter

# Import from pip-installable llm-adapter package
from llm_adapter import llm_adapter, model_registry, LLMAdapter

router = APIRouter()

# Custom registry configuration - can be overridden via environment variable
DEFAULT_CUSTOM_REGISTRY_PATH = Path(__file__).resolve().parents[4] / "examples" / "custom_registry.py"
CUSTOM_REGISTRY_PATH = Path(os.getenv("CUSTOM_REGISTRY_PATH", str(DEFAULT_CUSTOM_REGISTRY_PATH)))


def _get_adapter(merge_custom_registry: bool = False) -> LLMAdapter:
    """Get LLMAdapter instance, optionally with custom registry merged."""
    if not merge_custom_registry:
        return llm_adapter
    
    try:
        # Import custom registry from configured path
        examples_dir = CUSTOM_REGISTRY_PATH.parent
        if str(examples_dir) not in sys.path:
            sys.path.insert(0, str(examples_dir))
        
        # Import the registry module (filename without .py)
        module_name = CUSTOM_REGISTRY_PATH.stem
        registry_module = __import__(module_name)
        USER_REGISTRY = getattr(registry_module, 'REGISTRY')
        
        # Create new adapter with merged registry
        custom_adapter = LLMAdapter(model_registry=USER_REGISTRY)
        
        # Copy API keys from default adapter
        if hasattr(llm_adapter, 'openai_api_key'):
            custom_adapter.openai_api_key = llm_adapter.openai_api_key
        if hasattr(llm_adapter, 'gemini_api_key'):
            custom_adapter.gemini_api_key = llm_adapter.gemini_api_key
        
        return custom_adapter
    except Exception as e:
        # Fall back to default adapter if custom registry fails
        print(f"Failed to load custom registry from {CUSTOM_REGISTRY_PATH}: {e}")
        return llm_adapter


def get_model_options(adapter: Optional[LLMAdapter] = None) -> Dict[str, Any]:
    """Get model options from pip-installable llm-adapter registry.
    
    This mirrors the pattern from llm_adapter_demo/config.py
    
    Args:
        adapter: Optional LLMAdapter instance. If provided, uses its registry.
                If None, uses the default llm_adapter registry.
    """
    # Use provided adapter's registry or default registry
    registry = getattr(adapter, 'model_registry', model_registry.REGISTRY) if adapter else model_registry.REGISTRY
    
    out: Dict[str, Any] = {}
    for key, mi in (registry or {}).items():
        provider = str(getattr(mi, "provider", "") or "")
        if not provider:
            continue
        
        # Convert ModelInfo to dict with all fields
        model_data = {
            "key": key,
            "provider": provider,
            "model": str(getattr(mi, "model", "") or ""),
            "endpoint": str(getattr(mi, "endpoint", "") or ""),
            "pricing": getattr(mi, "pricing", None),
            "param_policy": getattr(mi, "param_policy", {}) or {},
            "reasoning_policy": getattr(mi, "reasoning_policy", {}) or {},
            "thinking_tax": getattr(mi, "thinking_tax", {}) or {},
            "reasoning_parameter": getattr(mi, "reasoning_parameter", None),
            "capabilities": getattr(mi, "capabilities", {}) or {},
        }
        
        # Convert Pricing object to dict if present
        if model_data["pricing"] is not None:
            pricing = model_data["pricing"]
            model_data["pricing"] = {
                "input_per_mm": getattr(pricing, "input_per_mm", 0.0),
                "output_per_mm": getattr(pricing, "output_per_mm", 0.0),
                "cached_input_per_mm": getattr(pricing, "cached_input_per_mm", 0.0),
            }
        
        out[key] = model_data
    return out


@router.get("/api/models", response_model=Dict[str, Dict[str, Any]])
async def get_models(merge_custom_registry: bool = False) -> Dict[str, Dict[str, Any]]:
    """Expose the pip-installable llm-adapter model registry for the frontend.

    Args:
        merge_custom_registry: If True, attempts to merge custom registry from examples/custom_registry.py
        
    Returns a JSON mapping of model_key -> lightweight model info
    suitable for building the frontend MODEL_REGISTRY.
    """
    if merge_custom_registry:
        # Get adapter with custom registry merged
        custom_adapter = _get_adapter(merge_custom_registry=True)
        return get_model_options(adapter=custom_adapter)
    else:
        # Use default registry
        return get_model_options()
