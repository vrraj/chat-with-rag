from typing import Any, Dict

from backend.core.config import settings


def resolve_embedding_spec(settings_obj: Any = settings) -> Dict[str, Any]:
    """Return embedding spec from model registry using embedding_model_key.

    Uses embedding_model_key to resolve provider, model, and dimensions
    from the pip-installable llm-adapter model registry.

    Output shape:
        {
          "provider": "openai" | "gemini",
          "model": "<provider-specific model name>",
          "dimensions": int | None,
        }
    """
    from backend.llm.llm_client import get_model_info
    
    model_key = getattr(settings_obj, "embedding_model_key", "openai:embed_small")
    model_info = get_model_info(model_key=model_key)
    
    return {
        "provider": model_info.provider,
        "model": model_info.model,
        "dimensions": model_info.capabilities.get("dimensions"),
    }
