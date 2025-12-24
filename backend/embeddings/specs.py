from typing import Any, Dict

from backend.core.config import settings


def resolve_embedding_spec(settings_obj: Any = settings) -> Dict[str, Any]:
    """Return a normalized embedding spec from settings.

    `embedding_model` must be a provider key ("openai" or "gemini"), and the
    concrete model id is resolved via the per-provider settings fields.

    Output shape:
        {
          "provider": "openai" | "gemini",
          "model": "<provider-specific model name>",
          "dimensions": int | None,
        }
    """
    raw = getattr(settings_obj, "embedding_model", "openai") or "openai"
    provider_key = str(raw).strip().lower()

    if provider_key not in ("openai", "gemini"):
        raise ValueError(
            f"Invalid embedding_model '{raw}'. Expected 'openai' or 'gemini'. "
            "Update your configuration to use a provider key and the per-provider model fields."
        )

    if provider_key == "gemini":
        model = getattr(settings_obj, "gemini_embedding_model", "gemini-embedding-001")
        try:
            dims = int(getattr(settings_obj, "gemini_embedding_dimensions", 1536) or 1536)
        except Exception:
            dims = 1536
        return {
            "provider": "gemini",
            "model": model,
            "dimensions": dims,
        }

    # provider_key == "openai"
    model = getattr(settings_obj, "openai_embedding_model", "text-embedding-3-small")
    return {
        "provider": "openai",
        "model": model,
        "dimensions": None,
    }
