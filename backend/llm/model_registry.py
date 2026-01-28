# backend/llm/model_registry.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Literal

Provider = Literal["openai", "gemini"]
Endpoint = Literal["responses", "chat_completions", "embeddings", "gemini_sdk"]

@dataclass(frozen=True)
class Pricing:
    # USD per 1,000,000 tokens
    input_per_mm: float
    output_per_mm: float
    cached_input_per_mm: float = 0.0  # optional

@dataclass(frozen=True)
class ModelInfo:
    key: str                    # stable alias key (what you store in settings/params)
    provider: Provider
    model: str                  # provider-native model id
    endpoint: Endpoint          # which API shape your handler will use
    pricing: Optional[Pricing]  # None if you don't want to cost this model
    capabilities: Dict[str, Any] = field(default_factory=dict)
    max_tokens_parameter: str = "max_output_tokens"  # Parameter name for token limits: "max_tokens" or "max_completion_tokens"
    # Optional “thinking tax” / reserve rules: used to derive a larger provider token cap
    # from a desired visible max_output_tokens, based on UI-provided effort levels.
    # Shape suggestion:
    # {
    #   "effort_map": {"none": {"reserve_ratio": 0.0}, "low": {"reserve_ratio": 0.25}, ...},
    #   "param_map": {"none": <value>, "low": <value>, ...}  # for thinking_level-style knobs
    # }
    thinking_tax: Dict[str, Any] = field(default_factory=dict)
    # Optional reasoning parameter mapping: (parameter_name, default_value)
    reasoning_parameter: Optional[Tuple[str, Any]] = None

# ---- Registry ----
REGISTRY: Dict[str, ModelInfo] = {
    # -----------------------
    # OpenAI (3)
    # -----------------------
    "openai:embed_small": ModelInfo(
        key="openai:embed_small",
        provider="openai",
        model="text-embedding-3-small",
        endpoint="embeddings",
        pricing=Pricing(input_per_mm=0.02, output_per_mm=0.0),
        capabilities={"dimensions": 1536},
        max_tokens_parameter="max_output_tokens",  # Embeddings don't use max_tokens but set for consistency
    ),
    "openai:embed_large": ModelInfo(
        key="openai:embed_large",
        provider="openai",
        model="text-embedding-3-large",
        endpoint="embeddings",
        pricing=Pricing(input_per_mm=0.13, output_per_mm=0.0),
        capabilities={"dimensions": 3072},
        max_tokens_parameter="max_output_tokens",  # Embeddings don't use max_tokens but set for consistency
    ),
    "openai:fast": ModelInfo(
        key="openai:fast",
        provider="openai",
        model="gpt-4o-mini",
        endpoint="responses",
        pricing=Pricing(input_per_mm=0.15, output_per_mm=0.60, cached_input_per_mm=0.075),
        capabilities={
            "tools": True, 
            "stream": True,
            "temperature": True,
            "reasoning_effort": False,
            "top_p": True,
        },
        max_tokens_parameter="max_output_tokens",  # OpenAI non-reasoning models use max_tokens
    ),
    "openai:best": ModelInfo(
        key="openai:best",
        provider="openai",
        model="gpt-4o",
        endpoint="responses",
        pricing=Pricing(input_per_mm=2.50, output_per_mm=10.00, cached_input_per_mm=1.25),
        capabilities={
            "tools": True, 
            "stream": True,
            "temperature": True,
            "reasoning_effort": False,
            "top_p": True,
        },
        max_tokens_parameter="max_output_tokens", 
    ),
    # Opt-in OpenAI Chat Completions variants (endpoint="chat_completions").
    # These are intentionally separate keys so existing "openai:*" models that
    # use the Responses API remain unchanged. Callers can explicitly choose
    # these keys when they want to route via chat.completions instead of
    # responses, while still targeting the same underlying model IDs.
    "openai:chat_fast": ModelInfo(
        key="openai:chat_fast",
        provider="openai",
        model="gpt-4o-mini",
        endpoint="chat_completions",
        pricing=Pricing(input_per_mm=0.15, output_per_mm=0.60, cached_input_per_mm=0.075),
        capabilities={
            "tools": True,
            "stream": True,
            "temperature": True,
            "reasoning_effort": False,
            "top_p": True,
        },
        # Chat Completions token limit parameter is "max_completion_tokens";
        # callers should continue to pass model-agnostic max_output_tokens and
        # let llm_handler map it using this field.
        max_tokens_parameter="max_completion_tokens",
    ),
    "openai:chat_best": ModelInfo(
        key="openai:chat_best",
        provider="openai",
        model="gpt-4o",
        endpoint="chat_completions",
        pricing=Pricing(input_per_mm=2.50, output_per_mm=10.00, cached_input_per_mm=1.25),
        capabilities={
            "tools": True,
            "stream": True,
            "temperature": True,
            "reasoning_effort": False,
            "top_p": True,
        },
        max_tokens_parameter="max_completion_tokens",
    ),
    "openai:reasoning_mini": ModelInfo(
        key="openai:reasoning_mini",
        provider="openai",
        model="o3-mini",
        endpoint="responses",
        pricing=Pricing(input_per_mm=1.10, output_per_mm=4.40),
        capabilities={
            "tools": True, 
            "stream": False,
            "temperature": False,  # o1/o3 models don't support temperature
            "reasoning_effort": True,  
            "top_p": False,
        },
        max_tokens_parameter="max_output_tokens",  
        reasoning_parameter=("reasoning_effort", "low"),  
    ),
    "openai:reasoning_mini_small": ModelInfo(
        key="openai:reasoning_mini_small",
        provider="openai",
        model="gpt-5-mini",
        endpoint="responses",
        pricing=Pricing(input_per_mm=.25, output_per_mm=2.00),
        capabilities={
            "tools": True, 
            "stream": False,
            "temperature": False,  # gpt-5-mini doesn't support temperature
            "reasoning_effort": True,  
            "top_p": False,
        },
        max_tokens_parameter="max_output_tokens",  
        reasoning_parameter=("reasoning_effort", "minimal"),  
    ),

    # -----------------------
    # Gemini via OpenAI adapter (3)
    # -----------------------
    "gemini:embed": ModelInfo(
        key="gemini:embed",
        provider="gemini",
        model="gemini-embedding-001",
        endpoint="embeddings",
        pricing=Pricing(input_per_mm=0.10, output_per_mm=0.0),  # use your real rates
        capabilities={
            "dimensions": 1536,
            "normalize_embedding": False,  # Let caller decide
        },
        max_tokens_parameter="max_tokens",  # Embeddings don't use max_tokens but set for consistency
    ),
    # Native Gemini SDK embedding profile. This is opt-in and used by the
    # gemini_sdk embedding path in llm_handler for experiments.
    # Task_type defaults are handled in core/config.py:
    # - gemini_embed_type_documents="RETRIEVAL_DOCUMENT" for indexing
    # - gemini_embed_type_query="RETRIEVAL_QUERY" for search
    "gemini:native-embed": ModelInfo(
        key="gemini:native-embed",
        provider="gemini",
        model="gemini-embedding-001",  # native embedding model id
        endpoint="gemini_sdk",
        pricing=Pricing(input_per_mm=0.10, output_per_mm=0.0),  # adjust to your real rates
        capabilities={
            "dimensions": 1536,
            "task_type": "RETRIEVAL_DOCUMENT",
            "output_dimensionality": 1536,
            "normalize_embedding": False,  # Let caller decide
        },
        max_tokens_parameter="max_tokens",
    ),
    "gemini:openai-fast": ModelInfo(
        key="gemini:openai-fast",
        provider="gemini",
        model="models/gemini-2.5-flash-lite",
        endpoint="chat_completions",
        pricing=Pricing(input_per_mm=0.20, output_per_mm=0.80),  # use your real rates
        capabilities={
            "tools": True, 
            "stream": True,
            "temperature": True,
            "reasoning_effort": False,
            "top_p": True,
        },
        max_tokens_parameter="max_completion_tokens", # Can also be max_completion_tokens or max_tokens 
        thinking_tax={
            "effort_map": {
                "none": {"reserve_ratio": 0.0},
                "low": {"reserve_ratio": 0.25},
                "medium": {"reserve_ratio": 0.50},
                "high": {"reserve_ratio": 0.80},
            },
            "kind": "budget",
        },
    ),
    "gemini:openai-3-flash": ModelInfo(
        key="gemini:openai-3-flash",
        provider="gemini",
        model="models/gemini-3-flash-preview",
        endpoint="chat_completions",
        pricing=Pricing(input_per_mm=0.50, output_per_mm=3.00),  # use your real rates
        capabilities={
            "tools": True, 
            "stream": True,
            "temperature": True,
            "reasoning_effort": True,  
            "top_p": True,
        },
        max_tokens_parameter="max_completion_tokens",  # Can also be max_output_tokens or max_tokens 
        reasoning_parameter=("thinking_level", "minimal"),  
        thinking_tax={
            "effort_map": {
                "none": {"reserve_ratio": 0.0},
                "minimal": {"reserve_ratio": 0.25},
                "low": {"reserve_ratio": 0.30},
                "medium": {"reserve_ratio": 0.50},
                "high": {"reserve_ratio": 0.80},
            },
            "param_map": {
                "none": "minimal",
                "minimal": "minimal",
                "low": "low",
                "medium": "medium",
                "high": "high",
            },
            "kind": "level",
        },
    ),
    "gemini:native-sdk-3-flash": ModelInfo(
        key="gemini:native-sdk-3-flash",
        provider="gemini",
        model="models/gemini-3-flash-preview",
        endpoint="gemini_sdk",
        pricing=Pricing(input_per_mm=0.50, output_per_mm=3.00),  # same rates as gemini:fast-3-flash
        capabilities={
            "tools": True,
            "stream": True,
            "temperature": True,
            "reasoning_effort": True,
            "top_p": True,
        },
        max_tokens_parameter="max_completion_tokens",
        reasoning_parameter=("thinking_level", "minimal"),
        thinking_tax={
            "effort_map": {
                "none": {"reserve_ratio": 0.0},
                "minimal": {"reserve_ratio": 0.25},
                "low": {"reserve_ratio": 0.30},
                "medium": {"reserve_ratio": 0.50},
                "high": {"reserve_ratio": 0.80},
            },
            "param_map": {
                "none": "minimal",
                "minimal": "minimal",
                "low": "low",
                "medium": "medium",
                "high": "high",
            },
            "kind": "level",
        },
    ),
    "gemini:openai-best": ModelInfo(
        key="gemini:openai-best",
        provider="gemini",
        model="models/gemini-2.5-flash",
        endpoint="chat_completions",
        pricing=Pricing(input_per_mm=0.30, output_per_mm=2.50),  # use your real rates
        capabilities={
            "tools": True, 
            "stream": True,
            "temperature": True,
            "reasoning_effort": True,  
            "top_p": True,
        },
        max_tokens_parameter="max_completion_tokens",  
        reasoning_parameter=("thinking_budget", 1000),  
        thinking_tax={
            "effort_map": {
                "none": {"reserve_ratio": 0.0},
                "low": {"reserve_ratio": 0.25},
                "medium": {"reserve_ratio": 0.50},
                "high": {"reserve_ratio": 0.80},
            },
            "kind": "budget",
        },
    ),
}

def get_model_info(key: str) -> ModelInfo:
    if key not in REGISTRY:
        raise KeyError(f"Unknown model key: {key}")
    return REGISTRY[key]


def resolve_model(
    provider: str | None,
    model: str | None,
    model_key: str | None = None,
) -> Optional[ModelInfo]:
    """Best-effort registry lookup for a model.

    Resolution order (mirrors existing heuristics in chat_manager):

      1. If ``model_key`` is provided and exists in REGISTRY, return that entry.
      2. If the provider-native ``model`` string itself is a REGISTRY key, return it.
      3. Otherwise, scan REGISTRY for an entry whose ``model`` field matches the
         provider-native ``model`` string, optionally filtered by ``provider``.

    Returns ``None`` if no match can be found or if REGISTRY is empty.
    """
    try:
        reg = REGISTRY or {}
        if not reg:
            return None

        # 1) Explicit registry key when provided.
        mk = str(model_key).strip() if model_key else ""
        if mk and mk in reg:
            return reg.get(mk)

        # 2) Direct key hit by provider-native model string.
        m = str(model or "").strip()
        if not m:
            return None
        if m in reg:
            return reg.get(m)

        # 3) Match by provider+model name.
        p = str(provider or "").strip().lower()
        for _k, _v in reg.items():
            try:
                if not _v:
                    continue
                if str(getattr(_v, "model", "")) != m:
                    continue
                if p and str(getattr(_v, "provider", "")).lower() != p:
                    continue
                return _v
            except Exception:
                continue
    except Exception:
        return None
    return None
