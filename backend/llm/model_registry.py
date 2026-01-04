# backend/llm/model_registry.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Literal

Provider = Literal["openai", "gemini"]
Endpoint = Literal["responses", "chat_completions", "embeddings"]

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
    max_tokens_parameter: str = "max_tokens"  # Parameter name for token limits: "max_tokens" or "max_completion_tokens"
    reasoning_parameter: str = "reasoning_effort"  # Parameter name for reasoning effort

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
    ),
    "openai:reasoning_mini": ModelInfo(
        key="openai:reasoning_mini",
        provider="openai",
        model="o3-mini",
        endpoint="responses",
        pricing=Pricing(input_per_mm=1.10, output_per_mm=4.40),
        capabilities={
            "tools": False, 
            "stream": False,
            "temperature": False,  # o1/o3 models don't support temperature
            "reasoning_effort": True,  # OpenAI reasoning models support this
            "top_p": False,
        },
        max_tokens_parameter="max_completion_tokens",
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
        capabilities={"dimensions": 1536},
        reasoning_parameter="thinking_budget",
    ),
    "gemini:fast": ModelInfo(
        key="gemini:fast",
        provider="gemini",
        model="models/gemini-2.5-flash-lite",
        endpoint="chat_completions",
        pricing=Pricing(input_per_mm=0.20, output_per_mm=0.80),  # use your real rates
        capabilities={
            "tools": False, 
            "stream": True,
            "temperature": True,
            "thinking_level": False,  # Gemini supports thinking_level
            "thinking_budget": True,  # Gemini does not support thinking_budget
        },
        reasoning_parameter="thinking_budget",
    ),
    "gemini:best": ModelInfo(
        key="gemini:best",
        provider="gemini",
        model="models/gemini-2.5-pro",
        endpoint="chat_completions",
        pricing=Pricing(input_per_mm=1.00, output_per_mm=4.00),  # use your real rates
        capabilities={
            "tools": False, 
            "stream": True,
            "temperature": True,
            "thinking_level": False,  # Gemini supports thinking_level
            "thinking_budget": True,  # Gemini does not support thinking_budget (but has similar concept)
            "top_p": True,
        },
        reasoning_parameter="thinking_budget",
    ),
}

def get_model_info(key: str) -> ModelInfo:
    if key not in REGISTRY:
        raise KeyError(f"Unknown model key: {key}")
    return REGISTRY[key]