# backend/llm/model_registry.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Literal

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
        capabilities={"dimensions": 1536},
        max_tokens_parameter="max_tokens",  # Embeddings don't use max_tokens but set for consistency
    ),
    "gemini:fast": ModelInfo(
        key="gemini:fast",
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
    # TEST MODEL: Budget-based thinking with reasoning support
    "gemini:fast-test": ModelInfo(
        key="gemini:fast-test",
        provider="gemini",
        model="models/gemini-2.5-flash-lite",  # Same model but WITH reasoning support
        endpoint="chat_completions",
        pricing=Pricing(input_per_mm=0.20, output_per_mm=0.80),
        capabilities={
            "tools": True, 
            "stream": True,
            "temperature": True,
            "reasoning_effort": True,  # ← IMPORTANT: Enable reasoning
            "top_p": True,
        },
        max_tokens_parameter="max_completion_tokens",
        reasoning_parameter=("thinking_budget", 2000),  # ← Uses numeric thinking_budget
        thinking_tax={
            "effort_map": {
                "none": {"reserve_ratio": 0.0},
                "low": {"reserve_ratio": 0.25},
                "medium": {"reserve_ratio": 0.50},
                "high": {"reserve_ratio": 0.80},
            },
            "kind": "budget",  # ← Uses thinking_budget parameter
        },
    ),
    "gemini:fast-3-flash": ModelInfo(
        key="gemini:fast-3-flash",
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
        reasoning_parameter=("thinking_level", "low"),  
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
    "gemini:best": ModelInfo(
        key="gemini:best",
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
