# custom_registry.py
# Example custom registry demonstrating the new parameter validation system
#
# KEY FEATURES DEMONSTRATED:
# 1. allowed lists - Explicitly define which provider-specific parameters are permitted
# 2. disabled lists - Parameters to filter out (cross-provider contamination prevention)
# 3. reasoning_policy - How reasoning_effort is converted to provider-specific format
# 4. Provider isolation - Gemini params can't reach OpenAI APIs and vice versa
#
# PARAMETER POLICY STRUCTURE:
# - allowed: Set of provider-specific parameters that can reach the API
# - disabled: Set of parameters to filter out (includes cross-provider params)
# - Framework params (model, input, messages, stream) are NOT in allowed lists
#
# EXAMPLES INCLUDED:
# - OpenAI reasoning models (reasoning_effort allowed, temperature/top_p disabled)
# - Gemini reasoning models (full Gemini parameter support)
# - OpenAI standard models (basic generation params only)
# - Embedding models (provider-specific dimension handling)
#
# REQUIREMENTS:
# - Set OPENAI_API_KEY and/or GEMINI_API_KEY for testing
# - export OPENAI_API_KEY=...
# - export GEMINI_API_KEY=...

from llm_adapter.model_registry import ModelInfo, Pricing, validate_registry

REGISTRY = {
    "openai:custom_reasoning_o3-mini": ModelInfo(
        provider="openai",
        model="o3-mini",
        endpoint="responses",
        pricing=Pricing(input_per_mm=1.10, output_per_mm=4.40),
        limits={"max_output_tokens": 2000},
        capabilities={
            "assistant_role": "assistant",
            "reasoning_effort": True,
        },
        param_policy={
            "allowed": {"max_output_tokens", "reasoning_effort", "reasoning", "tools", "tool_choice"},
            "disabled": {"stream", "temperature", "top_p", "include_thoughts"}
        },
        reasoning_policy={
            "mode": "openai_effort",
            "default": "low",
        },
        reasoning_parameter=("reasoning_effort", "low"),
    ),
    "openai:custom_reasoning_gpt-5-mini": ModelInfo(
        provider="openai",
        model="gpt-5-mini",
        endpoint="responses",
        pricing=Pricing(input_per_mm=0.25, output_per_mm=2.00),
        limits={"max_output_tokens": 2000},
        capabilities={
            "assistant_role": "assistant",
            "reasoning_effort": True,
        },
        param_policy={
            "allowed": {"max_output_tokens", "reasoning_effort", "reasoning", "tools", "tool_choice"},
            "disabled": {"stream", "temperature", "top_p", "include_thoughts"}
        },
        reasoning_policy={
            "mode": "openai_effort",
            "default": "minimal",
        },
        reasoning_parameter=("reasoning_effort", "minimal"),
    ),
    "gemini:custom_reasoning_flash": ModelInfo(
        provider="gemini",
        model="models/gemini-2.5-flash",
        endpoint="chat_completions",
        pricing=Pricing(input_per_mm=0.30, output_per_mm=2.50),
        limits={"max_output_tokens": 2000},
        capabilities={
            "assistant_role": "model",
            "reasoning_effort": True,
            "temperature": True,
            "top_p": True,
        },
        param_policy={
            "allowed": {"max_output_tokens", "reasoning_effort", "include_thoughts", "thinking_level", "thinking_budget", "temperature", "top_p", "tools", "tool_choice"},
            "disabled": set()
        },
        reasoning_policy={
            "mode": "gemini_budget",
            "param": "thinking_budget",
            "default": "medium",
            "budget_map": {
                "none": 0,
                "minimal": 500,
                "low": 1000,
                "medium": 2000,
                "high": 5000,
            },
            "counts_against_output": True,
        },
        reasoning_parameter=("thinking_budget", 1000),
    ),
    "openai:custom_standard_gpt-4o": ModelInfo(
        provider="openai",
        model="gpt-4o",
        endpoint="responses",
        pricing=Pricing(input_per_mm=2.50, output_per_mm=10.00, cached_input_per_mm=1.25),
        limits={"max_output_tokens": 2000},
        capabilities={
            "assistant_role": "assistant",
        },
        param_policy={
            "allowed": {"max_output_tokens", "temperature", "top_p", "tools", "tool_choice"},
            "disabled": {"reasoning_effort", "stream", "include_thoughts"}
        },
    ),
    "openai:custom_embedding": ModelInfo(
        provider="openai",
        model="text-embedding-3-small",
        endpoint="embeddings",
        pricing=Pricing(input_per_mm=0.02, output_per_mm=0.0),
        param_policy={
            "allowed": {"normalize_embedding", "dimensions"},
            "disabled": {"include_thoughts", "output_dimensionality"}
        },
        capabilities={
            "dimensions": 1536,
            "assistant_role": "assistant",
        },
    ),
}

# Optional (recommended in dev/CI)
validate_registry(REGISTRY, strict=False)
