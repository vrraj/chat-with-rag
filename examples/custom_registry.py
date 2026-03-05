"""
Custom Model Registry Example for chat-with-rag

This file demonstrates how to create custom model definitions
that extend or override the default llm-adapter registry.
"""

from llm_adapter.model_registry import ModelInfo, Pricing

# Custom registry - these models will be merged with the default registry
# Custom models can override existing keys or add new ones
REGISTRY = {
    # Override an existing model with custom pricing
    "openai:gpt-4o": ModelInfo(
        key="openai:gpt-4o",
        provider="openai",
        model="gpt-4o",
        endpoint="chat_completions",
        pricing=Pricing(
            input_per_mm=0.005,  # Custom pricing
            output_per_mm=0.015,
            cached_input_per_mm=0.0025
        ),
        capabilities={"reasoning": True, "tools": True},
        param_policy={
            "allowed": {"max_output_tokens", "temperature", "top_p", "tools", "tool_choice"},
            "disabled": set()
        }
    ),
    
    # Add a completely new custom model
    "custom:experimental": ModelInfo(
        key="custom:experimental",
        provider="openai",
        model="gpt-4-turbo-preview",
        endpoint="chat_completions",
        pricing=Pricing(
            input_per_mm=0.01,
            output_per_mm=0.03
        ),
        capabilities={
            "experimental": True,
            "max_tokens": 4096
        },
        param_policy={
            "allowed": {"max_output_tokens", "temperature", "top_p"},
            "disabled": {"tools"}  # Disable tools for this experimental model
        }
    ),
    
    # Add a Gemini model with custom reasoning policy
    "gemini:custom-reasoning": ModelInfo(
        key="gemini:custom-reasoning",
        provider="gemini",
        model="gemini-2.0-flash-exp",
        endpoint="gemini_sdk",
        pricing=Pricing(
            input_per_mm=0.001,
            output_per_mm=0.004,
            cached_input_per_mm=0.0005
        ),
        reasoning_policy={
            "mode": "gemini_level",
            "effort_map": {
                "minimal": {"thinking_level": 1},
                "low": {"thinking_level": 2},
                "medium": {"thinking_level": 3},
                "high": {"thinking_level": 4}
            }
        },
        param_policy={
            "allowed": {"reasoning_effort", "max_output_tokens", "temperature", "top_p"},
            "disabled": {"stream"}
        }
    ),
}
