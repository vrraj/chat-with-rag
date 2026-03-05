"""
LLM Client Interface

Thin wrapper around llm-adapter package providing a clean, simple interface
for text generation, embeddings, and model pricing.

Functions:
- generate(): Text generation with normalized response
- generate_raw(): Text generation returning raw provider response  
- embed(): Text embeddings
- get_pricing_for_model(): Model pricing information
- get_model_info(): Model information from registry
"""

from llm_adapter import llm_adapter, LLMError


def generate(*, model_key: str, input: str, tools=None, **kwargs):
    """
    Generate text using the specified model.
    
    Args:
        model_key: Registry model identifier (e.g., "openai:gpt-4o", "gemini:fast")
        input: Text prompt or messages
        tools: Optional tool definitions
        **kwargs: Additional parameters (stream, temperature, max_output_tokens, etc.)
    
    Returns:
        Normalized LLMResult dict with standardized fields
    """
    try:
        # Filter out conflicting parameters that might be in kwargs
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['model', 'input', 'model_key']}
        response = llm_adapter.create(
            model=model_key,
            input=input,
            tools=tools,
            **filtered_kwargs
        )
        return llm_adapter.normalize_adapter_response(response)
    except LLMError:
        raise


def generate_raw(*, model_key: str, input: str, tools=None, **kwargs):
    """
    Generate text without normalization - returns raw provider response.
    
    Useful when you need provider-specific data or debugging.
    
    Args:
        model_key: Registry model identifier
        input: Text prompt or messages
        tools: Optional tool definitions
        **kwargs: Additional parameters
    
    Returns:
        Raw provider response (not normalized)
    """
    try:
        return llm_adapter.create(
            model=model_key,
            input=input,
            tools=tools,
            **kwargs
        )
    except LLMError:
        raise


def embed(*, model_key: str, texts: str | list[str], **kwargs):
    """
    Generate embeddings for the given text(s).
    
    Args:
        model_key: Registry model identifier (e.g., "openai:text-embedding-3-small", "gemini:native-embed")
        texts: Single text or list of texts to embed
        **kwargs: Additional parameters (dimensions, task_type, normalize_embedding, etc.)
    
    Returns:
        Raw embedding response from provider
    """
    try:
        # Filter out conflicting parameters that might be in kwargs
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['model', 'input', 'model_key']}
        return llm_adapter.create_embedding(
            model=model_key,
            input=texts,
            **filtered_kwargs
        )
    except LLMError:
        raise


def get_pricing_for_model(*, model_key: str) -> dict | None:
    """
    Get pricing information for a model from the registry.
    
    Args:
        model_key: Registry model identifier
    
    Returns:
        Pricing information dict or None if not found
    """
    try:
        return llm_adapter.get_pricing_for_model(model_key)
    except LLMError:
        raise


def get_model_info(*, model_key: str):
    """
    Get model information from the registry.
    
    Args:
        model_key: Registry model identifier
    
    Returns:
        ModelInfo object from the registry
    """
    try:
        return llm_adapter._lookup_model_info_from_registry(model_key)
    except LLMError:
        raise
