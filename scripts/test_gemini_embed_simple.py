#!/usr/bin/env python3
"""Test script to check raw Gemini native embedding response and usage."""

import os
import sys

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), '..', 'backend')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

def test_raw_gemini_embedding():
    """Test raw Gemini SDK embedding without any wrappers."""
    
    try:
        from google import genai  # type: ignore
    except ImportError as e:
        print("google-genai not available:", e)
        return
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set")
        return
    
    client = genai.Client(api_key=api_key)
    
    # Test with the actual model name
    model = "gemini-embedding-001"
    contents = "Hello from Gemini native embedding"
    
    print(f"Testing raw Gemini embedding")
    print(f"Model: {model}")
    print(f"Contents: {contents}")
    print("-" * 50)
    
    try:
        resp = client.models.embed_content(
            model=model,
            contents=contents,
        )
        
        print(f"\n=== RESPONSE FIELDS ===")
        print(f"Type: {type(resp)}")
        print(f"Has embeddings: {hasattr(resp, 'embeddings')}")
        print(f"Has metadata: {hasattr(resp, 'metadata')}")
        print(f"Has usage_metadata: {hasattr(resp, 'usage_metadata')}")
        print(f"Has sdk_http_response: {hasattr(resp, 'sdk_http_response')}")
        
        print(f"\n=== METADATA FIELDS ===")
        metadata = getattr(resp, 'metadata', None)
        print(f"metadata: {metadata}")
        
        usage_metadata = getattr(resp, 'usage_metadata', None)
        print(f"usage_metadata: {usage_metadata}")
        
        # Try to access prompt_token_count directly
        if usage_metadata is not None:
            try:
                prompt_tokens = getattr(usage_metadata, 'prompt_token_count', None)
                print(f"usage_metadata.prompt_token_count: {prompt_tokens}")
            except AttributeError:
                print("usage_metadata has no prompt_token_count attribute")
        else:
            print("usage_metadata is None")
        
        # Also try accessing any token-related fields on the main response
        for attr_name in ['prompt_tokens', 'total_tokens', 'token_count', 'input_tokens']:
            if hasattr(resp, attr_name):
                value = getattr(resp, attr_name)
                print(f"{attr_name}: {value}")
            else:
                print(f"{attr_name}: not found")
        
        print(f"\n=== HTTP RESPONSE HEADERS ===")
        http_resp = getattr(resp, 'sdk_http_response', None)
        if http_resp:
            headers = getattr(http_resp, 'headers', None)
            if headers:
                print("Headers:")
                for key, value in headers.items():
                    print(f"  {key}: {value}")
        
        print(f"\n=== EMBEDDINGS INFO ===")
        embeddings = getattr(resp, 'embeddings', None)
        if embeddings:
            print(f"Number of embeddings: {len(embeddings)}")
            for i, emb in enumerate(embeddings):
                values = getattr(emb, 'values', None)
                if values:
                    print(f"  Embedding {i}: length={len(values)}, first_5={values[:5]}, last_5={values[-5:]}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_raw_gemini_embedding()
