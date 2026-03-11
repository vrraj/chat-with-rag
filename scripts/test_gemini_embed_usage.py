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
        
        print(f"\n=== RAW RESPONSE ===")
        print(f"Type: {type(resp)}")
        
        # Import pprint for better formatting
        from pprint import pprint
        
        # Create a dict representation without the long embedding values
        embeddings = getattr(resp, 'embeddings', None)
        embeddings_info = None
        if embeddings:
            embeddings_info = [{
                'values_length': len(getattr(emb, 'values', [])),
                'values_preview': getattr(emb, 'values', [])[:5] + ['...'] + getattr(emb, 'values', [])[-5:] if len(getattr(emb, 'values', [])) > 10 else getattr(emb, 'values', [])
            } for emb in embeddings]
        
        resp_dict = {
            'embeddings': embeddings_info,
            'metadata': getattr(resp, 'metadata', None),
            'usage_metadata': getattr(resp, 'usage_metadata', None),
            'sdk_http_response_headers': getattr(getattr(resp, 'sdk_http_response', None), 'headers', None),
        }
        
        print(f"\nResponse as dict:")
        pprint(resp_dict, width=120, compact=True)
        
        # Show the actual object without embedding values
        print(f"\nActual object (without embedding values):")
        print(f"EmbedContentResponse(")
        print(f"  embeddings=[{len(embeddings) if embeddings else 0} items]")
        print(f"  sdk_http_response=HttpResponse(headers=<dict len={len(getattr(getattr(resp, 'sdk_http_response', None), 'headers', {}))}>)")
        print(f")")
        
        # Check for usage metadata
        print(f"\n=== USAGE METADATA ===")
        usage_meta = getattr(resp, "usage_metadata", None)
        if usage_meta is not None:
            print(f"Type: {type(usage_meta)}")
            print(f"Value: {usage_meta}")
            print(f"Attributes: {[x for x in dir(usage_meta) if not x.startswith('_')]}")
        else:
            print("No usage_metadata found")
        
        # Check embeddings
        print(f"\n=== EMBEDDINGS ===")
        embeddings = getattr(resp, "embeddings", None)
        if embeddings:
            print(f"Number of embeddings: {len(embeddings)}")
            if embeddings:
                first = embeddings[0]
                values = getattr(first, "values", None)
                if values:
                    print(f"First embedding length: {len(values)}")
                    print(f"First 8 values: {values[:8]}")
        else:
            print("No embeddings found")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_raw_gemini_embedding()
