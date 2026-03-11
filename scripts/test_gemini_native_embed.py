#!/usr/bin/env python3
"""Test script to check Gemini native embedding response and usage."""

import os
import sys

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), '..', 'backend')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

try:
    from backend.llm.llm_client import embed
    from core.config import settings
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the project root with backend dependencies installed")
    sys.exit(1)

def test_gemini_native_embed():
    """Test gemini:native-embed and inspect response for usage."""
    
    # Test using llm_client.embed
    
    # Test texts
    test_inputs = [
        "This is a simple test text for embedding.",
        ["First text", "Second text", "Third text"]  # Batch test
    ]
    
    print("Testing Gemini Native Embedding")
    print("=" * 50)
    
    for i, test_input in enumerate(test_inputs, 1):
        print(f"\nTest {i}: {'Batch' if isinstance(test_input, list) else 'Single'}")
        print(f"Input: {test_input}")
        print("-" * 30)
        
        try:
            # Use llm_client.embed for native embedding
            resp = embed(
                model_key="gemini:native-embed",
                texts=test_input
            )
            
            # Inspect response structure
            print(f"Response type: {type(resp)}")
            print(f"Has 'usage' attribute: {hasattr(resp, 'usage')}")
            
            if hasattr(resp, 'usage') and resp.usage:
                usage = resp.usage
                print(f"Usage type: {type(usage)}")
                print(f"Usage attributes: {dir(usage) if usage else 'None'}")
                
                if hasattr(usage, 'prompt_tokens'):
                    print(f"Prompt tokens: {usage.prompt_tokens}")
                if hasattr(usage, 'total_tokens'):
                    print(f"Total tokens: {usage.total_tokens}")
            else:
                print("No usage information found in response")
            
            # Check data
            if hasattr(resp, 'data'):
                print(f"Number of embeddings: {len(resp.data)}")
                if resp.data:
                    first_emb = resp.data[0]
                    print(f"First embedding length: {len(first_emb.embedding)}")
                    print(f"First embedding attributes: {dir(first_emb)}")
            
            # Try to extract usage the way the system does
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("chat_manager", os.path.join(backend_path, "chat", "chat_manager.py"))
                chat_manager = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(chat_manager)
                usage_dict = chat_manager._extract_usage_from_responses(resp, provider="gemini")
                print(f"\nExtracted usage dict: {usage_dict}")
            except Exception as e:
                print(f"Could not extract usage: {e}")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("Test complete")

if __name__ == "__main__":
    # Check if Gemini API key is available
    if not hasattr(settings, 'GEMINI_API_KEY') or not settings.GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not found in settings")
        print("Please set GEMINI_API_KEY in your .env file")
        sys.exit(1)
    
    test_gemini_native_embed()
