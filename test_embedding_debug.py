#!/usr/bin/env python3
"""
Simple test to reproduce the embedding error and verify the fix
"""

import os
import sys
sys.path.append('/Users/raj/Documents/Raj/chat-with-rag')
sys.path.append('/Users/raj/Documents/Raj/llm-adapter/src')

from dotenv import load_dotenv
load_dotenv(override=False)

from backend.llm.llm_client import embed

def test_embedding_response_format():
    """Test the embedding response format to understand the structure"""
    print("Testing embedding response format...")
    
    try:
        # Test with a simple text
        response = embed(
            model_key="openai:text-embedding-3-small",  # or "gemini:native-embed"
            texts="Hello world"
        )
        
        print(f"Response type: {type(response)}")
        print(f"Has data attribute: {hasattr(response, 'data')}")
        
        if hasattr(response, 'data'):
            print(f"Data type: {type(response.data)}")
            print(f"Data length: {len(response.data)}")
            
            if response.data:
                print(f"First item type: {type(response.data[0])}")
                print(f"First item: {response.data[0][:5] if len(response.data[0]) > 5 else response.data[0]}")
                
                # This is what's causing the error
                try:
                    # This will fail if data[0] is a list, not an object
                    embedding = response.data[0].embedding
                    print(f"SUCCESS: Got embedding via .embedding attribute")
                except AttributeError as e:
                    print(f"ERROR: {e}")
                    print(f"The item does not have .embedding attribute")
                    print(f"Instead, the item itself IS the embedding vector")
        
        print(f"Has usage attribute: {hasattr(response, 'usage')}")
        if hasattr(response, 'usage'):
            print(f"Usage: {response.usage}")
            
        return response
        
    except Exception as e:
        print(f"Error during embedding test: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_current_embeddings_manager():
    """Test the current embeddings manager to reproduce the error"""
    print("\n" + "="*50)
    print("Testing current embeddings manager...")
    
    try:
        from backend.embeddings.embeddings_manager import EmbeddingsManager
        
        manager = EmbeddingsManager()
        
        # This should trigger the error
        result = manager.generate_embeddings("Hello world")
        print(f"SUCCESS: Embeddings manager worked! Result type: {type(result)}")
        return result
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("=" * 60)
    print("EMBEDDING DEBUG TEST")
    print("=" * 60)
    
    # Test 1: Check response format
    response = test_embedding_response_format()
    
    # Test 2: Try to reproduce the error
    test_current_embeddings_manager()
