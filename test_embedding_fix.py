#!/usr/bin/env python3
"""
Simple test to verify the embedding fix works
"""

import os
import sys
sys.path.append('/Users/raj/Documents/Raj/chat-with-rag')
sys.path.append('/Users/raj/Documents/Raj/llm-adapter/src')

# Mock the dependencies that might be missing
sys.modules['tiktoken'] = type(sys)('tiktoken')
sys.modules['tiktoken'].get_encoding = lambda name: type('MockEncoding', (), {'encode': lambda self, text, **kwargs: text.split()})()

from dotenv import load_dotenv
load_dotenv(override=False)

def test_embedding_format():
    """Test what the embedding response format looks like"""
    print("Testing embedding response format...")
    
    try:
        from backend.llm.llm_client import embed
        
        # Test with a simple text using a model that should work
        response = embed(
            model_key="openai:text-embedding-3-small",
            texts="Hello world"
        )
        
        print(f"✅ SUCCESS: Got embedding response")
        print(f"Response type: {type(response)}")
        print(f"Has data: {hasattr(response, 'data')}")
        
        if hasattr(response, 'data'):
            print(f"Data type: {type(response.data)}")
            print(f"Data length: {len(response.data)}")
            if response.data:
                print(f"First item type: {type(response.data[0])}")
                print(f"First item is list: {isinstance(response.data[0], list)}")
                if isinstance(response.data[0], list):
                    print(f"First 5 values: {response.data[0][:5]}")
        
        print(f"Has metadata: {hasattr(response, 'metadata')}")
        if hasattr(response, 'metadata') and response.metadata:
            print(f"Metadata keys: {list(response.metadata.keys())}")
            
        print(f"Has usage: {hasattr(response, 'usage')}")
        print(f"Has normalized: {hasattr(response, 'normalized')}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_embeddings_manager_fix():
    """Test that the embeddings manager now works"""
    print("\n" + "="*50)
    print("Testing embeddings manager fix...")
    
    try:
        # Mock more dependencies if needed
        import logging
        logging.basicConfig(level=logging.INFO)
        
        from backend.embeddings.embeddings_manager import EmbeddingsManager
        
        manager = EmbeddingsManager()
        
        # This should work now
        result = manager.generate_embeddings("Hello world")
        print(f"✅ SUCCESS: Embeddings manager worked!")
        print(f"Result type: {type(result)}")
        
        if isinstance(result, tuple):
            embedding, magnitude, normalized, provider = result
            print(f"Embedding length: {len(embedding) if embedding else 0}")
            print(f"Magnitude: {magnitude}")
            print(f"Normalized: {normalized}")
            print(f"Provider: {provider}")
        else:
            print(f"Embedding length: {len(result) if result else 0}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING EMBEDDING FIX")
    print("=" * 60)
    
    # Test 1: Check response format
    success1 = test_embedding_format()
    
    # Test 2: Test embeddings manager
    success2 = test_embeddings_manager_fix()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Response format test: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Embeddings manager test: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        print("🎉 All tests passed! The fix works.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
