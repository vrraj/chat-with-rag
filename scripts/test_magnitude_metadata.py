#!/usr/bin/env python3
"""
Test script to verify magnitude metadata is working correctly.
"""
import os
import sys

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from embeddings.embeddings_manager import EmbeddingsManager
from backend.llm.llm_client import embed

def test_magnitude_metadata():
    print("Testing magnitude metadata...")
    
    # Test 1: Single embedding
    print("\n1. Testing single embedding...")
    try:
        result = embed(model_key="gemini:native-embed", texts="Test text for magnitude metadata")
        
        if result.data:
            item = result.data[0]
            print(f"   Embedding length: {len(item.embedding)}")
            print(f"   Has magnitude: {hasattr(item, 'magnitude')}")
            print(f"   Magnitude: {getattr(item, 'magnitude', 'None')}")
            print(f"   Normalized: {getattr(item, 'normalized', 'None')}")
            print(f"   Provider: {getattr(item, 'provider', 'None')}")
        else:
            print("   No data returned")
            
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 2: Batch embeddings
    print("\n2. Testing batch embeddings...")
    try:
        result = embed(model_key="gemini:native-embed", texts=["Text 1", "Text 2", "Text 3"])
        
        for i, item in enumerate(result.data):
            print(f"   Item {i}:")
            print(f"     Has magnitude: {hasattr(item, 'magnitude')}")
            print(f"     Magnitude: {getattr(item, 'magnitude', 'None')}")
            print(f"     Normalized: {getattr(item, 'normalized', 'None')}")
            print(f"     Provider: {getattr(item, 'provider', 'None')}")
            
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 3: EmbeddingsManager
    print("\n3. Testing EmbeddingsManager...")
    try:
        manager = EmbeddingsManager()
        result = manager.generate_embeddings("Test text")
        
        if isinstance(result, tuple):
            embedding, magnitude, normalized, provider = result
            print(f"   Embedding length: {len(embedding)}")
            print(f"   Magnitude: {magnitude}")
            print(f"   Normalized: {normalized}")
            print(f"   Provider: {provider}")
        else:
            print(f"   Legacy format: {type(result)}")
            
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    test_magnitude_metadata()
