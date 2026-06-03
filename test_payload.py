#!/usr/bin/env python3
"""Test script to inspect Qdrant payload structure and diagnose retrieval issues."""

import sys
import os
from qdrant_client import QdrantClient
from pprint import pformat

def main():
    """Test Qdrant retrieval and inspect payload structure."""
    
    # Connect to Qdrant (hardcoded for testing)
    client = QdrantClient(
        host="localhost",
        port=6333
    )
    
    # Use the backpacking collection
    collection_name = "document_index_backpack"
    
    print(f"=== Testing Qdrant Collection: {collection_name} ===")
    print(f"Host: localhost:6333")
    print()
    
    try:
        # Get collection info
        collection_info = client.get_collection(collection_name)
        print(f"Collection exists: Yes")
        print(f"Points count: {collection_info.points_count}")
        print(f"Vectors config: {collection_info.config.params.vectors}")
        print()
    except Exception as e:
        print(f"Collection access error: {e}")
        return
    
    # Perform a simple search
    query = "how long are the trails in the Dolomites"
    
    print(f"=== Search Query: {query} ===")
    print()
    
    try:
        # Check if collection uses named vectors
        vectors_config = collection_info.config.params.vectors
        has_named_vectors = isinstance(vectors_config, dict) and "dense" in vectors_config
        
        # Generate embedding (simple approach - using a placeholder)
        # In production, you'd use the actual embedding model
        # For now, let's just scroll to get some sample points
        
        print("=== Scrolling for sample points ===")
        scroll_result = client.scroll(
            collection_name=collection_name,
            limit=5,
            with_payload=True
        )
        
        points = scroll_result[0]
        print(f"Found {len(points)} sample points")
        print()
        
        for i, point in enumerate(points):
            print(f"--- Point {i+1} ---")
            print(f"ID: {point.id}")
            print(f"Score: {point.score if hasattr(point, 'score') else 'N/A'}")
            print(f"Payload keys: {list(point.payload.keys()) if point.payload else []}")
            
            # Show sample payload structure
            if point.payload:
                print(f"Sample payload (truncated):")
                payload_sample = {k: (str(v)[:100] if isinstance(v, str) else v) for k, v in list(point.payload.items())[:5]}
                print(pformat(payload_sample))
                
                # Specifically check for 'text' field
                if 'text' in point.payload:
                    text = point.payload['text']
                    print(f"TEXT FIELD: '{text[:100]}...'" if len(text) > 100 else f"TEXT FIELD: '{text}'")
                else:
                    print("TEXT FIELD: MISSING")
            
            print()
        
        # Now try an actual search with a dummy vector (just to see the structure)
        # We'll use a random vector of the correct dimension
        import random
        vector_size = 768  # BAAI/bge-base-en-v1.5 dimension
        dummy_vector = [random.random() for _ in range(vector_size)]
        
        print("=== Search with dummy vector ===")
        if has_named_vectors:
            search_result = client.query_points(
                collection_name=collection_name,
                query=dummy_vector,
                limit=3,
                with_payload=True,
                using="dense"
            )
        else:
            search_result = client.query_points(
                collection_name=collection_name,
                query=dummy_vector,
                limit=3,
                with_payload=True
            )
        
        print(f"Search returned {len(search_result.points)} results")
        print()
        
        for i, point in enumerate(search_result.points):
            print(f"--- Search Result {i+1} ---")
            print(f"ID: {point.id}")
            print(f"Score: {point.score}")
            print(f"Payload keys: {list(point.payload.keys()) if point.payload else []}")
            
            if point.payload:
                # Check for text field
                if 'text' in point.payload:
                    text = point.payload['text']
                    print(f"TEXT FIELD: '{text[:100]}...'" if len(text) > 100 else f"TEXT FIELD: '{text}'")
                else:
                    print("TEXT FIELD: MISSING")
            
            print()
            
    except Exception as e:
        print(f"Search error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
