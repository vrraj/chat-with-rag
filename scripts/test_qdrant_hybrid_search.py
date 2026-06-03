#!/usr/bin/env python3
"""
Test hybrid search on the existing document_index_finance collection using QdrantDB.
This script uses the QdrantDB wrapper class to perform hybrid search.
"""

import sys
import os

# Set a dummy API key to bypass Settings validation
if not os.getenv("OPENAI_API_KEY") and not os.getenv("GEMINI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "dummy-key-for-testing"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.db.qdrant_db import QdrantDB

# Configuration
COLLECTION_NAME = "document_index_finance"
TEST_QUERY = "Redwell"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
EMBEDDING_MODEL_KEY = os.getenv("EMBEDDING_MODEL_KEY", "BAAI/bge-base-en-v1.5")

def main():
    """Main test function."""
    print("=" * 60)
    print("Qdrant Hybrid Search Test on Existing Collection")
    print("=" * 60)
    
    # Initialize QdrantDB
    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    qdrant_db = QdrantDB(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        collection_name=COLLECTION_NAME,
        embedding_model_key=EMBEDDING_MODEL_KEY,
    )
    
    # Perform hybrid search
    print(f"\nPerforming hybrid search for: '{TEST_QUERY}'")
    try:
        results = qdrant_db.search_similar_hybrid(
            query=TEST_QUERY,
            limit=5,
            score_threshold=None,
            exact=False,
            with_payload=True,
        )
        
        print(f"\nFound {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result.get('score', 0):.4f}")
            print(f"   Text: {result.get('payload', {}).get('text', '')[:300]}")
            print(f"   URL: {result.get('payload', {}).get('url_lower', '')}")
            
    except Exception as e:
        print(f"Error during hybrid search: {e}")
        print("\nFalling back to dense-only search...")
        try:
            results = qdrant_db.search_similar(
                query=TEST_QUERY,
                limit=5,
                score_threshold=None,
                exact=False,
                with_payload=True,
            )
            
            print(f"\nFound {len(results)} results (dense-only):")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Score: {result.get('score', 0):.4f}")
                print(f"   Text: {result.get('payload', {}).get('text', '')[:300]}")
                print(f"   URL: {result.get('payload', {}).get('url_lower', '')}")
        except Exception as e2:
            print(f"Error during dense search: {e2}")

if __name__ == "__main__":
    main()
