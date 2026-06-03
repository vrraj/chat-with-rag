#!/usr/bin/env python3
"""
Test the routing layer for embeddings (EmbeddingsManager → EmbeddingRouter → FastEmbedEmbeddingProvider).
This script tests the actual codebase routing instead of using FastEmbed directly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set dummy API key to satisfy Settings validation
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy-key")

from backend.embeddings.embeddings_manager import EmbeddingsManager
from backend.db import QdrantDB
from backend.core.config import settings
from qdrant_client import QdrantClient, models

# Configuration
COLLECTION_NAME = "test_routing_collection"
TEST_TEXT = "Finance policy updates and risk outlook for Q4 2024"
TEST_QUERY = "financial risk management"

def create_test_collection(client: QdrantClient, collection_name: str):
    """Create a test collection with named vectors."""
    try:
        client.get_collection(collection_name)
        print(f"Collection {collection_name} already exists. Recreating...")
        client.delete_collection(collection_name)
    except Exception:
        print(f"Creating new collection {collection_name}...")
    
    # Create collection with named dense + sparse vectors
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(size=768, distance=models.Distance.COSINE, on_disk=True)
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=True)
            )
        }
    )
    print(f"Created hybrid collection {collection_name} with named vectors: dense (768) + sparse")

def test_embeddings_manager_routing():
    """Test the EmbeddingsManager routing layer."""
    print("=" * 60)
    print("Testing EmbeddingsManager Routing Layer")
    print("=" * 60)
    
    # Check current active domain
    print(f"\nActive domain: {settings.active_domain}")
    domain_config = settings.DOMAIN_EMBEDDING_CONFIG.get(settings.active_domain, {})
    print(f"Domain config: {domain_config}")
    
    # Check model_type and vector_type
    model_type = domain_config.get("model_type")
    vector_type = domain_config.get("vector_type")
    print(f"Model type: {model_type}")
    print(f"Vector type: {vector_type}")
    
    # Create EmbeddingsManager with finance domain (local model)
    print(f"\nCreating EmbeddingsManager for 'finance' domain...")
    finance_config = settings.DOMAIN_EMBEDDING_CONFIG.get("finance", {})
    
    # Temporarily switch to finance domain for testing
    original_domain = settings.active_domain
    settings.active_domain = "finance"
    
    try:
        # Create EmbeddingsManager (it handles QdrantDB internally)
        embeddings_manager = EmbeddingsManager(active_domain="finance")
        
        print(f"EmbeddingsManager created:")
        print(f"  - Active domain: {embeddings_manager.active_domain}")
        print(f"  - Collection: {embeddings_manager.collection_name}")
        print(f"  - Model key: {embeddings_manager.embedding_model_key}")
        print(f"  - Model type: {embeddings_manager.model_type}")
        print(f"  - Vector type: {embeddings_manager.vector_type}")
        
        # Test dense embedding generation
        print(f"\n--- Testing dense embedding generation ---")
        dense_result = embeddings_manager.generate_embeddings(TEST_TEXT)
        print(f"Dense embedding generated:")
        print(f"  - Type: {type(dense_result)}")
        print(f"  - Shape: {len(dense_result) if isinstance(dense_result, list) else 'N/A'}")
        
        # Test sparse embedding generation (if hybrid)
        if embeddings_manager.vector_type == "hybrid":
            print(f"\n--- Testing sparse embedding generation ---")
            sparse_result = embeddings_manager.generate_sparse_embeddings(TEST_TEXT)
            print(f"Sparse embedding generated:")
            print(f"  - Type: {type(sparse_result)}")
            if isinstance(sparse_result, dict):
                print(f"  - Indices: {len(sparse_result.get('indices', []))}")
                print(f"  - Values: {len(sparse_result.get('values', []))}")
        
        # Test document processing
        print(f"\n--- Testing document processing ---")
        document = {
            "text": TEST_TEXT + " This is additional text to ensure the chunk is not empty and has sufficient content for embedding generation.",
            "url": "https://example.com/test",
            "title": "Test Document",
            "doc_type": "test"
        }
        
        processed_chunks = embeddings_manager.process_document(document)
        print(f"Document processed:")
        print(f"  - Chunks: {len(processed_chunks)}")
        if processed_chunks:
            first_chunk = processed_chunks[0]
            print(f"  - First chunk keys: {list(first_chunk.keys())}")
            if "dense" in first_chunk:
                print(f"  - Has dense vector: Yes")
            if "sparse" in first_chunk:
                print(f"  - Has sparse vector: Yes")
            if "vector" in first_chunk:
                print(f"  - Has unnamed vector: Yes")
        
        return embeddings_manager, processed_chunks
        
    finally:
        # Restore original domain
        settings.active_domain = original_domain

def test_qdrant_integration(embeddings_manager, processed_chunks):
    """Test Qdrant integration with named vectors."""
    print(f"\n--- Testing Qdrant integration ---")
    
    # Create test collection
    client = QdrantClient(host="localhost", port=6333)
    create_test_collection(client, COLLECTION_NAME)
    
    try:
        # Temporarily change collection name for testing
        original_collection = embeddings_manager.collection_name
        embeddings_manager.collection_name = COLLECTION_NAME
        embeddings_manager.qdrant.collection_name = COLLECTION_NAME
        embeddings_manager.qdrant_db.collection_name = COLLECTION_NAME
        
        # Index processed chunks using index_chunks
        print(f"Indexing {len(processed_chunks)} chunks into Qdrant...")
        result = embeddings_manager.index_chunks(
            chunks=processed_chunks,
            force_delete=False
        )
        print(f"Index result: {result}")
        
        if result:
            # Test search
            print(f"\n--- Testing search ---")
            try:
                results = embeddings_manager.qdrant_db.search_similar(
                    query=TEST_QUERY,
                    limit=5
                )
                print(f"Search results: {len(results)}")
                for i, result in enumerate(results[:3]):
                    print(f"  Result {i+1}: Score={result['score']:.4f}, Text={result['payload'].get('text', '')[:50]}...")
            except Exception as e:
                print(f"Search error: {e}")
        
    finally:
        # Restore original collection
        embeddings_manager.collection_name = original_collection
        embeddings_manager.qdrant.collection_name = original_collection
        embeddings_manager.qdrant_db.collection_name = original_collection
        
        # Cleanup test collection
        print(f"\nCleaning up test collection...")
        client.delete_collection(COLLECTION_NAME)

def main():
    """Main test function."""
    print("=" * 60)
    print("Routing Layer Test")
    print("=" * 60)
    
    # Test embeddings manager routing
    embeddings_manager, processed_chunks = test_embeddings_manager_routing()
    
    # Test Qdrant integration
    if processed_chunks:
        test_qdrant_integration(embeddings_manager, processed_chunks)
    
    print(f"\nTest complete!")

if __name__ == "__main__":
    main()
