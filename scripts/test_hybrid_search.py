#!/usr/bin/env python3
"""
Test Qdrant hybrid search with dense + sparse embeddings using FastEmbed.
This script tests the new hybrid search implementation with a fixed text.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

# Configuration
COLLECTION_NAME = "test_hybrid_collection"
DENSE_MODEL = "BAAI/bge-base-en-v1.5"
SPARSE_MODEL = "prithivida/Splade_PP_en_v1"
TEST_TEXT = "Finance policy updates and risk outlook for Q4 2024"
TEST_QUERY = "financial risk management"

def create_hybrid_collection(client: QdrantClient, collection_name: str):
    """Create a Qdrant collection with named dense + sparse vectors for hybrid search."""
    try:
        # Check if collection exists
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

def generate_hybrid_embeddings(text: str):
    """Generate both dense and sparse embeddings using FastEmbed."""
    print(f"Generating dense embeddings with {DENSE_MODEL}...")
    dense_model = TextEmbedding(model_name=DENSE_MODEL)
    dense_embeddings = list(dense_model.embed([text]))
    dense_vector = list(dense_embeddings[0])
    print(f"Dense embedding shape: {len(dense_vector)}")
    
    print(f"Generating sparse embeddings with {SPARSE_MODEL}...")
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    sparse_embeddings = list(sparse_model.embed([text]))
    sparse_embedding = sparse_embeddings[0]
    
    # Convert sparse embedding to dict format
    sparse_dict = {
        "indices": sparse_embedding.indices.tolist() if hasattr(sparse_embedding.indices, "tolist") else list(sparse_embedding.indices),
        "values": sparse_embedding.values.tolist() if hasattr(sparse_embedding.values, "tolist") else list(sparse_embedding.values)
    }
    print(f"Sparse embedding: {len(sparse_dict['indices'])} non-zero dimensions")
    
    return dense_vector, sparse_dict

def insert_test_document(client: QdrantClient, collection_name: str, text: str):
    """Insert a test document with both dense and sparse embeddings."""
    dense_vector, sparse_dict = generate_hybrid_embeddings(text)
    
    point = models.PointStruct(
        id=1,
        vector={
            "dense": dense_vector,
            "sparse": models.SparseVector(
                indices=sparse_dict["indices"],
                values=sparse_dict["values"]
            )
        },
        payload={
            "text": text,
            "source": "test_script"
        }
    )
    
    client.upsert(
        collection_name=collection_name,
        points=[point]
    )
    print(f"Inserted document with ID 1")

def hybrid_search(client: QdrantClient, collection_name: str, query: str):
    """Perform hybrid search using dense + sparse embeddings."""
    print(f"\nPerforming hybrid search for: '{query}'")
    
    # Generate query embeddings
    dense_model = TextEmbedding(model_name=DENSE_MODEL)
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    
    dense_query = list(dense_model.embed([query]))[0]
    sparse_query = sparse_model.embed([query])[0]
    
    # Convert sparse query to dict
    sparse_query_dict = {
        "indices": sparse_query.indices.tolist() if hasattr(sparse_query.indices, "tolist") else list(sparse_query.indices),
        "values": sparse_query.values.tolist() if hasattr(sparse_query.values, "tolist") else list(sparse_query.values)
    }
    
    # Perform hybrid search using fusion (RRF)
    # Note: Qdrant's query_points supports prefer parameter for fusion
    try:
        # Try using prefer for fusion search
        response = client.query_points(
            collection_name=collection_name,
            query=models.NamedVector(
                name="dense",
                vector=list(dense_query)
            ),
            using="dense",
            limit=5,
            with_payload=True
        )
        
        print(f"\nSearch results (dense only):")
        for hit in response:
            print(f"  Score: {hit.score:.4f}, Text: {hit.payload.get('text', '')[:50]}...")
        
        # Try sparse search
        response = client.query_points(
            collection_name=collection_name,
            query=models.NamedSparseVector(
                name="sparse",
                vector=models.SparseVector(
                    indices=sparse_query_dict["indices"],
                    values=sparse_query_dict["values"]
                )
            ),
            limit=5,
            with_payload=True
        )
        
        print(f"\nSearch results (sparse only):")
        for hit in response:
            print(f"  Score: {hit.score:.4f}, Text: {hit.payload.get('text', '')[:50]}...")
        
        # Try fusion search (dense + sparse)
        response = client.query_points(
            collection_name=collection_name,
            query=models.FusionQuery(
                fusion=models.Fusion(
                    rrf=models.RRF()  # Reciprocal Rank Fusion
                )
            ),
            prefetch=[
                models.Prefetch(
                    query=models.NamedVector(
                        name="dense",
                        vector=list(dense_query)
                    ),
                    using="dense"
                ),
                models.Prefetch(
                    query=models.NamedSparseVector(
                        name="sparse",
                        vector=models.SparseVector(
                            indices=sparse_query_dict["indices"],
                            values=sparse_query_dict["values"]
                        )
                    )
                )
            ],
            limit=5,
            with_payload=True
        )
        
        print(f"\nSearch results (hybrid fusion):")
        for hit in response:
            print(f"  Score: {hit.score:.4f}, Text: {hit.payload.get('text', '')[:50]}...")
            
    except Exception as e:
        print(f"Error during hybrid search: {e}")
        print("Falling back to dense-only search...")
        
        # Fallback to dense-only search
        response = client.query_points(
            collection_name=collection_name,
            query=list(dense_query),
            using="dense",
            limit=5,
            with_payload=True
        )
        
        print(f"\nSearch results (dense fallback):")
        for hit in response:
            print(f"  Score: {hit.score:.4f}, Text: {hit.payload.get('text', '')[:50]}...")

def main():
    """Main test function."""
    print("=" * 60)
    print("Qdrant Hybrid Search Test")
    print("=" * 60)
    
    # Connect to Qdrant
    client = QdrantClient(host="localhost", port=6333)
    print(f"Connected to Qdrant at localhost:6333")
    
    # Create hybrid collection
    create_hybrid_collection(client, COLLECTION_NAME)
    
    # Insert test document
    print(f"\nInserting test document: '{TEST_TEXT}'")
    insert_test_document(client, COLLECTION_NAME, TEST_TEXT)
    
    # Perform hybrid search
    hybrid_search(client, COLLECTION_NAME, TEST_QUERY)
    
    # Cleanup
    print(f"\nCleaning up test collection...")
    client.delete_collection(COLLECTION_NAME)
    print("Test complete!")

if __name__ == "__main__":
    main()
