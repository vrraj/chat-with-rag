#!/usr/bin/env python3
"""
Embed a paragraph and insert into hybrid collection for testing.

Usage:
  python scripts/test_hybrid_embed_and_search.py --text "Your paragraph here"
"""

import os
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

try:
    from qdrant_client import QdrantClient, models
except ImportError:
    print("ERROR: qdrant-client not installed. Run: pip install qdrant-client[fastembed]>=1.12.0")
    raise

try:
    from fastembed import TextEmbedding, SparseTextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder
except ImportError:
    print("ERROR: fastembed not installed. Run: pip install fastembed>=0.7.0,<0.9.0")
    raise


class HybridEmbedder:
    """Embed documents with dense + sparse and insert into Qdrant."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "document_index_finance",
        dense_model: str = "BAAI/bge-base-en-v1.5",
        sparse_model: str = "prithivida/Splade_PP_en_v1",
        cache_dir: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dense_model_name = dense_model
        self.sparse_model_name = sparse_model
        
        # Use environment variable for cache directory if not provided
        if cache_dir is None:
            cache_dir = os.path.expanduser(os.getenv("FASTEMBED_CACHE_PATH", "~/models/fastembed_cache"))
        
        # Initialize FastEmbed models
        print(f"Loading dense model: {dense_model}")
        self.dense_model = TextEmbedding(model_name=dense_model, cache_dir=cache_dir)
        
        # Probe dense vector size
        self.dense_vector_size = len(list(self.dense_model.embed(["dimension probe"]))[0])
        print(f"Dense vector size: {self.dense_vector_size}")
        
        print(f"Loading sparse model: {sparse_model}")
        self.sparse_model = SparseTextEmbedding(
            model_name=sparse_model,
            cache_dir=cache_dir,
            lazy_load=False
        )
        
        # Initialize Qdrant client
        self.client = QdrantClient(url=f"http://{host}:{port}")
        
        # Ensure collection exists
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Ensure hybrid collection exists with named vectors."""
        try:
            info = self.client.get_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' exists")
        except Exception as e:
            print(f"Collection '{self.collection_name}' not found: {e}")
            print("Creating hybrid collection with named vectors...")
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=self.dense_vector_size,
                        distance=models.Distance.COSINE,
                        on_disk=True
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=True)
                    )
                }
            )
            print(f"Created hybrid collection '{self.collection_name}' with named vectors: dense ({self.dense_vector_size}) + sparse")
    
    def embed_dense(self, text: str) -> List[float]:
        """Generate dense embedding."""
        vectors = list(self.dense_model.embed([text]))
        return vectors[0] if vectors else []
    
    def embed_sparse(self, text: str) -> Optional[models.SparseVector]:
        """Generate sparse embedding as Qdrant SparseVector."""
        vectors = list(self.sparse_model.embed([text]))
        if not vectors:
            return None
        
        sparse_embedding = vectors[0]
        
        if hasattr(sparse_embedding, "indices") and hasattr(sparse_embedding, "values"):
            indices = sparse_embedding.indices.tolist() if hasattr(sparse_embedding.indices, "tolist") else list(sparse_embedding.indices)
            values = sparse_embedding.values.tolist() if hasattr(sparse_embedding.values, "tolist") else list(sparse_embedding.values)
            return models.SparseVector(indices=indices, values=values)
        
        if hasattr(sparse_embedding, "items"):
            items = list(sparse_embedding.items())
            return models.SparseVector(
                indices=[int(index) for index, _ in items],
                values=[float(value) for _, value in items],
            )
        
        raise TypeError(f"Unsupported sparse embedding type: {type(sparse_embedding)!r}")
    
    def insert_document(self, text: str, title: str = "", url: str = "") -> int:
        """Embed and insert document into Qdrant."""
        dense_vec = self.embed_dense(text)
        sparse_vec = self.embed_sparse(text)
        
        if len(dense_vec) == 0:
            raise ValueError("Failed to generate dense embedding")
        if sparse_vec is None or len(sparse_vec.indices) == 0:
            raise ValueError("Failed to generate sparse embedding")
        
        # Generate point ID from URL (or hash of text if no URL)
        if url:
            point_id = hash(url.lower()) % (2**32)
        else:
            point_id = hash(text) % (2**32)
        
        point = models.PointStruct(
            id=point_id,
            vector={
                "dense": dense_vec,
                "sparse": sparse_vec
            },
            payload={
                "title": title or text[:50],
                "text": text,
                "url": url or "",
                "url_lower": url.lower() if url else "",
                "base_url": url or "",  # Required by list-docs-data
                "total_chunks": 1,  # Required by list-docs-data
                "updated_at": datetime.utcnow().isoformat()  # Required by list-docs-data
            }
        )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        
        print(f"Inserted document with ID: {point_id}")
        print(f"  Dense dim: {len(dense_vec)}")
        print(f"  Sparse non-zero: {len(sparse_vec.indices)}")
        
        return point_id
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search using hybrid approach."""
        dense_vec = self.embed_dense(query)
        sparse_vec = self.embed_sparse(query)
        
        if len(dense_vec) == 0:
            raise ValueError("Failed to generate dense embedding")
        if sparse_vec is None or len(sparse_vec.indices) == 0:
            raise ValueError("Failed to generate sparse embedding")
        
        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(query=dense_vec, using="dense", limit=25),
                    models.Prefetch(query=sparse_vec, using="sparse", limit=25),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
            
            return [
                {
                    "id": r.id,
                    "score": r.score,
                    "payload": r.payload,
                }
                for r in response.points
            ]
        except Exception as e:
            print(f"Hybrid search failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Embed and insert paragraph for hybrid search testing")
    parser.add_argument("--text", type=str, required=True, help="Text to embed and insert")
    parser.add_argument("--title", type=str, default="", help="Document title")
    parser.add_argument("--url", type=str, default="", help="Document URL")
    parser.add_argument("--collection", type=str, default="document_index_finance", help="Qdrant collection name")
    parser.add_argument("--dense-model", type=str, default="BAAI/bge-base-en-v1.5", help="Dense model")
    parser.add_argument("--sparse-model", type=str, default="prithivida/Splade_PP_en_v1", help="Sparse model")
    parser.add_argument("--search", type=str, default="", help="Search query after insertion")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Hybrid Embed and Insert")
    print("=" * 60)
    print(f"Text: {args.text[:100]}...")
    print(f"Collection: {args.collection}")
    print("=" * 60)
    
    try:
        embedder = HybridEmbedder(
            collection_name=args.collection,
            dense_model=args.dense_model,
            sparse_model=args.sparse_model,
        )
        
        # Insert document
        point_id = embedder.insert_document(args.text, args.title, args.url)
        print(f"\n✓ Document inserted with ID: {point_id}")
        
        # Search if query provided
        if args.search:
            print(f"\nSearching for: {args.search}")
            print("-" * 60)
            results = embedder.search(args.search, limit=5)
            
            for i, r in enumerate(results, 1):
                payload = r.get("payload", {})
                title = payload.get("title", "")[:50]
                text = payload.get("text", "")[:100]
                print(f"{i}. [{r['score']:.4f}] {title}")
                print(f"   {text}...")
                print()
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
