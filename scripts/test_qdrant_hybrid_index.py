#!/usr/bin/env python3
"""
Test Qdrant Hybrid Search with FastEmbed (Dense + Sparse)

This script demonstrates hybrid search using:
- Dense embeddings: BAAI/bge-small-en-v1.5
- Sparse embeddings: prithivida/Splade_PP_en_v1
- Fusion: RRF (Reciprocal Rank Fusion)

Usage:
  python scripts/test_qdrant_hybrid_index.py --query "finance policy updates"
"""

import os
import argparse
from typing import List, Dict, Any, Optional, Union

try:
    from qdrant_client import QdrantClient, models
    from qdrant_client.models import Distance, VectorParams, PayloadSchemaType
except ImportError:
    print("ERROR: qdrant-client not installed. Run: pip install qdrant-client[fastembed]>=1.12.0")
    raise

try:
    from fastembed import TextEmbedding, SparseTextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder
except ImportError:
    print("ERROR: fastembed not installed. Run: pip install fastembed>=0.7.0,<0.9.0")
    raise


class HybridSearcher:
    """Hybrid search using dense + sparse embeddings with Fusion."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "document_index_finance",
        dense_model: str = "BAAI/bge-small-en-v1.5",
        sparse_model: str = "prithivida/Splade_PP_en_v1",
        reranker_model: str = "BAAI/bge-reranker-base",
        cache_dir: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dense_model_name = dense_model
        self.sparse_model_name = sparse_model
        self.reranker_model_name = reranker_model
        
        # Use environment variable for cache directory if not provided
        if cache_dir is None:
            cache_dir = os.path.expanduser(os.getenv("FASTEMBED_CACHE_PATH", "~/models/fastembed_cache"))
        
        # Initialize FastEmbed models
        print(f"Loading dense model: {dense_model}")
        self.dense_model = TextEmbedding(model_name=dense_model, cache_dir=cache_dir)
        
        # Probe once so collection creation uses the actual dense vector size.
        # BGE-small is 384 dimensions, BGE-base is 768 dimensions, etc.
        self.dense_vector_size = len(self.embed_dense("dimension probe"))
        if self.dense_vector_size == 0:
            raise ValueError(f"Failed to generate probe embedding for dense model: {dense_model}")
        
        print(f"Loading sparse model: {sparse_model}")
        self.sparse_model = SparseTextEmbedding(
            model_name=sparse_model,
            cache_dir=cache_dir,
            lazy_load=False  # Force early crash if model is broken
        )
        
        print(f"Loading reranker model: {reranker_model}")
        self.reranker = TextCrossEncoder(model_name=reranker_model, cache_dir=cache_dir)
        
        # Initialize Qdrant client
        self.client = QdrantClient(url=f"http://{host}:{port}")
        
        # Check if collection exists and has named vectors
        self._ensure_collection_setup()
    
    def _ensure_collection_setup(self):
        """Ensure collection exists with named vectors for hybrid search."""
        try:
            info = self.client.get_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' exists")
            
            # Check if it has named vectors
            vectors_cfg = info.config.params.vectors
            if isinstance(vectors_cfg, dict):
                print(f"Collection has named vectors: {list(vectors_cfg.keys())}")
            else:
                print(f"Collection uses single vector (not hybrid-ready)")
                print("Warning: Hybrid search requires named vectors. Consider recreating the collection.")
        except Exception as e:
            print(f"Collection '{self.collection_name}' not found: {e}")
            print("Creating hybrid collection with named vectors...")
            
            # Create hybrid collection with named vectors
            self.client.create_collection(
                collection_name=self.collection_name,
                # 1. DENSE: The "Semantic" side
                vectors_config={
                    "dense": models.VectorParams(
                        size=self.dense_vector_size,
                        distance=models.Distance.COSINE,
                        on_disk=True  # Keep vectors on disk to save RAM
                    )
                },
                # 2. SPARSE: The "Keyword" side
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=True,
                        )
                    )
                }
            )
            print(f"Created hybrid collection '{self.collection_name}' with named vectors: dense ({self.dense_vector_size}) + sparse")
    
    def embed_dense(self, text: str) -> List[float]:
        """Generate dense embedding for query."""
        vectors = list(self.dense_model.embed([text]))
        return vectors[0] if vectors else []
    
    def embed_sparse(self, text: str) -> Optional[models.SparseVector]:
        """Generate sparse embedding for query as a Qdrant SparseVector."""
        vectors = list(self.sparse_model.embed([text]))
        if not vectors:
            return None

        sparse_embedding = vectors[0]

        # FastEmbed returns SparseEmbedding objects with indices/values arrays.
        if hasattr(sparse_embedding, "indices") and hasattr(sparse_embedding, "values"):
            indices = sparse_embedding.indices.tolist() if hasattr(sparse_embedding.indices, "tolist") else list(sparse_embedding.indices)
            values = sparse_embedding.values.tolist() if hasattr(sparse_embedding.values, "tolist") else list(sparse_embedding.values)
            return models.SparseVector(indices=indices, values=values)

        # Defensive fallback for dict-like sparse vectors from other libraries.
        if hasattr(sparse_embedding, "items"):
            items = list(sparse_embedding.items())
            return models.SparseVector(
                indices=[int(index) for index, _ in items],
                values=[float(value) for _, value in items],
            )

        raise TypeError(f"Unsupported sparse embedding type: {type(sparse_embedding)!r}")
    
    def search_hybrid(
        self,
        query: str,
        limit: int = 10,
        dense_limit: int = 25,
        sparse_limit: int = 25,
        use_reranker: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using dense + sparse embeddings with Fusion.
        
        Args:
            query: Query text
            limit: Final number of results to return
            dense_limit: Number of candidates from dense search
            sparse_limit: Number of candidates from sparse search
            use_reranker: Whether to apply cross-encoder reranking
        
        Returns:
            List of search results with scores and payloads
        """
        dense_vec = self.embed_dense(query)
        sparse_vec = self.embed_sparse(query)
        
        if len(dense_vec) == 0:
            raise ValueError("Failed to generate dense embedding")
        if sparse_vec is None or len(sparse_vec.indices) == 0:
            raise ValueError("Failed to generate sparse embedding - check if sparse model is supported in your FastEmbed version")
        
        print(f"Dense vector dim: {len(dense_vec)}")
        print(f"Sparse vector non-zero entries: {len(sparse_vec.indices)}")
        
        # Try hybrid search with query_points API
        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(query=dense_vec, using="dense", limit=dense_limit),
                    models.Prefetch(query=sparse_vec, using="sparse", limit=sparse_limit),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
            
            results = [
                {
                    "id": r.id,
                    "score": r.score,
                    "payload": r.payload,
                }
                for r in response.points
            ]
        except Exception as e:
            print(f"Hybrid search failed: {e}")
            print("Falling back to dense-only search...")
            
            # Fallback to dense-only search
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=dense_vec,
                using="dense",
                limit=limit,
                with_payload=True,
            )
            
            results = [
                {
                    "id": r.id,
                    "score": r.score,
                    "payload": r.payload,
                }
                for r in response.points
            ]
        
        # Apply reranker if requested
        if use_reranker:
            print("\nApplying cross-encoder reranking...")
            results = self.rerank(query, results)
        
        return results
    
    def search_dense_only(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Dense-only search (fallback)."""
        dense_vec = self.embed_dense(query)
        
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vec,
            using="dense",
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
    
    def rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank search results using cross-encoder.
        
        Args:
            query: Query text
            results: List of search results with payload containing 'text' field
        
        Returns:
            Reranked results with updated scores
        """
        # Extract document texts from payloads
        documents = []
        for r in results:
            payload = r.get("payload", {})
            # Try to get text from payload, fallback to title or url
            text = payload.get("text") or payload.get("title") or payload.get("url", "")
            documents.append(text)
        
        if not documents:
            return results
        
        # Rerank using cross-encoder
        scores = list(self.reranker.rerank(query, documents))
        
        # Update results with rerank scores
        reranked = []
        for r, score in zip(results, scores):
            r_copy = r.copy()
            r_copy["score"] = float(score)
            reranked.append(r_copy)
        
        # Sort by rerank score descending
        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked


def main():
    parser = argparse.ArgumentParser(description="Test Qdrant hybrid search with FastEmbed")
    parser.add_argument("--query", type=str, default="finance policy updates", help="Query text")
    parser.add_argument("--collection", type=str, default="document_index_finance", help="Qdrant collection name")
    parser.add_argument("--host", type=str, default="localhost", help="Qdrant host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--dense-model", type=str, default="BAAI/bge-base-en-v1.5", help="Dense model")
    parser.add_argument("--sparse-model", type=str, default="prithivida/Splade_PP_en_v1", help="Sparse model")
    parser.add_argument("--reranker-model", type=str, default="BAAI/bge-reranker-base", help="Reranker model")
    parser.add_argument("--limit", type=int, default=10, help="Number of results")
    parser.add_argument("--dense-only", action="store_true", help="Use dense-only search (no hybrid)")
    parser.add_argument("--use-reranker", action="store_true", help="Apply cross-encoder reranking")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Qdrant Hybrid Search Test")
    print("=" * 60)
    print(f"Collection: {args.collection}")
    print(f"Query: {args.query}")
    print(f"Dense model: {args.dense_model}")
    print(f"Sparse model: {args.sparse_model}")
    print(f"Reranker model: {args.reranker_model}")
    print(f"Use reranker: {args.use_reranker}")
    print("=" * 60)
    
    try:
        searcher = HybridSearcher(
            host=args.host,
            port=args.port,
            collection_name=args.collection,
            dense_model=args.dense_model,
            sparse_model=args.sparse_model,
            reranker_model=args.reranker_model,
        )
        
        if args.dense_only:
            print("\nRunning dense-only search...")
            results = searcher.search_dense_only(args.query, limit=args.limit)
        else:
            print("\nRunning hybrid search (dense + sparse + RRF fusion)...")
            results = searcher.search_hybrid(args.query, limit=args.limit, use_reranker=args.use_reranker)
        
        print(f"\nFound {len(results)} results:")
        print("-" * 80)
        
        for i, r in enumerate(results, 1):
            payload = r.get("payload", {})
            title = payload.get("title", "")[:50]
            url = payload.get("url", "")[:50]
            section = payload.get("section", "")[:30]
            
            print(f"{i}. [{r['score']:.4f}] {title}")
            print(f"   URL: {url}")
            print(f"   Section: {section}")
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
