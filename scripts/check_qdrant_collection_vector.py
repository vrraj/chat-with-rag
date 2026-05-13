#!/usr/bin/env python3
"""
Check if a Qdrant collection is hybrid (dense + sparse), sparse only, or dense only.
"""

import os
import sys
import argparse
from qdrant_client import QdrantClient

# Default Qdrant connection settings
DEFAULT_HOST = os.getenv("QDRANT_HOST", "localhost")
DEFAULT_PORT = int(os.getenv("QDRANT_PORT", "6333"))


def check_collection_vector_type(client: QdrantClient, collection_name: str) -> str:
    """Check the vector type configuration of a Qdrant collection."""
    try:
        collection_info = client.get_collection(collection_name=collection_name)
        
        # Check for dense vectors
        has_dense = collection_info.config.params.vectors is not None
        
        # Check for sparse vectors
        has_sparse = collection_info.config.params.sparse_vectors is not None
        
        if has_dense and has_sparse:
            return "hybrid (Both Dense and Sparse)"
        elif has_sparse:
            return "sparse only"
        else:
            return "dense only"
    except Exception as e:
        error_str = str(e)
        if "doesn't exist" in error_str or "404" in error_str:
            print(f"Collection '{collection_name}' does not exist")
            sys.exit(1)
        print(f"Error checking collection '{collection_name}': {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Check if a Qdrant collection is hybrid (dense + sparse), sparse only, or dense only."
    )
    parser.add_argument(
        "collection_name",
        help="Name of the Qdrant collection to check"
    )
    parser.add_argument(
        "--host",
        default=None,
        help=f"Qdrant host (default: {DEFAULT_HOST})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Qdrant port (default: {DEFAULT_PORT})"
    )
    
    args = parser.parse_args()
    
    host = args.host or DEFAULT_HOST
    port = args.port or DEFAULT_PORT
    
    client = QdrantClient(host=host, port=port)
    vector_type = check_collection_vector_type(client, args.collection_name)
    
    print(f"Collection '{args.collection_name}': {vector_type}")


if __name__ == "__main__":
    main()
