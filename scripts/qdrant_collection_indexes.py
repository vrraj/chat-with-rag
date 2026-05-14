"""
Create payload indexes and update vector index configuration for a Qdrant collection.

This script performs two operations:
1. Payload Index Creation: Creates KEYWORD payload indexes for fields commonly used in filters
2. Vector Index Configuration: Updates HNSW parameters and on_disk settings for vector indexes

Default payload fields indexed:
  - url_lower
  - base_url_lower
  - source
  - domain
  - doc_type

Default vector HNSW parameters:
  - m: 16
  - ef_construct: 100
  - full_scan_threshold: 1000
  - on_disk: True

Usage Examples:

  # Basic: create payload indexes and update vector config for default collection
  python scripts/qdrant_collection_indexes.py

  # Specify collection name
  python scripts/qdrant_collection_indexes.py --collection document_index_finance

  # Skip payload index creation, only update vector config
  python scripts/qdrant_collection_indexes.py --collection document_index --no-payload-indexes

  # Skip vector config update, only create payload indexes
  python scripts/qdrant_collection_indexes.py --collection document_index --no-vector-update

  # Update named vector (e.g., for hybrid collections with 'dense' and 'sparse' vectors)
  python scripts/qdrant_collection_indexes.py --collection document_index_finance --vector-name dense
  python scripts/qdrant_collection_indexes.py --collection document_index_finance --vector-name sparse

  # For hybrid collections, update both vectors (skip payload indexes on subsequent runs)
  python scripts/qdrant_collection_indexes.py --collection document_index_finance --vector-name dense
  python scripts/qdrant_collection_indexes.py --collection document_index_finance --vector-name sparse --no-payload-indexes

  # Customize HNSW parameters for better recall/performance
  python scripts/qdrant_collection_indexes.py --collection document_index --m 32 --ef-construct 200 --full-scan-threshold 2000

  # Store vector index in memory instead of on disk
  python scripts/qdrant_collection_indexes.py --collection document_index --no-on-disk

  # Index additional custom payload fields
  python scripts/qdrant_collection_indexes.py --collection document_index --fields url_lower base_url_lower custom_field

  # Combine multiple options
  python scripts/qdrant_collection_indexes.py --collection document_index --vector-name dense --m 24 --ef-construct 150

Notes:
  - Payload indexes are idempotent: if an index already exists, the script prints a warning and continues.
  - Vector config updates apply to existing collections; use with caution on production collections.
  - For hybrid collections with named vectors, use --vector-name to specify which vector to update.
  - The script uses Qdrant host/port from QDRANT_HOST/QDRANT_PORT environment variables (defaults: localhost:6333).
  - No API keys are required for this script — it only connects to Qdrant.
"""

import argparse
import os
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import PayloadSchemaType, VectorParamsDiff, HnswConfigDiff


def ensure_payload_indexes(
    client: QdrantClient,
    collection: str,
    fields: List[str],
) -> None:
    """Create KEYWORD payload indexes for the specified fields."""
    for field in fields:
        try:
            print(f"[INFO] Creating KEYWORD index for '{field}' on '{collection}'…")
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
            print(f"[OK]   Index created: {field}")
        except Exception as e:
            # Qdrant returns error if index exists; print and continue
            print(f"[WARN] Could not create index for '{field}': {e}")


def update_vector_index_config(
    client: QdrantClient,
    collection: str,
    vector_name: Optional[str] = None,
    m: int = 16,
    ef_construct: int = 100,
    full_scan_threshold: int = 1000,
    on_disk: bool = True,
) -> None:
    """Update vector index HNSW configuration and on_disk setting."""
    try:
        print(f"[INFO] Updating vector index config for '{collection}'" +
              (f" (vector: {vector_name})" if vector_name else "") + "…")

        # Handle sparse vectors separately (they use sparse_vectors_config)
        if vector_name == "sparse":
            sparse_config = {
                "index": {
                    "full_scan_threshold": full_scan_threshold,
                    "on_disk": on_disk,
                }
            }
            client.update_collection(
                collection_name=collection,
                sparse_vectors_config={"sparse": sparse_config},
            )
            print(f"[OK]   Sparse vector index config updated for 'sparse'")
            return

        # Handle dense vectors (named or unnamed)
        vector_config = VectorParamsDiff(
            hnsw_config=HnswConfigDiff(
                m=m,
                ef_construct=ef_construct,
                full_scan_threshold=full_scan_threshold,
            ),
            on_disk=on_disk,
        )

        # Qdrant update_collection expects a dict for vectors_config
        # Use empty string for unnamed vector, or the actual name for named vectors
        key = vector_name or ""
        client.update_collection(
            collection_name=collection,
            vectors_config={key: vector_config},
        )

        print(f"[OK]   Vector index config updated" +
              (f" for '{vector_name}'" if vector_name else " for unnamed vector"))
    except Exception as e:
        print(f"[WARN] Could not update vector index config: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create payload indexes and update vector index config for a Qdrant collection"
    )
    parser.add_argument(
        "--host",
        default=os.getenv("QDRANT_HOST", "localhost"),
        help="Qdrant host (default: localhost, or QDRANT_HOST env var)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("QDRANT_PORT", "6333")),
        help="Qdrant port (default: 6333, or QDRANT_PORT env var)",
    )
    parser.add_argument(
        "--collection",
        required=True,
        help="Collection name (required)",
    )
    parser.add_argument(
        "--fields",
        nargs="*",
        default=["url_lower", "base_url_lower", "source", "domain", "doc_type"],
        help="Field names to index as KEYWORD",
    )
    parser.add_argument(
        "--no-payload-indexes",
        action="store_true",
        help="Skip payload index creation",
    )
    parser.add_argument(
        "--no-vector-update",
        action="store_true",
        help="Skip vector index config update",
    )
    parser.add_argument(
        "--vector-name",
        default=None,
        help="Named vector to update (e.g., 'dense', 'sparse'). If omitted, updates default unnamed vector.",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=16,
        help="HNSW m parameter (default: 16)",
    )
    parser.add_argument(
        "--ef-construct",
        type=int,
        default=100,
        help="HNSW ef_construct parameter (default: 100)",
    )
    parser.add_argument(
        "--full-scan-threshold",
        type=int,
        default=1000,
        help="HNSW full_scan_threshold parameter (default: 1000)",
    )
    parser.add_argument(
        "--on-disk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store vector index on disk (default: True)",
    )
    args = parser.parse_args()

    client = QdrantClient(host=args.host, port=args.port)

    if not args.no_payload_indexes:
        ensure_payload_indexes(client, args.collection, args.fields)

    if not args.no_vector_update:
        update_vector_index_config(
            client,
            args.collection,
            vector_name=args.vector_name,
            m=args.m,
            ef_construct=args.ef_construct,
            full_scan_threshold=args.full_scan_threshold,
            on_disk=args.on_disk,
        )

    print("[DONE] Collection index operations completed.")


if __name__ == "__main__":
    main()
