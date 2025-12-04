"""
Create payload indexes for common filter fields on a Qdrant collection.

By default, creates KEYWORD indexes for:
  - url_lower
  - base_url_lower

Usage:
  python scripts/qdrant_create_payload_indexes.py --collection document_index
  python scripts/qdrant_create_payload_indexes.py --collection document_index --fields url_lower base_url_lower other_field
"""

import argparse
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http.models import PayloadSchemaType
from backend.core.config import settings


def ensure_indexes(client: QdrantClient, collection: str, fields: List[str]) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Create payload indexes on a Qdrant collection")
    parser.add_argument("--host", default=settings.qdrant_host, help="Qdrant host")
    parser.add_argument("--port", type=int, default=settings.qdrant_port, help="Qdrant port")
    parser.add_argument("--collection", default=settings.collection_name, help="Collection name")
    parser.add_argument(
        "--fields",
        nargs="*",
        default=["url_lower", "base_url_lower"],
        help="Field names to index as KEYWORD",
    )
    args = parser.parse_args()

    client = QdrantClient(host=args.host, port=args.port)
    ensure_indexes(client, args.collection, args.fields)


if __name__ == "__main__":
    main()

