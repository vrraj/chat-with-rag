#!/usr/bin/env python3
"""
Seed the Qdrant collection from a JSONL file of points, with zero env/config required.

Each JSONL line must look like:
  {"id": <int|str>, "vector": [..], "payload": {..}}

Defaults (chosen to match backend/core/config.py without importing it):
  - Qdrant URL:       http://localhost:6333
  - Collection name:  document_index
  - Vector size:      inferred from the first record (fallback 1536)
  - Distance:         COSINE
  - Batch size:       256

Usage:
  python scripts/seed_qdrant.py
  python scripts/seed_qdrant.py --path data/my-seed.jsonl
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Iterable, Dict, Any

# Import Qdrant configuration from backend/core/config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.core.config import settings

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse, ResponseHandlingException

# ---- Defaults (mirror backend/core/config.py) ----
DEFAULT_HOST = settings.qdrant_host
DEFAULT_PORT = settings.qdrant_port
DEFAULT_COLLECTION = settings.collection_name
DEFAULT_VECTOR_FALLBACK = getattr(settings, "vector_size", 1536)
DEFAULT_DISTANCE = models.Distance.COSINE
DEFAULT_BATCH = 256
DEFAULT_PATH = "data/docs-index-seed.jsonl"


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_client(host: str, port: int) -> QdrantClient:
    url = f"http://{host}:{port}"
    return QdrantClient(url=url)


def qdrant_is_reachable(client: QdrantClient) -> bool:
    try:
        # Lightweight call to verify connectivity
        client.get_collections()
        return True
    except Exception:
        return False


def main() -> None:
    print(f"Using Qdrant config from backend/core/config.py ({settings.qdrant_host}:{settings.qdrant_port}, collection={settings.collection_name})")
    print()
    print("This project includes sample data derived from Wikipedia.")
    print("The content is provided under the Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0).")
    print("For license details, visit: https://creativecommons.org/licenses/by-sa/4.0/")
    print()
    parser = argparse.ArgumentParser(description="Seed Qdrant from JSONL using built-in defaults")
    parser.add_argument("--path", default=DEFAULT_PATH, help=f"Path to JSONL seed file (default: {DEFAULT_PATH})")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Qdrant host (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Qdrant port (default: {DEFAULT_PORT})")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help=f"Collection name (default: {DEFAULT_COLLECTION})")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help=f"Upsert batch size (default: {DEFAULT_BATCH})")
    args = parser.parse_args()

    # Ensure seed file exists
    if not os.path.exists(args.path):
        print(f"Seed file not found: {args.path}")
        sys.exit(3)

    client = build_client(args.host, args.port)
    if not qdrant_is_reachable(client):
        print(f"Qdrant is not reachable at http://{args.host}:{args.port}")
        print("Hint: run 'make start-docker' and 'make start-qdrant' first, then retry.")
        sys.exit(2)

    # Peek first record to determine vector size if needed
    iterator = iter_jsonl(args.path)
    try:
        first = next(iterator)
    except StopIteration:
        print(f"No data found in {args.path}")
        sys.exit(3)

    vec = first.get("vector")
    if isinstance(vec, list) and len(vec) > 0:
        vector_size = len(vec)
    else:
        vector_size = DEFAULT_VECTOR_FALLBACK

    print()
    print("WARNING: This operation will delete any existing data in the collection (DOCUMENT_INDEX) before recreating it.")
    print("If you wish to keep your current data, cancel now (Ctrl+C) and back up your Qdrant storage directory.")
    print()
    resp = input("Do you want to continue with recreating the collection and deleting existing data? (y/N): ").strip().lower()
    if resp not in ("y", "yes"):
        print("Operation cancelled by user.")
        sys.exit(0)

    # Recreate collection
    url = f"http://{args.host}:{args.port}"
    print(f"Recreating collection '{args.collection}' (size={vector_size}, distance={DEFAULT_DISTANCE.value}) @ {url}")
    try:
        client.recreate_collection(
            collection_name=args.collection,
            vectors_config=models.VectorParams(size=vector_size, distance=DEFAULT_DISTANCE),
        )
    except (UnexpectedResponse, ResponseHandlingException) as e:
        print(f"Failed to recreate collection: {e}")
        sys.exit(5)

    def to_point(o: Dict[str, Any]) -> models.PointStruct:
        return models.PointStruct(id=o["id"], vector=o["vector"], payload=o.get("payload", {}))

    total = 0
    batch = [to_point(first)]
    BATCH = args.batch

    for rec in iterator:
        batch.append(to_point(rec))
        if len(batch) >= BATCH:
            client.upsert(collection_name=args.collection, points=batch)
            total += len(batch)
            print(f"Upserted {total} points...")
            batch.clear()

    if batch:
        client.upsert(collection_name=args.collection, points=batch)
        total += len(batch)

    print(f"Seeded {total} point(s) into '{args.collection}' from {args.path}")


if __name__ == "__main__":
    main()
