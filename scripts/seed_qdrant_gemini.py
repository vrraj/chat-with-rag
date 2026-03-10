#!/usr/bin/env python3
"""
Seed the Qdrant collection from a JSONL file of points using Gemini embeddings.

Each JSONL line must look like:
  {"id": <int|str>, "vector": [..], "payload": {..}}

This script is specifically for the Gemini collection (document_index_gemini).
It temporarily switches to the "oceans" domain to get the correct collection name
and embedding model configuration.

Defaults (chosen for Gemini domain):
  - Qdrant URL:       http://localhost:6333
  - Collection name:  document_index_gemini
  - Vector size:      inferred from the first record (fallback 1536)
  - Distance:         COSINE
  - Batch size:       256

Usage:
  python scripts/seed_qdrant_gemini.py
  python scripts/seed_qdrant_gemini.py --path data/docs-index-seed-gemini.jsonl
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Iterable, Dict, Any

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

# Import only Qdrant configuration to avoid API key validation
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from backend.core.config import settings
    DEFAULT_HOST = settings.qdrant_host
    DEFAULT_PORT = settings.qdrant_port
    DEFAULT_COLLECTION = "document_index_gemini"  # Override for Gemini collection
    DEFAULT_VECTOR_FALLBACK = getattr(settings, "vector_size", 1536)
except (ValueError, ImportError) as e:
    # Fallback defaults if config validation fails
    print(f"Warning: Could not load config ({e}), using defaults")
    DEFAULT_HOST = "localhost"
    DEFAULT_PORT = 6333
    DEFAULT_COLLECTION = "document_index_gemini"
    DEFAULT_VECTOR_FALLBACK = 1536

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse, ResponseHandlingException

# ---- Gemini-specific configuration ----
DEFAULT_DISTANCE = models.Distance.COSINE
DEFAULT_BATCH = 256
DEFAULT_PATH = "data/docs-index-seed-gemini.jsonl"

# Try to get vector size from settings if available
try:
    if 'settings' in locals():
        original_domain = settings.active_domain
        settings.active_domain = "oceans"
        DEFAULT_VECTOR_FALLBACK = getattr(settings, "vector_size", 1536)
        settings.active_domain = original_domain
    else:
        DEFAULT_VECTOR_FALLBACK = 1536
except:
    DEFAULT_VECTOR_FALLBACK = 1536


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
    print(f"Using Gemini domain configuration (oceans) -> collection: {DEFAULT_COLLECTION}")
    print(f"Qdrant: {DEFAULT_HOST}:{DEFAULT_PORT}")
    print()
    print("This project includes sample data derived from Wikipedia.")
    print("The content is provided under the Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0).")
    print("For license details, visit: https://creativecommons.org/licenses/by-sa/4.0/")
    print()
    print("This script seeds the Gemini embedding collection (document_index_gemini).")
    print("Make sure you have generated the seed data using Gemini embeddings first.")
    print()
    parser = argparse.ArgumentParser(description="Seed Qdrant Gemini collection from JSONL")
    parser.add_argument("--path", default=DEFAULT_PATH, help=f"Path to JSONL seed file (default: {DEFAULT_PATH})")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Qdrant host (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Qdrant port (default: {DEFAULT_PORT})")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help=f"Collection name (default: {DEFAULT_COLLECTION})")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help=f"Upsert batch size (default: {DEFAULT_BATCH})")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    # Ensure seed file exists
    if not os.path.exists(args.path):
        print(f"Seed file not found: {args.path}")
        print(f"Expected path: {args.path}")
        print("Make sure you have generated the Gemini seed data first.")
        sys.exit(3)

    client = build_client(args.host, args.port)
    if not qdrant_is_reachable(client):
        print(f"Qdrant is not reachable at http://{args.host}:{args.port}")
        print("Hint: run 'make start-docker' and 'make start-qdrant' first, then retry.")
        sys.exit(2)

    # Check if collection exists
    collection_exists = False
    try:
        collection_info = client.get_collection(collection_name=args.collection)
        collection_exists = True
        print(f"Found existing collection: {args.collection} (points: {collection_info.points_count})")
    except Exception as e:
        if "not found" not in str(e).lower():
            print(f"Warning: Error checking collection: {e}")
        print(f"Collection '{args.collection}' does not exist, will create it")

    # Only prompt if collection exists and --force is not set
    if collection_exists and not args.force:
        print("\nWARNING: This operation will delete all data in the Gemini collection.")
        print("If you wish to keep your current data, cancel now (Ctrl+C) and back up your Qdrant storage directory.")
        print()
        resp = input("Do you want to continue with recreating the collection and deleting existing data? (y/N): ").strip().lower()
        if resp not in ("y", "yes"):
            print("Operation cancelled by user.")
            sys.exit(0)

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

    # Recreate collection
    url = f"http://{args.host}:{args.port}"
    action = "Recreating" if collection_exists else "Creating"
    print(f"{action} Gemini collection '{args.collection}' (size={vector_size}, distance={DEFAULT_DISTANCE.value}) @ {url}")
    
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

    print(f"Seeded {total} point(s) into Gemini collection '{args.collection}' from {args.path}")
    print()
    print("Gemini collection is ready!")
    print("You can now switch to the 'oceans' domain in the configuration to use Gemini embeddings.")


if __name__ == "__main__":
    main()
