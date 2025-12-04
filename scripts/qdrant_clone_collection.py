"""
Clone points from one Qdrant collection to another, preserving IDs, vectors, and payloads.

Usage examples:
  python scripts/qdrant_clone_collection.py --src website_collection --dst document_index_v2
  python scripts/qdrant_clone_collection.py --src website_collection --dst document_index_v2 --batch-size 2048
  python scripts/qdrant_clone_collection.py --src website_collection --dst document_index_v2 --verify-only

Notes:
- Creates destination collection with the same vector size and distance as source.
- Does NOT modify aliases. Use alias actions separately to flip traffic.
- Safe by default: will not overwrite an existing destination unless --recreate is given.
"""

import argparse
import sys
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import models
from qdrant_client.models import Distance, VectorParams, PointStruct

from backend.core.config import settings


def get_distance_enum(name: str) -> Distance:
    name = (name or "").lower()
    if name == "cosine":
        return Distance.COSINE
    if name == "dot":
        return Distance.DOT
    if name in ("euclid", "euclidean"):
        return Distance.EUCLID
    # Fallback to COSINE
    return Distance.COSINE


def ensure_destination_collection(client: QdrantClient, src: str, dst: str, recreate: bool = False) -> None:
    try:
        # If destination exists and not recreating, keep it and return
        try:
            client.get_collection(dst)
            if not recreate:
                print(f"[INFO] Destination collection '{dst}' already exists; keeping existing.")
                return
        except Exception:
            # Does not exist; proceed to create
            pass

        # Inspect source collection to copy vector specs
        src_info = client.get_collection(src)
        vectors = src_info.config.params.vectors
        # vectors can be map or single config; here we assume single vector space
        if isinstance(vectors, dict) and "size" in vectors:
            size = int(vectors["size"])  # type: ignore
            distance = str(vectors.get("distance") or "Cosine")
        else:
            # Pydantic model access
            size = int(getattr(vectors, "size"))
            distance = str(getattr(vectors, "distance") or "Cosine")

        dist_enum = get_distance_enum(distance)

        print(f"[INFO] Creating destination '{dst}' with size={size}, distance={dist_enum.value}")
        client.recreate_collection(
            collection_name=dst,
            vectors_config=VectorParams(size=size, distance=dist_enum),
            on_disk_payload=getattr(src_info.config.params, "on_disk_payload", True),
        )
    except Exception as e:
        print(f"[ERROR] Failed to prepare destination '{dst}': {e}")
        sys.exit(1)


def count_points(client: QdrantClient, name: str) -> int:
    # Use scroll to count reliably
    total = 0
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=name,
            with_payload=False,
            with_vectors=False,
            limit=1000,
            offset=offset,
        )
        total += len(points)
        if not points or offset is None:
            break
    return total


def clone_points(client: QdrantClient, src: str, dst: str, batch_size: int = 1000, max_points: Optional[int] = None, skip_first: int = 0) -> int:
    total = 0
    skipped = 0
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=src,
            with_payload=True,
            with_vectors=True,
            limit=batch_size,
            offset=offset,
        )
        if not points:
            break
        # Skip leading points if requested
        if skip_first and skipped < skip_first:
            need_to_skip = skip_first - skipped
            if need_to_skip >= len(points):
                skipped += len(points)
                continue
            else:
                points = points[need_to_skip:]
                skipped = skip_first
        # Respect max_points limit if provided
        if max_points is not None and max_points >= 0:
            remaining = max_points - total
            if remaining <= 0:
                break
            if len(points) > remaining:
                points = points[:remaining]

        payloads = [p.payload for p in points]
        vectors = [p.vector for p in points]
        ids = [p.id for p in points]
        batch = [
            PointStruct(id=pid, vector=vec, payload=pl)
            for pid, vec, pl in zip(ids, vectors, payloads)
        ]
        client.upsert(collection_name=dst, points=batch)
        total += len(batch)
        print(f"[DEBUG] Copied {total} points…", end="\r", flush=True)
    print()
    return total


def main():
    parser = argparse.ArgumentParser(description="Clone Qdrant collection (IDs, vectors, payloads)")
    parser.add_argument("--src", required=True, help="Source collection name")
    parser.add_argument("--dst", required=True, help="Destination collection name")
    parser.add_argument("--host", default=settings.qdrant_host, help="Qdrant host")
    parser.add_argument("--port", type=int, default=settings.qdrant_port, help="Qdrant port")
    parser.add_argument("--batch-size", type=int, default=1000, help="Scroll/Upsert batch size")
    parser.add_argument("--recreate", action="store_true", help="Recreate destination collection if it exists")
    parser.add_argument("--max-points", type=int, default=None, help="Copy at most this many points (for testing)")
    parser.add_argument("--skip-first", type=int, default=0, help="Skip this many source points before copying")
    parser.add_argument("--verify-only", action="store_true", help="Only print src/dst counts and exit")

    args = parser.parse_args()

    client = QdrantClient(host=args.host, port=args.port)

    try:
        src_info = client.get_collection(args.src)
    except Exception as e:
        print(f"[ERROR] Source collection '{args.src}' not found: {e}")
        sys.exit(1)

    src_count = count_points(client, args.src)
    print(f"[INFO] Source '{args.src}' points: {src_count}")

    if args.verify_only:
        try:
            dst_count = count_points(client, args.dst)
            print(f"[INFO] Dest '{args.dst}' points: {dst_count}")
        except Exception as e:
            print(f"[INFO] Dest '{args.dst}' not found or error: {e}")
        return

    ensure_destination_collection(client, args.src, args.dst, recreate=args.recreate)

    copied = clone_points(client, args.src, args.dst, batch_size=args.batch_size, max_points=args.max_points, skip_first=args.skip_first)
    dst_count = count_points(client, args.dst)
    print(f"[SUCCESS] Copied {copied} points. Dest count now {dst_count}.")
    if dst_count != src_count:
        print("[WARN] Counts differ between source and destination. Investigate before flipping alias.")


if __name__ == "__main__":
    main()
