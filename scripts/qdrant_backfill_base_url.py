#!/usr/bin/env python3
"""
Backfill base_url and base_url_lower for points missing them by deriving from url_lower.

Behavior:
- Scans the collection via scroll.
- For points where payload lacks 'base_url' and/or 'base_url_lower' and has 'url_lower',
  sets both fields based on url_lower split at '#'. Both fields are stored in lowercase
  for consistency.
- Groups updates by base_url and performs batched set_payload calls.

Examples:
  python scripts/qdrant_backfill_base_url.py --dry-run
  python scripts/qdrant_backfill_base_url.py --batch-size 2000
  python scripts/qdrant_backfill_base_url.py --host 127.0.0.1 --port 6333

Notes:
- Uses settings for defaults (host/port/collection). Does not require the API server.
- If a point lacks url_lower, it is skipped.
-
"""

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

# Ensure project root is on sys.path so backend imports work
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointIdsList
from backend.core.config import settings


def derive_base_url(url_lower: str) -> str:
    if not url_lower:
        return url_lower
    return url_lower.split('#', 1)[0]


def backfill(
    host: str,
    port: int,
    collection: str,
    batch_size: int,
    dry_run: bool,
    max_updates: int,
) -> int:
    client = QdrantClient(host=host, port=port)

    total_scanned = 0
    total_needs_update = 0
    total_updated = 0

    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=None,
            with_payload=True,
            with_vectors=False,
            limit=batch_size,
            offset=offset,
        )

        if not points:
            break

        total_scanned += len(points)

        # Group point IDs by desired base_url_lower value
        groups: Dict[str, List] = defaultdict(list)
        for p in points:
            payload = p.payload or {}
            url_lower = payload.get('url_lower')
            if not isinstance(url_lower, str) or not url_lower:
                continue
            desired_base = derive_base_url(url_lower.strip().lower())

            has_base_url = isinstance(payload.get('base_url'), str) and bool(payload.get('base_url'))
            has_base_url_lower = isinstance(payload.get('base_url_lower'), str) and bool(payload.get('base_url_lower'))
            if has_base_url and has_base_url_lower:
                continue

            groups[desired_base].append(p.id)

        # Count how many need update in this page
        page_needs = sum(len(ids) for ids in groups.values())
        total_needs_update += page_needs

        if dry_run or page_needs == 0:
            if offset is None:
                break
            continue

        # Apply updates grouped by base_url (lower), batching ids for safety
        for base_url_lower, ids in groups.items():
            start = 0
            while start < len(ids):
                if 0 <= max_updates <= total_updated:
                    break
                end = min(start + 1000, len(ids))
                chunk_ids = ids[start:end]
                client.set_payload(
                    collection_name=collection,
                    payload={
                        'base_url': base_url_lower,
                        'base_url_lower': base_url_lower,
                    },
                    points=PointIdsList(points=chunk_ids),
                )
                total_updated += len(chunk_ids)
                start = end

        if offset is None:
            break

        if 0 <= max_updates <= total_updated:
            break

    print(f"Scanned: {total_scanned}")
    print(f"Needs update: {total_needs_update}")
    if dry_run:
        print("Dry run: no changes written")
    else:
        print(f"Updated: {total_updated}")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill base_url from url_lower where missing")
    parser.add_argument("--host", type=str, default=settings.qdrant_host, help="Qdrant host")
    parser.add_argument("--port", type=int, default=settings.qdrant_port, help="Qdrant port")
    parser.add_argument("--collection", type=str, default=settings.collection_name, help="Collection name")
    parser.add_argument("--batch-size", type=int, default=1000, help="Scroll batch size")
    parser.add_argument("--dry-run", action="store_true", help="Scan only; do not write changes")
    parser.add_argument("--max-updates", type=int, default=-1, help="Stop after updating approx this many points (-1 = unlimited)")
    args = parser.parse_args()

    raise SystemExit(
        backfill(
            host=args.host,
            port=args.port,
            collection=args.collection,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            max_updates=args.max_updates,
        )
    )


if __name__ == "__main__":
    main()
