#!/usr/bin/env python3
"""
Query Qdrant for points matching a given URL (case-insensitive via `url_lower`)
and print a simple listing of chunk_index and section name.

Usage examples:

  # Exact match against url_lower
  python scripts/qdrant_query_url.py --url https://example.com/page
  python scripts/qdrant_query_url.py --url-lower https://example.com/page

  # Include section fragments when you provide a base URL without '#...'
  # This returns rows for the base URL itself and any url_lower that starts with
  # "https://example.com/page#" (client-side filtered via scroll).
  python scripts/qdrant_query_url.py --url https://example.com/page --include-fragments

  # Limit output rows
  python scripts/qdrant_query_url.py --url https://example.com/page --limit 500

By default prints CSV: chunk_index,section,url_lower,base_url. Results are sorted by fragment
(URL hash, if present) and then by chunk_index.
Use --include-id to also include the Qdrant point id as the first column.
Use --no-header to omit the header row.
"""

import argparse
from typing import Optional, Set

import os
import sys
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

# Ensure project root is on sys.path when running as a script
# This lets `from backend...` imports work with `python scripts/xyz.py`
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.core.config import settings


def build_url_filter(url_lower_value: str) -> Filter:
    return Filter(
        must=[
            FieldCondition(
                key="url_lower",
                match=MatchValue(value=url_lower_value),
            )
        ]
    )


def run(
    url: Optional[str],
    url_lower: Optional[str],
    limit: int,
    include_id: bool,
    no_header: bool,
    include_fragments: bool,
    host: str,
    port: int,
    collection: str,
) -> int:
    if not url and not url_lower:
        raise SystemExit("Provide --url or --url-lower")

    value = (url_lower or url or "").lower()

    try:
        client = QdrantClient(host=host, port=port)
    except Exception as e:
        print(f"ERROR: Failed to create Qdrant client: {e}", file=sys.stderr)
        return 2

    flt = build_url_filter(value)

    # Collect rows that exactly match `url_lower == value`
    offset = None
    seen_ids: Set[str] = set()
    rows = []  # collect tuples for sorting/output
    remaining = None  # collect all, then apply limit after sorting
    while True:
        batch_limit = 1024
        try:
            points, offset = client.scroll(
                collection_name=collection,
                scroll_filter=flt,
                with_payload=True,
                with_vectors=False,
                limit=batch_limit,
                offset=offset,
            )
        except Exception as e:
            print(f"ERROR: Failed to scroll collection '{collection}': {e}", file=sys.stderr)
            return 3

        if not points:
            break

        for p in points:
            payload = p.payload or {}
            url_l = (payload.get("url_lower") or value).lower()
            chunk_index = payload.get("chunk_index")
            section = payload.get("section") or "Lead"
            fragment = url_l.split("#", 1)[1] if "#" in url_l else ""
            try:
                chunk_index_int = int(chunk_index) if chunk_index is not None else 10**12
            except Exception:
                chunk_index_int = 10**12
            base_url = url_l.split("#", 1)[0]
            row = {
                "id": str(p.id),
                "chunk_index": chunk_index,
                "chunk_index_int": chunk_index_int,
                "section": section,
                "fragment": fragment,
                "url_lower": url_l,
                "base_url": base_url,
            }
            rows.append(row)
            seen_ids.add(str(p.id))

        if offset is None:
            break

    # If requested and URL has no fragment, include any entries whose url_lower
    # starts with '<base>#' (client-side filter; Qdrant does not support prefix filtering).
    # We scan via scroll without a filter and stop once `remaining` is met.
    base_has_no_fragment = ("#" not in (url or url_lower or ""))
    if include_fragments and base_has_no_fragment:
        frag_prefix = value + "#"
        offset = None
        while True:
            batch_limit = 1024
            try:
                points, offset = client.scroll(
                    collection_name=collection,
                    scroll_filter=None,
                    with_payload=True,
                    with_vectors=False,
                    limit=batch_limit,
                    offset=offset,
                )
            except Exception as e:
                print(f"ERROR: Failed to scroll for fragments in '{collection}': {e}", file=sys.stderr)
                return 4

            if not points:
                break

            for p in points:
                if str(p.id) in seen_ids:
                    continue
                payload = p.payload or {}
                url_l = (payload.get("url_lower") or "").lower()
                if not url_l.startswith(frag_prefix):
                    continue

                chunk_index = payload.get("chunk_index")
                section = payload.get("section") or "Lead"
                fragment = url_l.split("#", 1)[1] if "#" in url_l else ""
                try:
                    chunk_index_int = int(chunk_index) if chunk_index is not None else 10**12
                except Exception:
                    chunk_index_int = 10**12
                base_url = url_l.split("#", 1)[0]
                rows.append(
                    {
                        "id": str(p.id),
                        "chunk_index": chunk_index,
                        "chunk_index_int": chunk_index_int,
                        "section": section,
                        "fragment": fragment,
                        "url_lower": url_l,
                        "base_url": base_url,
                    }
                )
                seen_ids.add(str(p.id))

            if offset is None:
                break

    # Sort by fragment (case-insensitive) then by chunk_index (numeric)
    rows.sort(key=lambda r: (r["fragment"].lower(), r["chunk_index_int"]))

    # Print header and rows (respect --limit)
    if not no_header:
        if include_id:
            print("id,chunk_index,section,url_lower,base_url")
        else:
            print("chunk_index,section,url_lower,base_url")

    to_print = rows if limit <= 0 else rows[:limit]
    for r in to_print:
        if include_id:
            print(f"{r['id']},{r['chunk_index']},{r['section']},{r['url_lower']},{r['base_url']}")
        else:
            print(f"{r['chunk_index']},{r['section']},{r['url_lower']},{r['base_url']}")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Query Qdrant by url_lower and list chunk_index + section")
    parser.add_argument("--url", type=str, help="URL to match (case-insensitive)")
    parser.add_argument("--url-lower", dest="url_lower", type=str, help="Direct url_lower value to match")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to print (0 = no limit)")
    parser.add_argument("--include-id", action="store_true", help="Include Qdrant point id as first column")
    parser.add_argument("--no-header", action="store_true", help="Do not print header row")
    parser.add_argument("--host", type=str, default=settings.qdrant_host, help="Qdrant host (default from settings)")
    parser.add_argument("--port", type=int, default=settings.qdrant_port, help="Qdrant port (default from settings)")
    parser.add_argument("--collection", type=str, default=settings.collection_name, help="Qdrant collection name (default from settings)")
    parser.add_argument(
        "--include-fragments",
        action="store_true",
        help=(
            "If provided URL has no '#', also include any entries whose "
            "url_lower starts with '<url>#'. Uses a client-side scan."
        ),
    )
    args = parser.parse_args()

    raise SystemExit(
        run(
            args.url,
            args.url_lower,
            args.limit,
            args.include_id,
            args.no_header,
            args.include_fragments,
            args.host,
            args.port,
            args.collection,
        )
    )


if __name__ == "__main__":
    main()
