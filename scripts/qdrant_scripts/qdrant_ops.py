#!/usr/bin/env python3
"""
NOTE: This script should be run within the project's Python virtual environment (venv)
to ensure all dependencies (e.g., qdrant-client) are available.
Activate venv using:
  . venv/bin/activate

qdrant_ops.py — Utility operations for a local Qdrant instance.

Functions (selected by CLI subcommands):
  1) Retrieve points by payload field/value and display `point_id` + `url_lower`.
  1b) Retrieve all points (optionally with a limit).
  2) Display payload field names (using the first point as reference).
  2b) List unique URLs with titles (30-char max)
  3) Count chunks for a specific base URL
  4) Delete points either by (a) payload field/value or (b) explicit point id(s).

Defaults try to import host/port/collection from backend/core/config.py, with
fallbacks to localhost:6333 and collection "document_index" if import fails.

Examples:
  # 1) Retrieve by payload field/value
  python scripts/qdrant_scripts/qdrant_ops.py get --field source --value wikipedia --limit 50

  # 1b) Retrieve all points (optionally with a limit)
  python scripts/qdrant_scripts/qdrant_ops.py get --limit 100

  # 2) List payload field names (from first point)
  python scripts/qdrant_scripts/qdrant_ops.py list-fields

  # 2b) List unique URLs with titles (30-char max)
  python scripts/qdrant_scripts/qdrant_ops.py list-titles --limit 100

  # 3) Count chunks for a specific base URL
  python scripts/qdrant_scripts/qdrant_ops.py count-chunks --base-url "https://example.com"

  # 4a) Delete by payload field/value (asks for confirmation)
  python scripts/qdrant_scripts/qdrant_ops.py delete --field doc_id --value 123

  # 4b) Delete by payload field/value (skip confirmation)
  python scripts/qdrant_scripts/qdrant_ops.py delete --field doc_id --value 123 --yes

  # 4c) Delete by explicit point IDs (skip confirmation)
  python scripts/qdrant_scripts/qdrant_ops.py delete --ids 10 11 12 --yes

  # 4d) Delete all points from a given source (interactive confirmation)
  python scripts/qdrant_scripts/qdrant_ops.py delete --field source --value wikipedia
"""
from __future__ import annotations
import argparse
import os
import sys
import json
from typing import Iterable, List, Optional

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from backend.core.config import settings  # type: ignore
    DEFAULT_HOST = settings.qdrant_host
    DEFAULT_PORT = settings.qdrant_port
    DEFAULT_COLLECTION = settings.collection_name
except Exception:
    DEFAULT_HOST = "localhost"
    DEFAULT_PORT = 6333
    DEFAULT_COLLECTION = "document_index"

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    Filter,
    FieldCondition,
    MatchValue,
)
from qdrant_client.http import models as rest


def build_client(host: str, port: int) -> QdrantClient:
    return QdrantClient(url=f"http://{host}:{port}")


def scroll_points(
    client: QdrantClient,
    collection: str,
    *,
    flt: Optional[Filter] = None,
    with_vectors: bool = False,
    with_payload: bool = True,
    page_size: int = 256,
    max_points: Optional[int] = None,
):
    """Generator yielding points with an optional filter."""
    offset = None
    yielded = 0
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            with_vectors=with_vectors,
            with_payload=with_payload,
            offset=offset,
            limit=page_size,
            scroll_filter=flt,
        )
        if not points:
            break
        for p in points:
            yield p
            yielded += 1
            if max_points is not None and yielded >= max_points:
                return
        if offset is None:
            break


def cmd_get(args) -> int:
    client = build_client(args.host, args.port)

    # Build optional filter only if both field and value are provided
    flt = None
    if args.field and args.value is not None:
        cond = FieldCondition(key=args.field, match=MatchValue(value=args.value))
        flt = Filter(must=[cond])

    count = 0
    for p in scroll_points(
        client,
        args.collection,
        flt=flt,
        page_size=args.page_size,
        max_points=args.limit,
        with_vectors=False,
        with_payload=True,
    ):
        title = ""
        base_url_lower = ""
        if p.payload and isinstance(p.payload, dict):
            title = p.payload.get("title") or ""
            base_url_lower = p.payload.get("base_url_lower") or ""
        # Truncate or pad title to 25 characters
        title_display = (title[:25]).ljust(25)
        print(f"id={p.id}\ttitle={title_display}\tbase_url_lower={base_url_lower}")
        count += 1

    if count == 0:
        print("No points matched the given payload filter.")
    else:
        print(f"Total: {count}")
    return 0


def cmd_list_fields(args) -> int:
    client = build_client(args.host, args.port)
    # Get a single point (any) to introspect payload keys
    pts, _ = client.scroll(
        collection_name=args.collection,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not pts:
        print("Collection is empty or not found.")
        return 0

    payload = pts[0].payload or {}
    if not isinstance(payload, dict) or not payload:
        print("No payload on the first point.")
        return 0

    keys = sorted(payload.keys())
    print("Payload fields (from first point):")
    for k in keys:
        print(f"- {k}")
    return 0



def cmd_list_titles(args) -> int:
    client = build_client(args.host, args.port)
    seen_urls = set()
    unique_points = []

    for p in scroll_points(
        client,
        args.collection,
        page_size=args.page_size,
        max_points=args.limit,   # reuse --limit if provided
        with_vectors=False,
        with_payload=True,
    ):
        if not p.payload or not isinstance(p.payload, dict):
            continue
        title = (p.payload.get("title") or "")
        base_url_lower = (p.payload.get("base_url_lower") or "")
        if not base_url_lower or base_url_lower in seen_urls:
            continue
        seen_urls.add(base_url_lower)
        title_display = (title[:30]).ljust(30)  # truncate to 30 and right-pad
        unique_points.append((title_display, base_url_lower))

    if not unique_points:
        print("No points found in the collection.")
        return 0

    print(f"{'TITLE':30}\tURL")
    print("-" * 80)
    for title_display, base_url_lower in unique_points:
        print(f"{title_display}\t{base_url_lower}")
    print(f"\nTotal unique URLs: {len(unique_points)}")
    return 0


# New command: vector-dims
def cmd_vector_dims(args) -> int:
    """Print vector configuration for the given collection.

    Outputs whether the collection uses named vectors, and prints
    dimension(s) + distance metric(s) when available.

    Supports both single-vector collections and named-vectors collections.
    """
    client = build_client(args.host, args.port)

    try:
        info = client.get_collection(args.collection)
    except Exception as e:
        print(f"Error retrieving collection '{args.collection}': {e}")
        return 1

    named_vectors = False
    vectors_cfg = None

    try:
        cfg = getattr(info, "config", None)
        if cfg is not None:
            params = getattr(cfg, "params", cfg)
            vectors_cfg = getattr(params, "vectors", None)
            named_vectors = isinstance(vectors_cfg, dict)
    except Exception:
        vectors_cfg = None
        named_vectors = False

    # Collect per-vector config: name -> {size, distance}
    vecs: dict[str, dict[str, object]] = {}

    try:
        # Single unnamed vector
        if vectors_cfg is not None and hasattr(vectors_cfg, "size"):
            vecs["default"] = {
                "size": getattr(vectors_cfg, "size", "unknown"),
                "distance": getattr(vectors_cfg, "distance", "unknown"),
            }

        # Named vectors (dict)
        elif isinstance(vectors_cfg, dict):
            for name, vcfg in vectors_cfg.items():
                vecs[str(name)] = {
                    "size": getattr(vcfg, "size", "unknown"),
                    "distance": getattr(vcfg, "distance", "unknown"),
                }

        # Best-effort fallback for other shapes
        else:
            size = getattr(vectors_cfg, "size", None)
            if size is not None:
                vecs["default"] = {
                    "size": size,
                    "distance": getattr(vectors_cfg, "distance", "unknown"),
                }
    except Exception:
        vecs = {}

    print(f"Collection: {args.collection}")
    print(f"Named vectors: {'yes' if named_vectors else 'no'}")

    if vecs:
        print("Vector config:")
        for name in sorted(vecs.keys()):
            size = vecs[name].get("size", "unknown")
            dist = vecs[name].get("distance", "unknown")
            print(f"- {name}: size={size}, distance={dist}")
    else:
        print("Vector config: unknown")

    return 0


def cmd_export(args) -> int:
    """
    Export all points from the given collection to a JSONL file.

    Each line in the output file will be:
      {"id": <int|str>, "vector": [..], "payload": {..}}

    The collection defaults to DEFAULT_COLLECTION (from backend/core/config.py
    when available) but can be overridden via --collection.

    The output file is placed under the project-level "data" directory. If no
    filename is provided, "docs-index-seed.jsonl" is used by default.

    Example usage:
      # Export default collection to default file (docs-index-seed.jsonl)
      python scripts/qdrant_scripts/qdrant_ops.py --export

      # Export specific collection to default file
      python scripts/qdrant_scripts/qdrant_ops.py --export --collection my_collection

      # Export specific collection to custom filename (saved under ./data)
      python scripts/qdrant_scripts/qdrant_ops.py --export --collection my_collection -f my-export.jsonl
    """
    client = build_client(args.host, args.port)

    # Resolve data directory relative to project root: ../../data from this script
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "data"))
    os.makedirs(data_dir, exist_ok=True)

    # Default filename if none provided
    filename = args.file or "docs-index-seed.jsonl"
    output_path = os.path.join(data_dir, filename)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for p in scroll_points(
            client,
            args.collection,
            with_vectors=True,
            with_payload=True,
        ):
            # Qdrant's Record.vector can be a list or a dict of named vectors.
            vec = getattr(p, "vector", None)
            if isinstance(vec, dict):
                # Take the first vector if multiple are present
                vec = next(iter(vec.values())) if vec else []
            if vec is None:
                vec = []

            payload = p.payload or {}

            record = {
                "id": p.id,
                "vector": vec,
                "payload": payload,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"Exported {count} point(s) from collection '{args.collection}' to {output_path}")
    return 0


def cmd_truncate(args) -> int:
    """
    Delete all points in a collection while preserving the collection configuration
    (distance, vector size, payload schema, etc.).

    The collection name is taken from --collection (defaulting to backend/core/config.py).
    This command is interactive and will ask you to re-type the collection name
    before proceeding.

    Example usage:
      # Truncate default collection (from backend/core/config.py)
      python scripts/qdrant_scripts/qdrant_ops.py --truncate

      # Truncate a specific collection
      python scripts/qdrant_scripts/qdrant_ops.py --truncate --collection my_collection

      # Expected prompt:
      #   All <N> point(s) will be deleted from the '<collection>' collection.
      #   This is a NON-reversible operation.
      #   To proceed, enter the collection name:
    """
    client = build_client(args.host, args.port)

    # Get collection info so we can show points, vectors, and dimensions
    try:
        info = client.get_collection(args.collection)
    except Exception as e:
        print(f"Error retrieving collection '{args.collection}': {e}")
        return 1

    points_count = getattr(info, "points_count", None)
    vectors_count = getattr(info, "vectors_count", None)

    # Try to infer vector dimension robustly
    dim = "unknown"
    try:
        cfg = getattr(info, "config", None)
        if cfg is not None:
            # qdrant-client 1.x usually exposes params.vectors
            params = getattr(cfg, "params", cfg)
            vectors = getattr(params, "vectors", None)
            if vectors is not None:
                if hasattr(vectors, "size"):
                    dim = vectors.size
                elif isinstance(vectors, dict) and vectors:
                    first = next(iter(vectors.values()))
                    dim = getattr(first, "size", "unknown")
    except Exception:
        dim = "unknown"

    print(f"Collection: {args.collection}")
    if points_count is not None:
        print(f"Points: {points_count}")
    if vectors_count is not None:
        print(f"Vectors: {vectors_count}")
    print(f"Vector dimension: {dim}")

    if not points_count:
        print("Collection has no points to delete.")
        return 0

    print(
        f"\nAll {points_count} point(s) will be deleted from the '{args.collection}' collection."
    )
    print("This is a NON-reversible operation.")
    confirm = input("To proceed, enter the collection name: ").strip()

    if confirm != args.collection:
        print("Collection name did not match. Aborting without deleting anything.")
        return 0

    # Delete all points but keep collection config: use All() selector
    client.delete(
        collection_name=args.collection,
        points_selector=rest.All(),
    )

    print(f"Truncated collection '{args.collection}'; deleted {points_count} point(s).")
    return 0


def cmd_delete(args) -> int:
    client = build_client(args.host, args.port)

    # Delete by explicit point ids
    if args.ids:
        targets: List[int] = []
        for s in args.ids:
            try:
                targets.append(int(s))
            except ValueError:
                print(f"Skipping non-integer id: {s}")
        if not targets:
            print("No valid point ids provided.")
            return 1
        if not args.yes:
            print(f"About to delete {len(targets)} point(s) by id from '{args.collection}'.")
            resp = input("Proceed? (y/N): ").strip().lower()
            if resp not in ("y", "yes"):
                print("Cancelled.")
                return 0
        client.delete(collection_name=args.collection, points_selector=targets)
        print(f"Deleted {len(targets)} point(s) by id.")
        return 0

    # Delete by payload field/value
    if args.field and args.value is not None:
        cond = FieldCondition(key=args.field, match=MatchValue(value=args.value))
        flt = Filter(must=[cond])

        # Optional pre-count for user feedback
        cnt = client.count(collection_name=args.collection, count_filter=flt, exact=True).count
        if cnt == 0:
            print("No points match the provided payload filter; nothing to delete.")
            return 0
        if not args.yes:
            print(f"About to delete {cnt} point(s) from '{args.collection}' where {args.field} == {args.value!r}.")
            resp = input("Proceed? (y/N): ").strip().lower()
            if resp not in ("y", "yes"):
                print("Cancelled.")
                return 0

        client.delete(collection_name=args.collection, points_selector=flt)
        print(f"Deleted {cnt} point(s) by payload filter.")
        return 0

    print("Provide either --ids or both --field and --value for deletion.")
    return 1


def cmd_list_collections(args) -> int:
    """List all collections in the Qdrant instance."""
    client = build_client(args.host, args.port)
    
    try:
        collections_info = client.get_collections()
        collections = collections_info.collections
        
        if not collections:
            print("No collections found.")
            return 0
        
        print("Collections:")
        for col in collections:
            print(f"- {col.name}")
        print(f"\nTotal: {len(collections)} collection(s)")
        return 0
    except Exception as e:
        print(f"Error listing collections: {e}")
        return 1


def count_chunks_by_base_url(args) -> int:
    """Count and display the number of chunks for a specific base URL."""
    client = build_client(args.host, args.port)
    collection = args.collection
    base_url = args.base_url
    
    if not base_url:
        print("Error: --base-url is required")
        return 1
    
    base_url_lower = base_url.lower()
    
    # Create filter for base_url_lower
    filter_condition = Filter(
        must=[
            FieldCondition(
                key="base_url_lower",
                match=MatchValue(value=base_url_lower),
            )
        ]
    )
    
    # Count points matching the filter
    count = 0
    for _ in scroll_points(client, collection, flt=filter_condition, with_payload=False):
        count += 1
    
    print(f"Found {count} chunks for base URL: {base_url}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qdrant operations utility")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Qdrant host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Qdrant port")
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION, 
                       help=f"Collection name (default: {DEFAULT_COLLECTION})")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Get command
    get_parser = subparsers.add_parser("get", help="Get points by filter")
    get_parser.add_argument("--field", type=str, help="Payload field to filter on")
    get_parser.add_argument("--value", type=str, help="Value to match in the specified field")
    get_parser.add_argument("--limit", type=int, default=10, help="Maximum number of points to return")
    get_parser.add_argument(
        "--page-size",
        type=int,
        default=256,
        help="Scroll page size for fetching points (default: 256)",
    )
    get_parser.set_defaults(func=cmd_get)
    
    # List fields command
    list_fields_parser = subparsers.add_parser("list-fields", help="List all payload field names")
    list_fields_parser.set_defaults(func=cmd_list_fields)
    
    # List titles command
    list_titles_parser = subparsers.add_parser("list-titles", help="List unique URLs with titles")
    list_titles_parser.add_argument("--limit", type=int, default=50, help="Maximum number of URLs to return")
    list_titles_parser.add_argument(
        "--page-size",
        type=int,
        default=256,
        help="Scroll page size for fetching points (default: 256)",
    )
    list_titles_parser.set_defaults(func=cmd_list_titles)

    # Vector dimensions command
    dims_parser = subparsers.add_parser(
        "vector-dims",
        help="Show vector config (dims + distance) for the collection (supports named vectors)",
    )
    dims_parser.set_defaults(func=cmd_vector_dims)
    
    # Count chunks by base_url command
    count_parser = subparsers.add_parser("count-chunks", help="Count chunks by base URL (case-insensitive)")
    count_parser.add_argument(
        "--base-url",
        type=str,
        required=True,
        dest="base_url",  # Store in args.base_url for backward compatibility
        help="Base URL to count chunks for (will be converted to lowercase)",
    )
    count_parser.set_defaults(func=count_chunks_by_base_url)
    
    # List collections command
    list_collections_parser = subparsers.add_parser("list-collections", help="List all collections in Qdrant")
    list_collections_parser.set_defaults(func=cmd_list_collections)
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export collection to JSONL file")
    export_parser.add_argument("-f", "--file", type=str, default="docs-index-seed.jsonl",
                              help="Output filename (saved in data/ directory)")
    export_parser.set_defaults(func=cmd_export)
    
    # Truncate command
    truncate_parser = subparsers.add_parser("truncate", help="Delete all points in a collection")
    truncate_parser.set_defaults(func=cmd_truncate)
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete points by filter or ID")
    delete_parser.add_argument("--field", type=str, help="Payload field to filter on")
    delete_parser.add_argument("--value", type=str, help="Value to match in the specified field")
    delete_parser.add_argument("--ids", nargs="+", help="List of point IDs to delete")
    delete_parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    delete_parser.set_defaults(func=cmd_delete)
    
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, 'func'):
        # No command provided, show help
        parser.print_help()
        return 1

    try:
        return args.func(args)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())