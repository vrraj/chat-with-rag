#!/usr/bin/env python3
"""Standalone FastEmbed supported models lister + embedding test.

What this script does:
1) Lists all FastEmbed supported models (dense, sparse, ColBERT)
2) Checks whether a target model is supported
3) Generates a query embedding
4) Optionally runs a Qdrant vector search with that embedding

Quick runs:

List all supported models:
  python scripts/test_fastembed_supported_models.py --list-models
  python scripts/test_fastembed_supported_models.py --list-models --filter bge

Test embedding with a specific model:
  python scripts/test_fastembed_supported_models.py --model BAAI/bge-large-en-v1.5 --text "latest finance market outlook"

Test embedding + retrieval:
  python scripts/test_fastembed_supported_models.py --model BAAI/bge-large-en-v1.5 --text "latest finance market outlook" --qdrant-host localhost --qdrant-port 6333 --collection document_index_finance --top-k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Iterable


def _safe_import_fastembed():
    try:
        from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding  # type: ignore
        return TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
    except Exception as exc:
        print(f"ERROR: fastembed import failed: {exc}")
        print("Install with: pip install \"qdrant-client[fastembed]>=1.12.0\" \"fastembed>=0.7.0,<0.9.0\"")
        return None


def _safe_import_qdrant():
    try:
        from qdrant_client import QdrantClient  # type: ignore
        return QdrantClient
    except Exception as exc:
        print(f"ERROR: qdrant-client import failed: {exc}")
        print("Install with: pip install \"qdrant-client[fastembed]>=1.12.0\"")
        return None


def _iter_supported_model_names(raw: Any) -> Iterable[str]:
    if not isinstance(raw, list):
        return []
    names: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            name = str(item.get("model") or item.get("model_name") or "").strip()
            if name:
                names.append(name)
        else:
            name = str(item).strip()
            if name:
                names.append(name)
    return names


def _print_supported_models(text_embedding_cls: Any, contains: str | None = None) -> list[str]:
    try:
        raw = text_embedding_cls.list_supported_models()
    except Exception as exc:
        print(f"ERROR: failed to list supported models: {exc}")
        return []

    names = list(_iter_supported_model_names(raw))
    if contains:
        key = contains.lower().strip()
        names = [n for n in names if key in n.lower()]

    print(f"Supported models count: {len(names)}")
    for n in names:
        print(f"- {n}")
    return names


def _build_embedding(text_embedding_cls: Any, *, model: str, text: str, batch_size: int) -> list[float]:
    emb = text_embedding_cls(model_name=model)
    vectors = list(emb.embed([text], batch_size=max(1, int(batch_size))))
    if not vectors:
        raise RuntimeError("No vectors returned from FastEmbed")
    vec0 = list(vectors[0])
    if not vec0:
        raise RuntimeError("Empty embedding vector returned")
    return vec0


def _qdrant_search(
    qdrant_client_cls: Any,
    *,
    host: str,
    port: int,
    collection: str,
    vector: list[float],
    top_k: int,
) -> list[dict[str, Any]]:
    client = qdrant_client_cls(url=f"http://{host}:{port}")
    hits = client.search(
        collection_name=collection,
        query_vector=vector,
        limit=max(1, int(top_k)),
        with_payload=True,
        with_vectors=False,
    )
    out: list[dict[str, Any]] = []
    for h in hits:
        out.append(
            {
                "id": str(getattr(h, "id", "")),
                "score": float(getattr(h, "score", 0.0) or 0.0),
                "url": ((getattr(h, "payload", {}) or {}).get("url") if isinstance(getattr(h, "payload", {}), dict) else None),
                "section": ((getattr(h, "payload", {}) or {}).get("section") if isinstance(getattr(h, "payload", {}), dict) else None),
                "preview": (((getattr(h, "payload", {}) or {}).get("text") or "")[:180] if isinstance(getattr(h, "payload", {}), dict) else ""),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="FastEmbed BGE-M3 embedding/retrieval smoke test")
    parser.add_argument("--model", default="BAAI/bge-m3", help="FastEmbed model name")
    parser.add_argument("--text", default="finance policy updates and risk outlook", help="Text to embed")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")

    parser.add_argument("--list-models", action="store_true", help="List all supported FastEmbed models and exit")
    parser.add_argument("--filter", default="", help="Optional substring filter for --list-models")

    parser.add_argument("--qdrant-host", default="", help="Qdrant host (optional)")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--collection", default="", help="Qdrant collection name (optional)")
    parser.add_argument("--top-k", type=int, default=5, help="Qdrant top-k")

    args = parser.parse_args()

    fastembed_imports = _safe_import_fastembed()
    if fastembed_imports is None:
        return 2
    TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding = fastembed_imports

    if args.list_models:
        print("=" * 50)
        print("DENSE EMBEDDING MODELS")
        print("=" * 50)
        _print_supported_models(TextEmbedding, contains=(args.filter or None))
        print()
        print("=" * 50)
        print("SPARSE (LEXICAL) MODELS")
        print("=" * 50)
        _print_supported_models(SparseTextEmbedding, contains=(args.filter or None))
        print()
        print("=" * 50)
        print("LATE INTERACTION (COLBERT) MODELS")
        print("=" * 50)
        _print_supported_models(LateInteractionTextEmbedding, contains=(args.filter or None))
        return 0

    print(f"Testing FastEmbed model: {args.model}")
    supported = _print_supported_models(TextEmbedding, contains="bge")
    if args.model not in supported:
        print(f"\nWARNING: '{args.model}' was not found in supported model list.")
        print("Try one from the list above or run: --list-models")

    try:
        vec = _build_embedding(
            TextEmbedding,
            model=args.model,
            text=args.text,
            batch_size=args.batch_size,
        )
    except Exception as exc:
        print(f"\nERROR: embedding failed for model '{args.model}': {exc}")
        return 1

    print("\nEmbedding OK")
    print(f"- vector_dim: {len(vec)}")
    print(f"- sample: {vec[:8]}")

    if not (args.qdrant_host and args.collection):
        print("\nSkipping Qdrant search (provide --qdrant-host and --collection to enable).")
        return 0

    QdrantClient = _safe_import_qdrant()
    if QdrantClient is None:
        return 2

    print("\nRunning Qdrant vector search...")
    try:
        hits = _qdrant_search(
            QdrantClient,
            host=args.qdrant_host,
            port=args.qdrant_port,
            collection=args.collection,
            vector=vec,
            top_k=args.top_k,
        )
    except Exception as exc:
        print(f"ERROR: Qdrant search failed: {exc}")
        return 1

    print(f"- hits: {len(hits)}")
    print(json.dumps(hits, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
