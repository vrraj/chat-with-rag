from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from backend.core.config import settings


_CONFIG_CACHE: Dict[str, Dict[str, Any]] = {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml(path: str) -> Dict[str, Any]:
    cache_key = str(path or "").strip()
    if cache_key in _CONFIG_CACHE:
        return _CONFIG_CACHE[cache_key]

    p = Path(cache_key)
    if not p.exists():
        _CONFIG_CACHE[cache_key] = {}
        return {}

    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}

    _CONFIG_CACHE[cache_key] = data
    return data


def clear_retrieval_config_cache() -> None:
    _CONFIG_CACHE.clear()


def resolve_retrieval_specs(*, domain: str | None, config_path: str | None = None) -> Dict[str, Dict[str, Any]]:
    path = str(config_path or getattr(settings, "retrieval_config_path", "prompts/local_models_registry.yaml") or "").strip()
    cfg = _read_yaml(path)

    retrieval_cfg = cfg.get("retrieval") if isinstance(cfg, dict) else {}
    retrieval_cfg = retrieval_cfg if isinstance(retrieval_cfg, dict) else {}

    defaults = retrieval_cfg.get("defaults") if isinstance(retrieval_cfg.get("defaults"), dict) else {}
    domains = retrieval_cfg.get("domains") if isinstance(retrieval_cfg.get("domains"), dict) else {}

    dom = str(domain or "").strip()
    dom_cfg = domains.get(dom) if dom and isinstance(domains.get(dom), dict) else {}

    effective = _deep_merge(defaults, dom_cfg)

    embedding = effective.get("embedding") if isinstance(effective.get("embedding"), dict) else {}
    rerank = effective.get("rerank") if isinstance(effective.get("rerank"), dict) else {}

    return {
        "embedding": embedding,
        "rerank": rerank,
    }
