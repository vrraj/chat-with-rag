#!/usr/bin/env python3
"""Gemini API smoke test.

Purpose
- Quickly verify that GEMINI_API_KEY is present and can authenticate to Google Gemini API.

What it does
- Loads GEMINI_API_KEY from environment.
- If missing, optionally attempts to read it from a local `.env` file (project root).
- Makes a minimal HTTPS request to Google Gemini REST API (`/v1beta/models`).

Exit codes
- 0: Success
- 2: Missing/invalid configuration (no API key)
- 3: Authentication/authorization error
- 4: Network/timeout error
- 5: Unexpected API error

Notes
- This intentionally avoids requiring Google Generative AI Python SDK.
- This script does NOT print API key.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple


def _load_key_from_dotenv(dotenv_path: Path) -> Optional[str]:
    """Very small .env reader: finds GEMINI_API_KEY=... and returns the value.

    Supports lines like:
      GEMINI_API_KEY=...
      GEMINI_API_KEY='...'
      GEMINI_API_KEY="..."

    Ignores comments and blank lines.
    """
    if not dotenv_path.exists() or not dotenv_path.is_file():
        return None

    for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("GEMINI_API_KEY="):
            continue
        val = line.split("=", 1)[1].strip()
        # Strip simple surrounding quotes
        if (val.startswith("\"") and val.endswith("\"")) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        val = val.strip()
        return val or None

    return None


def _get_api_key(allow_dotenv: bool) -> Optional[str]:
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key.strip() or None

    if not allow_dotenv:
        return None

    # Assume script lives in scripts/; repo root is parent directory.
    repo_root = Path(__file__).resolve().parent.parent
    return _load_key_from_dotenv(repo_root / ".env")


def _http_json(url: str, headers: Dict[str, str], timeout: float) -> Tuple[int, dict]:
    req = urllib.request.Request(url=url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        status = getattr(resp, "status", 200)
        body = resp.read().decode("utf-8", errors="replace")
    try:
        data = json.loads(body) if body else {}
    except json.JSONDecodeError:
        data = {"raw": body}
    return status, data


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test Gemini API credentials")
    parser.add_argument(
        "--base-url",
        default=os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com"),
        help="Gemini API base URL (default: https://generativelanguage.googleapis.com or env GEMINI_BASE_URL)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("GEMINI_HTTP_TIMEOUT", "15")),
        help="HTTP timeout in seconds (default: 15 or env GEMINI_HTTP_TIMEOUT)",
    )
    parser.add_argument(
        "--no-dotenv",
        action="store_true",
        help="Do not attempt to read GEMINI_API_KEY from a local .env file",
    )

    args = parser.parse_args()

    api_key = _get_api_key(allow_dotenv=not args.no_dotenv)
    if not api_key:
        print(
            "❌ GEMINI_API_KEY is not set. Set it in your environment or in local .env file.",
            file=sys.stderr,
        )
        return 2

    # Gemini API uses the API key as a query parameter
    url = f"{args.base_url.rstrip('/')}/v1beta/models?key={api_key}"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "chat-with-rag-smoke-test/1.0",
    }

    print(f"🧪 Testing Gemini API auth via GET {args.base_url.rstrip('/')}/v1beta/models ...")

    try:
        status, data = _http_json(url, headers=headers, timeout=args.timeout)
    except urllib.error.HTTPError as e:
        # HTTPError is also a file-like response; try to read body
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""

        # Try parse JSON error payload
        err = {}
        try:
            err = json.loads(body) if body else {}
        except json.JSONDecodeError:
            err = {"raw": body}

        status = getattr(e, "code", 0) or 0
        msg = (
            err.get("error", {}).get("message")
            if isinstance(err, dict)
            else None
        )

        if status in (401, 403):
            print("❌ Authentication failed (401/403).", file=sys.stderr)
            if msg:
                print(f"   {msg}", file=sys.stderr)
            return 3

        print(f"❌ Gemini API returned HTTP {status}.", file=sys.stderr)
        if msg:
            print(f"   {msg}", file=sys.stderr)
        elif body:
            print(f"   {body[:500]}", file=sys.stderr)
        return 5

    except urllib.error.URLError as e:
        print("❌ Network error while calling Gemini API.", file=sys.stderr)
        print(f"   {e}", file=sys.stderr)
        return 4

    # Success path
    if 200 <= status < 300:
        # Keep output short; just confirm we got a list-ish payload
        count = None
        if isinstance(data, dict):
            items = data.get("models")
            if isinstance(items, list):
                count = len(items)
        if count is not None:
            print(f"✅ Success! Auth OK. Models visible: {count}")
        else:
            print("✅ Success! Auth OK.")
        return 0

    # Unexpected non-2xx without exception
    print(f"❌ Unexpected HTTP status {status}.", file=sys.stderr)
    if isinstance(data, dict) and data.get("error", {}).get("message"):
        print(f"   {data['error']['message']}", file=sys.stderr)
    else:
        print(f"   {str(data)[:500]}", file=sys.stderr)
    return 5


if __name__ == "__main__":
    raise SystemExit(main())
