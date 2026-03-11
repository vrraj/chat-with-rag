#!/usr/bin/env python3

import argparse
import json
import os
import sys
from typing import Any


def _read_text_arg(value: str | None) -> str:
    if not value:
        return ""
    v = str(value)
    if v.startswith("@"):  # convenience: @path/to/file
        path = v[1:]
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description="Test the summary_update prompt without running the full chat pipeline")
    ap.add_argument("--domain", default=None, help="Prompt domain (e.g. mountains). Defaults to settings.prompt_domain_default.")
    ap.add_argument("--prior", default="", help="Prior summary text, or @/path/to/file")
    ap.add_argument("--recent", default="", help="Recent conversation text (verbatim), or @/path/to/file")
    ap.add_argument("--model", default=None, help="Override model (defaults to settings.summarizer_model)")
    ap.add_argument("--temperature", type=float, default=None, help="Override temperature (defaults to settings.summarizer_temperature)")
    ap.add_argument("--max-output-tokens", type=int, default=None, help="Override max output tokens (defaults to settings.summarizer_max_output_tokens)")
    ap.add_argument("--registry-path", default=None, help="Override prompt registry path (defaults to settings.inference_prompt_registry_path)")
    ap.add_argument("--call", action="store_true", help="Actually call the LLM. By default, prints the rendered prompt only.")
    ap.add_argument("--print-json", action="store_true", help="Print result as JSON")
    ap.add_argument("--show-messages", action="store_true", help="Print the exact messages array that would be sent to the LLM (system + user)")
    ap.add_argument("--messages-only", action="store_true", help="Print only the messages payload (implies --show-messages)")
    args = ap.parse_args()

    # Ensure repo root is on sys.path so backend imports work when invoked from anywhere.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from backend.chat.prompt_registry import resolve_summary_update_prompt, render_full_payload

    settings = None
    if args.call or (args.registry_path is None) or (args.domain is None):
        try:
            from backend.core.config import settings as _settings
            settings = _settings
        except Exception:
            settings = None

    llm_client = None
    if args.call:
        from backend.llm.llm_client import generate as _llm_client
        llm_client = _llm_client

    prior = _read_text_arg(args.prior).strip()
    recent = _read_text_arg(args.recent).strip()

    domain = args.domain
    if domain is None:
        domain = str(getattr(settings, "prompt_domain_default", "") if settings is not None else "")

    registry_path = args.registry_path
    if not registry_path:
        registry_path = str(getattr(settings, "inference_prompt_registry_path", "prompts/prompt_registry.yaml") if settings is not None else "prompts/prompt_registry.yaml")

    spec = resolve_summary_update_prompt(registry_path=registry_path, domain=domain)

    payload = render_full_payload(
        spec.full_payload_template,
        variables={
            "prior_chat_summary": prior,
            "recent_conversation": recent,
        },
    )

    messages_payload = [
        {"role": "system", "content": spec.system_instruction},
        {"role": "user", "content": payload},
    ]

    if not args.call:
        if args.messages_only:
            args.show_messages = True

        if args.show_messages:
            if args.print_json:
                print(json.dumps(messages_payload, indent=2, ensure_ascii=False))
            else:
                print(messages_payload)
            return 0

        out_obj: dict[str, Any] = {
            "domain": domain,
            "registry_path": registry_path,
            "messages": messages_payload,
            "note": "Dry run (no LLM call). Pass --call to run the summary_update model.",
        }
        if args.print_json:
            print(json.dumps(out_obj, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(out_obj, indent=2, ensure_ascii=False))
        return 0

    if llm_client is None:
        raise RuntimeError("--call was provided but llm_client could not be imported")

    _default_model = "gpt-4o-mini"
    if settings is not None:
        try:
            _default_model = getattr(settings, "summarizer_model", getattr(settings, "inference_model", _default_model))
        except Exception:
            _default_model = "gpt-4o-mini"
    model = args.model or _default_model

    temperature = args.temperature if args.temperature is not None else float(getattr(settings, "summarizer_temperature", 0.3) if settings is not None else 0.3)
    max_output_tokens = args.max_output_tokens if args.max_output_tokens is not None else int(getattr(settings, "summarizer_max_output_tokens", 128) if settings is not None else 128)

    resp = llm_client(
        model_key=f"openai:{model}",
        input=messages_payload,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        stream=False,
    )

    updated = ""
    try:
        updated = str((resp or {}).get("text") or "").strip()
    except Exception:
        updated = ""

    out_obj2: dict[str, Any] = {
        "domain": domain,
        "registry_path": registry_path,
        "model": model,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "updated_summary": updated,
    }
    if args.print_json:
        print(json.dumps(out_obj2, indent=2, ensure_ascii=False))
    else:
        print(updated)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
