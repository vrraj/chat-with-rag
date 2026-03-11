import argparse
import os
import sys
import time
from typing import Any, Dict, Optional


def _get_env_api_key() -> Optional[str]:
    return (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GENAI_API_KEY")
    )


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _truncate(s: Any, max_chars: int) -> str:
    try:
        txt = str(s)
    except Exception:
        try:
            txt = repr(s)
        except Exception:
            return "<unprintable>"
    if max_chars is not None and max_chars > 0 and len(txt) > max_chars:
        return txt[:max_chars] + "...<truncated>"
    return txt


def _dump_raw_to_file(resp: Any, path: str) -> None:
    chunks: list[str] = []
    chunks.append("=== Raw response type ===\n" + str(type(resp)) + "\n")
    try:
        chunks.append("=== Raw response repr ===\n" + repr(resp) + "\n")
    except Exception as e:
        chunks.append("=== Raw response repr failed ===\n" + str(e) + "\n")

    try:
        md = _safe_getattr(resp, "model_dump", None)
        if callable(md):
            dumped = md()
            chunks.append("=== Raw response model_dump ===\n" + str(dumped) + "\n")
    except Exception as e:
        chunks.append("=== Raw response model_dump failed ===\n" + str(e) + "\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(chunks))


def _print_response(resp: Any, *, raw_max_chars: int = 4000, dump_raw_file: Optional[str] = None) -> None:
    print("\n=== Raw response type ===")
    print(type(resp))

    if dump_raw_file:
        try:
            _dump_raw_to_file(resp, dump_raw_file)
            print(f"\n=== Raw response written to file ===\n{dump_raw_file}")
        except Exception as e:
            print("\n=== Raw response file dump failed ===")
            print(str(e))

    print("\n=== Raw response (repr, truncated) ===")
    if raw_max_chars == 0:
        print(repr(resp))
    else:
        print(_truncate(repr(resp), raw_max_chars))

    # If this is a Pydantic model, try a structured dump (bounded).
    try:
        md = _safe_getattr(resp, "model_dump", None)
        if callable(md):
            dumped = md()
            print("\n=== Raw response (model_dump, truncated) ===")
            if raw_max_chars == 0:
                print(str(dumped))
            else:
                print(_truncate(dumped, raw_max_chars))
    except Exception:
        pass

    # Common convenience fields
    resp_text = _safe_getattr(resp, "text", None)
    if isinstance(resp_text, str) and resp_text.strip():
        print("\n=== resp.text ===")
        print(resp_text)

    # google-genai responses usually have candidates -> content -> parts
    candidates = _safe_getattr(resp, "candidates", None)
    if not isinstance(candidates, list):
        print("\n(No candidates list found on response; printing str(resp))")
        print(str(resp))
        return

    print(f"\n=== candidates: {len(candidates)} ===")
    for ci, cand in enumerate(candidates):
        content = _safe_getattr(cand, "content", None)
        parts = _safe_getattr(content, "parts", None) if content is not None else None
        print(f"\n--- candidate[{ci}] ---")
        if not isinstance(parts, list):
            print("(No parts list found)")
            print("candidate:", str(cand))
            continue

        for pi, part in enumerate(parts):
            text = _safe_getattr(part, "text", None)
            thought = _safe_getattr(part, "thought", None)
            print(f"\npart[{pi}] type={type(part)}")
            if isinstance(thought, str) and thought.strip():
                print("[THOUGHT]")
                print(thought)
            if isinstance(text, str) and text.strip():
                print("[TEXT]")
                print(text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated native google-genai reasoning/thoughts test")
    parser.add_argument(
        "--model",
        default=os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash"),
        help="Gemini model name (e.g. models/gemini-2.5-flash)",
    )
    parser.add_argument(
        "--prompt",
        default="Explain why the sky is blue. Include your internal reasoning/thoughts.",
        help="Prompt to send",
    )
    parser.add_argument("--include-thoughts", action="store_true", help="Request thoughts")
    parser.add_argument("--thinking-budget", type=int, default=None, help="Thinking budget (int)")
    parser.add_argument("--thinking-level", type=str, default=None, help="Thinking level (string)")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=512,
        help="Max output tokens",
    )
    parser.add_argument(
        "--raw-max-chars",
        type=int,
        default=4000,
        help="Max characters to print for raw response dumps (repr/model_dump). Use 0 for unlimited.",
    )
    parser.add_argument(
        "--dump-raw-file",
        type=str,
        default=None,
        help="Write full raw response (repr + model_dump if available) to this file.",
    )
    parser.add_argument(
        "--no-dump-raw",
        action="store_true",
        help="Disable writing raw response to disk (default behavior writes a dump file).",
    )

    args = parser.parse_args()

    api_key = _get_env_api_key()
    if not api_key:
        print(
            "Missing API key. Set GOOGLE_API_KEY (or GEMINI_API_KEY/GENAI_API_KEY) in your environment.",
            file=sys.stderr,
        )
        return 2

    # Prefer the consolidated SDK import style:
    #   from google import genai
    #   from google.genai import types
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except Exception as e:
        print("Failed to import google-genai SDK (from google import genai).", file=sys.stderr)
        print("Error:", str(e), file=sys.stderr)
        return 2

    print("=== SDK diagnostics ===")
    print("python:", sys.version)
    print("GOOGLE_API_KEY present:", bool(os.getenv("GOOGLE_API_KEY")))
    print("GEMINI_API_KEY present:", bool(os.getenv("GEMINI_API_KEY")))
    print("GENAI_API_KEY present:", bool(os.getenv("GENAI_API_KEY")))

    ThinkingConfig = getattr(types, "ThinkingConfig", None)
    print("types.ThinkingConfig present:", ThinkingConfig is not None)

    # Build thinking_config
    tc_kwargs: Dict[str, Any] = {}
    if args.include_thoughts:
        tc_kwargs["include_thoughts"] = True
    if args.thinking_budget is not None:
        tc_kwargs["thinking_budget"] = int(args.thinking_budget)
    if args.thinking_level is not None:
        tc_kwargs["thinking_level"] = str(args.thinking_level)

    # Gemini rejects using both at once; prefer level if both specified.
    if "thinking_budget" in tc_kwargs and "thinking_level" in tc_kwargs:
        tc_kwargs.pop("thinking_budget", None)

    thinking_config: Any = None
    if tc_kwargs:
        if ThinkingConfig is not None:
            try:
                thinking_config = ThinkingConfig(**tc_kwargs)
            except Exception as e:
                print("ThinkingConfig(**tc_kwargs) failed; will fall back to dict.")
                print("Error:", str(e))
                thinking_config = dict(tc_kwargs)
        else:
            thinking_config = dict(tc_kwargs)

    # Build GenerateContentConfig
    cfg_kwargs: Dict[str, Any] = {
        "temperature": float(args.temperature),
        "max_output_tokens": int(args.max_output_tokens),
    }
    if thinking_config is not None:
        cfg_kwargs["thinking_config"] = thinking_config

    try:
        config = types.GenerateContentConfig(**cfg_kwargs)
    except Exception as e:
        print("Failed to build GenerateContentConfig.", file=sys.stderr)
        print("cfg_kwargs=", cfg_kwargs, file=sys.stderr)
        print("Error:", str(e), file=sys.stderr)
        return 2

    print("\n=== Request ===")
    print("model:", args.model)
    print("tc_kwargs:", tc_kwargs)
    print("thinking_config type:", type(thinking_config) if thinking_config is not None else None)

    dump_raw_file: Optional[str] = None
    if not args.no_dump_raw:
        if args.dump_raw_file:
            dump_raw_file = str(args.dump_raw_file)
        else:
            ts = int(time.time())
            safe_model = str(args.model).replace("/", "_").replace(":", "_")
            dump_raw_file = f"/tmp/gemini_raw_{safe_model}_{ts}.txt"

    # Make native SDK call
    try:
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=args.model,
            contents=args.prompt,
            config=config,
        )
    except Exception as e:
        print("Native SDK call failed.", file=sys.stderr)
        print("Error:", str(e), file=sys.stderr)
        return 1

    _print_response(
        resp,
        raw_max_chars=int(args.raw_max_chars),
        dump_raw_file=dump_raw_file,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
