# backend/utils/prompt_utils.py
import re
from typing import List, Dict

_SOURCES_RE = re.compile(r"(?:\r?\n)Sources:\s*\r?\n[\s\S]*\Z")

def _strip_trailing_sources(text: str) -> str:
    s = (text or "").rstrip()
    m = _SOURCES_RE.search(s)
    if m:
        s = s[:m.start()]
    return s.rstrip()

def convert_messages_to_prompt(
    messages: List[Dict[str, str]],
    *,
    strip_sources: bool = True,
    normalize_ws: bool = True,
) -> str:
    """
    Convert an OpenAI-style messages list to a single prompt string.
    - Optionally strips trailing 'Sources:' blocks from assistant messages.
    - Optionally normalizes whitespace (collapses runs of blank lines).
    """
    lines = []
    for m in messages or []:
        role = m.get("role", "user")
        content = m.get("content", "") or ""
        if strip_sources and role == "assistant":
            content = _strip_trailing_sources(content)
        # Keep role prefixes to make sections clear in a flat prompt
        lines.append(f"{role}: {content}".rstrip())

    prompt = "\n".join(lines).strip() + "\n"
    if normalize_ws:
        # collapse 3+ blank lines to 2, trim trailing spaces per line
        prompt = re.sub(r"[ \t]+(\r?\n)", r"\1", prompt)
        prompt = re.sub(r"(\r?\n){3,}", r"\n\n", prompt)
    return prompt