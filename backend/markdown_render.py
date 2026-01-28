from __future__ import annotations

import re
from typing import Optional

import bleach
from bs4 import BeautifulSoup, NavigableString
from markdown_it import MarkdownIt


_ALLOWED_TAGS = [
    "p",
    "br",
    "strong",
    "em",
    "code",
    "pre",
    "blockquote",
    "ul",
    "ol",
    "li",
    "hr",
    "a",
    "table",
    "thead",
    "tbody",
    "tr",
    "th",
    "td",
    "div",
]

_ALLOWED_ATTRS = {
    "a": ["href", "title", "target", "rel"],
    "th": ["colspan", "rowspan"],
    "td": ["colspan", "rowspan"],
    "div": ["class"],
}


def render_markdown_to_html(markdown_text: Optional[str]) -> str:
    src = "" if markdown_text is None else str(markdown_text)

    raw_html = ""
    try:
        # Prefer Python-Markdown for reliable GFM-style tables support.
        import markdown as _md  # type: ignore

        raw_html = _md.markdown(
            src,
            extensions=[
                "tables",
                "fenced_code",
            ],
            output_format="html",
        )
    except Exception:
        # Fallback: basic CommonMark rendering (no tables).
        md = MarkdownIt("commonmark", {"breaks": True, "html": False})
        raw_html = md.render(src)

    clean_html = bleach.clean(
        raw_html,
        tags=_ALLOWED_TAGS,
        attributes=_ALLOWED_ATTRS,
        protocols=["http", "https", "mailto"],
        strip=True,
    )

    # Post-process for UX parity with existing CSS: wrap tables and harden links.
    try:
        soup = BeautifulSoup(clean_html, "html.parser")

        for a in soup.find_all("a"):
            try:
                a["target"] = "_blank"
                a["rel"] = "noopener noreferrer"
            except Exception:
                pass

        for tbl in soup.find_all("table"):
            try:
                parent = tbl.parent
                if parent and getattr(parent, "name", "") == "div" and "md-table-wrap" in (parent.get("class") or []):
                    continue
                wrap = soup.new_tag("div")
                wrap["class"] = "md-table-wrap"
                tbl.wrap(wrap)
            except Exception:
                continue

        for p in list(soup.find_all("p")):
            try:
                txt = p.get_text("\n").strip()
                if "Sources:" not in txt:
                    continue

                before_txt = ""
                after_txt = txt
                if "Sources:" in txt:
                    parts = txt.split("Sources:", 1)
                    before_txt = (parts[0] or "").strip()
                    after_txt = (parts[1] or "").strip()

                if not after_txt and txt.startswith("Sources:"):
                    after_txt = ""

                sources_lines: list[str] = []
                if txt.startswith("Sources:"):
                    # Existing behavior: Sources block starts the paragraph.
                    sources_lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
                else:
                    # Split sources like: "[2] https://... [3] https://..." into per-source lines.
                    after_txt = after_txt.replace("\n", " ")
                    idx_matches = list(re.finditer(r"\[\d+\]", after_txt))
                    if idx_matches:
                        for i, m in enumerate(idx_matches):
                            start = m.start()
                            end = idx_matches[i + 1].start() if i + 1 < len(idx_matches) else len(after_txt)
                            seg = after_txt[start:end].strip()
                            if seg:
                                sources_lines.append(seg)
                    else:
                        if after_txt.strip():
                            sources_lines.append(after_txt.strip())
                    sources_lines = ["Sources:"] + sources_lines

                if not sources_lines:
                    continue

                insert_after_node = p

                if before_txt:
                    main_p = soup.new_tag("p")
                    main_p.append(NavigableString(before_txt))
                    p.replace_with(main_p)
                    insert_after_node = main_p
                else:
                    # If the paragraph was only sources, remove it and insert sources in its place.
                    p.extract()

                sources_p = soup.new_tag("p")
                sources_label = soup.new_tag("strong")
                sources_label.append(NavigableString(sources_lines[0]))
                sources_p.append(sources_label)
                sources_div = soup.new_tag("div")
                sources_div["class"] = "sources"
                for i, ln in enumerate(sources_lines[1:]):
                    if i > 0:
                        sources_div.append(soup.new_tag("br"))
                    sources_div.append(NavigableString(ln))

                insert_after_node.insert_after(sources_p)
                if sources_lines[1:]:
                    sources_p.insert_after(sources_div)
            except Exception:
                continue

        return str(soup)
    except Exception:
        return clean_html
