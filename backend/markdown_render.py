from __future__ import annotations

import html
import logging
import re
from typing import Optional

import bleach
from bs4 import BeautifulSoup, NavigableString
from markdown_it import MarkdownIt


logger = logging.getLogger(__name__)


_ALLOWED_TAGS = [
    "p",
    "br",
    "strong",
    "em",
    "code",
    "pre",
    "details",
    "summary",
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
    "svg",
    "polyline",
    "line",
    "rect",
    "path",
    "text",
    "g",
]

_ALLOWED_ATTRS = {
    "a": ["href", "title", "target", "rel"],
    "th": ["colspan", "rowspan"],
    "td": ["colspan", "rowspan"],
    "div": ["class"],
    "details": ["class"],
    "svg": [
        "width",
        "height",
        "viewBox",
        "role",
        "aria-hidden",
        "xmlns",
    ],
    "polyline": ["fill", "stroke", "stroke-width", "stroke-linecap", "stroke-linejoin", "points"],
    "line": ["x1", "y1", "x2", "y2", "stroke", "stroke-width", "stroke-linecap"],
    "rect": ["x", "y", "width", "height", "fill", "rx", "ry", "stroke", "stroke-width"],
    "path": ["d", "fill", "stroke", "stroke-width", "stroke-linecap", "stroke-linejoin"],
    "text": ["x", "y", "text-anchor", "fill", "font-size", "font-weight"],
    "g": ["transform", "fill", "stroke", "stroke-width"],
}


def render_markdown_to_html(markdown_text: Optional[str]) -> str:
    src = "" if markdown_text is None else str(markdown_text)

    thought_blocks: list[str] = []
    try:
        thought_blocks = [m.group(1).strip() for m in re.finditer(r"<thought>([\s\S]*?)</thought>", src, flags=re.IGNORECASE) if m.group(1).strip()]
        if thought_blocks:
            src = re.sub(r"<thought>[\s\S]*?</thought>", "", src, flags=re.IGNORECASE).strip()
    except Exception:
        thought_blocks = []

    # SVG-only raw HTML pass-through:
    # - stash <svg> blocks before markdown conversion,
    # - escape all other raw tags,
    # - restore only SVG blocks before bleach sanitization.
    svg_blocks: list[str] = []
    svg_token_prefix = "RAWSVGBLOCKTOKEN"

    def _svg_token(i: int) -> str:
        return f"{svg_token_prefix}{i}END"

    try:
        def _stash_svg(m: re.Match[str]) -> str:
            svg_blocks.append(m.group(0))
            return _svg_token(len(svg_blocks) - 1)

        src = re.sub(r"<svg\b[\s\S]*?</svg>", _stash_svg, src, flags=re.IGNORECASE)

        # Escape any remaining raw HTML-like tags so only stashed SVG can be injected.
        src = re.sub(r"</?[A-Za-z][^>]*>", lambda m: html.escape(m.group(0)), src)
        if svg_blocks:
            logger.info("[MD-SVG] stashed_svg_blocks=%d", len(svg_blocks))
    except Exception:
        svg_blocks = []

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

    if svg_blocks:
        try:
            restored = 0
            for i, block in enumerate(svg_blocks):
                token = _svg_token(i)
                if token in raw_html:
                    restored += 1
                raw_html = raw_html.replace(token, block)
                # Backward-compat fallback for previously used token format that markdown
                # could convert to strong text (e.g., __RAW_SVG_BLOCK_0__).
                raw_html = raw_html.replace(f"<strong>RAW_SVG_BLOCK_{i}</strong>", block)
                raw_html = raw_html.replace(f"RAW_SVG_BLOCK_{i}", block)

            # Any leftover token means markdown transformed or dropped placeholder text.
            if svg_token_prefix in raw_html:
                logger.warning("[MD-SVG] leftover_svg_tokens_detected_after_restore")

            if restored != len(svg_blocks):
                logger.warning(
                    "[MD-SVG] restored_token_count_mismatch restored=%d expected=%d",
                    restored,
                    len(svg_blocks),
                )
        except Exception:
            pass

    clean_html = bleach.clean(
        raw_html,
        tags=_ALLOWED_TAGS,
        attributes=_ALLOWED_ATTRS,
        protocols=["http", "https", "mailto"],
        strip=True,
    )

    if svg_blocks:
        try:
            # Basic structural sanity check for truncated SVG after sanitize.
            open_count = len(re.findall(r"<svg\b", clean_html, flags=re.IGNORECASE))
            close_count = len(re.findall(r"</svg>", clean_html, flags=re.IGNORECASE))
            logger.info(
                "[MD-SVG] post_sanitize_svg_counts open=%d close=%d",
                open_count,
                close_count,
            )
            if open_count != close_count:
                logger.warning(
                    "[MD-SVG] possible_svg_cutoff_detected open=%d close=%d",
                    open_count,
                    close_count,
                )
        except Exception:
            pass

    # Post-process for UX parity with existing CSS: wrap tables and harden links.
    try:
        soup = BeautifulSoup(clean_html, "html.parser")

        def _split_sources_entries(s: str) -> list[str]:
            """Split a Sources block tail into per-source entries.

            Handles indices like "[1] ..." and collapsed indices like "[3, 4] ...".
            """
            try:
                txt = (s or "").replace("\n", " ").strip()
                if not txt:
                    return []

                # Match bracketed index groups: [1], [12], [3, 4], [3,4,5]
                pat = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")
                matches = list(pat.finditer(txt))
                if not matches:
                    return [txt]

                out: list[str] = []
                for i, m in enumerate(matches):
                    start = m.start()
                    end = matches[i + 1].start() if i + 1 < len(matches) else len(txt)
                    seg = txt[start:end].strip()
                    if seg:
                        out.append(seg)
                return out
            except Exception:
                return [s.strip()] if (s or "").strip() else []

        # 1. Wrap tables in scrollable containers and harden links
        for table in soup.find_all("table"):
            wrapper = soup.new_tag("div", **{"class": "md-table-wrap"})
            table.wrap(wrapper)
            for a in table.find_all("a", href=True):
                a["target"] = "_blank"
                a["rel"] = "noopener noreferrer"

        # 2. Fix Sources that got incorrectly wrapped in table rows by the Markdown parser
        for tr in soup.find_all("tr"):
            txt = tr.get_text().strip()
            if txt.startswith("Sources:"):
                # Replace the <tr> with a plain <p> containing the text
                p = soup.new_tag("p")
                p.string = txt
                tr.replace_with(p)

        if thought_blocks:
            details = soup.new_tag("details")
            details["class"] = "thoughts"
            summary = soup.new_tag("summary")
            summary.append(NavigableString("Reasoning"))
            details.append(summary)
            pre = soup.new_tag("pre")
            pre.append(NavigableString("\n\n".join(thought_blocks)))
            details.append(pre)
            if soup.contents:
                soup.insert(0, details)
            else:
                soup.append(details)

        # 3. Ensure Sources: starts on a new line and each source is on its own line
        # Handle both paragraphs and list items; list items must remain valid HTML.
        for node in list(soup.find_all(["p", "li"])):
            try:
                # Never rewrite nodes that contain SVG; source-normalization rewrites
                # can otherwise remove mixed-content blocks that include charts.
                if node.find("svg") is not None:
                    continue

                txt = node.get_text("\n").strip()
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
                    raw_lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
                    if raw_lines:
                        tail = " ".join(raw_lines[1:]).strip()
                        sources_lines = [raw_lines[0]] + _split_sources_entries(tail)
                else:
                    # Split sources like: "[2] https://... [3] https://..." into per-source lines.
                    sources_lines = ["Sources:"] + _split_sources_entries(after_txt)

                if not sources_lines:
                    continue

                insert_after_node = node

                if before_txt:
                    if getattr(node, "name", None) == "li":
                        # Keep list structure valid: rewrite the LI in-place.
                        node.clear()
                        node.append(NavigableString(before_txt))
                        insert_after_node = node
                    else:
                        main_p = soup.new_tag("p")
                        main_p.append(NavigableString(before_txt))
                        node.replace_with(main_p)
                        insert_after_node = main_p
                else:
                    # If the paragraph was only sources, remove it and insert sources in its place.
                    node.extract()

                if getattr(insert_after_node, "name", None) == "li":
                    # Render sources inside the list item to keep valid HTML.
                    insert_after_node.append(soup.new_tag("br"))
                    sources_label = soup.new_tag("strong")
                    sources_label.append(NavigableString(sources_lines[0]))
                    insert_after_node.append(sources_label)
                    if sources_lines[1:]:
                        sources_div = soup.new_tag("div")
                        sources_div["class"] = "sources"
                        for i, ln in enumerate(sources_lines[1:]):
                            if i > 0:
                                sources_div.append(soup.new_tag("br"))
                            sources_div.append(NavigableString(ln))
                        insert_after_node.append(soup.new_tag("br"))
                        insert_after_node.append(sources_div)
                else:
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
