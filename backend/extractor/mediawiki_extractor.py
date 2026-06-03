import mwparserfromhell
from typing import List, Dict, Optional
import requests
import time
from requests import Response
from urllib.parse import unquote, urlparse
import tiktoken
import logging
from backend.core.config import settings
from .splitters import TextSplitter
from bs4 import BeautifulSoup


logger = logging.getLogger(__name__)

WIKI_API_URL = settings.wiki_api_url
WIKI_UA = settings.wiki_user_agent
DEFAULT_TIMEOUT = int(settings.wiki_timeout_secs)
MAX_RETRIES = int(settings.wiki_max_retries)

# Reusable session for connection pooling
_wiki_session = requests.Session()

import re

def clean_mediawiki_text(text: str) -> str:
    # Remove image/file references
    text = re.sub(r'\[\[File:[^\]]*\]\]', '', text)
    text = re.sub(r'\[\[Image:[^\]]*\]\]', '', text)
    # Remove templates
    text = re.sub(r'\{\{[^}]+\}\}', '', text)
    # Remove categories and special links
    text = re.sub(r'\[\[Category:[^\]]*\]\]', '', text)
    text = re.sub(r'\[\[[^\]]*:[^\]]*\]\]', '', text)  # Interwiki links
    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    # Remove references
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^/]*/>', '', text)
    # Remove table markup
    text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL)
    # Remove MediaWiki magic words
    text = re.sub(r'__\w+__', '', text)
    return text

def _wiki_get(params: Dict, api_url: str, ua: str) -> Response:
    """Resilient GET to Wikipedia Action API with UA + backoff."""
    backoff = 0.5
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = _wiki_session.get(
                api_url,
                params=params,
                headers={
                    "User-Agent": ua,
                    # Request uncompressed response to avoid rare client-side
                    # decompression issues (e.g., truncated gzip streams).
                    "Accept-Encoding": "identity",
                },
                timeout=DEFAULT_TIMEOUT,
            )
            # Retry on common throttle/forbidden statuses
            if r.status_code in (429, 403, 502, 503, 504):
                logger.warning(
                    "Wiki API status %s; attempt %s/%s. Backing off %.1fs",
                    r.status_code, attempt, MAX_RETRIES, backoff,
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                logger.error("Wiki API request failed after %s attempts: %s", MAX_RETRIES, e)
                raise
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError("Unreachable: _wiki_get loop")

# NOTE: MediaWiki extractor supports two fetch modes:
#  - action_api: per-section pulls via action=parse (exact hierarchy; multiple HTTP calls)
#  - parsoid: single fetch via REST /api/rest_v1/page/html/{Title} then DOM walk (native heading ids)
# The instance toggles below control section filtering and cleanup behavior.

class MediaWikiExtractor:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        api_url: Optional[str] = None,
        user_agent: Optional[str] = None,
        *,
        # --- MediaWiki toggles ---
        wiki_mode: str = "parsoid",  # or "action_api" . Set in MediaWikiExtractor instance in main.py
        wiki_drop_selectors: Optional[List[str]] = None,
        wiki_preserve_paragraphs: bool = True,
        wiki_index_tables: bool = False,
        wiki_table_rows_per_chunk: int = 12,
        wiki_drop_tables_from_prose: bool = False,
    ):
        # Per-instance API settings (fall back to global settings defaults)
        self.api_url = api_url or WIKI_API_URL
        self.user_agent = user_agent or WIKI_UA
        self.splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_manual_splitter=True
        )
        # wiki_mode: "action_api" for per-section API; "parsoid" for single fetch + DOM walk
        # wiki_drop_selectors: CSS selectors removed before text extraction
        # wiki_preserve_paragraphs: keep \n\n paragraph breaks (True) vs flatten (False)
        # Toggle defaults
        # IMPORTANT:
        # Intentionally drop `.infobox` from the main HTML-to-text extraction.
        # The MediaWiki extractor now pulls the infobox separately using `_infobox_html_to_text()` and emits a dedicated `section="INFOBOX"` payload.
        # Leaving `.infobox` here prevents the raw table HTML from leaking into the
        # Lead section or other prose sections, which would create duplication:
        #   - INFOBOX chunk (clean key/value format)
        #   - PLUS the messy table text again in the Lead/prose (undesired)
        # Do NOT remove `.infobox` unless you explicitly want the infobox table embedded in the prose — which will break current RAG assumptions.
        self.wiki_mode = wiki_mode
        self.wiki_drop_selectors = (
            wiki_drop_selectors
            if wiki_drop_selectors is not None
            else [
                "sup.reference",
                ".mw-references-wrap",
                ".navbox",
                ".infobox",
                "figure",
                "style",
                "script",
            ]
        )
        self.wiki_preserve_paragraphs = wiki_preserve_paragraphs
        # Optional: emit additional payloads for HTML tables (useful for list-pages).
        # This is additive and does not change existing prose extraction.
        self.wiki_index_tables = wiki_index_tables
        self.wiki_table_rows_per_chunk = wiki_table_rows_per_chunk
        # Optional: if True, remove ALL <table> elements from the prose extraction path
        # (_html_to_text). This prevents tables from being flattened into normal section
        # prose chunks when you prefer to index them only via the structured table path
        # (`_emit_table_payloads`, content_type="table").
        #
        # IMPORTANT: This is additive and non-breaking because the default is False.
        self.wiki_drop_tables_from_prose = wiki_drop_tables_from_prose

    def parse(self, wikitext: str, url: str, skip_sections: Optional[List[str]] = None) -> List[Dict]:
        """
        Parse wikitext into section/subsection chunks, split into token chunks, and return flattened payloads.
        """
        logger.debug("Wiki: parse url=%s", url)
        wikicode = mwparserfromhell.parse(wikitext)
        payloads = []
        section_index = 0
        subsection_index = 0  # Track subsection index

        _sections = wikicode.get_sections(include_lead=True, levels=[2])
        logger.debug("Wiki: top-level sections (incl lead)=%d", len(_sections))
        for top_section in _sections:
            heading_stack = []
            current_text = []
            section_title = None
            subsection_title = None
            subsubsection_title = None
            subsection_index = 0  # Reset subsection index for each section

            for node in top_section.nodes:
                if isinstance(node, mwparserfromhell.nodes.Heading):
                    level = node.level
                    title = node.title.strip_code().strip()

                    if level == 2:
                        section_title = title
                        if skip_sections and any(section_title.lower() == s.lower() for s in skip_sections):
                            # Flush any pending text as previous section, then mark to skip until next level 2
                            heading_stack = []
                            current_text = []
                            section_title = None
                            subsection_title = None
                            subsubsection_title = None
                            continue
                        subsection_index = 0  # Reset subsection index for new section
                        subsection_title = None
                        subsubsection_title = None
                    elif level == 3:
                        subsection_index += 1  # Increment for new subsection
                        subsection_title = title
                        subsubsection_title = None
                    elif level == 4:
                        subsubsection_title = title

                    if current_text:
                        payloads.extend(self._chunk_payload(
                            " ".join(current_text), url,
                            section_title, subsection_title, subsubsection_title, 
                            section_index, subsection_index if level >= 3 else 0
                        ))
                        current_text = []
                        if settings.debug_verbose:
                            logger.debug("Wiki: flushed text at level=%d, section=%s, subsection=%s, subsection_idx=%s", 
                                      level, section_title, subsection_title, 
                                      subsection_index if level >= 3 else 0)
                else:
                    current_text.append(str(node))
            if settings.debug_verbose:
                try:
                    logger.debug("Wiki: trailing text len=%d", len("".join(current_text)))
                except Exception:
                    logger.debug("Wiki: trailing text present (len unknown)")
            if current_text:
                payloads.extend(self._chunk_payload(
                    "\n".join(current_text), url,
                    section_title, subsection_title, subsubsection_title, 
                    section_index, subsection_index if subsection_title else 0
                ))
            section_index += 1

        #logger.info("Wiki: parsed url=%s payloads=%d", url, len(payloads))
        return payloads

    def _chunk_payload(self, text, url, section, subsection, subsubsection, section_index, subsection_index=0):
        # DEBUG: trace the Surveys / *century* pipeline (temporary, safe to remove later)
        is_surveys_century = (
            settings.debug_verbose
            and section is not None
            and subsection is not None
            and isinstance(section, str)
            and isinstance(subsection, str)
            and "surveys" in section.lower()
            and "century" in subsection.lower()
        )
        if is_surveys_century:
            preview_in = (text or "")[:200].replace("\n", " ")
            logger.debug(
                "Chunk DEBUG (raw) section=%r subsection=%r len=%d preview=%r",
                section,
                subsection,
                len(text or ""),
                preview_in,
            )

        # 1) Strip wikitext markup
        text = mwparserfromhell.parse(text).strip_code().strip()
        if is_surveys_century:
            preview_after_strip = (text or "")[:200].replace("\n", " ")
            logger.debug(
                "Chunk DEBUG (after strip_code) len=%d preview=%r",
                len(text or ""),
                preview_after_strip,
            )

        if not text:
            return []

        # 2) Clean mediawiki-specific noise (File:, refs, etc.)
        text = clean_mediawiki_text(text)
        if is_surveys_century:
            preview_after_clean = (text or "")[:200].replace("\n", " ")
            logger.debug(
                "Chunk DEBUG (after clean_mediawiki_text) len=%d preview=%r",
                len(text or ""),
                preview_after_clean,
            )

        # 3) Split into chunks
        chunks = self.splitter.split_text(text)
        total_chunks = len(chunks)
        if is_surveys_century:
            logger.debug(
                "Chunk DEBUG (split) produced %d chunks for section=%r subsection=%r",
                total_chunks,
                section,
                subsection,
            )

        if section is None:
            section = "Lead"
            # logger.debug(f"[DEBUG] Assigning 'Lead' to section for URL: {url}")
        title = getattr(self, 'page_title', None)
        description = getattr(self, 'page_description', "")
        return [{
            "text": chunk,
            "section": section,
            "subsection": subsection,
            "subsubsection": subsubsection,
            "chunk_index": idx,
            "total_chunks": total_chunks,
            "url": url,
            "doc_type": "mediawiki",  # For backward compatibility
            "document_type": "mediawiki",  # New standard field
            "source": url,
            "section_index": section_index,
            "subsection_index": subsection_index if subsection_index is not None else 0,
            "title": title,
            "description": description
        } for idx, chunk in enumerate(chunks)]

    def _fetch_section_tree(self, title: str) -> List[Dict]:
        params = {
            "action": "parse",
            "format": "json",
            "prop": "sections",
            "page": title,
            "redirects": 1,
        }
        r = _wiki_get(params, self.api_url, self.user_agent)
        data = r.json()
        sections = data.get("parse", {}).get("sections", [])
        #logger.debug("Wiki: section tree count=%d for %s", len(sections), title)
        return sections

    def _fetch_section_html(self, title: str, section_index: int) -> str:
        params = {
            "action": "parse",
            "format": "json",
            "prop": "text",
            "page": title,
            "section": section_index,
            "disabletoc": 1,
            "redirects": 1,
        }
        r = _wiki_get(params, self.api_url, self.user_agent)
        data = r.json()
        text_block = data.get("parse", {}).get("text", {})
        # MediaWiki returns HTML under the key "*"
        html = text_block.get("*", "") if isinstance(text_block, dict) else ""
        #logger.debug("Wiki: fetched section %s html len=%d", section_index, len(html or ""))
        return html

    def _html_to_text(self, html: str) -> str:
        """Convert a section's HTML to clean visible text.
        Drops boilerplate via self.wiki_drop_selectors and normalizes whitespace.
        """
        soup = BeautifulSoup(html or "", "html.parser")
        # Drop noisy or non-prose elements
        drop_selectors = self.wiki_drop_selectors or []
        for sel in drop_selectors:
            for el in soup.select(sel):
                el.decompose()
        # Optional: drop ALL tables from the prose path. When enabled, tables will not
        # be flattened into normal section text chunks, and can be indexed only via the
        # structured table extraction path (`_emit_table_payloads`).
        if getattr(self, "wiki_drop_tables_from_prose", False):
            for tbl in soup.find_all("table"):
                tbl.decompose()
        # Visible text only (anchor TEXT preserved automatically)
        text = soup.get_text(" ")
        # Normalize whitespace
        text = re.sub(r"\u00a0", " ", text)
        text = re.sub(r"[\t ]+", " ", text)
        if self.wiki_preserve_paragraphs:
            # Keep paragraph breaks: collapse 3+ newlines to 2; convert single newlines to spaces
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        else:
            # Fully flatten
            text = text.replace("\n", " ")
        if settings.debug_verbose:
            logger.debug("Wiki: html_to_text len=%d", len(text.strip()))
        return text.strip()

    def _infobox_html_to_text(self, infobox) -> str:
        """
        Convert a MediaWiki infobox <table> into a compact key/value style text block.

        This is used only for a dedicated INFOBOX section so that we preserve
        structured summary data (e.g., Elevation, Prominence, Location) without
        polluting the main section prose. The regular _html_to_text() path
        still drops .infobox elements via self.wiki_drop_selectors.
        """
        if infobox is None:
            return ""

        lines: List[str] = []

        try:
            # Many infoboxes have a title in a <caption> or in a top-level <th>.
            caption = infobox.find("caption")
            if caption:
                title_text = caption.get_text(" ", strip=True)
                if title_text:
                    lines.append(title_text)

            # Walk table rows and extract simple "Label: Value" pairs.
            for row in infobox.find_all("tr"):
                header = row.find("th")
                data = row.find("td")
                # Title-only rows may have just <th> without <td>.
                if header and not data:
                    title_text = header.get_text(" ", strip=True)
                    if title_text:
                        lines.append(title_text)
                    continue

                if header and data:
                    label = header.get_text(" ", strip=True)
                    value = data.get_text(" ", strip=True)
                    if label and value:
                        lines.append(f"{label}: {value}")
                    continue
        except Exception:
            # On any parsing error, fall back to plain text extraction.
            raw = infobox.get_text(" ", strip=True)
            raw = re.sub(r"\u00a0", " ", raw)
            raw = re.sub(r"[\t ]+", " ", raw)
            return raw.strip()

        # --- Subject prefixing for Infobox ---
        # Prefix facts with the page subject for better embedding semantics.
        subject = getattr(self, "page_title", "").strip()
        if subject:
            prefixed_lines: List[str] = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                prefixed_lines.append(f"{subject} — {line}")
            lines = prefixed_lines
        # --- End subject prefixing for Infobox ---

        text = "\n".join(lines)
        text = re.sub(r"\u00a0", " ", text)
        text = re.sub(r"[\t ]+", " ", text)
        return text.strip()

    def _is_table_nested(self, table_tag) -> bool:
        """Return True if this <table> is inside another <table>."""
        try:
            return table_tag.find_parent("table") is not None
        except Exception:
            return False

    def _should_skip_table(self, table_tag) -> bool:
        """Heuristics to skip non-content / boilerplate tables."""
        try:
            classes = " ".join(table_tag.get("class", [])).lower()
            # Common MediaWiki boilerplate / navigation tables.
            skip_markers = [
                "navbox",
                "vertical-navbox",
                "metadata",
                "ambox",
                "toccolours",
                "infobox",  # infobox is handled separately
            ]
            return any(m in classes for m in skip_markers)
        except Exception:
            return False

    def _extract_table_headers(self, table_tag) -> List[str]:
        """Extract column headers from the first header row, if present."""
        headers: List[str] = []
        try:
            # Prefer the first row that contains <th> cells.
            for tr in table_tag.find_all("tr", recursive=True):
                ths = tr.find_all("th", recursive=False)
                if ths:
                    headers = [th.get_text(" ", strip=True) for th in ths]
                    break
            # Fall back: sometimes header cells are nested; try a non-recursive search per row.
            if not headers:
                for tr in table_tag.find_all("tr"):
                    ths = tr.find_all("th")
                    if ths:
                        headers = [th.get_text(" ", strip=True) for th in ths]
                        break
        except Exception:
            headers = []

        # Normalize blanks
        headers = [h.strip() for h in headers if h and h.strip()]
        return headers

    def _flatten_cell_text(self, cell_tag) -> str:
        """Return readable text for a cell, flattening any nested tables into plain text."""
        try:
            # Replace nested tables with their visible text to avoid exploding structure.
            for nested in cell_tag.find_all("table"):
                nested_text = nested.get_text(" ", strip=True)
                nested_text = re.sub(r"\s+", " ", nested_text).strip()
                nested.replace_with(f" {nested_text} ")
            text = cell_tag.get_text(" ", strip=True)
            text = re.sub(r"\s+", " ", text).strip()
            return text
        except Exception:
            try:
                return re.sub(r"\s+", " ", cell_tag.get_text(" ", strip=True)).strip()
            except Exception:
                return ""

    def _table_rows_to_chunks(
        self,
        table_tag,
        *,
        rows_per_chunk: int,
        section: Optional[str],
        subsection: Optional[str],
        subsubsection: Optional[str],
        page_title: Optional[str],
        table_index: int,
    ) -> List[str]:
        """Convert a table to a list of chunk texts (row-batched) with header/context prefix."""
        headers = self._extract_table_headers(table_tag)

        # Collect row dicts
        row_texts: List[str] = []
        trs = table_tag.find_all("tr")
        for tr in trs:
            # Skip pure header rows
            tds = tr.find_all("td", recursive=False)
            ths = tr.find_all("th", recursive=False)
            if not tds and ths:
                continue
            if not tds:
                # Fallback: if cells are nested, try a broader search
                tds = tr.find_all("td")
            if not tds:
                continue

            cells = [self._flatten_cell_text(td) for td in tds]
            cells = [c for c in cells if c]
            if not cells:
                continue

            # If we have headers and the widths match (or are close), use key:value pairs.
            if headers and abs(len(headers) - len(cells)) <= 1:
                pairs = []
                for i, val in enumerate(cells):
                    key = headers[i] if i < len(headers) else f"col_{i+1}"
                    pairs.append(f"{key}: {val}")
                row_texts.append(" | ".join(pairs))
            else:
                # Otherwise keep as a compact row line.
                row_texts.append(" | ".join(cells))

        if not row_texts:
            return []

        # Build prefix context
        prefix_lines: List[str] = []
        if page_title:
            prefix_lines.append(f"Document: {page_title}")
        if section:
            prefix_lines.append(f"Section: {section}")
        if subsection:
            prefix_lines.append(f"Subsection: {subsection}")
        if subsubsection:
            prefix_lines.append(f"Subsubsection: {subsubsection}")
        if headers:
            prefix_lines.append("Columns: " + " | ".join(headers))
        prefix_lines.append(f"Table index: {table_index}")
        prefix = "\n".join(prefix_lines).strip() + "\n\n"

        # Chunk rows in batches
        rows_per_chunk = max(1, int(rows_per_chunk or 12))
        chunks: List[str] = []
        for start in range(0, len(row_texts), rows_per_chunk):
            batch = row_texts[start : start + rows_per_chunk]
            chunks.append(prefix + "\n".join(f"- {r}" for r in batch))
        return chunks

    def _emit_table_payloads(
        self,
        html: str,
        *,
        url: str,
        section: Optional[str],
        subsection: Optional[str],
        subsubsection: Optional[str],
        section_index: int,
        subsection_index: int,
    ) -> List[Dict]:
        """Extract top-level content tables from HTML and emit additive payloads.

        This is intentionally additive and non-breaking: it does NOT replace prose
        extraction. It emits extra payloads with `content_type="table"`.
        """
        if not html:
            return []
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return []

        tables = soup.find_all("table")
        if not tables:
            return []

        out: List[Dict] = []
        table_counter = 0
        for tbl in tables:
            if self._is_table_nested(tbl):
                continue
            if self._should_skip_table(tbl):
                continue

            table_counter += 1
            table_idx = table_counter

            chunk_texts = self._table_rows_to_chunks(
                tbl,
                rows_per_chunk=self.wiki_table_rows_per_chunk,
                section=section,
                subsection=subsection,
                subsubsection=subsubsection,
                page_title=getattr(self, "page_title", None),
                table_index=table_idx,
            )
            if not chunk_texts:
                continue

            total = len(chunk_texts)
            for i, chunk in enumerate(chunk_texts):
                # Mirror the core payload schema but avoid re-splitting by tokens.
                out.append(
                    {
                        "text": chunk,
                        "section": section or "Lead",
                        "subsection": subsection,
                        "subsubsection": subsubsection,
                        "chunk_index": i,
                        "total_chunks": total,
                        "url": url,
                        "doc_type": "mediawiki",
                        "document_type": "mediawiki",
                        "source": url,
                        "section_index": section_index,
                        "subsection_index": subsection_index if subsection_index is not None else 0,
                        "title": getattr(self, "page_title", None),
                        "description": getattr(self, "page_description", ""),
                        # Table-specific additive metadata (safe for downstream to ignore)
                        "content_type": "table",
                        "table_index": table_idx,
                        "table_chunk_index": i,
                        "table_total_chunks": total,
                    }
                )

        return out

    def _truncate_html_at_first_subheading(self, html: str, max_level: int = 2) -> str:
        """
        In Action API mode, the HTML for a given section level (e.g. an h2 section)
        often includes the full content of its deeper subsections (h3, h4, ...).
        This helper trims the HTML at the first heading deeper than `max_level`,
        so that parent sections don't "consume" the text of their child sections.

        For example:
          - max_level=2 → keep only content up to the first h3/h4/h5/h6.
          - max_level=3 → keep only content up to the first h4/h5/h6.
        """
        if not html:
            return html
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            # If parsing fails for any reason, fall back to the original HTML.
            return html

        # Headings deeper than max_level, e.g. ["h3", "h4", "h5", "h6"] for max_level=2.
        deeper_tags = [f"h{lvl}" for lvl in range(max_level + 1, 7)]
        first_sub = soup.find(deeper_tags)
        if not first_sub:
            # No deeper heading present; nothing to trim.
            return html

        # Remove the first deeper heading and everything that follows it at the same
        # root level. This preserves the parent section's own intro content while
        # allowing child sections to be fetched and chunked separately.
        for sib in list(first_sub.next_siblings):
            sib.extract()
        first_sub.extract()

        return str(soup)

    def _rest_base_from_api_url(self) -> str:
        """Derive REST base (scheme://host) from the configured Action API URL."""
        try:
            parsed = urlparse(self.api_url)
            return f"{parsed.scheme}://{parsed.netloc}"
        except Exception:
            return "https://en.wikipedia.org"

    def _fetch_parsoid_html(self, title: str) -> str:
        """Fetch Parsoid-rendered HTML for a page title via RESTBase."""
        #logger.debug("Wiki: fetching parsoid HTML for title=%s", title)
        base = self._rest_base_from_api_url()
        rest_url = f"{base}/api/rest_v1/page/html/{title.replace(' ', '_')}"
        headers = {"User-Agent": self.user_agent, "Accept": "text/html"}
        r = _wiki_session.get(rest_url, headers=headers, timeout=DEFAULT_TIMEOUT)
        #logger.debug("Wiki: parsoid HTTP %s for %s", r.status_code, rest_url)
        r.raise_for_status()
        return r.text

    def _walk_parsoid_sections(self, html: str, base_url: str, skip_sections: Optional[List[str]]) -> List[Dict]:
        """Walk Parsoid HTML, group content by section markers, and emit payloads.
        """
        #logger.debug("Wiki: walk parsoid sections start")
        soup = BeautifulSoup(html or "", "html.parser")
        root = soup.select_one("#mw-content-text") or soup

        payloads: List[Dict] = []
        section_index_counter = 0  # Lead = 0; first h2 = 1

        # Optional: extract the first infobox as a dedicated INFOBOX section.
        # We do this BEFORE the main section walk, so that the regular prose
        # extraction can still safely drop `.infobox` via self.wiki_drop_selectors
        # without losing the structured summary.
        try:
            infobox_chunks: List[Dict] = []
            infobox = root.select_one(".infobox")
            if infobox is not None:
                infobox_text = self._infobox_html_to_text(infobox)
                if infobox_text:
                    infobox_chunks = self._chunk_payload(
                        infobox_text,
                        base_url,
                        section="INFOBOX",
                        subsection=None,
                        subsubsection=None,
                        section_index=0,
                        subsection_index=0,
                    )
        except Exception as infobox_exc:
            logger.debug("Wiki: failed to extract INFOBOX via parsoid: %s", infobox_exc)
            infobox_chunks = []

        # Parsoid HTML is structured as <section data-mw-section-id="..."> blocks.
        # We treat section-id 0 as Lead, and all subsequent section-ids as
        # content sections keyed by their first heading (h2–h6).
        sections = root.find_all("section", attrs={"data-mw-section-id": True})
        if not sections:
            # If Parsoid sections are not present, raise so the caller can fall
            # back to the action_api flow.
            raise RuntimeError("Parsoid HTML does not contain section markers")
        # DEBUG: Parsoid section inventory (safe to remove after troubleshooting Everest/Surveys).
        if settings.debug_verbose:
            try:
                logger.debug(
                    "Wiki: parsoid found %d sections for base_url=%s",
                    len(sections),
                    base_url,
                )
                for sec in sections:
                    sec_id_dbg = sec.get("data-mw-section-id")
                    heading_dbg = sec.find(["h2", "h3", "h4", "h5", "h6"])
                    heading_txt_dbg = (
                        heading_dbg.get_text(" ").strip() if heading_dbg else "<no heading>"
                    )
                    logger.debug(
                        "Wiki: parsoid section id=%s heading=%s",
                        sec_id_dbg,
                        heading_txt_dbg[:120],
                    )
            except Exception as dbg_exc:
                logger.debug("Wiki: parsoid debug section inventory failed: %s", dbg_exc)

        # Handle lead (data-mw-section-id="0") if present.
        lead_section = root.find("section", attrs={"data-mw-section-id": "0"})
        if lead_section is not None:
            lead_html = "".join(str(child) for child in lead_section.contents)
            lead_text = self._html_to_text(lead_html)
            if lead_text and (not skip_sections or not any(s.lower() == "lead" for s in skip_sections)):
                payloads.extend(self._chunk_payload(
                    lead_text,
                    base_url,
                    section="Lead",
                    subsection=None,
                    subsubsection=None,
                    section_index=0,
                ))
        # Attach INFOBOX payloads (if any) after Lead, but before other sections.
        if "infobox_chunks" in locals() and infobox_chunks:
            payloads.extend(infobox_chunks)

        current_h2 = None
        current_h3 = None
        subsection_index_counter = 0
        current_subsection_index = None
        current_level = None  # Track last explicit heading level for heading-less sections

        # Walk non-lead sections in order of their data-mw-section-id
        def _section_id(sec) -> int:
            try:
                return int(sec.get("data-mw-section-id", "0"))
            except ValueError:
                return 0

        for sec in sorted(sections, key=_section_id):
            sec_id = sec.get("data-mw-section-id")
            if sec_id == "0":
                continue  # Lead already handled

            # Detect heading inside this <section>
            heading = sec.find(["h2", "h3", "h4", "h5", "h6"])
            if heading is not None:
                # Normal case: section starts with a heading
                try:
                    level = int(heading.name[1])
                except (TypeError, ValueError):
                    continue
                title = heading.get_text(" ").strip()
                anchor = heading.get("id") or sec.get("id") or ""
                current_level = level  # Remember for heading-less continuation sections
            else:
                # Heading-less section: treat as continuation of previous heading
                if current_h2 is None and current_h3 is None:
                    continue  # No context to attach this to
                level = current_level if current_level is not None else 2
                title = current_h3 or current_h2 or ""
                anchor = sec.get("id") or ""

            # Skip filtered h2 sections
            if level == 2 and skip_sections and any(title.lower() == s.lower() for s in skip_sections):
                continue

            # Update heading state
            if level == 2:
                current_h2 = title
                current_h3 = None
                section_index_counter += 1
                subsection_index_counter = 0
                current_subsection_index = None
            elif level == 3:
                current_h3 = title
                subsection_index_counter += 1
                current_subsection_index = subsection_index_counter
            else:
                # deeper levels reuse current H2/H3 context
                pass

            # Collect content for this section: everything inside the <section>
            # except the heading node itself and any nested subsection <section>
            # blocks (those have their own data-mw-section-id and will be
            # processed separately).
            collected_html = []
            for child in sec.contents:
                if child is heading:
                    continue
                # Skip nested subsection sections entirely to avoid duplicating
                # their content under the parent heading.
                if getattr(child, "name", None) == "section" and child.has_attr("data-mw-section-id"):
                    continue
                collected_html.append(str(child))

            collected_html_str = "".join(collected_html)
            # Compute titles / indices for this entry (used by both prose and optional table payloads)
            if level == 2:
                sec_title = current_h2
                sub_title = None
                subsub_title = None
                sec_index = section_index_counter
                subsec_index = 0
            elif level == 3:
                sec_title = current_h2
                sub_title = current_h3
                subsub_title = None
                sec_index = section_index_counter
                subsec_index = current_subsection_index
            else:  # level >= 4
                sec_title = current_h2
                sub_title = current_h3
                subsub_title = title
                sec_index = section_index_counter
                subsec_index = current_subsection_index

            section_url = f"{base_url}#{anchor}" if anchor else base_url

            # Additive table indexing (optional): emit separate payloads for tables
            # in list-style pages so row fragments carry column/header context.
            if getattr(self, "wiki_index_tables", False):
                try:
                    payloads.extend(
                        self._emit_table_payloads(
                            collected_html_str,
                            url=section_url,
                            section=sec_title,
                            subsection=sub_title,
                            subsubsection=subsub_title,
                            section_index=sec_index,
                            subsection_index=subsec_index if subsec_index is not None else 0,
                        )
                    )
                except Exception as table_exc:
                    logger.debug("Wiki: table extraction failed for url=%s: %s", section_url, table_exc)

            text = self._html_to_text(collected_html_str)

            # DEBUG: show a snippet of extracted text for Surveys / *century* to
            # understand why some subsections (e.g., "20th century") might be
            # missing or empty. Safe to remove after troubleshooting.
            if settings.debug_verbose and ("Surveys" in title or "century" in title):
                preview = (text or "")[:200].replace("\n", " ")
                logger.debug(
                    "Wiki: parsoid DEBUG text preview for %s (sec_id=%s): %r",
                    title,
                    sec_id,
                    preview,
                )
                if not text:
                    logger.debug(
                        "Wiki: parsoid DEBUG text is empty for %s (sec_id=%s); section will be skipped.",
                        title,
                        sec_id,
                    )

            if not text:
                continue

            payloads.extend(self._chunk_payload(
                text,
                section_url,
                sec_title,
                sub_title,
                subsub_title,
                sec_index,
                subsec_index,
            ))

        logger.info("Wiki: parsoid sections payloads=%d", len(payloads))
        return payloads

    def parse_from_title(self, title: str, url: Optional[str] = None, skip_sections: Optional[List[str]] = None) -> List[Dict]:
        """
        Build payloads using MediaWiki Action API per-section HTML (clean text, preserved hierarchy).
        """
        #logger.debug("Wiki: parse_from_title title=%s mode=%s", title, self.wiki_mode)
        # Resolve canonical URL and description first (keeps previous behavior for metadata)
        if url is None:
            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        # Fetch description via query API (keeps 'description' field)
        meta_params = {
            "action": "query",
            "format": "json",
            "prop": "description",
            "titles": title,
            "formatversion": "2",
            "redirects": 1,
        }
        try:
            meta_resp = _wiki_get(meta_params, self.api_url, self.user_agent)
            meta_data = meta_resp.json()
            pages = meta_data.get("query", {}).get("pages", [])
            if not pages or "missing" in pages[0]:
                raise ValueError(f"Page titled '{title}' not found.")
            self.page_title = pages[0].get("title", title)
            self.page_description = pages[0].get("description") or ""
        except Exception:
            # Fall back gracefully
            self.page_title = title
            self.page_description = ""

        # Dispatch by mode
        if self.wiki_mode == "parsoid":
            # Try Parsoid first; if it fails (e.g., REST not enabled on a corporate wiki),
            # fall back to the existing action_api per-section behavior.
            try:
                logger.debug("Wiki: fetching parsoid HTML for title=%s url=%s", self.page_title, url)
                html_doc = self._fetch_parsoid_html(self.page_title)
                payloads = self._walk_parsoid_sections(html_doc, url, skip_sections)
                logger.info("Wiki: parsed via parsoid url=%s payloads=%d", url, len(payloads))
                return payloads
            except Exception as e:
                logger.warning(
                    "Wiki: parsoid mode failed for title=%s url=%s; falling back to action_api. Error: %s",
                    self.page_title,
                    url,
                    e,
                )
                # Continue into the action_api section tree flow below.

        # Default: action_api per-section flow (existing behavior)
        sections = self._fetch_section_tree(self.page_title)
        #logger.debug("Wiki: fetched %d sections via action_api for %s", len(sections), self.page_title)

        payloads: List[Dict] = []
        table_payloads_all: List[Dict] = []

        # Lead content (section=0) + optional INFOBOX extraction from the lead HTML.
        infobox_chunks: List[Dict] = []
        lead_html = self._fetch_section_html(self.page_title, 0)
        try:
            # Attempt to extract the first infobox from the lead HTML as a dedicated INFOBOX section.
            soup_lead = BeautifulSoup(lead_html or "", "html.parser")
            infobox = soup_lead.select_one(".infobox")
            if infobox is not None:
                infobox_text = self._infobox_html_to_text(infobox)
                if infobox_text:
                    infobox_chunks = self._chunk_payload(
                        infobox_text,
                        url,
                        section="INFOBOX",
                        subsection=None,
                        subsubsection=None,
                        section_index=0,
                        subsection_index=0,
                    )
                # Remove the infobox from the lead HTML so it is not duplicated in the main Lead text.
                infobox.decompose()
                lead_html = str(soup_lead)
        except Exception as infobox_exc:
            logger.debug(
                "Wiki: failed to extract INFOBOX via action_api lead for title=%s url=%s: %s",
                self.page_title,
                url,
                infobox_exc,
            )
            infobox_chunks = []

        lead_text = self._html_to_text(lead_html)
        if lead_text and (not skip_sections or not any(s.lower() == "lead" for s in skip_sections)):
            payloads.extend(self._chunk_payload(
                lead_text,
                url,
                section="Lead",
                subsection=None,
                subsubsection=None,
                section_index=0,
            ))

        # Attach INFOBOX payloads (if any) after Lead, but before other sections.
        if infobox_chunks:
            payloads.extend(infobox_chunks)

        # Collect per-section text entries first; we'll de-duplicate parent/child
        # overlaps (e.g., Geography vs. Hydrology) in a second pass before
        # chunking. This keeps h2 sections from "swallowing" their h3+ children
        # when the Action API returns overlapping prose.
        section_entries: List[Dict] = []

        current_h2 = None
        current_h3 = None
        section_index_counter = 0  # Lead = 0; first h2 = 1
        subsection_index_counter = 0
        current_subsection_index = None

        for sec in sections:
            try:
                level = int(sec.get("level", 0))
            except (TypeError, ValueError):
                level = 0
            line = (sec.get("line") or "").strip()
            anchor = sec.get("anchor") or ""
            idx = sec.get("index")

            if level == 2 and skip_sections and any(line.lower() == s.lower() for s in skip_sections):
                current_h2 = None
                current_h3 = None
                subsection_index_counter = 0
                current_subsection_index = None
                continue

            if level == 2:
                current_h2 = line
                current_h3 = None
                section_index_counter += 1
                subsection_index_counter = 0
                current_subsection_index = None
            elif level == 3:
                current_h3 = line
                subsection_index_counter += 1
                current_subsection_index = subsection_index_counter
            # level >= 4 → handled below

            if not idx:
                continue
            html = self._fetch_section_html(self.page_title, int(idx))

            # In Action API mode, an h2 section's HTML typically includes the full
            # content of its h3+ subsections. To mirror Parsoid behavior (where we
            # do not duplicate child content under the parent), trim the HTML at
            # the first deeper heading for h2 and h3 levels.
            if level == 2:
                html = self._truncate_html_at_first_subheading(html, max_level=2)
            elif level == 3:
                html = self._truncate_html_at_first_subheading(html, max_level=3)

            # Compute titles / indices / URL for this entry (used by prose and optional table payloads)
            if level == 2:
                section_title = current_h2
                subsection_title = None
                subsubsection_title = None
                sec_index = section_index_counter
                subsec_index = 0
            elif level == 3:
                section_title = current_h2
                subsection_title = current_h3
                subsubsection_title = None
                sec_index = section_index_counter
                subsec_index = current_subsection_index
            else:  # >= 4
                section_title = current_h2
                subsection_title = current_h3
                subsubsection_title = line
                sec_index = section_index_counter
                subsec_index = current_subsection_index

            section_url = f"{url}#{anchor}" if anchor else url

            # Additive table indexing (optional) for Action API sections.
            if getattr(self, "wiki_index_tables", False):
                try:
                    table_payloads_all.extend(
                        self._emit_table_payloads(
                            html,
                            url=section_url,
                            section=section_title,
                            subsection=subsection_title,
                            subsubsection=subsubsection_title,
                            section_index=sec_index,
                            subsection_index=subsec_index if subsec_index is not None else 0,
                        )
                    )
                except Exception as table_exc:
                    logger.debug("Wiki: action_api table extraction failed for url=%s: %s", section_url, table_exc)

            text = self._html_to_text(html)
            if not text:
                continue

            section_entries.append(
                {
                    "level": level,
                    "text": text,
                    "url": section_url,
                    "section_title": section_title,
                    "subsection_title": subsection_title,
                    "subsubsection_title": subsubsection_title,
                    "section_index": sec_index,
                    "subsection_index": subsec_index,
                }
            )

        # Second pass: remove child subsection text from their parent sections
        # when the Action API has returned overlapping prose. For example, this
        # keeps the h2 "Geography" section from also containing the full text
        # of its h3 "Hydrology" subsection.
        for parent in section_entries:
            parent_level = parent.get("level", 0)
            if parent_level not in (2, 3):
                continue
            for child in section_entries:
                if child is parent:
                    continue
                if child.get("section_index") != parent.get("section_index"):
                    continue
                child_level = child.get("level", 0)
                if child_level <= parent_level:
                    continue
                child_text = (child.get("text") or "").strip()
                if not child_text:
                    continue
                parent_text = parent.get("text") or ""
                if not parent_text:
                    continue

                # Many Action API sections for h2 headings (e.g. "Geography")
                # contain the full body of their h3 children (e.g. "Hydrology")
                # *without* the child heading, while the child section's text
                # includes the heading label (e.g. "Hydrology [ edit ] ...").
                # This means `child_text` will not be an exact substring of the
                # parent, even though most of the body is duplicated.
                #
                # To handle this, we:
                #   1) Try an exact match on the full child text.
                #   2) Try a "body" variant with any leading "Heading [ edit ]"
                #      prefix stripped off.
                #   3) As a last resort, try matching the tail of the body
                #      (last N chars), which is usually enough to uniquely
                #      identify the duplicated prose within the parent.
                #
                # Each match simply removes the duplicated span from the parent
                # text; any surrounding whitespace clean-up is handled later by
                # the splitter / normalisation.
                def _strip_leading_heading_prefix(text: str, child_dict: Dict) -> str:
                    t = text.lstrip()
                    sub = (child_dict.get("subsection_title") or "").strip()
                    if sub and t.startswith(sub):
                        # Remove the leading subsection title and any trailing
                        # "[ edit ]" boilerplate if present.
                        t = t[len(sub):].lstrip()
                        if t.startswith("[ edit ]"):
                            t = t[len("[ edit ]"):].lstrip()
                    return t

                replaced = False

                # 1) Exact full-text match
                if child_text and child_text in parent_text:
                    parent["text"] = parent_text.replace(child_text, " ")
                    replaced = True
                else:
                    # 2) Try the body-only variant without heading prefix
                    body = _strip_leading_heading_prefix(child_text, child)
                    if body and body in parent_text:
                        parent["text"] = parent_text.replace(body, " ")
                        replaced = True
                    else:
                        # 3) Tail heuristic: use the last N characters of the body
                        # (typically enough to capture the main duplicated prose
                        # while being robust to minor leading differences).
                        if body:
                            candidate = body
                        else:
                            candidate = child_text
                        tail_len = 400
                        if len(candidate) > 80:  # avoid ultra-short, noisy matches
                            tail = candidate[-min(tail_len, len(candidate)):]
                            if tail in parent_text:
                                parent["text"] = parent_text.replace(tail, " ")
                                replaced = True

                if settings.debug_verbose and replaced:
                    logger.debug(
                        "Wiki: action_api de-duplicated child level=%s (%r) from parent level=%s (%r) for section_index=%s",
                        child_level,
                        child.get("subsection_title") or child.get("section_title"),
                        parent_level,
                        parent.get("subsection_title") or parent.get("section_title"),
                        parent.get("section_index"),
                    )

        # Now chunk all section entries into payloads.
        for entry in section_entries:
            text = (entry.get("text") or "").strip()
            if not text:
                continue
            payloads.extend(
                self._chunk_payload(
                    text,
                    entry.get("url") or url,
                    entry.get("section_title"),
                    entry.get("subsection_title"),
                    entry.get("subsubsection_title"),
                    entry.get("section_index", 0),
                    entry.get("subsection_index", 0),
                )
            )

        # Append any additive table payloads last (non-breaking for prose indexing).
        if "table_payloads_all" in locals() and table_payloads_all:
            payloads.extend(table_payloads_all)

        #logger.info("Wiki: parsed via action_api url=%s payloads=%d", url, len(payloads))
        return payloads

    def parse_from_url(self, url: str, skip_sections: Optional[List[str]] = None) -> List[Dict]:
        """
        Extract the page title from a Wikipedia URL and parse the page.
        """
        #logger.debug("Wiki: parse_from_url %s", url)
        logger.debug("Wiki: parse_from_url mode=%s", self.wiki_mode)
        parsed_url = urlparse(url)
        path = parsed_url.path
        if not path.startswith("/wiki/"):
            raise ValueError("URL does not appear to be a standard Wikipedia article URL.")
        title_encoded = path[len("/wiki/"):]
        title = unquote(title_encoded).replace('_', ' ')
        return self.parse_from_title(title, url, skip_sections=skip_sections)
