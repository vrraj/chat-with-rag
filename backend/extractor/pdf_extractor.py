import re
from typing import List, Dict, Optional, Tuple
import httpx
from io import BytesIO
from .splitters import TextSplitter
import html

import logging
from backend.core.config import settings

logger = logging.getLogger(__name__)


import fitz  # PyMuPDF
HAS_PYMUPDF = True

# --- pymupdf4llm optional import and feature flag ---
import os
import tempfile

try:
    import pymupdf4llm  # type: ignore
    HAS_PYMUPDF4LLM = True
except Exception:
    pymupdf4llm = None  # type: ignore
    HAS_PYMUPDF4LLM = False


HEADING_NUM_RE = re.compile(r"^(\d+(?:\.\d+){0,2})\s+(.+)$")

# Lines that look like table-of-contents entries, e.g., "1.2 Title .... 12"
TOC_RE = re.compile(r"^\s*\d+(?:\.\d+)*\s+.+?\.{2,}\s*\d+\s*$")

# Bold-only markdown headings (e.g., "**Hydrology**") that pymupdf4llm emits for
# some subsection titles in Wikipedia-style PDFs.
BOLD_HEADING_RE = re.compile(r"^\*\*(?P<title>[^*][^*]{0,100})\*\*\s*$")

# Superscript and citation patterns used to drop tiny reference spans at extraction time

SUP_UNI_RE = re.compile(r'^[\u00B9\u00B2\u00B3\u2070-\u2079]+$')  # ¹²³⁴⁵⁶⁷⁸⁹⁰
BRACKET_REF = re.compile(r'^\[(?:\d{1,3})(?:[\s,–-]+\d{1,3})*\]$')
SHORT_DIGITS = re.compile(r'^\d{1,3}$')

# --- Infobox extraction helpers (generic) ---
#
# We intentionally avoid a fixed list of labels.
# Many PDFs (including Wikipedia exports) represent sidebars / infoboxes as
# small 2-column tables (Label | Value). For other PDFs the labels differ
# (SEC summaries, medical demographics, etc.).
#
# IMPORTANT: Do NOT treat large multi-column data tables as INFOBOX.
# Doing so can cause tables to be mislabeled as INFOBOX and skipped by
# downstream filters.

def _table_md_column_count(table_md: str) -> int:
    """Return the number of columns in a markdown table header row."""
    if not table_md:
        return 0
    first = table_md.splitlines()[0].strip()
    if not first.startswith("|"):
        return 0
    # Remove leading/trailing pipes and split.
    parts = [p.strip() for p in first.strip("|").split("|")]
    return len([p for p in parts if p or p == ""])  # keep empty cells


def _clean_table_cell(text: str) -> str:
    """Clean a markdown table cell into plain text."""
    if text is None:
        return ""
    s = str(text)
    # Convert common HTML line breaks used by pymupdf4llm.
    s = s.replace("<br>", " ").replace("<br/>", " ").replace("<br />", " ")
    # Drop strike-through tildes that sometimes appear as OCR / markdown artifacts.
    s = s.replace("~~", "")
    # Collapse whitespace.
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _looks_like_kv_infobox_table(table_md: str) -> bool:
    """Heuristic: True if a markdown table looks like a 2-column Label/Value infobox."""
    if not table_md:
        return False
    lines = [ln for ln in table_md.splitlines() if ln.strip()]
    if len(lines) < 3:
        return False

    col_count = _table_md_column_count(table_md)
    # Only 2-column tables are considered INFOBOX candidates.
    # (Some PDFs may have a 3rd column for units; handle that later if needed.)
    if col_count != 2:
        return False

    # Ensure we have at least 2 data rows.
    data_rows = [ln for ln in lines[2:] if ln.strip().startswith("|")]
    if len(data_rows) < 2:
        return False

    # Label column should be relatively short and non-numeric for most rows.
    labelish = 0
    for r in data_rows[:10]:
        cells = [c.strip() for c in r.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        label = _clean_table_cell(cells[0])
        val = _clean_table_cell(cells[1])
        if not label or not val:
            continue
        if len(label) <= 40 and not re.fullmatch(r"[\d.,%-]+", label):
            labelish += 1

    return labelish >= 2


def _parse_kv_infobox_table(table_md: str) -> Dict[str, str]:
    """Parse a 2-column markdown table into key/value fields."""
    result: Dict[str, str] = {}
    if not table_md:
        return result

    lines = [ln for ln in table_md.splitlines() if ln.strip()]
    if len(lines) < 3:
        return result

    # Data rows start after header + separator.
    for r in lines[2:]:
        if not r.strip().startswith("|"):
            continue
        cells = [c.strip() for c in r.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        k = _clean_table_cell(cells[0]).strip(" :;,-")
        v = _clean_table_cell(cells[1]).strip(" :;,-")
        if k and v:
            # If the same key repeats, append (rare but happens in messy PDFs).
            if k in result:
                result[k] = f"{result[k]} ; {v}"
            else:
                result[k] = v

    return result

# NOTE: All toggles are per-instance init parameters. If you prefer global control,
# you can pass them from a Settings object, but keeping them here makes behavior
# explicit at the call site and easier to A/B.
class PDFExtractor:
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        timeout: int = 20,
        *,
        # --- Normalization toggles (lightweight, per-instance control) ---
        dehyphenate: bool = True,
        preserve_lists: bool = True,
        list_threshold: float = 0.6,
        paragraph_break_policy: str = "double-newline",  # or "strict" to flatten paragraphs
        normalize_quotes: bool = True,
        html_entity_unescape: bool = True,
        # --- Heuristics toggles ---
        heading_min_len: int = 3,
        toc_filter: bool = False,
        strip_reference_markers: bool = True,
        strip_unicode_superscripts: bool = True,
        drop_small_superscripts: bool = True,
        superscript_size_ratio: float = 0.8,
        # --- Layout toggles (placeholders; add implementation later) ---
        header_footer_filter: bool = False,
        multicolumn_sort: bool = False,
        # --- Section filtering ---
        skip_sections: Optional[List[str]] = None,
        # --- Table extraction / chunking (PDF) ---
        pdf_index_tables: bool = True,
        pdf_table_rows_per_chunk: int = 12,
        pdf_repeat_table_header: bool = True,
        pdf_table_min_rows: int = 2,
        pdf_drop_tables_from_prose: bool = False,
    ):
        """PDF text extractor with configurable cleanup heuristics.

        Parameters
        ----------
        chunk_size, chunk_overlap, timeout : core splitter/network options.
        dehyphenate : If True, join words split across line wraps (e.g., "Kili-\nmanjaro" → "Kilimanjaro").
        preserve_lists : If True, keep bullets/numbered lists one item per line when a paragraph is list-dominant.
        list_threshold : Fraction (0–1). If ≥ this fraction of lines in a paragraph look like bullets/numbers, treat as a list.
        paragraph_break_policy : "double-newline" preserves blank-line paragraph breaks; "strict" flattens paragraphs to a single stream.
        normalize_quotes : If True, map curly quotes/dashes (’‘“”–—) to straight equivalents (' " -).
        html_entity_unescape : If True, unescape HTML/XML entities (e.g., &amp; → &).
        heading_min_len : Minimum alpha length for heading detection via all-caps/size heuristics (reduces false headings).
        toc_filter : If True, skip TOC-like lines (e.g., "1.2 Title .... 12") during parsing.
        strip_reference_markers : If True, remove bracketed numeric citations like [2], [12, 14].
        strip_unicode_superscripts : If True, remove standalone superscript numerals (¹²³⁴⁵⁶⁷⁸⁹⁰) that act as citation markers.
        drop_small_superscripts : If True, drop tiny superscript spans (e.g., citation digits) during PyMuPDF extraction.
        superscript_size_ratio : A span is considered tiny if span.size <= line_max_size * this ratio (default 0.8).
        header_footer_filter : Placeholder toggle to drop repeating header/footer lines across pages (not implemented yet).
        multicolumn_sort : Placeholder toggle to re-order text for multi-column layouts (not implemented yet).
        """
        self.splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_manual_splitter=False,
        )
        self.timeout = timeout

        # Store toggles
        self.dehyphenate = dehyphenate
        self.preserve_lists = preserve_lists
        self.list_threshold = float(list_threshold)
        self.paragraph_break_policy = paragraph_break_policy
        self.heading_min_len = int(heading_min_len)
        self.toc_filter = toc_filter
        self.header_footer_filter = header_footer_filter
        self.multicolumn_sort = multicolumn_sort
        self.normalize_quotes = normalize_quotes
        self.html_entity_unescape = html_entity_unescape
        self.strip_reference_markers = strip_reference_markers
        self.strip_unicode_superscripts = strip_unicode_superscripts
        self.drop_small_superscripts = drop_small_superscripts
        self.superscript_size_ratio = float(superscript_size_ratio)

        # Table extraction controls (used by pymupdf4llm backend; safe no-ops otherwise)
        self.pdf_index_tables = bool(pdf_index_tables)
        self.pdf_table_rows_per_chunk = int(pdf_table_rows_per_chunk)
        self.pdf_repeat_table_header = bool(pdf_repeat_table_header)
        self.pdf_table_min_rows = int(pdf_table_min_rows)
        self.pdf_drop_tables_from_prose = bool(pdf_drop_tables_from_prose)

        # Normalized list of section titles to skip entirely (case-insensitive)
        # e.g., ["References", "See also"]. These are matched against the
        # "section" name as extracted from headings / markdown.
        self.skip_sections = [s.strip().casefold() for s in (skip_sections or []) if s and s.strip()]
        logger.debug("PDF: skip_sections=%s", self.skip_sections)
    def _strip_markdown_tables(self, md: str) -> str:
        """Remove GitHub-style markdown table blocks from markdown text.

        This is used only when `pdf_drop_tables_from_prose=True` to avoid double-indexing
        tables (once as prose, once as structured table payloads).

        We keep this conservative: only removes blocks that look like a table header row
        followed by a separator row (|---|---|...).
        """
        if not md:
            return ""

        lines = md.splitlines()
        out: List[str] = []
        i = 0

        def is_table_sep(line: str) -> bool:
            s = line.strip()
            if not s.startswith("|"):
                return False
            # Accept standard and alignment separators like:
            # |---|---|, |:---|---:|, |:---:|---|
            return bool(re.match(r"^\|\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$", s))

        def is_table_row(line: str) -> bool:
            s = line.strip()
            return s.startswith("|") and "|" in s[1:]

        while i < len(lines):
            # Detect a markdown table header + separator
            if i + 1 < len(lines) and is_table_row(lines[i]) and is_table_sep(lines[i + 1]):
                # Skip until the table block ends
                i += 2
                while i < len(lines) and is_table_row(lines[i]):
                    i += 1
                # Also skip trailing blank lines after table
                while i < len(lines) and not lines[i].strip():
                    i += 1
                continue

            out.append(lines[i])
            i += 1

        return "\n".join(out)

    def _extract_markdown_tables(self, md: str) -> List[str]:
        """Extract GitHub-style markdown table blocks from markdown text.

        This is used as a *fallback* when the pymupdf4llm chunk metadata does not
        provide table objects (or when table objects fail to convert).

        Returns a list of markdown strings, each containing a full table block
        (header + separator + rows).
        """
        if not md:
            return []

        lines = md.splitlines()
        tables: List[str] = []
        i = 0

        def is_table_sep(line: str) -> bool:
            s = line.strip()
            if not s.startswith("|"):
                return False
            return bool(re.match(r"^\|\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$", s))

        def is_table_row(line: str) -> bool:
            s = line.strip()
            return s.startswith("|") and "|" in s[1:]

        while i < len(lines):
            # Detect a markdown table header + separator
            if i + 1 < len(lines) and is_table_row(lines[i]) and is_table_sep(lines[i + 1]):
                start = i
                i += 2
                while i < len(lines) and is_table_row(lines[i]):
                    i += 1
                block = "\n".join(lines[start:i]).strip()
                if block:
                    tables.append(block)
                continue
            i += 1

        return tables

    def _chunk_markdown_table_by_rows(self, table_md: str) -> List[str]:
        """Chunk a markdown table into multiple markdown tables by row count.

        Expected format:
          |h1|h2|
          |---|---|
          |r1|r1|
          ...

        Returns a list of markdown strings, each preserving the header when
        `self.pdf_repeat_table_header=True`.
        """
        if not table_md:
            return []

        lines = [ln for ln in table_md.splitlines() if ln.strip()]
        if len(lines) < 3:
            return []

        header = lines[0]
        sep = lines[1]
        rows = lines[2:]

        # Basic validation: must look like a markdown table
        if not header.strip().startswith("|") or "|" not in header.strip()[1:]:
            return []
        if not sep.strip().startswith("|") or "---" not in sep:
            return []

        # Enforce minimum row count
        if len(rows) < max(0, self.pdf_table_min_rows):
            return []

        n = max(1, int(self.pdf_table_rows_per_chunk))
        chunks: List[str] = []
        for i in range(0, len(rows), n):
            block_rows = rows[i:i + n]
            if self.pdf_repeat_table_header:
                chunks.append("\n".join([header, sep] + block_rows))
            else:
                chunks.append("\n".join(block_rows))
        return chunks
    def _should_skip_section(
        self,
        section: Optional[str],
        subsection: Optional[str] = None,
        subsubsection: Optional[str] = None,
    ) -> bool:
        """Return True if any of the heading names are in the skip list.

        We check subsection and subsubsection first so that a page-wide section
        like "Mount Whitney" does not get skipped just because a specific
        subsection such as "References" should be ignored.
        """
        if not self.skip_sections:
            return False

        for name in (subsection, subsubsection, section):
            if not name:
                continue
            if name.strip().casefold() in self.skip_sections:
                return True
        return False

    def _fetch(self, url: str) -> bytes:
        try:
            #logger.debug("PDF: fetching %s", url)
            with httpx.Client(timeout=self.timeout) as client:
                r = client.get(url)
                r.raise_for_status()
                #logger.debug("PDF: HTTP %s for %s", r.status_code, url)
                return r.content
        except Exception as e:
            logger.exception("PDF: fetch failed for %s: %s", url, e)
            raise

    def _read_pages_pymupdf(self, pdf_bytes: bytes) -> Tuple[Optional[str], List[Tuple[int, List[Dict]]], List[str], List[str]]:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        title = doc.metadata.get("title") if doc.metadata else None
        #logger.debug("PDF: opened via PyMuPDF pages=%d title=%s", len(doc), title)
        pages = []
        for i, page in enumerate(doc):
            if settings.debug_verbose:
                logger.debug("PDF: processing page %d", i + 1)
            blocks = page.get_text("dict").get("blocks", [])
            page_width = page.rect.width if page.rect is not None else 0.0

            # Detect candidate infobox blocks on the first page based on geometry.
            infobox_block_idxs = set()
            if i == 0 and page_width:
                for bi, b in enumerate(blocks):
                    bbox = b.get("bbox")
                    if not bbox or len(bbox) != 4:
                        continue
                    x0, y0, x1, y1 = bbox
                    width = x1 - x0
                    # Heuristic: narrow block on the right side of the page with some text.
                    if x0 > page_width * 0.55 and width < page_width * 0.6:
                        has_text = False
                        for l in b.get("lines", []):
                            for s in l.get("spans", []):
                                if (s.get("text") or "").strip():
                                    has_text = True
                                    break
                            if has_text:
                                break
                        if has_text:
                            infobox_block_idxs.add(bi)

            lines = []
            for bi, b in enumerate(blocks):
                is_infobox_block = (i == 0 and bi in infobox_block_idxs)
                for l in b.get("lines", []):
                    spans = l.get("spans", [])
                    if not spans:
                        continue
                    line_max = max(s.get("size", 0) for s in spans) or 0
                    filtered = []
                    if self.drop_small_superscripts and line_max:
                        ratio = self.superscript_size_ratio
                        for s in spans:
                            t = (s.get("text") or "").strip()
                            sz = s.get("size", 0) or 0
                            # Drop if the span is tiny AND looks like a citation marker
                            if sz <= line_max * ratio and (
                                SUP_UNI_RE.match(t)
                                or BRACKET_REF.match(t)
                                or SHORT_DIGITS.match(t)
                                or t in {"[", "]"}
                            ):
                                continue
                            filtered.append(s)
                    else:
                        filtered = spans

                    if not filtered:
                        continue
                    max_size = max(s.get("size", 0) for s in filtered)
                    text = "".join(s.get("text", "") for s in filtered).strip()
                    is_bold = any((s.get("flags", 0) & 2) != 0 for s in filtered)  # flag 2 indicates bold in many cases
                    lines.append({"text": text, "size": max_size, "bold": is_bold, "infobox": is_infobox_block})
            pages.append((i + 1, lines))

        header_meta: List[str] = []
        footer_meta: List[str] = []

        # Optionally detect and scrub repeating header/footer lines across pages.
        if self.header_footer_filter and pages:
            header_patterns, footer_patterns = self._detect_header_footer_patterns(pages)
            pages, header_meta, footer_meta = self._filter_header_footer_lines(pages, header_patterns, footer_patterns)

        return title, pages, header_meta, footer_meta

    def _normalize_header_footer_text(self, text: str) -> str:
        """Normalize header/footer candidate text for stable comparison.

        We collapse whitespace and strip the result so that minor spacing
        differences across pages do not block detection.
        """
        return re.sub(r"\s+", " ", (text or "")).strip()

    def _detect_header_footer_patterns(
        self,
        pages: List[Tuple[int, List[Dict]]],
        max_lines: int = 3,
    ) -> Tuple[set, set]:
        """Detect repeating header/footer lines across pages.

        Strategy:
        - For each page, take the first/last few non-empty lines as candidates.
        - Normalize their text and count occurrences.
        - Any candidate that appears on a majority of pages (or at least twice)
          is treated as a header/footer pattern.
        """
        header_counts: Dict[str, int] = {}
        footer_counts: Dict[str, int] = {}
        total_pages = len(pages)

        for _, lines in pages:
            # Collect indices of non-empty lines for this page.
            nonempty_indices = [i for i, l in enumerate(lines) if (l.get("text") or "").strip()]
            if not nonempty_indices:
                continue
            # First / last few non-empty lines are candidates.
            header_idxs = nonempty_indices[:max_lines]
            footer_idxs = nonempty_indices[-max_lines:]

            for idx in header_idxs:
                txt = (lines[idx].get("text") or "").strip()
                norm = self._normalize_header_footer_text(txt)
                # Skip very short candidates (likely just bare page numbers).
                if len(norm) < 5:
                    continue
                header_counts[norm] = header_counts.get(norm, 0) + 1

            for idx in footer_idxs:
                txt = (lines[idx].get("text") or "").strip()
                norm = self._normalize_header_footer_text(txt)
                if len(norm) < 5:
                    continue
                footer_counts[norm] = footer_counts.get(norm, 0) + 1

        # Consider a line a true header/footer if it occurs on most pages, or at least twice.
        min_pages = max(2, int(0.6 * total_pages)) if total_pages else 0
        header_patterns = {t for t, c in header_counts.items() if c >= min_pages}
        footer_patterns = {t for t, c in footer_counts.items() if c >= min_pages}
        return header_patterns, footer_patterns

    def _filter_header_footer_lines(
        self,
        pages: List[Tuple[int, List[Dict]]],
        header_patterns: set,
        footer_patterns: set,
        max_lines: int = 3,
    ) -> Tuple[List[Tuple[int, List[Dict]]], List[str], List[str]]:
        """Remove repeating header/footer lines from pages, but keep one copy.

        We:
        - Identify header/footer candidate lines per page (first/last few).
        - If a candidate's normalized text matches a detected pattern, we keep
          it the first time we see it and drop subsequent occurrences.
        - Return the filtered pages and the list of header/footer texts we kept,
          which can be stored in payload metadata.
        """
        seen_headers: set = set()
        seen_footers: set = set()
        header_meta: List[str] = []
        footer_meta: List[str] = []
        new_pages: List[Tuple[int, List[Dict]]] = []

        for _, (page_num, lines) in enumerate(pages):
            # Determine non-empty line indices for this page.
            nonempty_indices = [i for i, l in enumerate(lines) if (l.get("text") or "").strip()]
            header_idxs = set(nonempty_indices[:max_lines])
            footer_idxs = set(nonempty_indices[-max_lines:])
            filtered_lines: List[Dict] = []

            for idx, line in enumerate(lines):
                txt = (line.get("text") or "").strip()
                if not txt:
                    filtered_lines.append(line)
                    continue

                norm = self._normalize_header_footer_text(txt)

                # Header candidate
                if idx in header_idxs and norm in header_patterns:
                    if norm in seen_headers:
                        # Drop repeated header line on later pages.
                        continue
                    seen_headers.add(norm)
                    header_meta.append(txt)
                    filtered_lines.append(line)
                    continue

                # Footer candidate
                if idx in footer_idxs and norm in footer_patterns:
                    if norm in seen_footers:
                        # Drop repeated footer line on later pages.
                        continue
                    seen_footers.add(norm)
                    footer_meta.append(txt)
                    filtered_lines.append(line)
                    continue

                # Regular content line
                filtered_lines.append(line)

            new_pages.append((page_num, filtered_lines))

        return new_pages, header_meta, footer_meta

    def _looks_like_header_or_footer(
        self,
        text: str,
        header_meta: Optional[List[str]] = None,
        footer_meta: Optional[List[str]] = None,
    ) -> bool:
        """
        Heuristic check to decide if a small markdown block is really just a
        repeating header / footer instead of real content.

        We deliberately keep this conservative so we don't drop legitimate text:
        - very short, no sentence-ending punctuation;
        - contains one of the known header/footer patterns; OR
        - looks like a doc-id + page-number line (e.g., 'NIR-SiPMs-AN100 3').
        """
        if not text:
            return False

        raw = text.strip()
        if not raw:
            return False

        # Normalize whitespace
        norm = re.sub(r"\s+", " ", raw)

        # If it has sentence punctuation, assume it's real content
        if any(ch in norm for ch in (".", "?", "!")):
            return False

        # Very long lines are unlikely to be pure headers/footers
        if len(norm) > 120:
            return False

        patterns: List[str] = []
        if header_meta:
            patterns.extend([p for p in header_meta if isinstance(p, str)])
        if footer_meta:
            patterns.extend([p for p in footer_meta if isinstance(p, str)])

        lowered = norm.lower()
        for pat in patterns:
            p = str(pat).strip()
            if not p:
                continue
            pl = p.lower()

            # Exact match to a known header/footer token
            if lowered == pl:
                return True

            # Common case: "DocId 3" style footer, or "DocId - 3"
            if lowered.startswith(pl + " "):
                tail = lowered[len(pl):].strip()
                # Very short numeric / rev suffix
                if len(tail) <= 8 and re.fullmatch(r"[\d\-–rev ]+", tail):
                    return True

        # Generic "DOCID page-number" pattern without relying on meta text
        if re.fullmatch(r"[A-Za-z0-9\-_.]+ \d{1,3}", norm):
            return True

        return False

    def _read_pages_pypdf2(self, pdf_bytes: bytes) -> Tuple[Optional[str], List[Tuple[int, List[Dict]]]]:
        reader = PdfReader(BytesIO(pdf_bytes))
        title = None
        try:
            meta = reader.metadata or {}
            title = meta.get('/Title') or meta.get('Title')
        except Exception:
            title = None
        #  logger.debug("PDF: opened via PyPDF2 pages=%d title=%s", len(reader.pages), title)
        pages = []
        for i in range(len(reader.pages)):
            page = reader.pages[i]
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            lines = [{"text": ln.strip(), "size": 0, "bold": False} for ln in text.splitlines()]
            if settings.debug_verbose:
                logger.debug("PDF: processed page %d lines=%d", i + 1, len(lines))
            pages.append((i + 1, lines))
        return title, pages

    def _read_pages(self, pdf_bytes: bytes) -> Tuple[Optional[str], List[Tuple[int, List[Dict]]], List[str], List[str]]:
        return self._read_pages_pymupdf(pdf_bytes)

    def _is_heading(self, line: Dict, median_size: float) -> Tuple[int, Optional[str]]:
        s = line["text"].strip()
        if not s:
            return 0, None
        # Numeric heading
        m = HEADING_NUM_RE.match(s)
        if m:
            dotted = m.group(1)
            title = m.group(2).strip()
            depth = dotted.count('.') + 1
            level = 1 if depth == 1 else 2 if depth == 2 else 3
            return level, title
        # Font-size based heuristic (PyMuPDF only; for PyPDF2 median_size==0)
        if line.get("size", 0) >= max(median_size * 1.2, median_size + 1):
            return 1, s
        # Bold short line (guarded by minimum length to avoid false positives)
        letters = [c for c in s if c.isalpha()]
        if (
            len(letters) >= self.heading_min_len
            and len(letters) <= 60
            and (sum(c.isupper() for c in letters) / len(letters)) > 0.8
        ):
            return 1, s.title()
        return 0, None

    def _chunk_payload(
        self,
        text: str,
        url: str,
        section: Optional[str],
        subsection: Optional[str],
        subsubsection: Optional[str],
        section_index: int,
        subsection_index: Optional[int],
        page_number: Optional[int],
        doc_title: Optional[str],
        header_meta: Optional[List[str]] = None,
        footer_meta: Optional[List[str]] = None,
    ) -> List[Dict]:
        chunks = self.splitter.split_text(text)
        total_chunks = len(chunks)
        payloads = []
        for idx, chunk in enumerate(chunks):
            payload: Dict[str, object] = {
                "text": chunk,
                "section": section or "Lead",
                "subsection": subsection,
                "subsubsection": subsubsection,
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "url": url,
                "document_type": "pdf",
                "source": url,
                "section_index": section_index,
                "subsection_index": subsection_index,
                "title": doc_title,
                "description": "",
                "page_number": page_number,
            }
            # Stash document-level header/footer metadata once per payload so that
            # downstream consumers can inspect it without re-scanning the PDF.
            if header_meta:
                payload["document_headers"] = header_meta
            if footer_meta:
                payload["document_footers"] = footer_meta
            payloads.append(payload)
        return payloads

    def _strip_markdown_links(self, text: str) -> str:
        """
        Remove inline markdown hyperlinks of the form [label](url) → label.
        This keeps the human‑readable text while removing noisy URLs that
        do not help embeddings or retrieval.

        Example:
            "[Mount Whitney](https://example.com)" → "Mount Whitney"
        """
        # Regex: capture [label](url) and replace with label only.
        return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    def _normalize_text(self, raw: str) -> str:
        """Normalize PDF-extracted text:
        - Dehyphenate across line breaks
        - Collapse single newlines within paragraphs to spaces
        - Preserve paragraph breaks (blank lines)
        - Light bullet/numbered-list guard: keep line breaks if a paragraph looks like a list
        """
        if settings.debug_verbose:
            try:
                before = len(raw)
            except Exception:
                before = 0
        if not raw:
            return ""
        if self.html_entity_unescape:
            raw = html.unescape(raw)
        # Normalize NBSP
        raw = raw.replace("\u00A0", " ")
        if self.normalize_quotes:
            # Map curly quotes/dashes to straight equivalents for consistency in embeddings/search
            trans_map = {
                "\u2019": "'",  # right single quote
                "\u2018": "'",  # left single quote
                "\u201C": '"',  # left double quote
                "\u201D": '"',  # right double quote
                "\u2013": "-",  # en dash
                "\u2014": "-",  # em dash
            }
            for k, v in trans_map.items():
                raw = raw.replace(k, v)

        # Remove markdown-style inline links (label-only keeps semantics)
        text_before_strip = raw
        raw = self._strip_markdown_links(raw)

        # Split into paragraphs on blank lines (2+ newlines)
        paragraphs = re.split(r"\n{2,}", raw)
        bullet_re = re.compile(r"^(\d+[\.)]|[-*•▪])\s+")
        norm_paragraphs = []
        for para in paragraphs:
            lines = [ln.strip() for ln in para.splitlines()]
            nonempty = [ln for ln in lines if ln]
            if not nonempty:
                norm_paragraphs.append("")
                continue
            # Detect bullet/numbered list heavy paragraphs
            bullet_count = sum(1 for ln in nonempty if bullet_re.match(ln))
            is_listy = (bullet_count / max(1, len(nonempty))) >= self.list_threshold
            if is_listy and self.preserve_lists:
                joined = "\n".join(nonempty)
            else:
                joined = " ".join(nonempty)
            norm_paragraphs.append(joined)

        # Rejoin paragraphs according to policy
        if self.paragraph_break_policy == "strict":
            sep = " "  # flatten paragraphs; no blank-line gaps
        else:
            sep = "\n\n"  # preserve paragraph breaks
        text = sep.join(norm_paragraphs)
        # Collapse excessive newlines and spaces
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[\t ]{2,}", " ", text)

        # Optionally remove inline numeric reference markers (e.g., [2], [12, 14])
        if self.strip_reference_markers:
            text = re.sub(r"(?<=\w)\s*\[(?:\d{1,3})(?:[\s,–-]+\d{1,3})*\]", "", text)
        # Optionally remove Unicode superscript numerals used as citations (¹²³⁴⁵⁶⁷⁸⁹⁰)
        if self.strip_unicode_superscripts:
            text = re.sub(r"(?<=\w)\s*[\u00B9\u00B2\u00B3\u2070-\u2079]+", "", text)

        if settings.debug_verbose:
            try:
                after = len(text)
                #logger.debug("PDF: normalize_text len %s -> %s", before, after)
            except Exception:
                pass
        return text.strip()

    def _parse_with_pymupdf(self, url: str, pdf_bytes: bytes) -> List[Dict]:
        #logger.debug("PDF: parsing url=%s", url)
        doc_title, pages, header_meta, footer_meta = self._read_pages(pdf_bytes)
        payloads: List[Dict] = []
        section_index = 0
        subsection_index = 0

        current_section: Optional[str] = None
        current_subsection: Optional[str] = None
        current_subsub: Optional[str] = None
        buffer: List[str] = []
        current_page: Optional[int] = None
        infobox_buffer: List[str] = []
        have_infobox = False

        def flush():
            nonlocal buffer, payloads, current_section, current_subsection, current_subsub, section_index, subsection_index, current_page
            if buffer:
                text = "\n".join(buffer).strip()
                text = self._normalize_text(text)
                if text and not self._should_skip_section(current_section, current_subsection, current_subsub):
                    payloads.extend(
                        self._chunk_payload(
                            text,
                            url,
                            current_section,
                            current_subsection,
                            current_subsub,
                            section_index,
                            subsection_index if current_subsection else None,
                            current_page,
                            doc_title,
                            header_meta,
                            footer_meta,
                        )
                    )
            buffer = []

        for (page_num, lines) in pages:
            current_page = page_num
            # Compute median size per page for heading heuristic
            sizes = [l.get("size", 0) for l in lines if l.get("size", 0) > 0]
            median_size = 0
            if sizes:
                srt = sorted(sizes)
                median_size = srt[len(srt)//2]
            for line in lines:
                # Collect infobox lines separately and exclude them from normal section parsing.
                if line.get("infobox"):
                    txt = line.get("text", "")
                    if txt:
                        infobox_buffer.append(txt)
                    continue

                # Optionally skip Table-Of-Contents style lines
                if self.toc_filter and TOC_RE.match(line.get("text", "")):
                    continue
                level, title = self._is_heading(line, median_size)
                if level > 0 and title:
                    flush()
                    if level == 1:
                        current_section = title
                        current_subsection = None
                        current_subsub = None
                        section_index += 1
                        subsection_index = 0
                    elif level == 2:
                        current_subsection = title
                        current_subsub = None
                        subsection_index += 1
                    else:
                        current_subsub = title
                    continue
                else:
                    txt = line.get("text", "")
                    if txt:
                        buffer.append(txt)
            flush()
        flush()
        #logger.info("PDF: parsed %s -> payloads=%d", url, len(payloads))

        # Build an INFOBOX payload from geometry-detected blocks on page 1, if any.
        if infobox_buffer:
            # Use double-newline join to preserve line breaks for subject-prefixing, then normalize
            raw_infobox = "\n\n".join(infobox_buffer).strip()
            normalized_infobox = self._normalize_text(raw_infobox)
            # Prefix each infobox line with the document title for stronger subject binding
            # --- Start suffix-stripping helper ---
            raw_subject = (doc_title or "").strip()
            title_suffixes_to_strip = [" - wikipedia", " — wikipedia", " | wikipedia"]
            subject = raw_subject
            lower = raw_subject.lower()
            for suf in title_suffixes_to_strip:
                if lower.endswith(suf):
                    subject = raw_subject[: -len(suf)].strip()
                    break
            # --- End suffix-stripping helper ---
            if subject:
                raw_lines = [ln.strip() for ln in normalized_infobox.split("\n") if ln.strip()]
                merged: List[str] = []
                i = 0
                while i < len(raw_lines):
                    cur = raw_lines[i]
                    nxt = raw_lines[i + 1] if i + 1 < len(raw_lines) else None

                    # Merge label + value lines (generic heuristic)
                    if (
                        nxt
                        and ":" not in cur
                        and ":" not in nxt
                        and len(cur) <= 40
                        and len(nxt) <= 80
                    ):
                        merged.append(f"{subject} — {cur} — {nxt}")
                        i += 2
                    else:
                        merged.append(f"{subject} — {cur}")
                        i += 1

                normalized_infobox = "\n".join(merged)
            if normalized_infobox:
                info_payload = {
                    "text": normalized_infobox,
                    "section": "INFOBOX",
                    "subsection": None,
                    "subsubsection": None,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "url": url,
                    "document_type": "pdf",
                    "source": url,
                    "section_index": 0,
                    "subsection_index": None,
                    "title": subject or doc_title,
                    "description": "",
                    "page_number": 1,
                }
                if header_meta:
                    info_payload["document_headers"] = header_meta
                if footer_meta:
                    info_payload["document_footers"] = footer_meta
                payloads.append(info_payload)
                have_infobox = True

        # Post-process Lead chunks to carve out a Wikipedia-style infobox when present
        # as a fallback if we did not detect an INFOBOX geometrically.
        if not have_infobox:
            updated_payloads: List[Dict] = []
            infobox_payloads: List[Dict] = []
            for p in payloads:
                text = p.get("text", "") or ""
                section = p.get("section") or "Lead"
                if section == "Lead" and "Highest point" in text:
                    infobox_text, remaining_text = _split_infobox_from_lead(text)
                    if infobox_text and remaining_text != text:
                        # Update the existing Lead payload text
                        p["text"] = remaining_text
                        # Build a separate INFOBOX payload without infobox field
                        info_p = dict(p)  # shallow copy of metadata
                        info_p["text"] = infobox_text
                        info_p["section"] = "INFOBOX"
                        info_p["subsection"] = None
                        info_p["subsubsection"] = None
                        info_p["section_index"] = 0
                        info_p["subsection_index"] = None
                        info_p["chunk_index"] = 0
                        info_p["total_chunks"] = 1
                        # Sanitize title for infobox payload (strip suffixes)
                        raw_subject = (info_p.get("title") or "").strip()
                        title_suffixes_to_strip = [" - wikipedia", " — wikipedia", " | wikipedia"]
                        subject = raw_subject
                        lower = raw_subject.lower()
                        for suf in title_suffixes_to_strip:
                            if lower.endswith(suf):
                                subject = raw_subject[: -len(suf)].strip()
                                break
                        info_p["title"] = subject or info_p.get("title")
                        if header_meta:
                            info_p["document_headers"] = header_meta
                        if footer_meta:
                            info_p["document_footers"] = footer_meta
                        infobox_payloads.append(info_p)
                updated_payloads.append(p)

            payloads = updated_payloads + infobox_payloads

        return payloads

    def _parse_with_pymupdf4llm(self, url: str, pdf_bytes: bytes) -> List[Dict]:
        """Parse a PDF using pymupdf4llm as the primary extractor.

        This relies on pymupdf4llm.to_markdown(page_chunks=True) to handle layout,
        headings, and table detection, and then maps the resulting markdown into
        the same section/subsection/chunk schema used elsewhere in the project.
        """
        if not HAS_PYMUPDF4LLM or pymupdf4llm is None:
            raise RuntimeError("pymupdf4llm is not available but pdf_use_pymupdf4llm is enabled")

        # --- Geometry-based infobox extraction using PyMuPDF ---
        geom_infobox_payload = None
        doc_title = None
        header_meta: List[str] = []
        footer_meta: List[str] = []
        try:
            doc_title, pages, header_meta, footer_meta = self._read_pages_pymupdf(pdf_bytes)
            # Collect infobox lines from page 1 (page_num == 1 and line.get("infobox"))
            infobox_lines = []
            for page_num, lines in pages:
                if page_num != 1:
                    continue
                for line in lines:
                    if line.get("infobox") and line.get("text", "").strip():
                        infobox_lines.append(line["text"])
            infobox_lines = [ln for ln in infobox_lines if ln.strip()]
            if infobox_lines:
                # Use double-newline join to preserve line breaks for subject-prefixing, then normalize
                raw_infobox = "\n\n".join(infobox_lines).strip()
                normalized_infobox = self._normalize_text(raw_infobox)
                # Prefix each infobox line with the document title for stronger subject binding
                # --- Start suffix-stripping helper ---
                raw_subject = (doc_title or "").strip()
                title_suffixes_to_strip = [" - wikipedia", " — wikipedia", " | wikipedia"]
                subject = raw_subject
                lower = raw_subject.lower()
                for suf in title_suffixes_to_strip:
                    if lower.endswith(suf):
                        subject = raw_subject[: -len(suf)].strip()
                        break
                # --- End suffix-stripping helper ---
                if subject:
                    raw_lines = [ln.strip() for ln in normalized_infobox.split("\n") if ln.strip()]
                    merged: List[str] = []
                    i = 0
                    while i < len(raw_lines):
                        cur = raw_lines[i]
                        nxt = raw_lines[i + 1] if i + 1 < len(raw_lines) else None

                        # Merge label + value lines (generic heuristic)
                        if (
                            nxt
                            and ":" not in cur
                            and ":" not in nxt
                            and len(cur) <= 40
                            and len(nxt) <= 80
                        ):
                            merged.append(f"{subject} — {cur} — {nxt}")
                            i += 2
                        else:
                            merged.append(f"{subject} — {cur}")
                            i += 1

                    normalized_infobox = "\n".join(merged)
                if normalized_infobox:
                    info_payload = {
                        "text": normalized_infobox,
                        "section": "INFOBOX",
                        "subsection": None,
                        "subsubsection": None,
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "url": url,
                        "document_type": "pdf",
                        "source": url,
                        "section_index": 0,
                        "subsection_index": None,
                        "title": subject or doc_title,
                        "description": "",
                        "page_number": 1,
                    }
                    if header_meta:
                        info_payload["document_headers"] = header_meta
                    if footer_meta:
                        info_payload["document_footers"] = footer_meta
                    geom_infobox_payload = info_payload
        except Exception:
            doc_title = None
            geom_infobox_payload = None

        # Write bytes to a temporary file so pymupdf4llm can open it reliably.
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            # Use page_chunks so we get per-page segmentation and tables metadata.
            # NOTE: Some PDFs fail to yield any detected tables under stricter strategies.
            # We start strict (better structure) and fall back to a looser strategy if needed.
            chunks = pymupdf4llm.to_markdown(
                tmp_path,
                page_chunks=True,
                table_strategy="lines_strict",
            )

            # If no tables were detected anywhere, retry with a more permissive strategy.
            try:
                total_tables = sum(len((c.get("tables") or [])) for c in (chunks or []))
            except Exception:
                total_tables = 0

            if total_tables == 0:
                chunks = pymupdf4llm.to_markdown(
                    tmp_path,
                    page_chunks=True,
                    table_strategy="lines",
                )
        finally:
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        # Debug log: concise summary of detected tables/chunks (if debug_verbose)
        if settings.debug_verbose:
            try:
                total_chunks = len(chunks or [])
                total_tables = sum(len((c.get("tables") or [])) for c in (chunks or []))
                logger.debug(
                    "PDF: pymupdf4llm produced chunks=%d total_tables=%d (table_strategy fallback may have run)",
                    total_chunks,
                    total_tables,
                )
            except Exception:
                pass

        payloads: List[Dict] = []

        current_section: Optional[str] = None
        current_subsection: Optional[str] = None
        current_subsub: Optional[str] = None
        section_index = 0
        subsection_index = 0

        for ch in chunks:
            # Prose markdown for this chunk (may include tables, depending on the backend).
            md_raw = (ch.get("text") or "").strip()
            md = md_raw

            # Tables for this chunk (we must collect these before possibly `continue`-ing).
            tables = ch.get("tables") or []
            meta = ch.get("metadata") or {}
            # Page number metadata can vary by pymupdf4llm version.
            # We accept a few common keys.
            page = None
            try:
                page = meta.get("page_number")
            except Exception:
                page = None
            if page is None:
                try:
                    page = meta.get("page")
                except Exception:
                    page = page
            if page is None:
                try:
                    page = meta.get("pageno")
                except Exception:
                    page = page
            if settings.debug_verbose:
                try:
                    logger.debug(
                        "PDF: chunk page=%s md_raw_len=%d tables_meta_len=%d drop_tables_from_prose=%s index_tables=%s",
                        page,
                        len(md_raw or ""),
                        len(tables or []),
                        self.pdf_drop_tables_from_prose,
                        self.pdf_index_tables,
                    )
                except Exception:
                    pass

            # If configured, remove markdown table blocks from the *prose* path to avoid
            # double-indexing. IMPORTANT: even if stripping tables leaves `md` empty,
            # we still must index `tables` below.
            if md and self.pdf_drop_tables_from_prose:
                md = self._strip_markdown_tables(md).strip()

            # Fallback: if metadata tables are missing (or later fail conversion),
            # we can still recover markdown table blocks from the original markdown.
            fallback_table_mds: List[str] = []
            if self.pdf_index_tables and md_raw:
                try:
                    fallback_table_mds = self._extract_markdown_tables(md_raw)
                except Exception:
                    fallback_table_mds = []

            # Prose path: only run when there is remaining markdown content.
            if md:
                lines = md.splitlines()
                local_buffer: List[str] = []

                def flush_local() -> None:
                    nonlocal local_buffer
                    if not local_buffer:
                        return
                    body_text = "\n".join(local_buffer).strip()
                    local_buffer = []
                    if not body_text:
                        return
                    norm = self._normalize_text(body_text)
                    if not norm:
                        return
                    if self._should_skip_section(current_section, current_subsection, current_subsub):
                        return
                    payloads.extend(
                        self._chunk_payload(
                            norm,
                            url,
                            current_section,
                            current_subsection,
                            current_subsub,
                            section_index,
                            subsection_index if current_subsection else None,
                            page,
                            doc_title,
                            header_meta,
                            footer_meta,
                        )
                    )

                for line in lines:
                    stripped = line.strip()

                    # Preserve blank lines inside the local buffer for paragraph detection.
                    if not stripped:
                        local_buffer.append("")
                        continue

                    # Drop header/footer-only lines when filtering is enabled.
                    if self.header_footer_filter and self._looks_like_header_or_footer(
                        stripped,
                        header_meta=header_meta,
                        footer_meta=footer_meta,
                    ):
                        # We still keep these strings in document_headers / document_footers
                        # via header_meta/footer_meta, but they don't become content chunks.
                        continue

                    # Markdown heading detection (e.g., "# Title", "## Section", "### Subsection").
                    if stripped.startswith("#"):
                        # Count leading '#' to determine heading level.
                        level = 0
                        for ch_header in stripped:
                            if ch_header == "#":
                                level += 1
                            else:
                                break
                        title = stripped[level:].strip()
                        # Strip surrounding bold markup if present ("**Title**").
                        if title.startswith("**") and title.endswith("**"):
                            title = title.strip("* ")

                        # Flush any buffered content before changing heading context.
                        flush_local()

                        if level == 1:
                            # Treat H1 as document title / Lead marker, not as a section.
                            # If PyMuPDF metadata did not provide a title, use this.
                            if not doc_title:
                                doc_title = title
                            # Do not modify current_section / subsection here; subsequent
                            # content will still be associated with the current section
                            # (or Lead if none has been set yet).
                        elif level == 2:
                            # H2 → new section (to mirror MediaWiki: page headings).
                            current_section = title
                            current_subsection = None
                            current_subsub = None
                            section_index += 1
                            subsection_index = 0
                        else:
                            # H3+ → subsection under the current section.
                            current_subsection = title
                            current_subsub = None
                            subsection_index += 1

                        continue

                    # Bold-only heading heuristic (e.g., "**Hydrology**", "**Climate**").
                    m = BOLD_HEADING_RE.match(stripped)
                    if m:
                        title = m.group("title").strip()
                        flush_local()
                        if current_section is None:
                            # Treat as a top-level section if we don't have one yet.
                            current_section = title
                            current_subsection = None
                            current_subsub = None
                            section_index += 1
                            subsection_index = 0
                        else:
                            # Otherwise treat as a subsection under the current section.
                            current_subsection = title
                            current_subsub = None
                            subsection_index += 1
                        continue

                    # Regular content line → accumulate into the current buffer.
                    local_buffer.append(line)

                # Flush any remaining buffered content for this chunk.
                flush_local()

            # Handle tables from this chunk as separate payloads.
            if not self.pdf_index_tables:
                if settings.debug_verbose:
                    logger.debug("PDF: table indexing disabled; skipping table payload generation for this chunk")
                continue

            # `tables` was collected above, before any prose-only early exits.
            def _table_to_markdown(obj) -> Optional[str]:
                """Best-effort conversion of a pymupdf4llm table object to markdown.

                pymupdf4llm table representations can vary by version / strategy.
                We support a few common shapes:
                - objects with .to_markdown()
                - raw markdown strings
                - dicts containing a markdown field

                Returns None when the object cannot be converted.
                """
                if obj is None:
                    return None
                # Most common: pymupdf4llm Table object
                if hasattr(obj, "to_markdown"):
                    try:
                        return obj.to_markdown()
                    except Exception:
                        return None
                # Sometimes already a markdown string
                if isinstance(obj, str):
                    s = obj.strip()
                    return s or None
                # Sometimes a dict-like representation
                if isinstance(obj, dict):
                    for key in ("markdown", "md", "table_md", "text"):
                        val = obj.get(key)
                        if isinstance(val, str) and val.strip():
                            return val.strip()
                return None

            # Combine metadata-provided tables with any tables recovered directly
            # from markdown. We de-dupe by exact markdown text.
            combined_table_mds: List[str] = []
            seen_tbl: set = set()

            # First: metadata tables (preferred)
            for t in tables:
                md_candidate = _table_to_markdown(t)
                if not md_candidate:
                    continue
                key = md_candidate.strip()
                if not key or key in seen_tbl:
                    continue
                seen_tbl.add(key)
                combined_table_mds.append(key)

            # Second: markdown-extracted tables (fallback)
            for tmd in (fallback_table_mds or []):
                key = (tmd or "").strip()
                if not key or key in seen_tbl:
                    continue
                seen_tbl.add(key)
                combined_table_mds.append(key)

            if settings.debug_verbose:
                try:
                    logger.debug(
                        "PDF: tables metadata=%d fallback=%d combined=%d section=%s subsection=%s",
                        len(tables or []),
                        len(fallback_table_mds or []),
                        len(combined_table_mds),
                        current_section,
                        current_subsection,
                    )
                except Exception:
                    pass
            if settings.debug_verbose and (tables or []) and not combined_table_mds:
                try:
                    first = (tables or [None])[0]
                    logger.debug(
                        "PDF: WARNING table metadata present but no markdown produced; first_table_type=%s first_table_repr=%s",
                        type(first).__name__,
                        repr(first)[:500],
                    )
                except Exception:
                    pass

            for tbl_md in combined_table_mds:
                # Determine where this table belongs before doing any work.
                # This lets us skip unwanted sections (e.g., References) early and
                # avoid confusing logs that look like we "indexed" something we later dropped.
                section_name = current_section or "Table"
                subsection_name = current_subsection or "Table"

                # Skip tables that belong to sections we want to ignore (e.g., "References").
                if self._should_skip_section(section_name, subsection_name, None):
                    if settings.debug_verbose:
                        logger.debug(
                            "PDF: skipping table due to skip_sections section=%s subsection=%s",
                            section_name,
                            subsection_name,
                        )
                    continue
                # DEBUG: table detected and ready for chunking/indexing
                if settings.debug_verbose:
                    try:
                        row_count = max(
                            0,
                            len([l for l in tbl_md.splitlines() if l.strip().startswith("|")]) - 2,
                        )
                    except Exception:
                        row_count = 0
                    logger.debug(
                        "PDF: indexing table after skipping sections rows=%d section=%s subsection=%s",
                        row_count,
                        current_section,
                        current_subsection,
                    )
                # Optionally chunk table by rows to preserve column context per chunk.
                table_blocks = self._chunk_markdown_table_by_rows(tbl_md)
                if not table_blocks:
                    # Fall back to whole-table indexing if chunking fails.
                    table_blocks = [tbl_md]

                for tbl_block in table_blocks:
                    # Detect INFOBOX tables using table structure (2-column label/value),
                    # not by matching a fixed label list. This prevents large multi-column
                    # data tables (e.g., lists of mountains) from being mislabeled as INFOBOX.
                    is_infobox = False
                    subject = None
                    kv_fields: Dict[str, str] = {}

                    if _looks_like_kv_infobox_table(tbl_block):
                        kv_fields = _parse_kv_infobox_table(tbl_block)

                    if kv_fields:
                        # --- Start suffix-stripping helper ---
                        raw_subject = (doc_title or "").strip()
                        title_suffixes_to_strip = [" - wikipedia", " — wikipedia", " | wikipedia"]
                        subject = raw_subject
                        lower = raw_subject.lower()
                        for suf in title_suffixes_to_strip:
                            if lower.endswith(suf):
                                subject = raw_subject[: -len(suf)].strip()
                                break
                        # --- End suffix-stripping helper ---

                        # Convert key/value fields to subject-bound lines (better retrieval).
                        # We keep one fact per line.
                        lines_out: List[str] = []
                        for k, v in kv_fields.items():
                            if subject:
                                lines_out.append(f"{subject} — {k} — {v}")
                            else:
                                lines_out.append(f"{k}: {v}")
                        tbl_norm = "\n".join(lines_out)
                        is_infobox = True
                    else:
                        # For INFOBOX-like tables, preserve per-row newlines by doubling them before normalization
                        tbl_norm = self._normalize_text(tbl_block.replace("\n", "\n\n"))

                    if not tbl_norm:
                        continue

                    # Re-label for INFOBOX after detection.
                    if is_infobox:
                        section_name = "INFOBOX"
                        subsection_name = "INFOBOX"
                    else:
                        # Keep the earlier placement for non-infobox tables.
                        section_name = section_name
                        subsection_name = subsection_name

                    tbl_payload: Dict[str, object] = {
                        "text": tbl_norm,
                        "section": "INFOBOX" if is_infobox else section_name,
                        "subsection": None if is_infobox else subsection_name,
                        "subsubsection": None,
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "url": url,
                        "document_type": "pdf",
                        "source": url,
                        "section_index": section_index,
                        "subsection_index": subsection_index if current_subsection else None,
                        "title": (subject if is_infobox else doc_title) if is_infobox else doc_title,
                        "description": "",
                        "page_number": page,
                    }
                    # Do NOT add infobox field for INFOBOX tables.
                    if header_meta:
                        tbl_payload["document_headers"] = header_meta
                    if footer_meta:
                        tbl_payload["document_footers"] = footer_meta
                    payloads.append(tbl_payload)

        # At the end, append geometry-based INFOBOX payload if no INFOBOX already present via table extraction
        has_infobox = any(p.get("section") == "INFOBOX" for p in payloads)
        if not has_infobox and geom_infobox_payload is not None:
            payloads.append(geom_infobox_payload)

        if settings.debug_verbose:
            try:
                total_payloads = len(payloads)
                infobox_payloads = sum(1 for p in payloads if p.get("section") == "INFOBOX")
                tableish_payloads = sum(1 for p in payloads if (p.get("section") in {"Table"} or p.get("subsection") in {"Table"}))
                logger.debug(
                    "PDF: pymupdf4llm final payloads=%d infobox=%d tableish=%d",
                    total_payloads,
                    infobox_payloads,
                    tableish_payloads,
                )
            except Exception:
                pass
        return payloads

    def _parse(self, url: str, pdf_bytes: bytes) -> List[Dict]:
        """Dispatch between pymupdf4llm and the legacy PyMuPDF parser.

        If settings.pdf_use_pymupdf4llm is True and pymupdf4llm is available,
        use the new _parse_with_pymupdf4llm backend. Otherwise, fall back to
        the existing PyMuPDF-based pipeline.
        """
        use_llm = getattr(settings, "pdf_use_pymupdf4llm", False)
        if settings.debug_verbose:
            logger.debug(
                "PDF: dispatcher use_llm=%s HAS_PYMUPDF4LLM=%s", use_llm, HAS_PYMUPDF4LLM
            )
        if use_llm and HAS_PYMUPDF4LLM and pymupdf4llm is not None:
            try:
                return self._parse_with_pymupdf4llm(url, pdf_bytes)
            except Exception as e:
                logger.exception("PDF: pymupdf4llm parse failed for %s, falling back to PyMuPDF: %s", url, e)

        # Default / fallback path: existing PyMuPDF-based parser.
        return self._parse_with_pymupdf(url, pdf_bytes)

    def parse_from_url(self, url: str) -> List[Dict]:
        #logger.debug("PDF: parse_from_url %s", url)
        pdf_bytes = self._fetch(url)
        return self._parse(url, pdf_bytes)

    def parse_from_bytes(self, data: bytes, url_for_source: str) -> List[Dict]:
        #logger.debug("PDF: parse_from_bytes src=%s bytes=%d", url_for_source, len(data) if data else 0)
        return self._parse(url_for_source, data)
