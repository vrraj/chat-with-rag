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

# --- Infobox extraction helpers and constants ---
INFOBOX_LABELS = [
    "Elevation",
    "Prominence",
    "Parent peak",
    "Isolation",
    "Listing",
    "Coordinates",
    "Naming",
    "Etymology",
]

def _split_infobox_from_lead(text: str) -> Tuple[Optional[str], str]:
    """Attempt to carve out a Wikipedia-style mountain infobox from Lead text.

    Strategy:
    - Look for the first occurrence of "Highest point".
    - If found, treat everything from that phrase to the end of the string as the
      infobox blob.
    - Return (infobox_text, remaining_text). If not found, return (None, original_text).

    This is intentionally conservative and only applies when the expected anchor
    phrase is present. It is safe for non-Wikipedia PDFs because it will simply
    be a no-op when the pattern is absent.
    """
    if not text:
        return None, text

    # Work on a normalized single-line representation to simplify span math,
    # but keep the original text for the remaining part.
    normalized = re.sub(r"\s+", " ", text).strip()
    anchor = "highest point"
    idx = normalized.lower().find(anchor)
    if idx == -1:
        return None, text

    infobox_text = normalized[idx:].strip()
    remaining_text = normalized[:idx].strip()
    return infobox_text, remaining_text


def _parse_infobox_key_values(infobox_text: str) -> Dict[str, str]:
    """Parse a flattened infobox blob into key/value pairs using known labels.

    Example input (single-line normalized):
      "Highest point Elevation 14,505 ft (4,421 m) NAVD 88 Prominence 10,075 ft ..."

    Returns a dict like:
      { 'Elevation': '14,505 ft (4,421 m) NAVD 88', 'Prominence': '10,075 ft (3,071 m)', ... }

    If labels are not found, returns an empty dict. This is meant as a best-effort
    helper for Wikipedia-style mountain pages and is a no-op for other PDFs.
    """
    result: Dict[str, str] = {}
    if not infobox_text:
        return result

    text = re.sub(r"\s+", " ", infobox_text).strip()
    if not text:
        return result

    # Build a regex that matches any of the known labels as a group, e.g.:
    # (Elevation|Prominence|Parent\ peak|Isolation|Listing|Coordinates|Naming|Etymology)
    label_pattern = r"(" + "|".join(re.escape(lbl) for lbl in INFOBOX_LABELS) + r")\b"
    matches = list(re.finditer(label_pattern, text))
    if not matches:
        return result

    for i, m in enumerate(matches):
        label = m.group(1)
        start_val = m.end()
        if i + 1 < len(matches):
            end_val = matches[i + 1].start()
        else:
            end_val = len(text)
        value = text[start_val:end_val].strip(" :;,-")
        if value:
            result[label] = value.strip()

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

        # Normalized list of section titles to skip entirely (case-insensitive)
        # e.g., ["References", "See also"]. These are matched against the
        # "section" name as extracted from headings / markdown.
        self.skip_sections = [s.strip().casefold() for s in (skip_sections or []) if s and s.strip()]
        logger.debug("PDF: skip_sections=%s", self.skip_sections)
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

    def _read_pages_pymupdf(self, pdf_bytes: bytes) -> Tuple[Optional[str], List[Tuple[int, List[Dict]]]]:
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
        return title, pages

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

    def _read_pages(self, pdf_bytes: bytes) -> Tuple[Optional[str], List[Tuple[int, List[Dict]]]]:
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

    def _chunk_payload(self, text: str, url: str, section: Optional[str], subsection: Optional[str], subsubsection: Optional[str], section_index: int, subsection_index: Optional[int], page_number: Optional[int], doc_title: Optional[str]) -> List[Dict]:
        chunks = self.splitter.split_text(text)
        total_chunks = len(chunks)
        #logger.debug(
        #    "PDF: chunked section=%s subsection=%s subsub=%s page=%s chunks=%d",
        #    section or "Lead", subsection, subsubsection, page_number, total_chunks,
        #)
        payloads = []
        for idx, chunk in enumerate(chunks):
            payloads.append({
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
            })
        return payloads

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
        doc_title, pages = self._read_pages(pdf_bytes)
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
            raw_infobox = "\n".join(infobox_buffer).strip()
            normalized_infobox = self._normalize_text(raw_infobox)
            if normalized_infobox:
                infobox_kv = _parse_infobox_key_values(normalized_infobox)
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
                    "title": doc_title,
                    "description": "",
                    "page_number": 1,
                    "infobox": infobox_kv,
                }
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
                        # Build a separate INFOBOX payload with structured key/values
                        infobox_kv = _parse_infobox_key_values(infobox_text)
                        info_p = dict(p)  # shallow copy of metadata
                        info_p["text"] = infobox_text
                        info_p["section"] = "INFOBOX"
                        info_p["subsection"] = None
                        info_p["subsubsection"] = None
                        info_p["section_index"] = 0
                        info_p["subsection_index"] = None
                        info_p["chunk_index"] = 0
                        info_p["total_chunks"] = 1
                        info_p["infobox"] = infobox_kv
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
        try:
            doc_title, pages = self._read_pages_pymupdf(pdf_bytes)
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
                raw_infobox = "\n".join(infobox_lines).strip()
                normalized_infobox = self._normalize_text(raw_infobox)
                if normalized_infobox:
                    infobox_kv = _parse_infobox_key_values(normalized_infobox)
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
                        "title": doc_title,
                        "description": "",
                        "page_number": 1,
                    }
                    if infobox_kv:
                        info_payload["infobox"] = infobox_kv
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
            chunks = pymupdf4llm.to_markdown(
                tmp_path,
                page_chunks=True,
                table_strategy="lines_strict",
            )
        finally:
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        payloads: List[Dict] = []

        current_section: Optional[str] = None
        current_subsection: Optional[str] = None
        current_subsub: Optional[str] = None
        section_index = 0
        subsection_index = 0

        for ch in chunks:
            md = (ch.get("text") or "").strip()
            if not md:
                continue

            page = None
            meta = ch.get("metadata") or {}
            try:
                page = meta.get("page_number")
            except Exception:
                page = None

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
                        doc_title,  # wire doc_title through
                    )
                )

            for line in lines:
                stripped = line.strip()

                # Preserve blank lines inside the local buffer for paragraph detection.
                if not stripped:
                    local_buffer.append("")
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
            tables = ch.get("tables") or []
            for tbl in tables:
                try:
                    tbl_md = tbl.to_markdown()
                except Exception:
                    continue
                tbl_norm = self._normalize_text(tbl_md)
                if not tbl_norm:
                    continue

                # Best-effort attempt to extract structured key/values from infobox-like tables.
                infobox_kv = _parse_infobox_key_values(tbl_norm)
                is_infobox = bool(infobox_kv)

                section_name = current_section or ("INFOBOX" if is_infobox else "Table")
                subsection_name = current_subsection if current_subsection else ("INFOBOX" if is_infobox else "Table")

                # Skip tables that belong to sections we want to ignore (e.g., "References").
                if self._should_skip_section(section_name, subsection_name, None):
                    continue

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
                    "title": doc_title,
                    "description": "",
                    "page_number": page,
                }
                if is_infobox:
                    tbl_payload["infobox"] = infobox_kv

                payloads.append(tbl_payload)

        # At the end, append geometry-based INFOBOX payload if no INFOBOX already present via table extraction
        has_infobox = any(p.get("section") == "INFOBOX" for p in payloads)
        if not has_infobox and geom_infobox_payload is not None:
            payloads.append(geom_infobox_payload)

        return payloads

    def _parse(self, url: str, pdf_bytes: bytes) -> List[Dict]:
        """Dispatch between pymupdf4llm and the legacy PyMuPDF parser.

        If settings.pdf_use_pymupdf4llm is True and pymupdf4llm is available,
        use the new _parse_with_pymupdf4llm backend. Otherwise, fall back to
        the existing PyMuPDF-based pipeline.
        """
        use_llm = getattr(settings, "pdf_use_pymupdf4llm", False)
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
