import re
from typing import List, Dict, Optional, Tuple
import httpx
from io import BytesIO
from backend.embeddings.text_splitter import TextSplitter
import html

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False
    from PyPDF2 import PdfReader


HEADING_NUM_RE = re.compile(r"^(\d+(?:\.\d+){0,2})\s+(.+)$")

# Lines that look like table-of-contents entries, e.g., "1.2 Title .... 12"
TOC_RE = re.compile(r"^\s*\d+(?:\.\d+)*\s+.+?\.{2,}\s*\d+\s*$")

# Superscript and citation patterns used to drop tiny reference spans at extraction time
SUP_UNI_RE = re.compile(r'^[\u00B9\u00B2\u00B3\u2070-\u2079]+$')  # ¹²³⁴⁵⁶⁷⁸⁹⁰
BRACKET_REF = re.compile(r'^\[(?:\d{1,3})(?:[\s,–-]+\d{1,3})*\]$')
SHORT_DIGITS = re.compile(r'^\d{1,3}$')

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

    def _fetch(self, url: str) -> bytes:
        with httpx.Client(timeout=self.timeout) as client:
            r = client.get(url)
            r.raise_for_status()
            return r.content

    def _read_pages_pymupdf(self, pdf_bytes: bytes) -> Tuple[Optional[str], List[Tuple[int, List[Dict]]]]:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        title = doc.metadata.get("title") if doc.metadata else None
        pages = []
        for i, page in enumerate(doc):
            blocks = page.get_text("dict").get("blocks", [])
            lines = []
            for b in blocks:
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
                    lines.append({"text": text, "size": max_size, "bold": is_bold})
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
        pages = []
        for i in range(len(reader.pages)):
            page = reader.pages[i]
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            lines = [{"text": ln.strip(), "size": 0, "bold": False} for ln in text.splitlines()]
            pages.append((i + 1, lines))
        return title, pages

    def _read_pages(self, pdf_bytes: bytes) -> Tuple[Optional[str], List[Tuple[int, List[Dict]]]]:
        if HAS_PYMUPDF:
            return self._read_pages_pymupdf(pdf_bytes)
        else:
            return self._read_pages_pypdf2(pdf_bytes)

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

        return text.strip()

    def _parse(self, url: str, pdf_bytes: bytes) -> List[Dict]:
        doc_title, pages = self._read_pages(pdf_bytes)
        payloads: List[Dict] = []
        section_index = 0
        subsection_index = 0

        current_section: Optional[str] = None
        current_subsection: Optional[str] = None
        current_subsub: Optional[str] = None
        buffer: List[str] = []
        current_page: Optional[int] = None

        def flush():
            nonlocal buffer, payloads, current_section, current_subsection, current_subsub, section_index, subsection_index, current_page
            if buffer:
                text = "\n".join(buffer).strip()
                text = self._normalize_text(text)
                if text:
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
        return payloads

    def parse_from_url(self, url: str) -> List[Dict]:
        pdf_bytes = self._fetch(url)
        return self._parse(url, pdf_bytes)

    def parse_from_bytes(self, data: bytes, url_for_source: str) -> List[Dict]:
        return self._parse(url_for_source, data)
