import re
from typing import List, Dict, Optional, Tuple
import httpx
from io import BytesIO
from backend.embeddings.text_splitter import TextSplitter

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False
    from PyPDF2 import PdfReader


HEADING_NUM_RE = re.compile(r"^(\d+(?:\.\d+){0,2})\s+(.+)$")


class PDFExtractor:
    def __init__(self, chunk_size: int, chunk_overlap: int, timeout: int = 20):
        self.splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_manual_splitter=False,
        )
        self.timeout = timeout

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
                    # Each line has spans with font size; take max size and concatenated text
                    spans = l.get("spans", [])
                    if not spans:
                        continue
                    max_size = max(s.get("size", 0) for s in spans)
                    text = "".join(s.get("text", "") for s in spans).strip()
                    is_bold = any((s.get("flags", 0) & 2) != 0 for s in spans)  # flag 2 indicates bold in many cases
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
        # Bold short line
        letters = [c for c in s if c.isalpha()]
        if len(letters) and len(letters) <= 60 and sum(c.isupper() for c in letters) / len(letters) > 0.8:
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
