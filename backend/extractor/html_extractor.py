# NOTE: This module handles **HTML-only** extraction. For PDFs use `pdf_extractor.py`;
# for MediaWiki (Wikipedia, etc.) use `mediawiki_extractor.py`.
import warnings
from typing import Optional, List, Dict
import re
import html as _html
import trafilatura
import logging
from backend.core.config import settings
from bs4 import BeautifulSoup
from .splitters import TextSplitter

logger = logging.getLogger(__name__)

# Markdown heading pattern and reference cleanup
H_RE = re.compile(r'^(#{1,6})\s+(.*)\s*$')
REF_BRACKETS_RE = re.compile(r"(?<=\w)\s*\[(?:\d{1,3})(?:[\s,–-]+\d{1,3})*\]")
SLUG_RE = re.compile(r"[^a-z0-9]+")

class ContentExtractor:
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ContentExtractor is deprecated for indexing. Use HTMLExtractor to emit chunked payloads (html only).",
            DeprecationWarning,
            stacklevel=2,
        )
        pass

    def extract_content(self, *args, **kwargs):
        warnings.warn(
            "ContentExtractor.extract_content returns a flat dict. For chunked, section-aware HTML payloads, use HTMLExtractor.parse().",
            DeprecationWarning,
            stacklevel=2,
        )
        # Actual extraction logic would go here
        pass

    def parse(self, html_content: str, url: str, skip_sections: Optional[List[str]] = None) -> List[Dict]:
        """HTML-only wrapper that delegates to HTMLExtractor (section-aware, chunked).

        This maintains backward compatibility with older call sites that still import
        ContentExtractor but expect chunked payloads. For fine-grained control, import
        and use HTMLExtractor directly.
        """
        warnings.warn(
            "ContentExtractor.parse() now delegates to HTMLExtractor.parse() (html only).",
            DeprecationWarning,
            stacklevel=2,
        )
        # Sensible defaults consistent with other extractors
        extractor = HTMLExtractor(
            chunk_size=500,
            chunk_overlap=100,
            html_mode="dom",  # DOM ensures we honor real <h2>/<h3>/<h4> headings
        )
        return extractor.parse(html_content, url, skip_sections=skip_sections)


class HTMLExtractor:
    """Section-aware HTML extractor that emits chunked payloads compatible with PDF/MediaWiki.

    Emits a list of dicts with keys:
    text, section, subsection, subsubsection, chunk_index, total_chunks,
    url, document_type, source, section_index, subsection_index, title, description.
    """
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        *,
        html_mode: str = "markdown",  # or "dom"
        drop_selectors: Optional[List[str]] = None,
        preserve_paragraphs: bool = True,
        normalize_quotes: bool = True,
        html_entity_unescape: bool = True,
        strip_reference_markers: bool = True,
    ) -> None:
        self.splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_manual_splitter=False,
        )
        self.html_mode = html_mode
        self.drop_selectors = drop_selectors if drop_selectors is not None else [
            "nav","footer","header",".sidebar",".toc",".infobox",".reference",".references",
            "table","figure","script","style",
        ]
        self.preserve_paragraphs = preserve_paragraphs
        self.normalize_quotes = normalize_quotes
        self.html_entity_unescape = html_entity_unescape
        self.strip_reference_markers = strip_reference_markers

    # ----------------- public API -----------------
    def parse(self, html_content: str, url: str, skip_sections: Optional[List[str]] = None) -> List[Dict]:
        """Return list of chunk payloads for the given HTML.

        skip_sections: case-insensitive titles to exclude at the h2 level (and children in DOM mode).
        """
        logger.debug("HTML: parse url=%s mode=%s", url, self.html_mode)
        title, description = self._extract_meta(html_content)
        if settings.debug_verbose:
            maxc = int(getattr(settings, "debug_log_truncate_chars", 500))
            t_snip = (title if len(title) <= maxc else title[:maxc] + "…") if title else ""
            d_snip = (description if len(description) <= maxc else description[:maxc] + "…") if description else ""
            logger.debug("HTML: meta title='%s' desc='%s'", t_snip, d_snip)
        if self.html_mode == "markdown":
            payloads = self._parse_markdown(html_content, url, title, description, skip_sections)
            # Auto-fallback: if Markdown path produced only Lead (no h2/h3 blocks), try DOM mode
            if not payloads or all((p.get("section") == "Lead" for p in payloads)):
                payloads = self._parse_dom(html_content, url, title, description, skip_sections)
        else:
            payloads = self._parse_dom(html_content, url, title, description, skip_sections)

        logger.debug("HTML: initial payloads count=%d", len(payloads) if payloads else 0)
        # Fill total_chunks uniformly
        total = len(payloads)
        for p in payloads:
            p["total_chunks"] = total
        logger.info("HTML: parsed url=%s payloads=%d", url, len(payloads))
        return payloads

    # ----------------- internals -----------------
    def _extract_meta(self, html_content: str) -> (str, str):
        logger.debug("HTML: extracting meta tags")
        soup = BeautifulSoup(html_content, 'html.parser')
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        meta_description = ""
        meta = soup.find('meta', attrs={'name': 'description'})
        if meta:
            meta_description = meta.get('content', '')
        if not meta_description:
            og_desc = soup.find('meta', attrs={'property': 'og:description'})
            if og_desc:
                meta_description = og_desc.get('content', '')
            if not meta_description:
                twitter_desc = soup.find('meta', attrs={'name': 'twitter:description'})
                if twitter_desc:
                    meta_description = twitter_desc.get('content', '')
        return title, (meta_description or "")

    def _normalize(self, text: str) -> str:
        if not text:
            return ""
        if self.html_entity_unescape:
            text = _html.unescape(text)
        # Curly quotes/dashes → straight equivalents
        if self.normalize_quotes:
            trans = {
                "\u2019": "'", "\u2018": "'",  # ’ ‘
                "\u201C": '"', "\u201D": '"',  # “ ”
                "\u2013": "-",  "\u2014": "-",  # – —
            }
            for k, v in trans.items():
                text = text.replace(k, v)
        # Collapse whitespace
        text = re.sub(r"\u00a0", " ", text)
        text = re.sub(r"[\t ]+", " ", text)
        if self.preserve_paragraphs:
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        else:
            text = text.replace("\n", " ")
        # Optional: strip [12], [2, 5]
        if self.strip_reference_markers:
            text = REF_BRACKETS_RE.sub("", text)
        return text.strip()

    def _slug(self, s: str) -> str:
        s = s.strip().lower()
        s = SLUG_RE.sub('-', s).strip('-')
        return s or "section"

    def _chunk_payload(self,
                       text: str,
                       base_url: str,
                       section: Optional[str],
                       subsection: Optional[str],
                       subsubsection: Optional[str],
                       section_index: int,
                       title: str,
                       description: str,
                       chunk_start_index: int,
                       ) -> List[Dict]:
        logger.debug(
            "HTML: chunking section=%s subsection=%s subsub=%s start_idx=%d",
            section or "Lead", subsection, subsubsection, chunk_start_index,
        )
        chunks = self.splitter.split_text(text)
        # Build section URL with slug if we have a title and no DOM id
        anchor = None
        if subsubsection:
            anchor = self._slug(subsubsection)
        elif subsection:
            anchor = self._slug(subsection)
        elif section and section != "Lead":
            anchor = self._slug(section)
        # Only append an anchor if the base_url doesn't already have a fragment
        section_url = base_url
        if anchor and "#" not in base_url:
            section_url = f"{base_url}#{anchor}"
        logger.debug("HTML: chunked into %d chunks for section '%s'", len(chunks), section or "Lead")
        out = []
        for i, c in enumerate(chunks):
            out.append({
                "text": c,
                "section": section or "Lead",
                "subsection": subsection,
                "subsubsection": subsubsection,
                "chunk_index": chunk_start_index + i,
                "total_chunks": 0,  # filled later
                "url": section_url,
                "document_type": "html",
                "source": section_url,
                "section_index": section_index,
                "subsection_index": None,  # set in DOM mode when available
                "title": title,
                "description": description,
            })
        return out

    # -------- markdown mode --------
    def _parse_markdown(self, html_content: str, url: str, title: str, description: str, skip_sections: Optional[List[str]]) -> List[Dict]:
        logger.debug("HTML: parse_markdown start")
        md = trafilatura.extract(
            html_content,
            include_comments=False,
            include_tables=False,
            include_images=False,
            include_formatting=True,
            output_format="markdown",
            include_links=False,
            favor_recall=True,
        )
        if not md:
            logger.debug("HTML: trafilatura returned empty markdown")
            return []

        # Walk markdown lines into blocks
        blocks = []  # list of dict(level,title,text)
        cur_level, cur_title, cur_buf = 2, "Lead", []
        for line in md.splitlines():
            m = H_RE.match(line)
            if m:
                level = len(m.group(1))
                title_line = m.group(2).strip()
                if level >= 2:  # ignore h1
                    blocks.append({"level": cur_level, "title": cur_title, "text": "\n".join(cur_buf).strip()})
                    cur_level, cur_title, cur_buf = level, title_line, []
                continue
            cur_buf.append(line)
        blocks.append({"level": cur_level, "title": cur_title, "text": "\n".join(cur_buf).strip()})
        blocks = [b for b in blocks if b.get("text")]
        if settings.debug_verbose:
            logger.debug("HTML: markdown blocks=%d", len(blocks))

        payloads: List[Dict] = []
        global_idx = 0
        section_index = 0
        current_h2 = None
        current_h3 = None

        for b in blocks:
            lvl, ttl, txt = b["level"], b["title"], self._normalize(b["text"])
            if not txt:
                continue
            # skip list applies to h2 only
            if lvl == 2 and skip_sections and any(ttl.lower() == s.lower() for s in skip_sections):
                current_h2 = None
                current_h3 = None
                continue
            if lvl == 2:
                current_h2 = ttl
                current_h3 = None
                section_index += 1
                sec, sub, subsub = current_h2, None, None
            elif lvl == 3:
                current_h3 = ttl
                sec, sub, subsub = current_h2, current_h3, None
            else:  # h4+
                sec, sub, subsub = current_h2, current_h3, ttl

            chunked = self._chunk_payload(txt, url, sec or "Lead", sub, subsub, section_index if sec else 0, title, description, global_idx)
            payloads.extend(chunked)
            global_idx += len(chunked)

        logger.debug("HTML: markdown payloads=%d", len(payloads))
        return payloads

    # -------- dom mode --------
    def _parse_dom(self, html_content: str, url: str, page_title: str, description: str, skip_sections: Optional[List[str]]) -> List[Dict]:
        logger.debug("HTML: parse_dom start")
        soup = BeautifulSoup(html_content, 'html.parser')
        for sel in self.drop_selectors:
            for el in soup.select(sel):
                el.decompose()
        logger.debug("HTML: drop_selectors applied count=%d selectors", len(self.drop_selectors))
        root = (
            soup.select_one(
                "article,[role=article],main .prose,main .content,main .post,main .article,.article-content,.post-content"
            )
            or soup
        )

        payloads: List[Dict] = []
        global_idx = 0
        section_index = 0

        # Lead: gather only "texty" nodes before first h2 to avoid hero/nav chrome
        first_h2 = root.find('h2')
        logger.debug("HTML: first_h2 found=%s", bool(first_h2))
        if first_h2:
            collected = []
            allowed = 'p, ul, ol, blockquote, pre, code'
            for node in list(root.children):
                if getattr(node, 'name', None) == 'h2':
                    break
                # From each pre-h2 sibling, only take text from allowed tags beneath it
                if hasattr(node, 'select'):
                    parts = [el.get_text(" ") for el in node.select(allowed)]
                    if parts:
                        collected.append(" ".join(parts))
            lead_text = self._normalize(" ".join(collected))
            if lead_text:
                chunks = self._chunk_payload(lead_text, url, "Lead", None, None, 0, page_title, description, global_idx)
                payloads.extend(chunks)
                logger.debug("HTML: lead chunks added=%d", len(chunks))
                global_idx += len(chunks)

        current_h2 = None
        current_h3 = None
        subsection_counts = {}

        for h in root.find_all(['h2','h3','h4']):
            lvl = int(h.name[1])
            ttl = h.get_text(" ").strip()
            anchor = h.get('id')
            if lvl == 2 and skip_sections and any(ttl.lower() == s.lower() for s in skip_sections):
                current_h2 = None
                current_h3 = None
                continue
            if lvl == 2:
                current_h2 = ttl
                current_h3 = None
                section_index += 1
                subsection_counts[section_index] = 0
            elif lvl == 3:
                current_h3 = ttl
                subsection_counts[section_index] = subsection_counts.get(section_index, 0) + 1

            # collect content until next heading of same/higher level
            collected_nodes = []
            for sib in h.next_siblings:
                nm = getattr(sib, 'name', None)
                if nm in ('h2','h3','h4'):
                    break
                collected_nodes.append(sib.get_text(" ") if hasattr(sib, 'get_text') else str(sib))
            txt = self._normalize(" ".join(collected_nodes))
            if settings.debug_verbose:
                logger.debug("HTML: heading lvl=%d title='%s' text_len=%d", lvl, ttl, len(txt))
            if not txt:
                continue

            sec = current_h2
            sub = current_h3 if lvl >= 3 else None
            subsub = ttl if lvl >= 4 else None

            # If we have a real DOM id, build URL with it; else slug will be added in _chunk_payload
            section_url = f"{url}#{anchor}" if anchor else url
            chunks = self._chunk_payload(txt, section_url, sec or "Lead", sub, subsub, section_index if sec else 0, page_title, description, global_idx)
            if sub is not None:
                for c in chunks:
                    c['subsection_index'] = subsection_counts.get(section_index, 0) - 1

            payloads.extend(chunks)
            logger.debug("HTML: heading chunks added=%d (total so far=%d)", len(chunks), len(payloads))
            global_idx += len(chunks)

        return payloads


# HTML-only helper function for chunked payloads

def extract_html_payloads(
    html_content: str,
    url: str,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    html_mode: str = "dom",
    drop_selectors: Optional[List[str]] = None,
    preserve_paragraphs: bool = True,
    normalize_quotes: bool = True,
    html_entity_unescape: bool = True,
    strip_reference_markers: bool = True,
    skip_sections: Optional[List[str]] = None,
) -> List[Dict]:
    """HTML-only entrypoint that returns chunked payloads matching PDF/MediaWiki schema.

    This is intentionally **HTML-only**. For PDFs, use `PDFExtractor`.
    For MediaWiki, use `MediaWikiExtractor`.
    """
    logger.debug("HTML: extract_html_payloads url=%s mode=%s", url, html_mode)
    extractor = HTMLExtractor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        html_mode=html_mode,
        drop_selectors=drop_selectors,
        preserve_paragraphs=preserve_paragraphs,
        normalize_quotes=normalize_quotes,
        html_entity_unescape=html_entity_unescape,
        strip_reference_markers=strip_reference_markers,
    )
    return extractor.parse(html_content, url, skip_sections=skip_sections)


__all__ = ["HTMLExtractor", "extract_html_payloads", "ContentExtractor"]
