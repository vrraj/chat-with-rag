import mwparserfromhell
from typing import List, Dict, Optional
import requests
import time
from requests import Response
from urllib.parse import unquote, urlparse
import tiktoken
import logging
from backend.core.config import settings
from backend.embeddings.text_splitter import TextSplitter
from bs4 import BeautifulSoup


logger = logging.getLogger(__name__)

WIKI_API_URL = settings.wiki_api_url
WIKI_UA = settings.wiki_user_agent
DEFAULT_TIMEOUT = int(settings.wiki_timeout_secs)
MAX_RETRIES = int(settings.wiki_max_retries)

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
    # Remove occurrences like 'thumb|...' that are not inside brackets or templates
    text = re.sub(r'thumb\|[^\n]*', '', text)
    return text

def _wiki_get(params: Dict, api_url: str, ua: str) -> Response:
    """Resilient GET to Wikipedia Action API with UA + backoff."""
    backoff = 0.5
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(
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
                    f"Wiki API status {r.status_code}; attempt {attempt}/{MAX_RETRIES}. Backing off {backoff:.1f}s"
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                logger.error(f"Wiki API request failed after {MAX_RETRIES} attempts: {e}")
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
        wiki_mode: str = "action_api",  # or "parsoid"
        wiki_drop_selectors: Optional[List[str]] = None,
        wiki_preserve_paragraphs: bool = True,
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
        self.wiki_mode = wiki_mode
        self.wiki_drop_selectors = (
            wiki_drop_selectors
            if wiki_drop_selectors is not None
            else [
                "sup.reference",
                ".mw-references-wrap",
                ".navbox",
                ".infobox",
                "table",
                "figure",
                "style",
                "script",
            ]
        )
        self.wiki_preserve_paragraphs = wiki_preserve_paragraphs

    def parse(self, wikitext: str, url: str, skip_sections: Optional[List[str]] = None) -> List[Dict]:
        """
        Parse wikitext into section/subsection chunks, split into token chunks, and return flattened payloads.
        """
        wikicode = mwparserfromhell.parse(wikitext)
        payloads = []
        section_index = 0

        for top_section in wikicode.get_sections(include_lead=True, levels=[2]):
            heading_stack = []
            current_text = []
            section_title = None
            subsection_title = None
            subsubsection_title = None

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
                        subsection_title = None
                        subsubsection_title = None
                    elif level == 3:
                        subsection_title = title
                        subsubsection_title = None
                    elif level == 4:
                        subsubsection_title = title

                    if current_text:
                        payloads.extend(self._chunk_payload(
                            " ".join(current_text), url,
                            section_title, subsection_title, subsubsection_title, section_index
                        ))
                        current_text = []
                        print(f"[mediawiki] Flush: {current_text}")
                else:
                    current_text.append(str(node))
            print(f"[mediawiki] Current Text: {current_text}")
            if current_text:
                payloads.extend(self._chunk_payload(
                    "\n".join(current_text), url,
                    section_title, subsection_title, subsubsection_title, section_index
                ))
            section_index += 1

        return payloads

    def _chunk_payload(self, text, url, section, subsection, subsubsection, section_index):
        text = mwparserfromhell.parse(text).strip_code().strip()
        if not text:
            return []

        text = clean_mediawiki_text(text)
        chunks = self.splitter.split_text(text)
        total_chunks = len(chunks)
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
            "document_type": "mediawiki",
            "source": url,
            "section_index": section_index,
            "subsection_index": None,
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
        return text.strip()

    def _rest_base_from_api_url(self) -> str:
        """Derive REST base (scheme://host) from the configured Action API URL."""
        try:
            parsed = urlparse(self.api_url)
            return f"{parsed.scheme}://{parsed.netloc}"
        except Exception:
            return "https://en.wikipedia.org"

    def _fetch_parsoid_html(self, title: str) -> str:
        """Fetch Parsoid-rendered HTML for a page title via RESTBase."""
        base = self._rest_base_from_api_url()
        rest_url = f"{base}/api/rest_v1/page/html/{title.replace(' ', '_')}"
        headers = {"User-Agent": self.user_agent, "Accept": "text/html"}
        r = requests.get(rest_url, headers=headers, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        return r.text

    def _walk_parsoid_sections(self, html: str, base_url: str, skip_sections: Optional[List[str]]) -> List[Dict]:
        """Walk Parsoid HTML, group content by h2/h3/h4, and emit payloads.
        h2 → section, h3 → subsection, h4+ → subsubsection. Use heading ids for anchors.
        """
        soup = BeautifulSoup(html or "", "html.parser")
        root = soup.select_one("#mw-content-text") or soup

        payloads: List[Dict] = []
        section_index_counter = 0  # Lead = 0; first h2 = 1

        # Handle Lead: all content before the first h2
        first_h2 = root.find("h2")
        if first_h2:
            lead_nodes = []
            for sib in first_h2.find_previous_siblings():
                # previous_siblings goes backwards; we actually want content before first h2
                pass  # Parsoid rarely has significant prose before first h2 inside #mw-content-text
            # Simpler: collect from start up to first_h2 by iterating next siblings of a fake cursor
            cursor = root.contents[0] if root.contents else None
            collected_html = []
            for node in list(root.children):
                if getattr(node, 'name', None) == 'h2':
                    break
                collected_html.append(str(node))
            lead_text = self._html_to_text("".join(collected_html))
            if lead_text and (not skip_sections or not any(s.lower() == "lead" for s in skip_sections)):
                payloads.extend(self._chunk_payload(
                    lead_text,
                    base_url,
                    section="Lead",
                    subsection=None,
                    subsubsection=None,
                    section_index=0,
                ))

        # Iterate all h2/h3/h4 headings in order
        current_h2 = None
        current_h3 = None

        for h in root.find_all(["h2", "h3", "h4"]):
            level = int(h.name[1])  # 2, 3, or 4
            title = h.get_text(" ").strip()
            anchor = h.get("id") or ""

            # Skip filtered h2 sections (and implicitly their children)
            if level == 2 and skip_sections and any(title.lower() == s.lower() for s in skip_sections):
                current_h2 = None
                current_h3 = None
                # advance until the next h2 of same/higher level by just continuing; children won't be emitted
                continue

            if level == 2:
                current_h2 = title
                current_h3 = None
                section_index_counter += 1
            elif level == 3:
                current_h3 = title
            # level >= 4 → subsubsection title is `title`

            # Collect content until the next heading of same or higher level
            collected = []
            for sib in h.next_siblings:
                name = getattr(sib, 'name', None)
                if name in ("h2", "h3", "h4"):
                    # Stop at next heading
                    break
                collected.append(str(sib))

            text = self._html_to_text("".join(collected))
            if not text:
                continue

            if level == 2:
                sec_title = current_h2
                sub_title = None
                subsub_title = None
                sec_index = section_index_counter
            elif level == 3:
                sec_title = current_h2
                sub_title = current_h3
                subsub_title = None
                sec_index = section_index_counter
            else:  # level >= 4
                sec_title = current_h2
                sub_title = current_h3
                subsub_title = title
                sec_index = section_index_counter

            section_url = f"{base_url}#{anchor}" if anchor else base_url
            payloads.extend(self._chunk_payload(
                text,
                section_url,
                sec_title,
                sub_title,
                subsub_title,
                sec_index,
            ))

        return payloads

    def parse_from_title(self, title: str, url: Optional[str] = None, skip_sections: Optional[List[str]] = None) -> List[Dict]:
        """
        Build payloads using MediaWiki Action API per-section HTML (clean text, preserved hierarchy).
        """
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
            html_doc = self._fetch_parsoid_html(self.page_title)
            payloads = self._walk_parsoid_sections(html_doc, url, skip_sections)
            return payloads

        # Default: action_api per-section flow (existing behavior)
        sections = self._fetch_section_tree(self.page_title)

        payloads: List[Dict] = []
        # Lead content (section=0)
        lead_html = self._fetch_section_html(self.page_title, 0)
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

        current_h2 = None
        current_h3 = None
        section_index_counter = 0  # Lead = 0; first h2 = 1

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
                continue

            if level == 2:
                current_h2 = line
                current_h3 = None
                section_index_counter += 1
            elif level == 3:
                current_h3 = line
            # level >= 4 → handled below

            if not idx:
                continue
            html = self._fetch_section_html(self.page_title, int(idx))
            text = self._html_to_text(html)
            if not text:
                continue

            if level == 2:
                section_title = current_h2
                subsection_title = None
                subsubsection_title = None
                sec_index = section_index_counter
            elif level == 3:
                section_title = current_h2
                subsection_title = current_h3
                subsubsection_title = None
                sec_index = section_index_counter
            else:  # >= 4
                section_title = current_h2
                subsection_title = current_h3
                subsubsection_title = line
                sec_index = section_index_counter

            section_url = f"{url}#{anchor}" if anchor else url
            payloads.extend(self._chunk_payload(
                text,
                section_url,
                section_title,
                subsection_title,
                subsubsection_title,
                sec_index,
            ))

        return payloads

    def parse_from_url(self, url: str, skip_sections: Optional[List[str]] = None) -> List[Dict]:
        """
        Extract the page title from a Wikipedia URL and parse the page.
        """
        parsed_url = urlparse(url)
        path = parsed_url.path
        if not path.startswith("/wiki/"):
            raise ValueError("URL does not appear to be a standard Wikipedia article URL.")
        title_encoded = path[len("/wiki/"):]
        title = unquote(title_encoded).replace('_', ' ')
        return self.parse_from_title(title, url, skip_sections=skip_sections)
