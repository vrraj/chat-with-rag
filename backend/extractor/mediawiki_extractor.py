import mwparserfromhell
from typing import List, Dict, Optional
import requests
import time
from requests import Response
from urllib.parse import unquote, urlparse
import tiktoken
import logging

from backend.embeddings.text_splitter import TextSplitter


logger = logging.getLogger(__name__)

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
WIKI_UA = "WebsiteChatAgent/0.1 (contact: you@example.com)"
DEFAULT_TIMEOUT = 15
MAX_RETRIES = 5

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

def _wiki_get(params: Dict) -> Response:
    """Resilient GET to Wikipedia Action API with UA + backoff."""
    backoff = 0.5
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(
                WIKI_API_URL,
                params=params,
                headers={"User-Agent": WIKI_UA},
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

class MediaWikiExtractor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, use_manual_splitter=True)

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
                            "\n".join(current_text), url,
                            section_title, subsection_title, subsubsection_title, section_index
                        ))
                        current_text = []
                else:
                    current_text.append(str(node))

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
            "description": description,
        } for idx, chunk in enumerate(chunks)]

    def parse_from_title(self, title: str, url: Optional[str] = None, skip_sections: Optional[List[str]] = None) -> List[Dict]:
        """
        Fetch wikitext from Wikipedia API given a page title, then parse it.
        """
        params = {
            "action": "query",
            "format": "json",
            "prop": "revisions|description",
            "rvprop": "content",
            "titles": title,
            "formatversion": "2",
        }
        response = _wiki_get(params)
        data = response.json()
        pages = data.get("query", {}).get("pages", [])
        if not pages or "missing" in pages[0]:
            raise ValueError(f"Page titled '{title}' not found.")
        self.page_title = title
        self.page_description = pages[0].get("description") or ""
        wikitext = pages[0].get("revisions", [{}])[0].get("content", "")
        if url is None:
            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        return self.parse(wikitext, url, skip_sections=skip_sections)

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