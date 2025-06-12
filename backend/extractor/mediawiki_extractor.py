import mwparserfromhell
from typing import List, Dict, Optional
import requests
from urllib.parse import unquote, urlparse
import tiktoken
import logging

from backend.embeddings.text_splitter import TextSplitter


logger = logging.getLogger(__name__)


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
            "title": self.page_title,
            "description": self.page_description,
        } for idx, chunk in enumerate(chunks)]

    def parse_from_title(self, title: str, url: Optional[str] = None, skip_sections: Optional[List[str]] = None) -> List[Dict]:
        """
        Fetch wikitext from Wikipedia API given a page title, then parse it.
        """
        api_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "prop": "revisions",
            "rvprop": "content",
            "titles": title,
            "formatversion": "2"
        }
        response = requests.get(api_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", [])
        if not pages or "missing" in pages[0]:
            raise ValueError(f"Page titled '{title}' not found.")
        self.page_title = title
        self.page_description = pages[0].get("description", "")
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