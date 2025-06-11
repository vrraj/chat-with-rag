import mwparserfromhell
from typing import List, Dict, Optional
import requests
from urllib.parse import unquote, urlparse
import tiktoken

from backend.embeddings.text_splitter import TextSplitter


class MediaWikiExtractor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, use_manual_splitter=True)

    def parse(self, wikitext: str, url: str, skip_sections: Optional[List[str]] = None) -> List[Dict]:
        """
        Parse wikitext into section/subsection chunks, split into token chunks, and return flattened payloads.
        """
        wikicode = mwparserfromhell.parse(wikitext)
        payloads = []
        sections = wikicode.get_sections(include_lead=True, levels=[2, 3])
        section_index = 0
        for section in sections:
            # Get section title
            section_title = section.filter_headings()[0].title.strip() if section.filter_headings() else "Lead"
            # Skip sections if specified
            if skip_sections and section_title in skip_sections:
                continue
            # Split into subsections (level 3)
            subsections = section.get_sections(include_lead=True, levels=[3])
            if len(subsections) > 1:
                # There are subsections
                subsection_index = 0
                for subsection in subsections:
                    subsection_title = subsection.filter_headings()[0].title.strip() if subsection.filter_headings() else None
                    print(f"DEBUG: Section: {section_title}, Subsection: {subsection_title}")
                    # Extract text (remove headings)
                    text = subsection.strip_code().strip()
                    if not text:
                        continue
                    chunks = self.splitter.split_text(text)
                    total_chunks = len(chunks)
                    for idx, chunk in enumerate(chunks):
                        payloads.append({
                            "text": chunk,
                            "section": str(section_title),
                            "subsection": str(subsection_title) if subsection_title else None,
                            "chunk_index": idx,
                            "total_chunks": total_chunks,
                            "url": url,
                            "document_type": "mediawiki",
                            "source": url,
                            "section_index": section_index,
                            "subsection_index": subsection_index,
                            "title": self.page_title,
                            "description": self.page_description,
                        })
                    subsection_index += 1
            else:
                # No subsections, chunk the whole section
                print(f"DEBUG: Section: {section_title}, No subsections")
                text = section.strip_code().strip()
                if not text:
                    continue
                chunks = self.splitter.split_text(text)
                total_chunks = len(chunks)
                for idx, chunk in enumerate(chunks):
                    payloads.append({
                        "text": chunk,
                        "section": str(section_title),
                        "subsection": None,
                        "chunk_index": idx,
                        "total_chunks": total_chunks,
                        "url": url,
                        "document_type": "mediawiki",
                        "source": url,
                        "section_index": section_index,
                        "subsection_index": None,
                        "title": self.page_title,
                        "description": self.page_description,
                    })
            section_index += 1
        return payloads

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