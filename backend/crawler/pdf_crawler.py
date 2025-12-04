import httpx
import PyPDF2
import logging
from io import BytesIO
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class PDFCrawler:
    def __init__(self):
        self.client = httpx.Client()

    def fetch_pdf(self, url: str) -> Optional[bytes]:
        """Fetch a PDF file from URL"""
        try:
            #logger.debug("PDFCrawler: fetching %s", url)
            response = self.client.get(url)
            response.raise_for_status()
            logger.debug("PDFCrawler: HTTP %s for %s (bytes=%d)", response.status_code, url, len(response.content or b""))
            return response.content
        except httpx.RequestError as e:
            logger.exception("PDFCrawler: error fetching %s: %s", url, e)
            return None

    def extract_sections(self, pdf_content: bytes) -> Dict[str, Any]:
        """Extract sections from PDF content"""
        try:
            sections: List[Dict] = []
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))

            # Extract metadata
            metadata = getattr(pdf_reader, "metadata", None) or {}
            title = metadata.get('/Title') if isinstance(metadata, dict) else getattr(metadata, 'title', None)
            title = title or 'Untitled PDF'
            #logger.debug("PDFCrawler: pages=%d title=%s", len(pdf_reader.pages), title)

            # Extract sections by page
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text:
                    section = {
                        'title': f"Page {page_num + 1}",
                        'content': text,
                        'page_number': page_num + 1
                    }
                    sections.append(section)
                else:
                    logger.debug("PDFCrawler: empty text on page %d", page_num + 1)

            logger.info("PDFCrawler: extracted %d sections", len(sections))
            return {
                'title': title,
                'sections': sections
            }
        except Exception as e:
            logger.exception("PDFCrawler: error extracting sections: %s", e)
            return {
                'title': 'Untitled PDF',
                'sections': []
            }

    def crawl(self, url: str) -> List[Dict]:
        """Crawl a PDF document"""
        #logger.debug("PDFCrawler: crawl %s", url)
        pdf_content = self.fetch_pdf(url)
        if not pdf_content:
            logger.warning("PDFCrawler: no content fetched for %s", url)
            return []

        sections = self.extract_sections(pdf_content)
        result = [{
            'url': url,
            'title': sections.get('title', 'Untitled PDF'),
            'sections': sections.get('sections', [])
        }]
        #logger.info("PDFCrawler: crawl complete url=%s sections=%d", url, len(result[0]['sections']))
        return result
