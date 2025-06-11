import httpx
import PyPDF2
from io import BytesIO
from typing import Dict, List

class PDFCrawler:
    def __init__(self):
        self.client = httpx.Client()

    def fetch_pdf(self, url: str) -> bytes:
        """Fetch a PDF file from URL"""
        try:
            response = self.client.get(url)
            response.raise_for_status()
            return response.content
        except httpx.RequestError as e:
            print(f"Error fetching PDF from {url}: {e}")
            return None

    def extract_sections(self, pdf_content: bytes) -> List[Dict]:
        """Extract sections from PDF content"""
        try:
            sections = []
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
            
            # Extract metadata
            metadata = pdf_reader.metadata
            title = metadata.get('/Title', 'Untitled PDF')
            
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
            
            return {
                'title': title,
                'sections': sections
            }
        except Exception as e:
            print(f"Error extracting sections from PDF: {e}")
            return []

    def crawl(self, url: str) -> List[Dict]:
        """Crawl a PDF document"""
        pdf_content = self.fetch_pdf(url)
        if not pdf_content:
            return []
            
        sections = self.extract_sections(pdf_content)
        return [{
            'url': url,
            'title': sections['title'],
            'sections': sections['sections']
        }]
