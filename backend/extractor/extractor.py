import trafilatura
from bs4 import BeautifulSoup
from typing import Dict, List
from urllib.parse import urlparse

class ContentExtractor:
    @staticmethod
    def extract_content(html_content: str, url: str) -> Dict:
        """Extract meaningful content from HTML using trafilatura"""
        try:
            extracted = trafilatura.extract(
                html_content,
                include_comments=False,
                include_tables=False,
                include_images=False,
                include_formatting=False
            )
            
            if not extracted:
                return {}

            # Parse HTML for additional metadata
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title
            title = soup.title.string if soup.title else ""
            
            # Extract headers
            headers = []
            for header in soup.find_all(['h1', 'h2', 'h3']):
                headers.append(header.get_text(strip=True))
            
            # Extract meta description
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

            from trafilatura import load_html
            from trafilatura.metadata import extract_metadata

            downloaded = load_html(html_content)
            metadata = extract_metadata(downloaded)
            date_published = metadata.get("date") if isinstance(metadata, dict) else None
            if not date_published:
                time_tag = soup.find('time')
                if time_tag and time_tag.has_attr('datetime'):
                    date_published = time_tag['datetime']

            # Supplement with htmldate if date is still missing
            if not date_published:
                try:
                    from htmldate import find_date
                    date_published = find_date(html_content)
                except Exception:
                    pass
            
            extracted_data = {
                'title': title,
                'text': extracted,
                'headers': headers,
                'description': meta_description,
                'date': date_published,
                'url': url,
                'document_type': 'HTML',
                'domain': urlparse(url).netloc
            }
            return extracted_data
        except Exception as e:
            print(f"Error extracting content: {e}")
            return {}
