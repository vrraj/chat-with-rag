import httpx
from bs4 import BeautifulSoup
from typing import List, Dict
from urllib.parse import urljoin

class WebCrawler:
    def __init__(self, base_url: str, skip_references: bool = True, force_crawl: bool = False):
        self.base_url = base_url
        self.skip_references = skip_references
        self.visited_urls = set()
        self.force_crawl = force_crawl
        self.client = httpx.AsyncClient(headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.114 Safari/537.36"
            )
        })

    async def fetch_page(self, url: str) -> str:
        """Fetch a single page content"""
        if not self.force_crawl and url in self.visited_urls:
            return ""

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.114 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": self.base_url,
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        try:
            response = await self.client.get(url, headers=headers)
            response.raise_for_status()
            self.visited_urls.add(url)
            print(f"[DEBUG] Successfully fetched page: {url}")
            return response.text
        except httpx.HTTPStatusError as e:
            print(f"403 or other error from server on {url}: {e}")
            return ""
        except httpx.RequestError as e:
            print(f"Error fetching {url}: {e}")
            return ""

    async def crawl(self, url: str, depth: int = 1) -> List[Dict]:
        """Crawl a website up to specified depth"""
        if depth <= 0:
            return []

        content = await self.fetch_page(url)
        if not content:
            return []

        soup = BeautifulSoup(content, 'html.parser')
        removed_reference_count = 0
        reference_excerpt = ""

        if self.skip_references:
            print("[DEBUG] skip_references is True: removing references from page content.")
            # Remove inline references like [1], [2], etc.
            references = soup.find_all("sup", class_="reference")
            removed_reference_count += len(references)
            for tag in references:
                tag.decompose()

            # Remove Wikipedia-style reference list sections by id
            reference_sections = soup.find_all(id="References")
            for div in reference_sections:
                reference_excerpt += div.get_text(separator="\n")[:1000]
                div.decompose()

            # Remove ordered lists of references
            reference_ols = soup.find_all("ol", class_="references")
            for ol in reference_ols:
                reference_excerpt += ol.get_text(separator="\n")[:1000]
                ol.decompose()

            # Look for headers like "References" and remove all following siblings until next section
            reference_headers = soup.find_all(["h2", "h3", "h4"], string=lambda s: s and "reference" in s.lower())
            for header in reference_headers:
                next_siblings = []
                sibling = header.find_next_sibling()
                while sibling and sibling.name not in ["h2", "h3", "h4"]:
                    next_siblings.append(sibling)
                    sibling = sibling.find_next_sibling()

                for tag in [header] + next_siblings:
                    reference_excerpt += tag.get_text(separator="\n")[:1000]
                    tag.decompose()

            print(f"[DEBUG] References removed. Count: {removed_reference_count}")
            print(f"[DEBUG] Reference excerpt (first 300 chars): {reference_excerpt[:300]}")
        
        # Extract content and metadata
        print(f"[DEBUG] Initial content length: {len(content)}")
        print(f"[DEBUG] Filtered content length: {len(str(soup))}")
        page_data = {
            'url': url,
            'content': str(soup),
            'title': soup.title.string if soup.title else "No title",
            'links': [],
            'removed_references_count': removed_reference_count,
            'reference_section_excerpt': reference_excerpt.strip()
        }

        # Find all links
        for link in soup.find_all('a', href=True):
            full_url = urljoin(self.base_url, link['href'])
            if full_url.startswith(self.base_url) and full_url not in self.visited_urls:
                page_data['links'].append(full_url)

        # Recursively crawl links
        for link in page_data['links']:
            await self.crawl(link, depth - 1)

        print(f"[DEBUG] Crawled page title: {page_data['title']} with {len(page_data['links'])} links.")
        return [page_data]

    async def close(self):
        await self.client.aclose()
