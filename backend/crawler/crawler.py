import httpx
from bs4 import BeautifulSoup
from typing import List, Dict
from urllib.parse import urljoin

class WebCrawler:
    def __init__(self, base_url: str, force_crawl: bool = False):
        self.base_url = base_url

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
