import httpx
import brotli
from bs4 import BeautifulSoup
from typing import List, Dict
from urllib.parse import urljoin
import asyncio, random

class WebCrawler:
    def __init__(self, base_url: str, force_crawl: bool = False):
        self.base_url = base_url

        self.visited_urls = set()
        self.force_crawl = force_crawl
        self.client = httpx.AsyncClient(
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/91.0.4472.114 Safari/537.36"
                ),
                "Accept-Encoding": "gzip, deflate, br"
            },
            transport=httpx.AsyncHTTPTransport(http2=True)
        )

    async def fetch_page(self, url: str) -> str:
        """Fetch a single page content"""
        if not self.force_crawl and url in self.visited_urls:
            return ""

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "DNT": "1",
            "Referer": self.base_url,
            "Upgrade-Insecure-Requests": "1",
        }

        try:
            response = await self.client.get(url, headers=headers)
            response.raise_for_status()
            print(f"[DEBUG] Content-Encoding: {response.headers.get('Content-Encoding')}")
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
        print(f"[DEBUG] Page data: {page_data}")
        # Find all links
        for link in soup.find_all('a', href=True):
            full_url = urljoin(self.base_url, link['href'])
            if full_url.startswith(self.base_url) and full_url not in self.visited_urls:
                page_data['links'].append(full_url)

        # Recursively crawl links with proper task management
        if depth > 0 and page_data['links']:
            # Create tasks for each link with delay
            tasks = []
            for link in page_data['links']:
                # Add a short randomized delay to respect target servers and avoid detection
                delay_task = asyncio.create_task(asyncio.sleep(random.uniform(1.5, 3.0)))
                crawl_task = asyncio.create_task(self.crawl(link, depth - 1))
                tasks.extend([delay_task, crawl_task])

            # Wait for all tasks to complete
            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                print(f"[DEBUG] Crawler tasks cancelled for {url}")
                # Cancel remaining tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()
                raise
            except Exception as e:
                print(f"[ERROR] Error crawling {url}: {str(e)}")
                raise

        print(f"[DEBUG] Crawled page title: {page_data['title']} with {len(page_data['links'])} links.")
        return [page_data]

    async def close(self):
        await self.client.aclose()
