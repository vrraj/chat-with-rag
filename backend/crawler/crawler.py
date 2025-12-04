import httpx
import brotli
from bs4 import BeautifulSoup
from typing import List, Dict
from urllib.parse import urljoin
import asyncio, random
import logging
from backend.core.config import settings

logger = logging.getLogger(__name__)

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
            logger.debug("Crawler: skipping already-visited %s", url)
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
            logger.debug("Crawler: content-encoding=%s", response.headers.get('Content-Encoding'))
            self.visited_urls.add(url)
            logger.info("Crawler: fetched %s (len=%d)", url, len(response.text or ""))
            return response.text
        except httpx.HTTPStatusError as e:
            logger.warning("Crawler: HTTP error on %s: %s", url, e)
            return ""
        except httpx.RequestError as e:
            logger.error("Crawler: request error on %s: %s", url, e, exc_info=True)
            return ""

    async def crawl(self, url: str, depth: int = 1) -> List[Dict]:
        """Crawl a website up to specified depth"""
        logger.debug("Crawler: crawl url=%s depth=%d", url, depth)
        if depth <= 0:
            return []

        content = await self.fetch_page(url)
        if not content:
            return []

        soup = BeautifulSoup(content, 'html.parser')
        removed_reference_count = 0
        reference_excerpt = ""

        # Extract content and metadata
        logger.debug("Crawler: initial_len=%d filtered_len=%d", len(content), len(str(soup)))
        page_data = {
            'url': url,
            'content': str(soup),
            'title': soup.title.string if soup.title else "No title",
            'links': [],
            'removed_references_count': removed_reference_count,
            'reference_section_excerpt': reference_excerpt.strip()
        }
        if settings.debug_verbose:
            try:
                from pprint import pformat
                maxc = int(getattr(settings, "debug_log_truncate_chars", 500))
                logger.debug("Crawler: page_data %s", pformat({k: (v if k != 'content' else f'<html len {len(v)}>' ) for k, v in page_data.items()})[:maxc])
            except Exception:
                logger.debug("Crawler: page_data keys=%s", list(page_data.keys()))
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
                logger.debug("Crawler: tasks cancelled for %s", url)
                # Cancel remaining tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()
                raise
            except Exception as e:
                logger.exception("Crawler: error crawling %s: %s", url, e)
                raise

        logger.info("Crawler: crawled title='%s' links=%d", page_data.get('title'), len(page_data.get('links') or []))
        return [page_data]

    async def close(self):
        await self.client.aclose()
