import requests
import logging
from typing import List, Dict
from backend.core.config import settings

logger = logging.getLogger(__name__)

class WebSearchClient:
    def __init__(self):
        self.base_url = "https://api.duckduckgo.com"
        
    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Perform a web search and return results
        Args:
            query: Search query
            num_results: Number of results to return
        Returns:
            List of search results with title, snippet, and URL
        """
        if settings.debug_verbose:
            maxc = int(getattr(settings, "debug_log_truncate_chars", 500))
            q_snip = query if len(query) <= maxc else (query[:maxc] + "…")
            logger.debug("[WEB] search query='%s' num_results=%d", q_snip, num_results)
        try:
            params = {
                "q": query,
                "format": "json",
                "pretty": 1,
                # Reduce HTML/redirect noise in responses API
                "no_redirect": 1,
                "no_html": 1,
            }
            
            response = requests.get(f"{self.base_url}/?", params=params)
            response.raise_for_status()
            logger.debug("[WEB] HTTP %s from %s", response.status_code, self.base_url)
            
            data = response.json()

            def _mk_item(text: str | None, url: str | None) -> Dict:
                t = (text or "").strip()
                title = t.split(" - ", 1)[0] if " - " in t else (t[:80] + ("…" if len(t) > 80 else ""))
                return {"title": title, "snippet": t, "url": (url or "").strip()}

            results: List[Dict] = []

            # 1) Prefer the abstract if present (often Wikipedia or similar)
            abs_url = (data.get("AbstractURL") or "").strip()
            abs_text = (data.get("AbstractText") or "").strip()
            if abs_url and abs_text:
                results.append(_mk_item(abs_text, abs_url))

            # 2) Direct Results list (often empty in DDG Instant Answer)
            for item in (data.get("Results") or []) or []:
                results.append(_mk_item(item.get("Text"), item.get("FirstURL")))

            # 3) RelatedTopics can contain either direct items or grouped topics
            def _extract_related(items):
                for it in items or []:
                    if "FirstURL" in it or "Text" in it:
                        yield _mk_item(it.get("Text"), it.get("FirstURL"))
                    # Grouped topics
                    topics = it.get("Topics") if isinstance(it, dict) else None
                    if isinstance(topics, list):
                        for sub in topics:
                            yield _mk_item(sub.get("Text"), sub.get("FirstURL"))

            results.extend(list(_extract_related(data.get("RelatedTopics"))))

            # Deduplicate by URL then by title
            seen_urls = set()
            deduped: List[Dict] = []
            for r in results:
                url = r.get("url", "")
                title = r.get("title", "")
                key = (url or title)
                if key and key not in seen_urls and url:
                    seen_urls.add(key)
                    deduped.append(r)

            final = deduped[: max(1, int(num_results))]
            logger.debug("[WEB] extracted %d results (final=%d)", len(results), len(final))
            return final
            
        except Exception as e:
            logger.exception("[WEB] Error in web search: %s", e)
            return []

    def get_additional_context(self, query: str, existing_context: List[Dict]) -> List[Dict]:
        """
        Get additional context from web search
        Args:
            query: Search query
            existing_context: Current context to avoid duplicates
        Returns:
            List of new context items
        """
        if settings.debug_verbose:
            maxc = int(getattr(settings, "debug_log_truncate_chars", 500))
            q_snip = query if len(query) <= maxc else (query[:maxc] + "…")
            logger.debug("[WEB] additional_context query='%s' existing=%d", q_snip, len(existing_context))
        existing_urls = {item.get("url", "") for item in existing_context}
        
        # Get web search results
        search_results = self.search(query)
        logger.debug("[WEB] search returned %d results", len(search_results))
        
        # Filter out existing URLs
        new_results = [
            result for result in search_results 
            if result["url"] not in existing_urls
        ]
        logger.debug("[WEB] new unique results=%d (cap 3)", len(new_results))
        
        return new_results[:3]  # Limit to 3 new results
