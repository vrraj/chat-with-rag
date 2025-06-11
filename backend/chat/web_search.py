import requests
from typing import List, Dict
from backend.core.config import settings

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
        try:
            params = {
                "q": query,
                "format": "json",
                "pretty": 1
            }
            
            response = requests.get(f"{self.base_url}/?", params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract relevant results
            results = []
            for item in data.get("Results", [])[:num_results]:
                results.append({
                    "title": item.get("Text", ""),
                    "snippet": item.get("Snippet", ""),
                    "url": item.get("FirstURL", "")
                })
            
            return results
            
        except Exception as e:
            print(f"Error in web search: {e}")
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
        existing_urls = {item.get("url", "") for item in existing_context}
        
        # Get web search results
        search_results = self.search(query)
        
        # Filter out existing URLs
        new_results = [
            result for result in search_results 
            if result["url"] not in existing_urls
        ]
        
        return new_results[:3]  # Limit to 3 new results
