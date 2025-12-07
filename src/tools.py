"""Tools for conversational search agent."""
import os
from typing import List, Dict, Optional, Literal
from tavily import TavilyClient
from datetime import datetime
from braintrust import traced


class TavilySearchTool:
    """Wrapper for Tavily search API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Tavily client.

        Args:
            api_key: Tavily API key (defaults to TAVILY_API_KEY env var)
        """
        self.client = TavilyClient(api_key=api_key or os.getenv("TAVILY_API_KEY"))

    @traced(type="tool", name="tavily_search")
    def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: Literal["basic", "advanced"] = "advanced"
    ) -> List[Dict[str, str]]:
        """Search the web using Tavily.

        Args:
            query: Search query
            max_results: Maximum number of results to return
            search_depth: "basic" or "advanced"

        Returns:
            List of sources with url, title, snippet, timestamp
        """
        response = self.client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_raw_content=False
        )

        # Format results into our source schema
        sources = []
        for result in response.get("results", []):
            sources.append({
                "url": result.get("url", ""),
                "title": result.get("title", ""),
                "snippet": result.get("content", ""),
                "timestamp": datetime.now().isoformat()
            })

        return sources
