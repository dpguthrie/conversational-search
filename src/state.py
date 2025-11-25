"""Agent state schema for LangGraph."""
from typing import TypedDict, List, Dict
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """State for conversational search agent.

    Attributes:
        messages: Conversation history
        sources: Retrieved sources with metadata
        search_results: Raw Tavily API responses
        needs_search: Whether current query requires new search
        current_query: Current user query being processed
    """
    messages: List[BaseMessage]
    sources: List[Dict[str, str]]  # {url, title, snippet, timestamp}
    search_results: List[Dict]
    needs_search: bool
    current_query: str
