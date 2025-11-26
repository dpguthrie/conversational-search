"""Synthesis and citation utilities."""
import re
from typing import List, Dict, Tuple


def extract_citations(text: str) -> List[int]:
    """Extract citation numbers from text.

    Args:
        text: Text containing citations like [1], [2], etc.

    Returns:
        List of citation numbers found
    """
    pattern = r'\[(\d+)\]'
    matches = re.findall(pattern, text)
    return [int(m) for m in matches]


def validate_citations(text: str, num_sources: int) -> Tuple[bool, List[str]]:
    """Validate that all citations exist in sources.

    Args:
        text: Text with citations
        num_sources: Number of available sources

    Returns:
        (is_valid, list_of_errors)
    """
    citations = extract_citations(text)
    errors = []

    for cite_num in citations:
        if cite_num < 1 or cite_num > num_sources:
            errors.append(f"Citation [{cite_num}] out of range (1-{num_sources})")

    return len(errors) == 0, errors


def format_sources(sources: List[Dict[str, str]]) -> str:
    """Format sources list for display.

    Args:
        sources: List of source dicts with url, title, snippet

    Returns:
        Formatted string with numbered sources
    """
    if not sources:
        return ""

    formatted = "\n\nSources:\n"
    for i, source in enumerate(sources, 1):
        title = source.get("title", "Untitled")
        url = source.get("url", "")
        formatted += f"[{i}] {title}\n    {url}\n"

    return formatted


def create_synthesis_prompt(query: str, sources: List[Dict[str, str]]) -> str:
    """Create system prompt for answer synthesis.

    Args:
        query: User query
        sources: Retrieved sources

    Returns:
        Formatted prompt for OpenAI
    """
    sources_text = "\n\n".join([
        f"Source [{i+1}]:\nTitle: {s['title']}\nURL: {s['url']}\nContent: {s['snippet']}"
        for i, s in enumerate(sources)
    ])

    prompt = f"""You are a helpful research assistant. Answer the user's query using ONLY the provided sources.

CITATION RULES:
1. Cite sources using [1], [2], etc. format
2. Every factual claim MUST have a citation
3. Only cite sources that are provided below
4. If sources don't contain enough information, say so

USER QUERY: {query}

SOURCES:
{sources_text}

Provide a comprehensive answer with proper citations."""

    return prompt
