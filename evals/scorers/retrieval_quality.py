"""Retrieval quality scorer using precision@K and LLM-based recall."""
import json
import os
from typing import Dict, Any, List

from openai import OpenAI


class RetrievalQualityScorer:
    """Evaluate quality of retrieved documents."""

    def __init__(self, model: str = "gpt-4o-mini", k: int = 5):
        """Initialize scorer.

        Args:
            model: OpenAI model for judging
            k: Number of top results to evaluate (for Precision@K)
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.k = k

    def score_precision_at_k(self, sources: List[Dict], query: str) -> float:
        """Score Precision@K using LLM to judge relevance.

        Args:
            sources: Retrieved sources
            query: User query

        Returns:
            Precision@K score
        """
        if not sources:
            return 0.0

        # Take top-K sources
        top_k = sources[:self.k]

        # Use LLM to judge relevance of each source
        relevant_count = 0

        for source in top_k:
            snippet = source.get("snippet", "")
            title = source.get("title", "")

            judge_prompt = f"""Is this source relevant to the query?

QUERY: {query}

SOURCE:
Title: {title}
Content: {snippet}

Answer only "RELEVANT" or "NOT_RELEVANT"."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0,
            )

            content = response.choices[0].message.content or ""
            if "RELEVANT" in content.upper() and "NOT_RELEVANT" not in content.upper():
                relevant_count += 1

        precision = relevant_count / len(top_k)
        return precision

    def score_recall_approximation(self, sources: List[Dict], query: str) -> float:
        """Approximate recall by asking LLM if important info is missing.

        Args:
            sources: Retrieved sources
            query: User query

        Returns:
            Recall approximation score
        """
        if not sources:
            return 0.0

        sources_text = "\n\n".join([
            f"Source {i+1}: {s.get('snippet', '')}"
            for i, s in enumerate(sources)
        ])

        judge_prompt = f"""Given the query, evaluate if the retrieved sources contain sufficient information.

QUERY: {query}

RETRIEVED SOURCES:
{sources_text}

Are there obvious important aspects of the query that are NOT covered by these sources?

Output format (JSON):
{{
    "coverage_score": 0.0-1.0,
    "missing_topics": ["list of missing topics if any"],
    "reasoning": "brief explanation"
}}

Output only valid JSON."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0,
        )

        content = response.choices[0].message.content or ""

        try:
            result = json.loads(content)
            return result.get("coverage_score", 0.5)
        except json.JSONDecodeError:
            return 0.5  # Default if parsing fails

    def score(self, output: Dict[str, Any], expected: Dict[str, Any] = None) -> Dict[str, Any]:
        """Score retrieval quality.

        Args:
            output: Agent output with query and sources
            expected: Expected output (optional, may contain ground truth sources)

        Returns:
            Score dict with retrieval quality metrics
        """
        query = output.get("query", "")
        sources = output.get("sources", [])

        if not sources:
            return {
                "name": "retrieval_quality",
                "score": 0.0,
                "metadata": {"reason": "No sources retrieved"}
            }

        # Compute metrics
        precision = self.score_precision_at_k(sources, query)
        recall_approx = self.score_recall_approximation(sources, query)

        # F1-like composite score
        if precision + recall_approx == 0:
            composite = 0.0
        else:
            composite = 2 * (precision * recall_approx) / (precision + recall_approx)

        return {
            "name": "retrieval_quality",
            "score": composite,
            "metadata": {
                "precision_at_k": precision,
                "recall_approximation": recall_approx,
                "k": self.k,
                "num_sources": len(sources)
            }
        }


# Braintrust-compatible scorer function
def retrieval_quality_scorer(output, expected=None):
    """Braintrust-compatible wrapper for retrieval quality scorer."""
    scorer = RetrievalQualityScorer(k=5)
    return scorer.score(output, expected)
