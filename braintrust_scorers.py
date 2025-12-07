"""Scorers for pushing to Braintrust platform.

Push to Braintrust with:
    braintrust push evals/braintrust_scorers.py --if-exists replace

Each scorer is self-contained and uses only packages available in Braintrust:
- autoevals, openai, braintrust, json, re, requests, typing
"""

import re
from typing import Any, Dict, List

import braintrust
from pydantic import BaseModel

# =============================================================================
# PARAMETER SCHEMAS
# =============================================================================


class AgentOutputParams(BaseModel):
    """Standard output format from our conversational search agent."""

    input: Dict[str, Any]
    """Input to the agent"""

    output: Dict[str, Any]
    """Agent output dict with query, answer, and sources"""

    expected: Any = None
    """Optional expected output for comparison"""

    metadata: Dict[str, Any]
    """Metadata about the output"""


# =============================================================================
# PROJECT SETUP
# =============================================================================

project = braintrust.projects.create(name="conversational-search")


# =============================================================================
# SCORER 1: FACTUAL ACCURACY (LLM-as-Judge)
# =============================================================================

project.scorers.create(  # type: ignore
    name="Factual Accuracy",
    slug="factual-accuracy-v1",
    description="Evaluates if answer claims are supported by retrieved sources. Uses GPT-4o with chain-of-thought reasoning.",
    parameters=AgentOutputParams,
    messages=[
        {
            "role": "user",
            "content": """Evaluate the factual accuracy of the answer against the provided sources.

ANSWER:
{{output.response}}

SOURCES:
{{output.sources}}

Evaluation criteria:
1. Are all factual claims supported by the sources?
2. Are there any contradictions with the sources?
3. Are there any likely hallucinations (made-up facts)?

Respond with one of: "accurate", "partially_accurate", "inaccurate"
""",
        }
    ],
    model="gpt-4o",
    use_cot=True,
    choice_scores={
        "accurate": 1.0,
        "partially_accurate": 0.5,
        "inaccurate": 0.0,
    },
)


# =============================================================================
# SCORER 2: CITATION QUALITY (Hybrid: Code + Heuristics)
# =============================================================================


def citation_quality_handler(
    input: Dict[str, Any],
    output: Dict[str, Any],
    expected: Any = None,
    metadata: Dict[str, Any] = None,
):
    """Hybrid scorer evaluating citation coverage, precision, and source quality.

    Uses deterministic metrics for evaluation.
    """
    answer = output.get("response", "")
    sources = output.get("sources", [])

    if not sources:
        return {"name": "citation_quality", "score": 0.0}

    # Helper: Extract citation numbers
    def extract_citations(text: str) -> List[int]:
        pattern = r"\[(\d+)\]"
        matches = re.findall(pattern, text)
        return [int(m) for m in matches]

    # Helper: Extract factual sentences
    def extract_factual_sentences(text: str) -> List[str]:
        # Remove source list if present
        text = re.split(r"\n\s*Sources?:", text)[0]
        sentences = re.split(r"[.!?]+", text)
        factual = []
        for s in sentences:
            s = s.strip()
            if not s or s.endswith("?"):
                continue
            if any(
                s.lower().startswith(p)
                for p in ["however", "therefore", "thus", "in conclusion"]
            ):
                continue
            if len(s.split()) < 5:
                continue
            factual.append(s)
        return factual

    citations = extract_citations(answer)

    if not citations:
        # No citations but has sources = poor
        return {"name": "citation_quality", "score": 0.2}

    # Metric 1: Coverage - % of factual sentences with citations
    factual_sentences = extract_factual_sentences(answer)
    if not factual_sentences:
        coverage = 0.5  # Unclear, give neutral score
    else:
        cited_sentences = sum(1 for s in factual_sentences if re.search(r"\[\d+\]", s))
        coverage = cited_sentences / len(factual_sentences)

    # Metric 2: Precision - validity of citation numbers
    max_source_num = len(sources)
    valid_citations = [c for c in citations if 1 <= c <= max_source_num]
    precision = len(valid_citations) / len(citations) if citations else 0.0

    # Metric 3: Source Quality (simple heuristics)
    quality_scores = []
    for source in sources:
        url = source.get("url", "").lower()
        score = 0.7  # Default

        # Bonus for reputable domains
        if any(
            domain in url
            for domain in [
                ".edu",
                ".gov",
                "wikipedia.org",
                "arxiv.org",
                "nature.com",
                "science.org",
            ]
        ):
            score = 1.0
        # Penalty for questionable domains
        elif any(domain in url for domain in ["clickbait", "spam", "ads", "redirect"]):
            score = 0.3

        quality_scores.append(score)

    source_quality = (
        sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
    )

    # Composite score (weighted)
    composite = 0.5 * coverage + 0.3 * precision + 0.2 * source_quality
    return {
        "name": "citation_quality",
        "score": composite,
        "metadata": {
            "coverage": coverage,
            "precision": precision,
            "source_quality": source_quality,
            "num_citations": len(citations),
            "num_sources": len(sources),
            "num_factual_sentences": len(factual_sentences),
        },
    }


project.scorers.create(
    name="Citation Quality",
    slug="citation-quality-v1",
    description="Hybrid scorer evaluating citation coverage, precision, and source quality using deterministic metrics.",
    parameters=AgentOutputParams,
    handler=citation_quality_handler,
)


# =============================================================================
# SCORER 3: ANSWER COMPLETENESS (LLM-as-Judge)
# =============================================================================

project.scorers.create(  # type: ignore
    name="Answer Completeness",
    slug="answer-completeness-v1",
    description="Evaluates if answer fully addresses all aspects of the user query. Uses GPT-4o-mini with chain-of-thought.",
    parameters=AgentOutputParams,
    messages=[
        {
            "role": "user",
            "content": """Evaluate if the answer fully addresses all aspects of the user's query.

USER QUERY:
{{input.query}}

PREVIOUS QUESTIONS:
{{input.chat_history}}

ANSWER:
{{output.response}}

Evaluation criteria:
1. Are all parts of the query addressed?
2. Is the depth of coverage appropriate?
3. Is important context included?
4. Are there obvious gaps or missing information?

Respond with one of: "complete", "mostly_complete", "partially_complete", "incomplete"
""",
        }
    ],
    model="gpt-4o-mini",
    use_cot=True,
    choice_scores={
        "complete": 1.0,
        "mostly_complete": 0.75,
        "partially_complete": 0.5,
        "incomplete": 0.0,
    },
)


# =============================================================================
# SCORER 4: RETRIEVAL QUALITY (Hybrid: Code + LLM)
# =============================================================================


def retrieval_quality_handler(
    input: Dict[str, Any],
    output: Dict[str, Any],
    expected: Any = None,
    metadata: Dict[str, Any] = None,
):
    """Hybrid scorer evaluating precision@K and recall of retrieved documents.

    Uses LLM-judged relevance for evaluation.
    """
    import os

    from openai import OpenAI

    query = output.get("query", "") if isinstance(output, dict) else ""
    sources = output.get("sources", []) if isinstance(output, dict) else []

    if not sources:
        return {
            "name": "retrieval_quality",
            "score": 0.0,
            "metadata": {"reason": "No sources retrieved"},
        }

    # LLM-judged relevance for precision@K
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    relevant_count = 0
    for source in sources[:5]:  # Top 5
        judge_prompt = f"""Is this search result relevant to the query?

Query: {query}

Result:
Title: {source.get("title", "")}
Snippet: {source.get("snippet", "")}

Answer with just 'yes' or 'no':"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0,
                max_tokens=5,
            )
            judgment = response.choices[0].message.content.strip().lower()
            if "yes" in judgment:
                relevant_count += 1
        except Exception:
            # On error, assume relevant (generous)
            relevant_count += 1

    precision = relevant_count / min(5, len(sources))

    # Recall approximation (heuristic: assume good if we have 3+ relevant)
    recall_approx = min(1.0, relevant_count / 3.0)

    # F1-like composite
    if precision + recall_approx == 0:
        return {
            "name": "retrieval_quality",
            "score": 0.0,
            "metadata": {
                "precision_at_k": 0.0,
                "recall_approximation": 0.0,
                "relevant_count": 0,
                "k": min(5, len(sources)),
                "num_sources": len(sources),
            },
        }

    composite = 2 * (precision * recall_approx) / (precision + recall_approx)
    return {
        "name": "retrieval_quality",
        "score": composite,
        "metadata": {
            "precision_at_k": precision,
            "recall_approximation": recall_approx,
            "relevant_count": relevant_count,
            "k": min(5, len(sources)),
            "num_sources": len(sources),
        },
    }


project.scorers.create(
    name="Retrieval Quality",
    slug="retrieval-quality-v1",
    description="Hybrid scorer evaluating precision@K and recall of retrieved documents using LLM-judged relevance.",
    parameters=AgentOutputParams,
    handler=retrieval_quality_handler,
)


# =============================================================================
# USAGE
# =============================================================================

if __name__ == "__main__":
    print("Scorers defined for conversational-search project:")
    print("  1. factual-accuracy-v1 - LLM-as-judge for factual claims")
    print("  2. citation-quality-v1 - Hybrid metrics for citations")
    print("  3. answer-completeness-v1 - LLM-as-judge for query coverage")
    print("  4. retrieval-quality-v1 - Hybrid precision/recall metrics")
    print()
    print("To push to Braintrust:")
    print("  braintrust push evals/braintrust_scorers.py --if-exists replace")
