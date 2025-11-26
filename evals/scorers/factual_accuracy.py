"""Factual accuracy scorer using LLM-as-judge."""
import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage


class FactualAccuracyScorer:
    """Evaluate factual accuracy of answers against sources."""

    def __init__(self, model: str = "gpt-4"):
        """Initialize scorer.

        Args:
            model: OpenAI model for judging
        """
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model,
            temperature=0
        )

    def score(self, output: Dict[str, Any], expected: Dict[str, Any] = None) -> Dict[str, Any]:
        """Score factual accuracy of answer.

        Args:
            output: Agent output containing answer and sources
            expected: Expected output (optional, not used for this scorer)

        Returns:
            Score dict with accuracy score and reasoning
        """
        answer = output.get("answer", "")
        sources = output.get("sources", [])

        if not sources:
            return {
                "name": "factual_accuracy",
                "score": 0.5,
                "metadata": {
                    "reason": "No sources provided to verify against"
                }
            }

        # Build sources text
        sources_text = "\n\n".join([
            f"Source {i+1}:\n{s.get('snippet', '')}"
            for i, s in enumerate(sources)
        ])

        # LLM judging prompt
        judge_prompt = f"""Evaluate the factual accuracy of the answer against the provided sources.

ANSWER:
{answer}

SOURCES:
{sources_text}

Evaluation criteria:
1. Are all factual claims supported by the sources?
2. Are there any contradictions with the sources?
3. Are there any likely hallucinations (made-up facts)?

Output format (JSON):
{{
    "accuracy_score": 0.0-1.0,
    "unsupported_claims": ["list of unsupported claims"],
    "contradictions": ["list of contradictions"],
    "reasoning": "brief explanation"
}}

Output only valid JSON."""

        response = self.llm.invoke([SystemMessage(content=judge_prompt)])

        try:
            import json
            result = json.loads(response.content)

            return {
                "name": "factual_accuracy",
                "score": result.get("accuracy_score", 0.0),
                "metadata": {
                    "unsupported_claims": result.get("unsupported_claims", []),
                    "contradictions": result.get("contradictions", []),
                    "reasoning": result.get("reasoning", "")
                }
            }
        except json.JSONDecodeError:
            return {
                "name": "factual_accuracy",
                "score": 0.0,
                "metadata": {
                    "error": "Failed to parse LLM judge response",
                    "raw_response": response.content
                }
            }


# Braintrust-compatible scorer function
def factual_accuracy_scorer(output, expected=None):
    """Braintrust-compatible wrapper for factual accuracy scorer."""
    scorer = FactualAccuracyScorer()
    return scorer.score(output, expected)
