"""Answer completeness scorer using LLM-as-judge."""
import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage


class AnswerCompletenessScorer:
    """Evaluate if answer addresses all aspects of query."""

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
        """Score answer completeness.

        Args:
            output: Agent output with query and answer
            expected: Expected output (optional)

        Returns:
            Score dict with completeness score
        """
        query = output.get("query", "")
        answer = output.get("answer", "")

        if not query or not answer:
            return {
                "name": "answer_completeness",
                "score": 0.0,
                "metadata": {"error": "Missing query or answer"}
            }

        # LLM judging prompt
        judge_prompt = f"""Evaluate if the answer fully addresses all aspects of the user's query.

USER QUERY:
{query}

ANSWER:
{answer}

Evaluation criteria:
1. Are all parts of the query addressed?
2. Is the depth of coverage appropriate?
3. Is important context included?
4. Are there obvious gaps or missing information?

Output format (JSON):
{{
    "completeness_score": 0.0-1.0,
    "addressed_aspects": ["list of addressed aspects"],
    "missing_aspects": ["list of missing aspects"],
    "reasoning": "brief explanation"
}}

Output only valid JSON."""

        response = self.llm.invoke([SystemMessage(content=judge_prompt)])

        try:
            import json
            result = json.loads(response.content)

            return {
                "name": "answer_completeness",
                "score": result.get("completeness_score", 0.0),
                "metadata": {
                    "addressed_aspects": result.get("addressed_aspects", []),
                    "missing_aspects": result.get("missing_aspects", []),
                    "reasoning": result.get("reasoning", "")
                }
            }
        except json.JSONDecodeError:
            return {
                "name": "answer_completeness",
                "score": 0.0,
                "metadata": {
                    "error": "Failed to parse LLM judge response",
                    "raw_response": response.content
                }
            }


# Braintrust-compatible scorer function
def answer_completeness_scorer(output, expected=None):
    """Braintrust-compatible wrapper for answer completeness scorer."""
    scorer = AnswerCompletenessScorer()
    return scorer.score(output, expected)
