"""Factual accuracy scorer using BaseEvaluator pattern."""

from typing import Any
from evals.scorers.base_evaluator import BaseEvaluator, scorer_wrapper


# Prompt template using Braintrust's placeholder format
FACTUAL_ACCURACY_PROMPT = """Evaluate the factual accuracy of the answer against the provided sources.

ANSWER:
{{{output}}}

SOURCES:
{{{expected}}}

Evaluation criteria:
1. Are all factual claims supported by the sources?
2. Are there any contradictions with the sources?
3. Are there any likely hallucinations (made-up facts)?

Provide your reasoning, then respond with one of: "accurate", "partially_accurate", "inaccurate"
"""


class FactualAccuracyEvaluator(BaseEvaluator):
    """Evaluate factual accuracy of answers against sources.

    This version uses the BaseEvaluator pattern for cleaner code
    and better Braintrust integration.
    """

    name = "factual_accuracy"
    model = "gpt-4o"
    use_cot = True

    def get_choice_scores(self) -> dict[str, float]:
        """Map LLM choices to numeric scores."""
        return {
            "accurate": 1.0,
            "partially_accurate": 0.5,
            "inaccurate": 0.0,
        }

    def get_prompt_template(self) -> str:
        """Return the evaluation prompt."""
        return FACTUAL_ACCURACY_PROMPT

    def _run_eval_sync(self, output: Any, expected: Any = None, **kwargs: Any):
        """Override to extract answer and sources from output dict.

        The output from our agent is a dict with:
        - query: the user's question
        - answer: the agent's response
        - sources: retrieved sources

        We need to extract answer and format sources for the prompt.
        """
        if isinstance(output, dict):
            answer = output.get("response", "")
            sources = output.get("sources", [])

            # Format sources for the prompt
            sources_text = "\n\n".join([
                f"[{i+1}] {s.get('title', 'Untitled')}\n"
                f"URL: {s.get('url', 'N/A')}\n"
                f"Content: {s.get('snippet', '')}"
                for i, s in enumerate(sources)
            ])

            # If no sources, indicate that
            if not sources_text:
                sources_text = "No sources available."

            # Pass answer as output and formatted sources as expected
            return super()._run_eval_sync(answer, expected=sources_text, **kwargs)
        else:
            # Fallback to direct call if output is just a string
            return super()._run_eval_sync(output, expected, **kwargs)

    async def _run_eval_async(self, output: Any, expected: Any = None, **kwargs: Any):
        """Async version with same data extraction logic."""
        if isinstance(output, dict):
            answer = output.get("response", "")
            sources = output.get("sources", [])

            # Format sources for the prompt
            sources_text = "\n\n".join([
                f"[{i+1}] {s.get('title', 'Untitled')}\n"
                f"URL: {s.get('url', 'N/A')}\n"
                f"Content: {s.get('snippet', '')}"
                for i, s in enumerate(sources)
            ])

            # If no sources, indicate that
            if not sources_text:
                sources_text = "No sources available."

            # Pass answer as output and formatted sources as expected
            return await super()._run_eval_async(answer, expected=sources_text, **kwargs)
        else:
            # Fallback to direct call if output is just a string
            return await super()._run_eval_async(output, expected, **kwargs)


# Braintrust-compatible scorer function
factual_accuracy_scorer = scorer_wrapper(FactualAccuracyEvaluator)


