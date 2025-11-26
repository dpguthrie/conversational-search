"""Factual accuracy scorer using BaseEvaluator pattern."""

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


# Braintrust-compatible scorer function
factual_accuracy_scorer = scorer_wrapper(FactualAccuracyEvaluator)


