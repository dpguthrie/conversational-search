"""Answer completeness scorer using BaseEvaluator pattern."""

from evals.scorers.base_evaluator import BaseEvaluator, scorer_wrapper

COMPLETENESS_PROMPT = """Evaluate if the answer fully addresses all aspects of the user's query.

USER QUERY:
{{{input}}}

ANSWER:
{{{output}}}

Evaluation criteria:
1. Are all parts of the query addressed?
2. Is the depth of coverage appropriate?
3. Is important context included?
4. Are there obvious gaps or missing information?

Provide your reasoning, then respond with one of: "complete", "mostly_complete", "partially_complete", "incomplete"
"""


class AnswerCompletenessEvaluator(BaseEvaluator):
    """Evaluate if answer addresses all aspects of query.

    Uses the BaseEvaluator pattern with custom data extraction
    to handle the query/answer/sources structure.
    """

    name = "answer_completeness"
    model = "gpt-4o-mini"
    use_cot = True

    def get_choice_scores(self) -> dict[str, float]:
        """Map LLM choices to numeric scores."""
        return {
            "complete": 1.0,
            "mostly_complete": 0.75,
            "partially_complete": 0.5,
            "incomplete": 0.0,
        }

    def get_prompt_template(self) -> str:
        """Return the evaluation prompt."""
        return COMPLETENESS_PROMPT

    def _run_eval_sync(self, output, expected=None, **kwargs):
        """Override to extract query and answer from output dict.

        The output from our agent is a dict with:
        - query: the user's question
        - answer: the agent's response
        - sources: retrieved sources

        We need to map this to Braintrust's input/output format.
        """
        # Extract structured data
        if isinstance(output, dict):
            query = output.get("query", "")
            answer = output.get("response", "")

            # Pass query as 'input' and answer as 'output' to the classifier
            kwargs["input"] = query

            # Call parent with the answer as output
            return super()._run_eval_sync(answer, expected, **kwargs)
        else:
            # Fallback to direct call if output is just a string
            return super()._run_eval_sync(output, expected, **kwargs)

    async def _run_eval_async(self, output, expected=None, **kwargs):
        """Async version with same data extraction logic."""
        if isinstance(output, dict):
            query = output.get("query", "")
            answer = output.get("response", "")
            kwargs["input"] = query
            return await super()._run_eval_async(answer, expected, **kwargs)
        else:
            return await super()._run_eval_async(output, expected, **kwargs)


# Braintrust-compatible scorer function
answer_completeness_scorer = scorer_wrapper(AnswerCompletenessEvaluator)
