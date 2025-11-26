"""Base evaluator for LLM-as-judge scorers."""

import abc
from typing import Any

from autoevals import LLMClassifier
from autoevals.score import Score, Scorer


class BaseEvaluator(Scorer, abc.ABC):
    """Base class for all LLM evaluators.

    Reduces code duplication by providing common LLM-as-judge infrastructure.
    Subclasses only need to define the prompt template and choice scores.
    """

    name: str = "Evaluator"
    model: str = "gpt-4o"
    use_cot: bool = True

    def __init__(self, model: str | None = None, use_cot: bool | None = None):
        """Initialize evaluator.

        Args:
            model: LLM model to use. Supports multiple providers via Braintrust proxy:
                  - OpenAI: "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", etc.
                  - Anthropic: "claude-3-5-sonnet-latest", "claude-sonnet-4-5-20250929", etc.
                  - Or any custom model configured in Braintrust
                  Defaults to class attribute.
            use_cot: Whether to use chain-of-thought (defaults to class attribute)

        Note: Set BRAINTRUST_API_KEY environment variable to use non-OpenAI models.
        """
        self.model = model or self.model
        self.use_cot = use_cot if use_cot is not None else self.use_cot

        # Create classifier once at instantiation
        self._classifier = LLMClassifier(
            name=self.name,
            prompt_template=self.get_prompt_template(),
            choice_scores=self.get_choice_scores(),
            model=self.model,
            use_cot=self.use_cot,
        )

    @abc.abstractmethod
    def get_choice_scores(self) -> dict[str, float]:
        """Return mapping of LLM choices to numeric scores.

        Example: {"correct": 1.0, "incorrect": 0.0}
        """
        pass

    @abc.abstractmethod
    def get_prompt_template(self) -> str:
        """Return the evaluation prompt template.

        Should use {{{input}}}, {{{output}}}, {{{expected}}} placeholders.
        """
        pass

    def _name(self) -> str:
        """Return scorer name for Braintrust."""
        return self.name

    def _run_eval_sync(
        self, output: Any, expected: Any = None, **kwargs: Any
    ) -> Score:
        """Synchronous evaluation.

        Args:
            output: The output to evaluate (typically the LLM's answer)
            expected: Expected/reference data (optional)
            **kwargs: Additional context (input, metadata, etc.)

        Returns:
            Score object with name, score (0-1), and metadata
        """
        # Call LLMClassifier
        response = self._classifier(output, expected, **kwargs)

        return Score(
            name=self.name,
            score=response.score,
            metadata=response.metadata,
        )

    async def _run_eval_async(
        self, output: Any, expected: Any = None, **kwargs: Any
    ) -> Score:
        """Async evaluation for better parallelism.

        Args:
            output: The output to evaluate
            expected: Expected/reference data (optional)
            **kwargs: Additional context

        Returns:
            Score object with name, score (0-1), and metadata
        """
        # Use async version
        response = await self._classifier.eval_async(output, expected, **kwargs)

        return Score(
            name=self.name,
            score=response.score,
            metadata=response.metadata,
        )


# Braintrust-compatible wrapper function
def scorer_wrapper(evaluator_class):
    """Wrap an evaluator class for use with Braintrust Eval().

    Usage:
        factual_accuracy_scorer = scorer_wrapper(FactualAccuracyEvaluator)

        Eval(
            scores=[factual_accuracy_scorer],
            ...
        )
    """
    instance = evaluator_class()
    return instance
