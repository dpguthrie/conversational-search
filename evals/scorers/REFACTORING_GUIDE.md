# BaseEvaluator Refactoring Guide

This guide explains how to refactor your existing scorers to use the `BaseEvaluator` pattern, reducing code duplication and improving Braintrust integration.

## Benefits of BaseEvaluator

1. **Reduces code duplication**: Common LLM-as-judge logic centralized
2. **Better Braintrust integration**: Uses `autoevals.LLMClassifier` (Braintrust's recommended approach)
3. **Async support built-in**: Better parallelism for evaluations
4. **Consistent interface**: All scorers follow the same pattern
5. **Easier to test**: Mock the classifier once, test all scorers
6. **Type-safe**: Implements `Scorer` protocol correctly

## Quick Start

### 1. Basic Scorer (No Data Transformation)

For scorers that work directly with `output` and `expected`:

```python
from evals.scorers.base_evaluator import BaseEvaluator, scorer_wrapper

PROMPT = """Your evaluation prompt here.

OUTPUT:
{{{output}}}

EXPECTED:
{{{expected}}}

Respond with: "good", "fair", "poor"
"""

class MyEvaluator(BaseEvaluator):
    name = "my_evaluator"
    model = "gpt-4o"
    use_cot = True

    def get_choice_scores(self) -> dict[str, float]:
        return {"good": 1.0, "fair": 0.5, "poor": 0.0}

    def get_prompt_template(self) -> str:
        return PROMPT

# Create Braintrust-compatible scorer
my_scorer = scorer_wrapper(MyEvaluator)
```

### 2. Advanced Scorer (Custom Data Extraction)

For scorers that need to extract specific fields from structured output:

```python
class MyAdvancedEvaluator(BaseEvaluator):
    name = "my_advanced_evaluator"
    model = "gpt-4o"
    use_cot = True

    def get_choice_scores(self) -> dict[str, float]:
        return {"excellent": 1.0, "good": 0.75, "poor": 0.0}

    def get_prompt_template(self) -> str:
        return """Your prompt with {{{input}}}, {{{output}}}, {{{expected}}}"""

    def _run_eval_sync(self, output, expected=None, **kwargs):
        """Override to extract custom fields."""
        # If output is a dict, extract what you need
        if isinstance(output, dict):
            answer = output.get("answer", "")
            query = output.get("query", "")
            sources = output.get("sources_text", "")

            # Pass as kwargs to the parent
            kwargs["input"] = query
            kwargs["context"] = sources  # Custom field

            return super()._run_eval_sync(answer, expected, **kwargs)
        else:
            return super()._run_eval_sync(output, expected, **kwargs)

    async def _run_eval_async(self, output, expected=None, **kwargs):
        """Async version with same logic."""
        if isinstance(output, dict):
            answer = output.get("answer", "")
            query = output.get("query", "")
            sources = output.get("sources_text", "")
            kwargs["input"] = query
            kwargs["context"] = sources
            return await super()._run_eval_async(answer, expected, **kwargs)
        else:
            return await super()._run_eval_async(output, expected, **kwargs)

my_advanced_scorer = scorer_wrapper(MyAdvancedEvaluator)
```

## Migration Examples

### Before (Original Pattern)

```python
"""Factual accuracy scorer using LLM-as-judge."""
import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

class FactualAccuracyScorer:
    def __init__(self, model: str = "gpt-4"):
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model,
            temperature=0
        )

    def score(self, output: Dict[str, Any], expected: Dict[str, Any] = None):
        answer = output.get("answer", "")
        sources = output.get("sources", [])

        sources_text = "\\n\\n".join([...])

        judge_prompt = f"""Evaluate the factual accuracy...
        ANSWER: {answer}
        SOURCES: {sources_text}
        Output JSON: {{"accuracy_score": 0.0-1.0, ...}}
        """

        response = self.llm.invoke([SystemMessage(content=judge_prompt)])

        try:
            import json
            result = json.loads(response.content)
            return {
                "name": "factual_accuracy",
                "score": result.get("accuracy_score", 0.0),
                "metadata": {...}
            }
        except json.JSONDecodeError:
            return {"name": "factual_accuracy", "score": 0.0, ...}

def factual_accuracy_scorer(output, expected=None):
    scorer = FactualAccuracyScorer()
    return scorer.score(output, expected)
```

**Issues:**
- 50+ lines of boilerplate
- Manual JSON parsing
- Error handling repeated across scorers
- Direct OpenAI calls (not using Braintrust's infrastructure)
- No async support

### After (BaseEvaluator Pattern)

```python
"""Factual accuracy scorer using BaseEvaluator."""
from evals.scorers.base_evaluator import BaseEvaluator, scorer_wrapper

FACTUAL_ACCURACY_PROMPT = """Evaluate the factual accuracy of the answer.

ANSWER: {{{output}}}
SOURCES: {{{expected}}}

Respond with: "accurate", "partially_accurate", "inaccurate"
"""

class FactualAccuracyEvaluatorV2(BaseEvaluator):
    name = "factual_accuracy"
    model = "gpt-4o"
    use_cot = True

    def get_choice_scores(self) -> dict[str, float]:
        return {
            "accurate": 1.0,
            "partially_accurate": 0.5,
            "inaccurate": 0.0
        }

    def get_prompt_template(self) -> str:
        return FACTUAL_ACCURACY_PROMPT

factual_accuracy_scorer_v2 = scorer_wrapper(FactualAccuracyEvaluatorV2)
```

**Benefits:**
- ~20 lines (60% less code)
- No manual JSON parsing
- Built-in error handling
- Uses Braintrust's LLMClassifier
- Async support included
- Easier to test and maintain

## Using with Braintrust Eval

### Simple Usage

```python
from braintrust import Eval
from evals.scorers.factual_accuracy_v2 import factual_accuracy_scorer_v2
from evals.scorers.answer_completeness_v2 import answer_completeness_scorer_v2

result = Eval(
    project_name="my-project",
    data=test_cases,
    task=run_my_agent,
    scores=[
        factual_accuracy_scorer_v2,
        answer_completeness_scorer_v2,
        # Add more scorers...
    ],
)
```

### With Custom Configuration

```python
# Create evaluator with custom model
from evals.scorers.factual_accuracy_v2 import FactualAccuracyEvaluatorV2

custom_evaluator = FactualAccuracyEvaluatorV2(
    model="gpt-4o-mini",  # Use cheaper model
    use_cot=False         # Disable chain-of-thought for speed
)

result = Eval(
    project_name="my-project",
    data=test_cases,
    task=run_my_agent,
    scores=[custom_evaluator],  # Pass instance directly
)
```

## Prompt Template Guidelines

### Placeholder Format

Use Braintrust's triple-brace format:
- `{{{input}}}` - The input/query
- `{{{output}}}` - The output to evaluate
- `{{{expected}}}` - Expected/reference data

### Custom Fields

You can add custom fields via kwargs:

```python
def _run_eval_sync(self, output, expected=None, **kwargs):
    kwargs["sources"] = format_sources(output["sources"])
    kwargs["topic"] = output.get("topic", "")
    return super()._run_eval_sync(output["answer"], expected, **kwargs)
```

Then use in prompt:
```
SOURCES:
{{{sources}}}

TOPIC:
{{{topic}}}
```

## Testing Your Scorers

```python
import pytest
from evals.scorers.factual_accuracy_v2 import FactualAccuracyEvaluatorV2

def test_factual_accuracy_evaluator():
    evaluator = FactualAccuracyEvaluatorV2()

    output = {
        "query": "What is Python?",
        "answer": "Python is a programming language created by Guido van Rossum.",
        "sources_text": "Python was created by Guido van Rossum in 1991."
    }

    score = evaluator._run_eval_sync(output)

    assert score.name == "factual_accuracy"
    assert 0.0 <= score.score <= 1.0
    assert "reasoning" in score.metadata  # From use_cot=True
```

## Migration Checklist

For each existing scorer:

- [ ] Create new file: `{scorer_name}_v2.py`
- [ ] Define prompt template with `{{{placeholders}}}`
- [ ] Create evaluator class inheriting from `BaseEvaluator`
- [ ] Implement `get_choice_scores()` method
- [ ] Implement `get_prompt_template()` method
- [ ] Override `_run_eval_sync()` if you need custom data extraction
- [ ] Create scorer wrapper: `my_scorer_v2 = scorer_wrapper(MyEvaluator)`
- [ ] Test with sample data
- [ ] Update eval runner to use v2 scorer
- [ ] Compare results with original scorer
- [ ] Once validated, replace original

## Deterministic Scorers

For non-LLM scorers (like Citation Quality), you don't need `BaseEvaluator`. Keep them as-is:

```python
# Citation quality is deterministic - no LLM needed
class CitationQualityScorer:
    def score_coverage(self, answer: str) -> float:
        # Deterministic regex/heuristic logic
        ...
```

Only use `BaseEvaluator` for LLM-as-judge scorers.

## Advanced: Custom Classifier Options

```python
class MyEvaluator(BaseEvaluator):
    name = "my_evaluator"

    def __init__(self, **kwargs):
        # Pass custom options to LLMClassifier
        super().__init__(**kwargs)

        # Recreate classifier with custom options
        self._classifier = LLMClassifier(
            name=self.name,
            prompt_template=self.get_prompt_template(),
            choice_scores=self.get_choice_scores(),
            model=self.model,
            use_cot=self.use_cot,
            max_tokens=500,  # Custom option
            temperature=0.2,  # Custom option
        )
```

## Summary

**Key Benefits:**
- 📉 **60% less code** per scorer
- 🚀 **Better performance** with async support
- 🔧 **Easier maintenance** with centralized logic
- ✅ **Better Braintrust integration** using recommended patterns
- 🧪 **Easier testing** with shared infrastructure

**When to Use:**
- ✅ LLM-as-judge scorers (factual accuracy, completeness, etc.)
- ❌ Deterministic scorers (citation counting, regex-based, etc.)

**Next Steps:**
1. Start with simplest scorer (e.g., factual accuracy)
2. Migrate one scorer at a time
3. Validate results match original
4. Update eval runner
5. Remove old scorer once validated
