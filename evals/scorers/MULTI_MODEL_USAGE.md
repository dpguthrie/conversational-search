# Using Multiple LLM Models with BaseEvaluator

`BaseEvaluator` supports **multiple LLM providers** through Braintrust's AI proxy, not just OpenAI!

## Supported Models

Via Braintrust's proxy, you can use:
- ✅ **OpenAI**: gpt-4o, gpt-4-turbo, gpt-3.5-turbo, etc.
- ✅ **Anthropic**: claude-3-5-sonnet-latest, claude-sonnet-4-5-20250929, etc.
- ✅ **Custom models** you've configured in Braintrust

## Setup

### 1. Set Your Braintrust API Key

```bash
export BRAINTRUST_API_KEY=your_braintrust_api_key_here
```

This enables the Braintrust proxy which routes to different providers.

### 2. Use Any Supported Model

You can specify the model in three ways:

#### Option A: Class-Level Default

```python
from evals.scorers.base_evaluator import BaseEvaluator, scorer_wrapper

class MyClaudeEvaluator(BaseEvaluator):
    name = "my_claude_evaluator"
    model = "claude-3-5-sonnet-latest"  # Set default to Claude
    use_cot = True

    def get_choice_scores(self) -> dict[str, float]:
        return {"good": 1.0, "bad": 0.0}

    def get_prompt_template(self) -> str:
        return "Evaluate: {{{output}}}"

my_scorer = scorer_wrapper(MyClaudeEvaluator)
```

#### Option B: Instance-Level Override

```python
from evals.scorers.factual_accuracy_v2 import FactualAccuracyEvaluatorV2

# Override at instantiation
claude_evaluator = FactualAccuracyEvaluatorV2(
    model="claude-3-5-sonnet-latest"
)

# Use in Eval
Eval(
    scores=[claude_evaluator],
    ...
)
```

#### Option C: Mix Multiple Models

```python
from evals.scorers.factual_accuracy_v2 import FactualAccuracyEvaluatorV2
from evals.scorers.answer_completeness_v2 import AnswerCompletenessEvaluatorV2

# Use Claude for factual accuracy (better reasoning)
factual_claude = FactualAccuracyEvaluatorV2(
    model="claude-3-5-sonnet-latest"
)

# Use GPT-4o for completeness (faster)
completeness_gpt = AnswerCompletenessEvaluatorV2(
    model="gpt-4o"
)

# Run both in same eval
Eval(
    scores=[factual_claude, completeness_gpt],
    ...
)
```

## Complete Example

```python
"""Example: Using Claude for evaluation."""
import os
from braintrust import Eval
from dotenv import load_dotenv

from evals.scorers.factual_accuracy_v2 import FactualAccuracyEvaluatorV2
from evals.scorers.answer_completeness_v2 import AnswerCompletenessEvaluatorV2
from src.agent import ConversationalSearchAgent

load_dotenv()

# Verify Braintrust API key is set
assert os.getenv("BRAINTRUST_API_KEY"), "Set BRAINTRUST_API_KEY to use Claude!"

def run_agent(case):
    agent = ConversationalSearchAgent()
    query = case["query"]
    response, state = agent.run_with_state(query)
    return {
        "query": query,
        "answer": response,
        "sources": state.get("sources", []),
    }

# Create Claude-powered evaluators
factual_accuracy_claude = FactualAccuracyEvaluatorV2(
    model="claude-3-5-sonnet-latest",
    use_cot=True  # Claude excels at chain-of-thought reasoning
)

completeness_claude = AnswerCompletenessEvaluatorV2(
    model="claude-3-5-sonnet-latest"
)

# Run evaluation with Claude
result = Eval(
    project_name="conversational-search",
    experiment_name="claude_evaluation",
    data=[
        {"query": "What is quantum computing?"},
        {"query": "How does machine learning work?"},
    ],
    task=run_agent,
    scores=[
        factual_accuracy_claude,
        completeness_claude,
    ],
    metadata={
        "evaluator_model": "claude-3-5-sonnet-latest",
        "agent_model": "gpt-4o",
    }
)

print(f"Evaluation complete! Results: {result}")
```

## Model Selection Guidelines

### When to Use Claude

**✅ Use Claude for:**
- **Complex reasoning tasks** - Claude excels at nuanced evaluation
- **Long contexts** - Claude has larger context windows
- **Chain-of-thought** - Claude provides better reasoning explanations
- **Factual accuracy checks** - Claude is thorough and careful

**Example:**
```python
# Use Claude for complex factual accuracy evaluation
factual_scorer = FactualAccuracyEvaluatorV2(
    model="claude-3-5-sonnet-latest",
    use_cot=True  # Get detailed reasoning
)
```

### When to Use GPT-4o

**✅ Use GPT-4o for:**
- **Speed** - GPT-4o is faster for simple evaluations
- **Cost** - OpenAI models may be cheaper depending on usage
- **Structured outputs** - GPT-4o is well-tested for JSON outputs
- **High-volume evals** - Better rate limits with OpenAI directly

**Example:**
```python
# Use GPT-4o for fast citation quality checks
citation_scorer = CitationQualityEvaluatorV2(
    model="gpt-4o",
    use_cot=False  # Skip reasoning for speed
)
```

### Model Mixing Strategy

**Recommended approach:**
```python
# Use Claude for high-stakes judgments
factual_accuracy = FactualAccuracyEvaluatorV2(
    model="claude-3-5-sonnet-latest"
)

# Use GPT-4o for simpler checks
citation_quality = CitationQualityEvaluatorV2(
    model="gpt-4o"
)

# Mix in same evaluation
Eval(scores=[factual_accuracy, citation_quality], ...)
```

## Cost Comparison

| Model | Speed | Cost (per 1K tokens) | Best For |
|-------|-------|---------------------|----------|
| claude-3-5-sonnet-latest | Medium | Medium | Complex reasoning, factual accuracy |
| gpt-4o | Fast | Low-Medium | General purpose, high volume |
| gpt-4-turbo | Medium | Medium | Balanced quality/cost |
| gpt-3.5-turbo | Very Fast | Very Low | Simple checks, prototyping |

## Troubleshooting

### "Model not found" Error

**Problem:** Getting 404 or model not found errors

**Solution:**
1. Verify `BRAINTRUST_API_KEY` is set: `echo $BRAINTRUST_API_KEY`
2. Check model name spelling: "claude-3-5-sonnet-latest" (not "claude-3.5")
3. Ensure you're using Braintrust proxy (not direct OpenAI API)

### Using Without Braintrust Proxy

If you want to use OpenAI directly (without Braintrust proxy):

```bash
# Use OpenAI API key directly
export OPENAI_API_KEY=your_openai_key

# Only OpenAI models will work without BRAINTRUST_API_KEY
```

### Custom Models

If you've configured custom models in Braintrust:

```python
# Use your custom fine-tuned model
evaluator = FactualAccuracyEvaluatorV2(
    model="my-custom-model-id"  # As configured in Braintrust
)
```

## Testing Different Models

Quick script to compare models:

```python
"""Compare different models on same evaluation."""
from evals.scorers.factual_accuracy_v2 import FactualAccuracyEvaluatorV2

models_to_test = [
    "gpt-4o",
    "claude-3-5-sonnet-latest",
    "gpt-4-turbo",
]

test_case = {
    "query": "What is Python?",
    "answer": "Python is a programming language.",
    "sources_text": "Python was created by Guido van Rossum in 1991."
}

for model in models_to_test:
    print(f"\n=== Testing {model} ===")
    evaluator = FactualAccuracyEvaluatorV2(model=model)
    score = evaluator._run_eval_sync(test_case)
    print(f"Score: {score.score}")
    print(f"Reasoning: {score.metadata.get('reasoning', 'N/A')}")
```

## Summary

- ✅ **Multiple providers supported** via Braintrust proxy
- ✅ **Easy to switch**: Just change the `model` parameter
- ✅ **Mix models**: Use different models for different scorers
- ✅ **Same code**: No changes to your BaseEvaluator subclasses
- ✅ **Cost optimization**: Choose cheaper/faster models where appropriate

**Key takeaway:** You're not limited to OpenAI! Use Claude, mix models, or configure custom models—all with the same `BaseEvaluator` code.
