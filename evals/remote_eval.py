"""Simple single-turn evaluation using Braintrust Eval function.

This demonstrates the simplest way to run evaluations with Braintrust:
- Single-turn queries (no conversation context)
- Uses existing dataset filtered to first-turn queries only
- Uses Braintrust's Eval() function for easy setup
- Automatic parallelization and progress tracking

This is much simpler than run_evals.py but doesn't support multi-turn conversations.

Usage:
    uv run python evals/simple_eval.py
"""

from typing import Any

from braintrust import Eval, current_span
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Import local scorers (hosted scorers require API keys configured in Braintrust)
from evals.scorers.answer_completeness import answer_completeness_scorer
from evals.scorers.citation_quality import citation_quality_scorer
from evals.scorers.factual_accuracy import factual_accuracy_scorer
from evals.scorers.retrieval_quality import retrieval_quality_scorer
from src.agent import DEFAULT_SYSTEM_PROMPT, ConversationalSearchAgent

# Load environment variables
load_dotenv()


class SystemPromptParam(BaseModel):
    """System prompt parameter for conversational search agent."""

    value: str = Field(
        default=DEFAULT_SYSTEM_PROMPT,
        description="System prompt for the conversational search agent.",
    )


async def run_agent(input: dict, hooks: Any = None) -> dict:
    """Run agent on a single query.

    Args:
        input: Input dict with 'query' key

    Returns:
        Dict with query, response, sources, and metrics
    """

    params = hooks.parameters if hooks and hasattr(hooks, "parameters") else {}

    agent = ConversationalSearchAgent(system_prompt=params.get("system_prompt"))

    # Extract query from input
    query = input.get("query", input.get("input", ""))

    # Run agent (no thread_id = single-turn query)
    with current_span().start_span(name="run_agent") as span:
        response, thread_id = await agent.run_async(query)
        span.log(
            input={"query": query},
            output=response,
            metadata={"query": query},
        )

    # Get sources from state
    state = agent.get_state(thread_id)
    sources = state.get("sources", [])

    # Calculate and log custom metrics
    metrics = {
        "response_length": len(response),
        "response_word_count": len(response.split()),
        "source_count": len(sources),
        "query_length": len(query),
        "query_word_count": len(query.split()),
    }

    # Log metrics to current span
    current_span().log(metrics=metrics)

    return {
        "query": query,
        "response": response,
        "sources": sources,
    }


# Run evaluation with filtered dataset
Eval(
    name="conversational-search",
    data=[],
    task=run_agent,
    scores=[  # type: ignore
        factual_accuracy_scorer,
        citation_quality_scorer,
        answer_completeness_scorer,
        retrieval_quality_scorer,
    ],
    experiment_name="remote_eval",
    metadata={
        "model": "gpt-4o-mini",
        "evaluation_type": "single_turn",
        "description": "First-turn queries from conversational-search-eval-v1 dataset",
        "filter": "metadata.turn_index == 0",
    },
    parameters={"system_prompt": SystemPromptParam},
)
