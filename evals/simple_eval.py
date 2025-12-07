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

from braintrust import Eval, current_span, init_dataset
from dotenv import load_dotenv

# Import local scorers (hosted scorers require API keys configured in Braintrust)
from evals.scorers.answer_completeness import answer_completeness_scorer
from evals.scorers.citation_quality import citation_quality_scorer
from evals.scorers.factual_accuracy import factual_accuracy_scorer
from evals.scorers.retrieval_quality import retrieval_quality_scorer
from src.agent import ConversationalSearchAgent

# Load environment variables
load_dotenv()


def run_agent(input: dict) -> dict:
    """Run agent on a single query.

    Args:
        input: Input dict with 'query' key

    Returns:
        Dict with query, response, sources, and metrics
    """
    agent = ConversationalSearchAgent()

    # Extract query from input
    query = input.get("query", input.get("input", ""))

    # Run agent (no thread_id = single-turn query)
    response, thread_id = agent.run(query)

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
    data=init_dataset(
        project="conversational-search",
        name="conversational-search-eval-v1",
        _internal_btql={"filter": {"btql": "metadata.turn_index = 0"}},
    ),
    task=run_agent,
    scores=[  # type: ignore
        factual_accuracy_scorer,
        citation_quality_scorer,
        answer_completeness_scorer,
        retrieval_quality_scorer,
    ],
    experiment_name="simple_eval",
    metadata={
        "model": "gpt-4o-mini",
        "evaluation_type": "single_turn",
        "description": "First-turn queries from conversational-search-eval-v1 dataset",
        "filter": "metadata.turn_index == 0",
    },
)
