"""Eval runner using BaseEvaluator pattern with Braintrust.

This version uses the refactored scorers that inherit from BaseEvaluator,
providing cleaner code and better Braintrust integration.
"""

import json
import os
from typing import Any, Dict

from braintrust import Eval
from dotenv import load_dotenv

from evals.scorers.answer_completeness_v2 import answer_completeness_scorer_v2
from evals.scorers.factual_accuracy_v2 import factual_accuracy_scorer_v2
from src.agent import ConversationalSearchAgent

# Load environment variables
load_dotenv()


def load_test_data(filepath: str = "evals/test_data.json") -> list:
    """Load evaluation dataset.

    Args:
        filepath: Path to test data JSON

    Returns:
        List of test cases
    """
    with open(filepath, "r") as f:
        conversations = json.load(f)

    # Flatten conversations into individual test cases
    test_cases = []
    for conv in conversations:
        conv_id = conv["conversation_id"]
        topic = conv["topic"]

        for i, turn in enumerate(conv["turns"]):
            test_cases.append({
                "conversation_id": conv_id,
                "topic": topic,
                "turn_index": i,
                "query": turn["query"],
                "turn_type": turn.get("type", "unknown"),
            })

    return test_cases


def run_agent_on_query(case: Dict[str, Any]) -> Dict[str, Any]:
    """Task function: run agent on a single query.

    Args:
        case: Test case with query

    Returns:
        Output dict with query, answer, and sources
    """
    agent = ConversationalSearchAgent()

    query = case["query"]

    # Run agent with state to get sources
    response, final_state = agent.run_with_state(query)

    # Format sources for the evaluator
    sources = final_state.get("sources", [])
    sources_text = "\n\n".join(
        [
            f"Source {i+1}:\nTitle: {s['title']}\nURL: {s['url']}\nContent: {s['snippet']}"
            for i, s in enumerate(sources)
        ]
    )

    output = {
        "query": query,
        "answer": response,
        "sources": sources,
        "sources_text": sources_text,  # Formatted for LLM evaluation
        "conversation_id": case["conversation_id"],
        "turn_index": case["turn_index"],
        "turn_type": case.get("turn_type", "unknown"),
    }

    return output


def main():
    """Run evaluations with Braintrust using BaseEvaluator scorers."""
    print("Loading evaluation dataset...")
    test_cases = load_test_data()
    print(f"Loaded {len(test_cases)} test cases\n")

    print("Running evaluations with Braintrust...")
    print("Scorers: factual_accuracy_v2, answer_completeness_v2")
    print("\n")

    # Run Braintrust Eval with BaseEvaluator scorers
    result = Eval(
        project_name="conversational-search",
        experiment_name="base_evaluator_scorers",
        data=test_cases,
        task=run_agent_on_query,
        scores=[
            factual_accuracy_scorer_v2,
            answer_completeness_scorer_v2,
            # Add more BaseEvaluator scorers here
        ],
        metadata={
            "model": "gpt-4o",
            "tavily_depth": "advanced",
            "description": "Evaluation using BaseEvaluator pattern",
            "scorer_framework": "autoevals.LLMClassifier",
        },
    )

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"Results: {result}")
    print("\nView detailed results in Braintrust dashboard:")
    print("https://www.braintrust.dev")
    print("=" * 60)


if __name__ == "__main__":
    main()
