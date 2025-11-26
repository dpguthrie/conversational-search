"""Main evaluation runner using Braintrust."""
import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
from braintrust import Eval
from src.agent import ConversationalSearchAgent
from evals.scorers.factual_accuracy import factual_accuracy_scorer
from evals.scorers.citation_quality import citation_quality_scorer
from evals.scorers.answer_completeness import answer_completeness_scorer
from evals.scorers.retrieval_quality import retrieval_quality_scorer

# Load environment variables
load_dotenv()


def load_test_data(filepath: str = "evals/test_data.json") -> list:
    """Load evaluation dataset.

    Args:
        filepath: Path to test data JSON

    Returns:
        List of test cases
    """
    with open(filepath, 'r') as f:
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
                "turn_type": turn.get("type", "unknown")
            })

    return test_cases


def run_agent_on_query(case: Dict[str, Any]) -> Dict[str, Any]:
    """Task function: run agent on a single query.

    Args:
        case: Test case with query

    Returns:
        Output dict with answer and sources
    """
    agent = ConversationalSearchAgent()

    query = case["query"]

    # Run agent with state
    response, final_state = agent.run_with_state(query)

    output = {
        "query": query,
        "answer": response,
        "sources": final_state.get("sources", []),
        "conversation_id": case["conversation_id"],
        "turn_index": case["turn_index"],
        "turn_type": case.get("turn_type", "unknown")
    }

    return output


def main():
    """Run evaluations with Braintrust."""
    print("Loading evaluation dataset...")
    test_cases = load_test_data()
    print(f"Loaded {len(test_cases)} test cases\n")

    print("Running evaluations with Braintrust...")
    print(f"Scorers: factual_accuracy, citation_quality, answer_completeness, retrieval_quality")
    print("\n")

    # Run Braintrust Eval
    result = Eval(
        project_name="conversational-search",
        experiment_name="initial_eval",
        data=test_cases,
        task=run_agent_on_query,
        scores=[
            factual_accuracy_scorer,
            citation_quality_scorer,
            answer_completeness_scorer,
            retrieval_quality_scorer
        ],
        metadata={
            "model": "gpt-4",
            "tavily_depth": "advanced",
            "description": "Initial evaluation of conversational search agent"
        }
    )

    print("\n" + "="*60)
    print("Evaluation complete!")
    print(f"Results: {result}")
    print("\nView detailed results in Braintrust dashboard:")
    print("https://www.braintrust.dev")
    print("="*60)


if __name__ == "__main__":
    main()
