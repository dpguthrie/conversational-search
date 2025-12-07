"""Main evaluation runner using Braintrust with stateful conversations.

Uses Braintrust datasets to bridge offline and online evals:
- Dataset records match production trace format
- Can add production examples back to eval dataset
- Unified structure for continuous improvement
- Parallel conversation evaluation for speed
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from braintrust import init_dataset, init_experiment
from dotenv import load_dotenv

from evals.dataset_manager import DatasetManager
from evals.scorers.answer_completeness import answer_completeness_scorer
from evals.scorers.citation_quality import citation_quality_scorer
from evals.scorers.factual_accuracy import factual_accuracy_scorer
from evals.scorers.retrieval_quality import retrieval_quality_scorer
from src.agent import ConversationalSearchAgent

# Load environment variables
load_dotenv()


def load_conversations(filepath: str = "evals/test_data.json") -> List[Dict[str, Any]]:
    """Load evaluation dataset as conversations.

    Args:
        filepath: Path to test data JSON

    Returns:
        List of conversation objects with turns
    """
    with open(filepath, "r") as f:
        conversations = json.load(f)

    return conversations


def ensure_dataset_loaded(
    filepath: str = "evals/test_data.json",
    dataset_name: str = "conversational-search-eval-v1",
) -> Any:
    """Ensure data is loaded into Braintrust dataset.

    Args:
        filepath: Path to test data JSON
        dataset_name: Name of Braintrust dataset

    Returns:
        Braintrust dataset object
    """
    manager = DatasetManager()

    # Load data into dataset (init_dataset is get-or-create)
    print(f"Loading dataset: {dataset_name}")
    dataset = manager.load_from_json(filepath, dataset_name)

    return dataset


def run_scorers(output: Dict[str, Any], parent_span: Any = None) -> Dict[str, float]:
    """Run all scorers on an output with tracing (in parallel).

    Args:
        output: Agent output dict with answer and sources
        parent_span: Parent span to attach scorer spans to

    Returns:
        Dict of scorer name to score value
    """
    from braintrust import start_span

    scores = {}

    # Run each scorer with its own span for visibility
    scorers_list = [
        ("factual_accuracy", factual_accuracy_scorer),
        ("citation_quality", citation_quality_scorer),
        ("answer_completeness", answer_completeness_scorer),
        ("retrieval_quality", retrieval_quality_scorer),
    ]

    def run_single_scorer(scorer_name: str, scorer, output: Dict[str, Any], parent_span: Any):
        """Run a single scorer with error handling."""
        try:
            # Create span for scorer execution
            if parent_span:
                with parent_span.start_span(
                    name=f"scorer_{scorer_name}", span_attributes={"type": "score"}
                ) as scorer_span:
                    scorer_span.log(input={"output": output})
                    result = scorer(output)

                    # Handle both Score objects and dict returns
                    if isinstance(result, dict):
                        score_value = result["score"]
                        score_key = result["name"]
                        scorer_span.log(
                            output=result, scores={score_key: score_value}
                        )
                    else:
                        score_value = result.score
                        score_key = result.name
                        scorer_span.log(output=result, scores={score_key: score_value})

                    return (score_key, score_value)
            else:
                # Fallback if no parent span
                result = scorer(output)
                if isinstance(result, dict):
                    return (result["name"], result["score"])
                else:
                    return (result.name, result.score)

        except Exception as e:
            print(f"Warning: Scorer {scorer_name} failed: {e}")
            return (scorer_name, 0.0)

    # Run scorers in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(scorers_list)) as executor:
        futures = {
            executor.submit(run_single_scorer, scorer_name, scorer, output, parent_span): scorer_name
            for scorer_name, scorer in scorers_list
        }

        for future in as_completed(futures):
            score_key, score_value = future.result()
            scores[score_key] = score_value

    return scores


def evaluate_conversation(
    conversation: Dict[str, Any], experiment: Any
) -> List[Dict[str, Any]]:
    """Evaluate a single conversation with stateful context.

    Args:
        conversation: Conversation dict with topic and turns
        experiment: Braintrust experiment object

    Returns:
        List of results for each turn
    """
    agent = ConversationalSearchAgent()
    thread_id = None  # Start with no thread_id
    results = []

    test_thread_id = conversation["thread_id"]  # From test data (semantic ID)
    topic = conversation["topic"]

    for turn_idx, turn in enumerate(conversation["turns"]):
        query = turn["query"]
        turn_type = turn.get("type", "unknown")

        # Start a span for this turn (since agent uses @traced)
        with experiment.start_span(
            name=f"turn_{turn_idx}",
            input={"query": query, "turn_type": turn_type},
        ) as span:
            # Run agent with thread_id to maintain conversation context
            response, agent_thread_id = agent.run(query, thread_id=thread_id)
            thread_id = agent_thread_id  # Update for next turn

            # Get final state to extract sources
            state = agent.get_state(agent_thread_id)

            output = {
                "query": query,
                "response": response,
                "sources": state.get("sources", []),
                "thread_id": agent_thread_id,
            }

            # Run scorers with tracing (pass span for scorer sub-spans)
            scores = run_scorers(output, parent_span=span)

            # Calculate custom metrics
            metrics = {
                "response_length": len(response),
                "response_word_count": len(response.split()),
                "source_count": len(state.get("sources", [])),
                "query_length": len(query),
                "query_word_count": len(query.split()),
            }

            # Log to the span
            span.log(
                output=output,
                scores=scores,
                metrics=metrics,
                metadata={
                    "test_thread_id": test_thread_id,  # From test data
                    "agent_thread_id": agent_thread_id,  # Generated by agent
                    "topic": topic,
                    "turn_index": turn_idx,
                    "turn_type": turn_type,
                },
            )

        results.append(
            {
                "query": query,
                "output": output,
                "scores": scores,
                "turn_type": turn_type,
            }
        )

        # Print progress
        print(
            f"  Turn {turn_idx + 1}/{len(conversation['turns'])} ({turn_type}): {query[:60]}..."
        )

    return results


def main(use_braintrust_dataset: bool = True, max_concurrency: int = 5):
    """Run evaluations with Braintrust using stateful conversations.

    Args:
        use_braintrust_dataset: If True, load data into Braintrust dataset first
        max_concurrency: Maximum number of conversations to evaluate in parallel
    """
    print("=" * 60)
    print("STATEFUL CONVERSATIONAL EVALUATION")
    print("=" * 60)
    print()

    if use_braintrust_dataset:
        print("Ensuring data is loaded into Braintrust dataset...")
        dataset = ensure_dataset_loaded(
            filepath="evals/test_data.json",
            dataset_name="conversational-search-eval-v1",
        )
        print(f"✓ Dataset ready: conversational-search-eval-v1")
        print()

    print("Loading evaluation dataset...")
    conversations = load_conversations()
    total_turns = sum(len(conv["turns"]) for conv in conversations)
    print(f"Loaded {len(conversations)} conversations ({total_turns} total turns)")
    print()

    print("Initializing Braintrust experiment...")
    experiment = init_experiment(
        project="conversational-search",
        experiment="stateful_eval",
        metadata={
            "model": "gpt-4o-mini",
            "tavily_depth": "advanced",
            "evaluation_type": "stateful_conversations",
            "num_conversations": len(conversations),
            "total_turns": total_turns,
            "max_concurrency": max_concurrency,
        },
    )
    print(f"Experiment initialized: {experiment.id}")
    print()

    print("Running evaluations...")
    print(
        "Scorers: factual_accuracy, citation_quality, answer_completeness, retrieval_quality"
    )
    print(f"Concurrency: {max_concurrency} conversations in parallel")
    print()

    # Evaluate conversations in parallel
    all_results = []
    completed_count = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        # Submit all conversations
        future_to_conv = {
            executor.submit(evaluate_conversation, conversation, experiment): (
                idx,
                conversation,
            )
            for idx, conversation in enumerate(conversations, 1)
        }

        # Process as they complete
        for future in as_completed(future_to_conv):
            conv_idx, conversation = future_to_conv[future]
            completed_count += 1

            try:
                results = future.result()
                all_results.extend(results)

                elapsed = time.time() - start_time
                avg_time = elapsed / completed_count
                eta = avg_time * (len(conversations) - completed_count)

                print(
                    f"✓ [{completed_count}/{len(conversations)}] {conversation['topic'][:50]:<50} "
                    f"(ETA: {eta:.0f}s)"
                )
            except Exception as e:
                print(
                    f"✗ [{completed_count}/{len(conversations)}] {conversation['topic']}: {e}"
                )

    total_time = time.time() - start_time
    print()
    print(f"Parallel execution completed in {total_time:.1f}s")
    print(
        f"Average: {total_time / len(conversations):.1f}s per conversation, "
        f"{total_time / total_turns:.1f}s per turn"
    )
    print()

    # Finish experiment
    try:
        experiment.flush()
    except AttributeError:
        pass  # flush may not be available on all experiment types

    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Evaluated {len(conversations)} conversations")
    print(f"Total turns evaluated: {len(all_results)}")
    print()

    # Calculate average scores
    score_names = ["factual_accuracy", "citation_quality", "answer_completeness", "retrieval_quality"]
    print("Average Scores:")
    for score_name in score_names:
        scores = [r["scores"].get(score_name, 0.0) for r in all_results]
        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"  {score_name}: {avg:.4f}")
    print()

    print("View detailed results in Braintrust dashboard:")
    print(f"https://www.braintrust.dev/app/Braintrust%20Demos/p/conversational-search/experiments/{experiment.id}")
    print()
    print("💡 TIP: Add production examples to your eval dataset:")
    print("  from evals.dataset_manager import DatasetManager")
    print("  manager = DatasetManager()")
    print("  manager.add_production_example(...)")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run stateful conversational evaluations with Braintrust"
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help="Maximum number of conversations to evaluate in parallel (default: 5)",
    )
    parser.add_argument(
        "--no-dataset",
        action="store_true",
        help="Skip loading data into Braintrust dataset",
    )

    args = parser.parse_args()

    main(
        use_braintrust_dataset=not args.no_dataset,
        max_concurrency=args.max_concurrency,
    )
