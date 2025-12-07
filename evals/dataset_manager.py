"""Dataset management utilities for Braintrust.

Unified data structure for offline and online evals:
- Matches production trace format (from agent.run)
- Supports conversation context with chat_history
- Enables adding production examples back to eval dataset
"""

import json
from typing import Any, Dict, List, Optional

from braintrust import init_dataset
from dotenv import load_dotenv

load_dotenv()


class DatasetManager:
    """Manage Braintrust datasets with unified format."""

    def __init__(self, project_name: str = "conversational-search"):
        """Initialize dataset manager.

        Args:
            project_name: Braintrust project name
        """
        self.project_name = project_name

    def get_or_create_dataset(self, dataset_name: str):
        """Get existing dataset or create new one.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Braintrust dataset object
        """
        return init_dataset(
            project=self.project_name,
            name=dataset_name,
            description=f"Evaluation dataset for {self.project_name}",
        )

    def conversation_to_dataset_records(
        self, conversation: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convert a conversation to dataset records.

        Unified format matches production traces:
        - input: {"query": str, "chat_history": list of previous user queries}
        - expected: {"response": str, "sources": list} (optional for evals)
        - metadata: thread_id, topic, turn_index, turn_type, etc.

        Args:
            conversation: Conversation dict with turns

        Returns:
            List of dataset records (one per turn)
        """
        records = []
        chat_history = []  # Build up as we go through turns

        thread_id = conversation.get("thread_id", "unknown")
        topic = conversation.get("topic", "unknown")

        for turn_idx, turn in enumerate(conversation.get("turns", [])):
            query = turn["query"]
            turn_type = turn.get("type", "unknown")

            # Create record matching production trace format
            record = {
                "input": {
                    "query": query,
                    "chat_history": chat_history.copy(),  # Previous user queries
                },
                # expected output would go here if we have ground truth
                # "expected": {"response": "...", "sources": [...]},
                "metadata": {
                    "thread_id": thread_id,
                    "topic": topic,
                    "turn_index": turn_idx,
                    "turn_type": turn_type,
                },
            }

            records.append(record)

            # Add this query to chat history for next turn
            chat_history.append(query)

        return records

    def load_from_json(
        self, filepath: str, dataset_name: str, clear_existing: bool = False
    ):
        """Load conversations from JSON file into Braintrust dataset.

        Args:
            filepath: Path to JSON file with conversations
            dataset_name: Name of target dataset
            clear_existing: If True, clear dataset before adding records

        Returns:
            Braintrust dataset object
        """
        # Load conversations from file
        with open(filepath, "r") as f:
            conversations = json.load(f)

        # Get or create dataset
        dataset = self.get_or_create_dataset(dataset_name)

        # Convert and insert records
        all_records = []
        for conversation in conversations:
            records = self.conversation_to_dataset_records(conversation)
            all_records.extend(records)

        # Insert all records
        for record in all_records:
            dataset.insert(**record)

        # Flush to ensure writes complete
        dataset.flush()

        print(f"Loaded {len(all_records)} records from {len(conversations)} conversations")
        print(f"Dataset: {dataset_name}")

        return dataset

    def add_production_example(
        self,
        dataset_name: str,
        query: str,
        chat_history: List[str],
        response: str,
        sources: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add a production example to the dataset.

        Use this to add real production traces back to your eval dataset
        for testing edge cases or interesting examples.

        Args:
            dataset_name: Name of target dataset
            query: User query
            chat_history: Previous user queries in conversation
            response: Agent response
            sources: Retrieved sources
            metadata: Additional metadata (conversation_id, topic, etc.)
        """
        dataset = self.get_or_create_dataset(dataset_name)

        record = {
            "input": {"query": query, "chat_history": chat_history},
            "expected": {"response": response, "sources": sources},
            "metadata": metadata or {},
        }

        dataset.insert(**record)
        dataset.flush()

        print(f"Added production example to dataset: {dataset_name}")
        print(f"  Query: {query[:60]}...")

    def export_dataset_to_json(self, dataset_name: str, output_path: str):
        """Export Braintrust dataset to JSON file.

        Useful for version control or sharing datasets.

        Args:
            dataset_name: Name of source dataset
            output_path: Path to output JSON file
        """
        dataset = self.get_or_create_dataset(dataset_name)

        # Fetch all records
        records = []
        for record in dataset:
            records.append(
                {
                    "input": record.input,
                    "expected": record.expected if hasattr(record, "expected") else None,
                    "metadata": record.metadata if hasattr(record, "metadata") else {},
                }
            )

        # Write to file
        with open(output_path, "w") as f:
            json.dump(records, f, indent=2)

        print(f"Exported {len(records)} records to {output_path}")


def main():
    """Example usage."""
    manager = DatasetManager()

    # Load test data into Braintrust dataset
    print("Loading test data into Braintrust dataset...")
    dataset = manager.load_from_json(
        filepath="evals/test_data.json",
        dataset_name="conversational-search-eval-v1",
        clear_existing=False,
    )

    print("\nDataset ready for use in evaluations!")
    print("View in Braintrust dashboard:")
    print("  https://www.braintrust.dev")


if __name__ == "__main__":
    main()
