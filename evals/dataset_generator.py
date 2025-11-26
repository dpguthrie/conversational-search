"""Generate synthetic evaluation dataset."""
import json
import os
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage


class ConversationGenerator:
    """Generate synthetic multi-turn conversations for evaluation."""

    def __init__(self, model: str = "gpt-4"):
        """Initialize generator.

        Args:
            model: OpenAI model to use
        """
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model,
            temperature=0.7  # Higher temp for diversity
        )

    def generate_conversation(self, topic: str, num_turns: int = 3) -> Dict:
        """Generate a single conversation.

        Args:
            topic: Conversation topic
            num_turns: Number of turns in conversation

        Returns:
            Conversation dict with turns
        """
        prompt = f"""Generate a realistic multi-turn conversation about {topic}.

Requirements:
1. First turn: Initial question about the topic
2. Subsequent turns: Follow-ups, clarifications, or related questions
3. Make it natural - like a real person exploring a topic
4. {num_turns} total user queries

Output format (JSON):
{{
    "topic": "{topic}",
    "turns": [
        {{"query": "first question", "type": "new_query"}},
        {{"query": "follow-up question", "type": "followup"}},
        {{"query": "clarification", "type": "clarification"}}
    ]
}}

Output only valid JSON, no other text."""

        response = self.llm.invoke([SystemMessage(content=prompt)])

        try:
            conversation = json.loads(response.content)
            return {
                "conversation_id": f"conv_{topic.lower().replace(' ', '_')}",
                **conversation
            }
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "conversation_id": f"conv_{topic.lower().replace(' ', '_')}",
                "topic": topic,
                "turns": [{"query": f"Tell me about {topic}", "type": "new_query"}]
            }

    def generate_dataset(
        self,
        topics: List[str],
        turns_per_conversation: int = 3
    ) -> List[Dict]:
        """Generate full evaluation dataset.

        Args:
            topics: List of topics to generate conversations for
            turns_per_conversation: Number of turns per conversation

        Returns:
            List of conversation dicts
        """
        dataset = []

        for topic in topics:
            print(f"Generating conversation for: {topic}")
            conv = self.generate_conversation(topic, turns_per_conversation)
            dataset.append(conv)

        return dataset

    def save_dataset(self, dataset: List[Dict], filepath: str):
        """Save dataset to JSON file.

        Args:
            dataset: Generated dataset
            filepath: Path to save file
        """
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)

        print(f"Dataset saved to {filepath}")


def main():
    """Generate evaluation dataset."""
    generator = ConversationGenerator()

    # Diverse topics across domains
    topics = [
        "quantum computing",
        "climate change solutions",
        "artificial intelligence safety",
        "space exploration",
        "renewable energy",
        "gene editing technology",
        "cryptocurrency regulation",
        "urban planning",
        "mental health treatments",
        "autonomous vehicles",
        "Mars colonization",
        "vaccine development",
        "sustainable agriculture",
        "neural networks",
        "ocean conservation",
        "educational technology",
        "blockchain applications",
        "carbon capture",
        "drug discovery",
        "smart cities"
    ]

    print("Generating synthetic evaluation dataset...")
    print(f"Topics: {len(topics)}")
    print(f"Turns per conversation: 3-4")
    print("\n")

    dataset = generator.generate_dataset(topics, turns_per_conversation=3)

    # Save dataset
    generator.save_dataset(dataset, "evals/test_data.json")

    print(f"\nGenerated {len(dataset)} conversations")
    print(f"Total turns: {sum(len(c['turns']) for c in dataset)}")


if __name__ == "__main__":
    main()
