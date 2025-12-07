"""Production demo with Braintrust tracing.

Simulates realistic production usage by generating random conversations
with multiple turns. Each conversation has a unique thread_id for proper
grouping in Braintrust traces.

This version supports parallel execution of conversations for improved performance.
"""

import argparse
import json
import os
import random
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock

from braintrust import init_logger
from dotenv import load_dotenv
from openai import OpenAI

from src.agent import ConversationalSearchAgent

# Load environment variables
load_dotenv()

# Initialize Braintrust logger
init_logger(
    project="conversational-search", api_key=os.environ.get("BRAINTRUST_API_KEY")
)


class ConversationGenerator:
    """Generate random realistic conversations for production demo."""

    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize conversation generator.

        Args:
            model: OpenAI model to use for generation (cheap model recommended)
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def generate_conversations_batch(
        self, num_conversations: int, turns_per_conversation: list[int]
    ) -> list[dict]:
        """Generate multiple conversations in a single API call.

        Args:
            num_conversations: Number of conversations to generate
            turns_per_conversation: List of turn counts for each conversation

        Returns:
            List of dicts with 'topic' and 'queries' keys
        """
        prompt = f"""Generate {num_conversations} diverse conversation topics with questions for a search engine.

For each conversation, provide:
1. A topic (current, searchable, specific enough for follow-ups)
2. Questions that build naturally on each other

Number of questions per conversation: {turns_per_conversation}

Requirements:
- Topics should span: news, tech, science, culture, sports, health, finance, education, etc.
- Questions should feel like a natural conversation flow
- First question is the initial query, subsequent ones are follow-ups
- Each question should be standalone but contextually related

Output format (JSON array):
[
  {{"topic": "topic 1", "queries": ["question 1", "question 2", ...]}},
  {{"topic": "topic 2", "queries": ["question 1", "question 2", "question 3", ...]}},
  ...
]

Generate exactly {num_conversations} conversations:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,  # High temperature for diversity
            max_tokens=4000,  # Increase for larger batches
        )

        content = response.choices[0].message.content.strip()

        # Parse JSON response
        try:
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            conversations = json.loads(content)

            # Ensure we have the right number of conversations
            if len(conversations) < num_conversations:
                print(
                    f"Warning: Only generated {len(conversations)}/{num_conversations} conversations"
                )

            return conversations[:num_conversations]
        except json.JSONDecodeError as e:
            print(f"Error parsing batch response: {e}")
            print(f"Response content: {content[:500]}...")
            # Fallback: Generate simple conversations
            return [
                {
                    "topic": f"Topic {i + 1}",
                    "queries": [f"Tell me about topic {i + 1}"]
                    * turns_per_conversation[i],
                }
                for i in range(num_conversations)
            ]


def run_conversation(
    agent: ConversationalSearchAgent,
    queries: list[str],
    thread_id: str,
    conv_num: int,
    total_convs: int,
    print_lock: Lock,
):
    """Run a multi-turn conversation with automatic tracing.

    Braintrust automatically captures traces through @traced decorators.
    Turns within a conversation are executed sequentially to maintain context.

    Args:
        agent: Agent instance (shared across threads, but thread-safe)
        queries: List of queries in conversation
        thread_id: Unique thread ID for grouping turns
        conv_num: Current conversation number
        total_convs: Total number of conversations
        print_lock: Lock for synchronized printing across threads

    Returns:
        dict: Conversation result with metadata
    """
    # Re-initialize Braintrust logger in this thread context
    init_logger(
        project="conversational-search", api_key=os.environ.get("BRAINTRUST_API_KEY")
    )

    results = {
        "conv_num": conv_num,
        "thread_id": thread_id,
        "num_turns": len(queries),
        "success": True,
        "errors": [],
    }

    with print_lock:
        print(f"\n{'=' * 80}")
        print(
            f"CONVERSATION {conv_num}/{total_convs} | Thread ID: {thread_id[:8]}... | {len(queries)} turns"
        )
        print(f"{'=' * 80}")

    for turn_num, query in enumerate(queries, 1):
        with print_lock:
            print(f"\n[Turn {turn_num}/{len(queries)}] Query: {query}")

        try:
            # Run agent - Braintrust automatically traces everything
            # The agent.run() method internally creates a span with format:
            # "Agent Search [{thread_id[:8]}]({turn_number})"
            response, _ = agent.run(query, thread_id=thread_id)

            # Print shortened response for visibility
            response_preview = (
                response[:150] + "..." if len(response) > 150 else response
            )
            with print_lock:
                print(f"[Turn {turn_num}/{len(queries)}] Response: {response_preview}")

        except Exception as e:
            error_msg = f"[Turn {turn_num}/{len(queries)}] ERROR: {e}"
            with print_lock:
                print(error_msg)
            results["errors"].append(str(e))
            results["success"] = False

    return results


def main(
    parallel: bool = True, max_workers: int = 10, num_conversations: int | None = None
):
    """Run production demo with random conversations.

    Args:
        parallel: If True, run conversations in parallel. If False, run sequentially.
        max_workers: Maximum number of parallel workers (only used if parallel=True)
        num_conversations: Number of conversations to generate (if None, random between 50-100)
    """
    print("=" * 80)
    print("PRODUCTION DEMO - Random Conversation Generator")
    print("=" * 80)
    print()

    # Validate API keys
    print("Checking API keys...")
    print(f"  OpenAI API Key: {'✓' if os.getenv('OPENAI_API_KEY') else '✗ MISSING'}")
    print(f"  Tavily API Key: {'✓' if os.getenv('TAVILY_API_KEY') else '✗ MISSING'}")
    print(
        f"  Braintrust API Key: {'✓' if os.getenv('BRAINTRUST_API_KEY') else '✗ MISSING'}"
    )
    print()

    if not all(
        [
            os.getenv("OPENAI_API_KEY"),
            os.getenv("TAVILY_API_KEY"),
            os.getenv("BRAINTRUST_API_KEY"),
        ]
    ):
        print("ERROR: Missing required API keys. Please set them in .env file.")
        return

    # Generate random parameters
    if num_conversations is None:
        num_conversations = random.randint(1, 20)

    print(f"Generating {num_conversations} random conversations...")
    print("Each conversation will have 1-5 turns")
    print(f"Execution mode: {'PARALLEL' if parallel else 'SEQUENTIAL'}")
    if parallel:
        print(f"Max parallel workers: {max_workers}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Initialize components
    agent = ConversationalSearchAgent()
    generator = ConversationGenerator()

    # Thread-safe lock for printing
    print_lock = Lock()

    # Pre-generate all conversation data in batches
    print("Generating conversation topics and queries...")

    # Generate random turn counts for all conversations
    turns_per_conversation = [random.randint(1, 5) for _ in range(num_conversations)]

    # Batch size for API calls (to avoid token limits)
    batch_size = 50
    all_generated = []

    for batch_start in range(0, num_conversations, batch_size):
        batch_end = min(batch_start + batch_size, num_conversations)
        batch_turns = turns_per_conversation[batch_start:batch_end]
        batch_count = len(batch_turns)

        print(f"  Generating batch {batch_start + 1}-{batch_end}...")
        batch_conversations = generator.generate_conversations_batch(
            num_conversations=batch_count, turns_per_conversation=batch_turns
        )
        all_generated.extend(batch_conversations)

    # Build final conversation list with thread IDs
    conversations = []
    for conv_num, conv_data in enumerate(all_generated, 1):
        thread_id = str(uuid.uuid4())
        conversations.append(
            {
                "conv_num": conv_num,
                "thread_id": thread_id,
                "queries": conv_data.get("queries", []),
                "topic": conv_data.get("topic", f"Topic {conv_num}"),
            }
        )

    print(f"✓ Generated {len(conversations)} conversations. Starting execution...\n")

    # Track results
    results = []
    start_time = datetime.now()

    if parallel:
        # Parallel execution using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all conversation tasks
            future_to_conv = {
                executor.submit(
                    run_conversation,
                    agent=agent,
                    queries=conv["queries"],
                    thread_id=conv["thread_id"],
                    conv_num=conv["conv_num"],
                    total_convs=num_conversations,
                    print_lock=print_lock,
                ): conv
                for conv in conversations
            }

            # Process completed conversations
            for future in as_completed(future_to_conv):
                conv = future_to_conv[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    with print_lock:
                        print(f"ERROR in conversation {conv['conv_num']}: {e}")
                    results.append(
                        {
                            "conv_num": conv["conv_num"],
                            "thread_id": conv["thread_id"],
                            "success": False,
                            "errors": [str(e)],
                        }
                    )
    else:
        # Sequential execution (original behavior)
        for conv in conversations:
            result = run_conversation(
                agent=agent,
                queries=conv["queries"],
                thread_id=conv["thread_id"],
                conv_num=conv["conv_num"],
                total_convs=num_conversations,
                print_lock=print_lock,
            )
            results.append(result)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Summary
    print()
    print("=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print(f"Generated {num_conversations} conversations")
    print(f"Execution time: {duration:.2f} seconds")
    print(f"Average time per conversation: {duration / num_conversations:.2f} seconds")
    print(
        f"Successful conversations: {sum(1 for r in results if r['success'])}/{len(results)}"
    )
    if any(not r["success"] for r in results):
        print(
            f"Failed conversations: {sum(1 for r in results if not r['success'])}/{len(results)}"
        )
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("View traces in Braintrust dashboard:")
    print("  https://www.braintrust.dev")
    print()
    print("Tips for exploring traces:")
    print("  - Filter by thread_id to see complete conversations")
    print("  - Group by thread_id to analyze conversation patterns")
    print("  - Check span names for turn numbers: 'Agent Search [thread](turn)'")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Production demo with Braintrust tracing and parallel execution"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run conversations sequentially instead of in parallel (default: parallel)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Maximum number of parallel workers (default: 10, only used in parallel mode)",
    )
    parser.add_argument(
        "--num-conversations",
        type=int,
        default=None,
        help="Number of conversations to generate (default: random between 50-100)",
    )

    args = parser.parse_args()

    main(
        parallel=not args.sequential,
        max_workers=args.workers,
        num_conversations=args.num_conversations,
    )
