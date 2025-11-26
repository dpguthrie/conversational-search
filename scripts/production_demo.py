"""Production demo with Braintrust tracing."""
import os
from dotenv import load_dotenv
import braintrust
from src.agent import ConversationalSearchAgent

# Load environment variables
load_dotenv()

# Initialize Braintrust
braintrust.init(project="conversational-search")


def run_conversation(agent: ConversationalSearchAgent, queries: list[str], conv_id: str):
    """Run a multi-turn conversation with tracing.

    Args:
        agent: Agent instance
        queries: List of queries in conversation
        conv_id: Conversation ID for tracing
    """
    with braintrust.start_span(name="conversation", input={"id": conv_id}) as conv_span:
        state = None

        for i, query in enumerate(queries):
            print(f"\n{'='*60}")
            print(f"Query {i+1}: {query}")
            print(f"{'='*60}\n")

            with braintrust.start_span(
                name="query_turn",
                input={"query": query, "turn": i+1}
            ) as turn_span:
                # Run agent
                if state is None:
                    # First turn - initialize
                    response = agent.run(query)
                    # Get state for next turn (preserving context)
                    # Note: For true multi-turn, we'd need to modify run() to return state
                    # For demo purposes, each query is semi-independent
                else:
                    response = agent.run(query, conversation_state=state)

                print(response)
                print("\n")

                # Log turn output
                turn_span.log(output={"response": response})

        conv_span.log(output={"completed": True})


def main():
    """Run demo conversations."""
    print("Initializing Conversational Search Agent...")
    print(f"OpenAI API Key: {'✓' if os.getenv('OPENAI_API_KEY') else '✗'}")
    print(f"Tavily API Key: {'✓' if os.getenv('TAVILY_API_KEY') else '✗'}")
    print(f"Braintrust API Key: {'✓' if os.getenv('BRAINTRUST_API_KEY') else '✗'}")
    print("\n")

    agent = ConversationalSearchAgent()

    # Demo conversation 1: Quantum computing
    run_conversation(
        agent=agent,
        queries=[
            "What are the latest developments in quantum computing?",
            "How does this compare to classical computing?",
            "What companies are leading in this space?"
        ],
        conv_id="conv_quantum_001"
    )

    # Demo conversation 2: Current events
    run_conversation(
        agent=agent,
        queries=[
            "What are the major AI announcements from November 2024?",
            "Which companies made these announcements?"
        ],
        conv_id="conv_ai_news_001"
    )

    print("\n" + "="*60)
    print("Demo complete! Check Braintrust dashboard for traces:")
    print("https://www.braintrust.dev")
    print("="*60)


if __name__ == "__main__":
    main()
