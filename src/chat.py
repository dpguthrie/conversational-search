#!/usr/bin/env python3
"""Interactive CLI chat interface for conversational search agent."""

import os
import sys
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.status import Status
from rich.table import Table
from braintrust import init_logger

from src.agent import ConversationalSearchAgent

# Load environment
load_dotenv()

# Initialize console
console = Console()


class ChatSession:
    """Manages a chat session with conversation history."""

    def __init__(self):
        """Initialize chat session."""
        self.agent: Optional[ConversationalSearchAgent] = None
        self.thread_id: Optional[str] = None
        self.turn_count = 0
        self.start_time = datetime.now()

        # Initialize Braintrust
        if os.getenv("BRAINTRUST_API_KEY"):
            init_logger(
                project="conversational-search",
                api_key=os.getenv("BRAINTRUST_API_KEY"),
            )

    def start(self):
        """Start the chat session."""
        self._print_welcome()
        self._initialize_agent()
        self._run_chat_loop()

    def _print_welcome(self):
        """Print welcome message."""
        console.clear()
        console.print()

        # Create title panel
        title = Panel.fit(
            "[bold cyan]Conversational Search Agent[/bold cyan]\n"
            "[dim]Powered by OpenAI + Tavily + Braintrust[/dim]",
            border_style="cyan",
        )
        console.print(title)
        console.print()

        # Print commands
        commands_table = Table(show_header=False, box=None, padding=(0, 2))
        commands_table.add_column(style="yellow bold")
        commands_table.add_column(style="dim")

        commands_table.add_row("/new", "Start a new conversation")
        commands_table.add_row("/help", "Show this help message")
        commands_table.add_row("/stats", "Show conversation statistics")
        commands_table.add_row("/exit", "Exit the chat")

        console.print(
            Panel(
                commands_table,
                title="[bold]Commands[/bold]",
                border_style="blue",
                padding=(1, 2),
            )
        )
        console.print()

    def _initialize_agent(self):
        """Initialize the agent with API key checks."""
        # Check API keys
        missing_keys = []
        if not os.getenv("OPENAI_API_KEY"):
            missing_keys.append("OPENAI_API_KEY")
        if not os.getenv("TAVILY_API_KEY"):
            missing_keys.append("TAVILY_API_KEY")

        if missing_keys:
            console.print(
                f"[bold red]Error:[/bold red] Missing API keys: {', '.join(missing_keys)}",
                style="red",
            )
            console.print(
                "\n[dim]Set these in your .env file or environment variables[/dim]"
            )
            sys.exit(1)

        # Initialize agent
        with Status("[cyan]Initializing agent...", console=console):
            self.agent = ConversationalSearchAgent()

        console.print("[green]✓[/green] Agent initialized successfully")
        console.print()

    def _run_chat_loop(self):
        """Main chat loop."""
        while True:
            try:
                # Get user input
                user_input = Prompt.ask(
                    "[bold cyan]You[/bold cyan]",
                    console=console,
                )

                # Handle empty input
                if not user_input.strip():
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    if self._handle_command(user_input):
                        continue
                    else:
                        break  # Exit

                # Process query
                self._process_query(user_input)

            except KeyboardInterrupt:
                console.print("\n\n[yellow]Use /exit to quit[/yellow]")
                continue
            except EOFError:
                break

        self._print_goodbye()

    def _handle_command(self, command: str) -> bool:
        """Handle slash commands.

        Args:
            command: Command string

        Returns:
            True to continue, False to exit
        """
        cmd = command.lower().strip()

        if cmd == "/exit" or cmd == "/quit":
            return False

        elif cmd == "/new":
            self._start_new_conversation()

        elif cmd == "/help":
            self._print_welcome()

        elif cmd == "/stats":
            self._print_stats()

        else:
            console.print(f"[red]Unknown command:[/red] {command}")
            console.print("[dim]Type /help to see available commands[/dim]")

        return True

    def _start_new_conversation(self):
        """Start a new conversation."""
        console.print()
        console.print(Rule("[bold cyan]New Conversation[/bold cyan]"))
        console.print()

        self.thread_id = None
        self.turn_count = 0
        self.start_time = datetime.now()

        console.print("[green]✓[/green] Started new conversation")
        console.print()

    def _print_stats(self):
        """Print conversation statistics."""
        console.print()

        # Get state if we have a thread
        state = None
        if self.thread_id and self.agent:
            state = self.agent.get_state(self.thread_id)

        # Create stats table
        stats = Table(show_header=False, box=None, padding=(0, 2))
        stats.add_column(style="cyan bold")
        stats.add_column(style="white")

        stats.add_row("Thread ID", self.thread_id or "[dim]No active conversation[/dim]")
        stats.add_row("Turns", str(self.turn_count))

        if state:
            stats.add_row("Messages", str(len(state["messages"])))
            stats.add_row("Sources Retrieved", str(len(state["sources"])))

        duration = datetime.now() - self.start_time
        stats.add_row("Duration", f"{duration.seconds // 60}m {duration.seconds % 60}s")

        console.print(
            Panel(
                stats,
                title="[bold]Conversation Statistics[/bold]",
                border_style="cyan",
                padding=(1, 2),
            )
        )
        console.print()

    def _process_query(self, query: str):
        """Process a user query.

        Args:
            query: User query string
        """
        console.print()

        # Show thinking status
        with Status("[cyan]Thinking...", console=console, spinner="dots"):
            try:
                response, thread_id = self.agent.run(query, thread_id=self.thread_id)
                self.thread_id = thread_id
                self.turn_count += 1

            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {e}")
                console.print("[dim]Please try again or check your API keys[/dim]")
                console.print()
                return

        # Print response
        console.print(Rule("[bold green]Assistant[/bold green]", style="green"))
        console.print()

        # Render as markdown for better formatting
        md = Markdown(response)
        console.print(md)

        console.print()

    def _print_goodbye(self):
        """Print goodbye message."""
        console.print()
        console.print(Rule())

        if self.turn_count > 0:
            console.print(
                f"\n[cyan]Session Summary:[/cyan] {self.turn_count} turns",
                style="dim",
            )

        console.print("\n[bold cyan]Thanks for using Conversational Search![/bold cyan]")
        console.print()


def main():
    """Main entry point."""
    try:
        session = ChatSession()
        session.start()
    except Exception as e:
        console.print(f"\n[bold red]Fatal error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
