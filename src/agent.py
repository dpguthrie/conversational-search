"""Conversational search agent using minimal primitives (no LangChain)."""

import asyncio
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional, Union

from braintrust import current_span, start_span, traced
from braintrust.oai import wrap_openai
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from src.tools import TavilySearchTool

DEFAULT_SYSTEM_PROMPT = f"""You are a conversational search assistant. Your job is to answer user questions by searching the web when needed and providing cited answers.

INSTRUCTIONS:
1. If you need current information or facts, use the search_web tool
2. After searching, synthesize the information with inline citations [1], [2], etc.
3. Always cite sources at the end of your response in the format:

Sources:
[1] Title - URL
[2] Title - URL

4. If the query is a follow-up, use conversation context
5. Only search when you actually need fresh information
6. If you have enough information from previous searches, answer directly

CITATION RULES:
- Every factual claim MUST have an inline citation [N]
- Citation numbers correspond to the source list at the end
- Only cite sources that were returned from search_web tool
- If sources don't contain enough information, say so clearly

Always be aware of the current date and time.
{datetime.now().isoformat()}
"""


# Message Types (minimal primitives)
@dataclass
class Message:
    """Base message class."""

    content: str
    role: Literal["system", "user", "assistant", "tool"] = "user"


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""

    id: str
    name: str
    arguments: dict


@dataclass
class AssistantMessage:
    """Assistant message with optional tool calls."""

    content: str
    role: Literal["assistant"] = "assistant"
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class ToolMessage:
    """Tool execution result."""

    content: str
    role: Literal["tool"] = "tool"
    tool_call_id: str = ""
    name: str = ""


# Tool Definition (Pydantic BaseModel pattern from bitswired)
class Tool(BaseModel):
    """Base tool class."""

    async def __call__(self) -> str:
        raise NotImplementedError


class SearchWebTool(Tool):
    """Search the web for current information and facts.

    Returns search results with sources including title, URL, and content snippets.
    """

    query: str = Field(description="The search query string")

    def __init__(self, tavily_client: TavilySearchTool, **data):
        super().__init__(**data)
        self._tavily = tavily_client

    @traced(type="task", name="search_web")
    async def __call__(self) -> str:
        """Execute web search."""
        # Run synchronous Tavily search in thread pool
        sources = await asyncio.to_thread(
            self._tavily.search,
            query=self.query,
            max_results=5,
            search_depth="advanced",
        )

        # Store sources for later retrieval
        self._sources = sources

        # Format results for LLM
        result = "Search results:\n\n"
        for i, source in enumerate(sources, 1):
            result += f"[{i}] {source['title']}\n"
            result += f"URL: {source['url']}\n"
            result += f"Content: {source['snippet']}\n\n"

        return result

    @property
    def sources(self):
        """Get retrieved sources."""
        return getattr(self, "_sources", [])


# Agent Events (for streaming support)
@dataclass
class EventText:
    text: str
    type: str = "text"


@dataclass
class EventToolUse:
    tool: Tool
    type: str = "tool_use"


@dataclass
class EventToolResult:
    tool: Tool
    result: str
    type: str = "tool_result"


AgentEvent = EventText | EventToolUse | EventToolResult


MessageType = Union[Message, AssistantMessage, ToolMessage]


class ConversationMemory:
    """Simple in-memory conversation storage."""

    def __init__(self):
        self.conversations: dict[str, list[MessageType]] = {}
        self.sources: dict[str, list] = {}

    def get_messages(self, thread_id: str) -> list[MessageType]:
        """Get message history for thread."""
        return self.conversations.get(thread_id, [])

    def get_user_messages(self, thread_id: str) -> list[MessageType]:
        """Get user messages for thread."""
        return [
            message
            for message in self.get_messages(thread_id)
            if message.role == "user"
        ]

    def get_assistant_messages(self, thread_id: str) -> list[MessageType]:
        """Get assistant messages for thread."""
        return [
            message
            for message in self.get_messages(thread_id)
            if message.role == "assistant"
        ]

    def get_tool_messages(self, thread_id: str) -> list[MessageType]:
        """Get tool messages for thread."""
        return [
            message
            for message in self.get_messages(thread_id)
            if message.role == "tool"
        ]

    def save_messages(self, thread_id: str, messages: list[MessageType]):
        """Save message history for thread."""
        self.conversations[thread_id] = messages

    def add_sources(self, thread_id: str, sources: list):
        """Replace sources for thread (resets each turn)."""
        self.sources[thread_id] = sources

    def get_sources(self, thread_id: str) -> list:
        """Get all sources for thread."""
        return self.sources.get(thread_id, [])

    def get_state(self, thread_id: str) -> dict:
        """Get conversation state."""
        messages = self.get_messages(thread_id)
        sources = self.get_sources(thread_id)

        return {
            "messages": messages,
            "sources": sources,
            "thread_id": thread_id,
            "turn_count": len([m for m in messages if m.role == "user"]),
        }


class ConversationalSearchAgent:
    """Minimal while-loop based agent without LangChain dependencies."""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_iterations: int = 5,
        system_prompt: Optional[str] = None,
    ):
        """Initialize agent.

        Args:
            openai_api_key: OpenAI API key
            tavily_api_key: Tavily API key
            model: OpenAI model to use
            max_iterations: Maximum iterations per query
            system_prompt: System prompt to use
        """
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        self.client = wrap_openai(
            AsyncOpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        )
        self.model = model
        self.max_iterations = max_iterations

        # Initialize Tavily
        self.tavily = TavilySearchTool(api_key=tavily_api_key)

        # Tool definitions for OpenAI
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web for current information and facts. Returns search results with sources.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query string",
                            }
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

        self.memory = ConversationMemory()

    def _message_to_openai(self, msg: MessageType) -> dict:
        """Convert our Message to OpenAI format."""
        if isinstance(msg, AssistantMessage) and msg.tool_calls:
            return {
                "role": "assistant",
                "content": msg.content or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": str(tc.arguments)},
                    }
                    for tc in msg.tool_calls
                ],
            }
        elif isinstance(msg, ToolMessage):
            return {
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "name": msg.name,
                "content": msg.content,
            }
        else:
            return {"role": msg.role, "content": msg.content}

    async def _execute_tool(self, tool_call: ToolCall) -> tuple[str, list]:
        """Execute a tool call.

        Args:
            tool_call: Tool call from LLM

        Returns:
            Tuple of (result_string, sources_list)
        """
        if tool_call.name == "search_web":
            # Parse arguments
            import json

            args = (
                json.loads(tool_call.arguments)
                if isinstance(tool_call.arguments, str)
                else tool_call.arguments
            )

            # Create and execute tool
            tool = SearchWebTool(tavily_client=self.tavily, **args)
            result = await tool()
            sources = tool.sources

            return result, sources
        else:
            return f"Error: Unknown tool {tool_call.name}", []

    @traced(type="task", name="agent_decision")
    async def _agent_step(self, messages: list[MessageType], iteration: int):
        """Single agent step with LLM call.

        Args:
            messages: Current message history
            iteration: Current iteration number

        Returns:
            OpenAI ChatCompletion response
        """
        current_span().log(metadata={"iteration": iteration})

        # Convert messages to OpenAI format
        openai_messages = [self._message_to_openai(msg) for msg in messages]

        # Call OpenAI
        response = await self.client.chat.completions.create(
            model=self.model, messages=openai_messages, tools=self.tools, temperature=0
        )

        return response

    async def run_async(
        self, query: str, thread_id: Optional[str] = None
    ) -> tuple[str, str]:
        """Run agent asynchronously.

        Args:
            query: User query
            thread_id: Conversation thread ID

        Returns:
            Tuple of (response, thread_id)
        """
        # Generate or use thread_id
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        # Get conversation history to calculate turn number
        chat_history = self.memory.get_user_messages(thread_id)
        turn_number = len(chat_history) + 1

        # Format span name with thread_id and turn number
        short_thread_id = thread_id[:8]  # First 8 chars for readability
        span_name = f"Agent Search [{short_thread_id}]({turn_number})"

        # Start span with dynamic name
        with start_span(name=span_name, span_attributes={"type": "task"}) as span:
            # Get full conversation history
            messages = self.memory.get_messages(thread_id)

            # Log input and metadata
            span.log(
                input={"query": query, "chat_history": chat_history},
                metadata={
                    "thread_id": thread_id,
                    "turn_number": turn_number,
                },
            )

            # Add system prompt if first message
            if not messages:
                messages.append(Message(content=self.system_prompt, role="system"))

            # Add user query
            messages.append(Message(content=query, role="user"))

            # Agent loop
            iteration = 0
            final_response = None

            # Reset sources at the start of each turn
            self.memory.add_sources(thread_id, [])

            while iteration < self.max_iterations:
                iteration += 1

                # Call LLM
                response = await self._agent_step(messages, iteration)

                # Get the message from response
                message = response.choices[0].message

                # Check if LLM wants to use tools
                if message.tool_calls:
                    # Create assistant message with tool calls
                    tool_calls = []
                    for tc in message.tool_calls:
                        import json

                        tool_calls.append(
                            ToolCall(
                                id=tc.id,
                                name=tc.function.name,
                                arguments=json.loads(tc.function.arguments),
                            )
                        )

                    assistant_msg = AssistantMessage(
                        role="assistant",
                        content=message.content or "",
                        tool_calls=tool_calls,
                    )
                    messages.append(assistant_msg)

                    # Collect sources from all tool calls in this iteration
                    turn_sources = []

                    # Execute tools
                    for tool_call in tool_calls:
                        result, sources = await self._execute_tool(tool_call)

                        # Collect sources from this tool call
                        if sources:
                            turn_sources.extend(sources)

                        # Add tool result message
                        messages.append(
                            ToolMessage(
                                role="tool",
                                content=result,
                                tool_call_id=tool_call.id,
                                name=tool_call.name,
                            )
                        )

                    # Store all sources from this turn
                    if turn_sources:
                        self.memory.add_sources(thread_id, turn_sources)
                else:
                    # LLM has final answer
                    final_response = message.content
                    messages.append(
                        Message(content=final_response or "", role="assistant")
                    )
                    break

            # Save conversation
            self.memory.save_messages(thread_id, messages)

            # Return response
            if final_response is None:
                final_response = "I apologize, but I was unable to generate a response."

            span.log(
                output={
                    "response": final_response,
                    "sources": self.memory.get_sources(thread_id),
                },
                metadata={"turn_index": iteration, "thread_id": thread_id},
            )

            return final_response, thread_id

    def run(
        self,
        query: str,
        thread_id: Optional[str] = None,
        span_name: str = "Conversational Search",
    ) -> tuple[str, str]:
        """Synchronous wrapper for run_async.

        Args:
            query: User query
            thread_id: Conversation thread ID
            span_name: Custom span name (kept for API compatibility)

        Returns:
            Tuple of (response, thread_id)
        """
        _ = span_name  # Explicitly mark as intentionally unused
        return asyncio.run(self.run_async(query, thread_id))

    def get_state(self, thread_id: str) -> dict:
        """Get conversation state for a thread.

        Args:
            thread_id: Conversation thread ID

        Returns:
            State dict with messages, sources, etc.
        """
        return self.memory.get_state(thread_id)
