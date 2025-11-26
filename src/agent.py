"""Conversational search agent using LangGraph."""
import os
from typing import Dict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.tools import TavilySearchTool
from src.synthesis import create_synthesis_prompt, format_sources, validate_citations


class ConversationalSearchAgent:
    """LangGraph-based conversational search agent."""

    def __init__(
        self,
        openai_api_key: str = None,
        tavily_api_key: str = None,
        model: str = "gpt-4"
    ):
        """Initialize agent.

        Args:
            openai_api_key: OpenAI API key
            tavily_api_key: Tavily API key
            model: OpenAI model to use
        """
        self.llm = ChatOpenAI(
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
            model=model,
            temperature=0
        )
        self.search_tool = TavilySearchTool(api_key=tavily_api_key)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build LangGraph state machine."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("route_query", self._route_query)
        workflow.add_node("search", self._search)
        workflow.add_node("synthesize", self._synthesize)
        workflow.add_node("respond", self._respond)

        # Add edges
        workflow.set_entry_point("route_query")
        workflow.add_conditional_edges(
            "route_query",
            self._should_search,
            {
                True: "search",
                False: "synthesize"
            }
        )
        workflow.add_edge("search", "synthesize")
        workflow.add_edge("synthesize", "respond")
        workflow.add_edge("respond", END)

        return workflow.compile()

    def _route_query(self, state: AgentState) -> AgentState:
        """Decide if query needs new search or can use context.

        Args:
            state: Current agent state

        Returns:
            Updated state with needs_search flag
        """
        messages = state["messages"]
        current_query = messages[-1].content if messages else ""

        # Build routing prompt
        routing_prompt = """You are a routing assistant. Decide if this query needs a new web search or can be answered from conversation context.

SEARCH NEEDED if:
- Query asks about current events, recent information, or facts
- Query introduces a completely new topic
- Previous sources don't cover this topic

NO SEARCH NEEDED if:
- Query is a clarification of previous answer
- Query asks about something already in conversation
- Query is a greeting or meta-question

Conversation history:
"""
        for msg in messages[:-1]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            routing_prompt += f"{role}: {msg.content}\n"

        routing_prompt += f"\nCurrent query: {current_query}\n\nRespond with only 'SEARCH' or 'NO_SEARCH'"

        # Ask LLM to route
        response = self.llm.invoke([SystemMessage(content=routing_prompt)])
        needs_search = "SEARCH" in response.content.upper() and "NO_SEARCH" not in response.content.upper()

        return {
            **state,
            "current_query": current_query,
            "needs_search": needs_search
        }

    def _should_search(self, state: AgentState) -> bool:
        """Conditional edge function."""
        return state["needs_search"]

    def _search(self, state: AgentState) -> AgentState:
        """Search web using Tavily (placeholder)."""
        # Placeholder - will implement in next task
        return state

    def _synthesize(self, state: AgentState) -> AgentState:
        """Synthesize answer with citations (placeholder)."""
        # Placeholder - will implement in next task
        return state

    def _respond(self, state: AgentState) -> AgentState:
        """Format and return response (placeholder)."""
        # Placeholder - will implement in next task
        return state
