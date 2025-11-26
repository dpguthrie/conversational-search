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
        """Search web using Tavily.

        Args:
            state: Current agent state

        Returns:
            Updated state with search results and sources
        """
        query = state["current_query"]

        # Optionally reformulate query for better search
        # For now, use query directly

        # Call Tavily
        sources = self.search_tool.search(
            query=query,
            max_results=5,
            search_depth="advanced"
        )

        # Append to state (preserve previous sources)
        existing_sources = state.get("sources", [])
        all_sources = existing_sources + sources

        return {
            **state,
            "sources": all_sources,
            "search_results": sources  # Latest search results
        }

    def _synthesize(self, state: AgentState) -> AgentState:
        """Synthesize answer with citations.

        Args:
            state: Current agent state

        Returns:
            Updated state with AI message containing answer
        """
        query = state["current_query"]
        sources = state.get("sources", [])

        if not sources:
            # No sources available, respond from general knowledge
            messages = state["messages"] + [
                SystemMessage(content="Answer the user's query based on your general knowledge. Be honest if you don't have enough information."),
            ]
            response = self.llm.invoke(messages)
            answer = response.content
        else:
            # Generate answer with citations
            synthesis_prompt = create_synthesis_prompt(query, sources)
            response = self.llm.invoke([SystemMessage(content=synthesis_prompt)])
            answer = response.content

            # Validate citations
            is_valid, errors = validate_citations(answer, len(sources))
            if not is_valid:
                # Log validation errors (in production, might retry or fix)
                print(f"Citation validation errors: {errors}")

        # Add AI message to conversation
        new_messages = state["messages"] + [AIMessage(content=answer)]

        return {
            **state,
            "messages": new_messages
        }

    def _respond(self, state: AgentState) -> AgentState:
        """Format final response with source list.

        Args:
            state: Current agent state

        Returns:
            Updated state with formatted response
        """
        messages = state["messages"]
        sources = state.get("sources", [])

        # Get the last AI message (answer)
        answer = messages[-1].content if messages else ""

        # Append formatted sources
        sources_text = format_sources(sources)
        formatted_response = answer + sources_text

        # Update the last message with formatted version
        if messages:
            messages[-1].content = formatted_response

        return {
            **state,
            "messages": messages
        }

    def run(self, query: str, conversation_state: AgentState = None) -> str:
        """Run agent on a query.

        Args:
            query: User query
            conversation_state: Optional existing conversation state

        Returns:
            Agent response with citations
        """
        # Initialize or use existing state
        if conversation_state is None:
            state: AgentState = {
                "messages": [],
                "sources": [],
                "search_results": [],
                "needs_search": False,
                "current_query": ""
            }
        else:
            state = conversation_state

        # Add user message
        state["messages"].append(HumanMessage(content=query))

        # Run graph
        result = self.graph.invoke(state)

        # Return last message
        return result["messages"][-1].content
