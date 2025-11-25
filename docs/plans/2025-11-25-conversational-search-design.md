# Conversational Search & Research Agent Design

**Date:** 2025-11-25
**Purpose:** Build a conversational search engine that synthesizes information from the web with cited answers, comprehensive evaluation framework, and production tracing.

## Overview

A balanced general-purpose conversational search system using LangGraph for state management, Tavily for web search, OpenAI for synthesis, and Braintrust for evaluation and observability.

## Technology Stack

- **Agent Framework:** LangGraph (explicit state management, graph-based control flow)
- **LLM:** OpenAI GPT-4 (synthesis, routing, LLM-as-judge scoring)
- **Search:** Tavily API (web search with citations)
- **Observability & Evals:** Braintrust (tracing, evaluation framework)
- **Package Management:** uv with pyproject.toml
- **Language:** Python 3.11+

## Architecture

### System Components

1. **Conversational Search Agent** (LangGraph-based)
   - Graph nodes: `route_query` → `search` → `synthesize` → `respond`
   - State: conversation history, retrieved sources, search results
   - Routing logic: decides whether to search Tavily or answer from context (intelligent routing)

2. **Tavily Integration**
   - Tool wrapper for LangGraph tool calling
   - Returns: URLs, snippets, titles, relevance scores

3. **OpenAI Integration**
   - Primary LLM: GPT-4 for synthesis and routing decisions
   - Function calling for tool use

4. **Braintrust Integration**
   - Auto-tracing via LangChain/LangGraph integration
   - Manual spans for custom metrics (search quality, citation extraction)
   - Eval framework using `Eval()` function

### Data Flow

```
User Query → Route Node (search needed?)
  → If yes: Search Node (Tavily) → Synthesize Node (OpenAI + citations)
  → If no: Synthesize Node (use context)
→ Respond Node → User
```

### State Management

**LangGraph State Schema:**
```python
class AgentState(TypedDict):
    messages: List[BaseMessage]
    sources: List[Dict]  # {url, title, snippet, timestamp}
    search_results: List[Dict]  # Raw Tavily responses
    needs_search: bool
    current_query: str
```

State persists across conversation turns for context-aware follow-ups.

## Agent Implementation

### Graph Nodes

1. **route_query Node**
   - Analyzes user message + conversation history
   - Uses OpenAI to decide: search needed vs. answer from context
   - Decision criteria: new topic, needs updated info, clarification vs. new question
   - Sets `needs_search` boolean flag

2. **search Node** (conditional, only if needs_search=True)
   - Calls Tavily API with optimized query
   - May reformulate query for better search results
   - Appends results to `search_results` and `sources`
   - Wrapped in Braintrust custom span for search metrics tracking

3. **synthesize Node**
   - OpenAI generates answer with explicit citation instructions
   - System prompt enforces: "Cite sources using [1], [2] format. Include source list."
   - Extracts citations and links them to sources
   - Validates all citations exist in retrieved sources

4. **respond Node**
   - Formats final response: answer text + numbered source list
   - Appends to message history
   - Returns control to user

### Graph Edges

- `route_query` → `search` (if needs_search) or `synthesize` (if not)
- `search` → `synthesize`
- `synthesize` → `respond`
- `respond` → END (awaits next user input)

### Tool Integration

- Tavily as LangGraph tool with error handling
- Configurable parameters: max_results, search_depth, include_domains

## Evaluation Framework

### Core Metrics (Priority Implementation)

1. **Factual Accuracy Scorer** (LLM-as-judge)
   - Uses OpenAI to verify claims against retrieved source documents
   - Checks for: unsupported statements, contradictions, hallucinations
   - Implementation: Compare answer sentences against source snippets
   - Output: accuracy score (0-1) + list of problematic claims

2. **Citation Quality Scorer** (Hybrid: deterministic + LLM)
   - **Citation Coverage** (deterministic): % of factual claims with citations
   - **Citation Precision** (deterministic): Are citations relevant to claims?
   - **Source Quality** (deterministic): Domain reputation, recency scoring
   - Output: composite score + breakdown by dimension

3. **Answer Completeness Scorer** (LLM-as-judge)
   - Evaluates if answer addresses all aspects of the query
   - Checks for: key points covered, appropriate depth, missing context
   - Output: completeness score (0-1) + gap analysis

4. **Retrieval Quality Scorer** (Hybrid: deterministic + LLM)
   - **Precision@K** (deterministic): Relevance of top-K retrieved docs
   - **Recall approximation** (LLM-judge): Did we miss important sources?
   - Requires: ground truth relevant docs (from synthetic generation or manual annotation)
   - Output: precision/recall metrics + relevance distribution

### Incremental Scorers (Add Later)

- Source diversity (deterministic)
- Search query quality (LLM-judge)
- Coherence & readability (LLM-judge)
- Context awareness for conversations (LLM-judge)
- Query understanding (correct interpretation of ambiguous queries)
- Harmful content detection (classifier)
- Bias & fairness metrics

### Scorer Architecture

- Each scorer is a Braintrust-compatible function: `(output, expected) -> Score`
- Scorers can access: answer text, sources, search queries, conversation history
- Leverage AutoEvals library for built-in factuality/relevance scorers where applicable
- Custom scorers for domain-specific metrics

### Eval Dataset Generation (Synthetic)

1. **Conversation Generator**
   - Use OpenAI to generate diverse multi-turn conversations
   - Topics: varied domains (tech, science, current events, history)
   - Conversation patterns: follow-ups, clarifications, topic pivots, context references
   - Generate ~20-30 conversations initially (3-5 turns each)

2. **Reference Answer Creation**
   - For each conversation turn, generate "gold standard" answer
   - Include expected sources (manually validated or high-confidence)
   - Store: query, expected_answer, expected_sources[], conversation_context

3. **Test Case Structure**
   ```python
   {
     "conversation_id": "conv_001",
     "turns": [
       {
         "user_query": "...",
         "context": [...],  # Previous messages
         "expected_answer": "...",
         "expected_sources": [...],
         "metadata": {"topic": "...", "type": "new_query|followup|clarification"}
       }
     ]
   }
   ```

### Braintrust Eval Execution

```python
from braintrust import Eval

Eval(
    project_name="conversational-search",
    data=test_conversations,
    task=run_agent_conversation,  # Execute agent on test case
    scores=[
        factual_accuracy_scorer,
        citation_quality_scorer,
        answer_completeness_scorer,
        retrieval_quality_scorer
    ],
    metadata={"model": "gpt-4", "tavily_depth": "advanced"}
)
```

**Metrics Dashboard:**
- Aggregate scores across all conversations
- Per-scorer breakdowns
- Per-topic performance analysis
- Cost/latency tracking (built-in Braintrust)

## Production Tracing

### Tracing Script

```python
# scripts/production_demo.py
# Simulates "production" usage with Braintrust tracing

import braintrust
from agent import ConversationalSearchAgent

# Initialize Braintrust logging
braintrust.init(project="conversational-search")

# Run sample conversations
agent = ConversationalSearchAgent()

demo_conversations = [
    ["What are the latest developments in quantum computing?",
     "How does this compare to classical computing?",
     "What companies are leading in this space?"],

    ["Explain the 2024 US election results",
     "What were the key swing states?"],
]

for conv in demo_conversations:
    with braintrust.start_span(name="conversation") as span:
        for turn in conv:
            response = agent.run(turn)
            # Auto-traced via LangGraph integration
```

## Project Structure

```
perp/
├── docs/
│   └── plans/
│       └── 2025-11-25-conversational-search-design.md
├── src/
│   ├── agent.py              # LangGraph agent implementation
│   ├── tools.py              # Tavily wrapper
│   ├── synthesis.py          # Citation extraction & formatting
│   └── state.py              # AgentState schema
├── evals/
│   ├── scorers/
│   │   ├── factual_accuracy.py
│   │   ├── citation_quality.py
│   │   ├── answer_completeness.py
│   │   └── retrieval_quality.py
│   ├── dataset_generator.py  # Synthetic conversation generation
│   ├── test_data.json        # Generated eval dataset
│   └── run_evals.py          # Main eval script using Eval()
├── scripts/
│   └── production_demo.py    # Tracing demonstration
├── pyproject.toml            # uv-managed dependencies
├── uv.lock                   # Generated by uv
└── .env                      # API keys
```

## Dependency Management

### pyproject.toml

```toml
[project]
name = "conversational-search"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "langgraph>=0.2.0",
    "langchain>=0.3.0",
    "langchain-openai>=0.2.0",
    "braintrust>=0.0.140",
    "openai>=1.50.0",
    "tavily-python>=0.5.0",
    "python-dotenv>=1.0.0",
]

[tool.uv]
dev-dependencies = [
    "pytest>=8.0.0",
    "ruff>=0.6.0",
]
```

### Setup Commands

```bash
# Initialize project (if needed)
uv init

# Install dependencies
uv sync

# Run production demo with tracing
uv run scripts/production_demo.py

# Run evaluations
uv run evals/run_evals.py
```

## Key Design Decisions

1. **LangGraph over Pydantic AI**: Explicit state management and graph-based control flow provides better visibility and control over conversation routing logic.

2. **Intelligent Routing**: Agent decides when to search vs. answer from context, balancing freshness with efficiency and latency.

3. **Core Metrics First**: Start with 4 essential scorers (factual accuracy, citation quality, completeness, retrieval quality) before expanding to comprehensive suite.

4. **Synthetic Eval Data**: Use LLM-generated conversations to quickly scale eval coverage across diverse topics and conversation patterns.

5. **uv for Package Management**: Modern, fast Python package manager with deterministic dependency resolution.

## Implementation Priority

1. **Phase 1: Agent Implementation**
   - LangGraph agent with all nodes
   - Tavily integration
   - Citation extraction and formatting
   - Basic Braintrust tracing

2. **Phase 2: Core Evaluations**
   - Implement 4 priority scorers
   - Synthetic dataset generation
   - Eval runner script with Braintrust

3. **Phase 3: Production Demo**
   - Production tracing script
   - Sample conversations
   - Verify end-to-end tracing in Braintrust dashboard

4. **Phase 4: Expansion** (Future)
   - Additional scorers (diversity, query quality, etc.)
   - Real user conversation collection
   - Advanced routing strategies

## References

- [Braintrust LangGraph Integration](https://www.braintrust.dev/docs/integrations/langchain)
- [Braintrust Evaluation Tools 2025](https://www.braintrust.dev/articles/best-llm-evaluation-tools-integrations-2025)
- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [Tavily API](https://tavily.com)
