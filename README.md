# Conversational Search & Research Agent

A conversational search engine that synthesizes information from the web with proper citations, comprehensive evaluations, and production tracing.

## Features

- **Conversational Search**: Multi-turn conversations with intelligent routing (search vs. context)
- **Citation-Based Answers**: All factual claims backed by web sources
- **LangGraph State Machine**: Explicit control flow with route → search → synthesize → respond
- **Comprehensive Evaluations**: 4 core scorers (factual accuracy, citation quality, completeness, retrieval quality)
- **Production Tracing**: Braintrust integration for observability
- **Synthetic Eval Dataset**: 20 diverse conversations across domains

## Tech Stack

- **Agent Framework**: LangGraph
- **LLM**: OpenAI GPT-4
- **Search**: Tavily API
- **Observability & Evals**: Braintrust
- **Package Manager**: uv
- **Language**: Python 3.11+

## Setup

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and Install Dependencies

```bash
git clone <repo-url>
cd perp
uv sync
```

### 3. Configure API Keys

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
BRAINTRUST_API_KEY=...
```

Get API keys from:
- OpenAI: https://platform.openai.com/api-keys
- Tavily: https://tavily.com
- Braintrust: https://www.braintrust.dev

## Usage

### Production Demo (with Tracing)

Run sample conversations with Braintrust tracing:

```bash
uv run python scripts/production_demo.py
```

View traces at: https://www.braintrust.dev

### Generate Evaluation Dataset

Generate synthetic conversation dataset:

```bash
uv run python evals/dataset_generator.py
```

Output: `evals/test_data.json` (20 conversations, ~60 test cases)

### Run Evaluations

Execute full evaluation suite with Braintrust:

```bash
uv run python evals/run_evals.py
```

This will:
1. Load test dataset
2. Run agent on each test case
3. Score with 4 core metrics
4. Upload results to Braintrust

View results at: https://www.braintrust.dev

### Use Agent Programmatically

```python
from src.agent import ConversationalSearchAgent

agent = ConversationalSearchAgent()

# Single query
response = agent.run("What is quantum computing?")
print(response)

# With state for multi-turn
response1, state = agent.run_with_state("What is quantum computing?")
response2, state = agent.run_with_state("How does it compare to classical?", state)
```

## Architecture

### Agent Flow

```
User Query
    ↓
Route Node (decide: search vs. context)
    ↓
Search Node (if needed) → Tavily API
    ↓
Synthesize Node → OpenAI GPT-4 + citations
    ↓
Respond Node → Format with source list
    ↓
User
```

### State Schema

```python
{
    "messages": [HumanMessage, AIMessage, ...],
    "sources": [{url, title, snippet, timestamp}, ...],
    "search_results": [...],
    "needs_search": bool,
    "current_query": str
}
```

## Evaluation Scorers

### 1. Factual Accuracy (LLM-as-judge)
- Verifies claims against sources
- Detects hallucinations and contradictions
- Output: accuracy score + unsupported claims

### 2. Citation Quality (Hybrid)
- **Coverage**: % of factual claims with citations
- **Precision**: validity of citation references
- **Source Quality**: domain reputation heuristics

### 3. Answer Completeness (LLM-as-judge)
- Checks if all query aspects addressed
- Identifies missing information
- Output: completeness score + gap analysis

### 4. Retrieval Quality (Hybrid)
- **Precision@K**: relevance of top-K results (LLM-judged)
- **Recall Approximation**: coverage analysis (LLM-judged)
- Output: F1-like composite score

## Project Structure

```
perp/
├── docs/plans/                  # Design and implementation docs
├── src/
│   ├── agent.py                # LangGraph agent
│   ├── tools.py                # Tavily search wrapper
│   ├── synthesis.py            # Citation utilities
│   └── state.py                # Agent state schema
├── evals/
│   ├── scorers/                # Evaluation scorers
│   │   ├── factual_accuracy.py
│   │   ├── citation_quality.py
│   │   ├── answer_completeness.py
│   │   └── retrieval_quality.py
│   ├── dataset_generator.py   # Synthetic data generation
│   ├── test_data.json          # Generated test cases
│   └── run_evals.py            # Main eval runner
├── scripts/
│   └── production_demo.py      # Tracing demo
├── pyproject.toml              # Dependencies
└── README.md
```

## Development

### Run Tests

```bash
uv run pytest
```

### Lint Code

```bash
uv run ruff check .
```

### Add New Scorer

1. Create `evals/scorers/your_scorer.py`
2. Implement scorer class with `score(output, expected)` method
3. Add Braintrust-compatible wrapper function
4. Import and add to `evals/run_evals.py` scorers list

## Future Enhancements

- Additional scorers: source diversity, query quality, coherence, context awareness
- Real user conversation collection
- Advanced routing strategies
- Multi-turn conversation state persistence
- Streaming responses
- Cost optimization

## License

MIT

## References

- [LangGraph Docs](https://python.langchain.com/docs/langgraph)
- [Braintrust Docs](https://www.braintrust.dev/docs)
- [Tavily API](https://docs.tavily.com)
