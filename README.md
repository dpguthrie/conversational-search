# Conversational Search & Research Agent

A conversational search engine that synthesizes information from the web with proper citations, comprehensive evaluations, and production tracing.

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Set up API keys (copy .env.example to .env)
cp .env.example .env
# Edit .env with your API keys

# 3. Start the interactive CLI
uv run python src/chat.py
```

That's it! Start asking questions and the agent will search the web and provide cited answers.

## Features

- **Conversational Search**: Multi-turn conversations with intelligent LLM-driven tool calling
- **Citation-Based Answers**: All factual claims backed by web sources
- **Minimal While-Loop Architecture**: Simple, transparent agent loop with direct OpenAI SDK integration
- **Comprehensive Evaluations**: 4 core scorers (factual accuracy, citation quality, completeness, retrieval quality)
- **Production Tracing**: Braintrust integration with @traced decorators
- **Synthetic Eval Dataset**: 20 diverse conversations across domains
- **Zero Framework Bloat**: No LangChain or LangGraph - just clean primitives

## Tech Stack

- **Agent Pattern**: While loop with tool calling
- **LLM**: OpenAI GPT-4 (direct SDK)
- **Search**: Tavily API
- **Message Primitives**: Python dataclasses
- **Tool Definition**: Pydantic BaseModel
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

### Interactive CLI (Recommended)

The easiest way to use the agent is through the interactive CLI:

```bash
uv run python src/chat.py

# Or make executable
chmod +x src/chat.py
./src/chat.py
```

**Features:**
- 🎨 Beautiful rich terminal UI with colors and panels
- 💬 Multi-turn conversations with automatic context (thread_id)
- 🔄 Start new conversations with `/new`
- 📊 View conversation stats with `/stats`
- ⚡ Real-time "Thinking..." status indicators
- 📝 Markdown-rendered responses with proper formatting
- ✅ Automatic Braintrust tracing

**Commands:**
| Command | Description |
|---------|-------------|
| `/new` | Start a new conversation (resets thread_id) |
| `/help` | Show help message and available commands |
| `/stats` | Show conversation statistics (turns, sources, duration) |
| `/exit` | Exit the chat (or use Ctrl+D) |

**Example Session:**
```
You: What is quantum computing?
[Thinking...]
Assistant: [Detailed response with citations]

You: How does it compare to classical computing?
[Uses context from previous question]

You: /stats
┌─ Conversation Statistics ─┐
│  Thread ID    abc123...    │
│  Turns        2            │
│  Sources      5            │
└─────────────────────────────┘

You: /new
✓ Started new conversation

You: /exit
Thanks for using Conversational Search!
```

### Production Demo (with Automatic Tracing)

Run sample conversations with Braintrust tracing:

```bash
uv run python scripts/production_demo.py
```

This demo uses @traced decorators which automatically capture:
- Agent loop iterations
- LLM calls with prompts and responses
- Tool executions (web search)
- Conversation flow and metadata
- Timing and performance metrics

View traces at: https://www.braintrust.dev

### Generate Evaluation Dataset

Generate synthetic conversation dataset:

```bash
uv run python evals/dataset_generator.py
```

Output: `evals/test_data.json` (20 conversations, ~60 test cases)

### Run Evaluations

#### Simple Single-Turn Evaluation (Recommended for Quick Testing)

The easiest way to run evaluations using Braintrust's `Eval()` function:

```bash
uv run python evals/simple_eval.py
```

**Features:**
- Uses existing dataset (conversational-search-eval-v1)
- Filters to first-turn queries only
- Hosted scorers from Braintrust platform (no local imports)
- Automatic parallelization via `Eval()`
- Built-in progress tracking

**Limitations:**
- Single-turn only (no conversation state)
- For multi-turn conversations, use `run_evals.py`

#### Full Multi-Turn Evaluation

Execute full evaluation suite with conversation state:

```bash
# Run all conversations with parallel execution (default: 5 concurrent)
uv run python evals/run_evals.py

# Adjust concurrency for faster evaluation
uv run python evals/run_evals.py --max-concurrency 10

# Skip dataset loading for faster startup
uv run python evals/run_evals.py --no-dataset

# Small test (first conversation only)
uv run python evals/test_eval_small.py
```

This will:
1. Load test dataset from Braintrust
2. Run agent on each test case with conversation context
3. Score with 4 core metrics (local scorers)
4. Upload results to Braintrust

**Performance:**
- Default concurrency (5): ~20 conversations in parallel
- Each conversation maintains its own thread_id for context
- Conversations evaluated independently and safely in parallel
- Typical speedup: 3-5x faster than sequential execution

View results at: https://www.braintrust.dev

#### Remote Evaluation (Braintrust Playground)

Expose your evaluations as remote endpoints for testing in the Braintrust playground:

```bash
# Start development server (exposes eval at http://localhost:8300)
braintrust eval evals/remote_eval.py --dev

# Customize port if needed
braintrust eval evals/remote_eval.py --dev --dev-port 8301

# Restrict access to your organization only
braintrust eval evals/remote_eval.py --dev --dev-org-name YOUR_ORG
```

**Setup workflow:**
1. Start the dev server with `braintrust eval evals/remote_eval.py --dev`
2. In Braintrust UI, go to **Configuration > Remote evals**
3. Click **+ Remote eval source** and enter: `http://localhost:8300`
4. Create a new task in the Braintrust playground
5. Select **Remote eval** and choose your exposed evaluation
6. Run evaluations against playground datasets with custom parameters
7. Iterate and compare results directly in the UI

**Benefits:**
- Test parameter variations from the UI without code changes
- Run evals against different datasets without redeploying
- Collaborate with team members who can run your evals remotely
- Combine playground datasets with your existing evaluation logic
- Scorers from remote eval are combined with playground scorers

**Note:** The dataset defined in `remote_eval.py` is ignored; datasets come from the Braintrust playground interface.

### Use Agent Programmatically

#### Without Tracing

```python
from src.agent import ConversationalSearchAgent

agent = ConversationalSearchAgent()

# Single query with auto-generated thread_id
response, thread_id = agent.run("What is quantum computing?")
print(f"Response: {response}")
print(f"Thread ID: {thread_id}")

# Multi-turn conversation (same thread_id maintains context)
response1, thread_id = agent.run("What is quantum computing?")
response2, _ = agent.run("How does it compare to classical?", thread_id=thread_id)
response3, _ = agent.run("What companies are leaders?", thread_id=thread_id)

# Get conversation state
state = agent.get_state(thread_id)
print(f"Messages in conversation: {len(state['messages'])}")
```

#### With Automatic Tracing (REPL or Scripts)

Enable Braintrust tracing to capture all agent operations:

```python
import os
from dotenv import load_dotenv
from braintrust import init_logger
from src.agent import ConversationalSearchAgent

# Load environment (includes BRAINTRUST_API_KEY)
load_dotenv()

# Enable automatic tracing (do this once at startup)
init_logger(project="conversational-search", api_key=os.environ.get("BRAINTRUST_API_KEY"))

# Now use the agent normally - everything is automatically traced via @traced decorators!
agent = ConversationalSearchAgent()

# Auto-generate thread_id
response1, thread_id = agent.run("What is Python?")

# Continue conversation with same thread_id
response2, _ = agent.run("Tell me more about its history", thread_id=thread_id)

# Multi-turn conversation
response3, _ = agent.run(
    "What companies use it?",
    thread_id=thread_id,
)

# All agent steps, LLM calls, tool calls captured in Braintrust
# The thread_id is exposed in metadata for easy filtering and grouping
```

## Architecture

### Agent Flow

```
User Query
    ↓
While Loop (iteration < max_iterations):
    ↓
    LLM Call (with tools=[search_web])
    ↓
    ├─> Tool Calls? → Execute search_web → Add results to messages → Continue loop
    │
    └─> No Tools? → Final answer with citations → Done
```

**Key Pattern:**
- LLM decides when to search via tool calling (no explicit routing)
- Simple while loop with clear stop condition
- Direct OpenAI SDK calls (no framework wrapper)

### Message Primitives

Simple dataclasses for type-safe messages:

```python
@dataclass
class Message:
    content: str
    role: Literal["system", "user", "assistant", "tool"] = "user"

@dataclass
class AssistantMessage:
    content: str
    role: Literal["assistant"] = "assistant"
    tool_calls: list[ToolCall] = field(default_factory=list)

@dataclass
class ToolMessage:
    content: str
    role: Literal["tool"] = "tool"
    tool_call_id: str = ""
    name: str = ""
```

### Tool Definition

Pydantic BaseModel with `__call__` method:

```python
class SearchWebTool(Tool):
    """Search the web for current information."""
    query: str = Field(description="The search query string")

    async def __call__(self) -> str:
        # Execute search and return formatted results
        sources = await asyncio.to_thread(self.tavily.search, ...)
        return formatted_results
```

### State Schema

Simple dict-based memory:

```python
{
    "messages": [Message, AssistantMessage, ToolMessage, ...],
    "sources": [{url, title, snippet, timestamp}, ...],
    "thread_id": str,
    "turn_count": int
}
```

**Memory Management:**
- In-memory dict storage (ConversationMemory class)
- Thread isolation via thread_id
- Easy to extend with persistent storage (PostgreSQL, Redis, etc.)

## Output Format

All agent outputs use a consistent format across production, evaluations, and scorers:

```python
{
    "query": "What is quantum computing?",
    "response": "Quantum computing uses quantum bits...",
    "sources": [
        {
            "url": "https://example.com/quantum",
            "title": "Introduction to Quantum Computing",
            "snippet": "Quantum computers use qubits...",
            "timestamp": "2025-12-01T10:30:00Z"
        }
    ]
}
```

**Key:** Use `"response"` (not `"answer"`) to access the agent's reply.

This unified format enables:
- Consistent scorer implementation across local and hosted versions
- Easy addition of production examples to eval datasets
- Seamless integration between production tracing and evaluation

## Evaluation Scorers

The project includes 4 comprehensive scorers available in both local (for `run_evals.py`) and hosted (for Braintrust platform) versions:

### 1. Factual Accuracy (LLM-as-judge)
- Verifies claims against sources
- Detects hallucinations and contradictions
- Output: accuracy score + unsupported claims

### 2. Citation Quality (Hybrid - Custom Handler)
- **Coverage**: % of factual claims with citations
- **Precision**: validity of citation references
- **Source Quality**: domain reputation heuristics
- **Metadata**: Returns coverage, precision, source_quality, citation counts

### 3. Answer Completeness (LLM-as-judge)
- Checks if all query aspects addressed
- Considers conversation context (chat_history)
- Identifies missing information
- Output: completeness score (0.0-1.0)

### 4. Retrieval Quality (Hybrid - Custom Handler)
- **Precision@K**: relevance of top-K results (LLM-judged)
- **Recall Approximation**: coverage analysis (LLM-judged)
- **Metadata**: Returns precision_at_k, recall_approximation, relevant_count
- Output: F1-like composite score

## Project Structure

```
perp/
├── src/
│   ├── agent.py                # Minimal while-loop agent (no LangChain)
│   ├── tools.py                # Tavily search wrapper
│   ├── synthesis.py            # Citation utilities
│   └── chat.py                 # 🎯 Interactive CLI (start here!)
├── evals/
│   ├── scorers/                # Local evaluation scorers
│   │   ├── base_evaluator.py  # Base class for LLM-as-judge scorers
│   │   ├── factual_accuracy.py
│   │   ├── citation_quality.py
│   │   ├── answer_completeness.py
│   │   └── retrieval_quality.py
│   ├── dataset_generator.py   # Synthetic data generation
│   ├── dataset_manager.py     # Dataset management utilities
│   ├── simple_eval.py          # Simple single-turn eval with Eval()
│   ├── run_evals.py            # Full multi-turn eval runner
│   └── test_eval_small.py     # Small test eval (single conversation)
├── scripts/
│   ├── production_demo.py      # Production demo with tracing
│   └── add_production_to_eval.py  # Add production examples to dataset
├── braintrust_scorers.py       # Scorers for pushing to Braintrust
├── docs/plans/                 # Original design documents
├── pyproject.toml              # Dependencies
└── README.md                   # This file
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

#### Local Scorers (for run_evals.py)

LLM-as-judge scorers use the `BaseEvaluator` pattern for cleaner code:

```python
from evals.scorers.base_evaluator import BaseEvaluator, scorer_wrapper

PROMPT = """Evaluate the output.
OUTPUT: {{{output}}}
Respond with: "good", "fair", "poor"
"""

class MyEvaluator(BaseEvaluator):
    name = "my_evaluator"
    model = "gpt-4o"  # Or "claude-3-5-sonnet-latest"
    use_cot = True

    def get_choice_scores(self) -> dict[str, float]:
        return {"good": 1.0, "fair": 0.5, "poor": 0.0}

    def get_prompt_template(self) -> str:
        return PROMPT

# Create Braintrust-compatible scorer
my_scorer = scorer_wrapper(MyEvaluator)
```

Then add to `evals/run_evals.py`:
```python
from evals.scorers.my_evaluator import my_scorer

Eval(scores=[my_scorer, ...], ...)
```

#### Hosted Scorers (for Braintrust Platform)

To create scorers that can be pushed to Braintrust and used across projects, add them to `braintrust_scorers.py`:

**LLM-as-Judge Scorer (Native Format):**
```python
project.scorers.create(
    name="My Scorer",
    slug="my-scorer-v1",
    description="Evaluates output quality",
    parameters=AgentOutputParams,
    messages=[{
        "role": "user",
        "content": """Evaluate the output.
OUTPUT: {{output.response}}
Respond with: "good", "fair", "poor"
"""
    }],
    model="gpt-4o",
    use_cot=True,
    choice_scores={
        "good": 1.0,
        "fair": 0.5,
        "poor": 0.0,
    },
)
```

**Custom Handler Function (Deterministic/Hybrid):**
```python
def my_handler(input, output, expected=None, metadata=None):
    """Custom scoring logic with full control."""
    score = compute_my_metric(output)
    return {
        "name": "my_scorer",
        "score": score,
        "metadata": {"reason": "explanation..."}
    }

project.scorers.create(
    name="My Handler Scorer",
    slug="my-handler-v1",
    description="Custom scoring logic",
    parameters=AgentOutputParams,
    handler=my_handler,
)
```

**Push to Braintrust:**
```bash
uv run braintrust push braintrust_scorers.py --if-exists replace
```

**Use in Evaluations:**
```python
from braintrust import init_function

my_scorer = init_function(
    project_name="conversational-search",
    slug="my-scorer-v1",
)
```

### Using Multiple LLM Models

Scorers support multiple models via Braintrust's proxy:

**OpenAI**: `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`
**Anthropic**: `claude-3-5-sonnet-latest`, `claude-sonnet-4-5-20250929`

```python
# Use Claude for complex reasoning
factual_claude = FactualAccuracyEvaluator(model="claude-3-5-sonnet-latest")

# Use GPT-4o for speed
completeness_gpt = AnswerCompletenessEvaluator(model="gpt-4o")

# Mix in same evaluation
Eval(scores=[factual_claude, completeness_gpt], ...)
```

**Setup:** Set `BRAINTRUST_API_KEY` environment variable to use non-OpenAI models.

## Tracing & Observability

All agent operations are automatically traced in Braintrust with meaningful span names:

### Span Naming Convention

Spans are named with thread_id and turn number for easy identification:

```
Agent Search [627a7def](1)  ← Thread 627a7def, Turn 1
Agent Search [627a7def](2)  ← Thread 627a7def, Turn 2
Agent Search [80d7297c](1)  ← Thread 80d7297c, Turn 1 (different conversation)
```

**Format:** `Agent Search [<thread_id_prefix>](<turn_number>)`

This makes it easy to:
- Track multi-turn conversations
- Compare different conversation threads
- Filter by thread_id or turn number in dashboard
- Debug specific turns

### Metadata Logged

Each span includes:
```python
{
    "thread_id": "627a7def-1d08-453c-b7d4-53305d219f77",
    "turn_number": 2,
    "iterations": 3,  # Number of LLM calls
    "chat_history": [...],  # Previous messages
}
```

View traces at: https://www.braintrust.dev

## Architecture Patterns

This implementation follows proven patterns from:
- **[bitswired/agentic-loop](https://github.com/bitswired/demos/tree/main/projects/agentic-loop)** - Minimal agent primitives
- **[Braintrust Agent Loop Blog](https://www.braintrust.dev/blog/agent-while-loop)** - While loop pattern
- **[OpenAI Agents Guide](https://platform.openai.com/docs/guides/agents)** - Direct SDK usage

## Why No LangChain?

**Decision:** Use direct OpenAI SDK instead of LangChain/LangGraph

**Reasons:**
1. **Less Bloat**: 3 dependencies vs 7+
2. **More Control**: Direct API calls, no framework magic
3. **Easier Debugging**: Clear stack traces, no callback layers
4. **Simpler Code**: Dataclasses vs framework message types
5. **Better Performance**: No framework overhead

**Before:**
```python
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
```

**After:**
```python
from openai import AsyncOpenAI
from dataclasses import dataclass
```

## Offline→Online Eval Feedback Loop

This project demonstrates a complete eval workflow that bridges offline evals with production data:

### Unified Data Structure

**Production traces** and **eval dataset** use the same format:

```python
{
    "input": {
        "query": "What are the main differences?",
        "chat_history": ["Tell me about quantum computing"]
    },
    "expected": {  # Optional for evals
        "response": "...",
        "sources": [...]
    },
    "metadata": {
        "thread_id": "conv_123",
        "topic": "quantum computing",
        "turn_index": 1,
        "turn_type": "followup"
    }
}
```

This matches the production trace format from `agent.run()`, enabling seamless integration.

### The Feedback Loop

```
1. Production
   └─> Agent runs with @traced decorators
   └─> All interactions logged to Braintrust

2. Identify Edge Cases
   └─> Browse production traces in Braintrust dashboard
   └─> Find interesting/failing examples

3. Add to Eval Dataset
   └─> Use dataset_manager.py to add examples
   └─> Examples stored in Braintrust dataset

4. Run Evals
   └─> Evaluate agent on updated dataset
   └─> Measure performance on edge cases

5. Improve Agent
   └─> Fix issues identified in evals
   └─> Re-run evals to verify improvements

6. Deploy → back to step 1
```

### Usage

**Load eval data into Braintrust dataset:**

```bash
python -m evals.dataset_manager
```

**Push scorers to Braintrust platform:**

```bash
# Upload all 4 scorers to Braintrust for reuse across projects
uv run braintrust push braintrust_scorers.py --if-exists replace
```

This uploads:
- `factual-accuracy-v1` - LLM-as-judge (native format) for factual claims
- `citation-quality-v1` - Custom handler with hybrid metrics for citations
- `answer-completeness-v1` - LLM-as-judge (native format) for query coverage
- `retrieval-quality-v1` - Custom handler with hybrid precision/recall metrics

**Key Features:**
- **Native LLM Scorers**: Use Braintrust's declarative `messages` + `choice_scores` format
- **Custom Handlers**: Full control with Python functions for deterministic/hybrid scoring
- **Self-Contained**: All handlers use only Braintrust-available packages (no external imports)
- **Metadata Returns**: Scorers return diagnostic metadata for transparency

**Add production examples to eval dataset:**

```python
from evals.dataset_manager import DatasetManager

manager = DatasetManager()
manager.add_production_example(
    dataset_name="conversational-search-eval-v1",
    query="What about the controversy?",
    chat_history=["Tell me about AI safety", "What are the main concerns?"],
    response="I need more context...",
    sources=[],
    metadata={
        "source": "production",
        "reason": "Edge case: vague reference needing clarification",
    }
)
```

**Run evals with dataset:**

```bash
# Simple single-turn eval with hosted scorers (recommended for quick testing)
uv run python evals/simple_eval.py

# Full multi-turn eval with conversation context (default: 5 parallel conversations)
uv run python evals/run_evals.py

# Adjust concurrency for faster evaluation
uv run python evals/run_evals.py --max-concurrency 10

# Skip dataset loading for faster startup
uv run python evals/run_evals.py --no-dataset

# Small test (first conversation only)
uv run python evals/test_eval_small.py
```

**Simple Eval Features:**
- Uses `Eval()` function for simplicity
- Filters dataset to first-turn queries only
- Uses hosted scorers from Braintrust (no local imports)
- Automatic parallelization and progress tracking

**Full Eval Performance:**
- Default concurrency (5): ~20 conversations in parallel
- Each conversation maintains its own thread_id for context
- Conversations evaluated independently and safely in parallel
- Typical speedup: 3-5x faster than sequential execution

**Use hosted scorers in custom evals:**

```python
from braintrust import init_function

# Reference hosted scorers instead of local functions
factual_accuracy = init_function(
    project_name="conversational-search",
    slug="factual-accuracy-v1"
)

citation_quality = init_function(
    project_name="conversational-search",
    slug="citation-quality-v1"
)

# Use in Eval()
Eval(
    name="my-eval",
    data=my_data,
    task=my_task,
    scores=[factual_accuracy, citation_quality],
)
```

### Key Benefits

1. **Same Format**: Production and eval data use identical structure
2. **Easy Addition**: One function call to add production examples
3. **Continuous Improvement**: Edge cases feed back into eval dataset
4. **Realistic Evals**: Test on real user interactions, not just synthetic data

## Future Enhancements

- Additional scorers: source diversity, query quality, coherence, context awareness
- Real user conversation collection
- Multiple tool support (calculator, code execution, etc.)
- Persistent memory backend (PostgreSQL, Redis)
- Streaming responses
- Cost optimization and caching

## License

MIT

## References

- [OpenAI SDK](https://github.com/openai/openai-python)
- [Braintrust Docs](https://www.braintrust.dev/docs)
- [Tavily API](https://docs.tavily.com)
- [bitswired agentic-loop](https://github.com/bitswired/demos/tree/main/projects/agentic-loop)
