# AI Agents & Orchestration — Interview Study Guide

**Last updated:** 2026-04-01 | **Sources:** LangGraph Docs, 12-Factor Agents (475 HN pts), AI Engineering (Chip Huyen), PydanticAI Docs, TechDocs, HN practitioner discussions
**Purpose:** Definitive interview prep — same depth as transformers-fundamentals.md

---

## 1. What Is an AI Agent?

### One-liner
An AI agent is an LLM-powered system that can perceive its environment, plan actions, use tools, and take autonomous steps to accomplish a goal.

### The Agent Equation
```
Agent = LLM (brain) + Tools (hands) + Memory (context) + Planning (strategy)
```

- **LLM:** The reasoning engine that interprets tasks and decides next steps
- **Tools:** Functions the agent can call (APIs, databases, search, code execution)
- **Memory:** Short-term (conversation history) + long-term (vector store, KV store)
- **Planning:** Decomposing complex tasks into executable steps

### The Spectrum: From Chatbot to Multi-Agent System
```
Chatbot → Tool-User → Agent → Multi-Agent System
  │           │          │            │
  │           │          │            └─ Multiple agents coordinating
  │           │          └─ Autonomous loop: plan → act → observe → repeat
  │           └─ LLM calls functions but human controls the loop
  └─ Stateless Q&A, no tools, no memory
```

**Why this matters in interviews:** Most "agents" in production are actually tool-users or simple workflows. The interviewer wants to know if you understand the difference.

### Karpathy's Framing: "Software 3.0"
LLMs are "jagged intelligence" — brilliant at some tasks, terrible at others, with no smooth gradient between. The engineering challenge is NOT making the LLM smarter. It's building the **runtime** — the scaffolding that manages state, handles errors, retries failed tool calls, persists sessions, and keeps a human in the loop.

### Chip Huyen's Agent Definition (AI Engineering Book)
An agent is characterized by its **environment** and its **set of actions**. ChatGPT is an agent (web search, code execution, image gen). RAG systems are agents (text retrievers and SQL executors are tools).

**Compound mistakes are the core problem:** If model accuracy is 95% per step, over 10 steps accuracy drops to 60%. Over 100 steps: 0.6%. This is why agent reliability engineering matters.

### HN Practitioner Reality Check
> "Most 'AI Agents' that make it to production aren't actually that agentic. The best ones are mostly just well-engineered software with LLMs sprinkled in at key points." — HN, 12-Factor Agents thread (475 pts)

> "Plan for cost at scale. Whenever something might be handled by a deterministic component, try that first. Not only saves on hallucinations and latency, but could make a huge difference in your bottom line." — HN practitioner

---

## 2. Agent Patterns

### ReAct (Reason + Act) — Yao et al., 2023

**How it works:**
```
Loop:
  1. Think   → LLM reasons about what to do next
  2. Act     → LLM selects a tool and arguments
  3. Observe → Tool executes, returns result
  4. Repeat until done or max_iterations
```

**Complete ReAct loop in Python:**
```python
import json
from openai import OpenAI

client = OpenAI()

def react_agent(query: str, tools: dict, max_steps: int = 10) -> str:
    messages = [
        {"role": "system", "content": f"""You are a helpful agent.
Available tools: {json.dumps([t['schema'] for t in tools.values()])}
Think step by step. When you need information, call a tool.
When you have the final answer, respond directly."""},
        {"role": "user", "content": query}
    ]

    for step in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=[t["schema"] for t in tools.values()],
        )
        msg = response.choices[0].message
        messages.append(msg)

        # Terminal condition: no tool calls = final answer
        if not msg.tool_calls:
            return msg.content

        # Execute each tool call
        for tc in msg.tool_calls:
            fn = tools[tc.function.name]["fn"]
            args = json.loads(tc.function.arguments)
            result = fn(**args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result)
            })

    return "Max iterations reached"
```

**Strengths:** Simple, effective for tool use, well-supported by all frameworks
**Weaknesses:** Can loop indefinitely, no upfront planning, greedy (one step at a time)

### Plan-and-Execute

**How it works:**
```
1. Plan    → LLM creates a FULL PLAN upfront (list of steps)
2. Execute → Run each step sequentially
3. Re-plan → After each step, optionally re-plan if results differ
```

**Code skeleton:**
```python
from pydantic import BaseModel

class Plan(BaseModel):
    steps: list[str]
    current_step: int = 0

def plan_and_execute(query: str, tools: dict) -> str:
    # Step 1: Generate plan
    plan_response = llm.invoke(
        f"Create a step-by-step plan to answer: {query}\n"
        f"Available tools: {list(tools.keys())}\n"
        f"Return a numbered list of steps."
    )
    plan = parse_plan(plan_response)

    results = []
    for i, step in enumerate(plan.steps):
        # Step 2: Execute each step
        result = execute_step(step, tools, context=results)
        results.append(result)

        # Step 3: Check if re-planning needed
        if should_replan(step, result, plan.steps[i+1:]):
            remaining = replan(query, results, plan.steps[i+1:])
            plan.steps = plan.steps[:i+1] + remaining

    return synthesize(query, results)
```

**When to use:** Complex multi-step workflows, when you need predictable/auditable execution
**Weakness:** Upfront plan may be wrong; re-planning adds latency

**Why (from Chip Huyen):** "Planning should be decoupled from execution. You ask the agent to first generate a plan, and only after this plan is validated is it executed." This prevents fruitless execution of bad plans.

### Multi-Agent (Supervisor Pattern)

**How it works:**
```
User Query → Supervisor Agent
                 ├── "research query"  → Researcher Agent
                 ├── "code task"       → Coder Agent
                 ├── "review needed"   → Reviewer Agent
                 └── "send email"      → Comms Agent (+ human approval)
```

**LangGraph implementation:**
```python
from langchain.tools import tool
from langchain.agents import create_agent

# Specialized sub-agents
researcher = create_agent(model="claude-sonnet-4-20250514", tools=[search, fetch])
coder = create_agent(model="claude-sonnet-4-20250514", tools=[run_code, read_file])

# Wrap as tools for supervisor
@tool("research", description="Research a topic and return findings")
def call_researcher(query: str):
    result = researcher.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content

@tool("code", description="Write or debug code")
def call_coder(task: str):
    result = coder.invoke({"messages": [{"role": "user", "content": task}]})
    return result["messages"][-1].content

# Supervisor coordinates
supervisor = create_agent(
    model="claude-sonnet-4-20250514",
    tools=[call_researcher, call_coder]
)
```

**Key characteristics (from LangGraph docs):**
- Centralized control: all routing through supervisor
- Context isolation: each sub-agent gets clean context window
- Subagents are stateless — memory lives in supervisor
- Can execute sub-agents in parallel

**When to use:** Different domains need different expertise, context isolation needed, team-based development

### LATS (Language Agent Tree Search)

```
1. Generate multiple possible next actions (tree branches)
2. Evaluate each branch with a value function (LLM-as-judge)
3. Select best branch (Monte Carlo tree search style)
4. If branch fails → backtrack and try another
```

**When to use:** Tasks requiring exploration and backtracking (code generation, math proofs)
**Weakness:** Expensive (many LLM calls per step), high latency

### Decision Framework Table
| Task Type | Pattern | Why |
|---|---|---|
| Simple Q&A with tools | ReAct | Quick, low overhead, well-understood |
| Multi-step workflow | Plan-and-Execute | Predictable, auditable, can validate plan |
| Complex reasoning/code gen | LATS | Explores alternatives, can backtrack |
| Multi-domain tasks | Supervisor | Specialized expertise, context isolation |
| Production reliability | Stateless Reducer (Factor 12) | Testable, debuggable, persistent |
| Latency-critical | Single tool-call | Skip the loop, one LLM call with structured output |

---

## 3. LangGraph Deep Dive

### Mental Model
```
LangChain:  Linear chains (A → B → C)
LangGraph:  Directed graphs with conditional edges = state machines for agents
            "React Router for AI agents"
```

**The five-step process (from official docs):**
1. Map out your workflow as discrete steps (nodes)
2. Identify what each step needs to do (LLM/data/action/user-input)
3. Design your state (shared memory across nodes)
4. Build your nodes (functions that read/write state)
5. Wire it together (edges, conditionals, compile)

### Three Building Blocks

**1. State** — Shared typed dict with annotated reducers
```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # auto-appends, deduplicates by ID
    retrieved_docs: list[str]
    classification: dict | None
    attempt_count: int
```

**Why `Annotated[list, add_messages]`?** The `add_messages` reducer handles append + dedup. Without it, returning `{"messages": [new_msg]}` would REPLACE the entire list. With it, the new message is APPENDED. This is the single most common LangGraph gotcha.

**2. Nodes** — Functions that take state, return state updates
```python
def retrieve(state: AgentState) -> dict:
    """Data step: query vector store"""
    query = state["messages"][-1].content
    docs = vector_store.search(query, k=5)
    return {"retrieved_docs": [d.content for d in docs]}

def generate(state: AgentState) -> dict:
    """LLM step: generate response from context"""
    context = "\n".join(state["retrieved_docs"])
    prompt = f"Context: {context}\n\nQuestion: {state['messages'][-1].content}"
    response = llm.invoke(prompt)
    return {"messages": [response]}
```

**Key principle:** Keep state raw, format prompts on-demand. State stores data; nodes format it into prompts. This means different nodes can format the same data differently.

**3. Edges** — Transitions (fixed or conditional)
```python
# Fixed edge: always go from retrieve to generate
graph.add_edge("retrieve", "generate")

# Conditional edge: route based on state
def route_by_intent(state: AgentState) -> str:
    intent = state["classification"]["intent"]
    if intent == "billing" or state["classification"]["urgency"] == "critical":
        return "human_review"
    elif intent in ["question", "feature"]:
        return "search_docs"
    return "draft_response"

graph.add_conditional_edges("classify", route_by_intent)
```

### The Command Pattern — Dynamic Routing from Inside Nodes
```python
from langgraph.types import Command
from typing import Literal

def classify_intent(state: AgentState) -> Command[Literal["search", "human_review", "respond"]]:
    """Node that routes using Command instead of conditional edges"""
    classification = structured_llm.invoke(f"Classify: {state['messages'][-1].content}")

    # Command combines state update AND routing in one return
    if classification["urgency"] == "critical":
        return Command(update={"classification": classification}, goto="human_review")
    elif classification["intent"] == "question":
        return Command(update={"classification": classification}, goto="search")
    else:
        return Command(update={"classification": classification}, goto="respond")
```

**Why Command over conditional edges?** Routing logic lives WITH the node that has the context to make the decision. Type hints (`Literal[...]`) make the possible routes explicit and traceable.

### Complete Code: RAG Agent with Retrieve/Generate/Fallback
```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, RetryPolicy
from typing import TypedDict, Annotated, Literal

class RAGState(TypedDict):
    messages: Annotated[list, add_messages]
    docs: list[str]
    answer_quality: str | None

def retrieve(state: RAGState) -> dict:
    query = state["messages"][-1].content
    docs = vector_store.similarity_search(query, k=5)
    return {"docs": [d.page_content for d in docs]}

def generate(state: RAGState) -> Command[Literal["grade", END]]:
    context = "\n---\n".join(state["docs"])
    response = llm.invoke(
        f"Answer based on context:\n{context}\n\nQ: {state['messages'][-1].content}"
    )
    return Command(
        update={"messages": [response]},
        goto="grade"
    )

def grade_answer(state: RAGState) -> Command[Literal["retrieve", END]]:
    """LLM-as-judge: is the answer grounded in the docs?"""
    grade = grader_llm.invoke(
        f"Is this answer grounded in the docs? Answer: {state['messages'][-1].content}"
    )
    if grade.score < 0.7 and state.get("attempt_count", 0) < 3:
        return Command(update={"answer_quality": "retry"}, goto="retrieve")
    return Command(update={"answer_quality": "accepted"}, goto=END)

# Wire it together
workflow = StateGraph(RAGState)
workflow.add_node("retrieve", retrieve, retry_policy=RetryPolicy(max_attempts=3))
workflow.add_node("generate", generate)
workflow.add_node("grade", grade_answer)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")

app = workflow.compile(checkpointer=MemorySaver())

# Run
config = {"configurable": {"thread_id": "rag-session-1"}}
result = app.invoke({"messages": [HumanMessage("How does Flash Attention work?")]}, config)
```

---

## 4. Session Persistence & Checkpointing

### Why Persistence Matters
1. **Resume sessions** — user comes back hours later, conversation continues
2. **Human-in-the-loop** — pause at interrupt(), resume after approval
3. **Fault tolerance** — crash recovery from last checkpoint
4. **Debugging** — replay any state to reproduce bugs
5. **Audit trail** — full history of agent decisions

### Checkpointer Hierarchy
| Backend | Use Case | Durability |
|---|---|---|
| `MemorySaver` | Dev/testing | Lost on restart |
| `SqliteSaver` | Local persistence | Single process |
| `PostgresSaver` | Production | Multi-process, distributed |

**Production setup:**
```python
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost:5432/agents"
)

graph = builder.compile(checkpointer=checkpointer)

# thread_id = one conversation session
config = {"configurable": {"thread_id": "user-123-session-1"}}
result = graph.invoke({"messages": [user_message]}, config=config)

# Hours later — state automatically loaded from Postgres
result2 = graph.invoke({"messages": [follow_up]}, config=config)
```

### Cross-Thread Memory (LangGraph Store)
For data that spans sessions — user preferences, learned facts, relationship history:
```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
# Store user preferences (persists across thread_ids)
store.put(("user_123", "preferences"), "theme", {"value": "dark_mode"})
# Retrieve in any session
prefs = store.get(("user_123", "preferences"), "theme")
```

### Memory Hierarchy
```
┌─────────────────────────────────────────────┐
│  Working Memory (current context window)     │ ← messages in state
├─────────────────────────────────────────────┤
│  Session Memory (checkpointed state)         │ ← PostgresSaver per thread_id
├─────────────────────────────────────────────┤
│  Summary Memory (compressed old turns)       │ ← LLM summarizes, replaces old msgs
├─────────────────────────────────────────────┤
│  Long-Term Memory (cross-session)            │ ← Vector store (pgvector) + KV store
└─────────────────────────────────────────────┘
```

---

## 5. Human-in-the-Loop

### Why: Safety, Approval, Escalation
Not every agent action should be autonomous. High-stakes actions (send email, deploy code, transfer money) need human approval. The agent should PAUSE, not skip.

### LangGraph `interrupt()` Mechanism

**How it works:**
1. Node calls `interrupt(payload)` — execution suspends
2. State saved to checkpointer
3. Payload returned to caller under `__interrupt__`
4. Graph waits indefinitely
5. Caller resumes with `Command(resume=value)`
6. Resume value becomes the return of `interrupt()` inside the node
7. **Critical:** Node re-executes from the beginning on resume

```python
from langgraph.types import interrupt, Command

def send_email_node(state: AgentState) -> Command[Literal["confirm", "cancel"]]:
    # interrupt() MUST come first — code before it re-runs on resume
    approval = interrupt({
        "action": "send_email",
        "to": state["email_to"],
        "draft": state["email_draft"],
        "message": "Approve sending this email?"
    })

    if approval.get("approved"):
        send_email(state["email_draft"])
        return Command(update={"status": "sent"}, goto="confirm")
    return Command(update={"status": "cancelled"}, goto="cancel")
```

**Resume after human input:**
```python
config = {"configurable": {"thread_id": "email-workflow-42"}}

# Initial run — pauses at interrupt
result = graph.invoke(initial_state, config)
print(result["__interrupt__"])  # Shows the approval request

# Human reviews and approves (could be hours later)
final = graph.invoke(
    Command(resume={"approved": True, "edited_draft": "..."}),
    config
)
```

### Interrupt Rules (from LangGraph docs — easy to get wrong)
1. **Never wrap `interrupt()` in try/except** — it throws an exception internally
2. **Keep interrupt order consistent** — matching is index-based, not content-based
3. **Side effects before `interrupt()` must be idempotent** — node re-runs from start
4. **Only JSON-serializable payloads** — no functions, no class instances

### Interrupt in Tools (approval inside tool itself)
```python
from langchain.tools import tool

@tool
def transfer_money(to: str, amount: float):
    """Transfer money to a recipient."""
    response = interrupt({
        "action": "transfer_money",
        "to": to, "amount": amount,
        "message": f"Approve ${amount} transfer to {to}?"
    })
    if response.get("approved"):
        return execute_transfer(to, amount)
    return "Transfer cancelled by user"
```

---

## 6. The 12-Factor Agents

From HumanLayer's open-source guide (475 pts on HN). Production principles, not academic patterns.

### Overview Table
| # | Factor | Core Idea |
|---|---|---|
| 1 | **NL to Tool Calls** | LLM's primary job: convert natural language to structured tool calls |
| 2 | **Own Your Prompts** | Don't hide prompts behind framework abstractions. Version and debug them. |
| 3 | **Own Your Context Window** | Explicitly manage what goes in. Pre-fetch likely needs. Remove irrelevant. |
| 4 | **Tools Are Structured Outputs** | Tool calls = JSON outputs your code interprets. Not magic. |
| 5 | **Unify LLM and Code Errors** | Hallucinated tool names and API timeouts flow through same error handling. |
| 6 | **Launch / Iterate / Evaluate** | Ship fast, measure with evals, iterate. Don't over-design. |
| 7 | **Contact Humans with Tool Calls** | Human-in-the-loop = a tool (`ask_human`), not a separate system. |
| 8 | **Own Your Control Flow** | Don't let the framework decide when to loop/retry/stop. Write explicit flow. |
| 9 | **Compact Errors** | Don't dump full stack traces to LLM. Summarize into actionable messages. |
| 10 | **Small, Focused Agents** | Composition over monolith. Clear responsibilities per agent. |
| 11 | **Trigger from Anywhere** | CLI, API, webhook, cron — not tied to one interface. |
| 12 | **Stateless Reducer** | `(state, event) -> (new_state, effects)` — agent as pure function. |

### Deep Dive: Factor 8 — Own Your Control Flow

**The problem:** Frameworks often control your agent loop. You call `agent.run()` and hope for the best. But you need to: break the loop for human approval, handle different tool types differently, implement custom caching, add rate limiting.

**The solution:** Write your own loop.
```python
def handle_next_step(thread: Thread):
    while True:
        next_step = await determine_next_step(thread_to_prompt(thread))

        if next_step.intent == 'request_clarification':
            # Async: break loop, wait for human webhook
            await send_message_to_human(next_step)
            await db.save_thread(thread)
            break

        elif next_step.intent == 'fetch_open_issues':
            # Sync: execute immediately, continue loop
            issues = await linear_client.issues()
            thread.events.append({"type": "issues_result", "data": issues})
            continue

        elif next_step.intent == 'deploy_backend':
            # High-stakes: break for human approval
            await request_human_approval(next_step)
            await db.save_thread(thread)
            break
```

**Why this matters:** The #1 feature request from practitioners is interrupting between tool SELECTION and tool INVOCATION. Without this, you either restrict agents to low-risk calls or "yolo hope it doesn't screw up."

### Deep Dive: Factor 12 — Make Your Agent a Stateless Reducer

```python
def agent_step(state: AgentState, event: Event) -> tuple[AgentState, list[Effect]]:
    """Pure function: state + event -> new_state + side effects"""
    messages = state.messages + [event.message]
    response = llm.chat(messages)

    if response.tool_calls:
        effects = [ExecuteTool(tc) for tc in response.tool_calls]
        new_state = state.with_messages(messages + [response])
        return new_state, effects
    else:
        return state.with_messages(messages + [response]), [Complete(response.content)]
```

**Why this is the most important factor:**
- **Testable** — mock the LLM, assert state transitions
- **Debuggable** — replay any state to reproduce bugs exactly
- **Persistent** — serialize state to DB, resume anywhere
- **Composable** — chain agent_step calls, build pipelines

**Testing a stateless reducer:**
```python
def test_agent_routes_to_search_on_question():
    state = AgentState(messages=[HumanMessage("What is RAG?")])
    event = Event(message=HumanMessage("What is RAG?"))

    # Mock LLM to return a search tool call
    with mock_llm(returns=ToolCall(name="search", args={"q": "RAG"})):
        new_state, effects = agent_step(state, event)

    assert len(effects) == 1
    assert isinstance(effects[0], ExecuteTool)
    assert effects[0].tool_call.name == "search"

def test_agent_completes_on_final_answer():
    state = AgentState(messages=[HumanMessage("Hi"), AIMessage("Hello!")])
    event = Event(message=HumanMessage("Thanks"))

    with mock_llm(returns=AIMessage("You're welcome!")):
        new_state, effects = agent_step(state, event)

    assert len(effects) == 1
    assert isinstance(effects[0], Complete)
```

---

## 7. Concurrency, Retries, Error Handling

### Error Handling Strategy (from LangGraph docs)
| Error Type | Who Fixes It | Strategy |
|---|---|---|
| Transient (network, rate limits) | System (automatic) | Retry policy with backoff |
| LLM-recoverable (tool failure, parse error) | LLM | Feed error back to LLM, let it retry |
| User-fixable (missing info, ambiguous) | Human | `interrupt()` to collect input |
| Unexpected (bugs) | Developer | Let it bubble up, don't catch |

### Async Tool Execution with Retry
```python
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
async def call_tool_with_retry(tool_fn, args: dict) -> str:
    """Retry with exponential backoff, timeout per call."""
    return await asyncio.wait_for(tool_fn(**args), timeout=30.0)

async def execute_tools_parallel(tool_calls: list) -> list:
    """Execute multiple tool calls concurrently."""
    tasks = [
        call_tool_with_retry(get_fn(tc.name), tc.arguments)
        for tc in tool_calls
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [
        compact_error(r) if isinstance(r, Exception) else str(r)
        for r in results
    ]
```

### LangGraph Retry Policy (built-in)
```python
from langgraph.types import RetryPolicy

workflow.add_node(
    "search_documentation",
    search_documentation,
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0)
)
```

### Circuit Breaker Pattern
```python
class CircuitBreaker:
    """Prevents hammering a failing service."""
    def __init__(self, failure_threshold=5, reset_timeout=60):
        self.failures = 0
        self.threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure = None
        self.state = "CLOSED"  # CLOSED → OPEN → HALF_OPEN

    async def call(self, fn, *args):
        if self.state == "OPEN":
            if time.time() - self.last_failure > self.reset_timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitOpenError("Circuit is open — service unavailable")
        try:
            result = await fn(*args)
            self.failures = 0
            self.state = "CLOSED"
            return result
        except Exception:
            self.failures += 1
            self.last_failure = time.time()
            if self.failures >= self.threshold:
                self.state = "OPEN"
            raise
```

### Factor 9: Compact Errors (don't dump traces to LLM)
```python
def compact_error(error: Exception) -> str:
    """Convert exceptions to LLM-friendly messages."""
    if isinstance(error, TimeoutError):
        return "Tool timed out after 30s. Try a simpler query or different tool."
    elif isinstance(error, RateLimitError):
        return "Rate limited. Wait and retry, or use a different data source."
    elif isinstance(error, ValidationError):
        return f"Invalid parameters: {error.message}. Check the tool schema."
    else:
        return f"Tool failed: {type(error).__name__}. Try an alternative approach."
```

**Why:** Full stack traces waste context tokens and confuse the LLM. Actionable summaries help it recover.

---

## 8. Agent Evaluation & Testing

### Chip Huyen's Failure Mode Taxonomy

**Planning Failures:**
| Failure | Example |
|---|---|
| Invalid tool | Plan calls `bing_search` but it's not in inventory |
| Valid tool, invalid params | Calls `lbs_to_kg` with 2 params (needs 1) |
| Valid tool, wrong values | Calls `lbs_to_kg(lbs=100)` when it should be 120 |
| Goal failure | Asked for SF→Hanoi trip, plans SF→Ho Chi Minh City |
| Reflection error | Assigns 40/50 people to rooms, insists task is done |

**Tool Failures:** Correct tool used, but wrong output (bad SQL, wrong image caption, missing tool for domain).

**Efficiency Failures:** Valid plan, correct tools, but wasteful (too many steps, too many LLM calls, too expensive).

### Planning Evaluation Metrics
1. What % of generated plans are valid?
2. How many attempts to get a valid plan?
3. What % of tool calls are valid?
4. How often are invalid tools called?
5. How often are valid tools called with invalid parameters?

### Testing Pyramid for Agents
```
         ┌─────────────┐
         │  Eval Suite  │  Real LLM, golden set, scored
         │  (expensive) │  on correctness + safety
         ├─────────────┤
         │ Integration  │  Mocked LLM, assert state
         │   Tests      │  transitions + tool sequences
         ├─────────────┤
         │  Unit Tests  │  Each tool function tested
         │  (fast)      │  independently, no LLM
         └─────────────┘
```

**Unit tests:** Test each tool in isolation. Mock APIs, assert return values.
```python
def test_search_tool_returns_docs():
    with mock_vector_store(returns=["doc1", "doc2"]):
        result = search_tool("What is RAG?")
    assert len(result) == 2
    assert "doc1" in result
```

**Integration tests:** Mock the LLM, test the agent loop.
```python
def test_agent_calls_search_then_generates():
    mock_responses = [
        ToolCall(name="search", args={"q": "RAG"}),      # First LLM call
        AIMessage("RAG stands for Retrieval Augmented...") # Second LLM call
    ]
    with mock_llm(responses=mock_responses):
        result = agent.invoke({"messages": [HumanMessage("What is RAG?")]})

    assert "Retrieval Augmented" in result["messages"][-1].content
    assert agent.tool_calls_made == ["search"]
```

**Eval suite:** Real LLM, golden set of (query, expected_behavior) pairs.
```python
GOLDEN_SET = [
    {"query": "Reset my password", "expected_tool": "search_docs", "expected_topic": "password"},
    {"query": "I was charged twice", "expected_escalation": True, "expected_urgency": "critical"},
]

def run_eval(agent, golden_set):
    results = []
    for case in golden_set:
        output = agent.invoke({"messages": [HumanMessage(case["query"])]})
        results.append({
            "correct_tool": check_tool_used(output, case),
            "correct_escalation": check_escalation(output, case),
            "latency": output.metadata["latency_ms"],
            "cost": output.metadata["total_tokens"] * COST_PER_TOKEN,
        })
    return aggregate_metrics(results)
```

---

## 9. Framework Comparison (2026)

| | **LangGraph** | **PydanticAI** | **CrewAI** | **Strands SDK** |
|---|---|---|---|---|
| **Model** | State machine (graph) | Agent + tools + deps | Role-based crews | Event-driven |
| **By** | LangChain | Pydantic team | Community | AWS |
| **Strength** | Flexibility, checkpointing, multi-agent | Type safety, DI, clean code | Multi-agent ease | AWS integration |
| **Weakness** | Verbose, learning curve | Less orchestration | Less control | New, less community |
| **Production** | Most mature | Growing fast | Limited | Early |
| **When** | Complex workflows, persistence needed | Type-safe agents, simple flows | Quick prototypes | AWS-native |

### PydanticAI Example (for contrast)
```python
from pydantic_ai import Agent, RunContext

support_agent = Agent(
    'openai:gpt-4o',
    deps_type=DatabaseConn,       # Dependency injection
    output_type=SupportResponse,  # Structured output via Pydantic model
    system_prompt="You are a customer support agent..."
)

@support_agent.tool
async def lookup_order(ctx: RunContext[DatabaseConn], order_id: str) -> str:
    """Look up order details."""
    return await ctx.deps.query(f"SELECT * FROM orders WHERE id = '{order_id}'")

# Type-safe, dependency-injected run
result = support_agent.run_sync("Where is my order #12345?", deps=db_conn)
print(result.output)  # SupportResponse with validated fields
```

**Why PydanticAI matters:** Type safety via generics (`Agent[DepsType, OutputType]`), dependency injection for testability, and Pydantic's validation for structured outputs. No graph complexity when you don't need it.

### When to Pick What (Interview Answer)
"For complex orchestration with persistence, multi-agent, and human-in-the-loop, I'd use **LangGraph** — it's the most mature and gives full control over the state machine. For simpler agents where type safety and clean code matter, **PydanticAI** — its dependency injection makes testing easy. For AWS-native projects, evaluate **Strands SDK**. And for quick prototyping, **CrewAI**. The key principle from 12-Factor Agents is **own your control flow** — don't let any framework make decisions you should be making."

---

## 10. Interview Q&A

### Q1: "How would you build an agent that searches documents, queries a database, and sends emails?"

**What they're testing:** System design, orchestration patterns, safety awareness.

**Strong answer:** "I'd use LangGraph with a supervisor pattern. Three tool nodes: `search_docs`, `query_db`, `send_email`. The supervisor routes based on intent classification. Email sending gets a human-in-the-loop `interrupt()` since it's irreversible. State is checkpointed to PostgreSQL for session persistence. Each tool call has retry with exponential backoff and a 30-second timeout. The agent is a stateless reducer — `(state, event) -> (new_state, effects)` — so it's testable and debuggable. For the context window, I'd pre-fetch likely-needed data (Factor 3) rather than waiting for the LLM to ask for it."

**Follow-ups:**
- "How do you handle the email tool failing?" → Factor 5 (unify errors), compact the error (Factor 9), feed back to LLM to retry or use alternative
- "What if the user comes back tomorrow?" → PostgresSaver + thread_id, state loads automatically
- "How do you test this?" → Three levels: unit (tools), integration (mocked LLM), eval (golden set)

**Your angle:** "At Lexsi AI, I built MoE backends with dynamic routing across expert models — that's essentially the same pattern as a supervisor routing to specialized sub-agents. And SafeSpecch was a multi-step pipeline (transcribe → detect → filter → flag) which is structurally identical to a plan-and-execute agent."

---

### Q2: "Tool call fails mid-execution. What happens?"

**What they're testing:** Error handling, resilience, 12-Factor awareness.

**Strong answer:** "Following Factor 5 (unify LLM and code errors): I compact the error into an actionable message — not a stack trace — and feed it back to the LLM (Factor 9). The LLM can decide to retry with different parameters, use an alternative tool, or ask the user. For transient failures, I use retry with exponential backoff via tenacity or LangGraph's built-in RetryPolicy. For persistent failures, a circuit breaker prevents hammering a dead service. The failed state is checkpointed, so I can debug and resume from the exact failure point."

**Follow-ups:**
- "What if the LLM keeps retrying the same broken call?" → Max iterations + circuit breaker
- "How do you distinguish transient from permanent failures?" → HTTP status codes, error types, circuit breaker state

**Your angle:** "In our RAG chatbot at Lexsi, we had to handle embedding API failures gracefully — retry transient errors, fall back to cached embeddings for persistent failures. Same error unification pattern."

---

### Q3: "How do you test an agent?"

**What they're testing:** Engineering rigor, understanding of non-determinism.

**Strong answer:** "Three levels. *Unit tests:* each tool function tested independently with mocked external services. *Integration tests:* the agent loop with a mocked LLM returning deterministic responses — I assert that given input state, the agent produces expected tool calls and state transitions. The stateless reducer pattern (Factor 12) makes this straightforward — it's just `assert agent_step(state, event) == (expected_state, expected_effects)`. *Eval suite:* golden set of (query, expected_behavior) pairs run against the real LLM, scored on task completion, tool accuracy, and safety. I track metrics like valid plan %, correct tool usage %, and cost per task."

**Follow-ups:**
- "How do you handle LLM non-determinism in tests?" → Set temperature=0 for evals, use semantic similarity for output matching, focus on behavior (correct tool called) not exact text
- "What's in your golden set?" → Representative cases per intent category + edge cases + adversarial inputs

---

### Q4: "Explain ReAct vs Plan-and-Execute"

**What they're testing:** Pattern knowledge, ability to reason about tradeoffs.

**Strong answer:** "ReAct is a tight loop — think, act, observe, repeat. The LLM decides the next action based only on the current observation. It's greedy: no upfront plan. Great for simple tool use, but can loop and has no global strategy. Plan-and-Execute decouples planning from execution — the LLM first generates a complete plan, that plan gets validated (is it feasible? does it use valid tools?), and only then is it executed step by step, with optional re-planning after each step.

The choice depends on the task. Simple Q&A with tools? ReAct. Complex multi-step workflow where you need auditability? Plan-and-Execute. The AI Engineering book emphasizes that decoupling planning from execution prevents 'fruitless execution' — you don't want an agent running a 1,000-step bad plan for hours before you notice."

**Follow-ups:**
- "When would you combine them?" → Plan-and-Execute for the high-level strategy, ReAct for executing each individual step
- "What about LATS?" → Tree search with backtracking — for tasks where the first path might fail (code gen, proofs)

---

### Q5: "How do you persist agent state across requests?"

**What they're testing:** Architecture, production readiness.

**Strong answer:** "LangGraph checkpointing to PostgreSQL. Each session has a `thread_id`. When a new request arrives, the checkpointer loads the full state — messages, retrieved docs, classification results, user preferences — from the last checkpoint. The agent continues from where it left off. For cross-session memory (user preferences, learned facts), I use LangGraph Store or a vector database (pgvector) for semantic retrieval of relevant past interactions. There's a memory hierarchy: working memory (current context window) → session memory (checkpointed) → summary memory (LLM-compressed old turns) → long-term memory (vector store)."

**Follow-ups:**
- "What about context window limits?" → Summary memory: LLM compresses old turns, sliding window keeps recent + summary
- "How do you handle stale memory?" → Relevance decay, importance scoring, active forgetting of outdated facts

---

### Q6: "How do you prevent an agent from going rogue?"

**What they're testing:** Safety thinking, production maturity.

**Strong answer:** "Multiple layers. *Structural:* human-in-the-loop `interrupt()` for all high-stakes actions (email, deployment, financial). *Operational:* max iteration limits (LangGraph recursion_limit), timeouts per tool call (asyncio.wait_for), cost caps per session. *Observability:* OTEL tracing on every agent invocation, tool execution, and LLM call with token counts and latency. *Guardrails:* output validation before execution — if the LLM hallucinates a tool name, it's caught by the tool registry, not executed blindly. Factor 8 (own your control flow) is key — you decide when to loop, when to stop, when to escalate."

**Follow-ups:**
- "What about prompt injection?" → Input validation, separate system/user message roles, tool result sanitization
- "How do you monitor in production?" → OTEL spans (invoke_agent, execute_tool, chat), dashboards for latency/cost/failure rates

**Your angle:** "At Lexsi, I built DLBacktrace — an explainability engine that traces information flow through neural networks layer by layer. Debugging agents is philosophically similar: you need to see why the agent made each decision, which is why observability and the stateless reducer pattern (replayable states) are essential."

---

### Q7: "What are the 12-Factor Agents?"

**What they're testing:** Awareness of production best practices, not just academic patterns.

**Strong answer:** "It's an open-source guide from HumanLayer (475 points on HN) that codifies production principles for LLM agents. The most important ones: *Factor 4* — tools are just structured outputs, not magic. *Factor 8* — own your control flow, don't let frameworks decide when to loop or stop. *Factor 9* — compact errors before feeding them to the LLM. *Factor 12* — make the agent a stateless reducer: `(state, event) -> (new_state, effects)`. This makes agents testable, debuggable, and persistent.

The meta-principle is: treat agents like well-engineered software, not black boxes. Own your prompts, own your context window, own your control flow. The LLM is a library you call, not a framework that calls you."

**Follow-ups:**
- "Which factor do people get wrong most?" → Factor 3 (context window management) — most people dump everything in and hope the LLM figures it out. Pre-fetching and pruning are critical.
- "How does this relate to LangGraph?" → LangGraph implements several factors natively: checkpointing (Factor 12 persistence), interrupt (Factor 7 + 8), RetryPolicy (Factor 5)

---

### Q8: "Design a multi-agent system for code review"

**What they're testing:** System design, multi-agent architecture decisions.

**Strong answer:** "Supervisor pattern with three specialized agents: *Security Reviewer* (checks for vulnerabilities, secret leaks), *Style Reviewer* (linting, naming conventions, patterns), *Logic Reviewer* (correctness, edge cases, test coverage). The supervisor takes a PR diff, fans out to all three in parallel, collects their findings, deduplicates, and produces a unified review. Each sub-agent runs in an isolated context window (context isolation from the subagents pattern) so the security reviewer's vulnerability databases don't bloat the style reviewer's context.

I'd implement this as a LangGraph graph: START → fan_out → [security, style, logic] in parallel → aggregate → human_review (interrupt for the PR author) → END. Checkpointed to PostgreSQL so the review persists even if the service restarts."

---

### Q9: "Compare LangGraph and PydanticAI — when would you use each?"

**What they're testing:** Framework judgment, not framework loyalty.

**Strong answer:** "LangGraph is for complex orchestration — state machines, conditional routing, multi-agent, checkpointing, human-in-the-loop. It's the most mature for production but has a steeper learning curve. PydanticAI is for simpler agents where type safety matters — its dependency injection system makes testing easy, and Pydantic validation gives you structured outputs with guarantees. If I need a RAG agent with persistence and approval gates, LangGraph. If I need a type-safe customer support agent with clean DI, PydanticAI. The 12-Factor principle applies to both: own your control flow, regardless of framework."

---

### Q10: "How would you add observability to an agent in production?"

**What they're testing:** Production operations, monitoring maturity.

**Strong answer:** "OpenTelemetry with the GenAI semantic conventions. Every agent invocation gets a trace with nested spans: `invoke_agent` (top-level) → `chat` (LLM call with model name, token counts) → `execute_tool` (tool name, duration, success/failure) → `embeddings` (for RAG). Key attributes: `gen_ai.agent.name`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.conversation.id`. For tooling, Langfuse (open-source, self-hosted) or Arize Phoenix (OTEL-native). I'd dashboard: p50/p95 latency per agent, cost per conversation, tool failure rates, and human escalation rates."

---

### Q11: "What's the difference between a supervisor and a router in multi-agent systems?"

**What they're testing:** Architectural precision.

**Strong answer:** "A router is a single classification step — it looks at the input, picks the best agent, and dispatches. No memory, no multi-turn coordination. A supervisor is a full agent that maintains conversation context, dynamically decides which sub-agents to call across multiple turns, and can call multiple sub-agents per turn. The router is stateless dispatch; the supervisor is stateful orchestration. Use a router when tasks are independent and single-shot. Use a supervisor when the response requires coordinating multiple agents and maintaining context across their outputs."

---

### Q12: "Walk me through how you'd debug an agent that's giving wrong answers"

**What they're testing:** Debugging methodology, systematic thinking.

**Strong answer:** "Systematic approach following Chip Huyen's failure taxonomy. First, classify the failure: *Planning failure?* (wrong tool, wrong params, wrong values — inspect the tool calls). *Tool failure?* (right tool called but wrong output — test the tool independently). *Efficiency failure?* (correct but wasteful — too many steps).

Concrete steps: (1) Look at the OTEL trace — which node produced the wrong output? (2) Inspect the state at that checkpoint — what did the LLM see in its context? (3) Check if it's a context problem (missing relevant info, too much noise) or a reasoning problem (had the info but made wrong decision). (4) If context: improve retrieval, pre-fetch relevant data (Factor 3). If reasoning: improve the prompt, add few-shot examples, or use a more capable model. (5) Add the failing case to the golden eval set so it doesn't regress."

**Your angle:** "This is exactly like debugging DLBacktrace — I'd trace relevance scores layer by layer through the network to find where attribution broke down. For agents, I trace state checkpoint by checkpoint to find where the agent's 'reasoning' went wrong."

---

## 11. Connection to Your Experience

### MoE Backends → Agent Routing
You built MoE backends for Qwen3-MoE, JetMoE, OLMoE with dynamic expert routing. A supervisor agent routing to specialized sub-agents is architecturally identical — a gating function decides which expert/agent handles each input.

### DLBacktrace → Agent Debugging & Observability
Layer-wise relevance propagation through transformers = tracing agent decisions through checkpointed states. Both require understanding information flow through a complex system to find where things went wrong.

### SafeSpecch → Plan-and-Execute Pipeline
SafeSpecch pipeline (transcribe → detect → filter → flag) is a concrete plan-and-execute pattern. Each step is a node with clear input/output. If detection fails, you don't proceed to filtering.

### RAG Chatbot → LangGraph Agent
You built a RAG chatbot with LangChain at Lexsi. LangGraph is the evolution — same retrieval/generation pattern but with state machines, checkpointing, and conditional routing for production reliability.

### Interview Positioning
"I've been building the components that make agents work — dynamic routing (MoE), multi-step pipelines (SafeSpecch), explainable decision tracing (DLBacktrace), and retrieval-augmented generation (RAG chatbot). Agentic engineering is the unifying framework that connects all of these into autonomous, reliable systems."
