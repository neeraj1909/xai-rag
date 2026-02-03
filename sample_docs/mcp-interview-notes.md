# MCP (Model Context Protocol) - Interview Notes

## 1. MCP -- Core Concept

### One-liner
An open protocol that standardizes how LLMs connect to external tools, data, and services -- "USB-C for AI integrations."

### Why it exists (the fragmentation problem)
Before MCP, every LLM-tool integration was bespoke. Anthropic had function calling, OpenAI had a different format, Google had yet another. If you built a database tool, you needed N adapters for N providers. MCP collapses this to 1 adapter that works everywhere. Same pattern as LSP (Language Server Protocol) standardizing IDE-language integration.

### Architecture: Host -> Client -> Server

```
+--------------------------------------------------+
|  HOST (Claude Desktop, IDE, your FastAPI app)     |
|  +---------+  +---------+  +---------+           |
|  | Client 1|  | Client 2|  | Client 3|           |
|  +----+----+  +----+----+  +----+----+           |
+-------|-----------|-----------|-----------------+
        |           |           |
   +----v----+  +----v----+  +----v----+
   |Server A |  |Server B |  |Server C |
   |(DB)     |  |(API)    |  |(Files)  |
   +---------+  +---------+  +---------+
```

- **Host**: LLM application that creates and manages clients. Controls security policy.
- **Client**: 1:1 connector to a single server. Maintains a stateful session.
- **Server**: Exposes capabilities (tools, resources, prompts) to clients.

**Why this layering?** The host is the trust boundary. It decides which servers to connect, what permissions to grant, and whether to allow tool calls. Clients are isolated -- one compromised server cannot affect another client's session.

### Protocol fundamentals
- **Wire format**: JSON-RPC 2.0 over stateful connections
- **Transports**: stdio (local, for desktop apps) or HTTP+SSE (remote, for production)
- **Session lifecycle**: initialize (capability negotiation) -> operate -> shutdown
- **Capability negotiation**: Client and server declare what they support during init. Server declares tools/resources/prompts capabilities; client declares sampling/elicitation/roots.

### Key facts (2026)
- Created by Anthropic, November 2024
- Donated to Linux Foundation AAIF (Dec 2025) alongside OpenAI's AGENTS.md
- 97M+ npm installs
- Adopted by OpenAI, Google, Microsoft, Block
- Spec revisions: 2024-11-05 -> 2025-03-26 -> 2025-06-18 -> 2025-11-25 -> draft

### MCP vs Function Calling vs A2A

|                  | MCP                        | Function Calling         | A2A (Google)            |
|------------------|----------------------------|--------------------------|--------------------------|
| **What**         | Protocol: LLM <-> tools    | LLM-native invocation    | Protocol: agent <-> agent|
| **Discovery**    | Dynamic (`tools/list`)     | Static (in prompt)       | Agent Cards              |
| **Statefulness** | Stateful sessions          | Stateless per call       | Task-based lifecycle     |
| **Transport**    | JSON-RPC 2.0               | Part of LLM API          | HTTP                     |
| **Scope**        | Cross-provider standard    | Provider-specific        | Cross-framework          |

**When to use each**: Function calling is how an LLM *decides* to use a tool. MCP is how the tool is *discovered, invoked, and managed*. A2A is how two autonomous agents delegate tasks to each other. They compose: an MCP server exposes tools that the LLM calls via function calling; A2A orchestrates multiple such agents.

---

## 2. The Three Primitives

### 2a. Tools (model-controlled functions)

**Intuition**: Tools are actions the LLM can take. The model sees the schema, decides when to call, and interprets results. Think: "search_documents", "send_email", "run_query".

**Protocol flow**:
```
Client -> Server:  tools/list          (discovery, paginated)
Server -> Client:  [{name, description, inputSchema, outputSchema, annotations}, ...]
Client -> Server:  tools/call          (invocation)
Server -> Client:  {content: [...], structuredContent: {...}, isError: bool}
Server -> Client:  notifications/tools/list_changed   (when tools change)
```

**Schema structure** (from spec):
```json
{
  "name": "search_docs",
  "title": "Document Search",
  "description": "Search documents by semantic query",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "Search query"},
      "limit": {"type": "integer", "description": "Max results"}
    },
    "required": ["query"]
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "results": {"type": "array"},
      "total": {"type": "integer"}
    },
    "required": ["results", "total"]
  },
  "annotations": {
    "readOnlyHint": true,
    "openWorldHint": false
  }
}
```

**Content types in results**: text, image (base64), audio (base64), resource_link (URI for later fetch), embedded_resource (inline content). New in 2025-06-18: `structuredContent` field alongside `content` for machine-readable JSON.

**Annotations** (hints for clients, not enforced):
- `readOnlyHint`: safe to auto-approve (ChatGPT skips confirmation for these)
- `destructiveHint`: cannot be undone (show warning)
- `idempotentHint`: repeated calls are safe
- `openWorldHint`: interacts with external systems
- `audience`: ["user"], ["assistant"], or both
- `priority`: 0.0-1.0 importance

**Error handling -- two distinct mechanisms**:
1. **Protocol errors**: JSON-RPC error codes (e.g., -32602 "Unknown tool"). Structural problems.
2. **Tool execution errors**: `isError: true` in result. Actionable feedback the LLM can use to retry with different params (e.g., "Invalid date: must be in the future").

**FastMCP tool with typed params and structured output**:
```python
from dataclasses import dataclass
from typing import Annotated
from pydantic import Field
from fastmcp import FastMCP, Context

mcp = FastMCP("DocSearchServer")

@dataclass
class SearchResult:
    title: str
    snippet: str
    score: float

@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": False})
async def search_documents(
    query: Annotated[str, "Semantic search query"],
    limit: Annotated[int, Field(description="Max results", ge=1, le=50)] = 10,
    ctx: Context = None
) -> list[SearchResult]:
    """Search the document database for relevant content."""
    await ctx.info(f"Searching for: {query}")
    await ctx.report_progress(progress=0, total=100)
    results = await db.search(query, limit=limit)
    await ctx.report_progress(progress=100, total=100)
    return [SearchResult(title=r.title, snippet=r.text[:200], score=r.score)
            for r in results]
```
FastMCP auto-generates `inputSchema` from type hints, `outputSchema` from the `list[SearchResult]` return type. The `Context` parameter is hidden from the LLM schema and injected at runtime.

### 2b. Resources (application-controlled data)

**Intuition**: Resources are data the *application* (not the model) decides to include as context. Think: files, DB schemas, config. The host chooses which resources to attach; the model reads but does not discover them autonomously.

**Static resources vs resource templates**:
- Static: `resource://docs/schema.sql` -- fixed URI, fixed content
- Template: `resource://docs/{doc_id}` -- parameterized URI (RFC 6570), content generated on demand

**Protocol flow**:
```
resources/list                -> list all static resources
resources/templates/list      -> list all templates
resources/read {uri}          -> get content (text or base64 binary)
resources/subscribe {uri}     -> watch for changes
notifications/resources/updated {uri}  -> content changed
notifications/resources/list_changed   -> new resources available
```

**FastMCP resource and template**:
```python
@mcp.resource("config://app/settings")
def app_settings() -> str:
    """Current application configuration."""
    return json.dumps(load_config())

@mcp.resource("docs://{doc_id}")
async def get_document(doc_id: str) -> str:
    """Retrieve a document by ID."""
    doc = await db.get(doc_id)
    return doc.content
```

### 2c. Prompts (user-controlled templates)

**Intuition**: Prompts are reusable interaction patterns the *user* selects. Think: "summarize this", "review this PR", "explain this error". They are slash-commands, not autonomous actions.

**Protocol**: `prompts/list` -> `prompts/get {name, arguments}` -> returns message array

```python
@mcp.prompt()
def code_review(language: str, code: str) -> str:
    """Generate a code review prompt."""
    return f"Review this {language} code for bugs, style, and performance:\n\n```{language}\n{code}\n```"
```

### The control spectrum

| Primitive  | Controlled by | Discovery     | Example                    |
|------------|--------------|---------------|----------------------------|
| **Tools**  | Model        | `tools/list`  | `search_docs(query)`       |
| **Resources** | Application | `resources/list` | `config://app/settings` |
| **Prompts** | User        | `prompts/list`| "Summarize this document"  |

**Why this matters**: The separation enforces least-privilege. The model cannot autonomously read any resource -- the application gates access. The model cannot trigger prompts -- the user chooses. Only tools are model-initiated, and even those go through human-in-the-loop confirmation.

---

## 3. Server-Initiated Capabilities

### Sampling (server requests LLM completion)

**Why**: Enables agentic recursion. A tool executing on the server can ask the client's LLM for help mid-execution -- no separate API key needed.

```
Server -> Client:  sampling/createMessage {messages, modelPreferences, tools, maxTokens}
Client -> Server:  {role: "assistant", content: ..., model: "...", stopReason: "endTurn"}
```

The server specifies `modelPreferences` with `intelligencePriority`, `speedPriority`, `costPriority` (0-1 each) plus optional `hints` (e.g., "claude-3-sonnet"). The client maps to available models. New in 2025-11-25: sampling supports tool use -- the server can provide tools the LLM can call within the sampling request, enabling multi-turn agentic loops.

### Elicitation (server requests user input)

Two modes:
- **Form mode**: Server sends a JSON Schema, client renders a form, user fills it in. For non-sensitive data only.
- **URL mode** (new in 2025-11-25): Server sends a URL, client opens browser. For sensitive data (API keys, OAuth flows, payments). Data never passes through the MCP client.

**Why URL mode matters for security**: Token passthrough is forbidden in MCP. If a server needs third-party credentials, it cannot ask the client to forward tokens. Instead, it uses URL mode elicitation to send the user directly to the third-party auth page, receives tokens server-side, and stores them bound to the user identity.

---

## 4. Building MCP Servers with FastMCP

### Minimal complete server
```python
from fastmcp import FastMCP

mcp = FastMCP("ProductionServer")

@mcp.tool(annotations={"readOnlyHint": True})
async def search(query: str, limit: int = 10) -> list[dict]:
    """Search the knowledge base."""
    return await db.search(query, limit)

@mcp.resource("config://schema")
def db_schema() -> str:
    """Database schema for context."""
    return open("schema.sql").read()

@mcp.prompt()
def analyze(topic: str) -> str:
    """Analyze a topic using our knowledge base."""
    return f"Analyze {topic} using the available search tools and schema context."
```

### Context object (runtime capabilities)
```python
@mcp.tool
async def process(data_uri: str, ctx: Context) -> dict:
    await ctx.info("Starting processing")           # logging
    resource = await ctx.read_resource(data_uri)     # read resources
    await ctx.report_progress(50, 100)               # progress
    summary = await ctx.sample(f"Summarize: {resource[0].content[:500]}")  # LLM sampling
    return {"summary": summary.text}
```

### Dependency injection (hiding params from LLM)
```python
from fastmcp.dependencies import Depends

def get_current_user() -> str:
    return "user_123"  # from auth context

@mcp.tool
def get_my_data(user_id: str = Depends(get_current_user)) -> dict:
    # user_id injected at runtime, not visible in tool schema
    return db.get_user_data(user_id)
```

### Server composition

| Method          | Behavior                      | When to use                    |
|-----------------|-------------------------------|--------------------------------|
| `import_server` | One-time copy (static)        | Finalized components, perf-critical |
| `mount`         | Live link (dynamic delegation)| Modular runtime, hot-reload    |

```python
# Gateway pattern: parent handles auth, children handle domain logic
gateway = FastMCP("Gateway")
gateway.add_middleware(AuthMiddleware())

docs_service = FastMCP("DocsService")
@docs_service.tool
def search(query: str) -> str:
    return f"Results for: {query}"

gateway.mount(docs_service, prefix="docs")  # tools appear as "docs_search"
```

Performance note: mounted HTTP servers add 300-400ms latency per list operation that affects the entire parent. Use `import_server` for performance-critical setups.

### Transport and deployment
```bash
# Local (stdio) -- for Claude Desktop, IDE integrations
python server.py

# Remote (HTTP/SSE) -- for production
uvicorn server:mcp --host 0.0.0.0 --port 8000
```

---

## 5. Middleware and Production Patterns

### Pipeline model
```
request -> ErrorHandling -> RateLimiting -> Caching -> Timing -> Logging -> handler
response <- ErrorHandling <- RateLimiting <- Caching <- Timing <- Logging <- handler
```

**Hook hierarchy** (general to specific):
1. `on_message` -- ALL messages
2. `on_request` / `on_notification` -- by message type
3. `on_call_tool`, `on_read_resource`, `on_get_prompt` -- operation-specific

### Production middleware stack
```python
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.middleware.timing import TimingMiddleware
from fastmcp.server.middleware.logging import LoggingMiddleware
from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware

mcp = FastMCP("Production")
mcp.add_middleware(ErrorHandlingMiddleware())
mcp.add_middleware(RateLimitingMiddleware(max_requests_per_second=50))
mcp.add_middleware(TimingMiddleware())
mcp.add_middleware(LoggingMiddleware(include_payloads=True))
```

### Custom auth middleware
```python
class AuthMiddleware(Middleware):
    async def on_call_tool(self, context: MiddlewareContext, call_next):
        tool_name = context.message.name
        if tool_name.startswith("admin_") and not is_admin(context):
            raise ToolError("Requires admin privileges")
        return await call_next(context)
```

---

## 6. Authorization and Security

### OAuth 2.1 flow (from spec)
```
1. Client requests protected resource -> 401 + WWW-Authenticate
2. Client discovers auth server via RFC 9728 (Protected Resource Metadata)
3. Client registers dynamically via RFC 7591 (if supported)
4. Authorization code flow + PKCE -> access token
5. All requests: Authorization: Bearer <token>
6. Server validates audience, scope, expiry
```

**Transport-specific auth**:
| Transport | Auth Method |
|-----------|-------------|
| stdio     | Environment variables (no OAuth) |
| HTTP/SSE  | OAuth 2.1 as above |

### Security landscape (2026 reality)
**Critical fact**: 30+ CVEs in 60 days. 92% exploit probability with 10+ plugins.

**Attack vectors**:
1. **Tool poisoning**: Malicious server injects harmful instructions in tool descriptions that manipulate the LLM
2. **Rug pull**: Server changes tool behavior after initial user approval
3. **Token passthrough**: Server forwards client tokens to third parties (confused deputy)
4. **PII exfiltration**: Tools read sensitive data and send to external services
5. **Prompt injection via tool results**: Tool output contains instructions that hijack the LLM

**Defense checklist**:
- Sandbox tool execution (Docker, subprocess isolation)
- Pin tool versions, do not auto-update
- Validate all inputs AND outputs
- Audit log every tool call (who, what, when, result)
- Rate limit per-client and per-tool
- Human-in-the-loop for sensitive operations (destructiveHint)
- RBAC: scope tool access by user role
- Never trust tool annotations from untrusted servers

**Spec security requirements** (MUST-level):
- Validate all tool inputs
- Implement access controls
- Rate limit invocations
- Sanitize outputs
- Use HTTPS for all auth endpoints
- Implement PKCE
- Validate token audience
- MUST NOT pass through tokens to downstream APIs

### OWASP MCP Top 10 (know these exist)
1. Tool Poisoning Attacks
2. Excessive Agency / Privilege Escalation
3. Server Spoofing / Rug Pulls
4. Token/Credential Theft
5. Resource Injection
6. Prompt Injection via Tools
7. Insufficient Input Validation
8. Inadequate Logging/Monitoring
9. Insecure Transport
10. Denial of Service via Resource Exhaustion

---

## 7. MCP Ecosystem (2026)

### Protocol governance
Created by Anthropic (Nov 2024), donated to Linux Foundation AAIF (Dec 2025). Open governance alongside OpenAI's AGENTS.md and Block's goose.

### Competing/complementary protocols
- **A2A** (Google): Agent-to-agent communication. Complements MCP -- MCP connects agents to tools, A2A connects agents to each other.
- **AGENTS.md** (OpenAI): Static capability declaration file. Simpler than MCP, no dynamic discovery.
- **Function calling** (per-provider): The mechanism LLMs use to invoke tools. MCP wraps this with discovery, lifecycle, and cross-provider support.

### Notable MCP servers in the wild
GitHub (repo management), Slack (messaging), PostgreSQL (database queries), filesystem (local file access), Playwright (browser automation), TechDocs (documentation search -- used daily in Claude Code).

### FastAPI integration pattern
FastMCP can generate an MCP server FROM an existing FastAPI app, or mount an MCP server INTO FastAPI:
```python
# Generate MCP server from existing FastAPI routes
from fastmcp.integrations.fastapi import FastApiMCP
mcp = FastApiMCP(fastapi_app)  # auto-discovers routes as tools

# Or mount MCP into FastAPI
from fastmcp import FastMCP
mcp = FastMCP("MyServer")
fastapi_app.mount("/mcp", mcp.get_asgi_app())
```
**Why this matters**: If you already have a FastAPI service, you can expose it as MCP tools with minimal code. This is the fastest path from existing API to MCP server.

### HN practitioner insights (2026)
Key themes from real-world MCP discussions:
- **Value depends on adoption**: "If I need to write an MCP adapter for everything, the value is little. If API owners put in the work to have MCP-compatible interfaces, it is valuable." (Currently at critical mass with major vendors adopting.)
- **stdio is the killer feature**: MCP's decision not to just use HTTP APIs enables open-source, locally-running tools without server setup. This is why Claude Desktop adoption exploded.
- **The real challenge is tool descriptions**: Most production issues stem from poorly described tools, not protocol problems. Tool descriptions are essentially prompts -- they need the same care.

---

## 8. Interview Q&A

### Q1: "What is MCP? Why does it matter?"
**What they're testing**: Fundamentals + ability to explain clearly.

**Strong answer**: "MCP is an open protocol, originally from Anthropic, now under the Linux Foundation, that standardizes how LLMs integrate with external tools and data. It uses JSON-RPC 2.0 with three primitives: Tools the model calls, Resources the application provides as context, and Prompts the user selects. Before MCP, every tool integration was custom per provider -- N tools times M providers meant N*M adapters. MCP collapses that to N+M. It is the LSP of AI: like how LSP let any IDE work with any language server, MCP lets any AI app work with any tool server. With 97M+ installs and adoption by OpenAI, Google, and Microsoft, it is the de facto standard."

**Follow-ups**: How does capability negotiation work? (Init handshake.) What happens if a server adds tools mid-session? (listChanged notification, client re-fetches.) How does MCP handle backward compatibility across spec versions? (Protocol version in init, servers support multiple.)

**Your angle**: "I use MCP tools daily in Claude Code -- TechDocs for documentation search, Playwright for browser automation. This gives me firsthand experience with both the developer ergonomics and the failure modes."

### Q2: "Explain Tools vs Resources vs Prompts"
**What they're testing**: Protocol depth, understanding of control boundaries.

**Strong answer**: "They represent three control levels. Tools are model-controlled -- the LLM decides when to call `search_docs(query)`. Resources are application-controlled -- the host decides which data to include as context, like `config://schema`. Prompts are user-controlled -- the user picks a template like 'review this PR'. This separation enforces least-privilege: the model cannot autonomously access any resource, and the user controls interaction patterns. In practice, write operations are tools, read-only data is resources, and reusable workflows are prompts."

**Follow-up**: Why not just make everything a tool? (Resources are passive and cheaper -- no execution, no confirmation dialog. They are also cacheable and subscribable.)

### Q3: "Design an MCP server for a document management system"
**What they're testing**: System design with MCP primitives.

**Strong answer**: "Three layers. Tools: `search_documents(query, filters)` with readOnlyHint, `create_document(title, content)` without, `delete_document(doc_id)` with destructiveHint. Resources: `docs://{doc_id}` as a template for document content, `config://schema` as static resource for the DB schema. Prompts: `summarize_document` and `compare_documents` as user-selectable templates. Middleware stack: ErrorHandling, RateLimiting (50 req/s), AuthMiddleware checking user roles against tool annotations, TimingMiddleware for observability. Composition: mount a shared `AuthServer` as gateway, domain servers behind prefixes. Transport: HTTP/SSE for production with OAuth 2.1."

**Follow-up**: How do you handle versioning? (Mount v1 and v2 servers with different prefixes. Use `listChanged` to notify clients. Version the entire deployment, not individual tools.)

### Q4: "How do you secure an MCP server?"
**What they're testing**: Production awareness, not just protocol knowledge.

**Strong answer**: "Start with the threat model: tool poisoning, rug pulls, token passthrough, PII exfiltration. Defense in depth: (1) Sandbox execution in containers. (2) Pin tool versions -- never auto-update from untrusted registries. (3) Input validation via JSON Schema plus output validation. (4) Audit logging of every tool call with caller identity, arguments, results, and timing. (5) Rate limiting per client and per tool. (6) Human-in-the-loop for destructive operations -- use annotations to signal this. (7) OAuth 2.1 with PKCE for remote servers, MUST validate token audience to prevent confused deputy. (8) Never trust tool annotations from untrusted servers -- they are hints, not security boundaries. The 2026 reality is 30+ CVEs in 60 days, so security is not optional."

**Follow-up**: How does MCP prevent token passthrough? (Spec explicitly forbids it. URL mode elicitation exists specifically so servers can get third-party credentials without seeing the client's tokens.)

### Q5: "MCP vs function calling -- when do you use each?"
**What they're testing**: Architectural judgment.

**Strong answer**: "They are complementary, not competing. Function calling is the mechanism: the LLM decides to invoke a tool based on its schema. MCP is the infrastructure: discovery, lifecycle management, cross-provider standardization, middleware, auth. If you have a single LLM calling 3 hardcoded tools, function calling alone is fine. When you need dynamic tool discovery, multiple providers, production middleware (auth, rate limiting, logging), or tool servers maintained by different teams, you need MCP. An MCP server exposes tools that the LLM ultimately invokes via function calling."

### Q6: "How does MCP authorization work?"
**What they're testing**: Security depth, OAuth knowledge.

**Strong answer**: "MCP uses OAuth 2.1. The flow: client requests a protected resource, gets 401 with WWW-Authenticate. Client discovers the authorization server via RFC 9728, optionally registers dynamically via RFC 7591, then does an authorization code flow with PKCE. All requests include the Bearer token. The server validates audience, scope, and expiry. Critical: servers MUST NOT pass through tokens to downstream APIs. For third-party auth, the server uses URL mode elicitation -- sends the user directly to the third-party auth page via the browser, receives tokens server-side, stores them bound to user identity. This prevents the confused deputy problem."

### Q7: "What is sampling in MCP? Why would a server need it?"
**What they're testing**: Understanding of agentic patterns.

**Strong answer**: "Sampling lets an MCP server request an LLM completion from the client during tool execution. The server does not need its own API key -- it borrows the client's LLM. This enables agentic recursion: a tool can ask the LLM to analyze intermediate results, decide next steps, or even use other tools within the sampling request. The server specifies model preferences (intelligence vs speed vs cost tradeoffs) and hints, but the client makes the final model selection. Human-in-the-loop applies: the client should show the sampling request for user approval."

**Follow-up**: What about security? (Iteration limits for tool loops, user approval at each step, rate limiting.)

### Q8: "How would you observe/monitor MCP servers in production?"
**What they're testing**: Production readiness, OTEL knowledge.

**Strong answer**: "Three layers. (1) Middleware: TimingMiddleware for latency per tool call, LoggingMiddleware with payloads for debugging. (2) Structured logging: every tool call logged with request_id, client_id, tool_name, duration, success/error. (3) OTEL integration: create spans for each tool invocation following the GenAI Agent Spans semantic convention. Trace the full chain: user request -> LLM decision -> tool call -> external API -> response. Alert on error rate spikes, latency P99, and rate limit hits. The `meta` field in ToolResult can carry runtime metadata like execution_time_ms and model_version."

### Q9: "Walk me through the lifecycle of an MCP session"
**What they're testing**: Protocol fluency.

**Strong answer**: "Four phases. (1) **Initialize**: Client sends `initialize` with its capabilities (sampling, elicitation, roots) and protocol version. Server responds with its capabilities (tools, resources, prompts) and supported features. Client confirms with `initialized` notification. (2) **Discover**: Client calls `tools/list`, `resources/list`, `prompts/list` to learn what is available. (3) **Operate**: Client sends `tools/call`, `resources/read`, `prompts/get` as needed. Server may send `sampling/createMessage` or `elicitation/create` back to the client. Either side can send notifications (list_changed, resource_updated). (4) **Shutdown**: Clean disconnection. The session is stateful throughout -- both sides maintain context."

### Q10: "You use MCP daily in Claude Code. What have you learned?"
**What they're testing**: Real experience vs textbook knowledge.

**Your angle**: "Three things. First, tool descriptions are the most important design decision -- they are the LLM's only interface to your tool. Vague descriptions lead to wrong tool selection; overly detailed ones waste context. Second, the stdio transport is fast and reliable for local tools, but HTTP-mounted servers introduce real latency that affects the entire host. Third, structured output (outputSchema) is a game-changer for tool chains -- when one tool's output feeds another tool's input, having a validated schema prevents silent data corruption. My background building DLBacktrace across transformer architectures gives me an intuition for how LLMs process tool schemas -- they attend to parameter names and descriptions the same way they attend to natural language."
