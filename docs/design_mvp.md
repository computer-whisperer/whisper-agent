# MVP — pathfinder build

The smallest implementation that exercises whisper-agent's load-bearing architecture end-to-end. Not production. Designed to be ripped out and rewritten as we learn what the shape actually wants to be.

## What the MVP proves

- The headless-loop separation (`docs/design_headless_loop.md`) works in practice — loop runtime in one process, host actions in another, no shortcuts.
- An MCP-mediated tool boundary is workable as the *only* path for host actions, with no internal abstraction layer between the loop and MCP. Plan to revisit this if it pinches.
- The Anthropic-shaped canonical conversation state generalizes (or doesn't) to non-Anthropic providers later.
- The whisper-tensor UI stack (Axum + egui+WASM + WebSocket+CBOR) ports cleanly from graph-explorer to chat-style UI.

## What the MVP does NOT prove

- Multi-host orchestration (one host, local).
- Provider abstraction (Anthropic only, hard-coded).
- Real authn/authz (loopback, full-trust).
- Permission UI — auto-approve with audit logging only; see `docs/design_permissions.md` for the intended end state.
- Sandboxing (out-of-process isolation only; the MCP host runs as the user, no namespacing/seccomp).
- Persistence (in-memory state, JSON dump on shutdown).
- Scheduling, sub-agents, event-driven loops.

## Architecture

Three Rust crates, two running processes plus the browser.

```
┌──────────────────────────────┐
│  Anthropic Messages API      │
└──────────────┬───────────────┘
               │ HTTPS + SSE
               │
┌──────────────┴──────────────┐         ┌────────────────────────────┐
│  whisper-agent              │         │  whisper-agent-mcp-host    │
│  (loop runtime + server)    │  MCP    │  (tool server, separate    │
│                             │ stream- │   process)                 │
│  - Anthropic API client     │ able-   │                            │
│  - MCP HTTP client          │ HTTP    │  Tools:                    │
│  - Loop, conversation state │ ◄──────►│  - read_file               │
│  - Audit log writer         │         │  - write_file              │
│  - Axum WebSocket endpoint  │         │  - bash                    │
└──────────────┬──────────────┘         └────────────────────────────┘
               │
               │ WebSocket + CBOR
               │
┌──────────────┴──────────────┐
│  whisper-agent-webui        │
│  (browser, egui + WASM)     │
│                             │
│  - Chat input               │
│  - Streamed assistant text  │
│  - Tool-call panels         │
└─────────────────────────────┘
```

## Crate layout

- **`whisper-agent`** — core + server combined. Contains the loop runtime, conversation state, Anthropic client, MCP HTTP client, Axum host with WebSocket endpoint, audit log writer, CLI entry point. Split into separate crates later only when there's a reason; one crate is fine for now.
- **`whisper-agent-webui`** — WASM + egui chat interface. WebSocket+CBOR client to talk to the server. Mirror of whisper-tensor's UI pattern.
- **`whisper-agent-mcp-host`** — standalone Rust binary that runs on the host. MCP server over streamable-HTTP. Implements `read_file`, `write_file`, `bash`. A trusted *slice* of POSIX surface, not a generic shell wrapper.

Workspace `Cargo.toml` at root.

## Tool API shape

Two layers, both mirroring the MCP protocol since we're locking the loop directly to MCP. Modeled as **stateful streaming connections** from day one — easier to get right now than to retrofit later.

**Session layer** — one per MCP host, long-lived:

- Holds the streamable-HTTP connection.
- Caches negotiated capabilities, current tool list, roots.
- Provides a server→client channel for unsolicited messages: `notifications/tools/list_changed`, `elicitation/create`, `sampling/createMessage`, progress notifications. Harness polls this channel continuously regardless of any active call.

**Invocation layer** — per tool call within a session:

- Returns a stream of events.
- Stream lifetime is bounded by the call.

Starting trait shape (placeholder, expect evolution):

```rust
trait McpSession {
    async fn list_tools(&mut self) -> Result<Vec<ToolDescriptor>, McpError>;

    async fn invoke(&mut self, name: &str, args: Value)
        -> Result<impl Stream<Item = ToolEvent>, McpError>;

    fn server_events(&mut self) -> impl Stream<Item = ServerInitiated>;
}

enum ToolEvent {
    Progress(ProgressNotification),
    OutputChunk(Bytes),       // streaming output — deferred for MVP
    Completed(ToolResult),
    Failed(ToolError),
}

enum ServerInitiated {
    Elicitation(ElicitationRequest),
    Sampling(SamplingRequest),
    ToolListChanged,
    // ...
}
```

**Critical nuance**: even with streaming events flowing internally, the model still sees exactly **one `tool_result` per `tool_use`** at the LLM API layer. That's how every provider tool protocol works. Streaming events exist for the UI, the audit log, and harness-internal handling (elicitation pause/resume). The harness collects events, forwards them to the UI, and at completion synthesizes the single `tool_result` for the model. **Do not leak per-event chatter into the model's context.**

**Elicitation flow** — auto-approve doesn't need it for MVP, but the channel must exist:

1. Server sends `elicitation/create` on the session's incoming-events channel.
2. Harness pauses any in-flight tool calls *for that session*.
3. Harness presents the prompt to the user. In MVP: log and auto-respond with sensible defaults.
4. Harness sends the response back.
5. Server proceeds; the affected tool call completes normally.

## Conversation state

Internal canonical form modeled after Anthropic's content-block shape:

```rust
struct Message {
    role: Role,                      // User | Assistant | (System lives outside the array)
    content: Vec<ContentBlock>,
}

enum ContentBlock {
    Text { text: String },
    ToolUse { id: String, name: String, input: Value },
    ToolResult { tool_use_id: String, content: ToolResultContent, is_error: bool },
    Thinking { signature: Option<String>, content: String },
    // image/document blocks deferred
}
```

Reasoning: Anthropic's shape is the most expressive of the three majors (explicitly named block types, mixed content per message, first-class thinking). Adapting it down to OpenAI/Gemini later is mechanical; adapting *up* from a flatter representation requires reorganizing state.

For MVP: hard-coded Anthropic adapter only. Provider serializers become per-provider modules when we add others.

## Wire protocols

**UI ↔ server**: WebSocket + CBOR. Mirror of whisper-tensor's pattern. Event sketch:

- Client→server: `UserMessage(text)`, `Cancel`, `ApproveElicitation(...)` (later).
- Server→client: `MessageBegin`, `TextDelta(chunk)`, `ToolCallBegin(name, args_preview)`, `ToolEvent(...)`, `ToolCallComplete(result_preview)`, `MessageEnd`, `ElicitationPrompt(...)` (later), `Status(...)`.

**Server ↔ Anthropic API**: standard `https://api.anthropic.com/v1/messages`, SSE streaming response.

**Server ↔ MCP host**: MCP **streamable-HTTP** (2025-06-18 spec). POST for client→server JSON-RPC; the same HTTP connection holds an SSE stream for server→client notifications. Loopback-only, no auth, no TLS for MVP.

## Permission posture (MVP stub)

- Auto-approve every tool call.
- Log every approval to an append-only audit file at `<state-dir>/audit.jsonl`, one line per call:
  `{timestamp, session_id, host_id, tool_name, args (truncated to 4KB), decision: "auto-approve", who_decided: "stub"}`.
- Audit machinery exists from day one, so adding real prompts in v0.2 is purely a UX layer addition; the policy/log path doesn't need refactoring.

Full intended model: `docs/design_permissions.md`.

## MVP demo scenario

Same as our research smoke-test: prompt the agent to "Create a minimal snake game in Rust using crossterm" in a clean working directory. Watch it use `bash` (cargo init, cargo add, cargo check), `write_file` and `read_file` (write `src/main.rs`, possibly re-read to iterate), all streamed to the webui.

Success criterion: same end-state as the claude-code capture (`docs/research/captures/transcripts/claude.json`), with full visibility into every API call and tool invocation in the audit log. We can then directly compare token usage, turn count, and tool sequences against the captures.

## Default model

`claude-sonnet-4-6` (or current Sonnet ID at start of work). The smarter model is less likely to do destructive things by accident while sandboxing is still primitive. Drop to Haiku for cost once we trust the harness.

## Known shortcuts (will be replaced)

- One MCP host, hard-coded by URL in a config file.
- One provider, hard-coded in code.
- `bash` returns final stdout/stderr only — no live progress streaming for MVP. The streaming infrastructure exists in the trait; the bash tool just doesn't emit `OutputChunk` events yet.
- Loopback HTTP, no auth, no TLS between webui/server/mcp-host.
- In-memory conversation state; JSON dump on shutdown for inspection only.
- No request retry / backoff (will hit rate limits in long sessions).
- No prompt-cache breakpoints — let Anthropic auto-cache or pay full input tokens; optimize later.

## Sequencing the work

1. Bootstrap workspace + the three crates with placeholder `lib.rs` / `main.rs`.
2. **`whisper-agent-mcp-host`** first — MCP server speaking streamable-HTTP with the three tools. Verify with `curl` / `mitmproxy` before any client exists.
3. **`whisper-agent`** loop runtime — Anthropic client, MCP client, basic loop driven by a hard-coded prompt (no UI yet).
4. End-to-end smoke test, CLI-only: snake prompt runs to completion, audit log written, conversation state dumped to JSON.
5. **`whisper-agent-webui`** — basic chat UI scaffold.
6. Wire UI ↔ server; the snake-prompt runs visible in the browser.

Each step is a natural commit point.
