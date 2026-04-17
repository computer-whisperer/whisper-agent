# The Headless Agent Loop

Whisper-agent's organizing principle: **the agent loop runs on a central server, separate from the hosts it acts on.** Every operation that touches the outside world — a filesystem, a process, a service — happens through a remote tool boundary, not directly inside the loop process.

Companion docs: [`design_pod_thread_scheduler.md`](design_pod_thread_scheduler.md) describes how the loop is structured internally (pods + threads + resources, tasks-as-data, central scheduler). [`design_permissions.md`](design_permissions.md) covers the three patterns at the tool boundary and which of them are implemented today.

## Why this separation

A modern agentic CLI (Claude Code, Codex, Gemini) colocates the LLM client, the conversation harness, and host access into one process running on the user's machine. That works for the "human at a terminal" use case those tools target but doesn't extend cleanly to:

- **Multiple hosts in one conversation.** One agent loop interacting with host A's filesystem and host B's processes simultaneously, in a single context.
- **Long-running, event-driven loops.** An agent reacting to webhooks, scheduled triggers, or external event sources — none of which require a human in a terminal. See [`design_behaviors.md`](design_behaviors.md) for how this is captured as pod behaviors.
- **Heterogeneous LLM backends per loop.** Same loop architecture targeting Anthropic, OpenAI, Gemini, or local endpoints without rewriting host integration.
- **Centralized observability and audit.** Every tool call traceable and policied at one point.

Moving the loop to a server and putting the host-access boundary at the network gives all four.

## The three layers

```
┌──────────────────────────────────────────────────────────────────┐
│  Loop runtime (the whisper-agent server process)                 │
│  - Pods, threads, resources (design_pod_thread_scheduler.md)     │
│  - Tasks-as-data state machine, driven by a central scheduler    │
│  - Per-thread conversation state                                 │
│  - Provider clients (Anthropic, OpenAI Chat + Responses, Gemini, │
│    local openai-compat)                                          │
│  - Context assembly, prompt-cache breakpoint placement           │
│  - Tool dispatch via direct MCP client                           │
│  - Trigger sources for autonomous behaviors (design_behaviors.md)│
└────────────────────────────┬─────────────────────────────────────┘
                             │ tool calls / tool results
┌────────────────────────────┴─────────────────────────────────────┐
│  Tool boundary (MCP over streamable HTTP)                        │
│  - Authn/authz: deferred (see design_permissions.md Pattern 3)   │
│  - Transport: streamable HTTP today; loopback only               │
└────────────────────────────┬─────────────────────────────────────┘
                             │ over network
┌────────────────────────────┴─────────────────────────────────────┐
│  Tool servers (MCP hosts, per managed endpoint)                  │
│  - whisper-agent-mcp-host: POSIX tools (read/write_file, bash)   │
│    provisioned per-thread inside a host-env sandbox              │
│  - whisper-agent-mcp-fetch: shared HTTP fetch                    │
│  - whisper-agent-mcp-search: shared web search                   │
└──────────────────────────────────────────────────────────────────┘
```

## What stays in the loop (not at the boundary)

These concerns don't fit a tool-call request/response shape and live in the loop process:

- **Message-stream editing**: system-reminders, file-history snapshots, deferred-tool announcements — the in-band metadata a harness injects between turns.
- **Prompt caching strategy**: where to place `cache_control` breakpoints, when to invalidate them. See [cache-control architecture memory](../) — breakpoints live on `ModelRequest` as an explicit list; the scheduler picks the policy, backend adapters translate.
- **Provider-specific request shaping**: per-provider transports converge to the canonical `Conversation` + `ContentBlock` shape the loop manipulates, then serialize out to the provider's wire format.
- **Conversation-level state**: history, tool-call results, allowlist, usage accounting. The loop owns its internals; the boundary is for outside-world access only.

Future items that will also live in the loop: context compaction, sub-agent spawning (see `design_pod_thread_scheduler.md` "not in scope" for the shape), autonomous behavior triggers (see `design_behaviors.md`).

## Commitment to MCP

The loop is bound directly to our own MCP client (`src/tools/mcp.rs`) rather than to an abstract "tool transport" trait that MCP happens to implement. The reason: we don't yet know what shape a tool-transport abstraction should take, and guessing from first principles produces bad abstractions. We learn the shape by seeing where MCP pinches.

So: the loop's tool-call sites reference MCP types directly. **Expect this binding to be ripped out and rebuilt** when the first concrete requirement surfaces.

Known MCP gaps (see [`research/tool_protocols.md`](research/tool_protocols.md)):

- Grammar-constrained tool inputs (e.g. Codex's `apply_patch`) don't fit JSON Schema — flattening to strings loses the grammar constraint the model was trained against.
- Streaming / partial tool output is second-class; `notifications/progress` exists but is primitive and requires bidirectional transport (SSE/WebSocket), whereas our client speaks plain HTTP POST.
- Sub-agent spawning maps to MCP `sampling` but not as a full agent loop.
- Permission elicitation, cost accounting, and per-call observability aren't first-class.
- Server-initiated events for autonomous triggers don't fit (see `design_behaviors.md` for why we don't lean on MCP notifications for that).

When any of these becomes load-bearing, we'll either extend MCP, add a complementary transport for the specific case, or build the trait we deferred — with concrete requirements in hand.

## The host tool server

`whisper-agent-mcp-host` is the Rust binary that exposes POSIX tools over MCP. It's provisioned per-thread inside a host-env sandbox: the scheduler asks a `HostEnvProvider` (today: `local-landlock`) to set up the sandboxed context, then spawns the MCP host inside that context with tool access scoped to the allowed paths and network policy. See [sandbox architecture memory](../) for the full shape — `SandboxSpec` on each task, `SandboxBackend` trait, provisioning below the MCP layer.

The host binary's tool surface:

- `read_file`, `write_file`, `edit_file` (line-oriented)
- `bash` (exec with timeout, cwd, inside the sandbox)
- `grep`, `glob` (search helpers)

Shared (non-host) MCP servers that ride the same transport:

- `whisper-agent-mcp-fetch`: HTTP fetch, shared across pods, no host context.
- `whisper-agent-mcp-search`: web search, likewise shared.

Shared MCP hosts are configured server-side (`shared_mcp_hosts` table in the server's TOML); pods opt into them via `[allow].mcp_hosts`.

## Open questions

Still unresolved:

1. **Transport for remote hosts.** Today the tool boundary is loopback only. Streamable HTTP vs. WebSocket vs. SSH-tunneled stdio for the real remote case has tradeoffs for latency, multiplexing, firewall friendliness, and mutual auth.
2. **Authn/authz model.** Per-principal credentials, per-host policy, audit storage for remote deployments. MCP doesn't standardize this; see `design_permissions.md` Pattern 3.
3. **Streaming / long-running operations.** `tail -f`, file watches, build-progress — how these surface to the model.
4. **Sub-agent spawning across hosts.** When a sub-agent needs different host access than its parent.
5. **Tool latency mitigation.** Batched/composite tools to amortize network round-trips.

Decided since the original framing:

- Sandboxing is not in the loop. It's in the host-env provisioning layer below MCP (landlock today; bubblewrap/podman/bare-metal planned). The loop trusts the sandbox to enforce what it advertises.
- Provider abstraction is concrete: one `ModelProvider` trait, five backend adapters (anthropic, openai_chat, openai_responses, gemini, and gemini_auth / codex_auth variants).
- The tool boundary currently speaks MCP streamable HTTP 2025-06-18; we wrote our own server (`whisper-agent-mcp-host`) rather than adopting one from the ecosystem, because we need the host-env provisioning contract to be ours.

## What this design is NOT

- **Not a commitment to MCP forever.** MCP is the first-cut transport. If it stops fitting, we change — with concrete requirements learned from where it pinched, not speculatively.
- **Not finished.** Sandboxing is in; remote hosts, remote auth, and cross-host sub-agents aren't. Expect the architecture to evolve as those land.
