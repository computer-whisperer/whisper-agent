# The Headless Agent Loop

Whisper-agent's organizing principle: **the agent loop runs on a central server, separate from the hosts it acts on.** Every operation that touches the outside world — a filesystem, a process, a service — happens through a remote tool boundary, not directly inside the loop process.

Companion docs: [`design_task_scheduler.md`](design_task_scheduler.md) describes the internal structure of the loop runtime (tasks-as-data + central scheduler); [`design_permissions.md`](design_permissions.md) covers the three patterns at the tool boundary.

## Why this separation

A modern agentic CLI (Claude Code, Codex, Gemini) colocates the LLM client, the conversation harness, and host access into one process running on the user's machine. That works for the "human at a terminal" use case those tools target, but doesn't extend cleanly to:

- **Multiple hosts in one conversation.** One agent loop interacting with host A's filesystem and host B's processes simultaneously, in a single context.
- **Long-running, event-driven loops.** A "Jarvis"-style agent reacting to webhooks, k8s events, file changes, or scheduled triggers — none of which require a human in a terminal.
- **Heterogeneous LLM backends per loop.** Same loop architecture targeting Anthropic, OpenAI, Gemini, whisper-tensor, or local llama.cpp servers without rewriting the host integration.
- **Centralized observability and audit.** Every tool call traceable and policied at one point.

Moving the loop to a server and putting the host-access boundary at the network gives all four for free.

## The three layers

```
┌──────────────────────────────────────────────────────────────────┐
│  Loop runtime (server)                                           │
│  - Tasks-as-data + central scheduler (design_task_scheduler.md)  │
│  - Per-task conversation state, scratchpad/memory                │
│  - Provider client (Anthropic; OpenAI/Gemini/w-t come later)     │
│  - Context assembly, prompt-cache breakpoint placement           │
│  - System-reminder injection, compaction, sub-agent spawning     │
│  - Tool dispatch via direct MCP client (see "Commitment to MCP") │
│  - Trigger sources: human chat, cron, webhooks, events           │
└────────────────────────────┬─────────────────────────────────────┘
                             │ tool calls / tool results
┌────────────────────────────┴─────────────────────────────────────┐
│  Tool boundary (network protocol — initially MCP)                │
│  - Authn (which loop session may talk to which host)             │
│  - Authz (which tools may this session call on this host)        │
│  - Transport: streamable HTTP / WebSocket / SSH-tunneled stdio   │
└────────────────────────────┬─────────────────────────────────────┘
                             │ over network
┌────────────────────────────┴─────────────────────────────────────┐
│  Tool servers (per managed endpoint)                             │
│  - POSIX hosts: exec, read/write/edit_file, grep, glob, ...      │
│  - K8s clusters: kubectl-shaped tool surface                     │
│  - Obsidian vault: note read/write/search                        │
│  - whisper-tensor server: SuperGraph commissioning               │
└──────────────────────────────────────────────────────────────────┘
```

## What stays in the loop (not at the boundary)

These concerns don't fit a tool-call request/response shape and must live in the loop process:

- **Message-stream editing**: system-reminders, file-history snapshots, deferred-tool announcements — the things Claude Code injects between turns (visible in `docs/research/captures/transcripts/claude.json`).
- **Prompt caching strategy**: where to place `cache_control` breakpoints, when to invalidate them.
- **Context compaction**: what to summarize, what to drop.
- **Provider-specific request shaping**: messages-array vs. WebSocket events vs. Gemini's `request` wrapper. Per-provider transports converge to a uniform internal event stream the loop manipulates.
- **Sub-agent spawning**: full agent loops with their own context windows and tool sets, not single LLM completions.
- **Conversation-level state**: history, scratchpad, working memory. The loop owns its internals; the boundary is for outside-world access only.

## Commitment to MCP (for now)

We deliberately bound the loop directly to our own MCP client rather than introduce an internal trait abstraction that MCP is one implementation of. The reason: we don't yet know what shape a "tool transport" abstraction should actually take, and guessing from first principles leads to bad abstractions. We learn that shape by seeing where MCP specifically pinches.

So: the loop's tool-call sites reference MCP types directly. When any of the gaps below becomes load-bearing, we introduce the abstraction at that point with real requirements, not hypothetical ones. **Plan for this binding to be ripped out and rebuilt.**

Known MCP gaps we'll need to revisit (see [`research/tool_protocols.md`](research/tool_protocols.md)):

- Grammar-constrained tool inputs (like Codex's `apply_patch`) don't fit JSON Schema — flattening them to strings loses the grammar constraint the model was RL-trained against.
- Streaming / partial tool output is second-class (`notifications/progress` exists but is primitive).
- Sub-agent spawning maps to MCP `sampling`, but not as a full agent loop.
- Permission elicitation, cost accounting, and per-call observability aren't first-class.

When we hit any of these with a real need, we'll either extend MCP, introduce a complementary transport for the specific case, or build the trait we postponed — with the concrete requirements in hand.

## The host tool server

Probably an opinionated Rust binary we write, deployed per managed POSIX host. Exposes the *slice* of POSIX surface we actually want agents touching:

- `exec` (with timeout, env, cwd, sandbox policy)
- `read_file`, `write_file`, `edit_file` (line-oriented, formatted similarly to Claude Code's tools)
- `grep`, `glob`
- Host-specific extensions where applicable: `docker`/`podman`, `kubectl`, `systemctl`, package management

Sandboxing primitives — capabilities, mount namespaces, seccomp filters — live in this binary. The loop trusts that the host server enforces what it advertises; it doesn't try to sandbox at the loop layer.

## Open questions (deferred — not deciding now)

1. **Transport**: streamable HTTP vs. WebSocket vs. SSH-tunneled stdio. Each has tradeoffs for latency, multiplexing, firewall friendliness, mutual auth, observability.
2. **Authn/authz model**: per-loop credentials, per-host policy, audit storage. MCP doesn't standardize this.
3. **Use existing MCP servers** (filesystem, shell, git) from the ecosystem, or build our own?
4. **Streaming / long-running operations**: how to surface `tail -f`, file watches, build-progress to the model.
5. **Sub-agent spawning across hosts**: when a sub-agent needs different host access than its parent.
6. **Tool latency mitigation**: batched/composite tools to amortize network round-trips.

## What this design is NOT

- **Not a commitment to MCP forever.** MCP is the first-cut transport. If it stops fitting, we change — with concrete requirements learned from where it pinched, not speculatively.
- **Not finalized.** This is a starting frame, not a spec. Expect it to evolve as prototypes hit reality.
