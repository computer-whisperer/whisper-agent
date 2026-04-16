# Sessions, Threads, and Decoupled Resources

Architectural pivot following `design_task_scheduler.md`. The first scheduler design fused three concerns into the `Task` type: it was the scheduler's unit of work, the user's unit of organization, *and* the owner of every external resource the loop touched (MCP session, sandbox, model backend). That fusion was wrong, and it's now blocking the next round of features.

This document replaces `Task` with three layered concepts:

- **Session** — the user-facing umbrella; holds threads and metadata.
- **Thread** — the scheduler's unit; one conversation, one state machine.
- **Resources** (Sandbox, McpHost, Backend) — first-class entities with independent lifecycles, referenced by threads via a binding table.

`design_task_scheduler.md` is superseded in full. The wire protocol, persistence layout, scheduler internals, and webui state model all change.

## Why now

Five upcoming features all want shapes the current `Task` can't express:

| Feature | What it needs |
|---|---|
| `/compact` | Replace a thread's conversation with a summary while preserving the original as history. |
| Subagent | A thread that spawns a child thread with a different system prompt and (optionally) restricted tools, then resumes when the child terminates. |
| Rewind & fork | Branch a thread from a chosen midpoint; both branches live alongside each other. |
| Non-interactive tasks | Long-running work that may spin up helper threads without cluttering the UI's task list. |
| Customized agentic loops (Lua) | Threads whose `step()` is parameterized or fully scripted. *(Design-enabling only — not implemented in this refactor.)* |

All five want a *thread*-shaped abstraction nested inside a *session*-shaped one.

Independently, the resource-ownership story has been wrong from day one:

- Sandboxes are provisioned on first MCP connect and torn down at task terminal state. Two sessions working in the same project directory can't share a sandbox; a parallel fork can't reuse its parent's running container.
- MCP sessions are owned per-task. The "shared MCP host" feature added later carved out a special case for singletons, but it's grafted onto a model that fundamentally assumes 1:1.
- A thread can't swap its sandbox mid-conversation, can't have multiple sandboxes, can't switch backends to a cheaper model after a long planning phase.

The scheduler itself is *already* concurrency-friendly — `FuturesUnordered` carries any number of in-flight ops across any number of tasks, `McpSession` is `Arc`-shared and safe under concurrent JSON-RPC. The constraint is in the data model, not the runtime.

## The shift

```
Before:                            After:
  Task                               Session
   ├── conversation                   ├── threads: Vec<ThreadId>
   ├── state machine                  └── metadata
   ├── owns McpSession              Thread
   ├── owns SandboxHandle             ├── conversation
   └── refs Backend                   ├── state machine
                                      └── bindings: { sandbox?, mcp_hosts[], backend, tool_filter? }
                                    Sandbox  ───┐
                                    McpHost  ───┼── independent lifecycles, refcounted
                                    Backend  ───┘
```

Three independent registries (sandboxes, MCP hosts, backends) sit alongside the session/thread tables. Threads carry **reference IDs**, not owned handles. Auto-provisioning ("if no sandbox exists matching this spec, create one") is a *policy layer* on top of the bare data model, not baked into it.

## Data model

```rust
struct Scheduler {
    sessions:  HashMap<SessionId, Session>,
    threads:   HashMap<ThreadId, Thread>,         // flat — sessions store ThreadIds, not Threads
    sandboxes: HashMap<SandboxId, SandboxEntry>,
    mcp_hosts: HashMap<McpHostId, McpHostEntry>,
    backends:  HashMap<BackendId,  BackendEntry>, // promoted from config
    pending_io: FuturesUnordered<IoFuture>,
    router:     ThreadEventRouter,
}

struct Session {
    id: SessionId,
    created_at, last_active,
    title: Option<String>,
    archived: bool,
    threads: Vec<ThreadId>,                // root thread is threads[0]
    /// Defaults consulted when a thread under this session needs auto-provisioned
    /// resources. Threads never inherit *bindings* — they inherit *defaults* and
    /// the resolver decides what to bind.
    defaults: SessionDefaults,
}

struct SessionDefaults {
    sandbox: Option<SandboxSpec>,          // None ⇒ no sandbox unless thread asks
    mcp_hosts: Vec<McpHostSpec>,
    backend: Option<BackendId>,
    approval_policy: ApprovalPolicy,
    tool_allowlist: BTreeSet<String>,      // session-wide trust grants persist across threads
}

struct Thread {
    id: ThreadId,
    session_id: SessionId,
    parent: Option<ThreadId>,              // for spawn / fork / compact
    kind: ThreadKind,
    created_at, last_active,
    title: Option<String>,
    bindings: ThreadBindings,
    config: ThreadConfig,                  // model, system_prompt, max_tokens, max_turns, sampling
    conversation: Conversation,
    total_usage: Usage,
    turns_in_cycle: u32,
    internal: ThreadInternalState,
}

enum ThreadKind {
    Root,
    Spawn   { spawned_by_tool_use_id: String },   // child of a subagent call
    Fork    { from_msg_idx: usize },              // rewind & fork
    Compact { summary_of_msg_range: (usize, usize) },
}

struct ThreadBindings {
    sandbox:   Option<SandboxId>,          // None ⇒ tool calls that need fs access fail
    mcp_hosts: Vec<McpHostId>,             // ordered: earlier wins on tool-name collision
    backend:   BackendId,
    /// If Some, only tool names in this set are exposed to the model. Used by
    /// subagent spawns that should see a narrower tool surface than the parent.
    tool_filter: Option<HashSet<String>>,
}
```

The thread state machine collapses — it no longer carries the MCP setup phases:

```rust
enum ThreadInternalState {
    Idle,
    Completed,
    /// Waiting for one or more bound resources to reach Ready before issuing a
    /// model call. Scheduler nudges the thread when each id flips state.
    WaitingOnResources { needed: Vec<ResourceId> },
    NeedsModelCall,
    AwaitingModel       { op_id, started_at },
    AwaitingApproval    { tool_uses, dispositions },
    AwaitingTools       { pending_dispatch, pending_io, completed, approvals },
    Failed   { at_phase, message },
    Cancelled,
}
```

`NeedsMcpConnect / AwaitingMcpConnect / NeedsListTools / AwaitingListTools` are gone. MCP connection and tool listing belong to the `McpHost` resource's own lifecycle, not the thread's.

### Resource entries

Each resource is a small struct with its own state machine, refcount, and pin flag:

```rust
struct SandboxEntry {
    id: SandboxId,
    spec: SandboxSpec,
    state: ResourceState,                  // Provisioning | Ready | Errored | TornDown
    handle: Option<Box<dyn SandboxHandle>>,// populated when Ready
    threads_using: BTreeSet<ThreadId>,     // refcount, by id (so we can list inspectability)
    pinned: bool,                          // user said "keep alive"
    last_used: DateTime<Utc>,              // for idle timeout
    created_at: DateTime<Utc>,
}

struct McpHostEntry {
    id: McpHostId,
    spec: McpHostSpec,                     // url, optional sandbox to run inside, optional auth
    state: ResourceState,
    session: Option<Arc<McpSession>>,
    tools: Vec<ToolDescriptor>,            // populated after list_tools
    annotations: HashMap<String, ToolAnnotations>,
    threads_using: BTreeSet<ThreadId>,
    pinned: bool,
    last_used, created_at,
}

struct BackendEntry {
    id: BackendId,
    name: String,                          // "anthropic", "openai-compat", etc.
    provider: Arc<dyn ModelProvider>,
    default_model: Option<String>,
    threads_using: BTreeSet<ThreadId>,
    pinned: bool,                          // backends typically pinned-by-default
    last_used, created_at,
}

enum ResourceState {
    Provisioning { op_id: OpId },          // I/O in flight
    Ready,
    Errored { message: String },
    TornDown,                              // terminal — entry lingers briefly for inspection then GC'd
}
```

The wire protocol exposes a collapsed view (`ResourceStateLabel: Provisioning | Ready | Errored | TornDown`), same indirection as `TaskStateLabel` today.

## Scheduler shape

The single `select!` loop carries over. The dispatch surface grows but the *structure* doesn't:

```
Inbox messages now include:
  - Session/thread CRUD: CreateSession, CreateThread, RebindThread, CancelThread, ...
  - Resource CRUD: CreateSandbox, DestroySandbox, ReconnectMcpHost, PinResource, ...

I/O completions now include:
  - Resource provisioning: SandboxProvisioned, McpHostConnected, ToolsListed
  - Per-thread ops: ModelCall, ToolCall (unchanged shape; routed via thread.bindings)
```

`step_until_blocked(thread)` — same control flow as today, but the precondition check is new:

```
loop {
    match thread.step() {
        DispatchIo(req)       => { resolve req against thread.bindings, push to pending_io; break; }
        WaitOnResources(ids)  => { register thread as a waiter on each id; break; }
        Continue              => { continue; }
        Paused                => { break; }
    }
}
```

When a `ResourceState` flips to `Ready`, the scheduler walks the waiter list and re-runs `step_until_blocked` for each waiting thread.

## Auto-provisioning (policy layer)

A new `ResourceResolver` sits between thread creation and the registries. When `CreateThread` lands with `bindings.sandbox: None` and `prefer_sandbox: Some(spec)`:

```
1. Look up an existing Ready sandbox whose spec matches `spec` (deep equality on
   image + mounts + network + limits).
2. Match → bind the existing SandboxId, increment refcount, done.
3. No match → enqueue a CreateSandbox op, bind the new id, thread enters
   WaitingOnResources until provisioning completes.
```

Same logic for MCP hosts. The thread always ends up with a concrete binding — "auto" never means "no resource," only "I didn't pick one explicitly."

The resolver is the *only* code that can spawn resources implicitly. UI-driven `CreateSandbox` requests bypass it entirely. This separation keeps the data model honest: every binding traces back to either a user action or a documented auto-provision rule.

## Mid-thread rebinding

```rust
SchedulerMsg::RebindThread {
    thread_id: ThreadId,
    patch: ThreadBindingsPatch {
        sandbox:     Option<Option<SandboxId>>,   // outer Some ⇒ "change", inner None ⇒ "unbind"
        mcp_hosts:   Option<Vec<McpHostId>>,
        backend:     Option<BackendId>,
        tool_filter: Option<Option<HashSet<String>>>,
    },
}
```

Applied immediately. Future I/O ops use the new bindings; in-flight ops complete against their original resources (the future already captured the necessary `Arc`s when it was built). Refcounts are adjusted: old resources `threads_using.remove(&thread_id)`, new ones `.insert(thread_id)`.

**Semantic warning to the model.** When the sandbox or the MCP host set changes, the scheduler appends a synthetic system message to the conversation:

> *Note: the execution environment changed at this point. Files, processes, and tool availability may differ from earlier in this conversation.*

Without this, the model continues referencing files and outputs from the old environment with no awareness that they may not exist. The warning is automatic but cosmetic — users can disable it per-thread if they're scripting orchestration.

## Resource lifecycle

State machine for every resource kind:

```
   CreateSandbox / CreateMcpHost
            │
            ▼
     Provisioning ──── error ──→ Errored
            │                      │
         success                 retry?
            │                      │
            ▼                      ▼
          Ready ←──────────────────┘
            │
        decommission
        / GC eligible
            │
            ▼
        TornDown ──→ removed from registry after grace period
```

**GC policy: refcount + idle timeout, with pinning.**

- Default decommission is automatic: when `threads_using.is_empty() && !pinned && now() - last_used > idle_timeout`, the scheduler tears down the resource.
- The default idle timeout is generous (15 minutes for sandboxes, 30 for MCP hosts; backends never idle-out).
- `pinned: true` opts out of GC entirely. UI exposes a pin toggle.
- Explicit `DestroyResource` always wins, even over `pinned`. Refused only if `threads_using` is non-empty (returns an error suggesting which threads to cancel/rebind first).

This balances "don't leak containers" against "don't yank a thing the user is mid-conversation about."

## Wire protocol

### Client → Server (additions and replacements)

```rust
enum ClientToServer {
    // Session lifecycle
    CreateSession   { correlation_id, initial_message, session_defaults: Option<SessionDefaultsOverride> },
    ArchiveSession  { session_id },

    // Thread lifecycle
    CreateThread    { session_id, parent: Option<ThreadId>, kind: ThreadKind,
                      initial_message: Option<String>, bindings: Option<ThreadBindingsRequest>,
                      config: ThreadConfig },
    SendUserMessage { thread_id, text },
    CancelThread    { thread_id },
    RebindThread    { thread_id, patch: ThreadBindingsPatch },
    ApprovalDecision{ thread_id, approval_id, choice, remember },

    // Subscription
    SubscribeToSession   { session_id },     // sends SessionSnapshot, then session-tier events
    SubscribeToThread    { thread_id },      // sends ThreadSnapshot, then thread-tier events
    UnsubscribeFromSession { session_id },
    UnsubscribeFromThread  { thread_id },

    // Resource CRUD (UI direct, also reachable via tools)
    CreateSandbox        { spec: SandboxSpec, pinned: bool },
    DestroySandbox       { sandbox_id, force: bool },
    PinResource          { resource: ResourceRef, pinned: bool },
    ConnectMcpHost       { spec: McpHostSpec, pinned: bool },
    DisconnectMcpHost    { mcp_host_id },
    ListResources        { correlation_id },

    // Listing
    ListSessions { correlation_id },
}
```

### Server → Client (additions and replacements)

```rust
enum ServerToClient {
    // Session-list tier (broadcast)
    SessionCreated      { session_id, summary },
    SessionStateChanged { session_id, summary },
    SessionArchived     { session_id },

    // Thread-list tier (only to subscribers of the parent session)
    ThreadCreated       { session_id, thread_id, summary },
    ThreadStateChanged  { session_id, thread_id, state: ThreadStateLabel },
    ThreadBindingsChanged { thread_id, bindings: ThreadBindings },

    // Per-thread turn tier (only to thread subscribers)
    ThreadAssistantBegin / Text / Reasoning / End  { thread_id, ... },
    ThreadToolCallBegin / End                       { thread_id, ... },
    ThreadPendingApproval / ApprovalResolved        { thread_id, ... },
    ThreadLoopComplete                              { thread_id },
    ThreadAllowlistUpdated                          { thread_id, ... },

    // Resource tier (broadcast — every connected client sees the registry)
    ResourceCreated     { resource: ResourceSnapshot },
    ResourceStateChanged{ resource_id, state: ResourceStateLabel },
    ResourceUsageChanged{ resource_id, threads_using: Vec<ThreadId> },
    ResourceDestroyed   { resource_id },

    // Snapshots and lists
    SessionList         { correlation_id, sessions: Vec<SessionSummary> },
    SessionSnapshot     { session_id, snapshot: SessionSnapshot },     // includes thread summaries
    ThreadSnapshot      { thread_id, snapshot: ThreadSnapshot },       // includes conversation
    ResourceList        { correlation_id, resources: Vec<ResourceSnapshot> },

    Error { correlation_id, session_id, thread_id, message },
}
```

Three event tiers replace today's two: session-list (cheap, broadcast), thread-list/turn (per-subscription), resource (broadcast — small registry, every UI wants it).

## Persistence

- **Sessions** — JSON file per session at `<state-dir>/sessions/<session_id>.json`. Includes the session metadata + every thread under it (full thread bodies, conversations included). Written on any thread state transition.
- **Resources** — `<state-dir>/resources.json`, a single file holding the *specs* (not live state) of every persisted resource. Live state (container IDs, MCP session cookies) is reconstructed lazily on startup.
- **On startup** — read `resources.json`, populate registries with `state: Provisioning` deferred until first use. Read every session file, populate sessions + threads. Mark any `AwaitingModel / AwaitingTools / Provisioning` states as `Failed { at_phase: "resume" }` — same crash-recovery story as today.

Sessions and threads upgrade to SQLite when cross-session queries (search by title, by date, by tools used) get useful. Resources stay in a single file forever — the registry is small.

## What this enables

Each follow-on feature becomes a small, focused change once this lands:

| Feature | Implementation sketch |
|---|---|
| **Subagent** | A `spawn_thread` tool. The tool handler creates a child Thread (`kind: Spawn`), bindings inherit from parent with optional `tool_filter` narrowing, parent transitions to a new `AwaitingChildThread { child_id }` state. When the child reaches `Completed`, parent resumes and the child's summary is delivered as the synthesized tool_result. |
| **Compact** | A new Thread (`kind: Compact`) under the same session. Conversation seeded with one user message containing the prior thread's summary. Old thread untouched. UI default-shows the new one. |
| **Rewind & fork** | Same as compact but seeded with `parent.conversation[..from_msg_idx]` instead of a summary. |
| **Non-interactive task** | A session created without a subscribed UI; thread runs to terminal state and is preserved. Helper threads (subagents) spawn freely without polluting the top-level session list. |
| **Lua / scripted loops** | A `ThreadInternalState::Scripted { interpreter_state }` variant. The script makes the same decisions today's hardcoded `step()` makes (when to call the model, when to dispatch tools, when to terminate). Out of scope for this refactor; the data model just doesn't preclude it. |

## What this replaces

Directly superseded — gets deleted:

- `src/task.rs` — replaced by `src/thread.rs` + `src/session.rs`.
- The `Task` / `TaskInternalState` / `TaskConfig` types.
- The `TaskEvent` enum — split into `ThreadEvent` (turn-tier) and `SessionEvent` (list-tier).
- Per-task pool plumbing in `src/scheduler.rs` — replaced by the resource registries.
- The `IoRequest::McpConnect` and `IoRequest::ListTools` variants — moved into resource provisioning.
- `src/io_dispatch.rs::build_model_request` — refactored to read from `thread.bindings` instead of `task.config`.

Largely rewritten:

- `src/scheduler.rs` — the loop shape carries over but the apply/dispatch surface is mostly new code. Best to write it fresh against the new data model rather than morph the existing file.
- `src/task_router.rs` → `src/thread_router.rs` — three event tiers instead of two; new resource registry.
- `crates/whisper-agent-protocol/src/lib.rs` — the wire enums change wholesale. Worth versioning the protocol now (`PROTOCOL_VERSION` constant on connect handshake) so clients fail loudly against an old server.
- `crates/whisper-agent-webui/src/app.rs` — needs new resource-list pane and a thread-tree-under-session UI. Significant frontend work.

Compatible — minimal change:

- `src/mcp.rs` — `McpSession` is unchanged; only the lifecycle around it moves.
- `src/sandbox/*` — sandbox provider trait and backends unchanged; only the lifecycle around handles moves.
- `src/anthropic.rs`, `src/openai.rs`, `src/model.rs` — model providers unchanged. They're invoked the same way, just dispatched against `thread.bindings.backend` instead of `task.config.backend`.
- `crates/whisper-agent-mcp-*` — daemons unchanged.
- `src/audit.rs` — schema gets a `thread_id` column alongside the existing `task_id` (renamed `session_id`).

Persistence: **clean break.** Existing on-disk task files are not migrated. The directory is moved aside on first startup with a `.pre-thread-refactor/` suffix, and a fresh sessions tree starts. This is consistent with the project's "delete old code first when reworking a subsystem" stance — a one-way migration shim is more code and more risk than is justified for a personal-tool stage.

## Migration shape

Single PR is too big. Phased, each phase ends in a green CI:

1. **Resource registries (no behavior change yet).** Add the registries alongside the existing per-task pool. Auto-populate them from the per-task data on `McpConnect` completion. Webui gets a read-only resource pane. Everything else still routes through `task_id`.

2. **Thread = Task with a wrapper.** Rename `Task` → `Thread`, introduce `Session` with exactly one thread per session, route the wire protocol through the new `(session_id, thread_id)` keys. Every existing UI flow gets an extra layer of indirection but works identically.

3. **Decouple resource ownership.** Threads switch to `bindings: ThreadBindings`. The per-task pool goes away. Resource lifecycles become independent. Refcounts and GC come online. Mid-thread rebind ships.

4. **Subagent (`spawn_thread` tool).** First feature that pressure-tests the multi-thread-per-session story. Surfaces any remaining 1:1 assumptions.

5. **Compact + fork.** Trivially small once spawn works.

6. **Lua / non-interactive / parallel forks.** Separate efforts on top.

Phase boundaries are commit/PR boundaries. `run_one_shot` and the existing CLI keep working through every phase.

## Not in scope (deferred to follow-up designs)

- **Lua-scripted threads.** Data model accommodates them; the actual interpreter integration, sandboxing of script execution, and scripting API surface are future work.
- **Cross-session resource attachment.** Resources are global within the scheduler, but the wire protocol scopes most operations to a session. Whether the UI lets one session's threads attach to "another session's sandbox" or whether resources are presented as a flat global pool is a UI question, deferred until inspectability ships.
- **Real cancellation of in-flight tools.** Same as today — `Cancelled` marks the thread, in-flight futures complete and discard.
- **Resource snapshotting / restore.** A sandbox could in principle be checkpointed to disk and restored elsewhere. Not designed.
- **Quota / rate-limiting per resource.** Out of scope.
- **Auth changes.** The connection layer is unchanged; permissions doc still applies, just retargeted to threads instead of tasks.

## Open questions

These were raised during design discussion and a default has been chosen, but flagging explicitly so they can be revisited before implementation:

1. **GC policy.** Refcount + 15-minute idle timeout for sandboxes, 30 for MCP hosts, never for backends. Pin opt-out. Could instead be manual-only (simpler, more leak-prone) or per-resource-class with smarter heuristics (more code).
2. **Mid-thread swap warning.** Automatic synthetic system message. Could instead require the user (or the orchestrating script) to insert their own marker.
3. **Default thread on new session.** A "new session" UI action creates a session *and* a root thread, auto-binding to a default-spec sandbox + the configured MCP host set. The honest data-model answer would be "create the session, let the user pick bindings" — but the auto-provision case is the common path and shouldn't require two clicks.
4. **Compact: fork or in-place.** A new thread (`kind: Compact`) preserving history, not in-place mutation. Costs slightly more disk; gains the ability to scroll back to pre-compact state.
5. **Subagent approvals.** A subagent thread inherits its parent's `approval_policy` and surfaces approvals to the same subscribers as the parent thread. A spawned thread that wants to run autonomously sets `approval_policy: AutoApproveAll` explicitly in its `ThreadConfig`. The wire protocol carries a `parent_thread_id` so UIs can group child approvals visually.
