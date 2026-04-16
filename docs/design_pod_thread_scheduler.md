# Pods, Threads, and Decoupled Resources

Architectural pivot following `design_task_scheduler.md`. The first scheduler design fused three concerns into the `Task` type: it was the scheduler's unit of work, the user's unit of organization, *and* the owner of every external resource the loop touched (MCP session, sandbox, model backend). That fusion was wrong, and it's now blocking the next round of features.

This document replaces `Task` with three layered concepts:

- **Pod** — a *directory on disk* with a `pod.toml` config and per-thread JSON files. Persistent by definition: a pod IS its directory. The TOML defines what threads inside the pod are *allowed* to use (backends, MCP hosts, sandbox specs) and the defaults applied to fresh threads.
- **Thread** — the scheduler's unit; one conversation, one state machine, persisted as one JSON file under the pod's `threads/` subdirectory.
- **Resources** (Sandbox, McpHost, Backend) — first-class scheduler-level entities with independent lifecycles. Pods reference them by name; thread bindings resolve to concrete resource ids.

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

All five want a *thread*-shaped abstraction nested inside a stable container.

Independently, two more issues force the rework:

**Resource ownership is wrong.** Sandboxes are provisioned on first MCP connect and torn down at task terminal state. Two pods working in the same project directory can't share a sandbox; a parallel fork can't reuse its parent's running container. MCP sessions are owned per-task. The "shared MCP host" feature is grafted onto a model that fundamentally assumes 1:1.

**Capability/policy has no home.** Today the answer to "what can this work do?" is sprawled across `default_task_config`, `shared_mcp_hosts`, ad-hoc validation in `create_task`. There's no single artifact the user can read or edit to understand a project's capability boundary. With pods-as-directories, the answer becomes one TOML file you can `cat`.

The scheduler itself is *already* concurrency-friendly — `FuturesUnordered` carries any number of in-flight ops across any number of tasks, `McpSession` is `Arc`-shared and safe under concurrent JSON-RPC. The constraint is in the data model, not the runtime.

## The shift

```
Before:                            After:
  Task                               Pod (directory + pod.toml)
   ├── conversation                   ├── pod.toml         ← capability boundary, defaults
   ├── state machine                  ├── system_prompt.md ← (or any user files)
   ├── owns McpSession                ├── threads/
   ├── owns SandboxHandle             │   └── thread-*.json
   └── refs Backend                   └── (future) hooks/*.lua

                                    Thread (one JSON file)
                                     ├── conversation
                                     ├── state machine
                                     └── bindings: { sandbox?, mcp_hosts[], backend, tool_filter? }
                                                ↓ resolves against ↓
                                    Sandbox / McpHost / Backend
                                      (scheduler-level registry, refcounted, GC'd)
```

A pod is the *project*: stable, durable, version-controllable. A thread is a *conversation* within that project: born, run to completion, archived. Resources are *infrastructure*: live in the scheduler, shared across pods and threads where specs match.

## Pod directory layout

```
<pods_root>/
  whisper-agent-dev/                      # pod id == directory name (immutable)
    pod.toml                              # source of truth for config + capability
    system_prompt.md                      # referenced by thread_defaults; user-editable
    threads/
      thread-01HVABCDE.json               # one file per thread (full conversation)
      thread-01HVFGHIJ.json
    hooks/                                # reserved for future Lua scripts
  another-pod/
    pod.toml
    threads/
      thread-01HV...json
  .archived/                              # archived pods moved here, not deleted
    deleted-experiment/
      pod.toml
      ...
```

`<pods_root>` is configured at server startup. There is exactly one. Long-running servers point it at `~/.local/share/whisper-agent/pods/` (or wherever the user prefers); CLI invocations point it at `/tmp/whisper-agent-cli-<pid>/` and let the OS handle cleanup.

Pod ids are directory names. Immutable — renaming a pod would break thread-internal references and audit history. The TOML's `name = "..."` field carries the human-friendly display name and is freely editable.

## Pod config TOML

A pod is fully described by its `pod.toml`. Strawman shape:

```toml
# Identity
name = "whisper-agent dev"
description = "Working on whisper-agent itself"
created_at = "2026-04-16T10:23:11Z"      # set on creation; informational

# What this pod's threads are *allowed* to use. Threads can be more
# restrictive (a thread's bindings are a subset of these); they cannot
# escalate. Anything not in [allow] is unreachable from this pod.
[allow]
backends  = ["anthropic", "openai-compat"]   # references to server-catalog backends
mcp_hosts = ["fetch", "search"]              # references to server-catalog shared hosts

# Sandbox specs are inline — pods are self-contained for sandboxes since
# they describe project-specific filesystem layout.
[[allow.sandbox]]
name = "landlock-rw-workspace"
type = "landlock"
allowed_paths = [
  { path = "/home/me/projects/whisper-agent", mode = "read_write" },
  { path = "/",                                mode = "read_only"  },
]
network = "unrestricted"

[[allow.sandbox]]
name = "no-isolation"
type = "none"

# Defaults applied to every fresh thread. Each field is overridable per-thread.
[thread_defaults]
backend            = "anthropic"
model              = "claude-opus-4-7"
system_prompt_file = "system_prompt.md"     # path relative to pod dir
max_tokens         = 32000
max_turns          = 100
approval_policy    = "prompt_destructive"
sandbox            = "landlock-rw-workspace" # name from [[allow.sandbox]]
mcp_hosts          = ["fetch", "search"]    # subset of [allow].mcp_hosts

# Pod-level limits across all threads.
[limits]
max_concurrent_threads = 10
# (Token budgets, cost caps, etc. arrive when there's a concrete need.)

# Reserved for future Lua hooks.
# [[hooks]]
# event  = "before_model_call"
# script = "hooks/before_model.lua"
```

Why this shape:

- **`[allow]` is the capability boundary.** One table, one place to read. The webui builds an "edit pod" form directly from it; future self-modification tools mutate exactly these keys.
- **Backends / MCP hosts reference the server catalog by name.** Backends require API keys, daemons, etc. — server-level concerns. Pods opt into a subset of what the server already runs.
- **Sandboxes are inline.** Sandboxes are project-specific (path layouts, mounts, network rules) — making them server-catalog entries would force users to maintain two configs. A pod's `[[allow.sandbox]]` array is the full spec list.
- **`thread_defaults` separate from `[allow]`.** Defaults are a *starting point* for new threads; `[allow]` is the *cap*. A thread can pick a backend that's allowed-but-not-default; it can't pick one that's not in `[allow]`.
- **`system_prompt_file` references a sibling file**, so long prompts edit cleanly with normal text editors instead of being squashed into TOML strings.

## Data model (in-memory)

```rust
struct Scheduler {
    pods_root: PathBuf,
    pods: HashMap<PodId, Pod>,                   // PodId = directory name
    threads: HashMap<ThreadId, Thread>,          // flat — pods store ThreadIds
    resources: ResourceRegistry,                 // already in place from Phase 1
    pending_io: FuturesUnordered<IoFuture>,
    router: ThreadEventRouter,
    server_catalog: ServerCatalog,               // backends + shared MCP hosts known at server level
}

struct Pod {
    id: PodId,                                   // directory name; immutable
    dir: PathBuf,                                // <pods_root>/<id>
    config: PodConfig,                           // parsed pod.toml
    raw_toml: String,                            // for round-trip / UI editing
    threads: Vec<ThreadId>,
    archived: bool,                              // mirror of being under .archived/
}

struct PodConfig {
    name: String,
    description: Option<String>,
    created_at: DateTime<Utc>,
    allow: PodAllow,
    thread_defaults: ThreadDefaults,
    limits: PodLimits,
    // hooks: Vec<HookSpec>,                     // future
}

struct PodAllow {
    backends: Vec<String>,                       // server-catalog names
    mcp_hosts: Vec<String>,                      // server-catalog names
    sandbox: Vec<NamedSandboxSpec>,              // inline
}

struct NamedSandboxSpec {
    name: String,
    spec: SandboxSpec,                           // existing protocol type
}

struct Thread {
    id: ThreadId,
    pod_id: PodId,
    parent: Option<ThreadId>,                    // for spawn / fork / compact
    kind: ThreadKind,
    created_at, last_active,
    title: Option<String>,
    bindings: ThreadBindings,                    // resolves to resource ids
    config: ThreadConfig,                        // model, system_prompt, max_tokens, max_turns
    conversation: Conversation,
    total_usage: Usage,
    turns_in_cycle: u32,
    internal: ThreadInternalState,
}

enum ThreadKind {
    Root,
    Spawn   { spawned_by_tool_use_id: String },  // child of a subagent call
    Fork    { from_msg_idx: usize },             // rewind & fork
    Compact { summary_of_msg_range: (usize, usize) },
}

struct ThreadBindings {
    sandbox:     Option<SandboxId>,
    mcp_hosts:   Vec<McpHostId>,                 // ordered; earlier wins on tool collision
    backend:     BackendId,
    tool_filter: Option<HashSet<String>>,        // narrows visible tools
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
    AwaitingChildThread { child_id: ThreadId },  // subagent
    Failed   { at_phase, message },
    Cancelled,
}
```

`NeedsMcpConnect / AwaitingMcpConnect / NeedsListTools / AwaitingListTools` are gone. MCP connection and tool listing belong to the `McpHost` resource's own lifecycle, not the thread's — already in place from Phase 1.

## Server catalog

The pod TOML references backends and MCP hosts *by name*. Those names are defined in the server's own config (today: `whisper-agent.toml` at the repo root):

```rust
struct ServerCatalog {
    backends: HashMap<String, BackendSpec>,      // anthropic, openai-compat, ...
    mcp_hosts: HashMap<String, McpHostSpec>,     // fetch, search, ...
    pods_root: PathBuf,
    sandbox_provider: SandboxProviderSpec,       // BareMetal | DaemonClient(url)
}
```

The catalog defines what the server *can* run. A pod's `[allow]` table picks a subset. Threads bind to specific entries.

A pod that references a catalog entry the server doesn't know warns at load time and excludes that entry from the effective `[allow]`. (Same defensive behavior as today's `shared_mcp_hosts` validation in `create_task`.)

## Resource registries

Unchanged from Phase 1 — see `src/resources.rs`. Scheduler still owns three flat `HashMap`s of `SandboxEntry / McpHostEntry / BackendEntry` with refcount, pin, and `ResourceState` (`Provisioning | Ready | Errored | TornDown`). The pod model adds two new behaviors at the *resolver* layer:

- **Pod allowlist enforcement.** When a thread requests a binding, the resolver verifies the requested resource (or the spec to auto-provision) is in the pod's `[allow]` table. Not allowed → request rejected with a wire error.
- **Sandbox spec dedup across pods.** Two pods with identical inline sandbox specs share the same provisioned sandbox in the registry. Spec equality is the dedup key, not the pod boundary.

GC policy (refcount + idle timeout + pin) is unchanged.

## Auto-provisioning (policy layer)

A `ResourceResolver` sits between thread creation and the registries. When a thread is created with `bindings.sandbox = "landlock-rw-workspace"`:

```
1. Look up "landlock-rw-workspace" in the pod's [[allow.sandbox]] entries —
   reject if absent (pod doesn't allow this sandbox).
2. Look for an existing Ready sandbox in the registry whose spec deep-equals
   the named spec. Match → bind the existing SandboxId.
3. No match → enqueue CreateSandbox, bind the new id, thread enters
   WaitingOnResources until provisioning completes.
```

Same logic for MCP hosts (matched by catalog name + url) and backends (matched by catalog name).

The resolver is the *only* code that can spawn resources implicitly. UI-driven `CreateSandbox` requests bypass it entirely (those create unscoped resources, useful for diagnostic / shared infrastructure).

## Mid-thread rebinding

```rust
SchedulerMsg::RebindThread {
    thread_id: ThreadId,
    patch: ThreadBindingsPatch,                  // sandbox / mcp_hosts / backend / tool_filter
}
```

Applied immediately. Rebinds are validated against the pod's `[allow]` table — same rules as initial binding. Future I/O ops use the new bindings; in-flight ops complete against their original resources (already captured in the future). Refcounts adjust: old resources `users.remove(thread_id)`, new ones `.insert(thread_id)`.

When the sandbox or MCP host set changes, the scheduler appends a synthetic system message to the conversation:

> *Note: the execution environment changed at this point. Files, processes, and tool availability may differ from earlier in this conversation.*

Cosmetic but important: without it the model continues referencing files and outputs from the old environment with no awareness that they may not exist.

## Wire protocol

### Client → Server

```rust
enum ClientToServer {
    // Pod lifecycle (programmatic creation; manual creation via mkdir+toml works too)
    ListPods         { correlation_id },
    GetPod           { correlation_id, pod_id },                  // returns parsed config + thread summaries
    CreatePod        { correlation_id, pod_id, config: PodConfig }, // server creates dir + writes TOML
    UpdatePodConfig  { correlation_id, pod_id, toml_text },       // full TOML replacement, parsed for validity
    ArchivePod       { pod_id },                                   // mv to <pods_root>/.archived/

    // Thread lifecycle (replaces the old Task* messages)
    CreateThread     { correlation_id, pod_id, parent: Option<ThreadId>, kind: ThreadKind,
                       initial_message: Option<String>, bindings: Option<ThreadBindingsRequest>,
                       config_override: Option<ThreadConfigOverride> },
    SendUserMessage  { thread_id, text },
    CancelThread     { thread_id },
    RebindThread     { thread_id, patch: ThreadBindingsPatch },
    ApprovalDecision { thread_id, approval_id, choice, remember },

    // Subscription
    SubscribeToPod    { pod_id },               // pod-list tier (thread summaries under this pod)
    SubscribeToThread { thread_id },            // turn tier (assistant text, tool calls, approvals)
    UnsubscribeFromPod    { pod_id },
    UnsubscribeFromThread { thread_id },

    // Resource registry (unchanged from Phase 1b)
    ListResources    { correlation_id },
    CreateSandbox    { spec, pinned },          // unscoped, UI-driven diagnostic
    DestroySandbox   { sandbox_id, force },
    PinResource      { resource_ref, pinned },

    // Server catalog (read-only)
    ListBackends     { correlation_id },        // unchanged
    ListMcpHosts     { correlation_id },        // new — surfaces what pods can opt into
    ListModels       { correlation_id, backend },
}
```

### Server → Client

```rust
enum ServerToClient {
    // Pod-list tier (broadcast to every connected client)
    PodList            { correlation_id, pods: Vec<PodSummary> },
    PodCreated         { pod: PodSummary, correlation_id },
    PodConfigUpdated   { pod_id, toml_text, parsed: PodConfig, correlation_id },
    PodArchived        { pod_id },

    // Pod-detail tier (only to subscribers of the pod)
    PodSnapshot        { pod_id, snapshot: PodSnapshot },         // config + thread summaries
    ThreadCreated      { pod_id, thread_id, summary: ThreadSummary },
    ThreadStateChanged { pod_id, thread_id, state: ThreadStateLabel },
    ThreadBindingsChanged { thread_id, bindings: ThreadBindings },

    // Per-thread turn tier (only to thread subscribers; renamed from TaskAssistant* etc.)
    ThreadAssistantBegin / Text / Reasoning / End  { thread_id, ... },
    ThreadToolCallBegin / End                       { thread_id, ... },
    ThreadPendingApproval / ApprovalResolved        { thread_id, ... },
    ThreadLoopComplete                              { thread_id },
    ThreadAllowlistUpdated                          { thread_id, ... },

    // Resource tier (unchanged from Phase 1b)
    ResourceList        { correlation_id, resources },
    ResourceCreated / Updated / Destroyed { ... },

    // Catalog responses
    BackendsList        { correlation_id, default_backend, backends },
    McpHostsList        { correlation_id, hosts: Vec<McpHostCatalogEntry> },
    ModelsList          { correlation_id, backend, models },

    Error { correlation_id, pod_id, thread_id, message },
}
```

Three event tiers replace today's two: pod-list (cheap, broadcast), pod-detail/thread-turn (per-subscription), resource (broadcast — small registry, every UI wants it).

## Persistence

**Pods are persisted as their directories, not as separate snapshots.** Writing a pod = writing `pod.toml` and the `threads/*.json` files. Reading a pod = parsing those.

- **`pod.toml`** is rewritten only on `UpdatePodConfig` or `CreatePod`. Hand-edits on disk are picked up on the next read; in-process pods don't auto-reload (a future file watcher is straightforward but deferred).
- **`threads/<thread_id>.json`** is rewritten on every thread state transition. The flush schedule mirrors today's `Persister::flush`.
- **No SQLite, no aggregate index file.** `ListPods` walks `<pods_root>` (and `.archived/` if requested); the cost is negligible at expected scale (≤ a few hundred pods).

**On startup**: scheduler reads its server config (backends, MCP host catalog, `pods_root`), then walks `<pods_root>` and loads every pod's TOML + threads. Threads in `AwaitingModel / AwaitingTools / WaitingOnResources` are flipped to `Failed { at_phase: "resume" }` — same crash-recovery story as today.

**Migration from current task files**: clean break. `<state-dir>/tasks/` is moved aside on first startup with a `.pre-pod-refactor/` suffix, and a fresh `<pods_root>` tree starts. Per the project's "delete old code first when reworking a subsystem" stance — a one-way migration shim is more code and more risk than is justified at the current size.

## CLI one-shot

The CLI uses `/tmp/whisper-agent-cli-<pid>/` as its `pods_root`. Inside, it programmatically creates one pod (typically named `cli`), creates one thread under it with the requested message, runs to terminal state, and prints the conversation. The directory persists for post-run inspection (`<tmp>/cli/threads/*.json`); OS `/tmp` cleanup eventually removes it.

A `--pods-root <path>` flag overrides the default for users who want CLI runs to share a long-lived pod (e.g., a chat history they review later).

## What this enables

| Feature | Implementation sketch |
|---|---|
| **Subagent** | A `spawn_thread` tool. The handler creates a child Thread (`kind: Spawn`), bindings inherit from parent (further restriction allowed; pod allowlist still caps), parent transitions to `AwaitingChildThread { child_id }`. When the child reaches `Completed`, parent resumes and the child's summary is delivered as the synthesized tool_result. |
| **Compact** | A new Thread (`kind: Compact`) under the same pod. Conversation seeded with one user message containing the prior thread's summary. Old thread untouched. UI default-shows the new one. |
| **Rewind & fork** | Same as compact but seeded with `parent.conversation[..from_msg_idx]` instead of a summary. |
| **Non-interactive task** | A pod created without a subscribed UI; thread runs to terminal state and is preserved. Helper threads (subagents) spawn freely without polluting the top-level pod list. |
| **Lua scripts** | `pod.toml` `[[hooks]]` table references scripts under `<pod>/hooks/`. Scheduler calls into the interpreter at named events (`before_model_call`, `on_tool_result`, etc.). Out of scope for this refactor; the data model just doesn't preclude it. |
| **Pod self-modification** | An approval-gated `pod_config_edit { toml_patch }` tool. Server validates the patch against schema + invariants (e.g., can't remove a backend a live thread depends on), writes the new TOML, broadcasts `PodConfigUpdated`. Out of scope for this refactor. |

## What this replaces

Directly superseded — gets deleted:

- `src/task.rs` — replaced by `src/thread.rs` + `src/pod.rs`.
- The `Task` / `TaskInternalState` / `TaskConfig` types.
- The `TaskEvent` enum — split into `ThreadEvent` (turn-tier) and `PodEvent` (list-tier).
- Per-task pool plumbing in `src/scheduler.rs` — the resource registry from Phase 1 takes over; the per-task `mcp_pools` and `sandbox_handles` fields go away.
- The `IoRequest::McpConnect` and `IoRequest::ListTools` variants — moved into resource provisioning.
- `src/persist.rs` — replaced by a pod-directory loader / writer.

Largely rewritten:

- `src/scheduler.rs` — the `select!` loop carries over but the apply / dispatch surface is mostly new code. Best to write it fresh against the new data model rather than morph the existing file.
- `src/task_router.rs` → `src/thread_router.rs` — three event tiers instead of two.
- `crates/whisper-agent-protocol/src/lib.rs` — the wire enums change wholesale. Worth versioning the protocol now (`PROTOCOL_VERSION` constant on connect handshake) so clients fail loudly against an old server.
- `crates/whisper-agent-webui/src/app.rs` — needs a pod-list-with-threads-tree UI, plus a pod config editor (Phase 4 deliverable).

Compatible — minimal change:

- `src/mcp.rs` — `McpSession` is unchanged; only the lifecycle around it moves.
- `src/sandbox/*` — sandbox provider trait and backends unchanged; only the lifecycle around handles moves.
- `src/anthropic.rs`, `src/openai_chat.rs`, `src/model.rs` — model providers unchanged. They're invoked the same way, just dispatched against `thread.bindings.backend`.
- `crates/whisper-agent-mcp-*` — daemons unchanged.
- `src/audit.rs` — schema gets `pod_id` and `thread_id` columns alongside the existing `task_id` (renamed; or kept as `thread_id` since that's the new name).

## Migration shape

Each phase ends in a green CI; no phase leaves the tree in a broken intermediate state.

1. **Resource registries (DONE).** Phase 1a/1b/1c — registries shadow the per-task plumbing, wire surface added, webui resources pane shipped. No behavior change.

2. **Pod-as-directory layout, single thread per pod.**
   - Restructure persistence: `<state-dir>/tasks/<task_id>.json` → `<pods_root>/<pod_id>/pod.toml` + `<pods_root>/<pod_id>/threads/<thread_id>.json`.
   - Each existing task becomes a one-thread pod with `name = task.title`, `pod_id = task_id` (so id-stable across the migration even though the on-disk shape changes).
   - Rename `Task` → `Thread` internally. `TaskConfig` splits into `PodConfig` (allow + thread_defaults) and `ThreadConfig` (per-thread overrides).
   - Wire protocol gains pod-level CRUD (`ListPods`, `GetPod`, `CreatePod`, `UpdatePodConfig`, `ArchivePod`); thread messages route through `(pod_id, thread_id)`.
   - Webui: left panel becomes pod list; selecting a pod shows its single thread (multi-thread per pod arrives in Phase 4).

3. **Decouple resource ownership.** Threads switch to `bindings: ThreadBindings`. The per-task pool goes away. Resource lifecycles become independent. Refcounts and GC come online. Mid-thread rebind ships. Pod allowlist enforcement on thread creation and rebind.

4. **Multi-thread per pod + pod config editor.** UI gains "new thread in this pod" action and a pod config editor (form-based for `[allow]` and `thread_defaults`; raw TOML editor as fallback). Server validates `UpdatePodConfig` against schema + invariants.

5. **Subagent (`spawn_thread` tool).** First feature that pressure-tests the multi-thread-per-pod story. Surfaces any remaining 1:1 assumptions.

6. **Compact + fork.** Trivially small once spawn works.

7. **Lua / pod self-modification / parallel forks.** Separate efforts on top.

Phase boundaries are commit/PR boundaries. `run_one_shot` and the existing CLI keep working through every phase (CLI-pod-in-`/tmp` lands in Phase 2).

## Not in scope (deferred to follow-up designs)

- **Lua-scripted threads.** Data model accommodates them; the actual interpreter integration, sandboxing of script execution, and scripting API surface are future work.
- **Pod self-modification tools.** The `pod_config_edit` tool is sketched above but not part of this refactor. UI editing comes first; tool-driven editing once that's stable.
- **File watcher on `pod.toml`.** Hand-edits on disk are picked up on next read; live reload is straightforward (notify crate) but deferred until there's a concrete need.
- **Concurrent server access to the same `pods_root`.** Single-server-instance assumption holds for now. File locking, lease coordination, etc. are out of scope.
- **Real cancellation of in-flight tools.** Same as today — `Cancelled` marks the thread, in-flight futures complete and discard.
- **Resource snapshotting / restore.** A sandbox could in principle be checkpointed to disk and restored elsewhere. Not designed.
- **Quota / rate-limiting per pod.** `[limits]` table is reserved; only `max_concurrent_threads` is concretely planned. Token / cost budgets land when there's a concrete enforcement path.
- **Auth changes.** The connection layer is unchanged; permissions doc still applies, retargeted to threads inside pods.

## Open questions

These were deliberately deferred and a default has been chosen, but flagging so they can be revisited before implementation:

1. **Pod TOML parse-error recovery.** A pod with an unparseable `pod.toml` is shown in the UI as `Errored` (along with the parse error message); its threads are visible read-only but no new threads can be created and the pod can't be otherwise modified until the TOML is fixed. Could instead refuse to load the pod at all (simpler, but easy for a typo to hide a pod's threads).
2. **GC policy.** Refcount + 15-minute idle timeout for sandboxes, 30 for MCP hosts, never for backends. Pin opt-out. Carried over from the prior design — could revisit if pod-level usage patterns turn out different.
3. **Mid-thread swap warning.** Automatic synthetic system message. Could instead require the user (or orchestrating script) to insert their own marker.
4. **Default thread on new pod.** A "new pod" UI action creates the directory + writes a default `pod.toml`, but does *not* create a root thread. The user creates the first thread explicitly (with a "+ New thread" button), so the pod-without-threads state is observable / intentional.
5. **Subagent approvals.** A subagent thread inherits its parent's `approval_policy` and surfaces approvals to the same subscribers as the parent thread. A spawned thread that wants to run autonomously sets `approval_policy: AutoApproveAll` explicitly in its `ThreadConfig`. Wire protocol carries `parent_thread_id` so UIs can group child approvals visually.
