# Pods, Threads, and Decoupled Resources

This document describes how whisper-agent organizes long-lived agent work. Three layered concepts:

- **Pod** — a *directory on disk* with a `pod.toml` config and per-thread JSON files. Persistent by definition: a pod IS its directory. The TOML defines what threads inside the pod are *allowed* to use (backends, MCP hosts, host-env sandboxes) and the defaults applied to fresh threads.
- **Thread** — the scheduler's unit of work; one conversation, one state machine, persisted as one JSON file under the pod's `threads/` subdirectory.
- **Resources** (Backend, McpHost, HostEnv) — scheduler-level entities with independent lifecycles. Pods reference them by name; thread bindings resolve to concrete resource ids.

Companion docs: [`design_headless_loop.md`](design_headless_loop.md) (why the loop is server-side to begin with), [`design_permissions.md`](design_permissions.md) (the tool-boundary patterns), [`design_behaviors.md`](design_behaviors.md) (autonomous behaviors layered on top of pods and threads).

## The three-layer shape

```
Pod (directory + pod.toml)
 ├── pod.toml              ← capability boundary, defaults
 ├── system_prompt.md      ← referenced by thread_defaults; user-editable
 ├── threads/
 │   └── thread-*.json     ← one file per thread (full conversation)
 └── behaviors/            ← future (design_behaviors.md); not yet built
     └── <name>/...

Thread (one JSON file)
 ├── conversation
 ├── state machine (ThreadInternalState)
 └── bindings: { backend, host_env?, mcp_hosts[] }
              ↓ resolves against ↓
Resource registry
 ├── Backend   (named in server config; credentials server-side)
 ├── McpHost   (shared hosts configured server-side; host-env MCP per-thread)
 └── HostEnv   (provisioned from a spec by a named provider)
```

A pod is the *project*: stable, durable, version-controllable. A thread is a *conversation* within that project: born, run to completion, archived. Resources are *infrastructure*: live in the scheduler, shared across pods and threads where specs match, reference-counted, GC'd on idle.

## Pod directory layout

```
<pods_root>/
  whisper-agent-dev/                      # pod id == directory name (immutable)
    pod.toml                              # source of truth for config + capability
    system_prompt.md                      # referenced by thread_defaults; user-editable
    threads/
      thread-01HVABCDE.json               # one file per thread (full conversation)
      thread-01HVFGHIJ.json
  another-pod/
    pod.toml
    threads/
      thread-01HV...json
  .archived/                              # archived pods moved here, not deleted
    deleted-experiment-2026-04-16T09:12/
      pod.toml
      ...
```

`<pods_root>` is configured at server startup. Long-running servers point it at `~/.local/share/whisper-agent/pods/` or wherever the user configures; CLI one-shots point it at a `/tmp/` path the OS eventually cleans up.

Pod ids are directory names — immutable. Renaming a pod would break thread-internal references and audit history. `PodConfig.name` carries the human-friendly display name and is freely editable.

## Pod config TOML

A pod is fully described by its `pod.toml`. Shape (from `src/pod/mod.rs` and `crates/whisper-agent-protocol/src/pod.rs`):

```toml
# Identity
name = "whisper-agent dev"
description = "Working on whisper-agent itself"
created_at = "2026-04-16T10:23:11Z"      # set on creation; informational

# What this pod's threads are *allowed* to use. Threads can be more
# restrictive; they cannot escalate. Anything not in [allow] is
# unreachable from this pod.
[allow]
backends  = ["anthropic", "openai"]      # references to server-catalog backends
mcp_hosts = ["fetch", "search"]          # references to server-catalog shared hosts

# Host-env entries are named (provider, spec) pairs. The provider name
# resolves against [[host_env_providers]] in the server config; the
# spec is TOML-inline because it's project-specific.
[[allow.host_env]]
name = "default"
provider = "local-landlock"
type = "landlock"                        # HostEnvSpec discriminator
allowed_paths = [
  "/home/me/project:rw",
  "/home/me/.cargo:ro",
  "/:ro",
]
network = "unrestricted"                 # unrestricted | isolated

# Defaults applied to every fresh thread. Each field is overridable per-thread.
[thread_defaults]
backend            = "anthropic"
model              = "claude-sonnet-4-6"
system_prompt_file = "system_prompt.md"  # path relative to pod dir
max_tokens         = 16384
max_turns          = 30
approval_policy    = "prompt_pod_modify" # see design_permissions.md
host_env           = "default"           # name from [[allow.host_env]]
mcp_hosts          = ["fetch", "search"] # subset of [allow].mcp_hosts

# Pod-level limits across all threads.
[limits]
max_concurrent_threads = 10
```

Why this shape:

- **`[allow]` is the capability boundary.** One table, one place to read. The webui builds an "edit pod" form directly from it; self-modification tools (when they land) mutate exactly these keys.
- **Backends and shared MCP hosts reference the server catalog by name.** Backends hold credentials and the server handles those. Pods opt into a subset.
- **Host-envs are inline.** Filesystem layout, mount mode, and network policy are project-specific; making them server-catalog entries would force users to maintain two configs. The pod's `[[allow.host_env]]` array is the full spec list. The `provider` name still resolves against a server-level catalog because provider *implementations* (landlock today, bubblewrap/podman future) are cross-cutting.
- **`thread_defaults` separate from `[allow]`.** Defaults are a starting point for new threads; `[allow]` is the cap. A thread can pick a backend that's allowed-but-not-default; it can't pick one that isn't in `[allow]`.
- **`system_prompt_file` references a sibling file** so long prompts edit cleanly with normal text editors instead of being squashed into a TOML string.

Validation (see `src/pod/mod.rs::validate`):

- Sandbox names must be unique within a pod.
- `thread_defaults.backend` must be in `allow.backends`.
- `thread_defaults.host_env` must be in `allow.host_env` (or both must be empty — a shared-only pod with no host-env MCP).
- `thread_defaults.mcp_hosts` must all be in `allow.mcp_hosts`.

## Data model (in-memory)

```rust
struct Scheduler {
    default_pod_id: PodId,
    pods: HashMap<PodId, Pod>,
    tasks: HashMap<ThreadId, Thread>,         // flat — pods store ThreadIds
    backends: HashMap<String, BackendEntry>,  // keyed by pod-visible name
    resources: ResourceRegistry,
    router: ThreadEventRouter,
    persister: Option<Persister>,
    host_env_registry: HostEnvRegistry,
    // ...plus GC bookkeeping, provisioning-in-flight set, etc.
}

struct Pod {
    id: PodId,                                // directory name; immutable
    dir: PathBuf,                             // <pods_root>/<id>
    config: PodConfig,                        // parsed pod.toml
    raw_toml: String,                         // for round-trip / UI editing
    system_prompt: String,                    // contents of system_prompt_file
    threads: BTreeSet<ThreadId>,
    archived: bool,
}

struct Thread {
    id: ThreadId,
    pod_id: PodId,
    created_at: DateTime<Utc>,
    last_active: DateTime<Utc>,
    title: Option<String>,
    config: ThreadConfig,                     // model, system_prompt, max_tokens, max_turns, policy
    bindings: ThreadBindings,                 // resolves to resource ids
    conversation: Conversation,
    total_usage: Usage,
    archived: bool,
    turns_in_cycle: u32,
    tool_allowlist: BTreeSet<String>,         // per-thread remembered approvals
    internal: ThreadInternalState,
}

struct ThreadBindings {
    backend: String,                          // name; empty = server default
    host_env: Option<HostEnvBinding>,         // Named { name } | Inline { provider, spec }
    mcp_hosts: Vec<String>,                   // ordered; earlier wins on tool collision
}

enum ThreadInternalState {
    Idle,
    WaitingOnResources { needed: Vec<String> },
    NeedsModelCall,
    AwaitingModel       { op_id, started_at },
    AwaitingTools       { pending_dispatch, pending_io, completed, approvals },
    Completed,                                // end_turn reached; accepts follow-ups
    Failed              { at_phase, message },
    Cancelled,
}
```

The internal state is collapsed to a public `ThreadStateLabel` on the wire (`Idle | Working | Completed | Failed | Cancelled`). Clients never see the fine-grained distinctions; that stability is the point of the split. See `src/runtime/thread.rs::public_state`. Approval prompting no longer has its own state — the scheduler's Function registry holds the pending approval with the buffered IO request (see `docs/design_functions.md`).

Sub-agent / fork / compact kinds: not built. There's no `ThreadKind` enum, no `parent: Option<ThreadId>`. When those features land they'll introduce the types they need; we're not pre-committing to an enum shape without the features to validate it.

## Scheduler

A single tokio task running one `select!` loop (see `src/runtime/scheduler.rs::run`). Branches:

- **Inbox** — client frames from the WebSocket layer, plus client-registration/unregistration.
- **Pending I/O completions** — model responses, MCP tool calls, host-env provisioning, tool-list fetches.
- **GC tick** — periodic sweep that tears down idle resources (host-env sandboxes unused for 5 min, torn-down entries evicted after 1 hour; constants at the top of `scheduler.rs`).

Each event follows the same three-step shape:

1. `apply_*` — mutate thread state, collect `ThreadEvent`s from the state machine.
2. `router.dispatch_events` — translate task events to wire events and broadcast to subscribers.
3. `step_until_blocked` — keep calling `thread.step()` until the thread requests I/O or pauses; push fresh I/O futures to the `FuturesUnordered`.

Tasks-as-data discipline: thread mutation happens *only* through `thread.step()` and `thread.apply_io_result()`. The scheduler is the single writer; every other code path is read-only. This is what makes persistence and crash recovery tractable.

## Server catalog

Pod TOMLs reference backends, MCP hosts, and host-env providers *by name*. Names are defined in the server's top-level TOML (today: the file the `whisper-agent serve` command points at — typically `whisper-agent.toml`):

```toml
default_backend = "anthropic"

[backends.anthropic]
kind = "anthropic"
default_model = "claude-sonnet-4-6"
auth = { mode = "api_key", env = "ANTHROPIC_API_KEY" }

[backends.local-llama]
kind = "openai_chat"
base_url = "http://localhost:11434/v1"
default_model = "llama3.2"

[shared_mcp_hosts]
fetch  = "http://127.0.0.1:9831/mcp"
search = "http://127.0.0.1:9832/mcp"

[[host_env_providers]]
name = "local-landlock"
url  = "http://127.0.0.1:9840"
```

The catalog defines what the server *can* run. A pod's `[allow]` table picks a subset. Threads bind to specific entries.

A pod that references a catalog entry the server doesn't know warns at load time; the missing entry is excluded from the effective `[allow]`.

## Resource registries

Three flat `HashMap`s in the scheduler's `ResourceRegistry` (`src/pod/resources.rs`): `BackendEntry`, `McpHostEntry`, `HostEnvEntry`. Each carries a `ResourceState` (`Provisioning | Ready | Errored | TornDown`), a refcount (threads currently bound), and a `pinned` flag.

Two behaviors at the resolver layer:

- **Pod allowlist enforcement.** When a thread requests a binding, the resolver verifies the requested resource (or the spec to auto-provision) is in the pod's `[allow]` table. Not allowed → request rejected with a wire error.
- **Spec dedup across pods.** Two pods with identical inline host-env specs share the same provisioned sandbox in the registry. Spec equality (via `HostEnvId::for_provider_spec`) is the dedup key, not the pod boundary.

GC: refcount + idle timeout + pin. Ready resources with zero users for 5 minutes are torn down. Torn-down / errored entries linger for an hour for inspection ("yes, that host-env existed and failed because…") then evict. Backends never GC — they're cheap handles.

## Auto-provisioning

A `ResourceResolver` sits between thread creation/rebind and the registries. When a thread wants `bindings.host_env = Named { name: "default" }`:

```
1. Look up "default" in the pod's [[allow.host_env]] entries.
   Reject if absent (pod doesn't allow it).
2. Compute HostEnvId::for_provider_spec(provider, spec) — the dedup key.
3. Look for an existing Ready HostEnv in the registry with that id.
   Match → bind the existing id.
4. No match → enqueue CreateHostEnv, bind the new id, thread enters
   WaitingOnResources until provisioning completes.
```

Same logic for MCP hosts (matched by catalog name + url) and backends (matched by catalog name).

The resolver is the only code path that spawns resources implicitly. UI-driven `CreateHostEnv` requests bypass it (those create unscoped resources, used for diagnostic or shared infrastructure).

## Mid-thread rebinding

```rust
ClientToServer::RebindThread {
    thread_id: ThreadId,
    patch: ThreadBindingsPatch,   // backend / host_env / mcp_hosts
}
```

Applied immediately. Patches are validated against the pod's `[allow]` table — same rules as initial binding. Future I/O ops use the new bindings; in-flight ops complete against their original resources (already captured in the future). Refcounts adjust: old resources `users.remove(thread_id)`, new ones `.insert(thread_id)`.

When bindings change materially, the scheduler appends a synthetic system message to the conversation:

> *Note: the execution environment changed at this point. Files, processes, and tool availability may differ from earlier in this conversation.*

Without this marker, the model continues referencing files and outputs from the old environment with no awareness that they may not exist.

## Wire protocol

### Client → Server

```rust
enum ClientToServer {
    // Pod lifecycle
    ListPods         { correlation_id },
    GetPod           { correlation_id, pod_id },
    CreatePod        { correlation_id, pod_id, config: PodConfig },
    UpdatePodConfig  { correlation_id, pod_id, toml_text },
    ArchivePod       { pod_id },

    // Thread lifecycle
    CreateThread     { correlation_id, pod_id, initial_message: Option<String>,
                       bindings: Option<ThreadBindingsRequest>,
                       config_override: Option<ThreadConfigOverride> },
    SendUserMessage  { thread_id, text },
    CancelThread     { thread_id },
    RebindThread     { thread_id, patch: ThreadBindingsPatch },
    ApprovalDecision { thread_id, approval_id, choice, remember },

    // Subscription (two tiers)
    SubscribeToPod    { pod_id },           // pod-detail tier (thread summaries)
    SubscribeToThread { thread_id },        // turn tier (assistant text, tool calls, approvals)
    UnsubscribeFromPod    { pod_id },
    UnsubscribeFromThread { thread_id },

    // Resource registry
    ListResources    { correlation_id },
    CreateHostEnv    { spec, pinned },       // unscoped, UI-driven
    DestroyHostEnv   { id, force },
    PinResource      { resource_ref, pinned },

    // Server catalog (read-only)
    ListBackends     { correlation_id },
    ListMcpHosts     { correlation_id },
    ListModels       { correlation_id, backend },
}
```

### Server → Client

```rust
enum ServerToClient {
    // Pod-list tier (broadcast)
    PodList            { correlation_id, pods },
    PodCreated         { pod, correlation_id },
    PodConfigUpdated   { pod_id, toml_text, parsed, correlation_id },
    PodArchived        { pod_id },

    // Pod-detail tier (subscribers of that pod)
    PodSnapshot        { pod_id, snapshot },
    ThreadCreated      { pod_id, thread_id, summary },
    ThreadStateChanged { pod_id, thread_id, state },
    ThreadBindingsChanged { thread_id, bindings },

    // Per-thread turn tier (subscribers of that thread)
    ThreadAssistantBegin / Text / Reasoning / End   { thread_id, ... },
    ThreadToolCallBegin / End                        { thread_id, ... },
    ThreadPendingApproval / ApprovalResolved         { thread_id, ... },
    ThreadLoopComplete                               { thread_id },
    ThreadAllowlistUpdated                           { thread_id, allowlist },

    // Resource tier (broadcast — registry is small)
    ResourceList        { correlation_id, resources },
    ResourceCreated / Updated / Destroyed { ... },

    // Catalog responses
    BackendsList        { correlation_id, default_backend, backends },
    McpHostsList        { correlation_id, hosts },
    ModelsList          { correlation_id, backend, models },

    Error { correlation_id, pod_id, thread_id, message },
}
```

Three subscription tiers: pod-list (cheap, broadcast), pod-detail and thread-turn (per-subscription), resource (broadcast because the registry is small and every UI wants it).

## Persistence

Pods are persisted as their directories. Writing a pod = writing `pod.toml` and the `threads/*.json` files. Reading a pod = parsing those.

- **`pod.toml`** is rewritten on `UpdatePodConfig` / `CreatePod`. Hand-edits on disk are picked up on the next scheduler restart; there's no file watcher. `raw_toml` is kept in memory so the wire can return it verbatim without re-serializing.
- **`threads/<thread_id>.json`** is rewritten on every thread state transition. The flush batches within a single scheduler-loop iteration via the `dirty` set — threads that transitioned this tick get one write, not one-per-step.
- **No SQLite, no aggregate index file.** `ListPods` walks `<pods_root>` (and `.archived/` if requested). At expected scale (≤ a few hundred pods) the cost is negligible.

On startup the scheduler reads the server config, walks `<pods_root>` and loads every pod's TOML + threads. Threads in non-terminal non-idle states (`AwaitingModel`, `AwaitingTools`, `WaitingOnResources`) are flipped to `Failed { at_phase: "resume" }` — the in-flight model / tool calls can't be resumed across a restart, so the user restarts the turn if they want.

## CLI one-shot

The CLI uses `/tmp/whisper-agent-cli-<pid>/` as its `pods_root`. Inside, it creates one pod (named `cli`), creates one thread, runs to terminal state, and prints the conversation. The directory persists for post-run inspection; OS `/tmp` cleanup eventually removes it. A `--pods-root <path>` flag overrides this for users who want CLI runs to share a long-lived pod.

## Not in scope (reserved for later)

- **Sub-agent spawning.** The data model doesn't preclude it — a thread could transition to a `AwaitingChildThread` internal state and child threads could carry a `parent_thread_id`. The actual machinery (a `spawn_thread` tool, child-completion → parent-resume plumbing, approval inheritance) isn't built.
- **Fork (rewind a thread from a midpoint).** Same situation: the pod-directory layout accommodates it, nothing is wired up.
- **Compact (replace a thread's conversation with a summary).** Also deferred.
- **Lua hooks on thread lifecycle events.** The `pod.toml` has no `[[hooks]]` table yet; the design leaves room for one without committing to a scripting model.
- **Autonomous behaviors** (scheduled or event-driven threads). Separate design: [`design_behaviors.md`](design_behaviors.md).
- **Pod self-modification via tool.** Builtin tools can edit pod config today (`PromptPodModify` gates them); a more ergonomic patch-style interface is deferred.
- **Concurrent server access to the same `pods_root`.** Single-server-instance assumption holds. File locking, lease coordination, etc. are future work.
- **Real cancellation of in-flight tools.** `Cancelled` marks the thread; in-flight futures complete and discard. Proper cancellation (tool rollback, abort-after-timeout) is deferred.
- **Pod-level quota / rate-limiting.** `[limits]` reserves space; only `max_concurrent_threads` is concrete. Token/cost budgets land when there's a concrete enforcement path.
- **File watcher on `pod.toml`.** Hand-edits require restart today; adding `notify`-based live reload is straightforward when the need arrives.
- **Auth.** Loopback-only. Remote principals and per-identity policy are design_permissions.md Pattern 3 territory.

## Open questions

1. **Pod TOML parse-error recovery.** Today: a pod with an unparseable `pod.toml` fails to load and is invisible. Alternative: show it in the UI as `Errored` with the parse error, threads read-only until fixed. Revisit if pod-config editing becomes a common activity.
2. **GC policy.** 5-minute host-env idle timeout, 30-minute MCP-host timeout, backends never. Pin opt-out. Could revisit if autonomous-behavior patterns make different cadences natural.
3. **Mid-thread swap warning.** Today: automatic synthetic system message. Could instead require the user or orchestrator to insert their own marker — revisit if the automatic one ever gets in the way.
4. **Default thread on new pod.** A new pod is created empty; the user explicitly creates the first thread. Could auto-create a root thread — hasn't come up as friction.
