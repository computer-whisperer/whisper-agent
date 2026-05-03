# Host-Env Protocol (v2)

This document describes the second-generation host-env subsystem: how the
scheduler talks to remote sandbox daemons, how per-thread isolation and runtime
configuration are expressed on the wire, and how background processes
(long-running bash, watch-mode commands, etc.) are surfaced as first-class
state rather than hidden inside MCP tool calls.

It supersedes the host-env-specific portions of
[`design_pod_thread_scheduler.md`](design_pod_thread_scheduler.md) — the
resource-registry shape, auto-provisioning, dedup, and the
`[[host_env_providers]]` catalog — while leaving the broader pod/thread/resource
layering it describes intact.

Companion docs: [`design_pod_thread_scheduler.md`](design_pod_thread_scheduler.md),
[`design_permissions.md`](design_permissions.md),
[`design_behaviors.md`](design_behaviors.md).

## Why a rework

The v1 design treated each provisioned sandbox as "just another MCP server":
the `whisper-agent-sandbox` daemon forked a `whisper-agent-mcp-host` child
inside a landlock jail, the scheduler opened an HTTP+JSON-RPC session to it,
and the host's tool catalog merged into the thread's regular MCP tool surface.
That was the right scaffold to get isolation working, but it has aged into
several structural blockers:

1. **No place to express per-thread runtime configuration.** The MCP wire is
   `tools/list` + `tools/call`. There's no frame for "this thread runs `bash`
   as user X with env Y, with `view_pdf` denied"; that information has nowhere
   to live without either bending the MCP schema or shoving everything into
   the `HostEnvSpec` (which forces a reprovision per-permutation).

2. **Spec dedup vs per-thread state are in direct conflict.** The
   `HostEnvId::for_provider_spec` dedup means two threads with the same spec
   share a host. The moment any per-thread state diverges, dedup breaks.

3. **The IPC stack pays for an interop story that doesn't exist.** HTTP +
   bearer-on-stdin + SSE-or-JSON-by-content-type + the `rewrite_mcp_host`
   dance all exist because we said the inside is MCP. We never point external
   MCP clients at our hosts; the wire complexity buys us nothing.

4. **Background bash is unrepresentable.** Claude Code and similar agents can
   start a long-running bash process, leave it running, check on it later,
   kill it. The MCP `tools/call` shape is request/response — there's no
   first-class "running task" the UI can show, no place for the scheduler to
   track a process across multiple model turns.

5. **Plain HTTP + a static bearer file is structurally weak.** The bearer
   crosses the LAN in cleartext. Operators rotate by editing a file and
   restarting. There's no good answer for daemons on roaming hosts (laptops,
   home machines behind NAT, dev boxes with dynamic IPs).

6. **Catalog churn from polling reachability.** The scheduler polls every
   provider's `/health` every 30 s. State machine, broadcast on transitions,
   lots of moving parts to model "is the daemon up." The right answer is
   "you have a live connection to it, or you don't."

## Architecture in one paragraph

A daemon process runs on each host that should provision sandboxes. On
startup it dials home over HTTPS to whisper-agent and opens a multiplexed,
bidirectional WebSocket carrying CBOR-framed RPCs. The scheduler asks that
WebSocket to **open sessions** (each session is one provisioned sandbox for
one thread); within a session, it issues **tool calls** that stream chunks
back and **background-task** primitives that outlive a single call.
Per-thread runtime configuration (env, runas, denylist) rides in the session
context and can be mutated mid-flight. Sessions are not deduplicated across
threads — landlock is cheap; dedup management was not. Reachability is the
existence of the WebSocket; tokens authenticate the daemon at WS upgrade
against a dedicated `[[auth.daemons]]` table.

## Connection direction and security model

**Daemon dials whisper-agent.** Whisper-agent already terminates HTTPS with a
real cert (Let's Encrypt or operator-managed); daemons authenticate it via
the standard cert chain. This shifts the firewall/NAT story to the simpler
side: only whisper-agent needs an open inbound port, daemons can run anywhere
with outbound HTTPS.

The connection upgrades from `GET /v1/host_env_link HTTP/1.1` to a WebSocket.
The daemon presents `Authorization: Bearer <daemon-token>` on the upgrade.

```toml
# whisper-agent.toml — separate from [[auth.clients]] / [[auth.admins]],
# because daemon trust is a different surface from chat-client trust.
[[auth.daemons]]
name  = "alpha"
token = "REPLACE_WITH_LONG_RANDOM_STRING"
# Reserved for future per-daemon caps:
# max_sessions = 16
# allowed_spec_kinds = ["landlock"]
```

On accepted upgrade the scheduler registers the connection in a
`LiveDaemonRegistry` keyed by the name resolved from the token. From that
point on, "is daemon `alpha` reachable?" is a synchronous lookup — no probes,
no `mark_reachable`/`mark_unreachable` ticking, no 30-second cadence. The
absence of the connection *is* the unreachability.

### Conflict policy

A new connection arriving for a name that already has a live connection:

1. Scheduler sends `Ping` with a short deadline to the existing connection.
2. Existing responds → scheduler rejects the new connection with
   `Goodbye { reason: NameAlreadyConnected }`.
3. Existing times out → scheduler declares the old connection dead, accepts
   the new one, and emits `BackgroundTasksLost` events for the UI (the new
   daemon process won't have the old one's running tasks; users should see
   it).

This handles both the "operator restarted the daemon" and "split-brain
network partition" cases without surprises.

### Why not Noise / mTLS

Earlier sketches for v2 considered Noise (XK pattern, fingerprint-pinned
catalog) or mTLS. Both became unnecessary once the connection direction
flipped: the dispatcher's TLS cert provides authenticity for free, and the
daemon-side bearer covers the other direction at the application layer. We
also drop the per-sandbox bearer that v1 bounced from the daemon back to the
scheduler — there's no separate MCP wire to authenticate; the worker child
inside the daemon is reachable only via a daemon-owned Unix socket.

## Wire protocol

A new crate `whisper-agent-host-proto` owns the type definitions. It does not
ship implementations of either side — both the daemon-side dialer and the
scheduler-side server consume it as a shared protocol.

### Frame envelope

A single CBOR-tagged enum carries every frame. Direction (S→D vs D→S) is
documented per variant; we don't split into two enums because the asymmetry
is small and a single union is easier to log, route, and evolve. Length-
prefixed inside the WS data frames.

```rust
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum Frame {
    // ─── Connection ─────────────────────────────────────
    Hello {                                    // D→S, first frame after upgrade
        daemon_version: String,
        protocol_version: u32,
        capabilities: DaemonCapabilities,
    },
    Welcome {                                  // S→D, response to Hello
        scheduler_version: String,
        protocol_version: u32,
    },
    Goodbye {                                  // either direction; carries reason
        reason: GoodbyeReason,
        message: String,
    },
    // Heartbeats use WS-level Ping/Pong frames (RFC 6455 §5.5.2/3),
    // not application-layer frames. The implementation closes the
    // connection if no Pong arrives within `2 * heartbeat_interval`.

    // ─── Session lifecycle ──────────────────────────────
    OpenSession {                              // S→D
        session_id: SessionId,                 // scheduler-assigned (UUID)
        thread_id: String,                     // for daemon-side logging only
        spec: HostEnvSpec,                     // landlock paths + network policy
        context: ThreadContext,                // env, runas, denylist, etc.
    },
    UpdateSession {                            // S→D, mid-flight context change
        session_id: SessionId,
        context_delta: ThreadContextDelta,
    },
    CloseSession { session_id: SessionId },    // S→D

    SessionReady  { session_id: SessionId },   // D→S, sandbox is up
    SessionFailed {                            // D→S, provision failed
        session_id: SessionId,
        phase: ProvisionPhase,
        message: String,
    },
    SessionClosed {                            // D→S, session ended
        session_id: SessionId,
        reason: SessionEndReason,              // RequestedByScheduler |
                                               // WorkerExited | DaemonShutdown
    },

    // ─── Per-call (within a session) ────────────────────
    InvokeTool {                               // S→D
        session_id: SessionId,
        call_id: CallId,                       // scheduler-assigned, monotonic
        tool_name: String,
        arguments: Value,
    },
    CancelCall { session_id: SessionId, call_id: CallId },  // S→D

    ToolChunk {                                // D→S, streaming output
        session_id: SessionId,
        call_id: CallId,
        block: ContentBlock,
    },
    ToolFinal {                                // D→S, terminal frame for a call
        session_id: SessionId,
        call_id: CallId,
        result: CallToolResult,
    },

    // ─── Background tasks ───────────────────────────────
    BackgroundTaskUpdate {                     // D→S, state change or new output
        session_id: SessionId,
        task_id: BackgroundTaskId,             // daemon-assigned (opaque)
        state: BackgroundTaskState,
        bytes_available: u64,                  // cumulative buffered output
        last_output_at: Timestamp,
    },
    ListBackgroundTasks {                      // S→D
        session_id: SessionId,
    },
    BackgroundTaskList {                       // D→S, response
        session_id: SessionId,
        tasks: Vec<BackgroundTaskSummary>,
    },

    // ─── Future extension point ─────────────────────────
    HookEvent {                                // D→S, out-of-band event
        session_id: SessionId,
        kind: HookKind,
        payload: Value,
    },
}
```

### Identifiers

- **`SessionId`** — UUID, scheduler-assigned. The scheduler is the source of
  truth for session identity; daemons never invent one. This makes reconnect
  semantics tractable — a returning daemon cannot accidentally collide with
  a session the scheduler still considers live.
- **`CallId`** — `u64`, scheduler-assigned, monotonic per session. Cheap and
  debuggable in logs.
- **`BackgroundTaskId`** — opaque string, daemon-assigned. The scheduler
  treats it as a token and never parses it.

### Capabilities and tool catalog

`Hello.capabilities` carries the daemon's full advertised surface:

```rust
pub struct DaemonCapabilities {
    /// Tool descriptors the daemon's compiled-in worker exposes. Schema is
    /// the same shape the scheduler already consumes from MCP `tools/list`,
    /// so existing surface (allowlist filtering, model presentation) reuses.
    pub tools: Vec<ToolDescriptor>,

    /// Sandbox spec kinds this daemon can actually fulfill. Today this is
    /// `[Landlock]`; container support adds `Container`.
    pub spec_kinds: Vec<HostEnvSpecKind>,

    /// Optional cap on concurrent sessions the scheduler should not exceed.
    /// `None` means unbounded (scheduler still applies its own ceilings).
    pub max_concurrent_sessions: Option<u32>,

    /// Whether the daemon supports the `BackgroundTask*` frames. Until this
    /// is true, the scheduler hides background-mode tools from threads
    /// bound to this daemon.
    pub supports_background_tasks: bool,
}
```

Tools change rarely (they're compiled into the daemon binary), so a one-shot
advertisement at connect is enough. If we later need per-session catalogs
(for hook-loaded extension tools, etc.), we add a `ListTools` frame then.

This is the key design point that lets daemon and scheduler have **mismatched
versions**: as long as `Frame` itself is wire-compatible, the daemon's tool
set, spec capabilities, and background-task support are all *advertised*,
not assumed. Operators can roll out a new daemon version on one host while
others stay back; the scheduler discovers and adapts.

## Sessions

A session is one provisioned sandbox. The scheduler opens one per
`(thread, host_env_binding)` pair — there is no dedup. Spec equality is no
longer a registry key; landlock provisioning is cheap enough that the
management complexity of dedup is not worth its cost.

### Lifecycle

```
            OpenSession (S→D)
                 │
                 ▼
      ┌─── provisioning ───┐
      │                    │
      ▼                    ▼
 SessionReady          SessionFailed
      │                    │
      ▼                    ▼
   ready ──InvokeTool* ─→ done
      │
      ▼
  CloseSession (S→D)
      │
      ▼
  SessionClosed (D→S)
```

`SessionReady` is the daemon confirming the sandbox is provisioned and the
worker is ready to take tool calls. `SessionFailed` carries the phase
(`PreExec | Landlock | WorkerSpawn | WorkerHandshake`) so failures surface
the right diagnostic at the UI.

`UpdateSession` may be sent any time after `SessionReady`; the daemon applies
the delta before dispatching the next tool call. Concurrent in-flight calls
see the pre-update context — the daemon does not retroactively re-apply.

`CloseSession` from the scheduler tears down the worker (signal, wait,
collect exit) and replies with `SessionClosed`. A daemon-initiated close
(worker crashed, daemon shutting down) sends an unsolicited `SessionClosed`.
Either way, all background tasks owned by the session are terminated.

### Thread context

```rust
pub struct ThreadContext {
    /// Override the daemon's default workspace root for this session.
    /// `None` means "use the spec's writable allowed_paths root."
    pub workspace_root: Option<PathBuf>,

    /// Environment variables for processes the worker spawns (bash, etc.).
    pub env: BTreeMap<String, String>,

    /// Drop to this user before exec in bash. `None` means daemon's uid.
    pub runas: Option<String>,

    /// Tools the daemon refuses to invoke for this session. Hard refusal
    /// (defense-in-depth on top of scheduler-side allow filtering).
    pub tool_denylist: BTreeSet<String>,

    /// Bash hard timeout. `None` means daemon default.
    pub bash_timeout_secs: Option<u32>,

    /// Cap on streamed `ToolChunk` bytes per call (head + tail truncation).
    pub output_byte_cap: Option<usize>,

    // Reserved for future hook/behavior integration.
}
```

`ThreadContextDelta` mirrors this with the standard `Option<Option<T>>`
"absent = leave alone, `Some(None)` = clear" pattern for partial updates.

What's deliberately **not** in `ThreadContext`:

- **Landlock paths / network policy.** Those are `HostEnvSpec`. Changing
  them requires re-landlocking the worker, which means tearing down and
  reprovisioning — that's what `OpenSession` already is. `UpdateSession` is
  for things changeable without re-landlocking.
- **Tool *allowlist*.** The scheduler enforces this before the call ever
  leaves; the daemon never sees disallowed calls and doesn't need a list.
  `tool_denylist` is specifically a daemon-enforced backstop.

## Tool calls

Foreground call shape:

```
S→D: InvokeTool { session, call, name, args }
D→S: ToolChunk { session, call, block }   (zero or more, only for streaming tools)
D→S: ToolFinal { session, call, result }  (exactly one; terminates the call)
```

Cancellation:

```
S→D: CancelCall { session, call }
D→S: ToolFinal { session, call, result: CancelledByCaller }   (terminal as usual)
```

The daemon may not produce a `ToolFinal` instantly on cancel — there's a
graceful window for the worker to interrupt the underlying operation. The
scheduler treats the eventual `ToolFinal` as authoritative; the model sees
exactly one outcome per call.

## Background tasks

Background tasks are how Claude-Code-style "start a long bash, leave it
running, check in later" works. They are owned by a session: when the
session closes, all its background tasks die.

### How the model interacts

The model sees ordinary tools — there is no special "protocol" the model
knows about:

- `bash` with `background: true` returns immediately with `{ task_id, started_at }`.
- `bash_check { task_id, since_offset }` reads incremental output.
- `bash_kill { task_id, signal }` signals the task.
- `bash_list` lists active tasks in this session.

These are normal `InvokeTool` calls; the daemon implements them as
operations on its in-process task table.

### What the wire carries

The scheduler needs visibility into the task table for two reasons: surfacing
running tasks in the UI, and resyncing on reconnect. So in addition to the
tool calls (which carry the model-visible output), the daemon emits
**state-change summaries** out of band:

```rust
BackgroundTaskUpdate {
    session_id, task_id,
    state: Running | Exited { code, signal } | Killed { reason },
    bytes_available: u64,
    last_output_at: Timestamp,
}
```

These are rate-limited (not one per stdout byte; coalesced to e.g. 500 ms
windows) — they're for status, not for transferring output. The actual
output bytes go through `bash_check` like any other tool result.

The scheduler maintains a `BackgroundTaskTable` per session. UI subscribers
get `ServerToClient::BackgroundTaskCompleted` / `…Updated` events derived
from the daemon's frames. Threads can be queried for "what's still running."

### Resync on reconnect

When the WebSocket comes back after a drop, the scheduler issues
`ListBackgroundTasks` for every still-live session on this daemon. The
daemon replies with `BackgroundTaskList` and the scheduler reconciles —
dropping tasks the daemon no longer reports, updating state for tasks it
does. From the model's perspective, in-flight `bash_check` calls failed
during the disconnect window (per the disconnect policy below); after
resync, fresh `bash_check` calls work normally.

### Daemon orphan policy

Background tasks survive WebSocket drops by default — killing a user's bash
because their network blipped is hostile. Daemons should expose an
operator-side option to "kill all background tasks if no scheduler reconnects
within N minutes" so unattended crashes don't accumulate orphan processes
forever. Out of scope for v1; flagged for the daemon implementation.

## Disconnect policy

Per-thread, declared on `ThreadConfig` (and inheritable from `ThreadDefaults`
in `pod.toml`):

```rust
pub enum HostEnvDisconnectPolicy {
    /// Active call returns an error to the model immediately; thread
    /// continues. Best for interactive threads where progress matters
    /// more than not losing partial work.
    ContinueWithWarning,

    /// Active call returns an error; thread enters WaitingOnResources
    /// until the daemon reconnects, then unblocks. The model still sees
    /// the error — we are not silently retrying. Best for background /
    /// behavior-driven threads where waiting beats surfacing transient
    /// failures.
    PauseUntilReconnect,
}
```

Two important non-features:

- **Not on the wire.** The daemon does not need to know which mode a thread
  is in. The scheduler observes WS close (via WS-level pong timeout) and
  applies the policy locally to the threads holding sessions on that daemon.
- **Not "replay the in-flight call on reconnect."** A richer mode that holds
  the call mid-flight and replays it on reconnect would require daemon-side
  reattach with call-state durability. Designed-around (the wire frames
  don't preclude it), but not shipped in v1.

If a thread holds bindings to multiple daemons and only one disconnects,
only the sessions on the affected daemon transition; bindings to other
daemons keep working. The thread's overall state reflects the union.

## Catalog

Daemons are admission-controlled by name + token in `[[auth.daemons]]`.
There is no separate `[[host_env_providers]]` table in the new world — the
admission entry *is* the catalog entry. Pods reference daemons by name in
`[[allow.host_env]].provider`, exactly as today.

```toml
# whisper-agent.toml
[[auth.daemons]]
name  = "local-landlock"
token = "..."

[[auth.daemons]]
name  = "alpha"
token = "..."
```

`[[host_env_providers]]` is gone; `host_env_providers.toml` (the durable
runtime catalog file) is gone. There's no longer a "URL to dial" — the
daemon either connects in or it doesn't. CLI overlays for dispatcher-side
provider configuration retire with it.

The runtime registry the scheduler maintains is a `LiveDaemonRegistry`:

```rust
pub struct LiveDaemonRegistry {
    /// Currently connected daemons keyed by name.
    connected: HashMap<String, LiveDaemonHandle>,
    /// Names admitted by [[auth.daemons]] but not currently connected.
    admitted_offline: HashSet<String>,
}
```

`UpdatePodConfig` validation: a pod's `[[allow.host_env]].provider` must
reference a name in either bucket — admission, not liveness, is the gate.
Threads bound to an offline daemon enter `WaitingOnResources` until it
connects (analogous to how a thread today blocks if the sandbox daemon is
unreachable).

## Resource registry simplifications

With dedup gone, the changes to `src/pod/resources.rs`:

- `HostEnvId::for_provider_spec` — gone. Sessions are keyed by
  `(thread_id, binding_name)` directly.
- `pre_register_host_env` / `pre_register_host_env_mcp` /
  `complete_host_env_provisioning` / `CompleteHostEnvOutcome` /
  `provisioning_in_flight` set / unused-handle cleanup — most goes away.
  What remains is a per-thread provisioning state ("is the OpenSession in
  flight, ready, failed, closed").
- `HostEnvEntry.refcount` / `pinned` — gone. Sessions exist exactly as long
  as the thread holds them.
- Idle GC of host envs (5-minute timeout, 1-hour eviction of torn-down) —
  gone. Sessions die on thread close.
- Spec-equality logic across pods — gone.

The host-env entry shape becomes a per-thread session record, much smaller:

```rust
pub struct ThreadSession {
    pub session_id: SessionId,
    pub daemon_name: String,
    pub binding_name: String,
    pub state: SessionLifecycleState,    // Provisioning | Ready | Failed | Closed
    pub background_tasks: HashMap<BackgroundTaskId, BackgroundTaskSummary>,
}
```

`McpHostId::for_host_env` and the dual MCP-host-mirror entry it created
(see today's `pod/resources.rs`) also retire — there's no separate MCP host
to track, since the host-env tools come through this protocol directly.

## Migration phasing

The cutover was staged so existing remote daemons kept working through it.
**All phases below have shipped.** The v1 path (HTTP+JSON-RPC, the
`whisper-agent-sandbox` crate, `[[host_env_providers]]` TOML, the
dedup machinery) was retired in phase 7b and is no longer in the codebase
or wire — operators on v1 must move to `[[auth.daemons]]` plus a running
`whisper-agent-host-daemon`.

1. **Protocol crate** *(landed).* `whisper-agent-host-proto` ships with the
   `Frame` enum, supporting types, CBOR round-trip tests, and a
   connection-lifecycle doc comment. Pure types crate — no wire
   implementations.

2. **Scheduler-side wire endpoint** *(landed).* The scheduler mounts
   `/v1/host_env_link` over its existing TLS endpoint. Daemons dial in,
   present a bearer admitted by `[[auth.daemons]]`, complete the
   Hello/Welcome handshake. `LiveDaemonRegistry` tracks admitted vs. live
   names; `LiveDaemonHandle` and `SessionHandle` give consumers a clean
   API. The handle layer exists; **threads do not yet route through it**
   (that's phase 4).

3. **Daemon binary** *(landed).* New `whisper-agent-host-daemon` crate
   (forked from `whisper-agent-sandbox`'s landlock provisioning, not
   replacing it). Dials home over WS, runs OpenSession by spawning a
   landlock'd `whisper-agent-mcp-host` worker (loopback HTTP + per-session
   bearer — same machinery as v1, kept intentionally for now), proxies
   `tools/call` as `InvokeTool` → `ToolFinal`. Tool catalog discovered by
   probing the worker once at startup (deviation from the original
   "compiled in" plan — single source of truth, see
   `crates/whisper-agent-host-daemon/src/catalog.rs` for the why). The
   "daemon-owned Unix socket worker IPC" rewrite the original phase 3
   sketched is invisible at the wire and was deferred.

4. **Wire threads to v2 daemons** *(landed).* User-visible value: a thread
   bound to a v2 daemon name actually invokes its tools. Dispatch landed in
   four sub-steps:

   - **4a. Binding resolution + tool catalog merge.** v2 names resolve in
     scheduler binding lookup (currently the seed loop in `main.rs` skips
     `V2Ws` entries; that needs to flip to "skip v1-style runtime catalog
     seeding but accept the name as a v2 binding target"). `DaemonCapabilities.tools`
     from each bound daemon is merged into the thread's visible tool
     catalog at bind time — same shape the existing MCP tools/list already
     produces, so the model-presentation path doesn't change.
   - **4b. Per-thread `V2SessionHandle` lifecycle.** One session per
     `(thread, v2-binding)` pair, opened on first tool call (lazy avoids
     spawning workers for threads that never use the binding) and closed
     on thread teardown. Link-down surfaces as a tool error today; the
     pause-vs-warn policy lands in phase 5.
   - **4c. `io_dispatch` v2 arm.** The tool-call dispatcher checks whether
     a binding is v2 and routes through `SessionHandle::invoke_tool()`
     instead of the existing MCP/HostEnv path. v1 dispatch stays as-is —
     no shared code paths, no risk of regressing the v1 flow.
   - **4d. End-to-end integration test.** A thread bound to a v2 daemon
     invokes a tool; assert the `ToolFinal` propagates back to the
     model-visible result. One test against a real `LiveDaemonRegistry`
     and a mock daemon — the unit-level dispatch tests cover the matrix.

   Per-thread sessions are inherent to v2 (each `OpenSession` carries a
   fresh `SessionId`); there is no dedup machinery on the v2 side to
   remove. The legacy `HostEnvId::for_provider_spec` / `provisioning_in_flight`
   stays in place for the v1 path until phase 7 deletes it wholesale —
   cleaning v1 internals before deletion is churn for no shipped value.

5. **Per-thread control surface** *(landed).* `UpdateSession` with
   denylist / runas / env / output cap. UI surfaces for editing thread
   context. This is the feature work that motivated the rework. The
   per-thread disconnect policy (`pause_until_reconnect` vs
   `continue_with_warning`) landed here as one of the editable fields.

6. **Background tasks** *(landed at the IPC layer; user-visible tools
   pending).* Phase 6.0a/b ship `whisper-agent-worker-proto` and the
   daemon ↔ worker Unix-socketpair IPC that carries `BackgroundTask*`
   frames end-to-end. The user-facing pieces — a `bash` `background`
   argument, the `bash_check` / `bash_kill` / `bash_list` tools, and
   the UI surfacing of active background tasks — are tracked separately
   from this protocol effort.

7. **Retire the legacy path** *(landed across 7a + 7b).* 7a deleted the
   `whisper-agent-sandbox` crate. 7b finished the job in the dispatcher:
   `[[host_env_providers]]` TOML catalog, `HostEnvProviderKind`,
   `--host-env-provider` / `--host-env-provider-token` CLI overlays,
   `HostEnvId::for_provider_spec`, `provisioning_in_flight`,
   `McpHostId::for_host_env`, the v1 `io_dispatch` arms, the HTTP-out
   `DaemonClient`, and the v1 webui surfaces (provider editor modal,
   host-env-providers settings tab). `whisper-agent-mcp-host` was kept
   under its existing name — renaming it to `…-host-worker` was floated
   but deferred since the binary contract didn't change.

The `whisper-agent.toml` migration is hard: configs with
`[[host_env_providers]]` stop parsing as of 7b, and operators must move
to `[[auth.daemons]]` plus a running `whisper-agent-host-daemon`.

## Not in scope for v1

- **Sticky sessions across reconnect.** A returning daemon's existing
  sessions are not resumed; the scheduler treats them as gone and the
  thread's policy decides what to do. Reattach with call-state durability
  is a real feature for behavior-driven threads but a separate effort.
- **Daemon-on-demand spawning.** Daemons must be running and connected
  before a thread can bind to them. We do not SSH into hosts to start them.
- **Lua hooks / behavior glue.** The `HookEvent` frame is a stub; what hooks
  emit and how they bind to behaviors is `design_behaviors.md` territory.
- **Container provisioning.** `HostEnvSpec::Container` stays the placeholder
  it is today. Adding it does not require protocol changes — `spec_kinds`
  in `Hello.capabilities` already advertises what the daemon supports.
- **Multiple workers per session.** One session = one worker. If we ever
  want a session that exposes multiple isolated subprocesses (e.g. a tools
  worker + a separate sandbox for arbitrary bash), we revisit.
- **Per-session tool catalogs.** All tools come from `Hello.capabilities`;
  every session sees the same set (filtered by `tool_denylist`). Per-session
  tool injection (for hook-loaded extensions) is a future `ListTools` frame.

## Open questions

1. **`HookEvent` payload schema.** Left as `Value` until the behavior
   integration is concrete enough to pin down. When that lands, we either
   formalize `HookKind` as an enum or split into typed frames.

2. **Daemon orphan-process policy.** Daemons keep background tasks running
   across WS drops; operators may want a per-daemon ceiling on how long an
   unattended task lives. Default and configuration shape TBD when we
   actually implement it — flagging the requirement here so phase 6 doesn't
   land without it.

3. **Per-binding vs per-thread disconnect policy.** Currently per-thread.
   If a thread has bindings to several daemons and the user wants different
   policies per binding (pause for the slow remote daemon, continue for the
   fast local one), revisit. Easy to extend without breaking existing
   `ThreadConfig` users.

4. **Multiple whisper-agent instances.** A daemon can dial home to N
   instances simultaneously — the protocol does not preclude it. Useful for
   dev/prod splits or HA. No code path uses it day one, but we shouldn't
   accidentally rule it out in the daemon's connection management.

5. **Session resume across scheduler restarts.** Today's threads-in-flight
   are flipped to `Failed { at_phase: "resume" }` on scheduler restart
   (`design_pod_thread_scheduler.md` §Persistence). Sessions follow the
   same rule: a restarted scheduler does not re-attach to existing
   daemon-side sessions; it tells daemons to close them on reconnect (or
   the daemon notices the scheduler-disconnect and tears them down itself).
