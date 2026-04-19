# Functions

A unified model for in-flight operations in whisper-agent. Status: **design draft, no implementation yet**. This doc captures the agreed direction and the open questions that still need to be resolved before coding. Revise it in place as those questions get answered — don't accumulate speculative content.

## Why this exists

Today, operations that mutate scheduler state or do non-trivial asynchronous work are implemented per-caller-surface:

- "Create a thread" has one code path for WS clients (`ClientToServer::CreateThread`) and another for behavior firing (`run_behavior`) and a third for model-initiated dispatch (`try_intercept_dispatch_thread`). They converge reasonably well on `Scheduler::create_task`, but the ergonomics at the edges differ.
- "Run a behavior" has three entry points: cron tick, webhook POST, and manual `ClientToServer::RunBehavior` — `fire_trigger` and `run_behavior` are parallel implementations that overlap on payload rendering and overlap-policy handling.
- "Compact" is client-only today; the model has no way to request it and lua won't either unless we reimplement the entry point a third time.
- Tool dispatch splits cleanly between builtin and MCP, but every caller of a tool is a model — if lua wants to invoke an MCP tool, it has to reimplement the dispatch path, or we grow a new surface.

The upcoming lua-hook work will add a fourth caller surface. Without a unified model we end up with N × M implementations of the same operations, and the permission story will be different in each.

A **Function** is a single uniform representation of an in-flight operation, callable from any surface (WS client, model tool call, lua hook, scheduler-internal) with one implementation per operation.

## Scope

### Fits the Function model

Anything caller-initiated, asynchronous, and worth auditing as a single unit of work:

- `CreateThread` (interactive, behavior-spawned, and dispatched-child all collapse to one)
- `CompactThread`
- `RebindThread`
- `CancelThread` (admits synchronously with a no-op progress stream and immediate terminal; kept as a Function for uniform caller access rather than a parallel API)
- `RunBehavior` (manual / cron / webhook all converge)
- `BuiltinToolCall` (each builtin is a variant, or a sub-dispatch inside one variant)
- `McpToolUse` (one variant covering the open set of MCP tools)

### Doesn't fit, and stays out

- **Thread state-machine stepping** — pure in-memory mutation driven by `Thread::step()`; not caller-initiated.
- **ModelCall** — the thread calling its bound backend to drive its own turn. Not caller-initiated; it's a sub-operation of thread stepping, with model deltas streaming directly into the thread's conversation and event stream. Kept in the thread's existing state-machine path, not lifted into the Function registry. No caller needs to invoke it directly (lua or otherwise — models are called *by* threads, not *at* threads).
- **GC tick, retention sweep** — internal timers with no caller, no return path, no permission scope.
- **Event broadcasting** — pure infrastructure.
- **Host-env provisioning** — has find-or-attach semantics (many threads wait on one provision) unlike any other operation. Keeping it scheduler-internal avoids bending the Function contract for a single variant's needs.
- **Approval resolution** — `ApprovalDecision` resolves a pending Function; it doesn't initiate one. Modelled as a scheduler message that feeds back into the Function's progress/terminal path.

### One Function, many surfaces: CreateThread as example

`CreateThread` is deliberately the *only* thread-creation Function. Interactive create, behavior-spawned create, and model-dispatched sub-agent are the same Function with different argument shapes:

- Interactive: no `parent`, client-chosen bindings, `wait_mode: ThreadCreated`, interactive subscriber on the resulting thread.
- Behavior-spawned: no `parent`, bindings derived from behavior config, `wait_mode: ThreadCreated`, no interactive subscriber.
- Model dispatch (sync): `parent: Some(ParentLink { thread_id, tool_use_id })`, `wait_mode: ThreadTerminal`, bindings inherited or explicitly narrowed.
- Model dispatch (async): `parent: Some(ParentLink { thread_id, tool_use_id })`, `wait_mode: ThreadCreated`, bindings inherited or explicitly narrowed.

One Function, parameterized. The tool-pool layer (see below) is what restricts the model's `dispatch_thread` tool to a subset of these fields; the underlying Function still accepts the full range.

## Core concepts

### `Function` (enum)

A closed enum of operation variants. Each variant carries the operation's **spec** — the arguments needed to invoke it.

```
enum Function {
    CreateThread {
        pod_id: PodId,
        initial_message: Option<String>,
        config_override: Option<ThreadConfigPatch>,
        bindings_request: Option<ThreadBindingsRequest>,
        parent: Option<ParentLink>,
        wait_mode: WaitMode,   // meaningful primarily when `parent` is set
    },
    CompactThread { thread_id: ThreadId },
    RebindThread { thread_id: ThreadId, patch: ThreadBindingsPatch },
    CancelThread { thread_id: ThreadId },
    RunBehavior { pod_id: PodId, behavior_id: BehaviorId, payload: Value },
    BuiltinToolCall { name: ToolName, args: Value },
    McpToolUse { host: HostName, name: ToolName, args: Value },
    // ...
}

enum WaitMode {
    /// Function terminates as soon as the thread exists and is seeded.
    /// Used by interactive create, behavior fire, async dispatch.
    ThreadCreated,
    /// Function terminates only when the created thread reaches terminal.
    /// Used by sync dispatch — parent parks on this Function's terminal.
    ThreadTerminal,
}

struct ParentLink {
    thread_id: ThreadId,
    tool_use_id: ToolUseId,
    // Optional depth tracking, whatever dispatch needs.
}
```

(Exact shape TBD; the sketch above is illustrative, not normative.)

Why an enum rather than a trait object:

- Preserves the scheduler's single-writer discipline — the main loop can match-and-mutate without dynamic dispatch.
- The full operation set is visible in one place for audit and review.
- PermissionScope checks and cancel-safety contracts are naturally variant-specific and easy to match at the registration site.

MCP tools are the one open set. They collapse to a single variant (`McpToolUse { session, name, args }`) — the scheduler doesn't need to know about specific MCP tools at the type level.

### `ActiveFunction`

The server-owned runtime state of an in-flight Function. Conceptually:

```
struct ActiveFunction {
    id: FunctionId,
    spec: Function,
    scope: PermissionScope,
    caller: CallerLink,
    state: FunctionState,   // Pending | Running | Cancelling | Terminal
    cancel: CancelHandle,
    progress_tx: Sender<ProgressEvent>,   // typed streaming, see Progress and terminal
    terminal_tx: OneShot<FunctionOutcome>, // typed-per-variant result
}
```

The scheduler holds `active_functions: HashMap<FunctionId, ActiveFunction>`. This is the sole authoritative state for what's in flight.

- **`FunctionId` is a scheduler-assigned monotonic `u64` counter**, same pattern as `ConnId`. Not a UUID — Functions are non-persistent and single-process, so there's no cross-process identity need.
- **HashMap for O(1) lookup** on terminal delivery, cancel, and progress emission by id. Scans (by caller-link for cancel-propagation, by thread for UI filtering) are O(N); at expected scale (tens to hundreds of in-flight Functions per scheduler) that's fine, and an auxiliary index can be added later if profiling shows need.
- Mirrors today's `tasks: HashMap<String, Thread>` layout for consistency.

### `PermissionScope`

A pure capability set: *what* the Function (or Pod, or Thread) is allowed to do. Carries no origin information and no caller identity — those live on `CallerLink`.

**One type, three uses.** `PermissionScope` is the single type used at every layer of the capability stack:

- **Pod.allow** is a `PermissionScope` — the ceiling for anything running in the pod.
- **Thread effective scope** is a `PermissionScope` — derived from the pod's scope narrowed by the thread's bindings and config.
- **Caller scope** (for a Function invocation) is a `PermissionScope` — supplied or derived per caller surface.

Because they're the same type, the same subset-check function is used at every boundary: pod-level gate, thread creation, rebind validation, Function registration.

**Sketch of the v1 shape** (names indicative, not final):

```
struct PermissionScope {
    // What bindings may be used / granted. Each is an AllowMap with
    // tri-state dispositions (Allow / AllowWithPrompt / Deny).
    backends: AllowMap<BackendName>,
    host_envs: AllowMap<ProviderName>,
    mcp_hosts: AllowMap<HostName>,

    // What tools may be invoked. Same AllowMap shape — no special
    // treatment. Approval-required tools get AllowWithPrompt.
    tools: AllowMap<ToolName>,

    // Pod-level access.
    pods: PodsScope,
}

enum PodsScope {
    /// Caller may act on any pod — including pods created after scope
    /// construction — with the given PodOps. Used by scheduler-internal
    /// callers and (today) WS clients with full trust.
    All(PodOps),
    /// Caller may act only on the enumerated pods, with per-pod
    /// operation dispositions.
    Per(Map<PodId, PodOps>),
}

struct PodOps {
    threads:   AllowMap<ThreadOp>,     // Create | Read | Send | Cancel | Compact | Rebind | Archive
    behaviors: AllowMap<BehaviorOp>,   // Create | Modify | Delete | Run
    config:    AllowMap<PodConfigOp>,  // Read | Modify
}

enum Disposition { Allow, AllowWithPrompt, Deny }

struct AllowMap<T> {
    /// Disposition for items not in `overrides`.
    default: Disposition,
    /// Per-item disposition overrides.
    overrides: Map<T, Disposition>,
}
```

Rules:

- **Tri-state disposition.** `Deny > AllowWithPrompt > Allow` by restrictiveness. Scope check returns a `Disposition`, not a bool. `Allow` admits directly, `AllowWithPrompt` admits but triggers a prompt (see Approval section), `Deny` rejects synchronously at registration.
- **Monotonic composition.** When Function A's execution triggers Function B, B's scope is ≤ A's. Never widens. Narrowing picks the *more restrictive* disposition per item: `max(a, b)` under the ordering above. `AllowMap` intersection is `{ default: max(a.default, b.default), overrides: merged map taking max per key, with missing keys taking the other side's default }`.
- **Gated at the scheduler, not inside variants.** The scope check happens at registration, before the variant's logic runs. Variants don't re-check their own scope.
- **No per-field special cases.** Tools get the same `AllowMap` treatment as backends and mcp_hosts. Today's `ApprovalPolicy` presets (`AutoApproveAll` / `PromptDestructive` / `PromptPodModify`) become constructors that build the appropriate `AllowMap<ToolName>` at pod/thread config-load time. The `tool_allowlist` ("remember approval") becomes per-tool overrides that upgrade `AllowWithPrompt` entries to `Allow` for specific tool names.
- **Cross-pod is expressible at the type level; single-pod is a construction-path default.** `PodsScope::Per` with a single entry is what pod/thread construction produces today; `PodsScope::All` is used for scheduler-internal and WS-client full-trust callers. The first concrete multi-pod caller (likely a cross-pod lua hook) will construct a `PodsScope::Per` with multiple entries explicitly, and registration will check each operation's target pod is admitted by the scope's `PodsScope`.

### Scope carries permission; the scheduler supplies the resource

A subtle but load-bearing point: the `PermissionScope`'s `backends` / `mcp_hosts` / `host_envs` fields name *what the caller is permitted to reach* — they don't carry the backing resource handles themselves.

The actual clients (backend HTTP clients, MCP sessions, host-env handles) live in the scheduler's resource registry, owned by the server. When a Function runs, it names the resource it needs by name (e.g., `McpToolUse { host: HostName, ... }`) and the scheduler resolves that name against its registry at execution time.

Why this split matters:

- **Lua-without-a-thread has a natural path to MCP tools.** Lua's scope includes host names it's permitted to reach; when it invokes `McpToolUse`, the scheduler resolves the host name to whatever session exists (typically a shared pod-level session). The caller never holds the resource.
- **Thread rebind works cleanly.** A rebind swaps the thread's resource set; it doesn't touch the caller's scope. In-flight Functions continue against the resources they resolved at start; future Functions resolve against the new set.
- **Scope remains pure data.** No lifetime-entangled handles, no refcounts, no concurrency concerns. Two scopes can be intersected without touching the registry.
- **The scope check is pure authorization, not resource availability.** Registration may still fail later (resource name valid but the resource is down, e.g., MCP host unreachable) — that's a deferred error on the Function's terminal, not a scope-check reject.

**What this consolidates from today:**

- Pod `[allow]` table (backends, host_envs, mcp_hosts) → the bindings fields as `AllowMap`s. **pod.toml schema will be rewritten** to express `AllowMap`s / `Disposition`s directly rather than via the old preset names. The project is still early enough that we drop backwards compatibility outright — existing persistence directories will be wiped, not migrated.
- Thread `bindings` gating → the same bindings fields, narrowed.
- `ApprovalPolicy` presets → `AllowMap<ToolName>` constructors. `AutoApproveAll` = all-`Allow`. `PromptDestructive` = destructive tools `AllowWithPrompt`, rest `Allow`. `PromptPodModify` = pod-modifying tools `AllowWithPrompt`, rest `Allow`.
- Per-thread `tool_allowlist` ("remember approval") → per-tool `Allow` overrides in the thread's `AllowMap<ToolName>`.
- Client implicit-trust → a scope that admits everything, explicitly; no caller-type bypass.
- Lua hook scope → assembled per-hook from pod/thread context and hook declaration.

**What PermissionScope deliberately does *not* own:**

- **Caller identity / authn.** Lives on `CallerLink`.
- **Sandbox-level filesystem/network limits.** Those remain on the sandbox layer (landlock, etc.) below the MCP boundary.
- **Resource handles.** Names of reachable resources live in scope; the handles live in the scheduler's registry (see above).

This is a **refinement pass, not a reimagining.** A deeper rework (identity tiers, elicitation, per-path fine-grained gates) is deferred and expected to happen later, on top of this foundation.

### `CallerLink`

Separate from PermissionScope. Describes *who invoked this Function* and *where the results go*.

```
enum CallerLink {
    WsClient {
        conn_id: ConnId,
        correlation_id: CorrelationId,
    },
    ThreadToolCall {
        thread_id: ThreadId,
        tool_use_id: ToolUseId,
    },
    Lua {
        hook_id: HookId,
        delivery: LuaChannelId,   // opaque handle into the lua runtime
    },
    SchedulerInternal(InternalOriginator),
}

enum InternalOriginator {
    CronFire { pod_id: PodId, behavior_id: BehaviorId, fired_at: Timestamp },
    AutoCompact { thread_id: ThreadId },
    // Keep closed. New entries trigger design review — this is the primary
    // mechanism for preventing "just add a callback" escape-hatch patterns.
}
```

Functions are **opaque to the caller-link variant**. A variant's implementation produces progress and terminal events without knowing whether the result will land in a WebSocket, a model turn, or a lua coroutine. The scheduler's router reads the link and delivers to the right surface. This is the key move that makes lua additive rather than disruptive: no Function variant grows a "lua branch."

**Design principles:**

- **Identity over transport.** Each variant names *who* the caller is, in terms that are auditable. Delivery details (which mpsc channel, which WS sink) are implicit adjuncts the scheduler looks up — not fields on the link itself. `ThreadToolCall` needs no channel handle because the thread_id + tool_use_id suffices to route back through `Thread::apply_io_result()`.
- **Subject lives on the Function spec, not the link.** The CallerLink says who/where; the Function spec says what. Scope derivation is a function of `(caller identity, subject)` where the subject comes from the spec. This keeps routing and authorization orthogonal.
- **Lua keeps its runtime internals out of the scheduler.** The scheduler sees `hook_id` (for audit + scope derivation) and an opaque `LuaChannelId`. The scheduler posts results into that channel; the lua runtime handles resuming the right coroutine internally. Coroutines, VMs, and stack frames are lua-runtime concepts the scheduler never reasons about.
- **`SchedulerInternal` is closed-enum.** No boxed callbacks, no string tags, no opaque "run this on terminal" hooks. Every internal origination is a named variant. Adding a new one is a review step — which answers "does this actually belong as a Function?" explicitly rather than as drift.

**Per-variant cancel-on-caller-gone detection:**

| Variant | "Caller gone" means |
|---------|---------------------|
| `WsClient { conn_id }` | `conn_id` no longer registered in the server's conn registry |
| `ThreadToolCall { thread_id }` | Thread is in a terminal state (Cancelled/Failed/Completed) |
| `Lua { delivery }` | `LuaChannelId` has been dropped/closed by the lua runtime |
| `SchedulerInternal(_)` | Never gone — runs to its own terminal or cancels via other means |

**Default cancel-on-caller-gone policy** (per-variant Function policy still wins; this is the default if a variant doesn't specify):

- `WsClient` gone → **continue**. Most operations have persistent side effects the user still wants; the observation channel is what's gone, not the intent.
- `ThreadToolCall` gone → **cancel**. No consumer for the result.
- `Lua` gone → **continue** without delivery. Hook-initiated work shouldn't abort because the hook returned.
- `SchedulerInternal` → N/A.

**CorrelationId contract:** always supplied by the client (zero allowed for fire-and-forget), not optional. Makes routing code uniform and gives the client a consistent handle even when they don't care about matching a reply.

**Audit:** a helper on `CallerLink` renders a stable, log-friendly identifier (`caller.audit_tag() -> AuditTag`) so the Function registration and terminal audit logs can use one uniform "who did this" field regardless of variant. Keeps the audit log format in one place.

## Ownership

**The server owns Functions.** `active_functions` is the source of truth for what's in flight; no other component holds a Function's lifecycle.

**Threads do not own Functions.** This is worth stating explicitly because it's tempting to assume otherwise:

- A Thread's state machine continues to track what it's waiting on via its own fields (`AwaitingTools { tool_use_id }`, `AwaitingModel`, etc.). It does not hold a function id.
- When the Function terminates, the scheduler routes the result back via the caller-link — `ThreadToolCall { thread_id, tool_use_id }` tells the scheduler which thread and which pending tool-use to apply the result to.
- A Thread *may* optionally cache a handle to the ActiveFunction for UI ergonomics (e.g., "show what this thread is waiting on"), but this is not required for correctness.
- Cancellation of a Thread does not reach *into* its Functions. The scheduler, on seeing a Thread go to Cancelled, scans `active_functions` for entries whose `CallerLink` points at that thread and applies each variant's cancel-on-caller-gone policy.

Net: one-way reference (Function → caller-link → possibly thread). No bidirectional bookkeeping.

## Lifecycle

### Registration is synchronous

When a caller invokes a Function, the scheduler does a synchronous accept-or-deny:

1. **Precondition check.** Variant-specific guards — e.g., "thread is not already compacting," "thread exists," "patch is compatible with pod allow table."
2. **PermissionScope check.** The caller's scope must admit the operation.
3. If either fails, the scheduler returns an immediate synchronous error. Nothing enters `active_functions`.
4. If both pass, an `ActiveFunction` is constructed in `Pending` state, admitted to the registry, and its `FunctionId` is returned to the caller.

**Return type is uniform across caller surfaces:**

```rust
fn register(&mut self, spec: Function, scope: PermissionScope, caller: CallerLink)
    -> Result<FunctionId, RejectReason>;

enum RejectReason {
    ScopeDenied { detail: String },
    PreconditionFailed { detail: String },
    InvalidSpec { detail: String },
    ResourceBusy { detail: String },
    // ... closed set, reviewed on addition
}
```

The returned `FunctionId` is all the caller gets directly. Progress and terminal delivery is not bolted to the return — it flows through the caller-link routing the scheduler already owns:

- **WS client** observes progress/terminal via its existing outbound channel; the `correlation_id` in its `CallerLink::WsClient` is what it matches events against.
- **Thread tool call** doesn't directly hold the id; the scheduler routes the terminal into `Thread::apply_io_result()` keyed by the `tool_use_id`.
- **Lua** yields its coroutine; the lua runtime resumes it when the scheduler posts to the `LuaChannelId`. The lua binding can expose the `FunctionId` if needed for explicit cancel, but the normal path is just `local result = whisper.foo(...)`.
- **Scheduler internal** handles terminals inline per its `InternalOriginator`.

In short: the `FunctionId` is for programmatic use (explicit cancel, status query, UI cross-reference). All delivery happens via caller-link.

**Anything that requires the scheduler to spin** — awaiting I/O, awaiting a child thread, running a model stream — happens through the phone-home path *after* acceptance. Deferred errors, cancellation, and success terminals all flow through that same channel. Registration itself is always immediate.

This matters for two reasons. First, it keeps the scheduler single-writer: registration is one state mutation, bounded. Second, it gives callers a crisp signal for "was this even a legal request?" without having to subscribe to progress just to find out.

### Progress and terminal

Every ActiveFunction produces two event streams, ordered within themselves, routed by the scheduler to the caller-link's delivery surface (WS frame, thread's `apply_io_result`, lua resume, etc.).

**Ordering invariant: no progress after terminal.** The delivery path is ordered such that once a terminal is observed by the caller, no further progress events for that Function will arrive. Concretely: progress and terminal go through a single ordered channel (or the router enforces "drop progress after terminal forwarded"); either way, downstream consumers can treat terminal as a hard end-of-stream signal without racing against late progress.

- **Progress** — zero or more decorative/streaming status events during execution.
- **Terminal** — exactly one result event.

The two streams have **different typing philosophies**, because the tradeoffs hit them differently.

**Terminal is typed per Function variant.**

The terminal is what callers program against; its shape must be predictable without runtime probing. Each Function variant declares its own terminal payload type:

```rust
enum FunctionTerminal {
    CreateThread(CreateThreadTerminal),
    CompactThread(CompactThreadTerminal),
    RebindThread(RebindThreadTerminal),
    CancelThread(()),              // immediate, no payload
    RunBehavior(RunBehaviorTerminal),
    BuiltinToolCall(ToolResult),
    McpToolUse(ToolResult),
    // ... one per Function variant
}

// Per-variant shapes differ in meaningful ways. Example:
struct CreateThreadTerminal {
    thread_id: ThreadId,
    // Populated only for wait_mode = ThreadTerminal:
    final_result: Option<ThreadTerminalSummary>,
}
```

Orthogonal to variant, every Function can terminate in one of three overall outcomes:

```rust
enum FunctionOutcome {
    Success(FunctionTerminal),
    Error(FunctionError),    // typed-enough for the common cases; carries context
    Cancelled(CancelReason), // UserDenied | CallerGone | ExplicitCancel | ...
}
```

Lua, model tool-call glue, and the WS client all get a terminal they can match on directly, without digging through untyped JSON.

**Progress is a typed envelope with common kinds plus an escape hatch.**

Progress events are streaming-decorative — they represent intermediate state that drives UI rendering and audit logging but aren't what the caller ultimately programs against. The shape:

```rust
enum ProgressEvent {
    /// Content blocks — text, image, audio, structured. Uses the existing
    /// ContentBlock type so multimodal payloads ride the CBOR wire natively
    /// (no base64-in-JSON regression). Partial / streaming content uses
    /// the same type — ContentBlock carries its own "is this the final
    /// form" signaling.
    Content(ContentBlock),

    /// Status transitions — "starting", "waiting_approval", "retrying", etc.
    Status(StatusKind),

    /// Approval request — produced by tool Functions that need prompting.
    /// The resolution routes back into the Function (see Approval section).
    ApprovalRequest(ApprovalRequest),

    /// Escape hatch for variant-specific progress that isn't worth naming
    /// in the common enum. `kind` is a stable string tag.
    Custom { kind: String, payload: serde_json::Value },
}
```

**Why this shape:**

- **Typed common cases** so UI and audit don't have to special-case 20 kinds of JSON blobs. Model-delta streaming, content blocks, status transitions, and approval requests are the things every rendering/logging layer needs to distinguish.
- **`ContentBlock` as the multimodal carrier.** The codebase already has a `ContentBlock` type covering text / image / structured content. Progress events reuse it rather than reinventing. Binary rides the existing CBOR wire.
- **`Custom { kind, payload }` escape.** Variant-specific progress (compaction summary chunks, behavior-thread forwarded events, etc.) that doesn't deserve its own top-level variant lives here. `kind` is a stable string, not a typed enum — churn happens inside the payload, not inside the protocol.
- **No `serde_json::Value` at the top level.** The top-level variants name what *category* of event this is. `Value` only appears inside the `Custom` escape, where the untyped freedom is actually wanted.

**Audit log handling:**

- Every Function registration + terminal produces an audit record with structured kind + summary.
- Progress events are summarized in audit (kind + maybe a length/token count), not logged in full — avoids blowing up the log with streaming delta spam or multimodal binary.
- Large binary payloads in terminals or content blocks get elided to a hash/reference in audit; the full payload is not JSON-wrapped for storage.

This replaces the earlier "progress schema TBD" placeholder and commits the shape enough to build against. Further refinement (more typed variants as `Custom` entries prove their weight) is a normal evolution.

### Thread in-flight flags

Preconditions like "thread is not already compacting" are checked against per-thread **in-flight operation flags** rather than by scanning `active_functions`:

```rust
bitflags! {
    struct InFlightOps: u32 {
        const COMPACTING = 1 << 0;
        const REBINDING  = 1 << 1;
        // one bit per operation kind that has an exclusivity invariant
    }
}

struct Thread {
    // ... existing fields ...
    in_flight: InFlightOps,
}
```

- Functions that need exclusivity set their bit on start and clear it on terminal (success, error, or cancel).
- The scheduler owns the set/clear lifecycle (uniform across variants) rather than variant logic — reduces the chance a bit gets stranded.
- Preconditions are O(1) reads, not registry scans.
- Not every Function needs a bit — only those with "can't run concurrently with another of the same kind on the same thread" invariants.

**Restart note:** in-flight flags should be cleared as part of the "flip to `Failed { at_phase: ... }`" resume handling, since Functions don't persist. If forgotten, a thread mid-compaction at the moment of a crash becomes permanently un-compactable. Acceptable to defer a clean fix for this until it bites in practice — the workaround (wipe thread, re-create) is cheap in early dev.

### Execution model

Functions run as futures pumped through the scheduler's existing `FuturesUnordered` mechanism — same pattern threads and provisioning use today. Each `ActiveFunction` carries a future plus channels; completions are polled by the main scheduler loop and routed to caller-links. No new runtime primitive is introduced; this reuses the single-writer-scheduler-with-external-io pattern.

### Cancel-by-thread sweep

When a Thread transitions to a terminal state, the scheduler scans `active_functions` for entries whose work targets that thread, and applies each variant's cancel-on-caller-gone policy. "Targets the thread" means either:

- `CallerLink::ThreadToolCall { thread_id }` matches, or
- `CallerLink::SchedulerInternal(InternalOriginator::AutoCompact { thread_id })` matches, or
- (future internal variants carrying a `thread_id` field match analogously).

Worth naming explicitly because the scan must peer into `SchedulerInternal`'s payload — not just the obvious thread-bearing variants. A helper `CallerLink::targets_thread(&self, id) -> bool` keeps this logic in one place.

**`CancelThread` triggers the sweep.** When a `CancelThread` Function accepts, its implementation (a) flips the thread's state to Cancelled, then (b) initiates the cancel-by-thread sweep against every other Function targeting that thread. The sweep is a *signal*, not a unilateral kill: each matched Function receives a caller-gone event and applies its variant's cancel-on-caller-gone policy. Cooperative — a Function with committed side effects and a "continue on caller-gone" policy will keep running; most will cancel per the defaults. This puts CancelThread's correctness responsibility in one place rather than scattered across the scheduler loop.

### Persistence

Functions are **not persisted.** On restart, `active_functions` is empty.

A Thread that was parked awaiting a Function transitions to `Failed { at_phase: "awaiting_function" }` (or similar — exact phase tag TBD) on resume. Same shape as today's in-flight I/O restart failure; no new hazards introduced.

This is deliberate. Cross-process resumption of arbitrary side effects (MCP calls that hit remote systems, model calls mid-stream) isn't worth the complexity.

## Cancellation and cancel-safety

Every Function variant declares its cancel-safety contract as part of the enum documentation. Two dimensions:

### Cancel-on-caller-gone

What should the scheduler do if the caller's `CallerLink` becomes meaningless (WS client disconnects, owning Thread is cancelled, lua coroutine is collected)?

- **Cancel** — work has no remaining audience, no side effects worth preserving. Default for most variants.
- **Continue** — other subscribers may still care, or the work has committed side effects worth letting finish. Rare but real (e.g., a behavior-fired thread that produces persistent output).

Variant documents its choice.

### Post-cancel side-effect residue

What persists after a cancel?

- Local futures are dropped. Anything the variant did before the cancel signal is what it is.
- For MCP tool calls, the remote system may have already committed the operation; cancel only reaches the local awaiter. The variant must say so explicitly — lua hooks and model logic both need to know they can't rely on "cancel = undo."
- For builtin tools, the contract is per-tool.

This is a per-variant contract, enumerated alongside the Function enum definition.

## Approval prompting

**The scope already answered whether to prompt.** `PermissionScope` carries a tri-state `Disposition` per item; `AllowWithPrompt` means "admissible, but require user confirmation before proceeding." The scheduler discovers this at the registration scope check, tags the resulting `ActiveFunction` with "prompt required," and hands control to the Function.

Flow:

1. A Thread's step-loop decides a tool should be called. It registers a `BuiltinToolCall` or `McpToolUse` Function with the thread's effective scope.
2. Scheduler scope-checks. Result is one of three dispositions:
   - `Allow` → admit, no prompt flag. Function proceeds directly.
   - `AllowWithPrompt` → admit with `prompt_required` on the `ActiveFunction`. Function emits an `ApprovalRequest` progress event at start and blocks on resolution.
   - `Deny` → synchronous `RejectReason::ScopeDenied` at registration.
3. For `AllowWithPrompt`: the UI / client sees the approval request via the caller-link's routing (same path as any progress event). The user answers.
4. The decision routes back **into the Function**, not into the Thread. The Thread is oblivious — it's just waiting for its tool to finish.
5. On **approve** → Function resumes execution, eventually produces its `Success(payload)` terminal. Thread sees the tool result, continues its turn as normal.
6. On **deny** → Function terminates with `Cancelled(UserDenied)`. Thread sees that terminal, synthesizes a "tool denied" tool_result and continues the turn — the same UX as today.

Why this works cleanly:

- **Thread logic stays simple.** A Thread's only concern is "my Function is pending; I wait for its terminal." It doesn't distinguish "slow tool" from "awaiting approval" from "remote service is down."
- **No runtime prompt-policy lookup.** The Function does not inspect tool annotations or thread/pod config at runtime — the scope already encoded the disposition. All "which tools prompt under which conditions" decisions live in scope construction (pod config → `AllowMap<ToolName>`), not in Function logic.
- **Uniform across caller surfaces.** A WS client, model tool call, or lua hook all go through the same scope check; the disposition determines what happens next.
- **Denial is just a terminal.** The cancellation plumbing we need for other reasons carries denial for free.

The one real dependency: `ApprovalDecision` messages route to Function ids, not Thread ids. Routing-layer change, not a semantic one.

### Approval with no human available

If a Function scope-checks as `AllowWithPrompt` but the caller-link has no surface for delivering a prompt to a human (lua hook in a headless context, cron-fired thread with no subscribers, etc.), the scheduler **auto-denies** after a short bounded wait — it does not pend indefinitely. The resulting terminal is `Cancelled(NoApprovalPath)`, distinct from `Cancelled(UserDenied)` for audit.

The auto-deny policy applies at the caller-link router layer, which knows whether it has a delivery surface for a prompt. Variants that want different behavior (e.g., a cron-fired behavior that should auto-approve because its config pre-authorized it) should construct their caller's scope with `Allow` for those tools rather than relying on any runtime override.

## What counts as a caller

- **WS client** — user at the webui (or another client) invoking an operation directly. V1: scope is **hardcoded full-trust for every WS client**, constructed at connection open — matches today's implicit behavior and keeps the plumbing trivial. Tightening this to per-identity derivation is a future step (see Pattern 3 in `design_permissions.md`); until then the "conn_id → scope" map is a one-liner that always returns the full-trust scope.
- **Thread tool call** — a model invoking a tool during a turn. Scope derived from the thread's effective `PermissionScope` (which is the pod's scope narrowed by thread bindings).
- **Lua hook** — a lua script reacting to an event. Default scope is **the pod's `PermissionScope`** — most hooks are pod-level actions and inheriting the pod's scope is the right baseline. Finer-grained scope acquisition (per-hook declaration, event-derived narrowing, cross-pod hooks) is **deferred until concrete use cases emerge**; when they do, narrower-than-pod is the composition direction (never wider).
- **Scheduler internal** — cron fire, auto-compact, retention-driven actions. Scope is full/internal; caller-link records the originating event for audit purposes.

## Tool pools vs the Function registry

The **Function registry** is the server's internal operation surface: the full set of variants, each accepting its complete spec. Permission enforcement happens here, at the scheduler.

The **agentic tool pool** is a separate registry — a curated, narrower view exposed to a specific caller surface (notably the model during a thread turn, but also potentially lua hooks or even WS clients in restricted modes). Each tool in the pool:

- Has a model-facing name, description, and JSON schema (what the caller sees).
- Maps to a Function invocation with possibly-fixed, templated, or constrained arguments.

**Why it's separate from the Function registry:**

Adding arguments to a Function does **not** automatically expose them to every caller. Example: `CreateThread`'s spec includes `pod_id`, `bindings_request`, `config_override`, etc. The model's `dispatch_thread` tool presents only `{ prompt, sync }` and internally invokes `CreateThread` with `pod_id = parent_pod`, `parent = current_thread`, `wait_mode = sync ? ThreadTerminal : ThreadCreated`, `bindings_request = None` (inherit). The model cannot escape those fixed fields through any choice of tool arguments.

This gives us three narrowing layers that stack:

1. **PermissionScope** (the capability gate): what the caller is *admissible to do*.
2. **Tool pool** (the surface gate): what operations and argument shapes the caller *can express*.
3. **Function registration** (the precondition gate): whether the specific invocation is *consistent* (thread exists, not already compacting, scope admits, etc.).

Any one of the three can deny; all three must pass.

**What the tool pool enables:**

- **Model restrictions without Function-variant explosion.** One `CreateThread` Function; multiple pool tools (`dispatch_thread`, an interactive "new thread" tool for a future multi-thread UI, etc.) each constraining the same underlying Function differently.
- **Per-surface tool catalogs.** The model's pool is different from lua's pool is different from a restricted WS client's pool. Each surface gets a curated view.
- **Pool composition is where "model sees only these tools" lives.** Pool is constructed from (pod-level tool config, thread bindings, caller-surface rules). PermissionScope sits *underneath* — even if a caller manages to invoke a Function the pool didn't intend to expose, the scope check still gates it.

**What it doesn't do:**

- The tool pool is *not* a security boundary on its own. Restricting a tool's arg schema doesn't mean the Function is safe to call with other args — it just means this caller can't *express* those args through this pool. PermissionScope is the guarantee. Tool pool is ergonomics + surface shaping.

**Lifecycle and source:**

Tool pools are assembled per caller session (thread, lua hook, client connection) from the caller's context. The primary source is **pod config** — pools are declared on the pod and projected per caller type (model-facing pool, lua-facing pool, etc.) at session construction. Thread bindings and caller-surface rules can narrow further. Pools are not persisted independently — they're always derivable from pod/thread config + caller type. Changing pod config or rebinding a thread can change the pool available to a running caller (this is how today's MCP-tool list refresh works).

## Observability

Progress and terminal events route to observers via **two orthogonal mechanisms**, matched to the caller-link type:

**Thread-owned Functions forward progress into the owning thread's event stream.**

Functions with `CallerLink::ThreadToolCall { thread_id, .. }` have their progress events rendered as part of the thread's history — the same way model-call deltas and tool output appear today. This matches the existing UI contract: a WS client subscribed to a thread sees everything that's happening inside the thread, including the in-flight Function work. The terminal routes into `Thread::apply_io_result()` which produces its own thread event.

Note: forwarding `ProgressEvent::Content(ContentBlock)` into the thread's event stream may require extending the thread-event wire protocol to carry multimodal content. Today's wire encodes text/tool-use/reasoning natively; adding `ContentBlock` coverage is a normal wire extension, flagged here so it's planned rather than discovered.

**Non-thread-owned Functions deliver via their caller-link only.**

Functions with `CallerLink::WsClient`, `::Lua`, or `::SchedulerInternal` deliver progress/terminal to the caller's specific channel:

- `WsClient`: progress and terminal events go out the WS session keyed by the `correlation_id`. The client matches them to its original invocation.
- `Lua`: terminal resumes the coroutine; progress (if the lua binding exposes it) flows through the same delivery channel as an iterator.
- `SchedulerInternal`: handled inline per the `InternalOriginator` variant's terminal logic.

**Cross-observer subscription (e.g., "show me every in-flight Function in the system") is deferred.** The `FunctionId` is stable and the router can be extended to support by-id subscriptions later, but no current use case justifies building that now.

**Audit is the one universal observer.** Every Function registration, every progress event (summarized), and every terminal produces an audit log record — this is independent of caller-link routing and applies to all Functions uniformly.

## Open questions

These questions remain for later — not blockers for starting implementation, but worth revisiting:

- **Finer-grained lua scope acquisition.** Default is "inherit pod scope." When concrete lua use cases emerge that need narrower or differently-shaped scopes (per-hook declarations, event-derived narrowing, cross-pod hooks), design the composition rules then.
- **Generic by-id Function subscription.** Deferred until there's a concrete need for a "show me every in-flight Function" observer. `FunctionId` is already stable enough to add this later without protocol churn.
- **Exact error taxonomy inside `RejectReason` and `FunctionError`.** Start with the sketch; let real call sites tell us which distinctions matter.
- **Auxiliary indexes on `active_functions`.** HashMap-by-id is enough for v1. If profiling shows scan-heavy cancel-propagation or thread-filter workloads, add indexes then.
- **Auto-deny timeout for `AllowWithPrompt` without a delivery surface.** Policy committed (auto-deny with `NoApprovalPath`), but the concrete timeout duration and how the caller-link router signals "no delivery surface" is a per-caller-type implementation detail.
- **`AllowMap` default-override merge edge cases.** The narrowing rule is "more restrictive disposition wins per item"; first implementation should have unit tests for all combinations of `{default, override-present, override-absent}` on both sides.

## Implementation staging

Six commits. Infrastructure lands alongside the first operation that uses it rather than as standalone plumbing — each commit removes more code than it adds (or at least makes the addition load-bearing for a concrete simplification). Old code paths keep working until their operation's migration commit lands; the tree compiles and the server runs at every commit boundary.

**Commit 1 — Type scaffolding.**

New types only, no existing code touched:

- `src/permission/` — `PermissionScope`, `AllowMap`, `Disposition`, `PodsScope`, `PodOps`, per-layer op enums. Narrowing composition + unit tests.
- `src/functions/` — `Function` spec enum (variants declared, bodies stubbed), `FunctionId`, `ActiveFunction`-adjacent types (`FunctionOutcome`, `FunctionTerminal`, `ProgressEvent`), `CallerLink` with closed `InternalOriginator`, `RejectReason`, `InFlightOps` bitflags.

No pod.toml changes, no scheduler changes, no `ApprovalPolicy` changes. Types exist and are tested in isolation.

**Commit 2 — `CancelThread` migration + scheduler infrastructure.**

First operation migrated; scheduler infrastructure lands alongside it as load-bearing support:

- Scheduler gains `active_functions: HashMap<FunctionId, ActiveFunction>`, synchronous `register(spec, scope, caller) -> Result<FunctionId, RejectReason>`, terminal routing back through the caller-link.
- WS session attaches `CallerLink::WsClient { conn_id, correlation_id }` + hardcoded full-trust `PermissionScope` to inbound messages.
- `CancelThread` implemented as a Function: state flip + cancel-by-thread sweep (using `CallerLink::targets_thread`). Existing `ClientToServer::CancelThread` handler removed.

**Commit 3 — `CompactThread` migration + in-flight flags + auto-compact.**

- `Thread.in_flight: InFlightOps` field added with `COMPACTING` bit wired.
- `CompactThread` migrated; precondition checks the bit.
- Auto-compact path rewired to register a Function with `CallerLink::SchedulerInternal(AutoCompact { .. })`.
- Old compaction code path removed.

**Commit 4 — Tool invocation + approval + pod.toml rewrite.**

Heaviest commit — this is where the new scope model starts paying off:

- `BuiltinToolCall` and `McpToolUse` migrated as Functions.
- Scope check returns `Disposition`; `AllowWithPrompt` tags the `ActiveFunction` with a prompt-required flag.
- `ApprovalRequest` progress-event flow with decision routing back into the Function; `ApprovalDecision` messages route to `FunctionId`.
- `ProgressEvent::Content(ContentBlock)` for streaming tool output; thread-event wire protocol extended to carry `ContentBlock`.
- Tool-pool curation layer for model callers — pod.toml defines the pool, thread derives a model-facing view.
- **Pod.toml schema rewritten** to express the new permission model: `[allow.tools]` with `default` + `overrides` replaces `thread_defaults.approval_policy`. Existing `pods/` directories wiped (no migration).
- `ApprovalPolicy` enum removed from protocol crate; per-thread `tool_allowlist` becomes per-tool `Allow` overrides on the thread's scope.

Splittable if the diff gets unwieldy (builtin first, then MCP).

**Commit 5 — Thread lifecycle + behavior operations.**

- `CreateThread` migrated (collapsing interactive / behavior-spawned / model-dispatched paths via `ParentLink` + `WaitMode`).
- `RunBehavior` migrated (folding `fire_trigger` and `run_behavior`).
- `RebindThread` migrated.
- After this commit, every caller-visible operation is a Function.

**Commit 6 — Cleanup.**

- Remove any superseded types and parallel pre-Function dispatch paths still present.
- Update `design_pod_thread_scheduler.md` and `design_permissions.md` to reference the Function model.
- Prune any "Open questions" items resolved during implementation.

## Status

Design complete enough to build against. Phase 1 has not started. As implementation surfaces issues, amend the relevant sections in place; don't let the "open questions" list grow while the body calcifies around stale assumptions.
