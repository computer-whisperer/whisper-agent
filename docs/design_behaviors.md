# Autonomous Pod Behaviors

Pods and threads give us a project-scoped, sandbox-bounded conversation engine. This document extends that with **behaviors**: captured autonomous work that runs on a timer or an event, without a human present in the loop.

Companion docs: [`design_pod_thread_scheduler.md`](design_pod_thread_scheduler.md) (the pod/thread/resource model this builds on), [`design_headless_loop.md`](design_headless_loop.md) (why the loop runs server-side to begin with).

## Motivation

Three shapes of work we want to express cleanly:

- **Scheduled surveys** — "every weekday morning, check the state of the repo's CI across branches and emit a summary."
- **Document synchronization** — "once a day, walk a notes directory and update wiki entries where they've drifted."
- **Event-driven processing** — "as new emails arrive via an MCP service, triage them and act."

The common structure: a prompt + thread configuration that is spawned on a trigger, runs largely mechanically to completion, and is archived or woken again on the next event. None of the three needs a subscribed UI to run; all three should be *visible* in the UI after the fact so the user can audit what happened.

What the existing pod/thread abstraction already supports:
- `Completed` as a stable terminal state for the thread state machine.
- `ApprovalPolicy::AutoApproveAll` so unattended threads don't wedge on prompts.
- Sandbox + pod `[allow]` as the load-bearing safety boundary — the appropriate primitive for unattended runs, where no human is watching approvals anyway.
- Per-thread binding overrides (narrower than the pod default) so an autonomous thread can run with a tighter tool surface than interactive work.

What's missing:
- No notion of "behavior" — a captured, repeatable prompt + config + trigger, distinct from an ad-hoc thread creation.
- No trigger sources inside the scheduler. The run loop reacts to client messages and I/O completions only; there's no cron branch, no event ingress.
- No way to spawn a thread without at least one subscribed client (today `CreateThread` assumes a client that wants events). Unattended runs need headless thread creation.
- No on-completion retention policy. Interactive threads sit in `Completed` indefinitely; behaviors that fire daily would accumulate without bound.

## Three new concepts

1. **Behavior** — pod-scoped, self-contained under `<pod>/behaviors/<name>/`. A TOML config, a prompt markdown file, and a state file. The pod's `[allow]` table remains the capability cap; a behavior's thread config is a subset, not an escalation.

2. **Trigger** — the fire source. Variants: `Manual`, `Cron`, `Webhook`. Each trigger instance produces a `TriggerFired { pod_id, behavior_id, payload }` onto the scheduler inbox. Event-source triggers (imap, slack, github webhooks-via-MCP) are deferred — we add them as adapters once a concrete use case is driving the shape.

3. **BehaviorOrigin** — a field on `Thread`. Records which behavior spawned it, when, and with what payload. Absent for interactive threads. Not an enum, not a `ThreadKind` — just an `Option` carried alongside the existing fields. See ["Why not ThreadKind"](#why-not-threadkind).

## Behavior directory layout

```
<pod>/
  pod.toml
  system_prompt.md
  threads/
    thread-*.json                     # flat — every thread, interactive or behavior-spawned
  behaviors/
    daily_ci_check/
      behavior.toml
      prompt.md
      state.json
    process_emails/
      behavior.toml
      prompt.md
      state.json
  .archived/
    threads/thread-*.json              # retention sweeps land here; same shape as archived pods
```

Self-contained behavior dirs: the three files for a behavior evolve together (config, prompt, state), so grouping them makes `cp -r behaviors/foo behaviors/bar` a real way to clone, `rm -r` a safe way to retire, and `git diff` clean on edits. When we eventually add report output, it lands as `<pod>/behaviors/<name>/reports/` — the behavior's dir is where the behavior's stuff lives.

Flat `threads/`: a thread's "which behavior spawned me" is a data field on the thread record, not a directory boundary. Splitting threads into per-behavior subdirs would duplicate a runtime concept into filesystem structure, and becomes awkward for subagents spawned *by* behavior-spawned threads (do they live under the behavior dir?). The thread list is one `readdir`.

## `behavior.toml`

```toml
name = "daily-ci-check"
description = "Survey github CI across our branches each morning"

# The trigger table picks exactly one variant.
[trigger.cron]
schedule  = "0 9 * * *"                # standard cron, five-field
timezone  = "America/Los_Angeles"
overlap   = "skip"                     # skip | queue_one | allow
catch_up  = "one"                      # one | none | all  (on server restart after missed fires)

# Thread config overrides layered on top of pod.thread_defaults.
# Any field omitted here inherits from the pod.
[thread]
max_turns       = 30
approval_policy = "auto_approve_all"
model           = "claude-sonnet-4-6"

# Bindings override — must be a subset of pod.allow.
[thread.bindings]
host_env  = "read-only-workspace"
mcp_hosts = ["github", "fetch"]

# What to do when the thread reaches a terminal state.
[on_completion]
retention = "archive_after_days"       # keep | archive_after_days | delete_after_days
days      = 30                         # required for *_after_days variants
```

Alternative `[trigger.*]` shapes:

```toml
# Fires only in response to ClientToServer::RunBehavior. The default shape when
# a user is iterating on a behavior before wiring it to a schedule.
[trigger.manual]

# Exposes POST /triggers/<pod_id>/<behavior_id>; request body becomes the payload.
[trigger.webhook]
# (no required fields for v1; auth and path customization are deferred)
```

The prompt template lives in `prompt.md`. The trigger payload (if any) is available to the prompt via a simple substitution scheme (initial cut: `{{payload}}` expands to the JSON payload pretty-printed; richer templating deferred until we see what behaviors actually want).

## In-memory types

```rust
struct Behavior {
    id: String,                        // directory name, immutable
    pod_id: PodId,
    dir: PathBuf,                      // <pod>/behaviors/<id>
    config: BehaviorConfig,
    prompt: String,                    // contents of prompt.md, loaded eagerly
    state: BehaviorState,
}

struct BehaviorConfig {
    name: String,
    description: Option<String>,
    trigger: TriggerSpec,
    thread: BehaviorThreadOverride,    // same shape as ThreadConfigOverride + bindings patch
    on_completion: RetentionPolicy,
}

enum TriggerSpec {
    Manual,
    Cron { schedule: String, timezone: String, overlap: Overlap, catch_up: CatchUp },
    Webhook { /* v1: nothing; path is derived from pod_id + behavior_id */ },
}

enum Overlap { Skip, QueueOne, Allow }
enum CatchUp { None, One, All }

enum RetentionPolicy {
    Keep,
    ArchiveAfterDays(u32),
    DeleteAfterDays(u32),
}

struct BehaviorState {
    last_fired_at: Option<DateTime<Utc>>,
    last_thread_id: Option<ThreadId>,
    last_outcome: Option<BehaviorOutcome>,   // Completed | Failed { message } | Cancelled
    run_count: u64,
    queued_payload: Option<Value>,           // populated for QueueOne overlap
}
```

`BehaviorOrigin` on the thread:

```rust
struct BehaviorOrigin {
    behavior_id: String,
    fired_at: DateTime<Utc>,
    trigger_payload: serde_json::Value,
}

struct Thread {
    // ... existing fields ...
    #[serde(default)]
    origin: Option<BehaviorOrigin>,
}
```

## Scheduler integration

The scheduler grows one new in-memory structure and two new `select!` branches.

```rust
struct Scheduler {
    // ... existing fields ...
    behaviors: HashMap<(PodId, BehaviorId), Behavior>,
    cron_next_fire: BinaryHeap<CronEntry>,  // min-heap keyed by next scheduled fire time
}
```

New branches in the run loop:

- **`tokio::time::interval` tick (every 30s or so)** — pops ready cron entries, emits `TriggerFired` for each. Re-inserts each with its next fire time.
- **`SchedulerMsg::TriggerFired { pod_id, behavior_id, payload }`** — the handler resolves the behavior, consults overlap policy against `last_thread_id` (still running? already queued?), consults `pod.limits.max_concurrent_threads`, and either spawns a new thread or updates `queued_payload` or drops the fire (logging either way). Spawning is via the same internal pathway as `ClientToServer::CreateThread` but without a subscriber requirement.

Webhook trigger: the existing axum server gains a `/triggers/:pod_id/:behavior_id` POST route. Handler pushes `TriggerFired` onto the scheduler inbox, returns 202 immediately. Request body becomes the payload (JSON, or wrapped string for non-JSON bodies).

Manual trigger: a new client→server message.

```rust
ClientToServer::RunBehavior {
    correlation_id,
    pod_id,
    behavior_id,
    payload: Option<Value>,
}
```

All three trigger variants converge on the same internal `TriggerFired` event, so the overlap/cap/spawn code has one path.

## Headless thread spawning

Interactive thread creation today assumes the creator is a connected client and routes events to that client. For behavior-spawned threads, there's no creator connection. Two small changes:

1. The spawn function gains a `subscribers: Vec<ConnId>` parameter instead of implicitly using the requester's connection. For behavior spawns, the list is empty; events still flow through the router but drop on the floor if nobody is subscribed.
2. The router's pod-level subscription (new or existing subscribers of the pod's detail tier) receives `ThreadCreated` for the new behavior thread the same way it would for an interactive one — this is what makes behavior threads show up in the UI even though nobody created them interactively.

Existing code paths for interactive `CreateThread` adopt the new function shape; behavior is uniform.

## On-completion handling

When a behavior-spawned thread reaches a terminal state (`Completed`, `Failed`, `Cancelled`), the scheduler:

1. Reads the originating behavior's `on_completion` policy from the behavior registry (via `thread.origin.behavior_id`).
2. Updates the behavior's `state.json`: `last_outcome`, increment `run_count`, clear `queued_payload` if set (and, if non-empty and the overlap was `QueueOne`, immediately emit a fresh `TriggerFired` with the queued payload).
3. Applies retention lazily — a separate daily sweep task inspects all `<pod>/threads/*.json`, and moves those past the retention window to `<pod>/.archived/threads/` (for `ArchiveAfterDays`) or deletes them (for `DeleteAfterDays`). Interactive threads without an `origin` default to `Keep` — unaffected by sweeps.

Retention is lazy and non-critical: if the sweep skips a day because the server was down, nothing breaks, next sweep catches up.

## Wire protocol additions

```rust
// ClientToServer
ListBehaviors    { correlation_id, pod_id }
GetBehavior      { correlation_id, pod_id, behavior_id }
CreateBehavior   { correlation_id, pod_id, behavior_id,
                   config: BehaviorConfig, prompt: String }
UpdateBehavior   { correlation_id, pod_id, behavior_id,
                   config: BehaviorConfig, prompt: String }
DeleteBehavior   { correlation_id, pod_id, behavior_id }
RunBehavior      { correlation_id, pod_id, behavior_id, payload: Option<Value> }

// ServerToClient
BehaviorList            { correlation_id, pod_id, behaviors: Vec<BehaviorSummary> }
BehaviorSnapshot        { correlation_id, snapshot: BehaviorSnapshot }
BehaviorCreated         { pod_id, summary }
BehaviorUpdated         { pod_id, summary }
BehaviorDeleted         { pod_id, behavior_id }
BehaviorStateChanged    { pod_id, behavior_id, state: BehaviorStateSnapshot }
```

`PodSnapshot` gains `behaviors: Vec<BehaviorSummary>` so clients see the catalog without a second round-trip after `GetPod`.

`ThreadSummary` gains `origin: Option<BehaviorOriginSummary>` (behavior_id + fired_at; payload elided for the summary tier).

## WebUI surface

Pod detail view splits into two sections:

- **Threads** (top) — interactive threads + any children (future Spawn/Fork/Compact). As today.
- **Behaviors** (below) — one row per behavior with: name, trigger summary ("daily at 09:00 PT"), last-run outcome, last-run timestamp, next scheduled fire (for cron). Expanding the row reveals that behavior's run history: a list of threads filtered by `thread.origin.behavior_id == this`.

Behavior-spawned threads in the main thread list carry a small badge ("⚙ daily_ci") so the user can tell at a glance what spawned them. Default sort keeps interactive threads up top; behavior threads sort by recency within their section.

Deleted-behavior orphans: when a behavior is deleted, its historical threads keep their `origin.behavior_id` (the UI resolves it as "deleted: daily_ci_check" and groups them under a separate "Orphaned runs" section). Deleting the behavior does not destroy its audit trail.

Visibility posture: **all threads are visible, regardless of origin.** The UI does prioritization (sort order, section split, badge density), not filtering. Mechanical threads are legitimate first-class pod citizens.

## A note on MCP notifications

MCP has notifications in both directions — it's JSON-RPC, so any method call without an `id` is a notification. The spec defines a handful (`notifications/initialized`, `notifications/progress`, `notifications/resources/updated`, `notifications/resources/list_changed`, `notifications/tools/list_changed`, `notifications/message`). Our current code treats inbound notifications as no-ops (the server returns 202) and sends only `initialized` from the client side after handshake.

They are not the right abstraction for behavior triggers, for two reasons:

- **Semantics** — MCP notifications are status hints about resources/tools a server already exposes, not an event-delivery channel. "Resource X changed" is a re-check hint; "a new email arrived with payload Y" is a different thing.
- **Transport** — inbound notifications require a bidirectional transport (SSE or WebSocket). Our MCP client today uses plain HTTP POST; it can't receive server-initiated messages without a transport rework.

The right place for triggers is **in-process, on the server side.** MCP is the tool surface for the spawned thread, not the control plane for firing it. When we eventually wrap an event source (github webhooks, an imap watcher) as an MCP server, we'll map its notifications onto our `TriggerFired` internally — but that mapping lives in the trigger adapter, not in MCP's role.

## Why not ThreadKind

The pod/thread scheduler doc sketched a `ThreadKind` enum with `Root | Spawn | Fork | Compact` variants, anticipating subagents and forking. None of those are implemented yet, and `ThreadKind` doesn't appear in the actual code.

We don't need an enum for behavior origin. A thread either has a behavior origin or doesn't; that's one optional field, not a variant of a discriminated union. Adding a `ThreadKind::Behavior { .. }` would pick a direction on an enum that doesn't yet exist, based on speculated relatives.

When Spawn/Fork/Compact land, each is its own structured relationship with distinct payload fields. If by then a real `ThreadKind` enum naturally falls out of the implementation, `BehaviorOrigin` can move into it; until then, a flat `Option<BehaviorOrigin>` is the smaller, more honest shape.

## Phasing

Each phase ends in a mergeable PR, green CI. Phases are ordered so the earliest ones prove shape without touching scheduler-loop internals.

1. **Behavior data model + read-side wire.** `BehaviorConfig`, loader (parse `behavior.toml` + `prompt.md` + `state.json` under a pod at startup), persister. `ListBehaviors` / `GetBehavior` wire. `PodSnapshot` carries the behavior catalog. No triggers fire yet; the data is visible but inert.

2. **`RunBehavior` (manual trigger).** Adds `Thread.origin`, headless thread spawn path, retention sweep infrastructure, on-completion hook that updates `BehaviorState`. Proves the full spawn-and-run path with zero new scheduler branches — `RunBehavior` is just another `ClientToServer` message. Webui gains a "Run now" button per behavior.

3. **Cron trigger.** `tokio::time::interval` branch in the scheduler, `cron` dependency, overlap + catch-up policies, restart-safe `state.json`. First autonomous fires.

4. **Webhook trigger.** Route registration on the axum server at behavior load/unload, POST handler pushing `TriggerFired`. Small enough to roll in with phase 3 if the shapes converge.

5. **Webui behaviors panel.** Can land incrementally against earlier phases — the read-side lands with phase 1, the per-behavior detail expansion with phase 2, cron/webhook indicators with phases 3–4.

The phases let the abstraction (`Behavior` as a data shape) stabilize before triggers introduce timing concerns. If it turns out the manual-trigger phase reveals a shape problem in `BehaviorConfig`, we fix it before automated firing pressures the design.

## Not in scope

- **Cross-firing stateful threads.** Every behavior fire spawns a fresh thread. Cross-firing memory is carried by files the behavior writes (via builtin tools + host_env), not by a long-lived conversation. A "persistent observer" behavior that keeps one thread alive across firings is a later refinement driven by a concrete case where fresh-per-fire is insufficient.
- **Report / push destinations.** How a behavior reports its output (email, slack, browser push) is out of scope. The expectation is that any such destination is reached via MCP tools — the behavior's prompt tells it "when done, call `send_slack_message` with your summary." When we get to it, browser push notifications are probably the most useful first target, but not now.
- **Pod self-modification via tool.** A behavior could in principle edit its own `behavior.toml` via a builtin. Deferred until webui-driven editing is stable.
- **Lua hooks.** Still reserved in `pod.toml`; behaviors aren't hooks. Hooks fire around existing thread events; behaviors fire new threads.
- **Richer trigger sources.** Event streams (imap, slack, github push), file watchers, system events. Each wants a durable cursor and an adapter; we'll add them when a use case drives the shape.
- **MCP notification-driven triggers.** See above — needs bidirectional transport first, and the protocol fit is poor anyway.

## Open questions

1. **Retention defaults.** `ArchiveAfterDays(30)` for behavior-spawned threads, `Keep` for interactive threads. A cron firing every 5 min with 30-day retention is ~8600 thread files — readable on `ls`, fine at scale — but revisit if we end up with higher-cadence behaviors. Per-behavior tuning is the escape hatch.
2. **Cron catch-up semantics.** On server restart after missed fires, default to `catch_up = "one"` — fire at most one catch-up per behavior, log the skipped count. Configurable per-behavior; alternatives are `"none"` (skip silently) and `"all"` (fire every missed one; rarely what anyone wants).
3. **Webhook path gating.** Should the pod's `[allow]` table declare which webhook paths its behaviors may claim? Leaning **no** — the behavior declares its own path implicitly (derived from pod_id + behavior_id), the scheduler detects collisions at load time, and pod-level allow-listing is overkill for a namespace-per-behavior scheme. Revisit if we add path customization.
4. **Concurrency interaction.** `pod.limits.max_concurrent_threads` interacts with a behavior's `overlap` setting — e.g., `overlap = "allow"` on a pod at its cap: does the fire wait, drop, or error? Proposal: overlap policy governs *within the behavior*, pod cap governs *across behaviors*; a fire that would exceed the pod cap is always dropped and logged, regardless of overlap.
5. **Archived pods.** Archiving a pod stops its triggers from firing (cron entries de-registered, webhook routes removed). Unarchiving does **not** automatically resume — the user re-enables per behavior, to prevent surprise firings from a pod that may have been archived precisely because its behaviors were misbehaving.
6. **Corrupt `state.json` recovery.** A corrupt state file means we can't read `last_fired_at` or `queued_payload`. Rename to `state.json.corrupt` for forensics, treat the behavior as "never fired," log loudly. Better than refusing to load the pod.
7. **Template substitution in `prompt.md`.** Initial cut: `{{payload}}` expands to pretty-printed JSON. Richer templating (Handlebars, Tera, Jinja) is out of scope for v1. Revisit when we see what behaviors actually want — likely small accessor paths (`{{payload.from}}`) are next, full templates are last.
