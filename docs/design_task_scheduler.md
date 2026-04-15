# Task Scheduler and Wire Protocol

Central architectural pivot following the MVP (`design_mvp.md`). The MVP tied one agent conversation to one WebSocket connection and ran the loop as an async future inside the WebSocket handler. This document replaces that model with explicit tasks driven by a central scheduler, and a multiplexed wire protocol that decouples clients from tasks.

**Why**: our design goals — persistable state, multi-client observation, cancellation and approval pauses, tasks that outlive client connections — don't fit into `async fn`. A compiler-generated future state machine is opaque, non-serializable, and owns its state in a way that precludes out-of-band observation. An explicit state machine with a central scheduler solves all of those directly.

## The shift: tasks as data, scheduler as driver

```
Before (MVP):                            After:
  WebSocket handler                       Client(s)
   └── async fn run()                      └── WebSocket (subscribes to N tasks)
        ├── owns Conversation
        ├── owns McpSession               TaskManager
        └── owns AnthropicClient           ├── HashMap<TaskId, Task>   ← pure data
                                           └── Scheduler               ← single event loop
                                                ├── dispatches I/O
                                                └── applies results to tasks
```

The Task is **pure data**: serializable, observable, snapshotable. The Scheduler drives state transitions and handles I/O, but holds no state that isn't also in a Task. That discipline — "Task is the truth, scheduler is stateless glue" — is what makes persistence and resumption tractable.

## Task data structure

```rust
struct Task {
    id: TaskId,
    created_at: DateTime<Utc>,
    last_active: DateTime<Utc>,
    title: Option<String>,              // derived from first user message
    config: TaskConfig,                 // model, system prompt, MCP hosts, policy
    state: TaskState,
    conversation: Conversation,         // canonical shape from conversation.rs
    total_usage: Usage,
}
```

## Task state

Coarse states corresponding to "rest points" between I/O operations. Fine-grained mid-operation state (partial SSE parse, mid-flight HTTP) lives in the scheduler's scratch space, not the Task.

```rust
enum TaskState {
    Ready,                              // idle, awaiting user input
    AwaitingModel { started_at: DateTime<Utc>, op_id: OpId },
    AwaitingApproval { pending: Vec<PendingApproval> },  // v0.2
    AwaitingTools { pending: Vec<InFlightCall> },
    Failed { error: String, at_phase: &'static str },
    Cancelled,
    Completed,                          // stop_reason=end_turn; open for follow-ups
    Archived,                           // user explicitly closed
}
```

Tasks do **not** auto-terminate on `end_turn`. `Completed` means "this turn is done, waiting for either a follow-up message or an archive command."

The internal state is **collapsed to a public `TaskStateLabel`** on the wire. Internal distinctions (`AwaitingModel` vs `AwaitingTools`) become `Working` for clients. Keeps the protocol stable against future internal-state churn.

```rust
enum TaskStateLabel {
    Idle,              // Ready
    Working,           // AwaitingModel / AwaitingTools collapsed
    AwaitingApproval,
    Completed,
    Failed,
    Cancelled,
}
```

## Scheduler

A single tokio task running one `select!` loop.

```
Scheduler state:
  ├── task_manager: &mut TaskManager
  ├── pending_io:   FuturesUnordered<IoOp>    // HTTP/MCP ops, keyed by (task_id, op_id)
  ├── inbox:        mpsc::Receiver<Input>     // user msgs, cancels, approvals, subscriptions
  └── persister:    &mut Persister            // flushes Task state on transitions

Loop:
  select! {
    completion = pending_io.next() => {
      let task = task_manager.get_mut(completion.task_id);
      task.apply_io_result(completion);
      step_until_blocked(task);
    }
    input = inbox.recv() => {
      let task = task_manager.get_mut(input.task_id);
      task.apply_input(input);
      step_until_blocked(task);
    }
  }

step_until_blocked(task):
  loop {
    match task.step() {
      StepOutcome::DispatchIo(op)   => { pending_io.push(…); break; }
      StepOutcome::Continue         => { continue; }       // pure state transition
      StepOutcome::Paused           => { break; }          // Ready / AwaitingApproval
      StepOutcome::Transitioned     => { persister.maybe_flush(task); }
    }
  }
```

Expected scale: ~10 tasks in flight. Single select-loop is ample. Not a bottleneck we need to engineer against.

### Why this shape

- **All reads of task state are free.** TaskManager owns Tasks; observers just lock the entry.
- **All writes go through one code path** (the scheduler loop). No contention, no interleaving surprises.
- **I/O is multiplexed but not owned by tasks.** The scheduler holds the futures; when a task is cancelled we drop its entries from `pending_io` and transition the Task state. No cooperative-abort dance.
- **Persistence hooks on state transitions.** Every time a Task's state changes, the persister has a chance to flush.

## Persistence

For the current scope:

- JSON file per task at `<state-dir>/tasks/<task_id>.json`.
- Write on every state transition. With coarse states, writes are infrequent (~1 per turn per task).
- Resume on startup: read all files, populate TaskManager, mark any `AwaitingModel` / `AwaitingTools` states as `Failed { at_phase: "resume" }`. There's no clean way to resume in-flight HTTP or tool calls — the user can restart the turn if they want.

Upgrade to SQLite when either (a) cross-task queries become useful (list by date, by title, by state) or (b) write volume makes one-file-per-task unwieldy.

## Wire protocol

Multiplexed WebSocket. One connection per client (browser tab, CLI, remote operator). Client subscribes to the tasks it wants to observe.

**Two tiers of events:**

- **Task-list tier** — lightweight, broadcast to every connected client automatically. Drives task-list UIs without per-task subscription.
- **Per-task turn tier** — high-volume. Only sent to clients explicitly subscribed to that task.

### Client → Server

```rust
enum ClientToServer {
    // Task lifecycle
    CreateTask {
        correlation_id: Option<String>,
        initial_message: String,
        config_override: Option<TaskConfigOverride>,
    },
    SendUserMessage { task_id: String, text: String },
    CancelTask     { task_id: String },
    ArchiveTask    { task_id: String },

    // Observation
    SubscribeToTask     { task_id: String },   // triggers TaskSnapshot + turn event stream
    UnsubscribeFromTask { task_id: String },
    ListTasks { correlation_id: Option<String> },
}
```

### Server → Client

```rust
enum ServerToClient {
    // Task-list tier — broadcast to all clients
    TaskCreated      { task_id: String, summary: TaskSummary, correlation_id: Option<String> },
    TaskStateChanged { task_id: String, state: TaskStateLabel },
    TaskTitleUpdated { task_id: String, title: String },
    TaskArchived     { task_id: String },

    // Request/response
    TaskList     { correlation_id: Option<String>, tasks: Vec<TaskSummary> },
    TaskSnapshot { task_id: String, snapshot: TaskSnapshot },

    // Per-task turn tier — only to subscribers
    TaskAssistantBegin     { task_id: String, turn: u32 },
    TaskAssistantText      { task_id: String, text: String },         // complete text block
    TaskAssistantTextDelta { task_id: String, delta: String },        // streaming partials (v0.2)
    TaskToolCallBegin      { task_id: String, tool_use_id: String, name: String, args_preview: String },
    TaskToolCallEnd        { task_id: String, tool_use_id: String, result_preview: String, is_error: bool },
    TaskAssistantEnd       { task_id: String, stop_reason: Option<String>, usage: Usage },
    TaskLoopComplete       { task_id: String },

    Error { correlation_id: Option<String>, task_id: Option<String>, message: String },
}

struct TaskSummary {
    task_id: String,
    title: Option<String>,
    state: TaskStateLabel,
    created_at: DateTime<Utc>,
    last_active: DateTime<Utc>,
}

struct TaskSnapshot {
    task_id: String,
    config: TaskConfig,
    state: TaskStateLabel,
    conversation: Conversation,    // full canonical shape — client renders
    total_usage: Usage,
    created_at: DateTime<Utc>,
}
```

### Streaming text

`TaskAssistantText` carries complete text blocks (current non-streaming Anthropic behavior). `TaskAssistantTextDelta` will carry streaming partials once we enable SSE — separate variant so the "complete block" semantics are preserved. A complete `TaskAssistantText` is always emitted at turn end, so clients that reconnect mid-stream get a consistent view once the turn settles.

### Conversation rendering

`TaskSnapshot.conversation` is the full canonical `Conversation` (Messages with ContentBlocks — text, tool_use, tool_result, thinking). Client renders to its own display format. Keeps the protocol narrow; the client stays flexible in how it shows things.

### Title derivation

`TaskSummary.title` starts `None`. After the first user message is appended, the server derives a short title (first ~50 chars of the initial message, with trailing ellipsis if truncated) and emits `TaskTitleUpdated`. Later we can swap in model-generated titles; the protocol event is the same.

## Subscription and client model

- On connect: server sends nothing automatically. Client bootstraps with `ListTasks`.
- Task-list events broadcast to every connected client.
- Per-task turn events require explicit `SubscribeToTask`. The server replies with `TaskSnapshot` (full current state), then streams subsequent events.
- Client disconnect: subscriptions drop (cheap — a per-connection `HashSet<TaskId>`). Tasks continue running.
- Client reconnect: re-subscribe to whatever tasks the UI is showing; each subscribe yields a fresh snapshot.

## What this replaces from the MVP

Directly superseded:

- `handle_ws_session` in `src/server.rs`, which tied a single Conversation to a single WebSocket connection. The new handler manages only a per-connection subscription set; all conversation state lives in the TaskManager.
- The `ClientToServer` / `ServerToClient` enum shapes in `whisper-agent-protocol`. Replaced by the multiplexed variants above. Current variants become a small subset under new names.
- The `run_one_shot` CLI path: either reuses the scheduler in-process (probably overkill) or stays as a library-level helper that wraps a single-task run without going through TaskManager. Most likely the latter.

Compatible — no change needed:

- `Conversation`, `Message`, `ContentBlock` in `src/conversation.rs`. Moves into `Task.conversation`.
- The `AnthropicClient` interface. It's now invoked by the scheduler as an I/O op rather than inside the loop.
- The `McpSession` interface. Similarly invoked by the scheduler. A Task's config references which host(s) it has access to; scheduler routes `ToolCall` I/O ops to the right session.
- The `whisper-agent-mcp-host` binary. Nothing changes on the host side.
- The audit log format. The `session_id` field is now `task_id`; every other field is unchanged.

## Not in scope (deferred)

- **Multi-host MCP routing within a single task.** A Task's config mentions one MCP host for now. Federated tool routing comes next.
- **Real cancellation semantics for in-flight tools.** A tool may have executed before the cancel arrives; the current plan just marks Task state `Cancelled` without rolling back. Proper "rollback or mark-undone" semantics are a v0.3 problem.
- **Crash recovery during I/O.** Resume marks `AwaitingModel`/`AwaitingTools` states as `Failed`. Better recovery (retry idempotent ops, re-prompt user for non-idempotent ones) is deferred.
- **Authentication.** Still loopback-only. Per-client identity comes when multi-client auth lands — see `design_permissions.md` Pattern 3.
- **Cross-task operations** (fork conversation, merge turns, branch from a midpoint). Deferred.
- **SSE streaming for Anthropic.** The wire variant exists (`TaskAssistantTextDelta`) so the protocol doesn't need to change when we add it.
