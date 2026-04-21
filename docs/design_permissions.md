# Permissions

The most load-bearing problem whisper-agent has to solve. The same architecture that gives an agent the ability to fix a typo in a notes directory gives it the ability to `rm -rf $HOME` by mistake. The same protocol that lets a remote agent triage k8s alerts could let it read files it has no business reading. Getting this wrong makes the system either unsafe or unusable.

This doc describes what's implemented today, what the ceiling looks like, and what's deliberately deferred.

## Three distinct patterns

Permission decisions in agentic systems separate cleanly into three patterns. They solve different problems and don't substitute for one another.

### Pattern 1 — Pre-execution client-side approval *(implemented)*

The scheduler's Function registry decides *before* invoking a tool whether to prompt the user. This is our primary line of defense for interactive threads.

Every tool_use emitted by the model registers a `Function::BuiltinToolCall` or `Function::McpToolUse` (or for `dispatch_thread`, a `Function::CreateThread` via its pool-alias path). The Function's scope-check is a per-tool `Disposition`:

- `Allow` — admit; push the IO future immediately.
- `AllowWithPrompt` — admit, but buffer the IO and emit a `PendingApproval` wire event. User's decision resolves through `resolve_tool_approval` — approve pushes the buffered IO future; reject synthesizes a tool-result error and completes the Function with `Cancelled(UserDenied)`.
- `Deny` — refuse registration; the thread sees a tool-result error and continues.

Inputs to the disposition:

- **Tool annotations** from the MCP server (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`) — carried on `ToolAnnotations` in `src/tools/mcp.rs` and surfaced in the `PendingApproval` wire event for client rendering.
- **Pod's `[allow.tools]` table** — `AllowMap<String>` with a `default` Disposition plus per-tool overrides. Replaces the old `ApprovalPolicy` preset enum: pods now declare exactly which disposition applies by default and which tools override.
- **Per-thread `tools_scope`** — an `AllowMap<String>` snapshotted from the pod's `[allow.tools]` at thread creation (`src/runtime/scheduler.rs::create_task`). When the user approves a call with "remember," that tool name gets a per-tool `Allow` override on `tools_scope` (and the older name `tool_allowlist` is kept around on `Thread` for back-compat, mirroring the same narrow set). Future calls to that name skip the prompt. Persisted with the thread.

Pre-Function-model references to `ApprovalPolicy` / `AwaitingApproval` / `ApprovalDisposition` were removed during the Function-registry migration — see `docs/design_functions.md`. The scheduler's `register_tool_function` is the single dispatch entry point; the thread has no separate approval state.

**Known gap: `tools_scope` is a frozen snapshot.** `UpdatePodConfig` (`src/runtime/scheduler/config_updates.rs::apply_pod_config_update`) does not re-sync a thread's `tools_scope` from the pod's current `[allow.tools]`. A thread that self-edits its own pod.toml via `pod_write_file` will not see its own tool scope change for the rest of its lifetime — only newly-created threads pick up the new map. The permissions rework (`design_permissions_rework.md`) promotes this behavior from gap to intended semantic.

**Known gap: `dispatch_thread` with `AllowWithPrompt` does not prompt.** `src/runtime/scheduler/functions.rs::register_dispatch_thread_tool` logs a `warn!` and proceeds as `Allow`. Buffering the pre-launch state for a Function-spawning tool alias takes more plumbing than the current `pending_approval_io: IoRequest` buffer supports, and no concrete use case has required it yet. Tracked in `design_functions.md`'s open questions.

### Pattern 2 — Server-initiated mid-execution elicitation *(deferred)*

MCP's 2025-06-18 spec defines **elicitation** — a server→client RPC channel for "the server knows something at runtime that requires confirmation." Server pauses mid-call, sends `elicitation/create`, client surfaces the prompt, user responds, server continues or aborts.

We don't use elicitation. Our MCP client (`src/tools/mcp.rs`) sends `notifications/initialized` after handshake and otherwise doesn't implement the server→client notification channel. Adding it requires a bidirectional transport (SSE or WebSocket) on top of the current plain HTTP POST. When we hit a concrete use case — large-diff preview, "confirm deleting 47 files" — we'll wire it up. Until then, Pattern 1 carries the full load.

### Pattern 3 — Per-identity capability tiers *(deferred)*

Orthogonal to Patterns 1 and 2. Before any per-call decision, *who is running this thread* determines what it can see at all. "A remote agent shouldn't access local-only files" is not a per-call approval problem; it's an authn/authz problem MCP doesn't standardize. The intended shape:

- **Connection-level identity** via mTLS or signed bearer tokens; the identity carries into any threads that principal creates.
- **Per-thread tool catalog filtering** — the host MCP server returns a different `tools/list` based on the calling thread's identity.
- **Per-tool authz inside the tool implementation** — even if a tool is exposed, the implementation checks "is this thread allowed to read this path?" before acting.
- **Connection-level refusal** — sensitive hosts only accept connections from identities tagged appropriately.

Today this is out of scope: the transport is loopback-only, there is no identity layer. The pod/host-env/MCP layering is structurally compatible with Pattern 3 — host_env providers could refuse to provision sandboxes for unauthorized principals, MCP hosts could filter tools per principal — but nothing enforces that yet.

## How the current defenses compose

Pattern 1 is one layer. The others are **sandboxing** and **audit**.

**Sandboxing** is the load-bearing safety boundary for unattended work. A thread binds to one or more `host_env` entries — each a named (provider, spec) pair declared in the pod's `[[allow.host_env]]` table. The provider (today: `local-landlock`, via `whisper-agent-sandbox`) provisions the host-env for the thread and spawns the MCP host inside it, scoped to the allowed paths and network policy. Tools running inside can't escape the landlock ruleset regardless of what they're asked to do. The pod's `[[allow.host_env]]` entries are the only place allowed_paths can be declared — there is no mechanism for narrowing `allowed_paths` per-dispatch, so a thread runs with the full path set of every named host-env it is bound to. See [sandbox architecture memory](../) for the layering rationale: per-task `SandboxSpec`, `SandboxBackend` trait, provisioned below the MCP layer because MCP's roots (advisory `file://` URIs) don't carry an image, mount mode, or network policy — real isolation has to come from below.

This is why an all-`Allow` default `[allow.tools]` is sensible for autonomous behaviors: the sandbox bounds blast radius regardless of what the model decides to do. For interactive use where the sandbox is relaxed, per-tool `AllowWithPrompt` overrides — typically on the builtin pod-editing tools and any destructive MCP tools — give a second line of defense around privilege-escalation vectors.

**Audit.** Every tool call writes one line to a JSONL audit log (`src/runtime/audit.rs`). Shape:

```rust
struct ToolCallEntry {
    timestamp: DateTime<Utc>,
    thread_id: &str,
    host_id: &str,
    tool_name: &str,
    args: Value,
    decision: &str,       // "auto" | "approve" | "reject" | "dispatched"
    who_decided: &str,    // "scheduler_gated" | "policy" | "user:{conn_id}"
    outcome: ToolCallOutcome,  // Ok { is_error } | Failed { message }
}
```

The log is append-only, machine-readable, and has existed since the MVP. It's the input for any future policy tightening ("this thread keeps prompting on the same tool; auto-approve it") and the forensic trail if something goes wrong.

## The reframing the headless-loop architecture enables

In a colocated agent, "where the agent runs" and "what it can access" are the same thing — both are the user's machine. That conflation makes Pattern 3 hard to express: there's no clean way to say "this agent is on my machine but should be treated as untrusted."

In whisper-agent's headless-loop architecture, the loop is remote from the hosts it acts on, and every host connection is mediated by an MCP server with its own policy. That separation makes Pattern 3 natural: the host's MCP server is the single place that decides "does this incoming thread have desktop-level access?" — independent of where the loop runtime is. Both local and remote principals would connect to that server the same way; per-identity policy is what differs.

This is one of the cleanest architectural wins of the model, and the reason we're careful not to short-circuit it by slipping per-principal logic into the loop runtime.

## What's in the protocol wire

Pattern 1 surfaces as:

- `ServerToClient::ThreadPendingApproval { thread_id, approval_id, tool_use_id, name, args_preview, destructive, read_only }` when a call needs human review.
- `ClientToServer::ApprovalDecision { thread_id, approval_id, choice, remember }` client→server.
- `ServerToClient::ThreadApprovalResolved { thread_id, approval_id, decision, decided_by_conn }` when a decision lands.
- `ServerToClient::ThreadAllowlistUpdated { thread_id, tool_allowlist }` when the thread's allowlist changes (via remember-yes or explicit revoke). Internally the thread emits a `ThreadEvent::AllowlistChanged` which the router translates to the wire variant.

The pod config carries `[allow.tools]` as an `AllowMap<String>` (default + per-tool overrides); each thread snapshots the pod's tools map into its own `tools_scope` at creation, which it narrows further via remember-approvals. See `docs/design_functions.md` for the full scope-composition model.

## Open questions

- **Approval UX surfaces.** Web dashboard today. Future: mobile push, Slack DM, terminal pager. Different threads want different channels; the harness would need a pluggable approval-channel abstraction. No concrete need yet.
- **Approval lifetime granularity.** Today: "approve this one call" vs. "approve for this thread" (the remember toggle). No "approve for N minutes" or "remember always for this identity" — waiting for a use case.
- **Policy DSL vs. Rust code.** `AllowMap<Disposition>` is declarative enough for the current shape (default + per-tool overrides); a richer policy grammar (conditionals on args, time-of-day, etc.) is premature until there's a non-engineer who needs to edit policies.
- **Reaction to denial.** When a user denies, the synthesized `tool_result` carries the user's message. Is that the right framing? Does the model retry, give up, ask for clarification? Empirically it mostly gives up cleanly, but the right denial payload is underexplored.
- **Audit log retention.** Local JSONL, no rotation, no retention policy. Fine at current scale; revisit when log volume or durability becomes a concrete concern.

## What this design is NOT

- **Not a guarantee of safety.** Permissions reduce blast radius; they don't make agents foolproof. The sandbox is the hard perimeter. The audit log is what makes incidents recoverable.
- **Not committed to MCP-only.** If a tool can't be exposed cleanly through MCP, the same three permission patterns still apply at whatever boundary replaces MCP for that tool. The patterns are protocol-independent.
- **Not finished.** Pattern 1 is real; Patterns 2 and 3 wait for concrete need.
