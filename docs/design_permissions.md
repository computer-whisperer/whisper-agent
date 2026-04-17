# Permissions

The most load-bearing problem whisper-agent has to solve. The same architecture that gives an agent the ability to fix a typo in a notes directory gives it the ability to `rm -rf $HOME` by mistake. The same protocol that lets a remote agent triage k8s alerts could let it read files it has no business reading. Getting this wrong makes the system either unsafe or unusable.

This doc describes what's implemented today, what the ceiling looks like, and what's deliberately deferred.

## Three distinct patterns

Permission decisions in agentic systems separate cleanly into three patterns. They solve different problems and don't substitute for one another.

### Pattern 1 — Pre-execution client-side approval *(implemented)*

The harness decides *before* invoking a tool whether to prompt the user. This is our primary line of defense for interactive threads.

Inputs to the decision:

- **Tool annotations** from the MCP server: `readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`. Carried on `ToolAnnotations` in `src/tools/mcp.rs`.
- **Thread's `ApprovalPolicy`** (`crates/whisper-agent-protocol/src/lib.rs`):
  - `AutoApproveAll` — auto-approve every call. Appropriate for unattended threads where the sandbox is the safety boundary.
  - `PromptPodModify` *(default for new pods)* — auto-approve read-only and non-pod-modifying tools; prompt on the builtin pod-editing tools. The sandbox bounds blast radius for everything else; self-modification of the pod's own config gets a distinct gate because it's a privilege-escalation vector.
  - `PromptDestructive` — auto-approve only tools the MCP server marked `readOnlyHint: true`; prompt on destructive MCP tools *and* pod-modifying builtins. Strictest useful policy, for sandbox-free or human-in-the-loop configurations.
- **Per-thread `tool_allowlist`**. When the user approves a call with "remember," the tool name is added to the thread's allowlist and future calls to that name skip the prompt. Persisted with the thread.

If the user denies, the tool call is never dispatched — the model receives a synthesized `tool_result` saying it was denied and continues from there. The state-machine variant `AwaitingApproval` handles the pause; `ApprovalDisposition` tracks per-call outcome (`AutoApproved | Pending | UserApproved | UserRejected`).

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

**Sandboxing** is the load-bearing safety boundary for unattended work. Each thread binds to a `host_env` — a named (provider, spec) pair declared in the pod's `[[allow.host_env]]` table. The provider (today: `local-landlock`, via `whisper-agent-sandbox`) provisions the host-env for the thread and spawns the MCP host inside it, scoped to the allowed paths and network policy. Tools running inside can't escape the landlock ruleset regardless of what they're asked to do. See [sandbox architecture memory](../) for the layering rationale: per-task `SandboxSpec`, `SandboxBackend` trait, provisioned below the MCP layer because MCP's roots (advisory `file://` URIs) don't carry an image, mount mode, or network policy — real isolation has to come from below.

This is why `AutoApproveAll` is a sensible default for autonomous behaviors: the sandbox bounds blast radius regardless of what the model decides to do. For interactive use where the sandbox is relaxed, `PromptPodModify` gives a second line of defense around self-modification.

**Audit.** Every tool call writes one line to a JSONL audit log (`src/runtime/audit.rs`). Shape:

```rust
struct ToolCallEntry {
    timestamp: DateTime<Utc>,
    thread_id: &str,
    host_id: &str,
    tool_name: &str,
    args: Value,
    decision: &str,       // "auto" | "approve" | "reject"
    who_decided: &str,    // "policy:read_only" | "policy:auto_approve_all" |
                          // "policy:allowlist" | "user:{conn_id}"
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

- `PendingApproval { approval_id, tool_use_id, name, args_preview, destructive, read_only }` events when a call needs human review.
- `ClientToServer::ApprovalDecision { thread_id, approval_id, choice, remember }` client→server.
- `ApprovalResolved { approval_id, decision, decided_by_conn }` events when a decision lands.
- `AllowlistChanged { allowlist }` when the thread's allowlist changes (via remember-yes or explicit revoke).

The pod config carries `thread_defaults.approval_policy` (a `PodConfig` field); per-thread overrides live in `ThreadConfig`.

## Open questions

- **Approval UX surfaces.** Web dashboard today. Future: mobile push, Slack DM, terminal pager. Different threads want different channels; the harness would need a pluggable approval-channel abstraction. No concrete need yet.
- **Approval lifetime granularity.** Today: "approve this one call" vs. "approve for this thread" (the remember toggle). No "approve for N minutes" or "remember always for this identity" — waiting for a use case.
- **Policy DSL vs. Rust code.** Approval policy is a closed enum today; extending to something declarative is premature until there's a non-engineer who needs to edit policies.
- **Reaction to denial.** When a user denies, the synthesized `tool_result` carries the user's message. Is that the right framing? Does the model retry, give up, ask for clarification? Empirically it mostly gives up cleanly, but the right denial payload is underexplored.
- **Audit log retention.** Local JSONL, no rotation, no retention policy. Fine at current scale; revisit when log volume or durability becomes a concrete concern.

## What this design is NOT

- **Not a guarantee of safety.** Permissions reduce blast radius; they don't make agents foolproof. The sandbox is the hard perimeter. The audit log is what makes incidents recoverable.
- **Not committed to MCP-only.** If a tool can't be exposed cleanly through MCP, the same three permission patterns still apply at whatever boundary replaces MCP for that tool. The patterns are protocol-independent.
- **Not finished.** Pattern 1 is real; Patterns 2 and 3 wait for concrete need.
