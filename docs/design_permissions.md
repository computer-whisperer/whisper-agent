# Permissions

The most load-bearing problem whisper-agent has to solve. The same architecture that gives a local agent the ability to fix a typo in `~/notes/` gives it the ability to `rm -rf $HOME` by mistake. The same protocol that lets a cloud agent triage k8s alerts could let it read files it has no business reading. Get this wrong and the system is either unsafe or unusable.

This doc lays out the design space. Specifics (UX, storage, exact policy DSL) are deliberately left open until prototypes hit reality.

Note: after the task-scheduler redesign (see [`design_task_scheduler.md`](design_task_scheduler.md)), "session" throughout this doc refers to a **task** — the long-lived agent conversation owned by the TaskManager. Client connections are transient; tasks are durable. Policy attaches to tasks (and to the identity that created them), not to individual WebSocket connections.

## Three distinct patterns — we need all three

Permission decisions in agentic systems separate cleanly into three patterns. They solve different problems and don't substitute for one another.

### Pattern 1 — Pre-execution client-side approval

The harness/client decides *before* invoking the tool whether to prompt the user. This is what Claude Code's `--permission-mode` controls.

Inputs to the decision:
- **Tool annotations** declared by the MCP server (`destructiveHint`, `readOnlyHint`, `idempotentHint`, `openWorldHint`).
- **Per-task policy** (auto / prompt / deny per tool name or class, set at task creation).
- **Per-host policy** (this host requires confirmation for `exec`; that host doesn't).

If the user denies, the tool call is never dispatched. The model receives a `tool_result` saying "denied" and continues from there. Bread-and-butter case — fast, works with every MCP-compliant server, doesn't require any cooperation from the server.

### Pattern 2 — Server-initiated mid-execution elicitation

MCP added native support for this in the **2025-06-18 spec** under the name **elicitation**. It's the answer to "the server knows something at runtime that requires confirmation."

Flow:
- Server begins processing a tool call.
- Mid-execution, the server sends an `elicitation/create` request to the client over a separate server→client RPC channel.
- The client (the harness/fixture, *not* the LLM) surfaces the request to the user.
- The user responds (approve/reject/answer-question).
- The client returns the response. The server proceeds or aborts based on it.

The LLM never sees the elicitation flow — it doesn't appear in the conversation transcript at all. Useful for:
- "I'm about to delete 47 files, confirm?" — the server only realizes the operation is bulk after planning it.
- Diff preview before applying an edit.
- Multi-step operations where each step might warrant approval.

Server cooperation is required, and ecosystem adoption is uneven (the spec is recent), so we shouldn't lean on this exclusively. But it's a real escape valve for context-sensitive prompts.

### Pattern 3 — Per-task policy and per-identity capability tiers

Orthogonal to the first two. Two layered policy attachments determine what a task can do at all, before any per-call decision arises:

- **Task policy** (attached at task creation, carries with the task for its lifetime):
  - "This task is autonomous (cron-triggered) — auto-approve everything but log."
  - "This task is human-driven — prompt for anything destructive."
- **Client/creator identity** (attached to the principal that created the task):
  - "Cloud-originated tasks never access `/home/$USER/private/`."
  - "Only local-resident creators can spawn tasks that talk to the desktop MCP host."

This is where **"a cloud agent shouldn't access local-only files"** lives. It is *not* a per-call approval problem. It is an **authn/authz design problem that MCP does not standardize**. We build it on top:

- **Connection-level identity**: mTLS cert or per-connection bearer token identifies which principal is connecting. The identity carries into any tasks that principal creates.
- **Per-task tool catalog filtering**: the host MCP server returns different `tools/list` based on the calling task's identity (this task sees `read_file`; that one also sees `exec`).
- **Per-tool authz inside the tool implementation**: even if the tool is exposed, the implementation checks "is this task allowed to read this path?" before acting.
- **Connection-level refusal**: a "sensitive" host MCP server only accepts connections from task identities tagged `local`, with cert pinning.

MCP gives us the wire protocol; the policy and identity model are ours to design and enforce.

## A reframing the headless-loop architecture enables

In a colocated agent (Claude Code), "where the agent runs" and "what it can access" are the same thing — both are the user's machine. That conflation makes Pattern 3 hard to express: there's no clean way to say "this agent is on my machine but should be treated as untrusted."

In whisper-agent's headless-loop architecture, the loop is always remote from the host it acts on, and *every* host connection is mediated by an MCP server with its own policy. That separation makes Pattern 3 natural: the desktop's MCP server is the single place that decides "does this incoming task have desktop-level access?" — independent of where the loop runtime is. A cloud-originated task and a local-originated task both connect to the desktop's MCP server the same way, and the desktop's policy (per-identity) is what differs.

This is one of the cleanest architectural wins of the headless-loop model.

## Suggested posture

For the first build, all three patterns from day one — but with realistic phasing:

1. **Pattern 1 (pre-execution prompt) is non-negotiable.** Default-prompt for any tool annotated `destructiveHint`. Auto-approve `readOnlyHint` tools by default. User-configurable per task.
2. **Pattern 3 (per-task + per-identity policy) is the security boundary.** Build the identity model (mTLS or signed bearer tokens), the per-host policy file format, and the catalog-filtering hook from the start. Even if early policies are coarse ("this task = full access" / "this task = read-only"), the *machinery* must exist.
3. **Pattern 2 (elicitation) is opportunistic.** Wire the client to support it, but don't require host MCP servers to use it. As we build our own host MCP server, use elicitation for the operations that benefit (bulk-destructive ops, large diffs).

Tool annotations (`destructiveHint`, etc.) are the language between server and client for "should this be auto-approved or prompt-by-default." Whatever host MCP servers we write, annotate exhaustively.

**Audit log every approval and denial**, with `(task_id, client_identity, host, tool_name, args excerpt, decision, who_decided, timestamp)`. Non-negotiable. The audit log is also the input for refining policy over time ("this task keeps prompting on the same tool; auto-approve it").

## Open questions

- **Approval UX.** Web dashboard? Mobile push? Slack DM? Terminal pager? Different tasks want different channels. The harness needs a pluggable approval-channel abstraction.
- **Per-task lifetime of approvals.** "Approve once" vs. "approve for this task" vs. "approve for this task for the next N minutes" vs. "remember always for this client identity."
- **Audit log storage and retention.** Local file? Postgres? S3? How long do we keep approvals?
- **Policy DSL or just Rust code.** A declarative per-host policy file, or just Rust functions hooked at the right callbacks. Probably start with Rust, add a DSL only if a non-engineer needs to edit policies.
- **How the loop reacts to denial.** When a user denies a tool call, what does the model see in the `tool_result`? "Denied by user" verbatim, or something more structured? Affects whether the model retries differently, gives up, asks for clarification.
- **Sandboxing inside the host MCP server.** Capabilities, namespaces, seccomp — what's the trust boundary the server itself enforces independent of the policy? (E.g., even a fully-trusted session can't escape the server's chroot / namespace.) Defense in depth.

## What this design is NOT

- **Not a guarantee of safety.** Permissions are a layer of defense. They reduce blast radius; they don't make agents foolproof. The audit log is what makes incidents recoverable.
- **Not committed to MCP-only.** If a tool can't be exposed cleanly through MCP (e.g., needs streaming, or needs a non-JSON-schema input grammar), the same three permission patterns must apply at whatever boundary replaces MCP for that tool. The patterns are protocol-independent.
- **Not done.** Each open question above will need a follow-up doc once we've made enough progress to choose intelligently.
