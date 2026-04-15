# Permissions

The most load-bearing problem whisper-agent has to solve. The same architecture that gives a local agent the ability to fix a typo in `~/notes/` gives it the ability to `rm -rf $HOME` by mistake. The same protocol that lets a cloud agent triage k8s alerts could let it read files it has no business reading. Get this wrong and the system is either unsafe or unusable.

This doc lays out the design space. Specifics (UX, storage, exact policy DSL) are deliberately left open until prototypes hit reality.

## Three distinct patterns — we need all three

Permission decisions in agentic systems separate cleanly into three patterns. They solve different problems and don't substitute for one another.

### Pattern 1 — Pre-execution client-side approval

The harness/client decides *before* invoking the tool whether to prompt the user. This is what Claude Code's `--permission-mode` controls.

Inputs to the decision:
- **Tool annotations** declared by the MCP server (`destructiveHint`, `readOnlyHint`, `idempotentHint`, `openWorldHint`).
- **Per-session policy** (auto / prompt / deny per tool name or class).
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

### Pattern 3 — Per-session policy / capability tiers

Orthogonal to the first two. The harness has a session-level policy that determines what a session can do at all, before any per-call decision arises:

- "This session is autonomous (cron-triggered) — auto-approve everything but log."
- "This session is human-driven — prompt for anything destructive."
- "This session is a cloud agent — never let it touch `/home/$USER/private/`."
- "This session is local-only — refuse to even open connections to remote MCP servers."

This is where **"a cloud agent shouldn't access local-only files"** lives. It is *not* a per-call approval problem. It is an **authn/authz design problem that MCP does not standardize**. We build it on top:

- **Connection-level identity**: mTLS cert or per-session bearer token identifies which loop session is connecting to which host MCP server.
- **Per-session tool catalog filtering**: the server returns different `tools/list` based on identity (this session sees `read_file`; that session also sees `exec`).
- **Per-tool authz inside the tool implementation**: even if the tool is exposed, the implementation checks "is this session allowed to read this path?" before acting.
- **Connection-level refusal**: a "sensitive" host MCP server only accepts connections from sessions tagged `local`, with cert pinning.

MCP gives us the wire protocol; the policy and identity model are ours to design and enforce.

## A reframing the headless-loop architecture enables

In a colocated agent (Claude Code), "where the agent runs" and "what it can access" are the same thing — both are the user's machine. That conflation makes Pattern 3 hard to express: there's no clean way to say "this agent is on my machine but should be treated as untrusted."

In whisper-agent's headless-loop architecture, the loop is always remote from the host it acts on, and *every* host connection is mediated by an MCP server with its own policy. That separation makes Pattern 3 natural: the desktop's MCP server is the single place that decides "does this incoming session have desktop-level access?" — independent of where the loop happens to be running. A cloud-resident loop session and a local-resident loop session both connect to the desktop's MCP server the same way, and the desktop's policy (per-session-identity) is what differs.

This is one of the cleanest architectural wins of the headless-loop model.

## Suggested posture

For the first build, all three patterns from day one — but with realistic phasing:

1. **Pattern 1 (pre-execution prompt) is non-negotiable.** Default-prompt for any tool annotated `destructiveHint`. Auto-approve `readOnlyHint` tools by default. User-configurable per session.
2. **Pattern 3 (per-session policy) is the security boundary.** Build the identity model (mTLS or signed session tokens), the per-host policy file format, and the catalog-filtering hook from the start. Even if early policies are coarse ("this session = full access" / "this session = read-only"), the *machinery* must exist.
3. **Pattern 2 (elicitation) is opportunistic.** Wire the client to support it, but don't require host MCP servers to use it. As we build our own host MCP server, use elicitation for the operations that benefit (bulk-destructive ops, large diffs).

Tool annotations (`destructiveHint`, etc.) are the language between server and client for "should this be auto-approved or prompt-by-default." Whatever host MCP servers we write, annotate exhaustively.

**Audit log every approval and denial**, with `(session_id, host, tool_name, args excerpt, decision, who_decided, timestamp)`. Non-negotiable. The audit log is also the input for refining policy over time ("this session keeps prompting on the same tool; auto-approve it").

## Open questions

- **Approval UX.** Web dashboard? Mobile push? Slack DM? Terminal pager? Different sessions want different channels. The harness needs a pluggable approval-channel abstraction.
- **Per-loop-session lifetime of approvals.** "Approve once" vs. "approve for this session" vs. "approve for this session for the next N minutes" vs. "remember always."
- **Audit log storage and retention.** Local file? Postgres? S3? How long do we keep approvals?
- **Policy DSL or just Rust code.** A declarative per-host policy file, or just Rust functions hooked at the right callbacks. Probably start with Rust, add a DSL only if a non-engineer needs to edit policies.
- **How the loop reacts to denial.** When a user denies a tool call, what does the model see in the `tool_result`? "Denied by user" verbatim, or something more structured? Affects whether the model retries differently, gives up, asks for clarification.
- **Sandboxing inside the host MCP server.** Capabilities, namespaces, seccomp — what's the trust boundary the server itself enforces independent of the policy? (E.g., even a fully-trusted session can't escape the server's chroot / namespace.) Defense in depth.

## What this design is NOT

- **Not a guarantee of safety.** Permissions are a layer of defense. They reduce blast radius; they don't make agents foolproof. The audit log is what makes incidents recoverable.
- **Not committed to MCP-only.** If a tool can't be exposed cleanly through MCP (e.g., needs streaming, or needs a non-JSON-schema input grammar), the same three permission patterns must apply at whatever boundary replaces MCP for that tool. The patterns are protocol-independent.
- **Not done.** Each open question above will need a follow-up doc once we've made enough progress to choose intelligently.
