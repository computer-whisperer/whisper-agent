# Permissions (Rework)

This doc describes the target shape for whisper-agent's permission system. The active design is in [`design_permissions.md`](design_permissions.md); the state of the code behind it is a mix of landed infrastructure and dead scaffolding — `PermissionScope` claims to gate five categories but only `tools` is consulted at runtime; `AllowWithPrompt` carries a per-call-approval model inherited from Claude Code that doesn't fit our setting; pod-editing tools use a hand-coded `is_pod_modify` boolean rather than a typed capability; a 317-line `apply_rebind` supports mid-thread bindings changes no client invokes. This rework replaces that with a single enforcement type, an explicit-escalation approval model, and typed capabilities for sub-tool governance.

**Status:** design sketch. Implementation staging is an open question (see "Migration" at the end).

## The invariant

> A thread can never cause a Function to run with permissions exceeding its own scope.

Everything that follows — dispatch narrowing, behavior caps, pod-file protections, the approval flow — is a corollary. Permissions only reduce blast radius if this invariant holds; if it doesn't, the rest is theatre.

## Threat model

The primary threat is **privilege escalation by a misaligned or confused agent.** Concrete vectors we're closing:

- A narrowed thread dispatching a child thread with wider scope. Today's `dispatch_thread` does exactly this: child scope is computed from the pod's `[allow]`, not the parent's effective scope.
- A thread authoring a behavior whose scope exceeds its own. Behaviors run autonomously later; a thread that can widen a future run is a thread that can widen itself via proxy.
- A thread editing `pod.toml`'s `[allow]` table to widen future threads (or, if the editing thread's scope refreshes from pod.allow, itself).
- An autonomous thread silently stuck on an approval prompt because some tool carried `destructiveHint`. Not an escalation, but a correctness failure the current model makes easy.

Not in scope: remote-principal identity (Pattern 3 in the active doc, still deferred); sandbox escape (orthogonal — landlock remains the hard perimeter); model alignment at the content level (permissions constrain capability, not wisdom-of-use).

## Scope

One type, used at every layer: pod cap, thread effective permission, caller permission for a Function registration. Narrowing is the same operation everywhere.

```rust
struct Scope {
    // Resources the scope may bind to. Named-entry granularity; path
    // narrowing on a host-env is expressed by the pod author creating
    // additional `[[allow.host_env]]` entries, not a sub-structure on
    // the scope.
    backends:  SetOrAll<String>,
    host_envs: SetOrAll<String>,     // allow-table entry names
    mcp_hosts: SetOrAll<String>,     // shared catalog names

    // Simple allow/deny per tool. No AllowWithPrompt — prompting is now
    // a scope-widening action, not a tool attribute.
    tools: AllowMap<ToolName>,

    // Typed capabilities for governance concerns that need more grain
    // than "in the tool catalog / not." Each enum is totally ordered,
    // least- to most-privileged.
    pod_modify: PodModifyCap,        // None | Memories | Content | ModifyAllow
    dispatch:   DispatchCap,         // None | WithinScope
    behaviors:  BehaviorOpsCap,      // None | Read | AuthorNarrower | AuthorAny

    // Approval surface. Only Interactive scopes can request widening;
    // autonomous scopes cannot escalate.
    escalation: Escalation,          // Interactive { via_conn } | None
}

enum SetOrAll<T> {
    All,                 // every name the pod catalog admits
    Only(BTreeSet<T>),   // exactly these names
}
```

Narrowing is pointwise:

- `SetOrAll`: `All ∩ X = X`, `Only(a) ∩ Only(b) = Only(a ∩ b)`.
- `AllowMap<Tool>`: per-tool `min` on the `Allow < Deny` restrictiveness order; `default` takes the more restrictive of the two.
- Typed caps: `min(self, other)` on the enum's order.
- `Escalation`: `Interactive{c} ∩ Interactive{c} = Interactive{c}`; any mismatch or `None` on either side collapses to `None`.

**Scope is what a thread is *permitted to do*, not what it's *currently bound to*.** A thread whose scope admits `host-env = "narrow"` but has never been bound there can still author a behavior using `"narrow"` — it's within scope. Current bindings are runtime state; scope is permission.

## Typed capabilities

Tools that operate on a surface with internal permission structure check a **capability type** rather than taking a boolean scope gate. The tool implementation does the typed check at entry.

### `PodModifyCap`

Governs *writes* to the pod directory. Reads are gated separately by
the pod filename allowlist — inspecting pod-internal state
(`threads/*.json`, etc.) is not a capability-raising action and
doesn't need cap-level permission. The cap hierarchy is specifically
for *what an agent is authorized to modify*.

| Level | Write-admits |
|---|---|
| `None` | nothing |
| `Memories` | `memory/**` |
| `Content` | `Memories` + `system_prompt.md`, `behaviors/*/prompt.md`, `behaviors/*/behavior.toml` |
| `ModifyAllow` | `Content` + `pod.toml` |

`Memories` is the baseline most threads run with. `Content` is for a trusted interactive thread the user has authorized to maintain prompts or behaviors. `ModifyAllow` sits at the top — modifying `pod.toml` (especially `[allow]`) potentially widens *future* threads, so nearly all `ModifyAllow` actions should route through explicit user approval at the moment of action, not a blanket grant at thread creation.

Source of truth is a single helper:

```rust
impl PodModifyCap {
    fn admits(self, path: &RelPath) -> bool { ... }
}
```

Every pod-editing tool routes its path through this one predicate; adding a new subdirectory to the pod surface means extending the helper, not patching every call site.

### `DispatchCap`

| Level | Meaning |
|---|---|
| `None` | Thread cannot spawn child threads. |
| `WithinScope` | Thread can spawn children with `scope ≤ self.scope`. |

### `BehaviorOpsCap`

| Level | Meaning |
|---|---|
| `None` | No access to the behavior subsystem. |
| `Read` | Can list / read behavior configs and prompts. |
| `AuthorNarrower` | Can create/modify behaviors whose declared scope is strictly narrower than the thread's own. |
| `AuthorAny` | Can create/modify behaviors with any scope ≤ pod.allow. |

The `AuthorNarrower` vs `AuthorAny` distinction captures the "smart-trusted thread authoring a cheap less-trusted behavior" case. A smart thread running at pod-wide scope authoring a narrower behavior is the common path and falls under `AuthorNarrower`. Authoring a behavior at the author's own scope is `AuthorAny`.

## The catalog *is* the scope

The tool catalog presented to the model is a projection of its scope:

- Tools admitted by `scope.tools` → visible with full description.
- Tools denied → not in the catalog at all.
- `request_escalation` is visible iff `scope.escalation` is `Interactive`.
- Cap-gated tools (`pod_read_file`, `pod_write_file`, `pod_edit_file`) appear in the catalog if the relevant cap is above `None`; the tool description tells the model which paths its current cap admits, so the model doesn't have to fail-then-discover.

A model cannot invoke a tool it cannot see. A model cannot escalate if `request_escalation` is absent. Autonomous threads (`escalation: None`) are silent at any out-of-scope action — no pending state, no auto-deny timeout. The fail mode is just "tool not in catalog."

## Approval

**Approval is scope-widening, not call-gating.** There is no per-call approval prompt. A thread either has a capability or it doesn't; if it needs one it doesn't have, it uses an explicit tool to request a widening.

### `request_escalation`

One tool in the catalog, typed-union arg schema:

```jsonc
request_escalation({
  variant: "add_tool",
  name: "bash",
  reason: "need to run the test suite"
} | {
  variant: "raise_pod_modify",
  target: "content",
  reason: "want to update system_prompt.md to add X"
} | {
  variant: "raise_behaviors",
  target: "author_any",
  reason: "want to create a behavior at this scope"
} | ... )
```

Server-side, each variant maps to a typed `Function::RequestEscalation*` variant. Typed Functions give the approval UI a concrete payload shape (not stringly-typed JSON), clean per-variant audit, and a natural place to hang per-variant policy.

Model-facing surface stays economical: **one tool**, not one-per-kind — we already have too many tools and the sum type keeps the catalog compact.

Behavior on accept: the thread's scope widens in place, persisted with the thread; subsequent catalog snapshots include the new capability; the model's tool-result says "granted, <X> is now available" and the model continues.

Behavior on reject: tool-result carries the rejection reason; model adapts.

Behavior with `escalation: None`: the tool is absent from the catalog, so the model can't call it, so the server never sees a rejected-without-channel case. Autonomous = silent failure on out-of-scope, period.

### Thread-level vs pod-level widenings

- Widening a *tool* or *typed cap* on a thread affects only that thread. Thread-scope approval.
- Widening the *pod's* `[allow]` (raising its ceiling for *future* threads) is a different concern — pod-scope approval, and the approver should see it framed differently. Concretely it's a `pod_write_file` call on `pod.toml`, which requires `pod_modify: ModifyAllow` *and* typically a per-action approval, and the approval UI names the specific allow-table change being requested.

`request_escalation { add_tool: "bash" }` adds `bash` to *this thread's* catalog. It does not change `pod.toml`. If the user wants `bash` permanently, that's a separate `pod_write_file` action with its own approval.

### What about "approve just this call"?

The new model doesn't have that concept. Approval grants a capability for the thread's life. If the user wants tight control, they grant later rather than earlier; or they revoke after. "Grant for one call only" is a time-boxed capability — a future extension, not a current one.

## Dispatched sub-threads

A thread dispatches a child via `dispatch_thread` (mapped to `Function::CreateThread`). The child's scope is:

```
child.scope = parent.scope.narrow(requested_bindings_and_caps)
```

- If `parent.scope.dispatch == None`, dispatch fails at registration. The tool is also absent from the catalog in that case — same principle as `request_escalation`.
- The request can narrow — never widen — any field of the parent's scope.
- The child's `escalation` is whatever `parent.scope.escalation ∩ requested` resolves to. In practice an interactive parent can grant its channel to a child or withhold it; an autonomous parent can only produce autonomous children.

This is the key escalation-vector fix. Today `resolve_bindings_choice` validates a dispatch against the pod's `[allow]`; the new rule routes through scope narrowing, which is monotonic by construction.

## Behaviors

A behavior in `behavior.toml` declares its runtime scope inline (matching the pod's `[allow]` shape below):

```toml
[scope]
backends  = ["anthropic"]
host_envs = ["narrow-workspace"]
mcp_hosts = ["fetch"]

[scope.tools]
default = "allow"

[scope.tools.overrides]
bash = "deny"

[scope.caps]
pod_modify = "none"
dispatch   = "none"
behaviors  = "none"
```

Behavior-spawned threads run with `pod.allow.narrow(behavior.scope)` at fire time — pod `[allow]` is still the hard cap. If the pod's allow tightens, a previously-authored behavior scope may no longer fit; the thread fails to provision and the behavior run is marked failed.

Behaviors are always autonomous (`escalation: None`). They cannot request widening. Out-of-scope actions fail cleanly.

A thread authoring or editing a behavior declares that behavior's scope by writing the TOML. Two gates check:

- `pod_modify: Content` (or higher) admits the file write.
- `behaviors: AuthorNarrower` admits behavior scopes strictly narrower than the author's; `AuthorAny` admits scopes up to `pod.allow`.

Both gates must pass. This lets a pod author grant "can maintain behaviors" (via `Content` + `AuthorNarrower`) without also granting "can mint behaviors at your own power level" (which is `AuthorAny`).

## Pod config

Pod `[allow]` is the absolute ceiling. Thread and behavior scopes derive from it. Editing `pod.toml` is the highest-privilege action in the system.

Two key semantics:

1. **Pod edits do not affect the editing thread's own running scope.** Scope is snapshotted at thread creation; edits apply to future threads and behavior runs. This closes the direct "edit pod.toml → widen self" escalation path and also resolves the existing `tools_scope`-doesn't-refresh bug by making the freeze intentional.

2. **Compaction resets the successor thread's scope.** When a compacted thread's successor is created, its scope is derived fresh from `pod.allow.narrow(thread_defaults)` — prior widenings granted to the predecessor do *not* carry over. The compaction-boundary note in the successor's conversation surfaces this so the user knows they may need to re-grant. This falls out naturally from scope being a per-thread snapshot; no special data-model work needed.

Practical consequence of (1): a user asking an interactive thread to "add bash to my pod" does not silently widen the asking thread. It widens the *next* thread. If the thread itself wants `bash`, it uses `request_escalation { add_tool: "bash" }` — same user, different request, correctly framed as "widen this thread" vs "widen future threads."

### TOML schema

Pod config grows two sub-tables from today's shape. The `[allow.caps]` block declares the pod's ceiling for typed capabilities; `[thread_defaults.caps]` declares what a freshly-created thread starts with. The split mirrors the existing `backends` / `host_env` / `mcp_hosts` split between `[allow]` (what's permitted) and `[thread_defaults]` (what's picked on creation). Both UI panels and LLMs edit the same file via `pod_write_file` / `pod_edit_file`; the schema sticks to plain tables and string-enum values so either can produce valid output without knowing TOML niceties.

```toml
name        = "whisper-agent dev"
description = "Working on whisper-agent itself"
created_at  = "2026-04-20T10:23:11Z"

[allow]
backends  = ["anthropic", "openai"]
mcp_hosts = ["fetch", "search"]

[allow.tools]
default = "allow"

[allow.tools.overrides]
bash = "deny"

[allow.caps]
# Ceiling for each typed cap. Threads in this pod may hold a cap up
# to but not exceeding this level. `request_escalation` cannot widen
# a thread past these.
pod_modify = "modify_allow"   # none | memories | content | modify_allow
dispatch   = "within_scope"   # none | within_scope
behaviors  = "author_any"     # none | read | author_narrower | author_any

[[allow.host_env]]
name     = "default"
provider = "local-landlock"
type     = "landlock"
allowed_paths = ["/home/me/project:rw", "/:ro"]
network  = "unrestricted"

[thread_defaults]
backend            = "anthropic"
model              = "claude-sonnet-4-6"
system_prompt_file = "system_prompt.md"
max_tokens         = 16384
max_turns          = 30
host_env           = ["default"]
mcp_hosts          = ["fetch", "search"]

[thread_defaults.caps]
# Defaults for a freshly-created thread. Baseline: memories-scoped
# pod access, can dispatch narrower children, can read (but not
# author) behaviors. User or authoring thread may pick anything <=
# the matching [allow.caps] entry at create time.
pod_modify = "memories"
dispatch   = "within_scope"
behaviors  = "read"

[limits]
max_concurrent_threads = 10
```

Tools stay under `[allow.tools]` (unchanged from today, minus the `allow_with_prompt` disposition). Caps live under a separate `[allow.caps]` table because they're structurally different — typed enums rather than per-name maps. Everything the pod author or an LLM needs to edit is expressible as string-valued keys or short arrays of strings.

Validation (the replacement for today's `src/pod.rs::validate`):

- `thread_defaults.caps.*` ≤ `allow.caps.*` for each cap.
- `thread_defaults.backend` ∈ `allow.backends`.
- `thread_defaults.host_env` ⊆ names in `allow.host_env`.
- `thread_defaults.mcp_hosts` ⊆ `allow.mcp_hosts`.

## Scope construction per caller

- **WS client (interactive user).** Scope is full-trust today (`Scope::allow_all` — which, unlike the current `PermissionScope::allow_all`, actually admits everything). Per-identity scope is Pattern 3, still deferred.
- **Thread tool call.** Scope is the thread's effective scope: `pod.allow.narrow(thread.declared)` computed at creation, snapshotted on the thread.
- **Behavior firing.** Scope is `pod.allow.narrow(behavior.scope)` at fire time.
- **Scheduler internal (auto-compact, cron firing).** Scope is the target operation's scope — the owning thread's scope for auto-compact, the behavior's scope for cron.

The scope lives on the `ActiveFunction` alongside the `CallerLink`. Scope-checks happen at Function registration; variants don't re-check during execution.

## What dies

Concrete deletions this rework enables (ordered roughly by blast radius):

- `ClientToServer::RebindThread`, `ThreadBindingsPatch`, `Function::RebindThread`, `apply_rebind`, and companion resource-refcount-diff logic. No client invokes rebind; deleting it removes ~350 lines and takes "bindings change mid-thread" out of the permissions design entirely.
- `Disposition::AllowWithPrompt` at the tool level. Tool dispositions collapse to `Allow` / `Deny`. Prompting happens via `request_escalation`.
- `ThreadPendingApproval` / `ThreadApprovalResolved` wire variants and the `pending_approval_io` buffer on `ActiveFunction`. Replaced by the escalation-request wire shape.
- `tool_allowlist: BTreeSet<String>` on `Thread`. Redundant with `tools_scope` (now just `scope.tools`) and the source of the `remove_from_allowlist` revoke-doesn't-revoke bug.
- `PermissionScope::backends` / `host_envs` / `mcp_hosts` as unused `Vec<String>`. Replaced by `Scope.{backends,host_envs,mcp_hosts}: SetOrAll<T>` which are actually consulted at registration.
- `PermissionScope::backend()` / `host_env()` / `mcp_host()` unused admission methods — replaced by calls that run in real dispatch paths.
- `PermissionScope::allow_all()` / `deny_all()`'s misnamed semantics — `Scope::allow_all()` actually admits everything.
- `is_pod_modify` boolean check in `builtin_tools.rs`. Replaced by `PodModifyCap::admits(path)` at tool entry.
- `dispatch_thread`'s `AllowWithPrompt` warn-and-bypass (`register_dispatch_thread_tool`'s line 896 workaround). `AllowWithPrompt` doesn't exist anymore, and dispatch is gated by `DispatchCap` + scope narrowing instead.
- `register_dispatch_thread_tool`'s ~160-line duplication of `register_tool_function`. One registration path, parameterized by Function variant.

## What stays

- `AllowMap<T>` and `Disposition` (minus `AllowWithPrompt`) in `whisper-agent-protocol::permission`. Clean primitives; keep.
- `resolve_bindings_choice` — 94 lines, focused, fine. Reshaped to produce a `Scope` rather than a raw bindings struct, but conceptually unchanged.
- Audit log structure. Gains richer `who_decided` values and carries scope-widening requests / grants as first-class entries.
- Sandbox layer below MCP. Scope narrows named bindings; the sandbox is still the hard filesystem/network perimeter. Unchanged.
- Per-thread `[allow.tools]` snapshot at creation. Frozen is now the documented intended behavior, not an accident.

## What's new

- `Scope` type with typed capabilities as the single enforcement shape.
- `request_escalation` tool and typed Function variants.
- Tool-catalog projection from scope (today the catalog is pod `[allow]` minus a few filters).
- `PodModifyCap::admits(path)` check at entry of every pod-editing tool.
- Narrowing plumbed through `dispatch_thread`, behavior firing, and behavior authoring — enforcing the core invariant at every boundary.
- Scope-state surface in the webui thread panel (replaces today's allowlist row with a fuller capability summary).

## Open questions

- **Scope snapshot shape on the wire.** Thread snapshots need to carry the whole scope for the UI to render "what am I allowed to do" inline with the thread view. Proposal: a flat struct in the thread snapshot, refreshed whenever an escalation grant or rebind changes it. No per-field subscription — scope mutations are rare and the whole-struct payload is small. Detail to finalize when we shape the wire events.
- **Per-behavior scope ergonomics.** The behavior `[scope]` block duplicates the pod `[allow]` schema. Inline (current sketch) is verbose but self-contained; a future "scope template" indirection (behavior references a named scope in the pod) could compress it. Lean inline for now; revisit if behaviors become numerous and the duplication bites.
- **`request_escalation` argument schema.** The variant sketch covers the common cases (add_tool, raise_pod_modify, raise_behaviors). The full enumeration — including host-env widening, mcp-host widening, backend widening — can be filled in when we build the Function variants. Nothing structural depends on the complete list.

## Migration

Full rebuild. The deletions are substantial and interlocking — `AllowWithPrompt` removal changes the wire, rebind removal changes the scheduler, typed caps reshape the enforcement type — and a phased migration would leave the enforcement model inconsistent with itself for several commits. Clean removal is simpler to reason about.

Rough shape:

1. **Rip.** Delete `ClientToServer::RebindThread` / `ThreadBindingsPatch` / `Function::RebindThread` / `apply_rebind`. Delete `Disposition::AllowWithPrompt`. Delete `ThreadPendingApproval` / `ThreadApprovalResolved` wire variants and their handlers. Delete `tool_allowlist`. Delete `PermissionScope` entirely. The tree will not compile; that's expected.

2. **Rebuild.** Introduce the new `Scope` type with typed caps. Pipe it through thread creation, `dispatch_thread`, behavior firing, and the tool registration path. Add `PodModifyCap::admits(path)` and route every pod-editing tool through it.

3. **Escalation.** Add `request_escalation` as one builtin tool with a typed-union arg schema. Map each variant to a typed `Function::RequestEscalation*`. Wire the approval-grant flow back into the thread's scope.

4. **UI + persistence.** Update the webui scope-state panel to render the new Scope shape. Rewrite the `pod.toml` schema (delete the old `approval_policy` ghost, add `[allow.caps]` and `[thread_defaults.caps]`). Existing pod directories are wiped on upgrade — matches how we handled the prior `ApprovalPolicy` → `AllowMap` schema change; the project is early enough that a data-migration pass isn't warranted.

5. **Audit + tests.** Per-variant audit records, narrowing-composition property tests, end-to-end tests for the escalation flow and the dispatch-narrowing invariant.

Tree may be red for a while during (1) and (2). That's acceptable; tests guard the landing.
