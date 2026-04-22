//! Static documentation returned by the `about` builtin tool.
//!
//! One entry per topic. `topic(name)` returns the markdown body; `index`
//! is a synthetic topic that lists every other name with a one-line
//! summary. These strings are the agent's primary reference for how the
//! whisper-agent system works — thread/pod structure, the tool surface,
//! filesystem and sandbox rules, the memory system, and pod/behavior
//! authoring. Keep schemas in sync with the types in
//! `whisper-agent-protocol::{pod, behavior, permission, sandbox,
//! tool_surface}` by hand.

pub fn topic(name: &str) -> Option<&'static str> {
    Some(match name {
        "" | "index" | "topics" => INDEX,
        "overview" => OVERVIEW,
        "threads" => THREADS,
        "tools" => TOOLS,
        "sudo" => SUDO,
        "filesystem" => FILESYSTEM,
        "sandbox" => SANDBOX,
        "memory" => MEMORY,
        "behaviors" => BEHAVIORS,
        "pod.toml" => POD_TOML,
        "behavior.toml" => BEHAVIOR_TOML,
        "triggers" => TRIGGERS,
        "cron" => CRON,
        "retention" => RETENTION,
        "self-modification" => SELF_MOD,
        _ => return None,
    })
}

/// All valid topic names (minus the `index` aliases). Used for the
/// error message when the agent asks for an unknown topic.
pub const TOPIC_NAMES: &[&str] = &[
    "overview",
    "threads",
    "tools",
    "sudo",
    "filesystem",
    "sandbox",
    "memory",
    "behaviors",
    "pod.toml",
    "behavior.toml",
    "triggers",
    "cron",
    "retention",
    "self-modification",
];

const INDEX: &str = "# about topics

Orientation
- `overview` — what whisper-agent is; how threads, pods, and behaviors fit together
- `threads` — what a thread is, its lifecycle, what survives restart

Tools & permissions
- `tools` — the tool surface: admitted vs askable, categories, `describe_tool`/`find_tool`
- `sudo` — running a tool with explicit user approval (the escalation path)

Environment
- `filesystem` — what the `pod_*_file` tools can read/write; the pod-dir layout
- `sandbox` — per-thread jail that host-env MCP tools run inside

State
- `memory` — the persistent `memory/` directory and how to use it
- `retention` — `on_completion` policies for spawned threads

Authoring / composition
- `behaviors` — configured sub-agents; authoring + dispatch
- `pod.toml` — field reference for the pod's root config
- `behavior.toml` — field reference for a behavior's config
- `self-modification` — editing this pod's config via `pod_*_file` tools

Scheduling
- `triggers` — how a behavior fires: manual, cron, webhook
- `cron` — cron schedule syntax, timezones, overlap, catch-up

Call `about` with `{\"topic\": \"<name>\"}` for the details.
";

const OVERVIEW: &str = "# Overview

You are running inside a **thread** — a single ongoing conversation
between a model (you) and whatever started this thread (a user, a
trigger, or a parent thread that dispatched you). Threads persist to
disk: every turn is recorded, and if the server restarts mid-work
the thread resumes where it left off.

Each thread lives inside a **pod** — a directory that holds the
thread's configuration, system prompt, memory, and any **behaviors**
(pre-authored sub-agents that can fire on a schedule, on a webhook,
or be launched manually). The pod is the unit of identity for the
agent: \"your pod\" is the one these docs keep referring to, the one
your `pod_*_file` tools edit.

```
<pod_dir>/
  pod.toml              # config: backends, sandboxes, caps, defaults
  system_prompt.md      # your system prompt (filename from pod.toml)
  memory/               # persistent notes — see `memory` topic
    MEMORY.md           #   index
    <topic>.md          #   individual memory files
  behaviors/<id>/       # one per behavior
    behavior.toml       #   trigger + thread config + scope
    prompt.md           #   seed user-message for spawned threads
    state.json          #   runtime state (read-only to you)
  threads/<id>.json     # full message history of each thread
  pod_state.json        # pod-level runtime state (read-only)
```

Tools come from three sources: **builtins** (the `pod_*` tools,
`about`, `describe_tool`, `find_tool`, `dispatch_thread`, `sudo`);
**host-env MCP** tools from the sandbox this thread bound to (e.g.
a `rustdev_bash` tool if the sandbox is a Rust dev environment); and
**shared MCP** tools from third-party services the pod is allowed
to reach. See the `tools` topic.

Not every call you make runs unconditionally. A thread has a
**scope** — the set of tools and capabilities it currently holds.
Scope starts from the pod's `[allow]` ceiling, can be narrowed by the
parent (for dispatch-spawned threads) or by a behavior's declared
scope (for trigger-spawned threads), and can be widened call-by-call
at runtime via `sudo` when an interactive approver is attached. See
the `sudo` topic.

The full topic index lives at `about` with no arguments.
";

const THREADS: &str = "# Threads

A thread is the conversation you are currently in. It is the atomic
unit of execution in whisper-agent: one model, one message history,
one step-loop turning tool calls into next-turn inputs until the
model stops calling tools or hits `max_turns`.

## Initial context (the setup prefix)

Every thread opens with a fixed prefix the scheduler assembles once
at seed time. In order:

1. **System prompt** (`Role::System`) — from the pod's
   `system_prompt.md` (or a per-thread override).
2. **Tool manifest** (`Role::Tools`) — every admitted tool with
   schema + description. This message lives at position 1 and is
   **never rewritten** mid-conversation. Tools activated after seed
   are announced via a tail `Role::System` message; see the `tools`
   topic.
3. **Memory block** (`Role::System`) — today's date plus the
   `memory/MEMORY.md` index. See the `memory` topic.
4. (Optionally) a tool-catalog listing, if the pod's `tool_surface`
   config requested one.

After the prefix, the conversation alternates between `Role::User`
turns (your inputs) and `Role::Assistant` turns (your outputs, which
may include tool calls and their results).

## Lifecycle

Threads are persisted as `<pod>/threads/<id>.json` — the full message
history plus runtime state. A thread survives server restart: on
startup the scheduler loads every non-terminated thread and resumes
it.

A thread terminates when:
- The model ends its turn without calling any tools and the thread
  has no pending user input (idle);
- `max_turns` is exceeded (hard cap from `pod.toml`);
- The backend returns a non-retriable error;
- The caller cancels.

## Dispatch-spawned threads

Threads can spawn child threads via the `dispatch_thread` builtin
(see `behaviors` topic for details). Sync mode blocks the parent's
turn and returns the child's final assistant message as the tool
result; async mode returns immediately with the child's id and
delivers the final message later as a tagged user turn on the
parent. Child threads are first-class — same persistence, same
lifecycle — and their retention is governed by the spawning
behavior's `on_completion` if any.

## What survives what

| Event                        | Conversation? | Scope grants? | Memory? |
|------------------------------|:-------------:|:-------------:|:-------:|
| Server restart mid-thread    |       ✓       |       ✓       |    ✓    |
| Thread terminates            |       ✓       |       ✗       |    ✓    |
| New thread in same pod       |       ✗       |       ✗       |    ✓    |

Escalation grants are per-thread and in-memory only — a new thread
starts from the pod baseline every time. Persistent preferences go
in memory; persistent config widening goes in `pod.toml`.
";

const TOOLS: &str = "# Tools

## Two buckets: admitted vs askable

Every tool visible to you is either **admitted** (you can call it
now) or **askable** (it exists, but calling it requires wrapping
the invocation in `sudo` for user approval — see the `sudo` topic).

Admitted tools are on the wire from turn 0: their name, input
schema, and description land in the `Role::Tools` manifest at thread
seed. Askable tools live in the listing or in `find_tool` output;
their schemas are fetched on demand.

## Description verbosity

The manifest is sized to fit the pod's budget. A small set of
\"core\" tools (by default: `describe_tool`, `find_tool`, `sudo`)
carry their full prose description. Every other admitted tool
carries a first-line summary only. The input schema is always full —
the grammar needs it — but for details on what a tool does, fetch
it with `describe_tool`.

## Categories

Tools come from three sources; a tool's name prefix tells you which:

- **Builtins** — in-process, no network, always available when
  admitted. Includes:
  - `pod_read_file` / `pod_write_file` / `pod_edit_file` /
    `pod_remove_file` / `pod_list_files` / `pod_grep` /
    `pod_list_threads` — edit and search the pod directory. See
    `filesystem`.
  - `pod_run_behavior` / `pod_set_behavior_enabled` — fire or pause
    behaviors manually. See `behaviors`.
  - `dispatch_thread` — spawn a child thread from a named behavior
    or an ad-hoc config. See `behaviors`.
  - `sudo` — run a tool with explicit user approval. See `sudo`.
  - `about` / `describe_tool` / `find_tool` — documentation surfaces.

- **Host-env MCP** — tools from the sandbox this thread is bound to
  (e.g. `rustdev_bash`, `rustdev_read_file`). Names are prefixed with
  the host-env's name from `[[allow.host_env]]`. They run inside the
  jail described in `sandbox`; when they fail with a permission error,
  it's usually the sandbox talking.

- **Shared MCP** — tools from third-party MCP hosts the pod is
  allowed to reach (e.g. `fetch_url`, `search_web`). Names are
  prefixed with the host id from the pod's `mcp_hosts` list.
  They run in their own process; latency and failure modes track the
  external service.

## Discovering tools

- `describe_tool(name)` — full schema + description for any admitted
  OR askable tool. Use it before first call when only the summary is
  on the wire.
- `find_tool(pattern)` — regex search over tool names/descriptions.
  Returns everything visible to you at the current scope, with an
  `admitted` vs `askable` tag per hit.
- `about` (this tool) — system-level documentation, not per-tool.

If you call a name the scope denies, the tool call errors with the
denial reason. If the name isn't in the pod at all, you get an
unknown-tool error — try `find_tool` to see what actually exists.
";

const SUDO: &str = "# Sudo

Sudo runs a tool with explicit user approval. It's the single
runtime escalation path: when you discover you need a tool the
thread's scope denies, OR when a call you're allowed to make would
exceed your current caps (e.g. writing a content path with
`pod_modify: memories`), wrap it in `sudo`.

## Shape

```json
sudo({
  \"tool_name\": \"pod_write_file\",
  \"args\": { \"filename\": \"system_prompt.md\", \"content\": \"...\" },
  \"reason\": \"tightening the system prompt; inner edit preview below\"
})
```

- `tool_name` — the wrapped tool. Must be routable in this pod
  (check with `find_tool` if unsure). Sudo cannot wrap itself or
  `dispatch_thread`; call those directly.
- `args` — the arguments for the wrapped tool. Shape must match
  that tool's input schema. If you haven't called it before, run
  `describe_tool(name: \"<tool_name>\")` first to fetch the schema.
- `reason` — short justification shown to the user in the approval
  UI. Tie it to the specific call, not a generic explanation.
  \"Rewriting the prompt to add the new memory-dir instruction\"
  beats \"need write access.\"

## Approval: three outcomes

The user sees the tool name, the pretty-printed args, and your
reason, then picks one:

- **Approve once** — run this single call; scope stays as-is.
  Future direct calls of the same tool face the same gate.
- **Approve remember** — run the call AND flip `scope.tools` to
  admit the tool name for the rest of this thread. Future direct
  calls skip the prompt. Caps are NOT remembered — each cap-gated
  operation needs a fresh sudo.
- **Reject** — the sudo call errors with the user's reason (if any).
  Course-correct from there; don't just retry the same sudo.

The tool call parks until the user responds — there is no timeout
and no polling.

## Cap bypass

When approved (either flavor), the wrapped tool runs at the pod's
full ceiling for this one invocation: `pod.allow.caps.pod_modify`,
`pod.allow.caps.dispatch`, `pod.allow.caps.behaviors`. That's
deliberate — the user is explicitly OK-ing the call, and you
shouldn't also have to explain why the cap-widening is justified
separately from the call itself.

Caveat: sudo cannot cross the pod's ceiling. Tools not in
`pod.allow.tools` are unreachable; cap targets above
`pod.allow.caps.*` are unreachable. Persistent widening past those
requires editing `pod.toml` (and a scope with `pod_modify:
modify_allow`).

## Headless threads

Threads without an interactive approver (cron-triggered behaviors,
scheduled runs) cannot use sudo — it hard-errors at the tool layer.
The `sudo` tool is filtered from the catalog for autonomous
threads. Design property: autonomous threads run at their
author-declared scope and nothing wider.

## When to use it vs. alternatives

- Need a tool once, for this conversation only → `sudo`.
- Need a tool every time this agent runs → edit `pod.toml` (if the
  thread has `pod_modify: modify_allow`) or ask the user to.
- Need a tool every time a specific behavior fires → edit that
  behavior's `[scope]` block (see `behavior.toml`).
";

const FILESYSTEM: &str = "# Filesystem

Two filesystems are relevant to a thread: the **pod directory**
(edited by the `pod_*_file` builtins) and whatever the host-env
sandbox exposes (edited by sandbox-specific MCP tools, e.g.
`rustdev_read_file`, `rustdev_bash`). This topic covers the pod
directory; see `sandbox` for the other.

## Pod-directory tools

All operate on paths relative to the pod root and never escape it.

- `pod_list_files(path?)` — walk the pod tree. Each entry tagged
  `[rw]` (readable + writable), `[r-]` (readable, writable to
  scheduler only), or `[--]` (hidden from your tools). Use at thread
  start to orient.
- `pod_read_file(path, offset?, limit?, tail?)` — whole file or
  line range. Default cap is 500 lines when no slicing args are
  given. `tail: N` returns the last N lines directly (handy for the
  end of a long thread log). Any sliced response closes with a
  `[showing lines X-Y of N]` marker so you know the file's total
  length without a follow-up call. `tail` and `offset` are mutually
  exclusive.
- `pod_write_file(path, content)` — create or full overwrite. Prefer
  `pod_edit_file` for targeted changes to existing files.
- `pod_edit_file(path, old_string, new_string, replace_all?)` —
  literal-substring replace. Single match by default; pass
  `replace_all: true` to change every occurrence. When you extend
  `old_string` with surrounding context to disambiguate, extend
  `new_string` by the same text — extending only `old_string`
  deletes the intervening characters.
- `pod_remove_file(path)` — delete a `memory/<name>.md` or
  retire an entire behavior by removing its
  `behaviors/<id>/behavior.toml` (this also drops the directory's
  `prompt.md` + `state.json` and unregisters the behavior so it
  stops firing). `prompt.md` cannot be removed on its own; the
  top-level config files (`pod.toml`, the system-prompt file) are
  not removable through this tool.
- `pod_grep(pattern, path?)` — literal-substring search across the
  pod tree. Useful for locating which thread logged an error or
  which behavior references a particular tool.
- `pod_list_threads(filters)` — structured query over
  `threads/*.json`: filter by behavior, state, since-timestamp, turn
  count. Returns a one-line summary per hit, newest-first. Use
  before `pod_read_file` when you want \"the most recent run of
  behavior X.\"

## Access rules

| Path                              | Access | Notes                                        |
|-----------------------------------|--------|----------------------------------------------|
| `pod.toml`                        | rw*    | TOML + pod schema validated before disk      |
| pod-level system prompt file      | rw     | plain text                                   |
| `memory/MEMORY.md`, `memory/*.md` | rw     | the memory system; see `memory` topic        |
| `behaviors/<id>/behavior.toml`    | rw*    | TOML + behavior schema validated             |
| `behaviors/<id>/prompt.md`        | rw     | plain text                                   |
| `pod_state.json`                  | r-     | scheduler-owned runtime state                |
| `behaviors/<id>/state.json`       | r-     | scheduler-owned runtime state                |
| `threads/<id>.json`               | r-     | full message history of a spawned thread     |
| `.archived/`, other               | --     | not reachable through `pod_*_file`           |

\\* Write access is further gated by the thread's `pod_modify` cap:

- `none` — no writes at all.
- `memories` — only `memory/**`. (Default for threads spawned by
  behaviors without a widened scope.)
- `content` — `memories` plus `system_prompt.md`,
  `behaviors/*/prompt.md`, `behaviors/*/behavior.toml`.
- `modify_allow` — everything above plus `pod.toml`.

Check `pod_list_files` tags to see what you actually have. If a
write you expect to work returns a cap error, wrap the call in
`sudo` — an approved sudo'd call bypasses the cap for that single
invocation.

## Validation

Writes to `pod.toml` and `behavior.toml` parse and validate before
touching disk. A bad TOML or a schema violation returns the error
as a tool error; the on-disk file is unchanged. Fix and retry — no
cleanup needed.
";

const SANDBOX: &str = "# Sandbox

When a pod declares a `[[allow.host_env]]` entry, it describes an
isolated environment that host-env MCP tools run inside. The
thread picks one of those entries at seed time (from
`thread_defaults.host_env` or a per-thread override) and every
tool name prefixed with that entry's `name` (e.g. `rustdev_bash`)
executes inside the jail.

The `pod_*_file` builtins do NOT go through this sandbox — they
talk directly to the pod directory through a scheduler-owned
path. The jail only applies to host-env MCP tool calls.

## Variants

Two sandbox types are defined:

- **`landlock`** — Linux-native lightweight isolation. Fast, no
  container runtime needed. Enforces per-path read-only / read-write
  access and an all-or-nothing network policy. Good for local dev
  loops.

  ```toml
  [[allow.host_env]]
  name          = \"rustdev\"
  provider      = \"landlock-laptop\"
  type          = \"landlock\"
  allowed_paths = [\"/home/me/project:rw\", \"/:ro\", \"/tmp:rw\"]
  network       = \"isolated\"       # or \"unrestricted\"
  ```

- **`container`** — OCI container (podman/docker). Heavier but
  fully reproducible via an image; supports CPU/memory limits and
  an env var map.

  ```toml
  [[allow.host_env]]
  name     = \"fetcher\"
  provider = \"podman\"
  type     = \"container\"
  image    = \"ghcr.io/whisper-agent/curl:latest\"
  mounts   = [
    { host = \"/home/me/cache\", guest = \"/cache\", mode = \"read_write\" },
    { host = \"/etc/ssl\",       guest = \"/etc/ssl\", mode = \"read_only\"  },
  ]
  network  = { kind = \"allow_list\", hosts = [\"api.example.com\"] }
  limits   = { cpus = 2.0, memory_mb = 512, timeout_s = 60 }
  env      = { LOG_LEVEL = \"debug\" }
  ```

## Network policy

Three forms, applicable to both variants:

- `\"isolated\"` — no network access. Tool calls that try to reach
  the network fail immediately.
- `\"unrestricted\"` — full outbound access. Fine for dev, risky for
  autonomous pods.
- `{ kind = \"allow_list\", hosts = [\"host.example.com\", ...] }` —
  whitelist specific destinations.

## Typical failure modes

When a host-env tool call fails unexpectedly, the sandbox is the
usual suspect:

- **Path not allowed** — the tool tried to read/write outside
  `allowed_paths`. Check the list in `pod.toml`.
- **Network blocked** — the tool tried to reach something outside
  the declared network policy.
- **Binary not in PATH** — container variant, image doesn't have
  the binary the tool expects.
- **Timeout** — long-running command exceeded `limits.timeout_s`.

To widen the sandbox itself, edit `pod.toml` (requires
`pod_modify: modify_allow` — which you can get one-off via `sudo` on
the pod_write_file call). Sandbox widening typically wants an
explicit human decision.

## No host-env configured

A pod with no `[[allow.host_env]]` entries has no sandbox for its
threads. Only builtin tools and shared-MCP tools are available.
This is a valid configuration for pods that only orchestrate
external services or operate on their own pod directory.
";

const MEMORY: &str = "# Memory

The pod has a persistent `memory/` directory. Files survive thread
termination, server restart, and the conversation window that
compaction eventually prunes. Use memory to carry forward things
you'll genuinely need next time: user facts, corrections, project
context. Do NOT use it as a scratchpad for the current thread —
tasks and conversation handle that.

## Shape

```
memory/
  MEMORY.md              # index: one line per memory file
  user_role.md           # individual memory, any topic-based name
  feedback_testing.md
  project_release_cut.md
  ...
```

Each memory file has YAML-ish frontmatter plus a markdown body:

```markdown
---
name: {memory name}
description: {one-line hook used to decide relevance later}
type: user | feedback | project | reference
---

{body}
```

`MEMORY.md` is the index and is always loaded into every new
thread's initial context (as the memory block — see `threads`).
Individual files are NOT loaded eagerly; you read them with
`pod_read_file` when the index line suggests relevance.

## Types

- **user** — who the user is, their expertise, how they want to
  collaborate. Tailor explanations / choices to this.
- **feedback** — rules for how to work in this project. Include
  the *why* so you can judge edge cases. Save on both corrections
  (\"don't do X\") and confirmations (\"the bundled PR was right\").
- **project** — ongoing initiatives, decisions, deadlines. Decays
  fast; include motivation so you can tell when it's stale.
- **reference** — pointers to external systems (dashboards, ticket
  projects, Slack channels) and what they're for.

## What NOT to save

- Code patterns, conventions, file paths — derivable from the
  codebase.
- Git history or who-changed-what — `git log` is authoritative.
- Debugging recipes — the fix is in the code; the commit message
  has the context.
- Current-thread state, in-progress work, ephemeral todos.

These exclusions apply even when the user says \"save this.\" If
asked to save something ephemeral, ask what was *surprising* or
*non-obvious* about it — that's the durable kernel.

## Using memory

- Load `memory/MEMORY.md` (already in your initial context) when
  orienting.
- Read individual files when a description matches the task.
- Before acting on a memory claim (\"X function exists\"), verify —
  memories can go stale. Trust what you observe now and update the
  memory if it's wrong.

## Writing a memory

Two writes per new memory: the file itself and the index line.

```
pod_write_file(\"memory/feedback_testing.md\", \"<frontmatter + body>\")
pod_edit_file(\"memory/MEMORY.md\", <append one line>)
```

Keep the index line short — under ~150 chars. The index is
truncated past 200 lines when injected, so organize semantically,
not chronologically. Update or remove entries that turn out wrong.
Don't write duplicates — look for an existing memory to edit first.
";

const BEHAVIORS: &str = "# Behaviors

A behavior is a pre-authored sub-agent that lives in the pod. It
has its own system prompt, thread config, trigger, and declared
scope — together they define a repeatable unit of work the pod can
fire on demand, on a schedule, or in response to a webhook.

A thread spawned from a behavior is a full first-class thread —
same setup prefix, same persistence, same tool surface — but its
scope is capped by both the pod's `[allow]` and the behavior's own
`[scope]` block. Behaviors run autonomously: they have no
interactive approver and cannot use `sudo`.

## Anatomy

```
behaviors/<id>/
  behavior.toml   # name, trigger, thread overrides, scope
  prompt.md       # seed user message; {{payload}} substitutes the
                  # webhook body or the explicit payload passed in
  state.json      # runtime: run_count, last_fired_at, queued_payload
                  # (read-only to you)
```

See the `behavior.toml` topic for the full schema.

## Firing a behavior

Three trigger kinds (see `triggers` topic):

- **manual** — fires only on explicit request via
  `pod_run_behavior` or a UI button.
- **cron** — fires on a cron schedule (`cron` topic has syntax).
- **webhook** — fires on HTTP POST to
  `/triggers/<pod_id>/<behavior_id>`; the body becomes
  `{{payload}}`.

From a thread, you can fire any behavior manually with
`pod_run_behavior(behavior_id, payload?)`. The new thread runs
concurrently; its completion delivers nothing back to you (no
parent-child link). Use `pod_list_threads` to find its output
after the fact.

## Dispatching a child thread

Distinct from behaviors, the `dispatch_thread` builtin spawns an
ad-hoc child with an inline prompt:

```json
{
  \"prompt\": \"Summarize the error log at /tmp/out.log\",
  \"sync\": true,
  \"config_override\": { \"model\": \"claude-haiku-4-5\" },
  \"bindings_override\": { \"host_env\": \"readonly\" }
}
```

- `sync: true` — blocks the parent's turn; returns the child's
  final assistant message as the tool result.
- `sync: false` — returns immediately with the child's thread id;
  the child's final message arrives later as a tagged
  `[dispatched thread <id> completed]` user turn.

The child inherits the parent's scope by default, narrowed by any
`config_override` / `bindings_override`. Dispatch requires the
parent's scope to include `dispatch: within_scope`.

Dispatch is the right tool for: offloading a cheap sub-task to a
smaller model; isolating a read-only investigation; parallelizing
independent searches. Use behaviors for: recurring work; work
that needs its own persona; work triggered externally.

## Authoring a behavior

Requires `behaviors: author_narrower` or `author_any` cap.

1. `pod_write_file(\"behaviors/<new-id>/behavior.toml\", <toml>)` —
   creates the directory and seeds `state.json`.
2. `pod_write_file(\"behaviors/<new-id>/prompt.md\", <prompt>)` —
   must come AFTER the behavior.toml write.

`author_narrower` limits you to behaviors whose declared scope is
strictly narrower than your own; `author_any` lets you declare any
scope within the pod ceiling. See `behavior.toml`.
";

const POD_TOML: &str = "# pod.toml schema

```toml
name         = \"whisper-agent dev\"     # required; display name
description  = \"...\"                  # optional
created_at   = \"2026-04-16T10:00:00Z\" # required; RFC-3339 UTC

[allow]
backends  = [\"anthropic\", \"openai-compat\"]  # model-backend ids threads may bind to
mcp_hosts = [\"fetch\", \"search\"]              # shared MCP host names

# Sandbox entries; names must be unique across [[allow.host_env]]. See `sandbox` topic.
[[allow.host_env]]
name          = \"rustdev\"
provider      = \"landlock-laptop\"
type          = \"landlock\"
allowed_paths = [\"/home/me/project:rw\", \"/:ro\"]
network       = \"isolated\"

# Tool gate. `default` decides unlisted tools; `overrides` flips specific names.
# Omitted → `{ default = \"allow\", overrides = {} }` (every admitted tool allowed).
[allow.tools]
default   = \"allow\"                 # \"allow\" | \"deny\"
overrides = { pod_write_file = \"deny\" }

# Capability ceilings. No thread in this pod — no sudo grant — can exceed these.
[allow.caps]
pod_modify = \"modify_allow\"         # none < memories < content < modify_allow
dispatch   = \"within_scope\"         # none | within_scope
behaviors  = \"author_any\"           # none < read < author_narrower < author_any

[thread_defaults]
backend            = \"anthropic\"               # must appear in allow.backends
model              = \"claude-opus-4-7\"
system_prompt_file = \"system_prompt.md\"        # pod-relative path
max_tokens         = 32000
max_turns          = 100
host_env           = [\"rustdev\"]               # names from [[allow.host_env]]
mcp_hosts          = [\"fetch\", \"search\"]      # subset of allow.mcp_hosts

# Starting cap values for a freshly-created thread. Each must be
# ≤ the matching allow.caps ceiling. Omitted → conservative defaults:
# memories / within_scope / read.
[thread_defaults.caps]
pod_modify = \"memories\"
dispatch   = \"within_scope\"
behaviors  = \"read\"

# Tool-surface presentation knobs. See `tools` topic for semantics.
[thread_defaults.tool_surface]
core_tools       = [\"describe_tool\", \"find_tool\", \"sudo\"]
initial_listing  = \"none\"           # \"none\" | \"all_names\" | \"core_only\"
activation_surface = \"announce\"     # \"announce\" | \"inject_schema\"

# Compaction — automatic summarization when the thread grows long.
[thread_defaults.compaction]
enabled         = true
prompt_file     = \"\"                 # empty → built-in default
token_threshold = 120000              # auto-trigger over this; omit for manual-only

[limits]
max_concurrent_threads = 10           # default 10 if omitted
```

## Validation rules

- `thread_defaults.backend` must appear in `allow.backends`.
- Every entry in `thread_defaults.host_env` must name an entry in
  `[[allow.host_env]]`. Empty allowed iff `[[allow.host_env]]` is
  also empty (shared-MCP-only pod).
- Every `thread_defaults.mcp_hosts` entry must appear in
  `allow.mcp_hosts`.
- `thread_defaults.caps.*` must each be ≤ the matching
  `allow.caps.*`.
- `[[allow.host_env]]` names must be unique.

## Invalid writes

Writes to `pod.toml` parse and validate BEFORE hitting disk. A bad
TOML or a schema violation returns the parser/validator error as a
tool error; the on-disk file stays untouched.
";

const BEHAVIOR_TOML: &str = "# behavior.toml schema

```toml
name        = \"Daily cleanup\"       # required; display name
description = \"...\"                # optional

[trigger]
kind = \"cron\"                       # \"manual\" | \"cron\" | \"webhook\"; see `triggers`

# Per-behavior thread config. Every field optional; omitted → inherit the pod's
# thread_defaults value.
[thread]
model               = \"claude-haiku-4-5\"
max_tokens          = 16000
max_turns           = 30

# Per-behavior system prompt. One of:
#   { kind = \"file\", name = \"<pod-relative path>\" }
#   { kind = \"text\", text = \"<literal>\" }
# Omitted → inherit the pod's system_prompt.
system_prompt = { kind = \"file\", name = \"summarizer.md\" }

# Per-behavior runtime scope. Composed with the pod's [allow] ceiling at fire
# time: child_scope = pod.allow.narrow(this). Every field is an Option — a
# wholly-absent [scope] block means the behavior runs at the pod's full ceiling
# (minus escalation, which behaviors never have).
[scope]
backends  = [\"anthropic\"]           # Some(list) narrows; omit to inherit
host_envs = [\"readonly\"]
mcp_hosts = [\"fetch\"]

[scope.tools]
default   = \"allow\"
overrides = { pod_write_file = \"deny\" }

[scope.caps]
pod_modify = \"memories\"             # narrows if present, inherits if absent
dispatch   = \"none\"
behaviors  = \"read\"

# Tool-surface override; replaces the pod's thread_defaults.tool_surface
# wholesale when present. See `tools` topic.
[scope.tool_surface]
core_tools      = \"all\"
initial_listing = \"core_only\"

[on_completion]                       # default: retention = \"keep\"
retention = \"archive_after_days\"    # | \"keep\" | \"delete_after_days\"
days      = 30                        # required for the _after_days variants
```

All fields except `name` are optional — a minimal valid
behavior.toml is `name = \"x\"` (producing a manual-trigger behavior
that inherits everything from the pod).

## Cross-pod validation

- `thread.system_prompt` file path must resolve under the pod
  directory.
- `[scope]` lists (backends, host_envs, mcp_hosts) are checked at
  fire time against `pod.allow` — a behavior cannot escape its
  pod's capability cap. A pod-config change that invalidates a
  binding surfaces as a runtime error when the behavior next fires.
- Cron `schedule` + `timezone` are parsed at write time (a typo
  errors the tool call).

## Authoring sequence

Writing `behaviors/<new-id>/behavior.toml` creates the directory
and seeds `state.json`. Write `prompt.md` AFTER the behavior.toml.
Requires `behaviors: author_narrower` or `author_any` cap; see
`behaviors` topic.
";

const TRIGGERS: &str = "# Trigger variants

The `[trigger]` block picks ONE variant by its `kind` field.

## `kind = \"manual\"` (default)

```toml
[trigger]
kind = \"manual\"
```

Fires only on an explicit `RunBehavior` from the UI or the
`pod_run_behavior` tool. Does nothing on its own. Good default for
a behavior you're iterating on.

## `kind = \"cron\"`

```toml
[trigger]
kind     = \"cron\"
schedule = \"0 9 * * *\"             # 5-field crontab (min hour dom mon dow)
timezone = \"America/Los_Angeles\"   # IANA name; default \"UTC\"
overlap  = \"skip\"                  # \"skip\" | \"queue_one\" | \"allow\"
catch_up = \"one\"                   # \"none\" | \"one\" | \"all\"
```

Fires on the cron schedule. See the `cron` topic for syntax and
semantics.

## `kind = \"webhook\"`

```toml
[trigger]
kind    = \"webhook\"
overlap = \"skip\"                   # \"skip\" | \"queue_one\" | \"allow\"
```

Fires on HTTP POST to `/triggers/<pod_id>/<behavior_id>`. The request
body becomes the trigger payload — exposed to the thread as
`{{payload}}` in prompt.md (pretty-printed JSON) and carried on the
thread's `BehaviorOrigin.trigger_payload`.

No auth, no path customization in v1 — gate the endpoint externally
if you care who can POST to it.
";

const CRON: &str = "# Cron schedules

## Schedule format

Five-field UNIX crontab: `minute hour day-of-month month day-of-week`.

- `*` — every value for this field
- `*/5` — every 5th (minute, hour, etc.)
- `0,15,30,45` — comma list
- `9-17` — inclusive range
- Day-of-week: 0 or 7 = Sunday, 1 = Monday, ..., 6 = Saturday

Examples:

- `0 9 * * *` — daily at 09:00
- `*/15 * * * *` — every 15 minutes
- `0 */4 * * 1-5` — every 4 hours, Mon-Fri
- `30 2 1 * *` — 02:30 on the 1st of every month

The underlying library expects seconds too; we prepend `0 ` internally
so fires land on the zeroth second of each minute.

## Timezone

Any IANA name (`UTC`, `America/Los_Angeles`, `Europe/Berlin`, ...).
Defaults to `UTC` if omitted. A DST-shifting zone means `0 2 * * *`
may fire 0, 1, or 2 times on the spring-forward / fall-back day —
that's intrinsic to cron, not a bug.

## `overlap` — what happens when a fire arrives during a previous run

- `skip` (default) — drop the fire.
- `queue_one` — park the fire's payload in state.json; the next
  on-completion hook spawns a fresh run with it. At most one queued
  payload; a second queued fire overwrites the first.
- `allow` — spawn a concurrent thread.

## `catch_up` — what happens to fires missed while the server was down

Applied ONCE per behavior on scheduler startup.

- `none` — skip missed fires silently.
- `one` (default) — fire at most once to catch up, log the count.
- `all` — fire every missed run. Rarely what's wanted — long downtime
  plus a per-minute cron would hammer the scheduler with stale fires.

Catch-up is gated by the same enabled/paused checks as a normal tick:
paused behaviors don't catch up.
";

const RETENTION: &str = "# `on_completion` retention policies

```toml
[on_completion]
retention = \"keep\"                    # default
```

Never sweep. Spawned thread JSONs live in `threads/` until manually
removed. Fine for manual-only behaviors or low-cadence crons.

```toml
[on_completion]
retention = \"archive_after_days\"
days      = 30
```

After `days` past the thread's last_active timestamp, move the JSON
to `<pod>/.archived/threads/`. Preserves the history but keeps
`threads/` small. `days` must be > 0.

```toml
[on_completion]
retention = \"delete_after_days\"
days      = 7
```

After `days` past last_active, delete the JSON outright. Best for
high-cadence crons whose run history isn't otherwise interesting.
`days` must be > 0.

A high-cadence behavior without an `after_days` policy accumulates
thousands of JSONs over a few months — the sweeping path is the
default for anything firing more often than daily.
";

const SELF_MOD: &str = "# Self-modification

Your pod is editable from inside the thread. The `pod_*_file` and
`pod_*_behavior` tools below target either files or runtime state
in THIS pod — not your general-purpose filesystem, not an
arbitrary workspace. See the `filesystem` topic for the read/write
table and the `behaviors` topic for authoring sub-agents.

## File tools (declarative config)

- `pod_list_files` — walk the pod dir, tagging each entry `[rw]` /
  `[r-]` / `[--]` by access level at your current cap.
- `pod_read_file` — whole file or a line range (default cap: 500
  lines when no slicing args supplied). `tail: N` reads the last N
  lines directly; every sliced response reports total line count.
- `pod_write_file` — full overwrite or create.
- `pod_edit_file` — literal-substring replace; `replace_all` to
  change every occurrence.
- `pod_remove_file` — delete a `memory/<name>.md` or retire a
  behavior by removing its `behaviors/<id>/behavior.toml`.
- `pod_grep` — literal-substring search across the whole pod tree.
- `pod_list_threads` — structured query over `threads/` with
  behavior/state/since/turn-count filters.
- `about` — this tool.

## Orchestration tools (runtime state)

- `pod_run_behavior` — manually fire a behavior, bypassing
  cron/paused gates. Good for testing a new behavior without
  waiting for its schedule, or for kicking a webhook behavior with
  a custom payload.
- `pod_set_behavior_enabled` — pause or resume an individual
  behavior. Paused behaviors skip cron ticks, 503 webhook POSTs,
  and don't catch up at startup — but `pod_run_behavior` still
  works (explicit actions always run).
- `dispatch_thread` — spawn an ad-hoc child thread; see `behaviors`.
- `sudo` — run any admissible tool with explicit user approval; see
  `sudo`.

## Creating a new behavior

1. `pod_write_file(\"behaviors/<new-id>/behavior.toml\", \"<toml>\")`
   — creates the directory and seeds `state.json`. Requires the
   `behaviors` cap (`author_narrower` or `author_any`).
2. `pod_write_file(\"behaviors/<new-id>/prompt.md\", \"<prompt>\")`
   — must come AFTER the behavior.toml write.

See the `behavior.toml` topic for the schema.

## Editing vs rewriting

Prefer `pod_edit_file` for targeted changes so you don't have to
re-send the whole file. When extending `old_string` with context to
disambiguate a multi-match, extend `new_string` by the same text —
extending only `old_string` deletes the intervening lines. Use
`pod_write_file` for creates and full rewrites.

## Validation errors

Writes to `pod.toml` and `behavior.toml` parse and validate BEFORE
hitting disk. A bad TOML or a schema violation returns the error
as a tool error; the on-disk file stays untouched. Fix the error
and try again — no cleanup needed.
";
