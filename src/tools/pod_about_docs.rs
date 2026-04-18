//! Static documentation returned by the `pod_about` builtin tool.
//!
//! One entry per topic. `topic(name)` returns the markdown body; `index`
//! is a synthetic topic that lists every other name with a one-line
//! summary. These strings are the agent's primary reference for the
//! pod/behavior schemas — keep them in sync with the types in
//! `whisper-agent-protocol::{pod, behavior}` by hand.

pub fn topic(name: &str) -> Option<&'static str> {
    Some(match name {
        "" | "index" | "topics" => INDEX,
        "overview" => OVERVIEW,
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
    "pod.toml",
    "behavior.toml",
    "triggers",
    "cron",
    "retention",
    "self-modification",
];

const INDEX: &str = "# pod_about topics

- `overview` — what a pod is, how behaviors and threads relate
- `pod.toml` — field reference for the pod's root config
- `behavior.toml` — field reference for a behavior's config
- `triggers` — how a behavior fires: manual, cron, webhook
- `cron` — cron schedule syntax, timezones, overlap, catch-up
- `retention` — on_completion policies for spawned threads
- `self-modification` — editing this pod's config via pod_*_file tools

Call `pod_about` with `{\"topic\": \"<name>\"}` for the details.
";

const OVERVIEW: &str = "# Overview

A **pod** is a self-contained unit that holds a config, a system
prompt, and a set of behaviors. It lives on disk as a directory:

```
<pod_dir>/
  pod.toml              # pod config (backends, sandboxes, defaults)
  system_prompt.md      # (name may differ; see pod.toml's
                        #  thread_defaults.system_prompt_file)
  pod_state.json        # operational state (pause flag); read-only
  behaviors/<id>/
    behavior.toml       # the behavior's trigger + overrides
    prompt.md           # the user-message template threads fire with
    state.json          # run_count, last_fired_at, queued_payload;
                        # read-only via pod_*_file
  threads/              # per-thread JSON files; read-only via pod_*_file
```

A **behavior** is an entry under `behaviors/`. Its trigger (manual,
cron, webhook) decides when to spawn a **thread**; each thread runs a
fresh step-loop against the pod's backend, using `prompt.md` as the
initial user message (with `{{payload}}` substituted for the trigger
payload if present).

Spawned threads live in `threads/`. They are independent: one
behavior firing per schedule produces one thread per fire. The
`on_completion` retention policy decides whether their JSONs are kept,
archived, or deleted after the thread terminates.

You (the agent) are running inside one of these threads. The pod is
YOUR pod — the one your `pod_*_file` tools edit. You can read and
modify your pod's config to change how your own future selves behave.
";

const POD_TOML: &str = "# pod.toml schema

```toml
name = \"whisper-agent dev\"            # required; display name
description = \"...\"                  # optional
created_at = \"2026-04-16T10:00:00Z\"  # required; RFC-3339 UTC

[allow]
backends  = [\"anthropic\", \"openai-compat\"]   # model-backend ids
                                                  # threads may bind to
mcp_hosts = [\"fetch\", \"search\"]               # shared MCP host names
host_env  = [ { name = \"landlock-rw\",           # named sandbox entries;
                provider = \"landlock-laptop\",    # names must be unique.
                spec = { ... } } ]                # See `HostEnvSpec`.

[thread_defaults]
backend            = \"anthropic\"              # must be in allow.backends
model              = \"claude-opus-4-7\"
system_prompt_file = \"system_prompt.md\"       # path relative to pod dir
max_tokens         = 32000
max_turns          = 100
approval_policy    = \"prompt_destructive\"     # see below
host_env           = \"landlock-rw\"            # must reference allow.host_env
mcp_hosts          = [\"fetch\", \"search\"]     # subset of allow.mcp_hosts

[limits]
max_concurrent_threads = 10                  # default 10 if omitted
```

## Validation rules

- `thread_defaults.backend` must appear in `allow.backends`.
- `thread_defaults.host_env` must reference an entry in
  `[[allow.host_env]]`, OR both may be empty (for shared-MCP-only pods).
- Every entry in `thread_defaults.mcp_hosts` must appear in
  `allow.mcp_hosts`.
- Sandbox names in `[[allow.host_env]]` must be unique.

## `approval_policy` values

- `auto_approve` — every tool call runs without prompting.
- `prompt_destructive` — ask before tools marked `destructive`.
- `prompt_pod_modify` — ask only before pod_write_file / pod_edit_file.
- `prompt_all` — ask before every tool call.

## `HostEnvSpec`

See `pod_about` topic `self-modification` for how to inspect the
current pod's `[[allow.host_env]]` entries. The variant discriminator
is `type`:

- `landlock` — Linux landlock sandbox with path-level rw/ro access
  and `network` = `isolated | unrestricted`.
- Other variants depend on host support; inspect `pod.toml` for the
  live shape.

Invalid `pod.toml` writes return the TOML parser / validator error as
a tool error; the on-disk file is untouched.
";

const BEHAVIOR_TOML: &str = "# behavior.toml schema

```toml
name        = \"Daily cleanup\"      # required; display name
description = \"...\"               # optional

[trigger]
kind = \"cron\"  # \"manual\" | \"cron\" | \"webhook\"; see `triggers` topic

[thread]                              # optional; overrides pod defaults
model           = \"claude-opus-4-7\"
max_tokens      = 32000
max_turns       = 50
approval_policy = \"auto_approve\"   # see pod.toml topic for values

[thread.bindings]                     # optional; narrow the thread's
                                       # capability surface below the pod
backend   = \"anthropic\"             # each Some() replaces pod default
host_env  = \"landlock-ro\"           # each None inherits
mcp_hosts = [\"fetch\"]

[on_completion]                        # default: retention = \"keep\"
retention = \"archive_after_days\"     # | \"keep\" | \"delete_after_days\"
days      = 30                         # required for the _after_days variants
```

All fields except `name` are optional — a minimal valid behavior.toml
is `name = \"x\"` (producing a manual-trigger behavior).

Writing `behaviors/<new-id>/behavior.toml` creates a new behavior.
The directory is created on disk and `state.json` is seeded with a
default `BehaviorState`. Prompt.md for a new id must be written AFTER
its behavior.toml.

## Cross-pod validation (at trigger time)

`thread.bindings.backend` / `host_env` / `mcp_hosts` must resolve
against the pod's `[allow]` table — a behavior cannot escape its
pod's capability cap. Cron `schedule` + `timezone` are parsed at
write time (a typo errors the tool call); bindings are checked at
fire time (so a pod-config change that invalidates a binding surfaces
as a runtime error).
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

You have eight builtin tools that target either files or runtime
state in THIS pod. File targets live outside your workspace;
`read_file` / `write_file` / `list_dir` / `bash` cannot see them —
only these can.

## File tools (declarative config)

- `pod_list_files` — walk the pod dir, including per-behavior
  subdirs. `[rw]` entries are reachable via the tools below;
  `[--]` entries (state.json, threads/) are read-only.
- `pod_read_file` — whole file or a line range.
- `pod_write_file` — full overwrite or create.
- `pod_edit_file` — literal-substring replace (single match by
  default, `replace_all` to change every occurrence).
- `pod_grep` — literal-substring search across the whole pod tree
  (including threads/, behaviors/state.json, etc.). Useful for
  locating which thread logged a tool name or error before pulling
  the full file. Dotfiles and `.archived/` are skipped.
- `pod_about` — this tool.

## Orchestration tools (runtime state)

- `pod_run_behavior` — manually fire a behavior, bypassing
  cron/paused gates. Good for testing a new behavior without
  waiting for its schedule, or for kicking a webhook behavior with
  a custom payload.
- `pod_set_behavior_enabled` — pause or resume an individual
  behavior. Paused behaviors skip cron ticks, 503 webhook POSTs,
  and don't catch up at startup — but `pod_run_behavior` still
  works (explicit actions always run).

## What you can write

| Path                              | Writable? | Validation                          |
|-----------------------------------|-----------|-------------------------------------|
| `pod.toml`                        | yes       | TOML + pod schema before disk       |
| pod-level system prompt file      | yes       | none (plain text)                   |
| `behaviors/<id>/behavior.toml`    | yes       | TOML + behavior schema before disk  |
| `behaviors/<id>/prompt.md`        | yes       | none (plain text)                   |
| `behaviors/<id>/state.json`       | no        | runtime state, not config           |
| `pod_state.json`, `threads/`, etc.| no        | runtime state, not config           |

## Creating a new behavior

1. `pod_write_file(\"behaviors/<new-id>/behavior.toml\", \"<toml>\")`
   — creates the directory and seeds `state.json`.
2. `pod_write_file(\"behaviors/<new-id>/prompt.md\", \"<prompt>\")`
   — must come AFTER the behavior.toml write.

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
