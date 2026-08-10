//! Scripted (Lua) driver execution.
//!
//! A scripted driver is a Lua program at `<pod>/drivers/<name>.lua`
//! defining one global function:
//!
//! ```lua
//! function on_event(state, event, config)
//!   -- state: the driver's persisted table (empty {} on first event)
//!   -- event: { kind = "turn_start" | "agent_completed" | ..., ... }
//!   -- config: knob values frozen at creation (step 10) — the map the
//!   --         server validated against describe(); {} when knob-less.
//!   --         A program edited after creation may find knobs missing;
//!   --         read them like any table (absent = nil).
//!   return { effects = { { kind = "run_agent", thread_id = ... } },
//!            state = state }
//! end
//! ```
//!
//! Two further entry points are optional: `present(state, config) ->
//! blocks` (step 7b — the pure presentation function) and `describe() ->
//! { label?, description?, knobs? }` (step 10 — the configuration
//! declaration the new-thread form renders; evaluated at listing and
//! creation time, never during coordination).
//!
//! The handler is a pure policy step in the same sense as the builtin
//! driver's functions: durable thread events in, requested effects out,
//! with all mutable driver state carried explicitly through the
//! `state` table (persisted as JSON in `DriverState::Scripted`). A
//! parked coroutine could not be snapshotted or replayed after a
//! restart, so there deliberately is no coroutine-style API.
//!
//! Boundary events may be re-delivered (e.g. a parked thread's boundary
//! fires again when the thread is next stepped); handlers must be
//! idempotent — return no effects when the event asks about work the
//! state already tracks as in progress.
//!
//! Each event runs in a fresh, sandboxed VM: table/string/math/utf8
//! stdlib plus one injected helper — `regex_capture(pattern, text)`,
//! a deterministic Rust-regex extraction (step 11 slice 5) — with the
//! base-library escape hatches stripped on top
//! (pcall/xpcall — they could swallow the instruction-budget error;
//! load/dofile/loadfile — filesystem and stdin reach; collectgarbage,
//! print, math.random — nondeterminism), plus a memory ceiling and an
//! instruction budget. Driver programs are pure policy — every side
//! effect goes through the returned effect list, admitted and
//! journaled by the scheduler, and driver state must be a
//! deterministic function of program + events.

use mlua::{Lua, LuaSerdeExt};
use serde::{Deserialize, Serialize};

/// Per-event instruction budget. Policy scripts decide and return; they
/// do not compute. The budget is deliberately generous — a checker
/// composing a prompt from a few tool calls uses a tiny fraction.
const INSTRUCTION_BUDGET: u64 = 2_000_000;
/// Count granularity for the instruction hook.
const HOOK_EVERY: u32 = 10_000;
/// Per-VM memory ceiling.
const MEMORY_LIMIT_BYTES: usize = 16 * 1024 * 1024;

/// A durable thread event delivered to `on_event`, tagged with the
/// thread it came from — a scripted weave may tick several threads.
// (`Eq` dropped when `QueryCompleted` brought f32 rerank scores.)
#[derive(Serialize, Debug, Clone, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ScriptedEvent {
    /// External input was accepted into a coordinated thread. `text`
    /// is the accepted message's text so coordinating drivers can
    /// speak it into other threads (the roundtable pattern) — same
    /// parity argument as `AgentCompleted` carrying the response text.
    InputAccepted { thread_id: String, text: String },
    /// The thread is at a runnable turn boundary. `turn` counts
    /// model turns since the last external input on this thread.
    TurnStart { thread_id: String, turn: u32 },
    /// A model response integrated. `text` is the response's text
    /// content; `reasoning` its thinking content (empty when the model
    /// produced none — carried for the same parity reason as `text`:
    /// the builtin autoquery's default query source is
    /// reasoning-then-text, and a thinking-heavy model's terse text
    /// would starve a text-only driver); `tool_calls` the tools it
    /// requested (empty when none). Authoring note: reasoning can be
    /// large — drivers that stash it into state bloat their persisted
    /// JSON.
    ///
    /// `usage` is the completed model call's own token usage;
    /// `thread_usage` the thread's cumulative totals across its
    /// lifetime — the operand of the builtin auto-compaction
    /// threshold, exposed so a driver can own that policy (step 11
    /// slice 5: a driver self-detects "context is outgrowing the
    /// thread" and composes compaction from weave primitives).
    AgentCompleted {
        thread_id: String,
        participant_id: String,
        text: String,
        reasoning: String,
        tool_calls: Vec<ScriptedToolCall>,
        usage: ScriptedCallUsage,
        thread_usage: ScriptedCallUsage,
    },
    /// Every tool requested by the last agent turn has a result.
    ToolsCompleted {
        thread_id: String,
        participant_id: String,
    },
    /// A `derive_thread` effect from an earlier `on_event` call
    /// completed; the driver learns the new thread's id here.
    /// `relationship` echoes the effect's relationship kind (the
    /// enum tag occupies `kind`).
    ThreadDerived {
        thread_id: String,
        relationship: String,
    },
    /// A coordinated thread died outside the driver's own effects — a
    /// model/tool I/O failure, an external cancel, or (delivered at
    /// the weave's first activation after a restart) any ticked
    /// thread found dead at load, including threads the persister
    /// healed mid-flight and threads found Cancelled. Without this
    /// the driver waits forever on an `agent_completed` that can
    /// never arrive (a dead thread refuses `run_agent`; only external
    /// input heals it). Treat it as an idempotent fact — "this thread
    /// is dead" — not an edge: a death the driver already handled
    /// live may be re-reported after a restart. When several deaths
    /// replay at once they arrive primary-first then in ref order,
    /// and a PEER's death may still be undelivered when yours
    /// arrives — don't run a sibling from a death handler unless the
    /// program can tolerate that sibling being dead too. Not fired
    /// live for driver-fault failures (an erroring program would just
    /// error again on the notification), but such deaths DO replay
    /// after a restart — the program may have been fixed in between.
    ThreadFailed { thread_id: String, message: String },
    /// A `query_knowledge` effect resolved (step 11 slice 3). `id`
    /// echoes the effect's driver-supplied correlation token; `query`
    /// echoes the query text; `hits` carry rerank scores unfiltered —
    /// the driver judges relevance in Lua, the scheduler does not
    /// pre-filter.
    ///
    /// Contract for drivers holding a parked boundary on this query:
    /// STORE the fact in state and let the boundary act. After
    /// delivery the scheduler steps the weave's ticked threads, so a
    /// parked boundary re-fires immediately and its handler finds the
    /// stored fact. Moving the held thread from THIS handler instead
    /// works live but breaks across a restart, where a healed
    /// `query_failed` notice and the re-fired boundary share one
    /// event drain — the boundary event queued behind the notice goes
    /// stale the moment the notice's handler moves the thread.
    QueryCompleted {
        id: String,
        query: String,
        hits: Vec<ScriptedKnowledgeHit>,
    },
    /// A `query_knowledge` effect failed — an environmental refusal at
    /// admission (empty scope, nothing hot, missing providers), an
    /// engine error live, or (delivered at the weave's first
    /// activation after a restart, ahead of the triggering event) a
    /// query that was still pending when the process died. Like
    /// `thread_failed`, an idempotent fact; a driver that still wants
    /// the material re-issues the query. The store-don't-move contract
    /// on [`Self::QueryCompleted`] applies here identically.
    QueryFailed { id: String, message: String },
    /// An async `dispatch_thread(sync=false)` child of `thread_id`
    /// (the dispatching parent) reached Completed (step 11 slice 4).
    /// `tool_use_id` correlates with the dispatch call the driver saw
    /// in `agent_completed.tool_calls`; `result` is the child's final
    /// assistant text; `usage` its lifetime totals. May also be
    /// delivered at the weave's first activation after a restart when
    /// the child finished but the event hadn't landed — like
    /// `thread_failed`, treat it as an idempotent fact (dedupe by
    /// `tool_use_id`).
    ///
    /// Movement contract (amends slice 3's store-don't-move, ratified
    /// 2026-08-10): this event usually arrives while the parent is
    /// QUIESCENT — no boundary will re-fire on its own — so the
    /// handler MAY move a quiescent thread (`append_entry` +
    /// `run_agent` both admit Idle/Completed; that pair is the
    /// builtin-parity response). When the parent is parked at a
    /// boundary, store the fact and let the re-fired boundary move it
    /// exactly as slice 3 ratified — the driver knows its own parking
    /// discipline, and `run_agent` on a parked thread refuses loudly.
    /// After delivery the scheduler steps the weave's ticked threads,
    /// so parked boundaries re-fire either way.
    DispatchCompleted {
        thread_id: String,
        tool_use_id: String,
        child_thread_id: String,
        result: String,
        usage: ScriptedDispatchUsage,
    },
    /// The async dispatch terminated without a result: the child
    /// failed, was cancelled (message says which), no longer exists,
    /// or — delivered at the weave's first activation after a restart
    /// — was healed dead by the persister while the process was down.
    /// Idempotent fact; the movement contract on
    /// [`Self::DispatchCompleted`] applies here identically.
    DispatchFailed {
        thread_id: String,
        tool_use_id: String,
        child_thread_id: String,
        message: String,
    },
    /// A compaction of `thread_id` may proceed (step 11 slice 5).
    /// Delivered in answer to the driver's own `request_compaction`
    /// effect (`reason = "driver"`) or to the client's manual
    /// CompactThread message on a scripted weave (`reason =
    /// "manual"`). Carries the thread's RESOLVED compaction config —
    /// `prompt` (the pod-relative `prompt_file` read at delivery
    /// time, or the built-in default), `summary_regex` (Rust-regex
    /// source for [`regex_capture`]-based extraction, group 1 = the
    /// summary body), and `continuation_template` (`{{summary}}`
    /// substitutes) — because pod-dir resolution is scheduler
    /// business and freezing the texts at creation would let a pod's
    /// prompt-file edits silently stop applying.
    ///
    /// The scheduler validated only structure (the weave ticks the
    /// thread, the thread is the current primary, compaction is
    /// enabled). Idleness is the DRIVER's business: store the fact
    /// while the primary is mid-cycle and act at quiescence — better
    /// than the builtin, which rejects a non-idle compact outright.
    /// Like every event, may re-deliver; a driver already compacting
    /// stores or drops it.
    CompactionReady {
        thread_id: String,
        reason: String,
        prompt: String,
        summary_regex: String,
        continuation_template: String,
    },
    /// A `request_compaction` effect was refused — the thread is not
    /// the weave's ticked primary, or its config disables compaction.
    /// Delivered so the driver can clear its own in-flight marker
    /// (the `query_failed` precedent: effects must not fail
    /// invisibly). Manual refusals bounce to the requesting client
    /// instead and never reach the driver.
    CompactionRefused { thread_id: String, message: String },
}

/// Lifetime usage totals of a dispatched child, crossing into Lua on
/// [`ScriptedEvent::DispatchCompleted`]. Mirrors the builtin
/// `<dispatched-thread-notification>` usage block.
#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
pub struct ScriptedDispatchUsage {
    pub total_tokens: u32,
    pub tool_uses: u32,
    pub duration_ms: i64,
}

/// Token usage crossing into Lua on [`ScriptedEvent::AgentCompleted`]
/// — the wire shape of the protocol `Usage` struct, used both for the
/// completed call and for the thread's cumulative totals.
#[derive(Serialize, Debug, Clone, PartialEq, Eq, Default)]
pub struct ScriptedCallUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_read_input_tokens: u32,
    pub cache_creation_input_tokens: u32,
}

impl From<&whisper_agent_protocol::Usage> for ScriptedCallUsage {
    fn from(usage: &whisper_agent_protocol::Usage) -> Self {
        Self {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            cache_read_input_tokens: usage.cache_read_input_tokens,
            cache_creation_input_tokens: usage.cache_creation_input_tokens,
        }
    }
}

/// One reranked hit crossing into Lua. `bucket` is the resolved
/// `scope:name` label; `source_id`/`chunk_id` are the builtin dedup
/// key material (key on source when present, chunk otherwise);
/// `snippet` is chunk text clipped to the effect's `snippet_chars`.
#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct ScriptedKnowledgeHit {
    pub bucket: String,
    pub source_id: String,
    pub chunk_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub locator: Option<String>,
    pub score: f32,
    pub snippet: String,
}

#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
pub struct ScriptedToolCall {
    pub tool_use_id: String,
    pub name: String,
    pub args: serde_json::Value,
}

/// One entry of the effect list returned by `on_event`. Mirrors the
/// executor vocabulary (docs/design_configurable_threads.md): the
/// scheduler admits, journals, and applies each in order.
///
/// (`deny_unknown_fields` is unsupported on internally-tagged enums, so
/// unknown keys on an effect are ignored rather than rejected; missing
/// required fields still fail the decode loudly.)
#[derive(Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ScriptedEffect {
    /// Run one model turn on a ticked thread parked at a turn boundary.
    /// `participant` selects which registered Model participant speaks
    /// (its frozen execution profile — model, system prompt, tools —
    /// shapes the request); omitted, the thread's default responder
    /// runs. Unknown or non-Model ids are refused and journaled.
    RunAgent {
        thread_id: String,
        #[serde(default)]
        participant: Option<String>,
    },
    /// Dispatch every tool requested by the turn parked at the agent
    /// boundary.
    DispatchTools {
        thread_id: String,
    },
    /// Resolve the parked tool requests per-tool: denied calls get
    /// synthesized error results, admitted calls dispatch. Any request
    /// without a decision is denied.
    ResolveTools {
        thread_id: String,
        decisions: Vec<ScriptedToolDecision>,
    },
    /// Take another model turn after tools completed. `nudge`
    /// optionally appends one system-authored message (the
    /// `submit_server_nudge` shape) atomically before the model call —
    /// the only mid-cycle injection point; `append_entry` stays
    /// refused mid-generation.
    ContinueCycle {
        thread_id: String,
        #[serde(default)]
        nudge: Option<String>,
    },
    /// End the thread's cycle and yield for input.
    FinishCycle {
        thread_id: String,
    },
    /// Append one authored entry into a referenced thread's transcript
    /// (pure pollution — the target is not woken).
    AppendEntry {
        thread_id: String,
        author: String,
        text: String,
        #[serde(default)]
        source_thread_id: Option<String>,
        #[serde(default)]
        source_entry_index: Option<usize>,
    },
    /// Create a new thread in the weave's pod, referenced and ticked by
    /// this weave. The driver learns the created id via a follow-up
    /// `thread_derived` event carrying the same relationship kind.
    DeriveThread {
        /// Relationship kind stamped on the weave's ref ("check",
        /// "fork", ...). Named `relationship` on the wire because the
        /// enum tag already occupies `kind`.
        relationship: String,
        #[serde(default)]
        system_prompt: Option<String>,
        #[serde(default)]
        model: Option<String>,
        /// Backend catalog name for the derived thread (step 10 — the
        /// cross-provider cast case). `None` inherits the pod-default
        /// resolution like any created thread; a name is validated
        /// against the pod's `allow.backends` by the bindings resolver,
        /// so an unknown or disallowed choice refuses the derive.
        #[serde(default)]
        backend: Option<String>,
        /// Strip the tool surface: the derived thread's model sees no
        /// tools at all (the permission-checker shape).
        #[serde(default)]
        disable_tools: bool,
        #[serde(default)]
        max_turns: Option<u32>,
        #[serde(default)]
        seed: Vec<ScriptedSeedEntry>,
        #[serde(default)]
        source_thread_id: Option<String>,
        /// Copy the named thread's setup prefix verbatim — system
        /// prompt + tool manifest — into the derived thread,
        /// realigning the default responder's frozen profile (step 11
        /// slice 5: the compaction-continuation shape, where
        /// re-reading the pod's current `system_prompt.md` would
        /// silently swap personalities across the roll). The source
        /// must be referenced by this weave. Exclusive with
        /// `system_prompt` and `disable_tools` — the combination
        /// refuses at execution.
        #[serde(default)]
        setup_from: Option<String>,
        /// Copy the named thread's config — participants, profiles,
        /// model, max_tokens, max_turns, tunables, compaction,
        /// autoquery — and its origin marker. Explicit `model` /
        /// `backend` / `max_turns` fields layer on top afterward.
        /// The source must be referenced by this weave.
        #[serde(default)]
        config_from: Option<String>,
        /// Copy the named thread's bindings — backend, named host_env
        /// entries, mcp_hosts. The source must be referenced by this
        /// weave. Explicit `backend` layers on top afterward.
        #[serde(default)]
        bindings_from: Option<String>,
    },
    /// Set a referenced thread's display title (step 11). Last-write-
    /// wins: a driver's model-generated title overwrites the
    /// scheduler's first-input truncation placeholder, and a failed
    /// title model simply leaves the placeholder standing. The target
    /// only has to be referenced, not ticked — titling is metadata
    /// curation, not coordination.
    SetTitle {
        thread_id: String,
        title: String,
    },
    /// Declare the weave's triggered unit of work done (step 11 slice
    /// 2). For a behavior-spawned weave the scheduler routes the
    /// declaration into behavior bookkeeping (run_count, last_outcome,
    /// overlap-queue release); for any other weave it journals and
    /// moves nothing — drivers declare unconditionally and stay
    /// origin-agnostic. `outcome` defaults to `completed`; `failed`
    /// carries `message` into the recorded failure.
    CompleteRun {
        #[serde(default)]
        outcome: ScriptedRunOutcome,
        #[serde(default)]
        message: Option<String>,
    },
    /// Run one knowledge query asynchronously (step 11 slice 3 — the
    /// first async non-thread effect). `id` is a driver-supplied
    /// correlation token echoed by the resolving `query_completed` /
    /// `query_failed` event; a duplicate id while one is in flight
    /// refuses. `buckets` uses the config ref grammar (bare,
    /// `server:name`, `pod:name`; empty = every bucket in the weave's
    /// pod scope). `top_k` defaults 5, max 20, zero refused (the
    /// `knowledge_query` tool's bounds); `snippet_chars` clips chunk
    /// text crossing into Lua (default 500); `hot_only` (default
    /// true) skips buckets not already loaded — flip it explicitly
    /// for cold-capable queries (the scheduled-digest shape).
    QueryKnowledge {
        id: String,
        query: String,
        #[serde(default)]
        buckets: Vec<String>,
        #[serde(default)]
        top_k: Option<u32>,
        #[serde(default)]
        snippet_chars: Option<u32>,
        #[serde(default)]
        hot_only: Option<bool>,
    },
    /// Ask the scheduler for the thread's resolved compaction config
    /// (step 11 slice 5). Answered within the same activation's drain
    /// by `compaction_ready` (validation passed — prompt, regex, and
    /// template resolved) or `compaction_refused` (not the ticked
    /// primary, or compaction disabled). The driver then composes the
    /// flow itself: append the prompt, run the summary turn, extract
    /// with `regex_capture`, derive the continuation, advance the
    /// head. The round trip exists because `prompt_file` resolves
    /// against the pod directory — scheduler business — at request
    /// time, not creation time.
    RequestCompaction {
        thread_id: String,
    },
    /// Promote a referenced, self-ticked thread to primary; the previous
    /// primary becomes a dormant auxiliary. The compaction-roll primitive.
    AdvanceHead {
        thread_id: String,
    },
    AdoptTicker {
        thread_id: String,
    },
    ReleaseTicker {
        thread_id: String,
    },
}

/// What a `complete_run` effect declares about the triggered work.
/// Deliberately binary — `cancelled` is a fact about external
/// intervention, not something a driver declares about its own run.
#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ScriptedRunOutcome {
    #[default]
    Completed,
    Failed,
}

#[derive(Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ScriptedToolDecision {
    pub tool_use_id: String,
    pub allow: bool,
    /// Shown to the model as the denied call's error tool_result.
    #[serde(default)]
    pub message: Option<String>,
}

#[derive(Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ScriptedSeedEntry {
    pub author: String,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScriptedOutcome {
    pub effects: Vec<ScriptedEffect>,
    /// The driver's state after this event, persisted verbatim.
    pub state: serde_json::Value,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OutcomeWire {
    #[serde(default)]
    effects: Vec<ScriptedEffect>,
    /// Omitted state keeps the pre-event state.
    #[serde(default)]
    state: Option<serde_json::Value>,
}

/// Build the sandboxed VM every driver call runs in: minimal stdlib,
/// base-library escape hatches stripped, memory ceiling, instruction
/// budget.
fn sandboxed_vm() -> Result<Lua, String> {
    let lua = Lua::new_with(
        mlua::StdLib::TABLE | mlua::StdLib::STRING | mlua::StdLib::MATH | mlua::StdLib::UTF8,
        mlua::LuaOptions::default(),
    )
    .map_err(|e| format!("lua init: {e}"))?;
    lua.set_memory_limit(MEMORY_LIMIT_BYTES)
        .map_err(|e| format!("lua memory limit: {e}"))?;
    // mlua loads the Lua base library unconditionally (before the StdLib
    // mask applies), so strip what breaks the sandbox contract:
    // - pcall/xpcall would let a driver catch the instruction-budget
    //   error and loop forever, wedging the single-writer scheduler;
    // - dofile/loadfile/load reach the filesystem (dofile() with no
    //   argument reads the process's stdin!) outside the pod boundary;
    // - collectgarbage and print leak allocator state / spam stdout.
    // math.random/randomseed go too: driver state must be a
    // deterministic function of program + events or the persisted
    // journal stops explaining the run.
    let globals = lua.globals();
    for name in [
        "pcall",
        "xpcall",
        "load",
        "dofile",
        "loadfile",
        "collectgarbage",
        "print",
    ] {
        globals
            .set(name, mlua::Value::Nil)
            .map_err(|e| format!("lua sandbox strip: {e}"))?;
    }
    if let Ok(math) = globals.get::<mlua::Table>("math") {
        let _ = math.set("random", mlua::Value::Nil);
        let _ = math.set("randomseed", mlua::Value::Nil);
    }
    drop(globals);
    let spent = std::cell::Cell::new(0u64);
    let hook_installed = lua.set_hook(
        mlua::HookTriggers::new().every_nth_instruction(HOOK_EVERY),
        move |_lua, _debug| {
            let now = spent.get() + HOOK_EVERY as u64;
            spent.set(now);
            if now > INSTRUCTION_BUDGET {
                Err(mlua::Error::RuntimeError(format!(
                    "driver exceeded the {INSTRUCTION_BUDGET}-instruction budget"
                )))
            } else {
                Ok(mlua::VmState::Continue)
            }
        },
    );
    hook_installed.map_err(|e| format!("lua instruction hook: {e}"))?;
    // `regex_capture(pattern, text)` — the one injected helper (step
    // 11 slice 5). Lua patterns cannot express the configured
    // `summary_regex` (Rust-regex syntax; the default's end-anchor
    // handles nested tags), so extraction gets a real engine. Returns
    // `capture, nil` (group 1 when the pattern has groups — nil if
    // the group didn't participate in the match — or the whole match
    // for a group-less pattern), `nil, nil` on no match, or
    // `nil, err` for a bad/oversized pattern, oversized text, or
    // non-UTF-8 input. Every refusal is an error RETURN, never a Lua
    // error — pcall is stripped, so a raised error would be an
    // uncatchable activation fault.
    //
    // The instruction budget meters Lua instructions, not Rust time,
    // so both operands are bounded here (review fix, slice 5): the
    // compiled pattern at 64 KiB (the builtin summary regex needs a
    // few hundred bytes; regex search worst-cases at compiled-size ×
    // text-length when the lazy DFA degrades) and the text at 1 MiB
    // (a summary reply is bounded by max_tokens, far below this).
    const REGEX_PATTERN_SIZE_LIMIT: usize = 64 * 1024;
    const REGEX_TEXT_LIMIT: usize = 1024 * 1024;
    let regex_capture = lua
        .create_function(
            move |_, (pattern, text): (mlua::LuaString, mlua::LuaString)| {
                let Ok(pattern) = pattern.to_str() else {
                    return Ok((
                        None,
                        Some("regex_capture: pattern is not valid UTF-8".into()),
                    ));
                };
                let Ok(text) = text.to_str() else {
                    return Ok((None, Some("regex_capture: text is not valid UTF-8".into())));
                };
                if text.len() > REGEX_TEXT_LIMIT {
                    return Ok((
                        None,
                        Some(format!(
                            "regex_capture: text exceeds {REGEX_TEXT_LIMIT} bytes"
                        )),
                    ));
                }
                let compiled = regex::RegexBuilder::new(&pattern)
                    .size_limit(REGEX_PATTERN_SIZE_LIMIT)
                    .build();
                Ok(match compiled {
                    Ok(re) => match re.captures(&text) {
                        Some(caps) => {
                            let capture = if re.captures_len() > 1 {
                                caps.get(1)
                            } else {
                                caps.get(0)
                            }
                            .map(|m| m.as_str().to_string());
                            (capture, None)
                        }
                        None => (None, None),
                    },
                    Err(e) => (None, Some(format!("regex_capture: {e}"))),
                })
            },
        )
        .map_err(|e| format!("lua regex_capture: {e}"))?;
    lua.globals()
        .set("regex_capture", regex_capture)
        .map_err(|e| format!("lua regex_capture install: {e}"))?;
    Ok(lua)
}

/// Run one driver event through a fresh sandboxed VM.
///
/// `chunk_name` labels Lua error messages (use the program name).
/// Errors are driver failures, not scheduler failures — the caller
/// fails the weave's coordinated work with the message.
pub fn run_event(
    source: &str,
    chunk_name: &str,
    state: &serde_json::Value,
    event: &ScriptedEvent,
    config: &std::collections::BTreeMap<String, serde_json::Value>,
) -> Result<ScriptedOutcome, String> {
    let lua = sandboxed_vm()?;
    lua.load(source)
        .set_name(chunk_name)
        .exec()
        .map_err(|e| format!("driver `{chunk_name}` failed to load: {e}"))?;
    let handler: mlua::Function = lua
        .globals()
        .get("on_event")
        .map_err(|_| format!("driver `{chunk_name}` defines no `on_event` function"))?;

    let state_value = lua
        .to_value(state)
        .map_err(|e| format!("driver state to lua: {e}"))?;
    let event_value = lua
        .to_value(event)
        .map_err(|e| format!("driver event to lua: {e}"))?;
    let config_value = lua
        .to_value(config)
        .map_err(|e| format!("driver config to lua: {e}"))?;
    let returned: mlua::Value = handler
        .call((state_value, event_value, config_value))
        .map_err(|e| format!("driver `{chunk_name}` on_event: {e}"))?;

    if returned.is_nil() {
        // A nil return is the explicit "nothing to do, state unchanged".
        return Ok(ScriptedOutcome {
            effects: Vec::new(),
            state: state.clone(),
        });
    }
    let wire: OutcomeWire = lua
        .from_value(returned)
        .map_err(|e| format!("driver `{chunk_name}` returned a malformed outcome: {e}"))?;
    Ok(ScriptedOutcome {
        effects: wire.effects,
        state: wire.state.unwrap_or_else(|| state.clone()),
    })
}

/// Evaluate the program's optional `present(state, config) -> blocks`
/// entry point (migration step 7b): a pure function of persisted driver
/// state (plus the creation-frozen knob config) composing the weave's
/// presentation structure — replayable by construction, since it can see
/// nothing else.
///
/// `Ok(None)` means the program defines no `present`; the caller
/// synthesizes the degenerate presentation. Errors degrade the display
/// (the caller falls back to degenerate) — they never fail the weave's
/// coordinated work, per "presentation can degrade, ground truth
/// cannot".
pub fn run_present(
    source: &str,
    chunk_name: &str,
    state: &serde_json::Value,
    config: &std::collections::BTreeMap<String, serde_json::Value>,
) -> Result<Option<Vec<whisper_agent_protocol::weave::PresentationBlock>>, String> {
    let lua = sandboxed_vm()?;
    lua.load(source)
        .set_name(chunk_name)
        .exec()
        .map_err(|e| format!("driver `{chunk_name}` failed to load: {e}"))?;
    let Ok(presenter) = lua.globals().get::<mlua::Function>("present") else {
        return Ok(None);
    };
    let state_value = lua
        .to_value(state)
        .map_err(|e| format!("driver state to lua: {e}"))?;
    let config_value = lua
        .to_value(config)
        .map_err(|e| format!("driver config to lua: {e}"))?;
    let returned: mlua::Value = presenter
        .call((state_value, config_value))
        .map_err(|e| format!("driver `{chunk_name}` present: {e}"))?;
    if returned.is_nil() {
        return Ok(None);
    }
    let blocks: Vec<whisper_agent_protocol::weave::PresentationBlock> = lua
        .from_value(returned)
        .map_err(|e| format!("driver `{chunk_name}` returned malformed presentation: {e}"))?;
    Ok(Some(blocks))
}

/// Evaluate the program's optional `describe()` declaration (migration
/// step 10): a pure, argument-free entry point returning the driver's
/// display metadata and typed configuration knobs. Runs at listing time
/// (the `DescribeDriver` request feeding the new-thread form) and at
/// creation time (validating submitted knob values) — never during
/// coordination.
///
/// `Ok(None)` means the program defines no `describe` (or it returned
/// nil); the caller treats that as the empty declaration — no knobs, the
/// pre-step-10 contract. Errors here refuse thread creation the same way
/// an unloadable program does: a driver whose declaration can't be
/// evaluated shouldn't get a weave that will fail at its first event.
pub fn run_describe(
    source: &str,
    chunk_name: &str,
) -> Result<Option<whisper_agent_protocol::driver::DriverDescription>, String> {
    let lua = sandboxed_vm()?;
    lua.load(source)
        .set_name(chunk_name)
        .exec()
        .map_err(|e| format!("driver `{chunk_name}` failed to load: {e}"))?;
    let Ok(describer) = lua.globals().get::<mlua::Function>("describe") else {
        return Ok(None);
    };
    let returned: mlua::Value = describer
        .call(())
        .map_err(|e| format!("driver `{chunk_name}` describe: {e}"))?;
    if returned.is_nil() {
        return Ok(None);
    }
    // Deserialize through a self-describing Value first: an empty Lua
    // table is indistinguishable from an empty map and arrives as `{}`,
    // which a typed `Vec<KnobSpec>` would reject even though
    // `knobs = {}` is a perfectly good "no knobs" — while a NON-empty
    // map (`knobs = { foo = {...} }`, a natural Lua authoring mistake)
    // would silently deserialize as zero knobs via the sequence path.
    // Normalize the former, refuse the latter loudly.
    let mut wire: serde_json::Value = lua
        .from_value(returned)
        .map_err(|e| format!("driver `{chunk_name}` returned a malformed description: {e}"))?;
    if let Some(knobs) = wire.get_mut("knobs") {
        match knobs {
            serde_json::Value::Object(map) if map.is_empty() => {
                *knobs = serde_json::Value::Array(Vec::new());
            }
            serde_json::Value::Object(_) => {
                return Err(format!(
                    "driver `{chunk_name}` describe(): `knobs` must be an array of knob tables, \
                     not a map keyed by id"
                ));
            }
            _ => {}
        }
    }
    let description: whisper_agent_protocol::driver::DriverDescription =
        serde_json::from_value(wire)
            .map_err(|e| format!("driver `{chunk_name}` returned a malformed description: {e}"))?;
    description
        .validate_declaration()
        .map_err(|e| format!("driver `{chunk_name}` describe(): {e}"))?;
    Ok(Some(description))
}

/// Content hash stamped on the weave when a scripted driver starts
/// coordinating, so persisted behavior stays explainable after the
/// program file changes.
pub fn program_hash(source: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(source.as_bytes());
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Config-less shims: these tests predate step-10 knobs and their
    /// drivers ignore the third argument, exactly like production
    /// drivers written before it existed.
    fn run_event(
        source: &str,
        chunk_name: &str,
        state: &serde_json::Value,
        event: &ScriptedEvent,
    ) -> Result<ScriptedOutcome, String> {
        super::run_event(source, chunk_name, state, event, &Default::default())
    }

    fn run_present(
        source: &str,
        chunk_name: &str,
        state: &serde_json::Value,
    ) -> Result<Option<Vec<whisper_agent_protocol::weave::PresentationBlock>>, String> {
        super::run_present(source, chunk_name, state, &Default::default())
    }

    fn event_turn_start(thread: &str, turn: u32) -> ScriptedEvent {
        ScriptedEvent::TurnStart {
            thread_id: thread.into(),
            turn,
        }
    }

    #[test]
    fn echo_driver_reads_event_and_returns_effect_and_state() {
        let src = r#"
            function on_event(state, event)
              state.seen = (state.seen or 0) + 1
              if event.kind == "turn_start" then
                return { effects = { { kind = "run_agent", thread_id = event.thread_id } },
                         state = state }
              end
              return { state = state }
            end
        "#;
        let out = run_event(src, "echo", &json!({}), &event_turn_start("t-1", 1)).unwrap();
        assert_eq!(
            out.effects,
            vec![ScriptedEffect::RunAgent {
                thread_id: "t-1".into(),
                participant: None,
            }]
        );
        assert_eq!(out.state, json!({"seen": 1}));

        // State round-trips into the next event.
        let out2 = run_event(
            src,
            "echo",
            &out.state,
            &ScriptedEvent::InputAccepted {
                thread_id: "t-1".into(),
                text: "again".into(),
            },
        )
        .unwrap();
        assert!(out2.effects.is_empty());
        assert_eq!(out2.state, json!({"seen": 2}));
    }

    #[test]
    fn nil_return_means_no_effects_state_unchanged() {
        let src = "function on_event(state, event) return nil end";
        let state = json!({"keep": true});
        let out = run_event(src, "nil", &state, &event_turn_start("t", 1)).unwrap();
        assert!(out.effects.is_empty());
        assert_eq!(out.state, state);
    }

    #[test]
    fn omitted_state_keeps_prior_state() {
        let src = r#"
            function on_event(state, event)
              return { effects = { { kind = "finish_cycle", thread_id = "t" } } }
            end
        "#;
        let state = json!({"n": 3});
        let out = run_event(src, "keep", &state, &event_turn_start("t", 1)).unwrap();
        assert_eq!(out.state, state);
        assert_eq!(out.effects.len(), 1);
    }

    #[test]
    fn full_effect_vocabulary_decodes() {
        let src = r#"
            function on_event(state, event)
              return { effects = {
                { kind = "run_agent" },
              }, state = state }
            end
        "#;
        // Missing required field (thread_id) — decode must fail loudly,
        // not silently drop the effect.
        let err = run_event(src, "vocab", &json!({}), &event_turn_start("t", 1)).unwrap_err();
        assert!(err.contains("malformed outcome"), "{err}");

        let src_ok = r#"
            function on_event(state, event)
              return { effects = {
                { kind = "resolve_tools", thread_id = "t-main", decisions = {
                    { tool_use_id = "toolu-1", allow = true },
                    { tool_use_id = "toolu-2", allow = false, message = "checker denied" },
                } },
                { kind = "append_entry", thread_id = "t-main", author = "checker",
                  text = "verdict: allow", source_thread_id = "t-check" },
                { kind = "derive_thread", relationship = "check", disable_tools = true,
                  system_prompt = "You are a permission checker.",
                  seed = { { author = "user", text = "context" } } },
                { kind = "release_ticker", thread_id = "t-check" },
                { kind = "adopt_ticker", thread_id = "t-check" },
                { kind = "dispatch_tools", thread_id = "t-main" },
                { kind = "continue_cycle", thread_id = "t-main" },
                { kind = "continue_cycle", thread_id = "t-main",
                  nudge = "related material surfaced" },
                { kind = "complete_run" },
                { kind = "complete_run", outcome = "failed", message = "no quorum" },
                { kind = "query_knowledge", id = "q1", query = "sharded consensus" },
                { kind = "query_knowledge", id = "q2", query = "cold archive",
                  buckets = { "server:wiki" }, top_k = 3, snippet_chars = 200,
                  hot_only = false },
                { kind = "request_compaction", thread_id = "t-main" },
                { kind = "derive_thread", relationship = "compaction",
                  setup_from = "t-main", config_from = "t-main",
                  bindings_from = "t-main",
                  seed = { { author = "user", text = "summary" } },
                  source_thread_id = "t-main" },
              } }
            end
        "#;
        let out = run_event(src_ok, "vocab", &json!({}), &event_turn_start("t", 1)).unwrap();
        assert_eq!(out.effects.len(), 14);
        assert!(matches!(
            &out.effects[0],
            ScriptedEffect::ResolveTools { decisions, .. }
                if decisions.len() == 2 && !decisions[1].allow
        ));
        assert!(matches!(
            &out.effects[2],
            ScriptedEffect::DeriveThread { relationship, disable_tools: true, seed, .. }
                if relationship == "check" && seed.len() == 1
        ));
        assert_eq!(
            out.effects[6],
            ScriptedEffect::ContinueCycle {
                thread_id: "t-main".into(),
                nudge: None,
            },
            "bare continue_cycle carries no nudge"
        );
        assert_eq!(
            out.effects[7],
            ScriptedEffect::ContinueCycle {
                thread_id: "t-main".into(),
                nudge: Some("related material surfaced".into()),
            }
        );
        assert_eq!(
            out.effects[8],
            ScriptedEffect::CompleteRun {
                outcome: ScriptedRunOutcome::Completed,
                message: None,
            },
            "bare complete_run defaults to a completed outcome"
        );
        assert_eq!(
            out.effects[9],
            ScriptedEffect::CompleteRun {
                outcome: ScriptedRunOutcome::Failed,
                message: Some("no quorum".into()),
            }
        );
        assert_eq!(
            out.effects[10],
            ScriptedEffect::QueryKnowledge {
                id: "q1".into(),
                query: "sharded consensus".into(),
                buckets: Vec::new(),
                top_k: None,
                snippet_chars: None,
                hot_only: None,
            },
            "bare query_knowledge defaults every bound"
        );
        assert_eq!(
            out.effects[11],
            ScriptedEffect::QueryKnowledge {
                id: "q2".into(),
                query: "cold archive".into(),
                buckets: vec!["server:wiki".into()],
                top_k: Some(3),
                snippet_chars: Some(200),
                hot_only: Some(false),
            }
        );
        assert_eq!(
            out.effects[12],
            ScriptedEffect::RequestCompaction {
                thread_id: "t-main".into(),
            }
        );
        assert!(
            matches!(
                &out.effects[13],
                ScriptedEffect::DeriveThread {
                    setup_from: Some(s),
                    config_from: Some(c),
                    bindings_from: Some(b),
                    system_prompt: None,
                    disable_tools: false,
                    ..
                } if s == "t-main" && c == "t-main" && b == "t-main"
            ),
            "granular inheritance directives decode: {:?}",
            out.effects[13]
        );
    }

    #[test]
    fn agent_completed_carries_call_and_thread_usage() {
        // Pins the Lua-side shape of the usage payloads (step 11 slice
        // 5): per-call and cumulative, both plain integer tables — the
        // material a driver's self-detected compaction threshold reads.
        let src = r#"
            function on_event(state, event)
              if event.kind == "agent_completed" then
                return { state = {
                  call_in = event.usage.input_tokens,
                  call_out = event.usage.output_tokens,
                  cached = event.usage.cache_read_input_tokens,
                  total_in = event.thread_usage.input_tokens,
                  over = event.thread_usage.input_tokens > 200000,
                } }
              end
              return { state = state }
            end
        "#;
        let event = ScriptedEvent::AgentCompleted {
            thread_id: "t".into(),
            participant_id: "assistant".into(),
            text: "done".into(),
            reasoning: String::new(),
            tool_calls: Vec::new(),
            usage: ScriptedCallUsage {
                input_tokens: 120_000,
                output_tokens: 900,
                cache_read_input_tokens: 80_000,
                cache_creation_input_tokens: 0,
            },
            thread_usage: ScriptedCallUsage {
                input_tokens: 250_000,
                output_tokens: 4_200,
                cache_read_input_tokens: 200_000,
                cache_creation_input_tokens: 1_000,
            },
        };
        let out = run_event(src, "usage", &json!({}), &event).unwrap();
        assert_eq!(
            out.state,
            json!({
                "call_in": 120_000,
                "call_out": 900,
                "cached": 80_000,
                "total_in": 250_000,
                "over": true,
            })
        );
    }

    #[test]
    fn compaction_events_are_readable_and_regex_capture_extracts() {
        // The compaction_ready payload crosses into Lua intact, and
        // `regex_capture` runs the CONFIGURED Rust-regex verbatim —
        // including the default's end-anchor, which must pick the
        // OUTER close tag when the body verbatim-quotes a nested
        // <summary> pair (the case Lua patterns cannot express).
        let src = r#"
            function on_event(state, event)
              if event.kind == "compaction_ready" then
                local body, err = regex_capture(event.summary_regex, state.reply)
                return { state = {
                  reason = event.reason,
                  prompt = event.prompt,
                  template = event.continuation_template,
                  body = body,
                  err = err,
                } }
              end
              if event.kind == "compaction_refused" then
                return { state = { refused = event.message } }
              end
              return { state = state }
            end
        "#;
        let event = ScriptedEvent::CompactionReady {
            thread_id: "t".into(),
            reason: "manual".into(),
            prompt: "Summarize yourself.".into(),
            summary_regex: r"(?s)<summary>\s*(.*?\S)\s*</summary>\s*\z".into(),
            continuation_template: "Continue: {{summary}}".into(),
        };
        let state = json!({
            "reply": "<summary>outer <summary>inner</summary> tail</summary>",
        });
        let out = run_event(src, "compact", &state, &event).unwrap();
        assert_eq!(out.state["reason"], "manual");
        assert_eq!(out.state["prompt"], "Summarize yourself.");
        assert_eq!(out.state["template"], "Continue: {{summary}}");
        assert_eq!(
            out.state["body"], "outer <summary>inner</summary> tail",
            "end-anchored extraction lands on the outer close tag"
        );
        assert!(out.state["err"].is_null());

        let refused = ScriptedEvent::CompactionRefused {
            thread_id: "t".into(),
            message: "compaction is disabled for this thread".into(),
        };
        let out = run_event(src, "compact", &json!({}), &refused).unwrap();
        assert_eq!(
            out.state["refused"],
            "compaction is disabled for this thread"
        );
    }

    #[test]
    fn regex_capture_no_match_groupless_and_bad_patterns() {
        let src = r#"
            function on_event(state, event)
              local miss, miss_err = regex_capture("<x>(.*)</x>", "nothing here")
              local whole, whole_err = regex_capture("[a-z]+", "abc123")
              local bad, bad_err = regex_capture("(unclosed", "text")
              return { state = {
                miss = miss == nil,
                miss_err = miss_err == nil,
                whole = whole,
                whole_err = whole_err == nil,
                bad = bad == nil,
                bad_err = bad_err ~= nil,
              } }
            end
        "#;
        let out = run_event(src, "regex", &json!({}), &event_turn_start("t", 1)).unwrap();
        assert_eq!(
            out.state,
            json!({
                "miss": true,
                "miss_err": true,
                "whole": "abc",
                "whole_err": true,
                "bad": true,
                "bad_err": true,
            })
        );
    }

    #[test]
    fn query_completed_event_is_readable_from_lua() {
        // Pins the Lua-side shape of the async completion: hits arrive
        // as an array of tables with numeric scores and the dedup key
        // material (source_id / chunk_id) as plain strings.
        let src = r#"
            function on_event(state, event)
              if event.kind == "query_completed" then
                local best = event.hits[1]
                return { state = {
                  id = event.id,
                  query = event.query,
                  n = #event.hits,
                  bucket = best.bucket,
                  source = best.source_id,
                  chunk = best.chunk_id,
                  strong = best.score > 0.5,
                  snippet = best.snippet,
                  locator = best.locator,
                } }
              end
              return { state = state }
            end
        "#;
        let event = ScriptedEvent::QueryCompleted {
            id: "q1".into(),
            query: "sharded consensus".into(),
            hits: vec![
                ScriptedKnowledgeHit {
                    bucket: "server:wiki".into(),
                    source_id: "Paxos".into(),
                    chunk_id: "c-9".into(),
                    locator: Some("§2".into()),
                    score: 0.83,
                    snippet: "The synod protocol…".into(),
                },
                ScriptedKnowledgeHit {
                    bucket: "server:wiki".into(),
                    source_id: String::new(),
                    chunk_id: "c-12".into(),
                    locator: None,
                    score: 0.31,
                    snippet: "…".into(),
                },
            ],
        };
        let out = run_event(src, "qc", &json!({}), &event).unwrap();
        assert_eq!(
            out.state,
            json!({
                "id": "q1",
                "query": "sharded consensus",
                "n": 2,
                "bucket": "server:wiki",
                "source": "Paxos",
                "chunk": "c-9",
                "strong": true,
                "snippet": "The synod protocol…",
                "locator": "§2",
            })
        );

        let failed = ScriptedEvent::QueryFailed {
            id: "q1".into(),
            message: "lost to restart".into(),
        };
        let src_failed = r#"
            function on_event(state, event)
              return { state = { id = event.id, message = event.message } }
            end
        "#;
        let out = run_event(src_failed, "qf", &json!({}), &failed).unwrap();
        assert_eq!(out.state, json!({"id": "q1", "message": "lost to restart"}));
    }

    #[test]
    fn dispatch_events_are_readable_from_lua() {
        // Pins the Lua-side shape of the async dispatch terminal: the
        // usage block arrives as a nested table with numeric fields,
        // and correlation rides tool_use_id.
        let src = r#"
            function on_event(state, event)
              if event.kind == "dispatch_completed" then
                return { state = {
                  parent = event.thread_id,
                  tu = event.tool_use_id,
                  child = event.child_thread_id,
                  result = event.result,
                  tokens = event.usage.total_tokens,
                  tools = event.usage.tool_uses,
                  slow = event.usage.duration_ms > 1000,
                } }
              end
              return { state = state }
            end
        "#;
        let event = ScriptedEvent::DispatchCompleted {
            thread_id: "parent-1".into(),
            tool_use_id: "tu-7".into(),
            child_thread_id: "child-9".into(),
            result: "Найдено: the answer is 42.".into(),
            usage: ScriptedDispatchUsage {
                total_tokens: 1234,
                tool_uses: 3,
                duration_ms: 45_000,
            },
        };
        let out = run_event(src, "dc", &json!({}), &event).unwrap();
        assert_eq!(
            out.state,
            json!({
                "parent": "parent-1",
                "tu": "tu-7",
                "child": "child-9",
                "result": "Найдено: the answer is 42.",
                "tokens": 1234,
                "tools": 3,
                "slow": true,
            })
        );

        let failed = ScriptedEvent::DispatchFailed {
            thread_id: "parent-1".into(),
            tool_use_id: "tu-7".into(),
            child_thread_id: "child-9".into(),
            message: "child thread was cancelled".into(),
        };
        let src_failed = r#"
            function on_event(state, event)
              return { state = { tu = event.tool_use_id, message = event.message } }
            end
        "#;
        let out = run_event(src_failed, "df", &json!({}), &failed).unwrap();
        assert_eq!(
            out.state,
            json!({"tu": "tu-7", "message": "child thread was cancelled"})
        );
    }

    #[test]
    fn runaway_driver_hits_the_instruction_budget() {
        let src = r#"
            function on_event(state, event)
              local n = 0
              while true do n = n + 1 end
            end
        "#;
        let err = run_event(src, "runaway", &json!({}), &event_turn_start("t", 1)).unwrap_err();
        assert!(err.contains("instruction budget"), "{err}");
    }

    #[test]
    fn memory_bomb_hits_the_memory_limit() {
        let src = r#"
            function on_event(state, event)
              local s = "x"
              while true do s = s .. s end
            end
        "#;
        let err = run_event(src, "membomb", &json!({}), &event_turn_start("t", 1)).unwrap_err();
        assert!(!err.is_empty());
    }

    #[test]
    fn sandbox_has_no_io_os_require_pcall_load_or_random() {
        let src = r#"
            function on_event(state, event)
              return { effects = {}, state = {
                has_io = io ~= nil,
                has_os = os ~= nil,
                has_require = require ~= nil,
                has_pcall = pcall ~= nil,
                has_xpcall = xpcall ~= nil,
                has_load = load ~= nil,
                has_dofile = dofile ~= nil,
                has_loadfile = loadfile ~= nil,
                has_collectgarbage = collectgarbage ~= nil,
                has_random = math.random ~= nil,
                has_coroutine = coroutine ~= nil,
              } }
            end
        "#;
        let out = run_event(src, "sandbox", &json!({}), &event_turn_start("t", 1)).unwrap();
        let object = out.state.as_object().unwrap();
        for (key, value) in object {
            assert_eq!(value, &json!(false), "sandbox leaks {key}");
        }
        assert_eq!(object.len(), 11);
    }

    #[test]
    fn instruction_budget_is_not_catchable() {
        // pcall is stripped from the sandbox, so the budget error cannot
        // be swallowed by a hostile driver's catch-and-retry loop — the
        // demonstrated wedge vector against the single-writer scheduler.
        let src = r#"
            function on_event(state, event)
              while true do
                local ok = pcall and pcall(function() while true do end end)
              end
            end
        "#;
        let err = run_event(src, "hostile", &json!({}), &event_turn_start("t", 1)).unwrap_err();
        assert!(err.contains("instruction budget"), "{err}");
    }

    #[test]
    fn missing_handler_and_syntax_errors_are_driver_failures() {
        let err =
            run_event("x = 1", "nohandler", &json!({}), &event_turn_start("t", 1)).unwrap_err();
        assert!(err.contains("no `on_event`"), "{err}");

        let err = run_event(
            "function on_event(",
            "syntax",
            &json!({}),
            &event_turn_start("t", 1),
        )
        .unwrap_err();
        assert!(err.contains("failed to load"), "{err}");
    }

    #[test]
    fn program_hash_is_stable_content_addressing() {
        let a = program_hash("return 1");
        assert_eq!(a, program_hash("return 1"));
        assert_ne!(a, program_hash("return 2"));
        assert_eq!(a.len(), 64);
    }

    #[test]
    fn present_absent_or_nil_means_no_presentation() {
        let no_present = "function on_event(s, e) return nil end";
        assert_eq!(run_present(no_present, "d", &json!({})).unwrap(), None);
        let nil_present = r#"
            function on_event(s, e) return nil end
            function present(state) return nil end
        "#;
        assert_eq!(run_present(nil_present, "d", &json!({})).unwrap(), None);
    }

    #[test]
    fn present_composes_blocks_from_state_alone() {
        use whisper_agent_protocol::weave::PresentationBlock;
        let src = r#"
            function on_event(s, e) return nil end
            function present(state)
              local blocks = {
                { kind = "primary_transcript", thread_id = state.primary },
                { kind = "status", text = "checking " .. state.n .. " calls" },
              }
              if state.checker then
                blocks[#blocks + 1] =
                  { kind = "thread_list", thread_ids = { state.checker } }
              end
              return blocks
            end
        "#;
        let blocks = run_present(src, "d", &json!({"primary": "t1", "n": 2, "checker": "t2"}))
            .unwrap()
            .unwrap();
        assert_eq!(
            blocks,
            vec![
                PresentationBlock::PrimaryTranscript {
                    thread_id: "t1".into()
                },
                PresentationBlock::Status {
                    text: "checking 2 calls".into()
                },
                PresentationBlock::ThreadList {
                    thread_ids: vec!["t2".into()]
                },
            ]
        );
    }

    #[test]
    fn present_failures_are_reported_not_swallowed() {
        let erroring = r#"
            function on_event(s, e) return nil end
            function present(state) error("display bug") end
        "#;
        let err = run_present(erroring, "d", &json!({})).unwrap_err();
        assert!(err.contains("display bug"), "got: {err}");

        let malformed = r#"
            function on_event(s, e) return nil end
            function present(state) return { { kind = "status" } } end
        "#;
        let err = run_present(malformed, "d", &json!({})).unwrap_err();
        assert!(err.contains("malformed presentation"), "got: {err}");
    }

    #[test]
    fn present_runs_under_the_same_instruction_budget() {
        let runaway = r#"
            function on_event(s, e) return nil end
            function present(state)
              local n = 0
              while true do n = n + 1 end
            end
        "#;
        let err = run_present(runaway, "d", &json!({})).unwrap_err();
        assert!(err.contains("instruction budget"), "got: {err}");
    }

    #[test]
    fn config_reaches_on_event_and_present() {
        let src = r#"
            function on_event(state, event, config)
              local voice = config["voice.model"]
              state.model = voice and voice.model or "unset"
              return { effects = {}, state = state }
            end
            function present(state, config)
              return { { kind = "status", text = "motto: " .. (config.motto or "?") } }
            end
        "#;
        let config: std::collections::BTreeMap<String, serde_json::Value> = [
            (
                "voice.model".to_string(),
                json!({"backend": "anthropic", "model": "claude-sonnet-5"}),
            ),
            ("motto".to_string(), json!("onward")),
        ]
        .into_iter()
        .collect();
        let out =
            super::run_event(src, "d", &json!({}), &event_turn_start("t", 1), &config).unwrap();
        assert_eq!(out.state["model"], "claude-sonnet-5");
        let blocks = super::run_present(src, "d", &json!({}), &config)
            .unwrap()
            .unwrap();
        assert_eq!(
            blocks,
            vec![whisper_agent_protocol::weave::PresentationBlock::Status {
                text: "motto: onward".into()
            }]
        );
    }

    #[test]
    fn describe_declares_typed_knobs() {
        let src = r#"
            function on_event(state, event) return nil end
            function describe()
              return {
                label = "Test driver",
                knobs = {
                  { id = "voice.model", label = "Voice", type = "model", required = true },
                  { id = "rounds", type = "integer", default = 2, min = 1, max = 9 },
                },
              }
            end
        "#;
        let description = run_describe(src, "d").unwrap().unwrap();
        assert_eq!(description.label.as_deref(), Some("Test driver"));
        let knobs = &description.knobs;
        assert_eq!(knobs.len(), 2);
        assert_eq!(
            knobs[0].kind,
            whisper_agent_protocol::driver::KnobKind::Model
        );
        assert!(knobs[0].required);
        assert_eq!(knobs[1].default, Some(json!(2)));
        assert_eq!((knobs[1].min, knobs[1].max), (Some(1.0), Some(9.0)));
    }

    #[test]
    fn describe_normalizes_empty_knobs_and_refuses_maps_and_bad_declarations() {
        // `knobs = {}` is ambiguous in Lua (empty array == empty map);
        // it must land as the empty declaration, not a type error.
        let empty = r#"
            function on_event(state, event) return nil end
            function describe() return { label = "L", knobs = {} } end
        "#;
        let description = run_describe(empty, "d").unwrap().unwrap();
        assert!(description.knobs.is_empty());
        assert_eq!(description.label.as_deref(), Some("L"));

        // A map keyed by id would silently deserialize as zero knobs
        // via the sequence path — refuse it loudly instead.
        let map_shaped = r#"
            function on_event(state, event) return nil end
            function describe()
              return { knobs = { rounds = { type = "integer" } } }
            end
        "#;
        let err = run_describe(map_shaped, "d").unwrap_err();
        assert!(err.contains("must be an array"), "got: {err}");

        // Declaration validity is enforced at evaluation, so authoring
        // mistakes pin to the picker instead of minting dead controls.
        let duplicated = r#"
            function on_event(state, event) return nil end
            function describe()
              return { knobs = {
                { id = "x", type = "string" },
                { id = "x", type = "boolean" },
              } }
            end
        "#;
        let err = run_describe(duplicated, "d").unwrap_err();
        assert!(err.contains("duplicate knob id `x`"), "got: {err}");
    }

    #[test]
    fn describe_absent_nil_and_malformed() {
        let absent = "function on_event(state, event) return nil end";
        assert_eq!(run_describe(absent, "d").unwrap(), None);

        let nil_return = r#"
            function on_event(state, event) return nil end
            function describe() return nil end
        "#;
        assert_eq!(run_describe(nil_return, "d").unwrap(), None);

        let malformed = r#"
            function on_event(state, event) return nil end
            function describe()
              return { knobs = { { id = "x", type = "mystery" } } }
            end
        "#;
        let err = run_describe(malformed, "d").unwrap_err();
        assert!(err.contains("malformed description"), "got: {err}");
    }
}
