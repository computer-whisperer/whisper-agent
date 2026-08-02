//! Scripted (Lua) driver execution.
//!
//! A scripted driver is a Lua program at `<pod>/drivers/<name>.lua`
//! defining one global function:
//!
//! ```lua
//! function on_event(state, event)
//!   -- state: the driver's persisted table (empty {} on first event)
//!   -- event: { kind = "turn_start" | "agent_completed" | ..., ... }
//!   return { effects = { { kind = "run_agent", thread_id = ... } },
//!            state = state }
//! end
//! ```
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
//! stdlib only, with the base-library escape hatches stripped on top
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
#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
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
    /// content; `tool_calls` the tools it requested (empty when none).
    AgentCompleted {
        thread_id: String,
        participant_id: String,
        text: String,
        tool_calls: Vec<ScriptedToolCall>,
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
    /// model/tool I/O failure or an external cancel. Without this the
    /// driver waits forever on an `agent_completed` that can never
    /// arrive (a Failed thread refuses `run_agent`; only external
    /// input heals it). Deliberately NOT fired for driver-fault
    /// failures (an erroring program would just error again on the
    /// notification) or during load-path healing (drivers don't run
    /// at load).
    ThreadFailed { thread_id: String, message: String },
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
    /// Take another model turn after tools completed.
    ContinueCycle {
        thread_id: String,
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
    let returned: mlua::Value = handler
        .call((state_value, event_value))
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

/// Evaluate the program's optional `present(state) -> blocks` entry
/// point (migration step 7b): a pure function of persisted driver state
/// composing the weave's presentation structure — replayable by
/// construction, since it can see nothing else.
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
    let returned: mlua::Value = presenter
        .call(state_value)
        .map_err(|e| format!("driver `{chunk_name}` present: {e}"))?;
    if returned.is_nil() {
        return Ok(None);
    }
    let blocks: Vec<whisper_agent_protocol::weave::PresentationBlock> = lua
        .from_value(returned)
        .map_err(|e| format!("driver `{chunk_name}` returned malformed presentation: {e}"))?;
    Ok(Some(blocks))
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
              } }
            end
        "#;
        let out = run_event(src_ok, "vocab", &json!({}), &event_turn_start("t", 1)).unwrap();
        assert_eq!(out.effects.len(), 7);
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
}
