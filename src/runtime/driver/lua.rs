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
//! Each event runs in a fresh, sandboxed VM: no io/os/require (only
//! table/string/math/utf8 stdlib), a memory ceiling, and an instruction
//! budget. Driver programs are pure policy — every side effect goes
//! through the returned effect list, admitted and journaled by the
//! scheduler.

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
    /// External input was accepted into a coordinated thread.
    InputAccepted { thread_id: String },
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
    RunAgent {
        thread_id: String,
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
    let lua = Lua::new_with(
        mlua::StdLib::TABLE | mlua::StdLib::STRING | mlua::StdLib::MATH | mlua::StdLib::UTF8,
        mlua::LuaOptions::default(),
    )
    .map_err(|e| format!("lua init: {e}"))?;
    lua.set_memory_limit(MEMORY_LIMIT_BYTES)
        .map_err(|e| format!("lua memory limit: {e}"))?;
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
                thread_id: "t-1".into()
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
    fn sandbox_has_no_io_os_or_require() {
        let src = r#"
            function on_event(state, event)
              return { effects = {}, state = {
                has_io = io ~= nil,
                has_os = os ~= nil,
                has_require = require ~= nil,
              } }
            end
        "#;
        let out = run_event(src, "sandbox", &json!({}), &event_turn_start("t", 1)).unwrap();
        assert_eq!(
            out.state,
            json!({"has_io": false, "has_os": false, "has_require": false})
        );
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
}
