//! Scripted-driver execution: boundary events in, effect lists out.
//!
//! The scheduler-side half of the Lua driver contract
//! (`crate::runtime::driver::lua`). Thread boundaries and accepted
//! input become [`ScriptedEvent`]s; the driver's returned effects are
//! admitted, journaled on the weave, and applied through the same
//! mechanical thread transitions and cross-thread executors the
//! builtin path uses. The scheduler stays the single writer — the
//! script computes policy and nothing else.
//!
//! Parking: when the driver returns no effect that moves the boundary
//! thread, `apply_scripted_boundary` reports "not advanced" and the
//! step loop breaks instead of spinning. The parked boundary fires
//! again the next time the thread is stepped, so handlers are written
//! idempotent (see the contract notes in `driver/lua.rs`).

use tracing::warn;

use super::{Scheduler, SchedulerFuture, io_dispatch};
use crate::runtime::driver::lua::{self, ScriptedEffect, ScriptedEvent, ScriptedToolCall};
use crate::runtime::driver::{
    DriverEffectId, DriverFinishReason, DriverState, EntryRef, PersistedDriverEffect,
    ThreadRelationship,
};
use crate::runtime::thread::{Thread, ThreadBoundary, ThreadInternalState, ToolDecision};
use futures::stream::FuturesUnordered;
use whisper_agent_protocol::{
    AllowMap, ContentBlock, GenerationContext, Message, Role, SystemPromptChoice,
    ThreadConfigOverride,
};

/// VM activations allowed per external trigger (a boundary or accepted
/// input). Each derive re-enters the driver via `thread_derived`, so
/// legitimate cascades are short; a driver that ping-pongs effects
/// forever is a bug this ceiling converts into a clean failure.
const MAX_VM_CALLS_PER_ACTIVATION: usize = 16;

impl Scheduler {
    /// Route one thread boundary through the weave's scripted driver.
    /// Returns whether the boundary thread's state advanced — `false`
    /// parks it (the step loop breaks; the boundary re-fires on the
    /// next step).
    pub(super) fn apply_scripted_boundary(
        &mut self,
        weave_id: &str,
        thread_id: &str,
        boundary: ThreadBoundary,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) -> bool {
        let before = self
            .tasks
            .get(thread_id)
            .map(|task| std::mem::discriminant(&task.internal));
        let event = match boundary {
            ThreadBoundary::TurnStart => ScriptedEvent::TurnStart {
                thread_id: thread_id.to_string(),
                turn: self.scripted_turn_count(weave_id, thread_id) + 1,
            },
            ThreadBoundary::AgentCompleted {
                generation,
                effect_id,
                has_tool_calls: _,
            } => {
                if let Some(weave) = self.weaves.get_mut(weave_id) {
                    weave.complete_effect(effect_id);
                    self.mark_weave_dirty(weave_id);
                }
                let (text, tool_calls) = self
                    .tasks
                    .get(thread_id)
                    .map(|task| {
                        let calls = match &task.internal {
                            ThreadInternalState::AgentBoundary {
                                pending_tool_uses, ..
                            } => pending_tool_uses
                                .iter()
                                .map(|req| ScriptedToolCall {
                                    tool_use_id: req.tool_use_id.clone(),
                                    name: req.name.clone(),
                                    args: req.input.clone(),
                                })
                                .collect(),
                            _ => Vec::new(),
                        };
                        (last_assistant_text(task), calls)
                    })
                    .unwrap_or_default();
                ScriptedEvent::AgentCompleted {
                    thread_id: thread_id.to_string(),
                    participant_id: generation.participant_id.to_string(),
                    text,
                    tool_calls,
                }
            }
            ThreadBoundary::ToolsCompleted {
                generation,
                effect_id,
            } => {
                if let Some(weave) = self.weaves.get_mut(weave_id) {
                    weave.complete_effect(effect_id);
                    self.mark_weave_dirty(weave_id);
                }
                ScriptedEvent::ToolsCompleted {
                    thread_id: thread_id.to_string(),
                    participant_id: generation.participant_id.to_string(),
                }
            }
        };
        self.run_scripted_driver(weave_id, thread_id, event, pending_io);
        let after = self
            .tasks
            .get(thread_id)
            .map(|task| std::mem::discriminant(&task.internal));
        before != after
    }

    /// Deliver accepted input to a scripted weave: reset the thread's
    /// mechanical turn counter and hand the driver an `input_accepted`
    /// event.
    pub(super) fn scripted_input_accepted(
        &mut self,
        weave_id: &str,
        thread_id: &str,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        if let Some(weave) = self.weaves.get_mut(weave_id) {
            if let DriverState::Scripted { turns, .. } = &mut weave.driver_state {
                turns.remove(thread_id);
            }
            weave.touch();
            self.mark_weave_dirty(weave_id);
        }
        self.run_scripted_driver(
            weave_id,
            thread_id,
            ScriptedEvent::InputAccepted {
                thread_id: thread_id.to_string(),
            },
            pending_io,
        );
    }

    /// Feed events through the driver program until the queue drains,
    /// executing each returned effect in order. Any driver error —
    /// unreadable program, VM failure, malformed outcome, refused
    /// effect — fails the origin thread with the message; refusals of
    /// journal-bearing effects (run_agent, dispatch/resolve_tools, the
    /// cross-thread executors) additionally resolve their journal
    /// record as Failed.
    ///
    /// One activation per weave at a time: an event arriving while the
    /// weave is already draining (a step tail hook feeding input back
    /// in) is queued for the active drain, attributed to that
    /// activation's origin thread.
    fn run_scripted_driver(
        &mut self,
        weave_id: &str,
        origin_thread: &str,
        event: ScriptedEvent,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        self.scripted_events
            .entry(weave_id.to_string())
            .or_default()
            .push_back(event);
        if !self.scripted_active.insert(weave_id.to_string()) {
            return;
        }
        let mut vm_calls = 0usize;
        let mut failure: Option<String> = None;
        'drain: while let Some(event) = self
            .scripted_events
            .get_mut(weave_id)
            .and_then(|queue| queue.pop_front())
        {
            vm_calls += 1;
            if vm_calls > MAX_VM_CALLS_PER_ACTIVATION {
                failure = Some(format!(
                    "driver event cascade exceeded {MAX_VM_CALLS_PER_ACTIVATION} VM calls in one activation"
                ));
                break;
            }
            let Some(weave) = self.weaves.get(weave_id) else {
                break;
            };
            let whisper_agent_protocol::ThreadDriverConfig::Scripted { name } =
                weave.driver.clone()
            else {
                failure = Some(
                    "non-scripted weave routed to the scripted executor (scheduler bug)".into(),
                );
                break;
            };
            let pod_id = weave.pod_id.clone();
            let data = match &weave.driver_state {
                DriverState::Scripted { data, .. } => data.clone(),
                // Config/state mismatch (hand-edited JSON): heal to a
                // fresh script state rather than failing the load.
                _ => serde_json::Value::Object(Default::default()),
            };
            let source = match self.load_driver_program(&pod_id, &name) {
                Ok(source) => source,
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            };
            let hash = lua::program_hash(&source);
            {
                let weave = self.weaves.get_mut(weave_id).expect("present above");
                if weave.driver_program_hash.as_deref() != Some(hash.as_str()) {
                    weave.driver_program_hash = Some(hash);
                }
                self.mark_weave_dirty(weave_id);
            }
            let outcome = match lua::run_event(&source, &name, &data, &event) {
                Ok(outcome) => outcome,
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            };
            {
                let weave = self.weaves.get_mut(weave_id).expect("present above");
                match &mut weave.driver_state {
                    DriverState::Scripted { data, .. } => *data = outcome.state,
                    other => {
                        *other = DriverState::Scripted {
                            data: outcome.state,
                            turns: Default::default(),
                        };
                    }
                }
                weave.touch();
                self.mark_weave_dirty(weave_id);
            }
            for effect in outcome.effects {
                if let Err(error) =
                    self.apply_scripted_effect(weave_id, origin_thread, effect, pending_io)
                {
                    failure = Some(error);
                    break 'drain;
                }
            }
        }
        self.scripted_active.remove(weave_id);
        // Queued leftovers are dropped: after a failure they would run
        // against a failed origin, and a clean drain leaves the queue
        // empty anyway.
        self.scripted_events.remove(weave_id);
        if let Some(message) = failure {
            self.fail_scripted(weave_id, origin_thread, &message);
        }
        self.refresh_scripted_presentation(weave_id);
    }

    /// Recompute the weave's presentation cache from the program's
    /// `present(state)` after an activation, then re-send the wire
    /// snapshot to weave subscribers (step 7b). Presentation errors
    /// degrade the display to the degenerate fallback — they never fail
    /// the weave's coordinated work.
    fn refresh_scripted_presentation(&mut self, weave_id: &str) {
        let Some(weave) = self.weaves.get(weave_id) else {
            return;
        };
        let whisper_agent_protocol::ThreadDriverConfig::Scripted { name } = weave.driver.clone()
        else {
            return;
        };
        let pod_id = weave.pod_id.clone();
        let DriverState::Scripted { data, .. } = &weave.driver_state else {
            return;
        };
        let data = data.clone();
        let blocks = match self.load_driver_program(&pod_id, &name) {
            Ok(source) => match lua::run_present(&source, &name, &data) {
                Ok(Some(blocks)) => blocks,
                Ok(None) => Vec::new(),
                Err(error) => {
                    warn!(
                        weave_id,
                        error, "driver present() failed; degrading display"
                    );
                    Vec::new()
                }
            },
            // An unreadable program already failed the activation
            // itself; for the display it just means degenerate.
            Err(_) => Vec::new(),
        };
        let weave = self.weaves.get_mut(weave_id).expect("present above");
        let validated = weave.validate_presentation(blocks);
        if weave.presentation != validated {
            weave.presentation = validated;
            self.mark_weave_dirty(weave_id);
        }
        self.notify_weave_subscribers(weave_id);
    }

    fn apply_scripted_effect(
        &mut self,
        weave_id: &str,
        origin_thread: &str,
        effect: ScriptedEffect,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) -> Result<(), String> {
        match effect {
            ScriptedEffect::RunAgent { thread_id } => {
                self.scripted_run_agent(weave_id, &thread_id, pending_io)
            }
            ScriptedEffect::DispatchTools { thread_id } => {
                self.scripted_dispatch_tools(weave_id, &thread_id, origin_thread, None, pending_io)
            }
            ScriptedEffect::ResolveTools {
                thread_id,
                decisions,
            } => {
                let decisions: Vec<ToolDecision> = decisions
                    .into_iter()
                    .map(|d| ToolDecision {
                        tool_use_id: d.tool_use_id,
                        allow: d.allow,
                        message: d.message,
                    })
                    .collect();
                self.scripted_dispatch_tools(
                    weave_id,
                    &thread_id,
                    origin_thread,
                    Some(decisions),
                    pending_io,
                )
            }
            ScriptedEffect::ContinueCycle { thread_id } => {
                self.scripted_continue_cycle(weave_id, &thread_id, origin_thread, pending_io)
            }
            ScriptedEffect::FinishCycle { thread_id } => {
                self.scripted_finish_cycle(weave_id, &thread_id, origin_thread, pending_io)
            }
            ScriptedEffect::AppendEntry {
                thread_id,
                author,
                text,
                source_thread_id,
                source_entry_index,
            } => {
                let message = Message::user_text(text).with_author(author);
                let source = source_thread_id.map(|tid| EntryRef {
                    thread_id: tid,
                    entry_index: source_entry_index,
                });
                self.weave_append_entry(weave_id, &thread_id, message, source)
                    .map(|_| ())
            }
            ScriptedEffect::DeriveThread {
                relationship,
                system_prompt,
                model,
                disable_tools,
                max_turns,
                seed,
                source_thread_id,
            } => {
                let config_override = ThreadConfigOverride {
                    model,
                    max_turns,
                    system_prompt: system_prompt.map(|text| SystemPromptChoice::Text { text }),
                    tools: disable_tools.then(AllowMap::deny_all),
                    ..Default::default()
                };
                let seed_messages: Vec<Message> = seed
                    .into_iter()
                    .map(|entry| Message::user_text(entry.text).with_author(entry.author))
                    .collect();
                let relationship_meta = ThreadRelationship {
                    kind: relationship.clone(),
                    source: source_thread_id.map(|tid| EntryRef {
                        thread_id: tid,
                        entry_index: None,
                    }),
                };
                let new_id = self.weave_derive_thread(
                    weave_id,
                    Some(config_override),
                    None,
                    seed_messages,
                    relationship_meta,
                    None,
                    pending_io,
                )?;
                self.scripted_events
                    .entry(weave_id.to_string())
                    .or_default()
                    .push_back(ScriptedEvent::ThreadDerived {
                        thread_id: new_id,
                        relationship,
                    });
                Ok(())
            }
            ScriptedEffect::AdvanceHead { thread_id } => {
                self.weave_advance_head(weave_id, &thread_id)
            }
            ScriptedEffect::AdoptTicker { thread_id } => {
                self.weave_adopt_ticker(weave_id, &thread_id)
            }
            ScriptedEffect::ReleaseTicker { thread_id } => {
                self.weave_release_ticker(weave_id, &thread_id)
            }
        }
    }

    /// Run one model turn on a ticked thread. Valid when the thread is
    /// parked at a turn boundary, idle, or completed (driver-run turns
    /// on seeded/reused threads). Enforces the thread's `max_turns`
    /// ceiling mechanically — the counter lives outside script state.
    fn scripted_run_agent(
        &mut self,
        weave_id: &str,
        thread_id: &str,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) -> Result<(), String> {
        let Some(task) = self.tasks.get(thread_id) else {
            return Err(format!("run_agent: unknown thread `{thread_id}`"));
        };
        let max_turns = task.config.max_turns;
        let participant = task.config.participants.default_responder.clone();
        let turn = self.scripted_turn_count(weave_id, thread_id) + 1;
        // Journal before admission so every refusal resolves the same
        // record precisely (auditable driver bugs, not silent drops).
        let generation = GenerationContext::new(uuid::Uuid::new_v4().to_string(), participant);
        let effect_id = self
            .weaves
            .get_mut(weave_id)
            .expect("caller validated")
            .record_pending_effect(PersistedDriverEffect::RunAgent {
                generation: generation.clone(),
                turn,
            });
        self.mark_weave_dirty(weave_id);
        if self.thread_ticker.get(thread_id).map(String::as_str) != Some(weave_id) {
            let message = format!("run_agent: weave does not tick thread `{thread_id}`");
            self.fail_weave_effect(weave_id, effect_id, &message);
            return Err(message);
        }
        if turn > max_turns {
            // Builtin parity: the turn limit finishes the cycle, it
            // never fails the thread. The refused RunAgent resolves
            // Failed; a completed Finish{TurnLimit} records the
            // decision; the counter resets with the cycle.
            self.fail_weave_effect(weave_id, effect_id, "turn limit reached");
            self.record_completed_weave_effect(
                weave_id,
                PersistedDriverEffect::Finish {
                    generation: None,
                    reason: DriverFinishReason::TurnLimit,
                },
            );
            warn!(
                max_turns,
                thread_id = %thread_id,
                "scripted driver hit the cycle turn limit"
            );
            let mut events = Vec::new();
            if let Some(task) = self.tasks.get_mut(thread_id) {
                task.finish_cycle(&mut events);
            }
            self.router.dispatch_events(thread_id, events);
            self.reset_scripted_turns(weave_id, thread_id);
            self.mark_dirty(thread_id);
            return Ok(());
        }
        if let Some(weave) = self.weaves.get_mut(weave_id)
            && let DriverState::Scripted { turns, .. } = &mut weave.driver_state
        {
            turns.insert(thread_id.to_string(), turn);
        }
        let op_id = self.next_op_id;
        self.next_op_id += 1;
        let mut events = Vec::new();
        let request = self.tasks.get_mut(thread_id).and_then(|task| {
            task.begin_model_call(op_id, generation, effect_id, turn, &mut events)
        });
        self.router.dispatch_events(thread_id, events);
        self.mark_dirty(thread_id);
        match request {
            Some(request) => {
                let fut = io_dispatch::build_io_future(self, thread_id.to_string(), request);
                pending_io.push(fut);
                Ok(())
            }
            None => {
                self.fail_weave_effect(weave_id, effect_id, "thread not at a runnable state");
                Err(format!(
                    "run_agent: thread `{thread_id}` is not at a runnable state"
                ))
            }
        }
    }

    /// Dispatch (or per-tool resolve) the tool requests parked at a
    /// thread's agent boundary. `decisions: None` admits everything —
    /// the plain `dispatch_tools` effect; `Some` is `resolve_tools`.
    fn scripted_dispatch_tools(
        &mut self,
        weave_id: &str,
        thread_id: &str,
        origin_thread: &str,
        decisions: Option<Vec<ToolDecision>>,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) -> Result<(), String> {
        if self.thread_ticker.get(thread_id).map(String::as_str) != Some(weave_id) {
            return Err(format!(
                "dispatch_tools: weave does not tick thread `{thread_id}`"
            ));
        }
        let Some(task) = self.tasks.get(thread_id) else {
            return Err(format!("dispatch_tools: unknown thread `{thread_id}`"));
        };
        let (generation, requested): (GenerationContext, Vec<String>) = match &task.internal {
            ThreadInternalState::AgentBoundary {
                generation,
                pending_tool_uses,
                ..
            } => (
                generation.clone(),
                pending_tool_uses
                    .iter()
                    .map(|req| req.tool_use_id.clone())
                    .collect(),
            ),
            _ => {
                return Err(format!(
                    "dispatch_tools: thread `{thread_id}` is not at an agent boundary"
                ));
            }
        };
        let effect = match &decisions {
            None => PersistedDriverEffect::DispatchTools {
                generation: generation.clone(),
                tool_use_ids: requested.clone(),
            },
            Some(decisions) => {
                let approved: Vec<String> = requested
                    .iter()
                    .filter(|id| decisions.iter().any(|d| &d.tool_use_id == *id && d.allow))
                    .cloned()
                    .collect();
                let denied: Vec<String> = requested
                    .iter()
                    .filter(|id| !approved.contains(id))
                    .cloned()
                    .collect();
                PersistedDriverEffect::ResolveTools {
                    generation: generation.clone(),
                    approved,
                    denied,
                }
            }
        };
        let effect_id = {
            let weave = self.weaves.get_mut(weave_id).expect("caller validated");
            weave.record_pending_effect(effect)
        };
        self.mark_weave_dirty(weave_id);
        let mut events = Vec::new();
        let applied = {
            let task = self.tasks.get_mut(thread_id).expect("checked above");
            match &decisions {
                None => task.begin_tool_dispatch(effect_id),
                Some(decisions) => task.resolve_tool_dispatch(effect_id, decisions, &mut events),
            }
        };
        self.router.dispatch_events(thread_id, events);
        self.mark_dirty(thread_id);
        if !applied {
            self.fail_weave_effect(weave_id, effect_id, "thread left the agent boundary");
            return Err(format!(
                "dispatch_tools: thread `{thread_id}` left the agent boundary"
            ));
        }
        if thread_id != origin_thread {
            self.step_until_blocked(thread_id, pending_io);
        }
        Ok(())
    }

    fn scripted_continue_cycle(
        &mut self,
        weave_id: &str,
        thread_id: &str,
        origin_thread: &str,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) -> Result<(), String> {
        if self.thread_ticker.get(thread_id).map(String::as_str) != Some(weave_id) {
            return Err(format!(
                "continue_cycle: weave does not tick thread `{thread_id}`"
            ));
        }
        let generation = match self.tasks.get(thread_id).map(|task| &task.internal) {
            Some(ThreadInternalState::ToolsBoundary { generation, .. }) => generation.clone(),
            _ => {
                return Err(format!(
                    "continue_cycle: thread `{thread_id}` is not at a tools boundary"
                ));
            }
        };
        let applied = self
            .tasks
            .get_mut(thread_id)
            .map(|task| task.continue_cycle())
            .unwrap_or(false);
        if !applied {
            return Err(format!(
                "continue_cycle: thread `{thread_id}` did not accept the transition"
            ));
        }
        self.record_completed_weave_effect(
            weave_id,
            PersistedDriverEffect::Continue { generation },
        );
        self.mark_weave_dirty(weave_id);
        self.mark_dirty(thread_id);
        if thread_id != origin_thread {
            self.step_until_blocked(thread_id, pending_io);
        }
        Ok(())
    }

    fn scripted_finish_cycle(
        &mut self,
        weave_id: &str,
        thread_id: &str,
        origin_thread: &str,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) -> Result<(), String> {
        if self.thread_ticker.get(thread_id).map(String::as_str) != Some(weave_id) {
            return Err(format!(
                "finish_cycle: weave does not tick thread `{thread_id}`"
            ));
        }
        let mut events = Vec::new();
        let applied = self
            .tasks
            .get_mut(thread_id)
            .map(|task| task.finish_cycle(&mut events))
            .unwrap_or(false);
        self.router.dispatch_events(thread_id, events);
        if !applied {
            return Err(format!(
                "finish_cycle: thread `{thread_id}` is not at a finishable boundary"
            ));
        }
        self.record_completed_weave_effect(
            weave_id,
            PersistedDriverEffect::Finish {
                generation: None,
                reason: DriverFinishReason::DriverChoice,
            },
        );
        self.reset_scripted_turns(weave_id, thread_id);
        self.mark_weave_dirty(weave_id);
        self.mark_dirty(thread_id);
        if thread_id != origin_thread {
            // Run the terminal hooks (behavior outcomes, dispatch
            // fan-out, auto-compaction gates) the origin thread's own
            // step loop would run for itself.
            self.step_until_blocked(thread_id, pending_io);
        }
        Ok(())
    }

    /// Read `<pod>/drivers/<name>.lua`. Synchronous like the pod
    /// system-prompt read at thread creation; driver programs are small
    /// policy scripts.
    pub(super) fn load_driver_program(&self, pod_id: &str, name: &str) -> Result<String, String> {
        if name.is_empty() || name.contains('/') || name.contains('\\') || name.contains("..") {
            return Err(format!("invalid driver program name `{name}`"));
        }
        let Some(pod) = self.pods.get(pod_id) else {
            return Err(format!("unknown pod `{pod_id}`"));
        };
        let path = pod.dir.join("drivers").join(format!("{name}.lua"));
        std::fs::read_to_string(&path)
            .map_err(|error| format!("driver program `{name}` unreadable: {error}"))
    }

    fn scripted_turn_count(&self, weave_id: &str, thread_id: &str) -> u32 {
        match self.weaves.get(weave_id).map(|weave| &weave.driver_state) {
            Some(DriverState::Scripted { turns, .. }) => turns.get(thread_id).copied().unwrap_or(0),
            _ => 0,
        }
    }

    /// A finished cycle resets the thread's mechanical turn counter, so
    /// reused threads (a checker answering many questions) budget per
    /// cycle, not per lifetime.
    fn reset_scripted_turns(&mut self, weave_id: &str, thread_id: &str) {
        if let Some(weave) = self.weaves.get_mut(weave_id)
            && let DriverState::Scripted { turns, .. } = &mut weave.driver_state
            && turns.remove(thread_id).is_some()
        {
            self.mark_weave_dirty(weave_id);
        }
    }

    /// Is this thread's ticker a scripted weave? Compaction machinery
    /// checks this: builtin-style compaction (COMPACTING bit, summary
    /// prompt as input, continuation thread) is incoherent for
    /// scripted-driven threads — those compact via weave machinery when
    /// migration step 8 lands.
    pub(super) fn has_scripted_ticker(&self, thread_id: &str) -> bool {
        self.thread_ticker
            .get(thread_id)
            .and_then(|weave_id| self.weaves.get(weave_id))
            .is_some_and(|weave| {
                matches!(
                    weave.driver,
                    whisper_agent_protocol::ThreadDriverConfig::Scripted { .. }
                )
            })
    }

    fn fail_weave_effect(&mut self, weave_id: &str, effect_id: DriverEffectId, message: &str) {
        if let Some(weave) = self.weaves.get_mut(weave_id) {
            weave.fail_effect(effect_id, message);
            self.mark_weave_dirty(weave_id);
        }
    }

    fn fail_scripted(&mut self, weave_id: &str, origin_thread: &str, message: &str) {
        warn!(
            weave_id = %weave_id,
            origin_thread = %origin_thread,
            %message,
            "scripted driver failed"
        );
        self.fail_thread_at_boundary(origin_thread, "driver", message);
    }
}

/// Text of the most recent assistant entry — what the driver reads as
/// "the response" at an agent boundary (the checker's verdict, the
/// primary's prose).
fn last_assistant_text(task: &Thread) -> String {
    task.conversation
        .messages()
        .iter()
        .rev()
        .find(|message| message.role == Role::Assistant)
        .map(|message| {
            message
                .content
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}
