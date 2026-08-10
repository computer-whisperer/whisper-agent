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
                let (text, reasoning, tool_calls) = self
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
                        (
                            last_assistant_text(task),
                            last_assistant_reasoning(task),
                            calls,
                        )
                    })
                    .unwrap_or_default();
                ScriptedEvent::AgentCompleted {
                    thread_id: thread_id.to_string(),
                    participant_id: generation.participant_id.to_string(),
                    text,
                    reasoning,
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
        text: &str,
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
                text: text.to_string(),
            },
            pending_io,
        );
    }

    /// Tell a scripted ticking weave that a coordinated thread died
    /// outside the driver's own effects (model/tool I/O failure,
    /// external cancel) — otherwise a coordinating driver waits
    /// forever on a completion that can never arrive. Builtin weaves
    /// have no program to inform; driver-fault failures deliberately
    /// fire nothing (see `ScriptedEvent`). Deaths at load are not
    /// fired here — `load_state` collects them and the weave's first
    /// activation replays them ahead of its triggering event.
    pub(super) fn scripted_thread_failed(
        &mut self,
        thread_id: &str,
        message: &str,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        let Some(weave_id) = self.thread_ticker.get(thread_id).cloned() else {
            return;
        };
        let is_scripted = self.weaves.get(&weave_id).is_some_and(|weave| {
            matches!(
                weave.driver,
                whisper_agent_protocol::ThreadDriverConfig::Scripted { .. }
            )
        });
        if !is_scripted {
            return;
        }
        self.run_scripted_driver(
            &weave_id,
            thread_id,
            ScriptedEvent::ThreadFailed {
                thread_id: thread_id.to_string(),
                message: message.to_string(),
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
        // First activation since load: deliver the deferred death
        // facts collected by `load_state` BEFORE the triggering event,
        // so the driver's world-model catches up with the restart
        // before it decides anything new. Query losses follow the
        // deaths (a driver holding a continue for a lost query must
        // already know which threads are corpses when it reacts).
        if let Some(notices) = self.scripted_load_notices.remove(weave_id) {
            let queue = self
                .scripted_events
                .entry(weave_id.to_string())
                .or_default();
            for (thread_id, message) in notices {
                queue.push_back(ScriptedEvent::ThreadFailed { thread_id, message });
            }
        }
        if let Some(notices) = self.scripted_query_notices.remove(weave_id) {
            // Resolve the lost queries' journal records AT DELIVERY,
            // not at the load scan: a crash between scan and delivery
            // re-derives the notice from the still-Pending record next
            // load (the rebuild-at-load contract the dead-ticked scan
            // has by construction). Salvaged notices whose records
            // already resolved live no-op here.
            if let Some(weave) = self.weaves.get_mut(weave_id) {
                let mut dirty = false;
                for (id, message) in &notices {
                    let lost: Vec<DriverEffectId> = weave
                        .effect_journal
                        .records()
                        .iter()
                        .filter(|record| {
                            record.outcome == crate::runtime::driver::DriverEffectOutcome::Pending
                                && matches!(
                                    &record.effect,
                                    PersistedDriverEffect::QueryKnowledge { query_id, .. }
                                        if query_id == id
                                )
                        })
                        .map(|record| record.id)
                        .collect();
                    for record_id in lost {
                        weave.fail_effect(record_id, message.as_str());
                        dirty = true;
                    }
                }
                if dirty {
                    self.mark_weave_dirty(weave_id);
                }
            }
            let queue = self
                .scripted_events
                .entry(weave_id.to_string())
                .or_default();
            for (id, message) in notices {
                queue.push_back(ScriptedEvent::QueryFailed { id, message });
            }
        }
        if let Some(notices) = self.scripted_dispatch_notices.remove(weave_id) {
            // Same resolve-AT-DELIVERY contract as the query notices
            // above; salvaged notices whose records already resolved
            // live find no Pending match and no-op.
            if let Some(weave) = self.weaves.get_mut(weave_id) {
                let mut dirty = false;
                for notice in &notices {
                    let pending: Vec<DriverEffectId> = weave
                        .effect_journal
                        .records()
                        .iter()
                        .filter(|record| {
                            record.outcome == crate::runtime::driver::DriverEffectOutcome::Pending
                                && matches!(
                                    &record.effect,
                                    PersistedDriverEffect::DispatchCallback {
                                        tool_use_id,
                                        child_thread_id,
                                        ..
                                    } if tool_use_id == &notice.tool_use_id
                                        && child_thread_id == &notice.child_thread_id
                                )
                        })
                        .map(|record| record.id)
                        .collect();
                    for record_id in pending {
                        match &notice.outcome {
                            Ok(_) => weave.complete_effect(record_id),
                            Err(message) => weave.fail_effect(record_id, message.as_str()),
                        }
                        dirty = true;
                    }
                }
                if dirty {
                    self.mark_weave_dirty(weave_id);
                }
            }
            let queue = self
                .scripted_events
                .entry(weave_id.to_string())
                .or_default();
            for notice in notices {
                queue.push_back(notice.into_event());
            }
        }
        self.scripted_events
            .entry(weave_id.to_string())
            .or_default()
            .push_back(event);
        if !self.scripted_active.insert(weave_id.to_string()) {
            return;
        }
        let mut vm_calls = 0usize;
        let mut failure: Option<String> = None;
        'drain: loop {
            // Cap check BEFORE the pop (and only when another event is
            // actually waiting): a cap-tripping event must stay in the
            // queue for the salvage arm below — popping first would
            // drop it unprocessed, and for loss facts and dispatch
            // terminals (whose journal records the preamble already
            // resolved) that drop is permanent.
            if vm_calls >= MAX_VM_CALLS_PER_ACTIVATION
                && self
                    .scripted_events
                    .get(weave_id)
                    .is_some_and(|queue| !queue.is_empty())
            {
                failure = Some(format!(
                    "driver event cascade exceeded {MAX_VM_CALLS_PER_ACTIVATION} VM calls in one activation"
                ));
                break;
            }
            let Some(event) = self
                .scripted_events
                .get_mut(weave_id)
                .and_then(|queue| queue.pop_front())
            else {
                break;
            };
            vm_calls += 1;
            let Some(weave) = self.weaves.get(weave_id) else {
                break;
            };
            let whisper_agent_protocol::ThreadDriverConfig::Scripted { name, config } =
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
            let outcome = match lua::run_event(&source, &name, &data, &event, &config) {
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
        // empty anyway. EXCEPT loss facts — dropping an undelivered
        // `thread_failed` re-opens the waits-forever wedge until the
        // next restart re-detects it, and an undelivered `query_failed`
        // is worse (nothing re-detects a lost query once its journal
        // record is resolved) — both re-stash for the next activation
        // (queue order preserved keeps primary-first).
        if let Some(queue) = self.scripted_events.remove(weave_id) {
            let mut dead_threads: Vec<(String, String)> = Vec::new();
            let mut dead_queries: Vec<(String, String)> = Vec::new();
            let mut dead_dispatches: Vec<ScriptedDispatchNotice> = Vec::new();
            for event in queue {
                match event {
                    ScriptedEvent::ThreadFailed { thread_id, message } => {
                        dead_threads.push((thread_id, message))
                    }
                    ScriptedEvent::QueryFailed { id, message } => dead_queries.push((id, message)),
                    // An undelivered success degrades to a loss notice:
                    // the hits are gone with this queue, but the
                    // correlation must still resolve or a coordinating
                    // driver waits forever. (The journal keeps the
                    // Completed record — the scheduler ran the query;
                    // delivery is what failed.)
                    ScriptedEvent::QueryCompleted { id, .. } => dead_queries.push((
                        id,
                        "query completed but its result was dropped by a driver fault; \
                         re-issue if still needed"
                            .into(),
                    )),
                    // Dispatch terminals re-stash LOSSLESS (both
                    // directions): the payload is small and durable,
                    // so an undelivered completion keeps its result
                    // instead of degrading. Records that already
                    // resolved no-op at the next drain's match scan.
                    ScriptedEvent::DispatchCompleted {
                        thread_id,
                        tool_use_id,
                        child_thread_id,
                        result,
                        usage,
                    } => dead_dispatches.push(ScriptedDispatchNotice {
                        parent_thread_id: thread_id,
                        tool_use_id,
                        child_thread_id,
                        outcome: Ok((result, usage)),
                    }),
                    ScriptedEvent::DispatchFailed {
                        thread_id,
                        tool_use_id,
                        child_thread_id,
                        message,
                    } => dead_dispatches.push(ScriptedDispatchNotice {
                        parent_thread_id: thread_id,
                        tool_use_id,
                        child_thread_id,
                        outcome: Err(message),
                    }),
                    _ => {}
                }
            }
            if !dead_threads.is_empty() {
                self.scripted_load_notices
                    .entry(weave_id.to_string())
                    .or_default()
                    .extend(dead_threads);
            }
            if !dead_queries.is_empty() {
                self.scripted_query_notices
                    .entry(weave_id.to_string())
                    .or_default()
                    .extend(dead_queries);
            }
            if !dead_dispatches.is_empty() {
                self.scripted_dispatch_notices
                    .entry(weave_id.to_string())
                    .or_default()
                    .extend(dead_dispatches);
            }
        }
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
        let whisper_agent_protocol::ThreadDriverConfig::Scripted { name, config } =
            weave.driver.clone()
        else {
            return;
        };
        let pod_id = weave.pod_id.clone();
        let DriverState::Scripted { data, .. } = &weave.driver_state else {
            return;
        };
        let data = data.clone();
        let blocks = match self.load_driver_program(&pod_id, &name) {
            Ok(source) => match lua::run_present(&source, &name, &data, &config) {
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
            ScriptedEffect::RunAgent {
                thread_id,
                participant,
            } => self.scripted_run_agent(weave_id, &thread_id, participant, pending_io),
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
            ScriptedEffect::ContinueCycle { thread_id, nudge } => {
                self.scripted_continue_cycle(weave_id, &thread_id, origin_thread, nudge, pending_io)
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
                backend,
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
                let bindings_request =
                    backend.map(|backend| whisper_agent_protocol::ThreadBindingsRequest {
                        backend: Some(backend),
                        ..Default::default()
                    });
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
                    bindings_request,
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
            ScriptedEffect::SetTitle { thread_id, title } => {
                self.weave_set_title(weave_id, &thread_id, &title)
            }
            ScriptedEffect::CompleteRun { outcome, message } => {
                self.weave_complete_run(weave_id, outcome, message, pending_io)
            }
            ScriptedEffect::QueryKnowledge {
                id,
                query,
                buckets,
                top_k,
                snippet_chars,
                hot_only,
            } => self.weave_query_knowledge(
                weave_id,
                id,
                query,
                buckets,
                top_k,
                snippet_chars,
                hot_only,
                pending_io,
            ),
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
        participant: Option<String>,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) -> Result<(), String> {
        let Some(task) = self.tasks.get(thread_id) else {
            return Err(format!("run_agent: unknown thread `{thread_id}`"));
        };
        let max_turns = task.config.max_turns;
        // Which registered participant speaks. `None` runs the default
        // responder; an explicit id must name a Model member — profile
        // resolution silently falls back to thread defaults for unknown
        // ids, and a wrong-voice driver bug should journal, not
        // silently run.
        let requested = participant
            .map(whisper_agent_protocol::ParticipantId::new)
            .unwrap_or_else(|| task.config.participants.default_responder.clone());
        let member_kind = task
            .config
            .participants
            .member(&requested)
            .map(|member| member.kind);
        let turn = self.scripted_turn_count(weave_id, thread_id) + 1;
        // Journal before admission so every refusal resolves the same
        // record precisely (auditable driver bugs, not silent drops).
        let generation =
            GenerationContext::new(uuid::Uuid::new_v4().to_string(), requested.clone());
        let effect_id = self
            .weaves
            .get_mut(weave_id)
            .expect("caller validated")
            .record_pending_effect(PersistedDriverEffect::RunAgent {
                thread_id: thread_id.to_string(),
                generation: generation.clone(),
                turn,
            });
        self.mark_weave_dirty(weave_id);
        match member_kind {
            Some(whisper_agent_protocol::ThreadParticipantKind::Model) => {}
            Some(_) => {
                let message =
                    format!("run_agent: participant `{requested}` is not a model participant");
                self.fail_weave_effect(weave_id, effect_id, &message);
                return Err(message);
            }
            None => {
                let message = format!(
                    "run_agent: participant `{requested}` is not registered on thread \
                     `{thread_id}`"
                );
                self.fail_weave_effect(weave_id, effect_id, &message);
                return Err(message);
            }
        }
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
                    thread_id: thread_id.to_string(),
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
                thread_id: thread_id.to_string(),
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
                    thread_id: thread_id.to_string(),
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
        nudge: Option<String>,
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
        // The nudge appends WITH the transition (the builtin
        // `submit_server_nudge` shape) — the only mid-cycle injection
        // point. A driver cannot sequence `continue_cycle` +
        // `append_entry` instead: for a non-origin thread the step
        // below runs the next turn re-entrantly before the following
        // effect applies, and the append refuses mid-generation.
        // Pushed only after the transition succeeded so a refused
        // continue never strands the entry.
        let nudge_entry = self.tasks.get_mut(thread_id).and_then(|task| {
            nudge.map(|text| {
                let author = task.config.participants.default_responder.clone();
                task.conversation
                    .push(Message::system_text(text).with_author(author));
                (task.conversation.messages().len() - 1, task.snapshot())
            })
        });
        // Live-transcript parity with the sibling injection paths
        // (`inject_pending_knowledge_nudge`, `weave_append_entry`):
        // subscribers see the injected entry before the reply that
        // references it streams in.
        let nudge_entry = nudge_entry.map(|(index, snapshot)| {
            self.router.broadcast_to_subscribers(
                thread_id,
                whisper_agent_protocol::ServerToClient::ThreadSnapshot {
                    thread_id: thread_id.to_string(),
                    snapshot,
                },
            );
            index
        });
        self.record_completed_weave_effect(
            weave_id,
            PersistedDriverEffect::Continue {
                thread_id: thread_id.to_string(),
                generation,
                nudge_entry,
            },
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
                thread_id: thread_id.to_string(),
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

    // ---------- query_knowledge (step 11 slice 3) ----------

    /// Execute a `query_knowledge` effect — the first async non-thread
    /// effect: validate, journal pending, launch the
    /// embed→search→rerank future. Resolution routes back through
    /// [`Self::apply_scripted_query_completion`]. Static authoring
    /// bugs (empty/duplicate id, empty query, out-of-bounds top_k)
    /// journal Failed AND return `Err`, failing the activation like
    /// any refused effect — the run_agent shape: auditable driver
    /// bugs, not silent drops. Environmental conditions (empty scope,
    /// nothing hot, missing providers) journal Failed and queue a
    /// `query_failed` event instead — those are async-shaped facts a
    /// driver handles, not authoring faults that should kill the
    /// primary.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn weave_query_knowledge(
        &mut self,
        weave_id: &str,
        query_id: String,
        query: String,
        bucket_refs: Vec<String>,
        top_k: Option<u32>,
        snippet_chars: Option<u32>,
        hot_only: Option<bool>,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) -> Result<(), String> {
        use crate::tools::builtin_tools::knowledge_query::{DEFAULT_TOP_K, MAX_TOP_K};
        let Some(weave) = self.weaves.get(weave_id) else {
            return Err(format!("unknown weave `{weave_id}`"));
        };
        let pod_id = weave.pod_id.clone();
        if query_id.is_empty() {
            return self.fault_scripted_query(
                weave_id,
                &query_id,
                &query,
                "query_knowledge: `id` must be non-empty".into(),
            );
        }
        if query.trim().is_empty() {
            return self.fault_scripted_query(
                weave_id,
                &query_id,
                &query,
                format!("query_knowledge `{query_id}`: `query` must be non-empty"),
            );
        }
        let top_k = top_k.unwrap_or(DEFAULT_TOP_K);
        if top_k == 0 || top_k > MAX_TOP_K {
            return self.fault_scripted_query(
                weave_id,
                &query_id,
                &query,
                format!("query_knowledge `{query_id}`: `top_k` must be in 1..={MAX_TOP_K}"),
            );
        }
        let flight_key = (weave_id.to_string(), query_id.clone());
        if self.scripted_queries_in_flight.contains_key(&flight_key) {
            return self.fault_scripted_query(
                weave_id,
                &query_id,
                &query,
                format!("query_knowledge `{query_id}`: a query with this id is already in flight"),
            );
        }
        let snippet_chars = snippet_chars.unwrap_or(500) as usize;
        let hot_only = hot_only.unwrap_or(true);

        // Scope = the weave's pod knowledge ceiling. No participant
        // narrowing — the driver is not a participant.
        let mut in_scope_server: Vec<String> = Vec::new();
        let mut in_scope_pod: Vec<String> = Vec::new();
        for token in self.knowledge_scope_tokens_for_pod(&pod_id) {
            match super::split_knowledge_bucket_token(&token) {
                Some((crate::knowledge::BucketScope::Server, name)) => {
                    in_scope_server.push(name.into())
                }
                Some((crate::knowledge::BucketScope::Pod, name)) => in_scope_pod.push(name.into()),
                None => {}
            }
        }
        let targets = match super::functions::resolve_query_targets(
            &bucket_refs,
            &in_scope_server,
            &in_scope_pod,
            &pod_id,
        ) {
            Ok(targets) if !targets.is_empty() => targets,
            Ok(_) => {
                return self.refuse_scripted_query(
                    weave_id,
                    &query_id,
                    &query,
                    "no knowledge buckets in the pod's scope".into(),
                );
            }
            Err(message) => {
                return self.refuse_scripted_query(weave_id, &query_id, &query, message);
            }
        };
        let Some(reranker) = self
            .rerank_providers
            .values()
            .next()
            .map(|r| r.provider.clone())
        else {
            return self.refuse_scripted_query(
                weave_id,
                &query_id,
                &query,
                "no rerank providers configured".into(),
            );
        };

        let mut sources: Vec<ScriptedQueryBucket> = Vec::new();
        let mut embedder_name: Option<String> = None;
        for tgt in &targets {
            let label = format!("{}:{}", tgt.scope.as_str(), tgt.name);
            let Some(entry) =
                self.bucket_registry
                    .find_entry(tgt.scope, tgt.pod_id.as_deref(), &tgt.name)
            else {
                if hot_only {
                    continue;
                }
                return self.refuse_scripted_query(
                    weave_id,
                    &query_id,
                    &query,
                    format!("bucket `{label}` is in scope but no longer exists in the registry"),
                );
            };
            let Some(active) = entry.active_slot.as_ref() else {
                if hot_only {
                    continue;
                }
                return self.refuse_scripted_query(
                    weave_id,
                    &query_id,
                    &query,
                    format!("bucket `{label}` has no active slot to query"),
                );
            };
            let embedder = entry.config.defaults.embedder.clone();
            let source = if hot_only {
                let hot = match tgt.scope {
                    crate::knowledge::BucketScope::Server => {
                        self.bucket_registry.hot_bucket(&tgt.name)
                    }
                    crate::knowledge::BucketScope::Pod => match tgt.pod_id.as_deref() {
                        Some(pid) => self.bucket_registry.hot_bucket_pod(pid, &tgt.name),
                        None => None,
                    },
                };
                // Cold under hot_only: skipped, not failed — the
                // journal's bucket list records what actually ran.
                let Some(bucket) = hot else { continue };
                ScriptedQueryBucket::Ready { label, bucket }
            } else {
                ScriptedQueryBucket::Load {
                    label,
                    scope: tgt.scope,
                    pod_id: tgt.pod_id.clone(),
                    name: tgt.name.clone(),
                    slot_id: active.slot_id.clone(),
                    serving_mode: format!("{:?}", active.serving.mode).to_lowercase(),
                }
            };
            if embedder_name.is_none() {
                embedder_name = Some(embedder);
            }
            sources.push(source);
        }
        if sources.is_empty() {
            return self.refuse_scripted_query(
                weave_id,
                &query_id,
                &query,
                "no hot buckets among the resolved targets (hot_only)".into(),
            );
        }
        let Some(embedder) = embedder_name
            .as_ref()
            .and_then(|name| self.embedding_providers.get(name))
            .map(|e| e.provider.clone())
        else {
            return self.refuse_scripted_query(
                weave_id,
                &query_id,
                &query,
                format!(
                    "embedder `{}` is not configured",
                    embedder_name.unwrap_or_default()
                ),
            );
        };

        let labels: Vec<String> = sources
            .iter()
            .map(|source| match source {
                ScriptedQueryBucket::Ready { label, .. }
                | ScriptedQueryBucket::Load { label, .. } => label.clone(),
            })
            .collect();
        let effect_id = self
            .weaves
            .get_mut(weave_id)
            .expect("checked above")
            .record_pending_effect(PersistedDriverEffect::QueryKnowledge {
                query_id: query_id.clone(),
                query: query.clone(),
                buckets: labels,
            });
        self.mark_weave_dirty(weave_id);
        self.scripted_queries_in_flight
            .insert(flight_key, effect_id);

        let registry = self.bucket_registry.clone();
        let task_tx = self.bucket_task_sender();
        let sparse_timeout_ms = self.knowledge_config.query.sparse_timeout();
        let weave_id_s = weave_id.to_string();
        pending_io.push(Box::pin(async move {
            let cancel = tokio_util::sync::CancellationToken::new();
            let result = async {
                let mut buckets: Vec<std::sync::Arc<dyn crate::knowledge::Bucket>> =
                    Vec::with_capacity(sources.len());
                for source in sources {
                    match source {
                        ScriptedQueryBucket::Ready { bucket, .. } => buckets.push(bucket),
                        ScriptedQueryBucket::Load {
                            label,
                            scope,
                            pod_id,
                            name,
                            slot_id,
                            serving_mode,
                        } => {
                            let load = super::buckets::load_bucket_with_progress(
                                super::buckets::BucketLoadProgressRequest {
                                    registry: registry.clone(),
                                    bucket_id: name,
                                    pod_id: match scope {
                                        crate::knowledge::BucketScope::Server => None,
                                        crate::knowledge::BucketScope::Pod => pod_id,
                                    },
                                    slot_id,
                                    serving_mode,
                                    task_tx: task_tx.clone(),
                                    requester_conn: None,
                                    correlation_id: None,
                                    emit_cached: false,
                                },
                            )
                            .await;
                            match load {
                                Ok(bucket) => buckets.push(bucket),
                                Err(e) => return Err(format!("load bucket `{label}` failed: {e}")),
                            }
                        }
                    }
                }
                let engine = crate::knowledge::QueryEngine::new(embedder, reranker);
                let params = crate::knowledge::QueryParams {
                    top_k: top_k as usize,
                    sparse_timeout_ms,
                    ..Default::default()
                };
                engine
                    .query(&buckets, &query, &params, &cancel)
                    .await
                    .map_err(|e| e.to_string())
            }
            .await;
            io_dispatch::SchedulerCompletion::ScriptedQuery(io_dispatch::ScriptedQueryCompletion {
                weave_id: weave_id_s,
                query_id,
                effect_id,
                query,
                snippet_chars,
                result,
            })
        }));
        Ok(())
    }

    /// A static `query_knowledge` authoring fault: journal the record
    /// Failed (the run_agent journal-every-refusal shape) and return
    /// `Err`, failing the activation. No `query_failed` event — the
    /// activation's failure is the loud signal, and the faulting
    /// driver would only fault again on the notification.
    fn fault_scripted_query(
        &mut self,
        weave_id: &str,
        query_id: &str,
        query: &str,
        message: String,
    ) -> Result<(), String> {
        if let Some(weave) = self.weaves.get_mut(weave_id) {
            weave.record_failed_effect(
                PersistedDriverEffect::QueryKnowledge {
                    query_id: query_id.to_string(),
                    query: query.to_string(),
                    buckets: Vec::new(),
                },
                message.as_str(),
            );
            self.mark_weave_dirty(weave_id);
        }
        Err(message)
    }

    /// An environmental `query_knowledge` refusal: journal the record
    /// Failed and queue a `query_failed` event onto the weave's drain
    /// (the `thread_derived` delivery shape — same activation, after
    /// the current event's remaining effects).
    fn refuse_scripted_query(
        &mut self,
        weave_id: &str,
        query_id: &str,
        query: &str,
        message: String,
    ) -> Result<(), String> {
        let message = format!("query_knowledge: {message}");
        if let Some(weave) = self.weaves.get_mut(weave_id) {
            weave.record_failed_effect(
                PersistedDriverEffect::QueryKnowledge {
                    query_id: query_id.to_string(),
                    query: query.to_string(),
                    buckets: Vec::new(),
                },
                message.as_str(),
            );
            self.mark_weave_dirty(weave_id);
        }
        self.scripted_events
            .entry(weave_id.to_string())
            .or_default()
            .push_back(ScriptedEvent::QueryFailed {
                id: query_id.to_string(),
                message,
            });
        Ok(())
    }

    /// Resolve a `query_knowledge` completion: settle the journal
    /// record and deliver the fact to the driver. The weave's primary
    /// rides as the delivery origin — a driver fault in the completion
    /// handler fails the primary (the loud-victim containment story),
    /// and the primary's stepping is ours to run after delivery since
    /// no boundary caller owns a step loop here; every other thread the
    /// handler's effects touch is stepped by the effect executors as
    /// usual.
    pub(super) fn apply_scripted_query_completion(
        &mut self,
        completion: io_dispatch::ScriptedQueryCompletion,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        let io_dispatch::ScriptedQueryCompletion {
            weave_id,
            query_id,
            effect_id,
            query,
            snippet_chars,
            result,
        } = completion;
        self.scripted_queries_in_flight
            .remove(&(weave_id.clone(), query_id.clone()));
        let Some(weave) = self.weaves.get_mut(&weave_id) else {
            // The weave retired while the query flew (retention sweep,
            // pod removal); its journal went with it — nothing to
            // resolve, nobody to tell.
            warn!(%weave_id, %query_id, "knowledge query resolved for a gone weave; dropping");
            return;
        };
        let event = match result {
            Ok(hits) => {
                weave.complete_effect(effect_id);
                let hits = hits
                    .into_iter()
                    .map(|hit| lua::ScriptedKnowledgeHit {
                        bucket: hit.bucket_id.to_string(),
                        source_id: hit.source_ref.source_id,
                        chunk_id: hit.chunk_id.to_string(),
                        locator: hit.source_ref.locator.filter(|l| !l.is_empty()),
                        score: hit.rerank_score,
                        snippet: super::head_chars(&hit.chunk_text, snippet_chars),
                    })
                    .collect();
                ScriptedEvent::QueryCompleted {
                    id: query_id,
                    query,
                    hits,
                }
            }
            Err(message) => {
                warn!(%weave_id, error = %message, "scripted knowledge query failed");
                weave.fail_effect(effect_id, message.as_str());
                ScriptedEvent::QueryFailed {
                    id: query_id,
                    message,
                }
            }
        };
        self.mark_weave_dirty(&weave_id);
        let origin = self
            .weaves
            .get(&weave_id)
            .and_then(|weave| weave.primary_thread_id().map(str::to_string))
            .unwrap_or_default();
        self.run_scripted_driver(&weave_id, &origin, event, pending_io);
        // Step every thread this weave ticks (primary first). A driver
        // waiting on this query has PARKED a boundary (the
        // hold-the-continue shape); stepping re-fires it now that the
        // handler has stored the fact, and the driver acts from the
        // re-fired boundary — the contract that stays correct across a
        // restart, where a healed `query_failed` notice and the
        // re-fired boundary share one drain. Threads the driver didn't
        // park step to no-ops.
        let mut ticked: Vec<String> = self
            .weaves
            .get(&weave_id)
            .map(|weave| {
                weave
                    .ticked_thread_ids()
                    .filter(|thread_id| {
                        self.thread_ticker.get(*thread_id).map(String::as_str) == Some(&weave_id)
                    })
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        ticked.sort_by_key(|thread_id| thread_id != &origin);
        for thread_id in ticked {
            self.step_until_blocked(&thread_id, pending_io);
        }
    }

    /// Terminal fan-out for async dispatch callbacks owed to scripted
    /// weave drivers (step 11 slice 4). Fires at every site where
    /// `complete_functions_awaiting_thread` fires; no-ops unless
    /// `child_thread_id` is a watched child in a terminal state (a
    /// defensive non-terminal call leaves the watcher armed). Builds
    /// the terminal outcome from the child's state, resolves the
    /// pending `DispatchCallback` record, and delivers the event.
    pub(super) fn complete_scripted_dispatch_for_child(
        &mut self,
        child_thread_id: &str,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        if !self
            .scripted_dispatch_watchers
            .contains_key(child_thread_id)
        {
            return;
        }
        let outcome = match self.tasks.get(child_thread_id) {
            Some(task) => match task.public_state() {
                whisper_agent_protocol::ThreadStateLabel::Completed => Ok((
                    super::compaction::extract_last_assistant_text(task),
                    scripted_dispatch_usage(task),
                )),
                whisper_agent_protocol::ThreadStateLabel::Failed => Err(task
                    .failure_detail()
                    .unwrap_or_else(|| "child thread failed".to_string())),
                whisper_agent_protocol::ThreadStateLabel::Cancelled => {
                    Err("child thread was cancelled".to_string())
                }
                _ => return,
            },
            None => Err("child thread no longer exists".to_string()),
        };
        let watcher = self
            .scripted_dispatch_watchers
            .remove(child_thread_id)
            .expect("checked above");
        self.deliver_scripted_dispatch_terminal(watcher, outcome, pending_io);
    }

    /// Resolve a dispatch callback's journal record and deliver its
    /// terminal event to the dispatching weave, then step the weave's
    /// ticked threads (the slice-3 async-delivery shape — see
    /// `apply_scripted_query_completion` for the origin/stepping
    /// rationale).
    fn deliver_scripted_dispatch_terminal(
        &mut self,
        watcher: ScriptedDispatchWatcher,
        outcome: Result<(String, lua::ScriptedDispatchUsage), String>,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        let ScriptedDispatchWatcher {
            weave_id,
            parent_thread_id,
            tool_use_id,
            child_thread_id,
            effect_id,
        } = watcher;
        let Some(weave) = self.weaves.get_mut(&weave_id) else {
            // The weave retired while the child ran (retention sweep,
            // pod removal); its journal went with it — nobody to tell.
            warn!(%weave_id, %child_thread_id, "dispatch callback resolved for a gone weave; dropping");
            return;
        };
        match &outcome {
            Ok(_) => weave.complete_effect(effect_id),
            Err(message) => weave.fail_effect(effect_id, message.as_str()),
        }
        self.mark_weave_dirty(&weave_id);
        let event = match outcome {
            Ok((result, usage)) => ScriptedEvent::DispatchCompleted {
                thread_id: parent_thread_id,
                tool_use_id,
                child_thread_id,
                result,
                usage,
            },
            Err(message) => ScriptedEvent::DispatchFailed {
                thread_id: parent_thread_id,
                tool_use_id,
                child_thread_id,
                message,
            },
        };
        let origin = self
            .weaves
            .get(&weave_id)
            .and_then(|weave| weave.primary_thread_id().map(str::to_string))
            .unwrap_or_default();
        self.run_scripted_driver(&weave_id, &origin, event, pending_io);
        let mut ticked: Vec<String> = self
            .weaves
            .get(&weave_id)
            .map(|weave| {
                weave
                    .ticked_thread_ids()
                    .filter(|thread_id| {
                        self.thread_ticker.get(*thread_id).map(String::as_str) == Some(&weave_id)
                    })
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        ticked.sort_by_key(|thread_id| thread_id != &origin);
        for thread_id in ticked {
            self.step_until_blocked(&thread_id, pending_io);
        }
    }
}

/// One async dispatch callback owed to a scripted weave driver, indexed
/// by child thread id in `Scheduler::scripted_dispatch_watchers`. The
/// durable twin is the pending `DispatchCallback` journal record whose
/// id `effect_id` carries.
pub(super) struct ScriptedDispatchWatcher {
    pub weave_id: String,
    pub parent_thread_id: String,
    pub tool_use_id: String,
    pub child_thread_id: String,
    pub effect_id: DriverEffectId,
}

/// A dispatch terminal fact awaiting delivery at the weave's next
/// activation (load heal, or an activation-fault re-stash). Unlike a
/// query loss, the completed payload is durable and re-stashes
/// lossless.
pub(super) struct ScriptedDispatchNotice {
    pub parent_thread_id: String,
    pub tool_use_id: String,
    pub child_thread_id: String,
    pub outcome: Result<(String, lua::ScriptedDispatchUsage), String>,
}

impl ScriptedDispatchNotice {
    fn into_event(self) -> ScriptedEvent {
        match self.outcome {
            Ok((result, usage)) => ScriptedEvent::DispatchCompleted {
                thread_id: self.parent_thread_id,
                tool_use_id: self.tool_use_id,
                child_thread_id: self.child_thread_id,
                result,
                usage,
            },
            Err(message) => ScriptedEvent::DispatchFailed {
                thread_id: self.parent_thread_id,
                tool_use_id: self.tool_use_id,
                child_thread_id: self.child_thread_id,
                message,
            },
        }
    }
}

/// Lifetime usage totals for a dispatched child's terminal event —
/// the same numbers the builtin `<dispatched-thread-notification>`
/// envelope reports.
pub(super) fn scripted_dispatch_usage(task: &Thread) -> lua::ScriptedDispatchUsage {
    lua::ScriptedDispatchUsage {
        total_tokens: task
            .total_usage
            .input_tokens
            .saturating_add(task.total_usage.output_tokens),
        tool_uses: super::dispatch::count_tool_uses(&task.conversation),
        duration_ms: (task.last_active - task.created_at)
            .num_milliseconds()
            .max(0),
    }
}

/// One bucket a scripted query will search, resolved at effect
/// admission. `Ready` is a hot bucket grabbed from cache synchronously
/// (the `hot_only` path); `Load` carries the coordinates for a
/// cache-or-disk load inside the future (the `knowledge_query` tool's
/// shape — a first load pays the slot-load cost).
enum ScriptedQueryBucket {
    Ready {
        label: String,
        bucket: std::sync::Arc<dyn crate::knowledge::Bucket>,
    },
    Load {
        label: String,
        scope: crate::knowledge::BucketScope,
        pod_id: Option<String>,
        name: String,
        slot_id: String,
        serving_mode: String,
    },
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

/// Thinking content of the last assistant message, for
/// `agent_completed.reasoning`. Mirrors `last_assistant_text` (and the
/// builtin autoquery's source extraction): visible Thinking blocks
/// only — redacted thinking has no text to carry.
fn last_assistant_reasoning(task: &Thread) -> String {
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
                    ContentBlock::Thinking { thinking, .. } => Some(thinking.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}
