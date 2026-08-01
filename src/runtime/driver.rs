//! Driver policy — durable thread events in, requested effects out.
//!
//! The scheduler remains responsible for executing effects and integrating I/O.
//! This module owns only conversational policy: which participant runs next,
//! whether a model response enters the tool loop, and when a cycle finishes.

use serde::{Deserialize, Serialize};
use whisper_agent_protocol::{
    GenerationContext, ParticipantId, ThreadDriverConfig, ThreadParticipants,
};

pub type DriverEffectId = u64;

/// Persisted mutable state owned by the selected driver. The enum is tagged so
/// a future scripted driver can carry a different state shape without making
/// existing thread JSON ambiguous.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DriverState {
    BuiltinSingleAgentChat {
        #[serde(default)]
        cycles_started: u64,
        #[serde(default)]
        turns_in_cycle: u32,
    },
}

impl Default for DriverState {
    fn default() -> Self {
        Self::BuiltinSingleAgentChat {
            cycles_started: 0,
            turns_in_cycle: 0,
        }
    }
}

impl DriverState {
    /// Create fresh mutable state for a normalized driver definition. Keeping
    /// this constructor config-aware prevents thread creation, forks, and
    /// derived threads from silently inheriting another driver's state shape
    /// when additional driver variants land.
    pub fn for_config(config: &ThreadDriverConfig) -> Self {
        match config {
            ThreadDriverConfig::BuiltinSingleAgentChat => Self::default(),
        }
    }
}

/// Effect vocabulary currently consumed by `Thread`. These are requests, not
/// side effects themselves; provider/tool handles remain scheduler-owned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DriverEffect {
    RunAgent {
        participant_id: ParticipantId,
        turn: u32,
    },
    DispatchTools,
    Continue,
    Finish,
}

/// Durable, executor-enriched form of a driver request. The pure policy effect
/// selects behavior; this form captures the run/tool generation allocated by
/// `Thread` so an in-flight request can be correlated after serialization.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PersistedDriverEffect {
    RunAgent {
        generation: GenerationContext,
        turn: u32,
    },
    DispatchTools {
        generation: GenerationContext,
        tool_use_ids: Vec<String>,
    },
    Continue {
        generation: GenerationContext,
    },
    Finish {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        generation: Option<GenerationContext>,
        reason: DriverFinishReason,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DriverFinishReason {
    AgentCompleted,
    ToolsCompleted,
    TurnLimit,
}

/// Terminal state is kept on the record rather than deleting it. That makes a
/// restart/cancel distinguishable from both a completed effect and one that was
/// never requested.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum DriverEffectOutcome {
    Pending,
    Completed,
    Failed { message: String },
    Interrupted { reason: String },
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DriverEffectRecord {
    pub id: DriverEffectId,
    pub effect: PersistedDriverEffect,
    pub outcome: DriverEffectOutcome,
}

/// Append-only effect history for one thread. `next_id` is persisted so record
/// identifiers remain monotonic even if a future retention policy prunes old
/// terminal entries.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DriverEffectJournal {
    #[serde(default = "first_effect_id")]
    next_id: DriverEffectId,
    #[serde(default)]
    records: Vec<DriverEffectRecord>,
}

const fn first_effect_id() -> DriverEffectId {
    1
}

impl Default for DriverEffectJournal {
    fn default() -> Self {
        Self {
            next_id: first_effect_id(),
            records: Vec::new(),
        }
    }
}

impl DriverEffectJournal {
    pub fn records(&self) -> &[DriverEffectRecord] {
        &self.records
    }

    pub fn record(&mut self, effect: PersistedDriverEffect) -> DriverEffectId {
        let after_existing = self
            .records
            .iter()
            .map(|record| record.id)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        let id = self.next_id.max(after_existing).max(first_effect_id());
        self.next_id = id.saturating_add(1);
        self.records.push(DriverEffectRecord {
            id,
            effect,
            outcome: DriverEffectOutcome::Pending,
        });
        id
    }

    pub fn complete(&mut self, id: DriverEffectId) -> bool {
        self.resolve(id, DriverEffectOutcome::Completed)
    }

    pub fn fail(&mut self, id: DriverEffectId, message: impl Into<String>) -> bool {
        self.resolve(
            id,
            DriverEffectOutcome::Failed {
                message: message.into(),
            },
        )
    }

    pub fn interrupt(&mut self, id: DriverEffectId, reason: impl Into<String>) -> bool {
        self.resolve(
            id,
            DriverEffectOutcome::Interrupted {
                reason: reason.into(),
            },
        )
    }

    pub fn fail_pending(&mut self, message: impl Into<String>) {
        let message = message.into();
        for record in &mut self.records {
            if record.outcome == DriverEffectOutcome::Pending {
                record.outcome = DriverEffectOutcome::Failed {
                    message: message.clone(),
                };
            }
        }
    }

    pub fn interrupt_pending(&mut self, reason: impl Into<String>) {
        let reason = reason.into();
        for record in &mut self.records {
            if record.outcome == DriverEffectOutcome::Pending {
                record.outcome = DriverEffectOutcome::Interrupted {
                    reason: reason.clone(),
                };
            }
        }
    }

    pub fn has_pending(&self) -> bool {
        self.records
            .iter()
            .any(|record| record.outcome == DriverEffectOutcome::Pending)
    }

    fn resolve(&mut self, id: DriverEffectId, outcome: DriverEffectOutcome) -> bool {
        let Some(record) = self.records.iter_mut().find(|record| record.id == id) else {
            return false;
        };
        if record.outcome != DriverEffectOutcome::Pending {
            return false;
        }
        record.outcome = outcome;
        true
    }
}

/// Reset driver-local cycle state after external input is appended.
pub fn input_accepted(config: &ThreadDriverConfig, state: &mut DriverState) -> Result<(), String> {
    match (config, state) {
        (
            ThreadDriverConfig::BuiltinSingleAgentChat,
            DriverState::BuiltinSingleAgentChat {
                cycles_started,
                turns_in_cycle,
            },
        ) => {
            *cycles_started = cycles_started.saturating_add(1);
            *turns_in_cycle = 0;
            Ok(())
        }
    }
}

/// Ask the driver what to do at a runnable turn boundary.
pub fn next_effect(
    config: &ThreadDriverConfig,
    state: &mut DriverState,
    participants: &ThreadParticipants,
    max_turns: u32,
) -> Result<DriverEffect, String> {
    match (config, state) {
        (
            ThreadDriverConfig::BuiltinSingleAgentChat,
            DriverState::BuiltinSingleAgentChat { turns_in_cycle, .. },
        ) => {
            if *turns_in_cycle >= max_turns {
                return Ok(DriverEffect::Finish);
            }
            *turns_in_cycle += 1;
            Ok(DriverEffect::RunAgent {
                participant_id: participants.default_responder.clone(),
                turn: *turns_in_cycle,
            })
        }
    }
}

/// Interpret one completed agent response. The compatibility driver enters a
/// tool loop iff the response requested tools; otherwise it yields for input.
pub fn agent_completed(
    config: &ThreadDriverConfig,
    state: &DriverState,
    participant_id: &ParticipantId,
    participants: &ThreadParticipants,
    has_tool_calls: bool,
) -> Result<DriverEffect, String> {
    match (config, state) {
        (
            ThreadDriverConfig::BuiltinSingleAgentChat,
            DriverState::BuiltinSingleAgentChat { .. },
        ) => {
            if participant_id != &participants.default_responder {
                return Err(format!(
                    "builtin single-agent driver received completion from `{participant_id}`, expected `{}`",
                    participants.default_responder
                ));
            }
            Ok(if has_tool_calls {
                DriverEffect::DispatchTools
            } else {
                DriverEffect::Finish
            })
        }
    }
}

/// Interpret completion of all tools requested by an agent turn.
pub fn tools_completed(
    config: &ThreadDriverConfig,
    state: &DriverState,
    participant_id: &ParticipantId,
    participants: &ThreadParticipants,
) -> Result<DriverEffect, String> {
    match (config, state) {
        (
            ThreadDriverConfig::BuiltinSingleAgentChat,
            DriverState::BuiltinSingleAgentChat { .. },
        ) => {
            if participant_id != &participants.default_responder {
                return Err(format!(
                    "builtin single-agent driver received tools from `{participant_id}`, expected `{}`",
                    participants.default_responder
                ));
            }
            Ok(DriverEffect::Continue)
        }
    }
}

pub fn turns_in_cycle(state: &DriverState) -> u32 {
    match state {
        DriverState::BuiltinSingleAgentChat { turns_in_cycle, .. } => *turns_in_cycle,
    }
}

/// One-time bridge for thread JSON written before driver state existed.
pub fn import_legacy_turn_count(state: &mut DriverState, legacy_turns: u32) {
    if legacy_turns == 0 {
        return;
    }
    match state {
        DriverState::BuiltinSingleAgentChat { turns_in_cycle, .. } if *turns_in_cycle == 0 => {
            *turns_in_cycle = legacy_turns;
        }
        DriverState::BuiltinSingleAgentChat { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_driver_runs_responder_tools_then_responder() {
        let config = ThreadDriverConfig::default();
        let participants = ThreadParticipants::default();
        let mut state = DriverState::default();
        input_accepted(&config, &mut state).unwrap();

        let first = next_effect(&config, &mut state, &participants, 4).unwrap();
        assert_eq!(
            first,
            DriverEffect::RunAgent {
                participant_id: participants.default_responder.clone(),
                turn: 1,
            }
        );
        assert_eq!(
            agent_completed(
                &config,
                &state,
                &participants.default_responder,
                &participants,
                true,
            )
            .unwrap(),
            DriverEffect::DispatchTools
        );
        assert_eq!(
            tools_completed(
                &config,
                &state,
                &participants.default_responder,
                &participants,
            )
            .unwrap(),
            DriverEffect::Continue
        );
        assert!(matches!(
            next_effect(&config, &mut state, &participants, 4).unwrap(),
            DriverEffect::RunAgent { turn: 2, .. }
        ));
    }

    #[test]
    fn builtin_driver_finishes_at_turn_limit() {
        let config = ThreadDriverConfig::default();
        let participants = ThreadParticipants::default();
        let mut state = DriverState::default();
        input_accepted(&config, &mut state).unwrap();
        assert!(matches!(
            next_effect(&config, &mut state, &participants, 1).unwrap(),
            DriverEffect::RunAgent { turn: 1, .. }
        ));
        assert_eq!(
            next_effect(&config, &mut state, &participants, 1).unwrap(),
            DriverEffect::Finish
        );
    }

    #[test]
    fn legacy_turn_count_only_fills_empty_state() {
        let mut state = DriverState::default();
        import_legacy_turn_count(&mut state, 3);
        assert_eq!(turns_in_cycle(&state), 3);
        import_legacy_turn_count(&mut state, 7);
        assert_eq!(turns_in_cycle(&state), 3);
    }

    #[test]
    fn builtin_state_has_stable_tagged_wire_shape() {
        let state = DriverState::default();
        assert_eq!(
            serde_json::to_value(&state).unwrap(),
            serde_json::json!({
                "kind": "builtin_single_agent_chat",
                "cycles_started": 0,
                "turns_in_cycle": 0,
            })
        );
        assert_eq!(
            serde_json::from_value::<DriverState>(serde_json::json!({
                "kind": "builtin_single_agent_chat"
            }))
            .unwrap(),
            state
        );
    }

    #[test]
    fn effect_journal_is_monotonic_and_terminal_records_are_immutable() {
        let generation = GenerationContext::new("run-1", "agent");
        let mut journal = DriverEffectJournal::default();
        let run_id = journal.record(PersistedDriverEffect::RunAgent {
            generation: generation.clone(),
            turn: 1,
        });
        assert_eq!(run_id, 1);
        assert!(journal.has_pending());
        assert!(journal.complete(run_id));
        assert!(!journal.fail(run_id, "too late"));

        let tools_id = journal.record(PersistedDriverEffect::DispatchTools {
            generation,
            tool_use_ids: vec!["toolu-1".into()],
        });
        assert_eq!(tools_id, 2);
        assert!(journal.interrupt(tools_id, "restart"));
        assert!(!journal.has_pending());
        assert!(matches!(
            &journal.records()[1].outcome,
            DriverEffectOutcome::Interrupted { reason } if reason == "restart"
        ));
    }

    #[test]
    fn effect_journal_round_trip_preserves_pending_request_and_next_id() {
        let mut journal = DriverEffectJournal::default();
        journal.record(PersistedDriverEffect::RunAgent {
            generation: GenerationContext::new("run-1", "reviewer"),
            turn: 2,
        });
        let json = serde_json::to_value(&journal).unwrap();
        assert_eq!(json["next_id"], 2);
        assert_eq!(json["records"][0]["id"], 1);
        assert_eq!(json["records"][0]["effect"]["kind"], "run_agent");
        assert_eq!(json["records"][0]["outcome"]["status"], "pending");

        let mut decoded: DriverEffectJournal = serde_json::from_value(json).unwrap();
        assert_eq!(
            decoded.record(PersistedDriverEffect::Finish {
                generation: None,
                reason: DriverFinishReason::TurnLimit,
            }),
            2
        );
    }

    #[test]
    fn journal_repairs_stale_or_missing_next_id_before_append() {
        let mut journal: DriverEffectJournal = serde_json::from_value(serde_json::json!({
            "records": [{
                "id": 7,
                "effect": {
                    "kind": "finish",
                    "reason": "turn_limit"
                },
                "outcome": { "status": "completed" }
            }]
        }))
        .unwrap();
        assert_eq!(
            journal.record(PersistedDriverEffect::Finish {
                generation: None,
                reason: DriverFinishReason::TurnLimit,
            }),
            8
        );
    }
}
