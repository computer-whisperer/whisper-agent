//! Weave — a durable driver instance.
//!
//! The middle tier of the pod/weave/thread model (see
//! `docs/design_configurable_threads.md`): `pod : weave : thread ::
//! namespace : process : file`. A weave owns the driver program selection,
//! its persisted mutable state, the durable effect journal, and *references*
//! to the threads it coordinates. It holds no tokens of its own; threads
//! remain the materialized LLM contexts.
//!
//! Step 5 scope: every thread is coordinated by a singleton weave (weave id
//! = thread id) running the compatibility driver. The scheduler routes
//! thread boundary outcomes through the ticking weave's policy methods here
//! and executes the returned effects; `Thread` no longer consults driver
//! policy directly.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use whisper_agent_protocol::{ParticipantId, ThreadDriverConfig, ThreadParticipants};

use crate::runtime::driver::{
    self, DriverEffect, DriverEffectId, DriverEffectJournal, DriverState, PersistedDriverEffect,
};

pub type WeaveId = String;

/// Role a referenced thread plays in this weave. Step 5 knows only the
/// primary conversation context; later steps add derived roles (compaction
/// sources, permission checkers, subagents) carrying relationship metadata.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WeaveThreadRole {
    Primary,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct WeaveThreadRef {
    pub thread_id: String,
    pub role: WeaveThreadRole,
}

/// Durable driver instance. Persisted at
/// `<pods_root>/<pod_id>/weaves/<weave_id>.json`, flushed through the same
/// dirty-set batching as threads so the journal keeps its write-ahead
/// guarantee (records land on disk before newly queued lazy provider/tool
/// futures are polled).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Weave {
    pub id: WeaveId,
    pub pod_id: String,
    /// Driver program selection. Step 5 carries the builtin compatibility
    /// driver only; scripted (Lua) drivers arrive with migration step 7.
    #[serde(default)]
    pub driver: ThreadDriverConfig,
    /// Persisted mutable state owned by the selected driver.
    #[serde(default)]
    pub driver_state: DriverState,
    /// Durable requested/completed effect history for this driver instance.
    #[serde(default)]
    pub effect_journal: DriverEffectJournal,
    /// Threads this weave references. The single-ticker invariant is
    /// enforced by the scheduler's `thread_ticker` index, not here; this
    /// list is the weave's own record of what it coordinates.
    pub threads: Vec<WeaveThreadRef>,
    pub created: DateTime<Utc>,
    pub last_active: DateTime<Utc>,
}

impl Weave {
    /// Fresh singleton weave for one thread — the step-5 compatibility
    /// shape and the load-migration target for legacy thread JSON.
    pub fn singleton_for_thread(
        thread_id: impl Into<String>,
        pod_id: impl Into<String>,
        driver: ThreadDriverConfig,
    ) -> Self {
        let thread_id = thread_id.into();
        let now = Utc::now();
        Self {
            id: thread_id.clone(),
            pod_id: pod_id.into(),
            driver_state: DriverState::for_config(&driver),
            driver,
            effect_journal: DriverEffectJournal::default(),
            threads: vec![WeaveThreadRef {
                thread_id,
                role: WeaveThreadRole::Primary,
            }],
            created: now,
            last_active: now,
        }
    }

    pub fn primary_thread_id(&self) -> Option<&str> {
        self.threads
            .iter()
            .find(|r| r.role == WeaveThreadRole::Primary)
            .map(|r| r.thread_id.as_str())
    }

    pub fn references(&self, thread_id: &str) -> bool {
        self.threads.iter().any(|r| r.thread_id == thread_id)
    }

    pub fn touch(&mut self) {
        self.last_active = Utc::now();
    }

    // ---------- driver policy boundaries ----------
    //
    // Thin wrappers so the scheduler consults one object and `Thread` never
    // imports driver policy. Participants and max_turns are passed in from
    // the thread's config: execution setup stays thread-level until the
    // profile definitions migrate up in steps 6-7.

    /// External input was accepted into a coordinated thread; reset the
    /// driver's cycle state.
    pub fn input_accepted(&mut self) -> Result<(), String> {
        self.touch();
        driver::input_accepted(&self.driver, &mut self.driver_state)
    }

    /// Ask the driver what to do at a runnable turn boundary.
    pub fn next_effect(
        &mut self,
        participants: &ThreadParticipants,
        max_turns: u32,
    ) -> Result<DriverEffect, String> {
        driver::next_effect(
            &self.driver,
            &mut self.driver_state,
            participants,
            max_turns,
        )
    }

    /// Interpret one completed agent response.
    pub fn agent_completed(
        &self,
        participant_id: &ParticipantId,
        participants: &ThreadParticipants,
        has_tool_calls: bool,
    ) -> Result<DriverEffect, String> {
        driver::agent_completed(
            &self.driver,
            &self.driver_state,
            participant_id,
            participants,
            has_tool_calls,
        )
    }

    /// Interpret completion of all tools requested by an agent turn.
    pub fn tools_completed(
        &self,
        participant_id: &ParticipantId,
        participants: &ThreadParticipants,
    ) -> Result<DriverEffect, String> {
        driver::tools_completed(
            &self.driver,
            &self.driver_state,
            participant_id,
            participants,
        )
    }

    // ---------- journal ----------

    /// Record a pending effect. The caller must dispatch the matching I/O
    /// only after the weave has been flushed (the scheduler's dirty-set
    /// batching provides this write-ahead boundary).
    pub fn record_pending_effect(&mut self, effect: PersistedDriverEffect) -> DriverEffectId {
        self.touch();
        self.effect_journal.record(effect)
    }

    /// Record an effect that completes synchronously (Continue / Finish).
    pub fn record_completed_effect(&mut self, effect: PersistedDriverEffect) {
        self.touch();
        let effect_id = self.effect_journal.record(effect);
        let completed = self.effect_journal.complete(effect_id);
        debug_assert!(completed);
    }

    /// Resolve a pending record. Zero is the legacy sentinel for internal
    /// state persisted before effect journaling; it is accepted and ignored.
    pub fn complete_effect(&mut self, effect_id: DriverEffectId) {
        if effect_id == 0 {
            return;
        }
        self.touch();
        let completed = self.effect_journal.complete(effect_id);
        debug_assert!(completed);
    }

    pub fn fail_pending(&mut self, message: impl Into<String>) {
        self.touch();
        self.effect_journal.fail_pending(message);
    }

    pub fn interrupt_pending(&mut self, reason: impl Into<String>) {
        self.touch();
        self.effect_journal.interrupt_pending(reason);
    }

    /// One-time bridge for state migrated off pre-weave thread JSON.
    pub fn import_thread_driver_state(
        &mut self,
        state: DriverState,
        journal: DriverEffectJournal,
        legacy_turns: u32,
    ) {
        self.driver_state = state;
        self.effect_journal = journal;
        driver::import_legacy_turn_count(&mut self.driver_state, legacy_turns);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::driver::{DriverEffectOutcome, DriverFinishReason};
    use whisper_agent_protocol::GenerationContext;

    #[test]
    fn singleton_weave_shares_the_thread_id_and_ticks_it() {
        let weave =
            Weave::singleton_for_thread("t-1", "pod", ThreadDriverConfig::BuiltinSingleAgentChat);
        assert_eq!(weave.id, "t-1");
        assert_eq!(weave.primary_thread_id(), Some("t-1"));
        assert!(weave.references("t-1"));
        assert!(!weave.references("t-2"));
    }

    #[test]
    fn boundary_flow_mirrors_the_compatibility_driver() {
        let mut weave =
            Weave::singleton_for_thread("t-1", "pod", ThreadDriverConfig::BuiltinSingleAgentChat);
        let participants = ThreadParticipants::default();
        weave.input_accepted().unwrap();

        let effect = weave.next_effect(&participants, 4).unwrap();
        assert!(matches!(effect, DriverEffect::RunAgent { turn: 1, .. }));
        assert_eq!(
            weave
                .agent_completed(&participants.default_responder, &participants, true)
                .unwrap(),
            DriverEffect::DispatchTools
        );
        assert_eq!(
            weave
                .tools_completed(&participants.default_responder, &participants)
                .unwrap(),
            DriverEffect::Continue
        );
        assert!(matches!(
            weave.next_effect(&participants, 4).unwrap(),
            DriverEffect::RunAgent { turn: 2, .. }
        ));
    }

    #[test]
    fn journal_helpers_record_resolve_and_tolerate_legacy_zero() {
        let mut weave =
            Weave::singleton_for_thread("t-1", "pod", ThreadDriverConfig::BuiltinSingleAgentChat);
        let id = weave.record_pending_effect(PersistedDriverEffect::RunAgent {
            generation: GenerationContext::new("run-1", "agent"),
            turn: 1,
        });
        assert!(weave.effect_journal.has_pending());
        weave.complete_effect(0); // legacy sentinel — no-op, no panic
        assert!(weave.effect_journal.has_pending());
        weave.complete_effect(id);
        assert!(!weave.effect_journal.has_pending());

        weave.record_completed_effect(PersistedDriverEffect::Finish {
            generation: None,
            reason: DriverFinishReason::AgentCompleted,
        });
        assert!(
            weave
                .effect_journal
                .records()
                .iter()
                .all(|r| r.outcome == DriverEffectOutcome::Completed)
        );
    }

    #[test]
    fn weave_json_round_trips_and_tolerates_missing_driver_fields() {
        let weave =
            Weave::singleton_for_thread("t-1", "pod", ThreadDriverConfig::BuiltinSingleAgentChat);
        let mut json = serde_json::to_value(&weave).unwrap();
        assert_eq!(json["id"], "t-1");
        assert_eq!(json["threads"][0]["role"], "primary");

        // Older/foreign weave JSON without driver fields defaults cleanly.
        let object = json.as_object_mut().unwrap();
        object.remove("driver");
        object.remove("driver_state");
        object.remove("effect_journal");
        let decoded: Weave = serde_json::from_value(json).unwrap();
        assert_eq!(decoded.driver, ThreadDriverConfig::BuiltinSingleAgentChat);
        assert!(decoded.effect_journal.records().is_empty());
    }
}
