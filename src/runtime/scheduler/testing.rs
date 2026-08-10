//! Scheduler test harness.
//!
//! Constructs a real [`Scheduler`] over an empty resource surface: no
//! backends, no shared MCP hosts, no buckets, no persister. Everything
//! the constructor touches is cheap and local — `Scheduler::new` only
//! performs I/O for shared-MCP catalog entries (none) — and the lazy
//! provider/tool futures pushed into `pending_io` are never polled
//! unless a test polls them, so scheduler-level choreography (thread
//! creation, weave effects, ticker admission, sweeps) is testable
//! without a single network or provider dependency.
//!
//! First consumer: the step-6 cross-thread effect executors, whose
//! admission logic has no production caller until the scripted driver
//! of migration step 7.

use super::*;
use crate::knowledge::RerankedCandidate;
use crate::runtime::driver::{
    DriverEffectOutcome, DriverState, EntryRef, PersistedDriverEffect, ThreadRelationship,
};
use crate::runtime::thread::{IoResult, ThreadInternalState};
use crate::runtime::weave::WeaveThreadRole;
use std::sync::atomic::{AtomicU64, Ordering};
use whisper_agent_protocol::weave::PresentationBlock;
use whisper_agent_protocol::{
    AllowMap, ContentBlock, GenerationContext, Message, PodAllow, PodConfig, PodLimits,
    PodModifyCap, ThreadConfigOverride, ThreadDefaultCaps, ThreadDefaults, ThreadDriverConfig,
};

const CHECKER_DRIVER: &str = include_str!("../../../examples/drivers/auto_mode_checker.lua");
const ROUNDTABLE_DRIVER: &str = include_str!("../../../examples/drivers/roundtable.lua");
const TITLED_CHAT_DRIVER: &str = include_str!("../../../examples/drivers/titled_chat.lua");

static COUNTER: AtomicU64 = AtomicU64::new(0);

const TEST_POD: &str = "testpod";

fn temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("whisper-sched-test-{}-{}", std::process::id(), n))
}

fn test_pod_config() -> PodConfig {
    PodConfig {
        name: TEST_POD.into(),
        description: None,
        created_at: "2026-07-31T10:00:00Z".into(),
        allow: PodAllow {
            // Two names so cross-backend knob/derive paths are testable;
            // the harness registry tolerates unknown backend ids and no
            // model call ever leaves the building (respond_model).
            backends: vec!["anthropic".into(), "openai".into()],
            mcp_hosts: Vec::new(),
            host_env: Vec::new(),
            knowledge_buckets: Vec::new(),
            tools: AllowMap::allow_all(),
            caps: Default::default(),
        },
        thread_defaults: ThreadDefaults {
            backend: "anthropic".into(),
            model: "claude-sonnet-4-6".into(),
            system_prompt_file: "system_prompt.md".into(),
            driver: None,
            driver_config: Default::default(),
            max_tokens: 8000,
            max_turns: 30,
            host_env: Vec::new(),
            mcp_hosts: Vec::new(),
            compaction: Default::default(),
            autoquery: Default::default(),
            caps: Default::default(),
            tool_surface: Default::default(),
            tunables: Default::default(),
        },
        limits: PodLimits::default(),
    }
}

pub(super) struct Harness {
    pub sched: Scheduler,
    dir: PathBuf,
    // Keep the outbound channels alive so scheduler sends never error.
    _stream_rx: mpsc::UnboundedReceiver<StreamUpdate>,
    _usage_rx: mpsc::UnboundedReceiver<BackendUsageUpdate>,
    _bucket_rx: mpsc::UnboundedReceiver<buckets::BucketTaskUpdate>,
    _resync_rx: mpsc::UnboundedReceiver<(Option<String>, String)>,
}

impl Drop for Harness {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

pub(super) async fn harness() -> Harness {
    let dir = temp_dir();
    let pod = crate::pod::Pod::new(
        TEST_POD.into(),
        dir.join(TEST_POD),
        test_pod_config(),
        String::new(),
        "you are a test agent".into(),
    );
    let audit = crate::runtime::audit::AuditLog::open(dir.join("audit.log"))
        .await
        .expect("audit log in temp dir");
    let forensics = crate::runtime::forensics::ForensicSink::new(dir.join("forensics"));
    let catalog = crate::tools::shared_mcp_catalog::CatalogStore::load(dir.join("mcp.json"))
        .expect("missing catalog file loads empty");
    let (sched, stream_rx, usage_rx, bucket_rx, resync_rx) = Scheduler::new(
        pod,
        "test-host".into(),
        "test-install".into(),
        "test-session".into(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        crate::knowledge::registry::BucketRegistry::default(),
        audit,
        forensics,
        catalog,
        Vec::new(),
        Arc::new(crate::tools::host_env_link::LiveDaemonRegistry::new()),
        None,
        Default::default(),
    )
    .await
    .expect("empty-surface scheduler constructs");
    Harness {
        sched,
        dir,
        _stream_rx: stream_rx,
        _usage_rx: usage_rx,
        _bucket_rx: bucket_rx,
        _resync_rx: resync_rx,
    }
}

impl Harness {
    /// Create a plain thread in the default pod (mints its singleton
    /// weave, which shares the thread's id).
    pub(super) fn create_thread(&mut self) -> String {
        let mut pending_io = FuturesUnordered::new();
        self.sched
            .create_task(
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                &mut pending_io,
            )
            .expect("thread creation on empty surface")
    }

    fn weave_of(&self, thread_id: &str) -> String {
        self.sched
            .thread_ticker
            .get(thread_id)
            .cloned()
            .expect("thread has a ticking weave")
    }

    fn last_record(&self, weave_id: &str) -> &crate::runtime::driver::DriverEffectRecord {
        self.sched.weaves[weave_id]
            .effect_journal
            .records()
            .last()
            .expect("journal has records")
    }

    fn derive(&mut self, weave_id: &str, seed: Vec<Message>, kind: &str) -> Result<String, String> {
        let mut pending_io = FuturesUnordered::new();
        self.sched.weave_derive_thread(
            weave_id,
            None,
            None,
            seed,
            ThreadRelationship {
                kind: kind.into(),
                source: None,
            },
            None,
            &mut pending_io,
        )
    }

    /// Install a scripted driver program into the default pod.
    fn install_driver(&self, name: &str, source: &str) {
        let drivers_dir = self.dir.join(TEST_POD).join("drivers");
        std::fs::create_dir_all(&drivers_dir).expect("create drivers dir");
        std::fs::write(drivers_dir.join(format!("{name}.lua")), source).expect("write driver");
    }

    /// Create a thread coordinated by the named scripted driver.
    fn create_scripted_thread(&mut self, driver: &str) -> Result<String, String> {
        self.create_scripted_thread_with_config(driver, Default::default())
    }

    /// Create a scripted thread with submitted knob values (step 10) —
    /// the harness stand-in for the new-thread form's config controls.
    fn create_scripted_thread_with_config(
        &mut self,
        driver: &str,
        config: std::collections::BTreeMap<String, serde_json::Value>,
    ) -> Result<String, String> {
        let mut pending_io = FuturesUnordered::new();
        self.sched.create_task(
            None,
            None,
            None,
            Some(ThreadConfigOverride {
                driver: Some(ThreadDriverConfig::Scripted {
                    name: driver.into(),
                    config,
                }),
                ..Default::default()
            }),
            None,
            None,
            None,
            None,
            None,
            None,
            &mut pending_io,
        )
    }

    fn internal_of(&self, thread_id: &str) -> &ThreadInternalState {
        &self.sched.tasks[thread_id].internal
    }

    /// Complete the thread's in-flight model call with a synthetic
    /// response and pump its step loop — the test-side stand-in for a
    /// provider round trip.
    fn respond_model(
        &mut self,
        thread_id: &str,
        content: Vec<ContentBlock>,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        let op_id = match self.internal_of(thread_id) {
            ThreadInternalState::AwaitingModel { op_id, .. } => *op_id,
            other => panic!("thread `{thread_id}` is not awaiting a model call: {other:?}"),
        };
        let response = crate::providers::model::ModelResponse {
            content,
            stop_reason: Some("end_turn".into()),
            usage: Default::default(),
        };
        let mut events = Vec::new();
        self.sched
            .tasks
            .get_mut(thread_id)
            .unwrap()
            .apply_io_result(op_id, IoResult::ModelCall(Ok(response)), &mut events);
        self.sched.step_until_blocked(thread_id, pending_io);
    }

    /// Complete one in-flight tool call with a synthetic text result
    /// and pump the step loop — the test-side stand-in for a tool
    /// executor round trip.
    fn respond_tool(
        &mut self,
        thread_id: &str,
        tool_use_id: &str,
        text: &str,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        let op_id = match self.internal_of(thread_id) {
            ThreadInternalState::AwaitingTools {
                pending_io: ops, ..
            } => ops
                .iter()
                .find(|(_, tid)| tid.as_str() == tool_use_id)
                .map(|(op, _)| *op)
                .unwrap_or_else(|| {
                    panic!("thread `{thread_id}` has no pending tool op for `{tool_use_id}`")
                }),
            other => panic!("thread `{thread_id}` is not awaiting tools: {other:?}"),
        };
        let mut events = Vec::new();
        self.sched
            .tasks
            .get_mut(thread_id)
            .unwrap()
            .apply_io_result(
                op_id,
                IoResult::ToolCall {
                    tool_use_id: tool_use_id.to_string(),
                    result: Ok(crate::tools::mcp::CallToolResult {
                        content: vec![crate::tools::mcp::McpContentBlock::Text {
                            text: text.to_string(),
                        }],
                        is_error: false,
                    }),
                },
                &mut events,
            );
        self.sched.step_until_blocked(thread_id, pending_io);
    }

    /// Fail the thread's in-flight model call through the scheduler's
    /// real io-completion path — unlike [`Self::respond_model`], which
    /// applies straight to the task, this exercises the layer that
    /// resolves journal records and notifies scripted drivers of the
    /// death.
    fn fail_model(
        &mut self,
        thread_id: &str,
        message: &str,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        let op_id = match self.internal_of(thread_id) {
            ThreadInternalState::AwaitingModel { op_id, .. } => *op_id,
            other => panic!("thread `{thread_id}` is not awaiting a model call: {other:?}"),
        };
        self.sched.apply_io_completion(
            crate::runtime::io_dispatch::IoCompletion {
                thread_id: thread_id.to_string(),
                op_id,
                result: IoResult::ModelCall(Err(message.to_string())),
                pod_update: None,
                scheduler_command: None,
                knowledge_hit_keys: Vec::new(),
            },
            pending_io,
        );
    }

    /// Install an in-memory behavior on the default pod — the harness
    /// stand-in for a `<pod>/behaviors/<id>/` directory, firing
    /// through the real `run_behavior` path. Disk-state writes ride
    /// the (unpolled) dirty machinery, so nothing touches the
    /// filesystem.
    fn install_behavior(
        &mut self,
        id: &str,
        config: whisper_agent_protocol::BehaviorConfig,
        prompt: &str,
    ) {
        let pod = self.sched.pods.get_mut(TEST_POD).expect("test pod");
        let dir = pod.dir.join("behaviors").join(id);
        pod.behaviors.insert(
            id.to_string(),
            crate::pod::behaviors::Behavior {
                id: id.to_string(),
                pod_id: TEST_POD.to_string(),
                dir,
                config: Some(config),
                raw_toml: String::new(),
                prompt: prompt.to_string(),
                system_prompt: None,
                state: Default::default(),
                cron: None,
                load_error: None,
            },
        );
    }

    fn behavior_state(&self, id: &str) -> &whisper_agent_protocol::BehaviorState {
        &self.sched.pods[TEST_POD].behaviors[id].state
    }
}

/// A behavior whose `[thread]` names a scripted driver — the TOML
/// authoring surface of step 11 slice 2, built directly.
fn scripted_behavior_config(
    driver: &str,
    knobs: std::collections::BTreeMap<String, serde_json::Value>,
) -> whisper_agent_protocol::BehaviorConfig {
    whisper_agent_protocol::BehaviorConfig {
        name: "scripted test behavior".into(),
        description: None,
        trigger: Default::default(),
        thread: whisper_agent_protocol::BehaviorThreadOverride {
            driver: Some(driver.to_string()),
            driver_config: knobs,
            ..Default::default()
        },
        on_completion: Default::default(),
        scope: Default::default(),
    }
}

fn text_block(text: &str) -> ContentBlock {
    ContentBlock::Text { text: text.into() }
}

fn tool_use_block(id: &str, name: &str) -> ContentBlock {
    ContentBlock::ToolUse {
        id: id.into(),
        name: name.into(),
        input: serde_json::json!({}),
        replay: None,
    }
}

fn tool_use_block_with_input(id: &str, name: &str, input: serde_json::Value) -> ContentBlock {
    ContentBlock::ToolUse {
        id: id.into(),
        name: name.into(),
        input,
        replay: None,
    }
}

// ---------- derive_thread ----------

#[tokio::test]
async fn derive_thread_attaches_to_deriving_weave_without_singleton() {
    let mut h = harness().await;
    let primary = h.create_thread();
    let weave_id = h.weave_of(&primary);
    assert_eq!(weave_id, primary, "singleton weave shares the thread id");

    let seed = vec![Message::user_text("curated seed context")];
    let derived = h.derive(&weave_id, seed, "check").expect("derive succeeds");

    assert!(h.sched.tasks.contains_key(&derived));
    assert!(
        !h.sched.weaves.contains_key(&derived),
        "derived thread must not mint a singleton weave"
    );
    assert_eq!(h.sched.thread_ticker.get(&derived), Some(&weave_id));

    let weave = &h.sched.weaves[&weave_id];
    let aux = weave
        .threads
        .iter()
        .find(|r| r.thread_id == derived)
        .expect("deriving weave references the new thread");
    assert_eq!(aux.role, WeaveThreadRole::Auxiliary);
    assert!(aux.ticks);
    assert_eq!(aux.relationship.as_ref().unwrap().kind, "check");

    let record = h.last_record(&weave_id);
    assert!(matches!(
        &record.effect,
        PersistedDriverEffect::DeriveThread { thread_id: Some(t), seed_entries: 1, .. }
            if *t == derived
    ));
    assert_eq!(record.outcome, DriverEffectOutcome::Completed);

    // The seed landed after the setup prefix, author preserved.
    let conv = h.sched.tasks[&derived].conversation.messages();
    let last = conv.last().expect("seed entry present");
    assert_eq!(
        last.effective_author().as_str(),
        whisper_agent_protocol::DEFAULT_INPUT_PARTICIPANT_ID
    );
}

#[tokio::test]
async fn derive_thread_refuses_unauthored_seed_and_journals_the_refusal() {
    let mut h = harness().await;
    let primary = h.create_thread();
    let weave_id = h.weave_of(&primary);
    let threads_before = h.sched.tasks.len();

    let unauthored = Message {
        author: None,
        run_id: None,
        role: whisper_agent_protocol::Role::User,
        content: vec![whisper_agent_protocol::ContentBlock::Text {
            text: "no provenance".into(),
        }],
    };
    let err = h
        .derive(&weave_id, vec![unauthored], "check")
        .expect_err("unauthored seed refused");
    assert!(err.contains("no author"));
    assert_eq!(h.sched.tasks.len(), threads_before, "no thread created");

    let record = h.last_record(&weave_id);
    assert!(matches!(
        &record.effect,
        PersistedDriverEffect::DeriveThread {
            thread_id: None,
            ..
        }
    ));
    assert!(matches!(
        &record.outcome,
        DriverEffectOutcome::Failed { message } if message.contains("no author")
    ));
}

#[tokio::test]
async fn derived_scope_inherits_primary_and_caps_narrow_not_assign() {
    let mut h = harness().await;
    let primary = h.create_thread();
    let weave_id = h.weave_of(&primary);

    // Distinctively narrow the primary's active scope below both the
    // pod ceiling (ModifyAllow) and the thread default (Memories).
    h.sched.tasks.get_mut(&primary).unwrap().scope.pod_modify = PodModifyCap::None;

    // No override: derived base scope is the primary's scope verbatim.
    let inherited = h.derive(&weave_id, Vec::new(), "check").unwrap();
    assert_eq!(
        h.sched.tasks[&inherited].scope.pod_modify,
        PodModifyCap::None
    );

    // A caps override above the primary's level but under the pod
    // ceiling must NOT re-widen: caps compose by narrowing for derived
    // threads.
    let widening = whisper_agent_protocol::ThreadConfigOverride {
        caps: Some(ThreadDefaultCaps {
            pod_modify: PodModifyCap::Content,
            ..Default::default()
        }),
        ..Default::default()
    };
    let mut pending_io = FuturesUnordered::new();
    let derived = h
        .sched
        .weave_derive_thread(
            &weave_id,
            Some(widening),
            None,
            Vec::new(),
            ThreadRelationship {
                kind: "check".into(),
                source: None,
            },
            None,
            &mut pending_io,
        )
        .expect("override within pod ceiling is admitted");
    assert_eq!(
        h.sched.tasks[&derived].scope.pod_modify,
        PodModifyCap::None,
        "derived caps narrow against the primary scope, never widen toward the ceiling"
    );
}

// ---------- append_entry ----------

#[tokio::test]
async fn append_entry_pollutes_referenced_thread_and_journals_provenance() {
    let mut h = harness().await;
    let primary = h.create_thread();
    let weave_id = h.weave_of(&primary);
    let checker = h.derive(&weave_id, Vec::new(), "check").unwrap();

    let before = h.sched.tasks[&primary].conversation.messages().len();
    let index = h
        .sched
        .weave_append_entry(
            &weave_id,
            &primary,
            Message::user_text("verdict: allow").with_author("checker"),
            Some(EntryRef {
                thread_id: checker.clone(),
                entry_index: Some(0),
            }),
        )
        .expect("append into referenced idle thread");

    assert_eq!(index, before, "returned index addresses the appended entry");
    let conv = h.sched.tasks[&primary].conversation.messages();
    assert_eq!(conv.len(), before + 1);
    assert_eq!(conv[index].effective_author().as_str(), "checker");
    assert!(
        h.sched.tasks[&primary].is_idle(),
        "append is pure pollution — the target is not woken"
    );

    let record = h.last_record(&weave_id);
    assert!(matches!(
        &record.effect,
        PersistedDriverEffect::AppendEntry {
            entry_index: Some(i),
            source: Some(src),
            ..
        } if *i == index && src.thread_id == checker
    ));
    assert_eq!(record.outcome, DriverEffectOutcome::Completed);
}

#[tokio::test]
async fn append_entry_admission_refusals() {
    let mut h = harness().await;
    let primary = h.create_thread();
    let weave_id = h.weave_of(&primary);
    let stranger = h.create_thread(); // its own singleton; not referenced by weave_id

    // Unreferenced target: refused and journaled Failed.
    let err = h
        .sched
        .weave_append_entry(
            &weave_id,
            &stranger,
            Message::user_text("x").with_author("checker"),
            None,
        )
        .expect_err("unreferenced target refused");
    assert!(err.contains("does not reference"));
    assert!(matches!(
        &h.last_record(&weave_id).outcome,
        DriverEffectOutcome::Failed { .. }
    ));

    // Unauthored message: caller-contract violation — refused with no
    // journal record (there is no author to attribute one to).
    let records_before = h.sched.weaves[&weave_id].effect_journal.records().len();
    let unauthored = Message {
        author: None,
        run_id: None,
        role: whisper_agent_protocol::Role::User,
        content: vec![whisper_agent_protocol::ContentBlock::Text { text: "x".into() }],
    };
    let err = h
        .sched
        .weave_append_entry(&weave_id, &primary, unauthored, None)
        .expect_err("unauthored append refused");
    assert!(err.contains("authored"));
    assert_eq!(
        h.sched.weaves[&weave_id].effect_journal.records().len(),
        records_before
    );

    // Mid-generation target: refused so the materialized log never
    // carries an entry sequenced before output generated without it.
    {
        let task = h.sched.tasks.get_mut(&primary).unwrap();
        task.submit_user_message("hi".into(), Vec::new(), Vec::new());
        let mut events = Vec::new();
        let generation = GenerationContext::new("run-test", "agent");
        assert!(
            task.begin_model_call(1, generation, 7, 1, &mut events)
                .is_some(),
            "thread parks in AwaitingModel"
        );
        assert!(task.is_mid_generation());
    }
    let err = h
        .sched
        .weave_append_entry(
            &weave_id,
            &primary,
            Message::user_text("late").with_author("checker"),
            None,
        )
        .expect_err("mid-generation append refused");
    assert!(err.contains("mid-generation"));
    assert!(matches!(
        &h.last_record(&weave_id).outcome,
        DriverEffectOutcome::Failed { .. }
    ));
}

// ---------- adopt_ticker / release_ticker ----------

#[tokio::test]
async fn release_then_adopt_transfers_a_ticker_between_weaves() {
    let mut h = harness().await;
    let alpha = h.create_thread();
    let beta = h.create_thread();
    let alpha_weave = h.weave_of(&alpha);
    let beta_weave = h.weave_of(&beta);

    // Release: idle thread admitted; ref survives with ticks=false.
    h.sched
        .weave_release_ticker(&alpha_weave, &alpha)
        .expect("release at idle");
    assert!(!h.sched.thread_ticker.contains_key(&alpha));
    assert!(h.sched.weaves[&alpha_weave].references(&alpha));
    assert_eq!(h.sched.weaves[&alpha_weave].ticked_thread_ids().count(), 0);
    assert!(matches!(
        &h.last_record(&alpha_weave).effect,
        PersistedDriverEffect::ReleaseTicker { thread_id } if *thread_id == alpha
    ));

    // Dormant: user input is rejected without failing the thread.
    let conv_before = h.sched.tasks[&alpha].conversation.messages().len();
    let mut pending_io = FuturesUnordered::new();
    h.sched
        .send_user_message(&alpha, "hello?".into(), Vec::new(), &mut pending_io);
    assert_eq!(
        h.sched.tasks[&alpha].conversation.messages().len(),
        conv_before,
        "dormant thread accepted no input"
    );
    assert!(h.sched.tasks[&alpha].is_idle());

    // Adopt from another weave in the same pod: admitted, aux ref added.
    h.sched
        .weave_adopt_ticker(&beta_weave, &alpha)
        .expect("adopt of dormant thread");
    assert_eq!(h.sched.thread_ticker.get(&alpha), Some(&beta_weave));
    let aux = h.sched.weaves[&beta_weave]
        .threads
        .iter()
        .find(|r| r.thread_id == alpha)
        .expect("adopting weave references the thread");
    assert_eq!(aux.role, WeaveThreadRole::Auxiliary);
    assert_eq!(aux.relationship, None);

    // Double-adopt refused; release from a non-ticking weave refused.
    assert!(
        h.sched
            .weave_adopt_ticker(&beta_weave, &alpha)
            .expect_err("already ticked")
            .contains("already ticks")
    );
    assert!(
        h.sched
            .weave_release_ticker(&alpha_weave, &alpha)
            .expect_err("alpha weave no longer ticks it")
            .contains("does not tick")
    );
}

#[tokio::test]
async fn release_refused_while_a_cycle_is_active() {
    let mut h = harness().await;
    let alpha = h.create_thread();
    let weave_id = h.weave_of(&alpha);

    h.sched
        .tasks
        .get_mut(&alpha)
        .unwrap()
        .submit_user_message("go".into(), Vec::new(), Vec::new());

    let err = h
        .sched
        .weave_release_ticker(&weave_id, &alpha)
        .expect_err("queued turn blocks release");
    assert!(err.contains("active cycle"));
    assert!(matches!(
        &h.last_record(&weave_id).outcome,
        DriverEffectOutcome::Failed { .. }
    ));
    assert_eq!(
        h.sched.thread_ticker.get(&alpha),
        Some(&weave_id),
        "ticker unchanged after refused release"
    );
}

// ---------- sweep / teardown ----------

#[tokio::test]
async fn sweep_unreferences_from_every_weave_and_retires_emptied_ones() {
    let mut h = harness().await;
    let primary = h.create_thread();
    let weave_id = h.weave_of(&primary);
    let derived = h.derive(&weave_id, Vec::new(), "check").unwrap();

    // Sweep the derived thread: the weave survives on its primary ref.
    h.sched
        .sweep_thread(&derived, TEST_POD, RetentionAction::Delete);
    assert!(!h.sched.tasks.contains_key(&derived));
    assert!(!h.sched.thread_ticker.contains_key(&derived));
    let weave = &h.sched.weaves[&weave_id];
    assert!(!weave.references(&derived));
    assert!(weave.references(&primary));

    // Sweep the primary: the weave empties and retires.
    h.sched
        .sweep_thread(&primary, TEST_POD, RetentionAction::Delete);
    assert!(!h.sched.weaves.contains_key(&weave_id));
    assert!(!h.sched.dirty_weaves.contains(&weave_id));
    assert!(h.sched.thread_ticker.is_empty());
}

// ---------- scripted drivers ----------

#[tokio::test]
async fn scripted_driver_must_load_at_creation() {
    let mut h = harness().await;
    let err = h
        .create_scripted_thread("no_such_driver")
        .expect_err("missing program rejects creation");
    assert!(err.contains("unreadable"), "{err}");

    let err = h
        .create_scripted_thread("../escape")
        .expect_err("path traversal rejected");
    assert!(err.contains("invalid driver program name"), "{err}");
}

#[tokio::test]
async fn scripted_driver_error_fails_the_thread_not_the_scheduler() {
    let mut h = harness().await;
    h.install_driver("broken", "function on_event(s, e) error('boom') end");
    let thread = h.create_scripted_thread("broken").unwrap();

    let mut pending_io = FuturesUnordered::new();
    h.sched
        .send_user_message(&thread, "hello".into(), Vec::new(), &mut pending_io);
    h.sched.step_until_blocked(&thread, &mut pending_io);

    assert!(matches!(
        h.internal_of(&thread),
        ThreadInternalState::Failed { .. }
    ));
    assert!(
        h.sched.tasks[&thread]
            .failure_detail()
            .unwrap_or_default()
            .contains("boom")
    );
}

#[tokio::test]
async fn checker_driver_denies_tools_and_the_cycle_continues() {
    let mut h = harness().await;
    h.install_driver("auto_mode_checker", CHECKER_DRIVER);
    let primary = h.create_scripted_thread("auto_mode_checker").unwrap();
    let weave_id = h.weave_of(&primary);
    let mut pending_io = FuturesUnordered::new();

    // A weave subscriber (step 7b): receives the full snapshot on each
    // refresh. Before any activation, the cache is empty and the wire
    // snapshot degrades to the degenerate primary-transcript form.
    let (weave_tx, mut weave_rx) = tokio::sync::mpsc::unbounded_channel();
    h.sched.router.register_client(77, weave_tx);
    h.sched.router.subscribe_weave(77, &weave_id);
    assert_eq!(
        h.sched.weaves[&weave_id].wire_snapshot().presentation,
        vec![PresentationBlock::PrimaryTranscript {
            thread_id: primary.clone()
        }]
    );

    // Turn 1: the driver runs the primary.
    h.sched.send_user_message(
        &primary,
        "please delete everything".into(),
        Vec::new(),
        &mut pending_io,
    );
    h.sched.step_until_blocked(&primary, &mut pending_io);
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::AwaitingModel { .. }
    ));

    // The model requests a tool — the driver must intercept: primary
    // parks at its boundary, a checker thread is derived (custom
    // prompt, no tools) and set running.
    h.respond_model(
        &primary,
        vec![
            text_block("I'll delete it now."),
            tool_use_block("toolu-1", "delete_everything"),
        ],
        &mut pending_io,
    );
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::AgentBoundary { .. }
    ));
    let weave = &h.sched.weaves[&weave_id];
    assert_eq!(weave.threads.len(), 2, "checker referenced by the weave");
    let checker = weave
        .threads
        .iter()
        .find(|r| r.thread_id != primary)
        .expect("checker ref")
        .thread_id
        .clone();
    assert_eq!(
        weave
            .threads
            .iter()
            .find(|r| r.thread_id == checker)
            .unwrap()
            .relationship
            .as_ref()
            .unwrap()
            .kind,
        "check"
    );
    assert!(matches!(
        h.internal_of(&checker),
        ThreadInternalState::AwaitingModel { .. }
    ));
    let checker_conv = h.sched.tasks[&checker].conversation.messages();
    assert!(
        checker_conv[0].content.iter().any(
            |b| matches!(b, ContentBlock::Text { text } if text.contains("permission checker"))
        ),
        "checker got its custom system prompt"
    );
    assert!(
        checker_conv
            .last()
            .unwrap()
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::Text { text } if text.contains("toolu-1"))),
        "checker was asked about the requested call"
    );

    // Mid-check presentation: the driver's present(state) composed the
    // head, a status line, and the in-flight checker — validated,
    // cached on the weave, and pushed to the weave subscriber.
    let mid_check = vec![
        PresentationBlock::PrimaryTranscript {
            thread_id: primary.clone(),
        },
        PresentationBlock::Status {
            text: "permission check in flight (1 tool calls)".into(),
        },
        PresentationBlock::ThreadList {
            thread_ids: vec![checker.clone()],
        },
    ];
    assert_eq!(h.sched.weaves[&weave_id].presentation, mid_check);
    let mut pushed = None;
    while let Ok(event) = weave_rx.try_recv() {
        if let ServerToClient::WeaveSnapshot { snapshot, .. } = event {
            pushed = Some(snapshot);
        }
    }
    let pushed = pushed.expect("subscriber got a weave snapshot");
    assert_eq!(pushed.presentation, mid_check);
    assert_eq!(pushed.threads.len(), 2);

    // List-tier decoration: both threads carry the weave id; the
    // derived checker is tagged auxiliary.
    let decorated = h.sched.decorate_summary(h.sched.tasks[&checker].summary());
    assert_eq!(decorated.weave_id.as_deref(), Some(weave_id.as_str()));
    assert_eq!(
        decorated.weave_role,
        Some(whisper_agent_protocol::weave::WeaveThreadRole::Auxiliary)
    );
    let decorated = h.sched.decorate_summary(h.sched.tasks[&primary].summary());
    assert_eq!(decorated.weave_id.as_deref(), Some(weave_id.as_str()));
    assert_eq!(
        decorated.weave_role,
        Some(whisper_agent_protocol::weave::WeaveThreadRole::Primary)
    );

    // Typing into the checker (its drill-down view is one click away
    // via the weave strip) is refused at the input path: auxiliary
    // threads of a scripted weave aren't externally writable. The
    // check stays in flight, the primary stays parked, and no journal
    // record was interrupted.
    h.sched
        .send_user_message(&checker, "let me help".into(), Vec::new(), &mut pending_io);
    assert!(matches!(
        h.internal_of(&checker),
        ThreadInternalState::AwaitingModel { .. }
    ));
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::AgentBoundary { .. }
    ));

    // The checker denies. The checker finishes; the primary's tool
    // request is closed with a synthesized error result and the cycle
    // continues into turn 2 without executing anything.
    h.respond_model(&checker, vec![text_block("DENY")], &mut pending_io);
    assert!(matches!(
        h.internal_of(&checker),
        ThreadInternalState::Completed
    ));
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::AwaitingModel { .. }
    ));
    let primary_conv = h.sched.tasks[&primary].conversation.messages();
    assert!(
        primary_conv
            .iter()
            .any(|m| m.content.iter().any(|b| matches!(
                b,
                ContentBlock::ToolResult { tool_use_id, is_error: true, .. }
                    if tool_use_id == "toolu-1"
            ))),
        "denied call closed with an error tool_result"
    );
    let journal = h.sched.weaves[&weave_id].effect_journal.records();
    assert!(journal.iter().any(|r| matches!(
        &r.effect,
        PersistedDriverEffect::ResolveTools { approved, denied, .. }
            if approved.is_empty() && denied == &vec!["toolu-1".to_string()]
    )));
    assert!(
        journal
            .iter()
            .any(|r| matches!(&r.effect, PersistedDriverEffect::DeriveThread { .. }))
    );

    // Check resolved: the display drops the finished checker and
    // returns to the bare head (the checker stays in the drill-down
    // refs — curate, never conceal).
    let after_check = h.sched.weaves[&weave_id].wire_snapshot();
    assert_eq!(
        after_check.presentation,
        vec![PresentationBlock::PrimaryTranscript {
            thread_id: primary.clone()
        }]
    );
    assert!(
        after_check.threads.iter().any(|r| r.thread_id == checker),
        "finished checker still reachable via drill-down"
    );

    // Turn 2 responds without tools: the driver finishes the cycle.
    h.respond_model(
        &primary,
        vec![text_block("Understood, I won't.")],
        &mut pending_io,
    );
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::Completed
    ));
    assert_eq!(
        h.sched.weaves[&weave_id].driver_program_hash.as_deref(),
        Some(crate::runtime::driver::lua::program_hash(CHECKER_DRIVER).as_str()),
        "weave snapshots the program it ran"
    );
}

#[tokio::test]
async fn checker_driver_allows_tools_and_they_dispatch() {
    let mut h = harness().await;
    h.install_driver("auto_mode_checker", CHECKER_DRIVER);
    let primary = h.create_scripted_thread("auto_mode_checker").unwrap();
    let weave_id = h.weave_of(&primary);
    let mut pending_io = FuturesUnordered::new();

    h.sched
        .send_user_message(&primary, "look around".into(), Vec::new(), &mut pending_io);
    h.sched.step_until_blocked(&primary, &mut pending_io);
    h.respond_model(
        &primary,
        vec![tool_use_block("toolu-9", "list_images")],
        &mut pending_io,
    );
    let checker = h.sched.weaves[&weave_id]
        .threads
        .iter()
        .find(|r| r.thread_id != primary)
        .expect("checker derived")
        .thread_id
        .clone();

    h.respond_model(&checker, vec![text_block("ALLOW ALL")], &mut pending_io);

    // Approved: the primary is executing (or has executed) its tool
    // batch — not failed, not denied.
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::AwaitingTools { .. } | ThreadInternalState::AwaitingModel { .. }
    ));
    let primary_conv = h.sched.tasks[&primary].conversation.messages();
    assert!(
        !primary_conv
            .iter()
            .any(|m| m.content.iter().any(|b| matches!(
                b,
                ContentBlock::ToolResult { content, .. }
                    if format!("{content:?}").contains("denied by permission checker")
            ))),
        "no denial result on the allow path"
    );
    let journal = h.sched.weaves[&weave_id].effect_journal.records();
    assert!(journal.iter().any(|r| matches!(
        &r.effect,
        PersistedDriverEffect::ResolveTools { approved, denied, .. }
            if approved == &vec!["toolu-9".to_string()] && denied.is_empty()
    )));
}

// ---------- step 8: advance_head as a weave primitive ----------

/// `advance_head` executor admissions: promotion works for a
/// self-ticked reference and refuses (journaling the refusal) for
/// unreferenced threads, the current primary, and a mid-cycle primary.
#[tokio::test]
async fn advance_head_promotes_and_refuses_with_journaled_records() {
    let mut h = harness().await;
    let t1 = h.create_thread();
    let weave_id = h.weave_of(&t1);
    let aux = h
        .derive(&weave_id, Vec::new(), "helper")
        .expect("derive an auxiliary");

    // Refusal: unreferenced thread.
    let err = h
        .sched
        .weave_advance_head(&weave_id, "task-not-referenced")
        .unwrap_err();
    assert!(err.contains("does not reference"), "got: {err}");
    assert!(matches!(
        &h.last_record(&weave_id).outcome,
        DriverEffectOutcome::Failed { message } if message.contains("does not reference")
    ));

    // Refusal: mid-cycle primary. Park t1 in an active turn first.
    let mut pending_io = FuturesUnordered::new();
    h.sched
        .send_user_message(&t1, "working...".into(), Vec::new(), &mut pending_io);
    h.sched.step_until_blocked(&t1, &mut pending_io);
    let err = h.sched.weave_advance_head(&weave_id, &aux).unwrap_err();
    assert!(err.contains("active cycle"), "got: {err}");
    h.respond_model(&t1, vec![text_block("done")], &mut pending_io);

    // Promotion: aux becomes the head, t1 a dormant auxiliary.
    h.sched
        .weave_advance_head(&weave_id, &aux)
        .expect("idle primary advances");
    let weave = &h.sched.weaves[&weave_id];
    assert_eq!(weave.primary_thread_id(), Some(aux.as_str()));
    let old_ref = weave.threads.iter().find(|r| r.thread_id == t1).unwrap();
    assert_eq!(old_ref.role, WeaveThreadRole::Auxiliary);
    assert!(!old_ref.ticks);
    assert!(!h.sched.thread_ticker.contains_key(&t1));
    assert_eq!(h.sched.thread_ticker.get(&aux), Some(&weave_id));

    // Refusal: promoting the current primary is a driver bug.
    let err = h.sched.weave_advance_head(&weave_id, &aux).unwrap_err();
    assert!(err.contains("already the primary"), "got: {err}");
}

/// A scripted driver composes a compaction-shaped roll from primitives:
/// finish the cycle, derive along a `compaction` edge, advance the head.
#[tokio::test]
async fn scripted_driver_advances_head_via_effect() {
    const ROLLER: &str = r#"
function on_event(state, event)
  local k = event.kind
  if k == "turn_start" then
    return { effects = { { kind = "run_agent", thread_id = event.thread_id } }, state = state }
  end
  if k == "agent_completed" then
    if state.rolled then
      return { effects = { { kind = "finish_cycle", thread_id = event.thread_id } }, state = state }
    end
    state.rolled = true
    return { effects = {
      { kind = "finish_cycle", thread_id = event.thread_id },
      { kind = "derive_thread", relationship = "compaction", source_thread_id = event.thread_id },
    }, state = state }
  end
  if k == "thread_derived" then
    return { effects = { { kind = "advance_head", thread_id = event.thread_id } }, state = state }
  end
  return { state = state }
end
"#;
    let mut h = harness().await;
    h.install_driver("roller", ROLLER);
    let t1 = h.create_scripted_thread("roller").unwrap();
    let weave_id = h.weave_of(&t1);
    let mut pending_io = FuturesUnordered::new();

    h.sched
        .send_user_message(&t1, "roll me".into(), Vec::new(), &mut pending_io);
    h.sched.step_until_blocked(&t1, &mut pending_io);
    h.respond_model(
        &t1,
        vec![text_block("summary of everything")],
        &mut pending_io,
    );

    let weave = &h.sched.weaves[&weave_id];
    assert_eq!(weave.threads.len(), 2);
    let new_head = weave
        .primary_thread_id()
        .expect("weave has a promoted head")
        .to_string();
    assert_ne!(new_head, t1);
    let old_ref = weave.threads.iter().find(|r| r.thread_id == t1).unwrap();
    assert_eq!(old_ref.role, WeaveThreadRole::Auxiliary);
    assert!(
        !old_ref.ticks,
        "scripted roll also leaves the old head dormant"
    );
    assert_eq!(h.sched.thread_ticker.get(&new_head), Some(&weave_id));
    assert!(
        h.sched.weaves[&weave_id]
            .effect_journal
            .records()
            .iter()
            .any(|r| r.outcome == DriverEffectOutcome::Completed
                && matches!(
                    &r.effect,
                    PersistedDriverEffect::AdvanceHead { thread_id, previous: Some(p) }
                        if thread_id == &new_head && p == &t1
                ))
    );
}

// ---------- multi-agent enablers: participant selection + concurrency ----------

/// Two model participants alternate in ONE thread: the driver runs the
/// default responder, then (on its completion) finishes the cycle and
/// runs the `critic` participant on the same transcript. The
/// participant-selectable `run_agent` is the single-thread
/// multi-voice conversation primitive.
#[tokio::test]
async fn scripted_driver_alternates_participants_in_one_thread() {
    const DIALOGUE: &str = r#"
function on_event(state, event)
  local k = event.kind
  if k == "turn_start" then
    return { effects = { { kind = "run_agent", thread_id = event.thread_id } }, state = state }
  end
  if k == "agent_completed" then
    if event.participant_id == "agent" then
      return { effects = {
        { kind = "finish_cycle", thread_id = event.thread_id },
        { kind = "run_agent", thread_id = event.thread_id, participant = "critic" },
      }, state = state }
    end
    return { effects = { { kind = "finish_cycle", thread_id = event.thread_id } }, state = state }
  end
  return { state = state }
end
"#;
    use whisper_agent_protocol::{
        ParticipantExecutionProfileRequest, ParticipantId, SystemPromptChoice, ThreadParticipant,
        ThreadParticipantKind, ThreadParticipants,
    };
    let mut h = harness().await;
    h.install_driver("dialogue", DIALOGUE);

    let mut participants = ThreadParticipants::single_agent();
    participants.members.push(ThreadParticipant {
        id: "critic".into(),
        kind: ThreadParticipantKind::Model,
        display_name: Some("Critic".into()),
    });
    let mut profiles = std::collections::BTreeMap::new();
    profiles.insert(
        ParticipantId::new("critic"),
        ParticipantExecutionProfileRequest {
            system_prompt: Some(SystemPromptChoice::Text {
                text: "You are a harsh critic.".into(),
            }),
            ..Default::default()
        },
    );
    let mut pending_io = FuturesUnordered::new();
    let t1 = h
        .sched
        .create_task(
            None,
            None,
            None,
            Some(ThreadConfigOverride {
                driver: Some(ThreadDriverConfig::Scripted {
                    name: "dialogue".into(),
                    config: Default::default(),
                }),
                participants: Some(participants),
                participant_profiles: Some(profiles),
                ..Default::default()
            }),
            None,
            None,
            None,
            None,
            None,
            None,
            &mut pending_io,
        )
        .expect("multi-participant scripted thread creates");
    let weave_id = h.weave_of(&t1);

    h.sched
        .send_user_message(&t1, "draft something".into(), Vec::new(), &mut pending_io);
    h.sched.step_until_blocked(&t1, &mut pending_io);
    match h.internal_of(&t1) {
        ThreadInternalState::AwaitingModel { generation, .. } => {
            assert_eq!(generation.participant_id.as_str(), "agent");
        }
        other => panic!("expected default responder in flight, got {other:?}"),
    }

    h.respond_model(&t1, vec![text_block("here is my draft")], &mut pending_io);
    // The driver finished the agent's cycle and started the critic on
    // the same thread.
    match h.internal_of(&t1) {
        ThreadInternalState::AwaitingModel { generation, .. } => {
            assert_eq!(generation.participant_id.as_str(), "critic");
        }
        other => panic!("expected critic turn in flight, got {other:?}"),
    }

    h.respond_model(&t1, vec![text_block("the draft is weak")], &mut pending_io);
    assert!(matches!(h.internal_of(&t1), ThreadInternalState::Completed));

    // Journal explains both voices: RunAgent records carry the two
    // participant ids, in order, both completed.
    let voices: Vec<String> = h.sched.weaves[&weave_id]
        .effect_journal
        .records()
        .iter()
        .filter_map(|r| match &r.effect {
            PersistedDriverEffect::RunAgent { generation, .. }
                if r.outcome == DriverEffectOutcome::Completed =>
            {
                Some(generation.participant_id.to_string())
            }
            _ => None,
        })
        .collect();
    assert_eq!(voices, vec!["agent".to_string(), "critic".to_string()]);
}

/// An unregistered participant id is refused at admission and the
/// refusal journals — profile resolution silently falls back for
/// unknown ids, so a wrong-voice driver bug must not silently run.
#[tokio::test]
async fn run_agent_refuses_unregistered_participant() {
    const GHOST: &str = r#"
function on_event(state, event)
  if event.kind == "turn_start" then
    return { effects = { { kind = "run_agent", thread_id = event.thread_id, participant = "ghost" } }, state = state }
  end
  return { state = state }
end
"#;
    let mut h = harness().await;
    h.install_driver("ghost", GHOST);
    let t1 = h.create_scripted_thread("ghost").unwrap();
    let weave_id = h.weave_of(&t1);
    let mut pending_io = FuturesUnordered::new();

    h.sched
        .send_user_message(&t1, "speak".into(), Vec::new(), &mut pending_io);
    h.sched.step_until_blocked(&t1, &mut pending_io);

    assert!(
        h.sched.weaves[&weave_id]
            .effect_journal
            .records()
            .iter()
            .any(|r| matches!(
                &r.outcome,
                DriverEffectOutcome::Failed { message } if message.contains("not registered")
            )),
        "refusal must journal"
    );
}

/// One weave drives two derived worker threads whose model calls are in
/// flight SIMULTANEOUSLY, and their out-of-order completions resolve
/// the right journal records — the tangled-threads multi-agent
/// substrate.
#[tokio::test]
async fn weave_runs_two_worker_threads_concurrently() {
    const FANOUT: &str = r#"
function on_event(state, event)
  local k = event.kind
  if k == "input_accepted" then
    return { effects = {
      { kind = "derive_thread", relationship = "worker",
        seed = { { author = "user", text = "task A" } }, source_thread_id = event.thread_id },
      { kind = "derive_thread", relationship = "worker",
        seed = { { author = "user", text = "task B" } }, source_thread_id = event.thread_id },
    }, state = state }
  end
  if k == "thread_derived" then
    return { effects = { { kind = "run_agent", thread_id = event.thread_id } }, state = state }
  end
  if k == "agent_completed" then
    return { effects = { { kind = "finish_cycle", thread_id = event.thread_id } }, state = state }
  end
  return { state = state }
end
"#;
    let mut h = harness().await;
    h.install_driver("fanout", FANOUT);
    let primary = h.create_scripted_thread("fanout").unwrap();
    let weave_id = h.weave_of(&primary);
    let mut pending_io = FuturesUnordered::new();

    h.sched
        .send_user_message(&primary, "fan out".into(), Vec::new(), &mut pending_io);
    h.sched.step_until_blocked(&primary, &mut pending_io);

    let workers: Vec<String> = h.sched.weaves[&weave_id]
        .threads
        .iter()
        .filter(|r| r.thread_id != primary)
        .map(|r| r.thread_id.clone())
        .collect();
    assert_eq!(workers.len(), 2, "two workers derived");
    // The concurrency claim itself: both workers' model calls are in
    // flight at the same time.
    for worker in &workers {
        assert!(
            matches!(
                h.internal_of(worker),
                ThreadInternalState::AwaitingModel { .. }
            ),
            "worker `{worker}` should be awaiting its model call"
        );
    }

    // Complete them OUT OF ORDER (B first) — each completion must
    // resolve its own journal record and cycle, untouched by the
    // other's in-flight state.
    h.respond_model(&workers[1], vec![text_block("done B")], &mut pending_io);
    assert!(matches!(
        h.internal_of(&workers[1]),
        ThreadInternalState::Completed
    ));
    assert!(
        matches!(
            h.internal_of(&workers[0]),
            ThreadInternalState::AwaitingModel { .. }
        ),
        "worker A must stay in flight while B completes"
    );
    h.respond_model(&workers[0], vec![text_block("done A")], &mut pending_io);
    assert!(matches!(
        h.internal_of(&workers[0]),
        ThreadInternalState::Completed
    ));

    let records = h.sched.weaves[&weave_id].effect_journal.records();
    let run_agents: Vec<_> = records
        .iter()
        .filter(|r| matches!(&r.effect, PersistedDriverEffect::RunAgent { .. }))
        .collect();
    assert_eq!(run_agents.len(), 2);
    assert!(
        run_agents
            .iter()
            .all(|r| r.outcome == DriverEffectOutcome::Completed),
        "both RunAgent records resolve precisely despite interleaving"
    );
}

/// Journal precision (the step-8 gap, retired): cancelling one worker
/// of a multi-thread weave interrupts only THAT thread's pending
/// record; the sibling's in-flight record stays pending and resolves
/// Completed when its own model call lands.
#[tokio::test]
async fn interrupting_one_worker_leaves_the_siblings_records_pending() {
    const FANOUT: &str = r#"
function on_event(state, event)
  local k = event.kind
  if k == "input_accepted" then
    return { effects = {
      { kind = "derive_thread", relationship = "worker",
        seed = { { author = "user", text = "task A" } }, source_thread_id = event.thread_id },
      { kind = "derive_thread", relationship = "worker",
        seed = { { author = "user", text = "task B" } }, source_thread_id = event.thread_id },
    }, state = state }
  end
  if k == "thread_derived" then
    return { effects = { { kind = "run_agent", thread_id = event.thread_id } }, state = state }
  end
  if k == "agent_completed" then
    return { effects = { { kind = "finish_cycle", thread_id = event.thread_id } }, state = state }
  end
  return { state = state }
end
"#;
    let mut h = harness().await;
    h.install_driver("fanout2", FANOUT);
    let primary = h.create_scripted_thread("fanout2").unwrap();
    let weave_id = h.weave_of(&primary);
    let mut pending_io = FuturesUnordered::new();

    h.sched
        .send_user_message(&primary, "fan out".into(), Vec::new(), &mut pending_io);
    h.sched.step_until_blocked(&primary, &mut pending_io);

    let workers: Vec<String> = h.sched.weaves[&weave_id]
        .threads
        .iter()
        .filter(|r| r.thread_id != primary)
        .map(|r| r.thread_id.clone())
        .collect();
    assert_eq!(workers.len(), 2);

    // Cancel worker B mid-flight.
    h.sched.execute_cancel_thread(&workers[1], &mut pending_io);

    let outcome_for = |h: &Harness, thread: &str| -> Vec<DriverEffectOutcome> {
        h.sched.weaves[&weave_id]
            .effect_journal
            .records()
            .iter()
            .filter(|r| {
                matches!(
                    &r.effect,
                    PersistedDriverEffect::RunAgent { thread_id, .. } if thread_id == thread
                )
            })
            .map(|r| r.outcome.clone())
            .collect()
    };
    assert!(
        matches!(
            outcome_for(&h, &workers[1]).as_slice(),
            [DriverEffectOutcome::Interrupted { .. }]
        ),
        "cancelled worker's record interrupted"
    );
    assert!(
        matches!(
            outcome_for(&h, &workers[0]).as_slice(),
            [DriverEffectOutcome::Pending]
        ),
        "sibling's in-flight record must stay pending — the old bulk \
         interrupt clobbered it"
    );

    // The survivor completes normally and resolves its own record.
    h.respond_model(&workers[0], vec![text_block("done A")], &mut pending_io);
    assert!(matches!(
        outcome_for(&h, &workers[0]).as_slice(),
        [DriverEffectOutcome::Completed]
    ));
}

/// Conversation entries of a thread as (author, text) pairs —
/// setup-prefix messages (system prompt, tool manifest) are authored
/// by the thread's own responder and aren't part of the transcript
/// proper, so only User/Assistant roles count.
fn authored_tail(h: &Harness, thread: &str) -> Vec<(String, String)> {
    h.sched.tasks[thread]
        .conversation
        .messages()
        .iter()
        .filter(|m| {
            matches!(
                m.role,
                whisper_agent_protocol::Role::User | whisper_agent_protocol::Role::Assistant
            )
        })
        .filter_map(|m| match m.content.first() {
            Some(ContentBlock::Text { text }) => {
                Some((m.effective_author().as_str().to_string(), text.clone()))
            }
            _ => None,
        })
        .collect()
}

/// The shipped roundtable driver (migration step 9, tangled-threads
/// shape): the primary is a driver-maintained minutes view, each voice
/// is a derived private-context thread, and one accepted input runs a
/// one-pass round in CAST order — later voices hear earlier replies
/// from the same round, every reply lands back in the minutes
/// attributed to its speaker, and the round closes the primary's
/// cycle. A second input reuses the derived cast.
#[tokio::test]
async fn roundtable_driver_runs_one_pass_rounds() {
    let mut h = harness().await;
    h.install_driver("roundtable", ROUNDTABLE_DRIVER);
    let primary = h.create_scripted_thread("roundtable").unwrap();
    let weave_id = h.weave_of(&primary);
    let mut pending_io = FuturesUnordered::new();

    h.sched.send_user_message(
        &primary,
        "Should we rewrite it in Rust?".into(),
        Vec::new(),
        &mut pending_io,
    );
    h.sched.step_until_blocked(&primary, &mut pending_io);

    // The cast derived with voice-tagged relationship kinds.
    let voice_thread = |h: &Harness, id: &str| -> String {
        let kind = format!("voice:{id}");
        h.sched.weaves[&weave_id]
            .threads
            .iter()
            .find(|r| r.relationship.as_ref().is_some_and(|rel| rel.kind == kind))
            .unwrap_or_else(|| panic!("voice `{id}` derived"))
            .thread_id
            .clone()
    };
    let optimist = voice_thread(&h, "optimist");
    let skeptic = voice_thread(&h, "skeptic");

    // First voice holds the floor; the second waits; the primary is
    // parked open at its input boundary (no model runs on the minutes).
    assert!(matches!(
        h.internal_of(&optimist),
        ThreadInternalState::AwaitingModel { .. }
    ));
    assert!(matches!(h.internal_of(&skeptic), ThreadInternalState::Idle));
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::NeedsModelCall
    ));

    // Both voices heard the user's words as genuine user input, after
    // their own (distinct) setup prefixes.
    for voice in [&optimist, &skeptic] {
        let last = h.sched.tasks[voice.as_str()]
            .conversation
            .messages()
            .last()
            .unwrap()
            .clone();
        assert_eq!(
            last.effective_author().as_str(),
            whisper_agent_protocol::DEFAULT_INPUT_PARTICIPANT_ID
        );
        assert!(matches!(
            &last.content[0],
            ContentBlock::Text { text } if text == "Should we rewrite it in Rust?"
        ));
    }

    // Mid-round presentation names the floor-holder.
    assert!(
        h.sched.weaves[&weave_id].presentation.iter().any(|block| {
            matches!(block, PresentationBlock::Status { text } if text.contains("optimist"))
        }),
        "status block should say optimist is speaking"
    );

    // The optimist speaks: its reply lands in the minutes and in the
    // skeptic's context (both attributed), and the floor passes.
    h.respond_model(
        &optimist,
        vec![text_block("Do it - the wins compound.")],
        &mut pending_io,
    );
    assert!(matches!(
        h.internal_of(&optimist),
        ThreadInternalState::Completed
    ));
    assert!(matches!(
        h.internal_of(&skeptic),
        ThreadInternalState::AwaitingModel { .. }
    ));
    assert!(
        authored_tail(&h, &primary)
            .iter()
            .any(|(author, text)| author == "optimist" && text.contains("wins compound")),
        "minutes carries the optimist's reply attributed to it"
    );
    assert!(
        authored_tail(&h, &skeptic)
            .iter()
            .any(|(author, text)| author == "optimist" && text.contains("wins compound")),
        "skeptic hears the optimist before speaking"
    );

    // The skeptic closes the round: reply pollinated to the minutes
    // and the (idle, un-woken) optimist, and the primary's cycle ends.
    h.respond_model(
        &skeptic,
        vec![text_block("Migration cost says otherwise.")],
        &mut pending_io,
    );
    assert!(matches!(
        h.internal_of(&skeptic),
        ThreadInternalState::Completed
    ));
    assert!(matches!(
        h.internal_of(&optimist),
        ThreadInternalState::Completed
    ));
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::Completed
    ));
    let minutes = authored_tail(&h, &primary);
    assert_eq!(
        minutes
            .iter()
            .map(|(author, _)| author.as_str())
            .collect::<Vec<_>>(),
        vec!["user", "optimist", "skeptic"],
        "minutes reads as the full round in speaking order"
    );
    assert!(
        authored_tail(&h, &optimist)
            .iter()
            .any(|(author, _)| author == "skeptic"),
        "optimist's context caught the skeptic's reply for next round"
    );

    // Journal: 3 derives (2 voices + the round-close title thread), 2
    // completed runs (the voice turns — the title turn is in flight,
    // its record still Pending), 3 finishes (2 voices + the primary's
    // round close), 6 appends (input to 2 voices, then each reply to
    // minutes + the other voice).
    let count = |pred: &dyn Fn(&PersistedDriverEffect) -> bool| -> usize {
        h.sched.weaves[&weave_id]
            .effect_journal
            .records()
            .iter()
            .filter(|r| pred(&r.effect) && r.outcome == DriverEffectOutcome::Completed)
            .count()
    };
    assert_eq!(
        count(&|e| matches!(e, PersistedDriverEffect::DeriveThread { .. })),
        3
    );
    assert_eq!(
        count(&|e| matches!(e, PersistedDriverEffect::RunAgent { .. })),
        2
    );
    assert_eq!(
        count(&|e| matches!(e, PersistedDriverEffect::Finish { .. })),
        3
    );
    assert_eq!(
        count(&|e| matches!(e, PersistedDriverEffect::AppendEntry { .. })),
        6
    );

    // The round close derived the title thread (step 11): the
    // truncation placeholder stands until the title model replies,
    // then set_title overwrites it with the cleaned text.
    let title_thread = h.sched.weaves[&weave_id]
        .threads
        .iter()
        .find(|r| {
            r.relationship
                .as_ref()
                .is_some_and(|rel| rel.kind == "title")
        })
        .expect("title thread derived at round close")
        .thread_id
        .clone();
    assert_eq!(
        h.sched.tasks[&primary].title.as_deref(),
        Some("Should we rewrite it in Rust?"),
        "truncation placeholder before the title model replies"
    );
    h.respond_model(
        &title_thread,
        vec![text_block("  \"Rust Rewrite Debate.\"  ")],
        &mut pending_io,
    );
    assert_eq!(
        h.sched.tasks[&primary].title.as_deref(),
        Some("Rust Rewrite Debate"),
        "model title cleaned (unquoted, no trailing period) onto the minutes"
    );
    assert!(matches!(
        h.internal_of(&title_thread),
        ThreadInternalState::Completed
    ));

    // Round two reuses the cast: no new derives, same rotation.
    h.sched.send_user_message(
        &primary,
        "What about the team?".into(),
        Vec::new(),
        &mut pending_io,
    );
    h.sched.step_until_blocked(&primary, &mut pending_io);
    assert_eq!(
        h.sched.weaves[&weave_id].threads.len(),
        4,
        "second round derives no new threads (primary, 2 voices, lingering title)"
    );
    assert!(matches!(
        h.internal_of(&optimist),
        ThreadInternalState::AwaitingModel { .. }
    ));
    h.respond_model(
        &optimist,
        vec![text_block("They will learn.")],
        &mut pending_io,
    );
    h.respond_model(
        &skeptic,
        vec![text_block("Six months, minimum.")],
        &mut pending_io,
    );
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::Completed
    ));
    assert_eq!(
        authored_tail(&h, &primary)
            .iter()
            .map(|(author, _)| author.as_str())
            .collect::<Vec<_>>(),
        vec!["user", "optimist", "skeptic", "user", "optimist", "skeptic"]
    );
}

/// A voice dying mid-round must not wedge the roundtable: the
/// `thread_failed` event advances the floor past the dead voice, the
/// round completes with the survivors, and the next round derives a
/// fresh replacement (a Failed thread refuses `run_agent`). Also pins
/// the stacked-input path: input sent mid-round queues and runs as a
/// back-to-back round before the primary's single cycle close. Fails
/// against the pre-`thread_failed` vocabulary — the driver waited
/// forever on the dead voice and swallowed all future input.
#[tokio::test]
async fn roundtable_survives_voice_failure_and_stacked_input() {
    let mut h = harness().await;
    h.install_driver("roundtable", ROUNDTABLE_DRIVER);
    let primary = h.create_scripted_thread("roundtable").unwrap();
    let weave_id = h.weave_of(&primary);
    let mut pending_io = FuturesUnordered::new();

    h.sched
        .send_user_message(&primary, "round one".into(), Vec::new(), &mut pending_io);
    h.sched.step_until_blocked(&primary, &mut pending_io);

    let voice_threads = |h: &Harness, id: &str| -> Vec<String> {
        let kind = format!("voice:{id}");
        h.sched.weaves[&weave_id]
            .threads
            .iter()
            .filter(|r| r.relationship.as_ref().is_some_and(|rel| rel.kind == kind))
            .map(|r| r.thread_id.clone())
            .collect()
    };
    let optimist1 = voice_threads(&h, "optimist")[0].clone();
    let skeptic = voice_threads(&h, "skeptic")[0].clone();
    assert!(matches!(
        h.internal_of(&optimist1),
        ThreadInternalState::AwaitingModel { .. }
    ));

    // The optimist's backend dies mid-turn: the floor advances to the
    // skeptic instead of waiting forever, and the round still closes.
    h.fail_model(&optimist1, "backend 500", &mut pending_io);
    assert!(matches!(
        h.internal_of(&optimist1),
        ThreadInternalState::Failed { .. }
    ));
    assert!(matches!(
        h.internal_of(&skeptic),
        ThreadInternalState::AwaitingModel { .. }
    ));
    h.respond_model(
        &skeptic,
        vec![text_block("Alone this round.")],
        &mut pending_io,
    );
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::Completed
    ));

    // Round two: a fresh replacement optimist derives; the dead
    // thread stays referenced for drill-down but unmapped.
    h.sched
        .send_user_message(&primary, "round two".into(), Vec::new(), &mut pending_io);
    h.sched.step_until_blocked(&primary, &mut pending_io);
    assert_eq!(
        h.sched.weaves[&weave_id].threads.len(),
        5,
        "replacement derived; dead voice and round-one title thread still referenced"
    );
    let optimist2 = voice_threads(&h, "optimist")
        .into_iter()
        .find(|t| *t != optimist1)
        .expect("replacement optimist derived");
    assert!(matches!(
        h.internal_of(&optimist2),
        ThreadInternalState::AwaitingModel { .. }
    ));

    // Stacked input mid-round: queues without disturbing the voice in
    // flight.
    h.sched
        .send_user_message(&primary, "round three".into(), Vec::new(), &mut pending_io);
    h.sched.step_until_blocked(&primary, &mut pending_io);
    assert!(matches!(
        h.internal_of(&optimist2),
        ThreadInternalState::AwaitingModel { .. }
    ));

    h.respond_model(&optimist2, vec![text_block("Fresh eyes.")], &mut pending_io);
    h.respond_model(&skeptic, vec![text_block("Costs.")], &mut pending_io);
    // Round two closed and round three began immediately: the
    // primary's cycle spans back-to-back rounds (one finish closes
    // both stacked submissions).
    assert!(
        !matches!(h.internal_of(&primary), ThreadInternalState::Completed),
        "primary's cycle must span back-to-back rounds"
    );
    assert!(matches!(
        h.internal_of(&optimist2),
        ThreadInternalState::AwaitingModel { .. }
    ));
    h.respond_model(
        &optimist2,
        vec![text_block("Still worth it.")],
        &mut pending_io,
    );
    h.respond_model(
        &skeptic,
        vec![text_block("Agreed, cautiously.")],
        &mut pending_io,
    );
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::Completed
    ));

    // Minutes: the dead optimist is absent from round one; the
    // stacked input lands at submit time (before the replies of the
    // round it interrupted).
    assert_eq!(
        authored_tail(&h, &primary)
            .iter()
            .map(|(author, _)| author.as_str())
            .collect::<Vec<_>>(),
        vec![
            "user", "skeptic", "user", "user", "optimist", "skeptic", "optimist", "skeptic"
        ]
    );
    // The replacement's context is fresh — it never saw round one.
    assert!(
        !authored_tail(&h, &optimist2)
            .iter()
            .any(|(_, text)| text.contains("round one") || text.contains("Alone this round")),
        "replacement voice must not inherit the dead voice's history"
    );
}

/// Restart mid-round: the persister heals in-flight threads to Failed
/// and drivers don't run at load, so the persisted driver state still
/// holds the floor assignment. `load_state` collects the dead ticked
/// threads and the weave's first activation replays them as deferred
/// `thread_failed` facts (primary first) ahead of the triggering
/// event: the roundtable voids the stale round, re-derives the dead
/// voice, keeps the surviving voice's context, and the healing input
/// runs a clean round. Fails against the pre-notice behavior — the
/// driver queued all post-restart input forever.
#[tokio::test]
async fn roundtable_recovers_after_restart_mid_round() {
    let mut h = harness().await;
    h.install_driver("roundtable", ROUNDTABLE_DRIVER);
    let primary = h.create_scripted_thread("roundtable").unwrap();
    let weave_id = h.weave_of(&primary);
    let mut pending_io = FuturesUnordered::new();

    h.sched
        .send_user_message(&primary, "round one".into(), Vec::new(), &mut pending_io);
    h.sched.step_until_blocked(&primary, &mut pending_io);

    let voice_threads = |h: &Harness, id: &str| -> Vec<String> {
        let kind = format!("voice:{id}");
        h.sched.weaves[&weave_id]
            .threads
            .iter()
            .filter(|r| r.relationship.as_ref().is_some_and(|rel| rel.kind == kind))
            .map(|r| r.thread_id.clone())
            .collect()
    };
    let optimist1 = voice_threads(&h, "optimist")[0].clone();
    let skeptic = voice_threads(&h, "skeptic")[0].clone();
    assert!(matches!(
        h.internal_of(&optimist1),
        ThreadInternalState::AwaitingModel { .. }
    ));

    // "Restart": apply the persister's in-flight heal to cloned state
    // and load it into a fresh scheduler (same recovery `load_one`
    // performs — see pod/persist.rs).
    let weave = h.sched.weaves[&weave_id].clone();
    let mut threads = Vec::new();
    for tid in [&primary, &optimist1, &skeptic] {
        let mut task = h.sched.tasks[tid.as_str()].clone();
        if task.is_in_flight() {
            let mut events = Vec::new();
            task.heal_to_idle("task was in-flight at last shutdown", &mut events);
            task.fail("resume", "task was in-flight at last shutdown");
        }
        threads.push(task);
    }
    let mut fresh = harness().await;
    fresh.install_driver("roundtable", ROUNDTABLE_DRIVER);
    fresh.sched.load_state(crate::pod::persist::LoadedState {
        pods: Vec::new(),
        threads,
        weaves: vec![weave],
    });
    // The parked primary and the in-flight voice both healed to
    // Failed; the idle voice survived.
    assert!(matches!(
        fresh.internal_of(&primary),
        ThreadInternalState::Failed { .. }
    ));
    assert!(matches!(
        fresh.internal_of(&optimist1),
        ThreadInternalState::Failed { .. }
    ));
    assert!(matches!(
        fresh.internal_of(&skeptic),
        ThreadInternalState::Idle
    ));

    // Fresh input heals the primary; the driver hears the deferred
    // deaths first (round voided, dead voice unmapped), then the
    // input — deriving a replacement optimist and running a clean
    // round with the surviving skeptic.
    let mut pending_io = FuturesUnordered::new();
    fresh
        .sched
        .send_user_message(&primary, "round two".into(), Vec::new(), &mut pending_io);
    fresh.sched.step_until_blocked(&primary, &mut pending_io);

    assert_eq!(
        fresh.sched.weaves[&weave_id].threads.len(),
        4,
        "replacement optimist derived; dead voice still referenced"
    );
    let optimist2 = voice_threads(&fresh, "optimist")
        .into_iter()
        .find(|t| *t != optimist1)
        .expect("replacement optimist derived");
    assert!(matches!(
        fresh.internal_of(&optimist2),
        ThreadInternalState::AwaitingModel { .. }
    ));
    fresh.respond_model(
        &optimist2,
        vec![text_block("Back online.")],
        &mut pending_io,
    );
    assert!(matches!(
        fresh.internal_of(&skeptic),
        ThreadInternalState::AwaitingModel { .. }
    ));
    fresh.respond_model(
        &skeptic,
        vec![text_block("Prove it lasts.")],
        &mut pending_io,
    );
    assert!(matches!(
        fresh.internal_of(&primary),
        ThreadInternalState::Completed
    ));
    // Minutes: round one's input never got replies (the round died
    // with the restart); round two ran cleanly.
    assert_eq!(
        authored_tail(&fresh, &primary)
            .iter()
            .map(|(author, _)| author.as_str())
            .collect::<Vec<_>>(),
        vec!["user", "user", "optimist", "skeptic"]
    );
    // The surviving skeptic kept its context across the restart: it
    // heard round one's input AND round two's.
    let skeptic_tail = authored_tail(&fresh, &skeptic);
    assert!(skeptic_tail.iter().any(|(_, t)| t.contains("round one")));
    assert!(skeptic_tail.iter().any(|(_, t)| t.contains("round two")));
    // A second post-restart round: no re-derivation and no stale
    // re-voiding on later rounds. (The drained-once property itself is
    // pinned by round two above — deliver-on-every-activation would
    // have replayed the primary's death mid-round and voided it.)
    // Ref count note: the title thread derives at the first COMPLETED
    // round's close — that's post-restart "round two"; pre-restart
    // round one was voided before closing.
    let mut pending_io2 = FuturesUnordered::new();
    fresh
        .sched
        .send_user_message(&primary, "round three".into(), Vec::new(), &mut pending_io2);
    fresh.sched.step_until_blocked(&primary, &mut pending_io2);
    assert_eq!(
        fresh.sched.weaves[&weave_id].threads.len(),
        5,
        "no re-derivation on the second post-restart round (title thread from the first completed round included)"
    );
    assert!(matches!(
        fresh.internal_of(&optimist2),
        ThreadInternalState::AwaitingModel { .. }
    ));
}

/// A cancel can reach a thread's JSON while the weave's
/// post-notification driver state does not (threads flush before
/// weaves): load then finds a ticked Cancelled thread the persisted
/// driver still coordinates. Cancelled refuses `run_agent` exactly
/// like Failed, so the dead-at-load rule must treat it as dead —
/// without the notice the stale floor assignment routes every
/// post-restart input into the pending queue forever (or, once the
/// round is otherwise voided, `run_agent` on the mapped corpse fails
/// the weave).
#[tokio::test]
async fn roundtable_restart_notices_a_cancelled_voice() {
    let mut h = harness().await;
    h.install_driver("roundtable", ROUNDTABLE_DRIVER);
    let primary = h.create_scripted_thread("roundtable").unwrap();
    let weave_id = h.weave_of(&primary);
    let mut pending_io = FuturesUnordered::new();

    h.sched
        .send_user_message(&primary, "round one".into(), Vec::new(), &mut pending_io);
    h.sched.step_until_blocked(&primary, &mut pending_io);

    let voice_threads = |h: &Harness, id: &str| -> Vec<String> {
        let kind = format!("voice:{id}");
        h.sched.weaves[&weave_id]
            .threads
            .iter()
            .filter(|r| r.relationship.as_ref().is_some_and(|rel| rel.kind == kind))
            .map(|r| r.thread_id.clone())
            .collect()
    };
    let optimist1 = voice_threads(&h, "optimist")[0].clone();
    let skeptic = voice_threads(&h, "skeptic")[0].clone();

    // Partial-flush window: the weave snapshot predates the cancel
    // (driver state still holds the floor and the mapping), while the
    // thread snapshot carries it.
    let weave = h.sched.weaves[&weave_id].clone();
    let mut threads = Vec::new();
    for tid in [&primary, &optimist1, &skeptic] {
        let mut task = h.sched.tasks[tid.as_str()].clone();
        if *tid == optimist1 {
            let mut events = Vec::new();
            task.cancel(&mut events);
        } else if task.is_in_flight() {
            let mut events = Vec::new();
            task.heal_to_idle("task was in-flight at last shutdown", &mut events);
            task.fail("resume", "task was in-flight at last shutdown");
        }
        threads.push(task);
    }
    let mut fresh = harness().await;
    fresh.install_driver("roundtable", ROUNDTABLE_DRIVER);
    fresh.sched.load_state(crate::pod::persist::LoadedState {
        pods: Vec::new(),
        threads,
        weaves: vec![weave],
    });
    assert!(matches!(
        fresh.internal_of(&optimist1),
        ThreadInternalState::Cancelled
    ));

    // The healing input recovers cleanly: the cancelled voice's death
    // fact unmaps it, a replacement derives, and the round runs.
    let mut pending_io = FuturesUnordered::new();
    fresh
        .sched
        .send_user_message(&primary, "round two".into(), Vec::new(), &mut pending_io);
    fresh.sched.step_until_blocked(&primary, &mut pending_io);
    let optimist2 = voice_threads(&fresh, "optimist")
        .into_iter()
        .find(|t| *t != optimist1)
        .expect("replacement for the cancelled voice derived");
    assert!(matches!(
        fresh.internal_of(&optimist2),
        ThreadInternalState::AwaitingModel { .. }
    ));
    fresh.respond_model(&optimist2, vec![text_block("Recovered.")], &mut pending_io);
    fresh.respond_model(&skeptic, vec![text_block("Noted.")], &mut pending_io);
    assert!(matches!(
        fresh.internal_of(&primary),
        ThreadInternalState::Completed
    ));
}

/// Step-10 knobs, end to end on a purpose-built driver: creation
/// validates submitted values against the program's `describe()`
/// (refusing mismatches with the knob named), freezes the validated map
/// — declaration defaults materialized — onto the weave's driver
/// config, hands it to every activation as `on_event`'s third argument,
/// and a `model` knob's backend flows through `derive_thread` onto the
/// derived thread's bindings.
#[tokio::test]
async fn knob_config_validates_freezes_and_reaches_the_driver() {
    const KNOBBED: &str = r#"
        function describe()
          return {
            label = "Knobbed",
            knobs = {
              { id = "voice.model", type = "model", required = true },
              { id = "rounds", type = "integer", default = 2, min = 1, max = 5 },
            },
          }
        end
        function on_event(state, event, config)
          if event.kind == "input_accepted" and not state.derived then
            state.derived = true
            local voice = config["voice.model"]
            return { effects = { { kind = "derive_thread",
              relationship = "voice:one",
              system_prompt = "you are the voice",
              model = voice.model,
              backend = voice.backend,
              disable_tools = true,
              source_thread_id = event.thread_id } }, state = state }
          end
          return { state = state }
        end
    "#;
    let mut h = harness().await;
    h.install_driver("knobbed", KNOBBED);
    let model_knob = |backend: &str| -> std::collections::BTreeMap<String, serde_json::Value> {
        [(
            "voice.model".to_string(),
            serde_json::json!({"backend": backend, "model": "gpt-5"}),
        )]
        .into_iter()
        .collect()
    };

    // Refusals name the offending knob: missing required value, a key
    // describe() never declared, and a backend outside the pod ceiling.
    let err = h.create_scripted_thread("knobbed").unwrap_err();
    assert!(err.contains("`voice.model` is required"), "{err}");
    let mut with_unknown = model_knob("openai");
    with_unknown.insert("mystery".into(), serde_json::json!(true));
    let err = h
        .create_scripted_thread_with_config("knobbed", with_unknown)
        .unwrap_err();
    assert!(err.contains("unknown knob `mystery`"), "{err}");
    let err = h
        .create_scripted_thread_with_config("knobbed", model_knob("nonexistent"))
        .unwrap_err();
    assert!(err.contains("not in pod"), "{err}");

    // A valid submission creates; the frozen map on the weave carries
    // the materialized default beside the submitted value.
    let primary = h
        .create_scripted_thread_with_config("knobbed", model_knob("openai"))
        .unwrap();
    let weave_id = h.weave_of(&primary);
    let ThreadDriverConfig::Scripted { config, .. } = &h.sched.weaves[&weave_id].driver else {
        panic!("scripted weave lost its driver config");
    };
    assert_eq!(config.get("rounds"), Some(&serde_json::json!(2)));
    assert_eq!(
        config["voice.model"]["model"],
        serde_json::json!("gpt-5"),
        "submitted knob frozen verbatim"
    );

    // The activation hands the frozen config to on_event; the driver
    // derives its voice on the knob's backend + model.
    let mut pending_io = FuturesUnordered::new();
    h.sched
        .send_user_message(&primary, "go".into(), Vec::new(), &mut pending_io);
    h.sched.step_until_blocked(&primary, &mut pending_io);
    let derived = h.sched.weaves[&weave_id]
        .threads
        .iter()
        .find(|r| {
            r.relationship
                .as_ref()
                .is_some_and(|rel| rel.kind == "voice:one")
        })
        .expect("voice derived")
        .thread_id
        .clone();
    assert_eq!(h.sched.tasks[&derived].config.model, "gpt-5");
    assert_eq!(h.sched.tasks[&derived].bindings.backend, "openai");
}

/// The roundtable's `describe()` generates one optional model knob per
/// cast seat; a configured seat derives on that knob's backend + model
/// while unconfigured seats keep the pod defaults, and a replacement
/// for a dead voice re-reads the same frozen knob — a configured seat
/// keeps its provider across deaths.
#[tokio::test]
async fn roundtable_model_knobs_configure_seats_and_outlive_voices() {
    let mut h = harness().await;
    h.install_driver("roundtable", ROUNDTABLE_DRIVER);
    let primary = h
        .create_scripted_thread_with_config(
            "roundtable",
            [(
                "voice.optimist.model".to_string(),
                serde_json::json!({"backend": "openai", "model": "gpt-5"}),
            )]
            .into_iter()
            .collect(),
        )
        .unwrap();
    let weave_id = h.weave_of(&primary);
    let mut pending_io = FuturesUnordered::new();
    h.sched
        .send_user_message(&primary, "well?".into(), Vec::new(), &mut pending_io);
    h.sched.step_until_blocked(&primary, &mut pending_io);

    let voice_threads = |h: &Harness, id: &str| -> Vec<String> {
        let kind = format!("voice:{id}");
        h.sched.weaves[&weave_id]
            .threads
            .iter()
            .filter(|r| r.relationship.as_ref().is_some_and(|rel| rel.kind == kind))
            .map(|r| r.thread_id.clone())
            .collect()
    };
    let optimist1 = voice_threads(&h, "optimist")[0].clone();
    let skeptic = voice_threads(&h, "skeptic")[0].clone();
    assert_eq!(h.sched.tasks[&optimist1].config.model, "gpt-5");
    assert_eq!(h.sched.tasks[&optimist1].bindings.backend, "openai");
    assert_eq!(h.sched.tasks[&skeptic].config.model, "claude-sonnet-4-6");
    assert_eq!(h.sched.tasks[&skeptic].bindings.backend, "anthropic");

    // Kill the configured voice mid-round (it holds the floor), let the
    // skeptic close the round, then a fresh input derives a
    // replacement: same knob, same provider, fresh context.
    h.fail_model(&optimist1, "provider exploded", &mut pending_io);
    h.respond_model(&skeptic, vec![text_block("Noted.")], &mut pending_io);
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::Completed
    ));
    h.sched
        .send_user_message(&primary, "again".into(), Vec::new(), &mut pending_io);
    h.sched.step_until_blocked(&primary, &mut pending_io);
    let optimist2 = voice_threads(&h, "optimist")
        .into_iter()
        .find(|t| *t != optimist1)
        .expect("replacement derived");
    assert_eq!(h.sched.tasks[&optimist2].config.model, "gpt-5");
    assert_eq!(h.sched.tasks[&optimist2].bindings.backend, "openai");
}

/// titled_chat.lua (step 11, slice 1): the degenerate single-agent
/// chat loop in Lua plus model titling. Pins: the chat turn runs and
/// the cycle closes like the builtin; the first completed reply
/// derives a one-turn tools-off title thread on the knob's backend +
/// model; the truncation placeholder stands until the title model
/// replies, then `set_title` lands the cleaned text and journals it;
/// the title thread lingers referenced (drill-down provenance) and a
/// second exchange derives nothing new.
#[tokio::test]
async fn titled_chat_titles_the_thread_from_its_knob_model() {
    let mut h = harness().await;
    h.install_driver("titled_chat", TITLED_CHAT_DRIVER);
    let primary = h
        .create_scripted_thread_with_config(
            "titled_chat",
            [(
                "title.model".to_string(),
                serde_json::json!({"backend": "openai", "model": "gpt-5-nano"}),
            )]
            .into_iter()
            .collect(),
        )
        .unwrap();
    let weave_id = h.weave_of(&primary);
    let mut pending_io = FuturesUnordered::new();

    h.sched.send_user_message(
        &primary,
        "Explain the borrow checker to a C programmer".into(),
        Vec::new(),
        &mut pending_io,
    );
    h.sched.step_until_blocked(&primary, &mut pending_io);
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::AwaitingModel { .. }
    ));

    h.respond_model(
        &primary,
        vec![text_block("It is a compile-time owner tracker.")],
        &mut pending_io,
    );
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::Completed
    ));
    let title_thread = h.sched.weaves[&weave_id]
        .threads
        .iter()
        .find(|r| {
            r.relationship
                .as_ref()
                .is_some_and(|rel| rel.kind == "title")
        })
        .expect("title thread derived after the first reply")
        .thread_id
        .clone();
    assert_eq!(h.sched.tasks[&title_thread].config.model, "gpt-5-nano");
    assert_eq!(h.sched.tasks[&title_thread].bindings.backend, "openai");
    assert!(matches!(
        h.internal_of(&title_thread),
        ThreadInternalState::AwaitingModel { .. }
    ));
    assert_eq!(
        h.sched.tasks[&primary].title.as_deref(),
        Some("Explain the borrow checker to a C programmer"),
        "truncation placeholder until the title model replies"
    );

    h.respond_model(
        &title_thread,
        vec![text_block("'Borrow Checker for C Programmers.'")],
        &mut pending_io,
    );
    assert_eq!(
        h.sched.tasks[&primary].title.as_deref(),
        Some("Borrow Checker for C Programmers")
    );
    assert!(matches!(
        h.internal_of(&title_thread),
        ThreadInternalState::Completed
    ));
    assert!(
        h.sched.weaves[&weave_id]
            .effect_journal
            .records()
            .iter()
            .any(|r| matches!(
                &r.effect,
                PersistedDriverEffect::SetTitle { thread_id, title }
                    if thread_id == &primary && title == "Borrow Checker for C Programmers"
            )),
        "the journal explains where the title came from"
    );

    // Second exchange: a tool round-trip through the degenerate
    // loop's dispatch half — dispatch_tools without interception,
    // then tools_completed → continue_cycle — with no second title
    // job; the title thread lingers as the provenance record.
    h.sched.send_user_message(
        &primary,
        "And lifetimes?".into(),
        Vec::new(),
        &mut pending_io,
    );
    h.sched.step_until_blocked(&primary, &mut pending_io);
    h.respond_model(
        &primary,
        vec![tool_use_block("toolu-lt", "list_images")],
        &mut pending_io,
    );
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::AwaitingTools { .. }
    ));
    h.respond_tool(&primary, "toolu-lt", "no images here", &mut pending_io);
    assert!(
        matches!(
            h.internal_of(&primary),
            ThreadInternalState::AwaitingModel { .. }
        ),
        "tools_completed → continue_cycle took another model turn"
    );
    h.respond_model(
        &primary,
        vec![text_block("Scopes with names.")],
        &mut pending_io,
    );
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::Completed
    ));
    assert!(
        h.sched.weaves[&weave_id]
            .effect_journal
            .records()
            .iter()
            .any(
                |r| matches!(&r.effect, PersistedDriverEffect::DispatchTools { .. })
                    && r.outcome == DriverEffectOutcome::Completed
            ),
        "the dispatch ran through the journaled DispatchTools effect"
    );
    assert_eq!(
        h.sched.weaves[&weave_id].threads.len(),
        2,
        "primary + lingering title thread; nothing new derived"
    );
}

/// A dead title model must not wedge the chat or clobber the title:
/// `thread_failed` unmaps the title thread, the truncation placeholder
/// stands (set_title never fired), and the conversation continues
/// without a retry.
#[tokio::test]
async fn titled_chat_keeps_the_placeholder_when_the_title_model_dies() {
    let mut h = harness().await;
    h.install_driver("titled_chat", TITLED_CHAT_DRIVER);
    let primary = h.create_scripted_thread("titled_chat").unwrap();
    let weave_id = h.weave_of(&primary);
    let mut pending_io = FuturesUnordered::new();

    h.sched
        .send_user_message(&primary, "hello there".into(), Vec::new(), &mut pending_io);
    h.sched.step_until_blocked(&primary, &mut pending_io);
    h.respond_model(&primary, vec![text_block("Hi.")], &mut pending_io);
    let title_thread = h.sched.weaves[&weave_id]
        .threads
        .iter()
        .find(|r| {
            r.relationship
                .as_ref()
                .is_some_and(|rel| rel.kind == "title")
        })
        .expect("title thread derived")
        .thread_id
        .clone();

    h.fail_model(
        &title_thread,
        "title model quota exhausted",
        &mut pending_io,
    );
    assert_eq!(
        h.sched.tasks[&primary].title.as_deref(),
        Some("hello there"),
        "placeholder survives the dead title model"
    );

    h.sched
        .send_user_message(&primary, "still alive?".into(), Vec::new(), &mut pending_io);
    h.sched.step_until_blocked(&primary, &mut pending_io);
    h.respond_model(&primary, vec![text_block("Very.")], &mut pending_io);
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::Completed
    ));
    let derives = h.sched.weaves[&weave_id]
        .effect_journal
        .records()
        .iter()
        .filter(|r| matches!(&r.effect, PersistedDriverEffect::DeriveThread { .. }))
        .count();
    assert_eq!(derives, 1, "no title retry after the model died");
}

// ---------- behavior-spawned scripted weaves (step 11 slice 2) ----------

/// The full behavior→scripted-weave arc: a fire spawns a scripted weave
/// with the TOML-authored knob map frozen at fire time, the primary's
/// Completed does NOT record the run (driver-declared run-done), and the
/// driver's `complete_run` — emitted once the title resolves — records
/// it, with the journal naming the behavior that consumed the
/// declaration.
#[tokio::test]
async fn behavior_fires_a_scripted_weave_and_the_driver_declares_the_run() {
    let mut h = harness().await;
    h.install_driver("titled_chat", TITLED_CHAT_DRIVER);
    h.install_behavior(
        "digest",
        scripted_behavior_config(
            "titled_chat",
            [(
                "title.model".to_string(),
                serde_json::json!({"backend": "openai", "model": "gpt-5-nano"}),
            )]
            .into_iter()
            .collect(),
        ),
        "Summarize the day.",
    );
    let mut pending_io = FuturesUnordered::new();
    let primary = h
        .sched
        .run_behavior(None, None, TEST_POD, "digest", None, &mut pending_io)
        .expect("scripted behavior fire");
    let weave_id = h.weave_of(&primary);
    assert!(
        matches!(
            &h.sched.weaves[&weave_id].driver,
            ThreadDriverConfig::Scripted { name, config }
                if name == "titled_chat" && config.contains_key("title.model")
        ),
        "the weave froze the behavior's knob map at fire time"
    );
    assert_eq!(
        h.behavior_state("digest").last_thread_id.as_deref(),
        Some(primary.as_str())
    );

    h.respond_model(
        &primary,
        vec![text_block("Quiet day; two PRs landed.")],
        &mut pending_io,
    );
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::Completed
    ));
    // Driver-declared run-done: the primary going Completed at
    // finish_cycle is NOT the run finishing — the builtin hook's
    // semantics would have recorded here.
    assert_eq!(h.behavior_state("digest").run_count, 0);
    assert!(
        h.behavior_state("digest").last_outcome.is_none(),
        "primary Completed defers to the driver's declaration"
    );
    assert!(
        h.sched.behavior_has_inflight_run(TEST_POD, "digest"),
        "the Skip/QueueOne overlap gate holds while the run is undeclared"
    );

    let title_thread = h.sched.weaves[&weave_id]
        .threads
        .iter()
        .find(|r| {
            r.relationship
                .as_ref()
                .is_some_and(|rel| rel.kind == "title")
        })
        .expect("title thread derived after the first reply")
        .thread_id
        .clone();
    assert_eq!(
        h.sched.tasks[&title_thread].bindings.backend, "openai",
        "the TOML-authored model knob reached the derive"
    );
    h.respond_model(
        &title_thread,
        vec![text_block("Daily Digest.")],
        &mut pending_io,
    );

    assert_eq!(h.behavior_state("digest").run_count, 1);
    assert_eq!(
        h.behavior_state("digest").last_outcome,
        Some(whisper_agent_protocol::BehaviorOutcome::Completed)
    );
    assert!(
        !h.sched.behavior_has_inflight_run(TEST_POD, "digest"),
        "the declaration opens the overlap gate"
    );
    assert_eq!(
        h.sched.tasks[&primary].title.as_deref(),
        Some("Daily Digest")
    );
    assert!(
        h.sched.weaves[&weave_id]
            .effect_journal
            .records()
            .iter()
            .any(|r| matches!(
                &r.effect,
                PersistedDriverEffect::CompleteRun {
                    outcome: whisper_agent_protocol::BehaviorOutcome::Completed,
                    behavior_id: Some(behavior_id),
                } if behavior_id == "digest"
            )),
        "the journal explains which behavior consumed the declaration"
    );
}

/// Death stays mechanical: a behavior-spawned scripted primary that
/// fails records `Failed` through the terminal-hook backstop — no
/// declaration can arrive from a weave whose only work just died.
#[tokio::test]
async fn behavior_scripted_death_backstop_records_failed() {
    let mut h = harness().await;
    h.install_driver("titled_chat", TITLED_CHAT_DRIVER);
    h.install_behavior(
        "digest",
        scripted_behavior_config("titled_chat", Default::default()),
        "Summarize the day.",
    );
    let mut pending_io = FuturesUnordered::new();
    let primary = h
        .sched
        .run_behavior(None, None, TEST_POD, "digest", None, &mut pending_io)
        .expect("scripted behavior fire");

    h.fail_model(&primary, "provider exploded", &mut pending_io);
    // The production loop pairs every io completion with a step
    // (scheduler main loop); the terminal hook lives in the step
    // epilogue, so mirror the pairing here.
    h.sched.step_until_blocked(&primary, &mut pending_io);
    assert_eq!(h.behavior_state("digest").run_count, 1);
    assert!(
        matches!(
            h.behavior_state("digest").last_outcome,
            Some(whisper_agent_protocol::BehaviorOutcome::Failed { .. })
        ),
        "mechanical death recorded through the backstop"
    );
    assert!(
        !h.sched.behavior_has_inflight_run(TEST_POD, "digest"),
        "a recorded death opens the overlap gate"
    );
}

/// The declaration consumes a queued `QueueOne` payload exactly like
/// the mechanical hook does — the shared recorder owns the release.
#[tokio::test]
async fn declaration_releases_a_queued_payload() {
    let mut h = harness().await;
    h.install_driver("titled_chat", TITLED_CHAT_DRIVER);
    h.install_behavior(
        "digest",
        scripted_behavior_config("titled_chat", Default::default()),
        "Summarize the day.",
    );
    let mut pending_io = FuturesUnordered::new();
    let primary = h
        .sched
        .run_behavior(None, None, TEST_POD, "digest", None, &mut pending_io)
        .expect("scripted behavior fire");
    h.respond_model(&primary, vec![text_block("Done.")], &mut pending_io);
    let weave_id = h.weave_of(&primary);
    let title_thread = h.sched.weaves[&weave_id]
        .threads
        .iter()
        .find(|r| {
            r.relationship
                .as_ref()
                .is_some_and(|rel| rel.kind == "title")
        })
        .expect("title thread")
        .thread_id
        .clone();
    // A fire arrived while the run was in flight and parked its payload.
    h.sched
        .pods
        .get_mut(TEST_POD)
        .unwrap()
        .behaviors
        .get_mut("digest")
        .unwrap()
        .state
        .queued_payload = Some(serde_json::json!({"again": true}));

    h.respond_model(&title_thread, vec![text_block("Digest.")], &mut pending_io);
    assert_eq!(h.behavior_state("digest").run_count, 1);
    assert!(
        h.behavior_state("digest").queued_payload.is_none(),
        "the declaration released the queued payload"
    );
}

/// A behavior whose model knob names a backend outside its own
/// fire-time `[scope]` refuses at the fire — the knob analogue of the
/// bindings narrowing `base_scope_override` already enforces.
#[tokio::test]
async fn behavior_fire_refuses_out_of_scope_model_knob() {
    let mut h = harness().await;
    h.install_driver("titled_chat", TITLED_CHAT_DRIVER);
    let mut config = scripted_behavior_config(
        "titled_chat",
        [(
            "title.model".to_string(),
            serde_json::json!({"backend": "openai", "model": "gpt-5-nano"}),
        )]
        .into_iter()
        .collect(),
    );
    config.scope = whisper_agent_protocol::BehaviorScope {
        backends: Some(vec!["anthropic".into()]),
        ..Default::default()
    };
    h.install_behavior("digest", config, "Summarize the day.");
    let mut pending_io = FuturesUnordered::new();
    let err = h
        .sched
        .run_behavior(None, None, TEST_POD, "digest", None, &mut pending_io)
        .expect_err("out-of-scope knob must refuse the fire");
    assert!(
        err.contains("fire-time scope.backends"),
        "refusal names the fire scope: {err}"
    );
}

/// Retention treats a behavior-spawned scripted weave as one unit: the
/// origin rides only the primary, but the sweep takes every referenced
/// thread once all are terminal and past the window, and the emptied
/// weave retires with them.
#[tokio::test]
async fn retention_sweeps_a_behavior_spawned_weave_as_a_unit() {
    let mut h = harness().await;
    h.install_driver("titled_chat", TITLED_CHAT_DRIVER);
    let mut config = scripted_behavior_config("titled_chat", Default::default());
    config.on_completion = whisper_agent_protocol::RetentionPolicy::DeleteAfterDays { days: 1 };
    h.install_behavior("digest", config, "Summarize the day.");
    let mut pending_io = FuturesUnordered::new();
    let primary = h
        .sched
        .run_behavior(None, None, TEST_POD, "digest", None, &mut pending_io)
        .expect("scripted behavior fire");
    h.respond_model(&primary, vec![text_block("Done.")], &mut pending_io);
    let weave_id = h.weave_of(&primary);
    let members: Vec<String> = h.sched.weaves[&weave_id]
        .threads
        .iter()
        .map(|r| r.thread_id.clone())
        .collect();
    assert_eq!(members.len(), 2, "primary + title thread");
    let title_thread = members
        .iter()
        .find(|id| *id != &primary)
        .expect("title thread")
        .clone();
    h.respond_model(&title_thread, vec![text_block("Digest.")], &mut pending_io);

    let stale = chrono::Utc::now() - chrono::Duration::days(2);
    for id in &members {
        h.sched.tasks.get_mut(id).unwrap().last_active = stale;
    }
    h.sched.retention_sweep();
    for id in &members {
        assert!(
            !h.sched.tasks.contains_key(id),
            "member `{id}` swept with the weave unit"
        );
    }
    assert!(
        !h.sched.weaves.contains_key(&weave_id),
        "the emptied weave retired"
    );
}

/// A weave-unit candidate defers whole while any member still works —
/// the primary being terminal and stale is not enough to archive a
/// weave whose title thread is mid-flight.
#[tokio::test]
async fn retention_defers_a_weave_unit_while_a_member_works() {
    let mut h = harness().await;
    h.install_driver("titled_chat", TITLED_CHAT_DRIVER);
    let mut config = scripted_behavior_config("titled_chat", Default::default());
    config.on_completion = whisper_agent_protocol::RetentionPolicy::DeleteAfterDays { days: 1 };
    h.install_behavior("digest", config, "Summarize the day.");
    let mut pending_io = FuturesUnordered::new();
    let primary = h
        .sched
        .run_behavior(None, None, TEST_POD, "digest", None, &mut pending_io)
        .expect("scripted behavior fire");
    h.respond_model(&primary, vec![text_block("Done.")], &mut pending_io);
    let weave_id = h.weave_of(&primary);
    // The title thread is still awaiting its model call.
    let stale = chrono::Utc::now() - chrono::Duration::days(2);
    let members: Vec<String> = h.sched.weaves[&weave_id]
        .threads
        .iter()
        .map(|r| r.thread_id.clone())
        .collect();
    for id in &members {
        h.sched.tasks.get_mut(id).unwrap().last_active = stale;
    }
    h.sched.retention_sweep();
    for id in &members {
        assert!(
            h.sched.tasks.contains_key(id),
            "no member swept while the title thread works"
        );
    }
    assert!(h.sched.weaves.contains_key(&weave_id));
}

/// Weave-unit membership is resolved by reference, not ticker: an
/// origin thread left dormant (the `advance_head`/`release_ticker`
/// shape) still pulls its whole weave through retention instead of
/// quietly demoting to a per-thread sweep and leaking the auxiliaries.
#[tokio::test]
async fn retention_sweeps_a_weave_whose_origin_thread_went_dormant() {
    let mut h = harness().await;
    h.install_driver("titled_chat", TITLED_CHAT_DRIVER);
    let mut config = scripted_behavior_config("titled_chat", Default::default());
    config.on_completion = whisper_agent_protocol::RetentionPolicy::DeleteAfterDays { days: 1 };
    h.install_behavior("digest", config, "Summarize the day.");
    let mut pending_io = FuturesUnordered::new();
    let primary = h
        .sched
        .run_behavior(None, None, TEST_POD, "digest", None, &mut pending_io)
        .expect("scripted behavior fire");
    h.respond_model(&primary, vec![text_block("Done.")], &mut pending_io);
    let weave_id = h.weave_of(&primary);
    let members: Vec<String> = h.sched.weaves[&weave_id]
        .threads
        .iter()
        .map(|r| r.thread_id.clone())
        .collect();
    let title_thread = members
        .iter()
        .find(|id| *id != &primary)
        .expect("title thread")
        .clone();
    h.respond_model(&title_thread, vec![text_block("Digest.")], &mut pending_io);

    // The origin thread goes dormant — no ticker entry, still
    // referenced by the weave.
    h.sched.thread_ticker.remove(&primary);
    let stale = chrono::Utc::now() - chrono::Duration::days(2);
    for id in &members {
        h.sched.tasks.get_mut(id).unwrap().last_active = stale;
    }
    h.sched.retention_sweep();
    for id in &members {
        assert!(
            !h.sched.tasks.contains_key(id),
            "member `{id}` swept despite the dormant origin thread"
        );
    }
    assert!(!h.sched.weaves.contains_key(&weave_id));
}

/// A ref to a gone thread (crash between sweep and weave flush, or a
/// thread dropped at load) must not leave an unsweepable zombie weave:
/// the unit sweep prunes the dead ref, retires the emptied weave, and
/// clears the stale ticker entry the load path can leave behind.
#[tokio::test]
async fn retention_prunes_gone_refs_and_retires_the_weave() {
    let mut h = harness().await;
    h.install_driver("titled_chat", TITLED_CHAT_DRIVER);
    let mut config = scripted_behavior_config("titled_chat", Default::default());
    config.on_completion = whisper_agent_protocol::RetentionPolicy::DeleteAfterDays { days: 1 };
    h.install_behavior("digest", config, "Summarize the day.");
    let mut pending_io = FuturesUnordered::new();
    let primary = h
        .sched
        .run_behavior(None, None, TEST_POD, "digest", None, &mut pending_io)
        .expect("scripted behavior fire");
    h.respond_model(&primary, vec![text_block("Done.")], &mut pending_io);
    let weave_id = h.weave_of(&primary);
    let title_thread = h.sched.weaves[&weave_id]
        .threads
        .iter()
        .find(|r| r.thread_id != primary)
        .expect("title thread")
        .thread_id
        .clone();
    h.respond_model(&title_thread, vec![text_block("Digest.")], &mut pending_io);

    // The title thread vanishes out from under its ref; its ticker
    // entry stays stale (the load-path shape).
    h.sched.tasks.remove(&title_thread);
    h.sched.tasks.get_mut(&primary).unwrap().last_active =
        chrono::Utc::now() - chrono::Duration::days(2);
    h.sched.retention_sweep();
    assert!(!h.sched.tasks.contains_key(&primary));
    assert!(
        !h.sched.weaves.contains_key(&weave_id),
        "gone ref pruned, emptied weave retired"
    );
    assert!(
        !h.sched.thread_ticker.contains_key(&title_thread),
        "stale ticker entry for the gone thread pruned"
    );
}

// ---------- step 11 slice 3: query_knowledge ----------

fn thinking_block(text: &str) -> ContentBlock {
    ContentBlock::Thinking {
        replay: None,
        thinking: text.to_string(),
    }
}

fn wiki_hit(source: &str, chunk_byte: u8, score: f32, text: &str) -> RerankedCandidate {
    RerankedCandidate {
        bucket_id: crate::knowledge::BucketId::server("wiki"),
        chunk_id: crate::knowledge::ChunkId([chunk_byte; 32]),
        chunk_text: text.to_string(),
        source_ref: crate::knowledge::SourceRef {
            source_id: source.to_string(),
            locator: (!source.is_empty()).then(|| "§2".to_string()),
        },
        source_score: score,
        source_path: crate::knowledge::SearchPath::Dense,
        rerank_score: score,
    }
}

impl Harness {
    fn scripted_data(&self, weave_id: &str) -> serde_json::Value {
        match &self.sched.weaves[weave_id].driver_state {
            DriverState::Scripted { data, .. } => data.clone(),
            other => panic!("weave `{weave_id}` is not scripted: {other:?}"),
        }
    }

    /// Manufacture an in-flight `query_knowledge`: the driver-state,
    /// journal, and in-flight entries `weave_query_knowledge` would
    /// have created had a launch succeeded — the harness has no
    /// bucket/reranker fixtures, so the launch path itself refuses
    /// environmentally and is pinned by the refusal test instead.
    fn fake_query_in_flight(
        &mut self,
        weave_id: &str,
        query_id: &str,
        query: &str,
    ) -> crate::runtime::driver::DriverEffectId {
        let weave = self.sched.weaves.get_mut(weave_id).unwrap();
        if let DriverState::Scripted { data, .. } = &mut weave.driver_state
            && let Some(object) = data.as_object_mut()
        {
            object.insert("aq_pending".into(), serde_json::json!(query_id));
        }
        let effect_id = weave.record_pending_effect(PersistedDriverEffect::QueryKnowledge {
            query_id: query_id.to_string(),
            query: query.to_string(),
            buckets: vec!["server:wiki".into()],
        });
        self.sched
            .scripted_queries_in_flight
            .insert((weave_id.to_string(), query_id.to_string()), effect_id);
        effect_id
    }

    fn nudge_messages(&self, thread_id: &str) -> Vec<String> {
        self.sched.tasks[thread_id]
            .conversation
            .messages()
            .iter()
            .filter(|message| message.role == whisper_agent_protocol::Role::System)
            .flat_map(|message| {
                message.content.iter().filter_map(|block| match block {
                    ContentBlock::Text { text } if text.contains("knowledge bucket surfaced") => {
                        Some(text.clone())
                    }
                    _ => None,
                })
            })
            .collect()
    }

    /// The child thread id of the weave's (sole) journaled dispatch
    /// callback — how a test finds the thread `dispatch_thread` spawned.
    fn dispatch_child_of(&self, weave_id: &str) -> String {
        self.sched.weaves[weave_id]
            .effect_journal
            .records()
            .iter()
            .find_map(|r| match &r.effect {
                PersistedDriverEffect::DispatchCallback {
                    child_thread_id, ..
                } => Some(child_thread_id.clone()),
                _ => None,
            })
            .expect("dispatch callback journaled")
    }

    /// Text of entries titled_chat's dispatch flush appended (author
    /// "dispatch").
    fn dispatch_notifications(&self, thread_id: &str) -> Vec<String> {
        self.sched.tasks[thread_id]
            .conversation
            .messages()
            .iter()
            .filter(|message| message.effective_author().as_str() == "dispatch")
            .flat_map(|message| {
                message.content.iter().filter_map(|block| match block {
                    ContentBlock::Text { text } => Some(text.clone()),
                    _ => None,
                })
            })
            .collect()
    }
}

/// The autoquery knob on an unprovisioned pod (no buckets in scope)
/// must never stall or kill the chat: the effect emission is real (the
/// query text pins reasoning-then-text extraction end-to-end), the
/// refusal journals Failed and delivers `query_failed` in the same
/// activation, and the tools round-trip continues bare.
#[tokio::test]
async fn titled_chat_autoquery_refusal_never_stalls_the_chat() {
    let mut h = harness().await;
    h.install_driver("titled_chat", TITLED_CHAT_DRIVER);
    let primary = h
        .create_scripted_thread_with_config(
            "titled_chat",
            [("autoquery".to_string(), serde_json::json!(true))]
                .into_iter()
                .collect(),
        )
        .unwrap();
    let weave_id = h.weave_of(&primary);
    let mut pending_io = FuturesUnordered::new();

    h.sched.send_user_message(
        &primary,
        "How do weaves persist?".into(),
        Vec::new(),
        &mut pending_io,
    );
    h.sched.step_until_blocked(&primary, &mut pending_io);
    h.respond_model(
        &primary,
        vec![
            thinking_block("weave storage internals"),
            tool_use_block("toolu-aq", "list_images"),
        ],
        &mut pending_io,
    );
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::AwaitingTools { .. }
    ));
    let refused = h.sched.weaves[&weave_id]
        .effect_journal
        .records()
        .iter()
        .find(|r| matches!(&r.effect, PersistedDriverEffect::QueryKnowledge { .. }))
        .expect("the driver issued a real query_knowledge effect");
    assert!(
        matches!(
            &refused.effect,
            PersistedDriverEffect::QueryKnowledge { query_id, query, .. }
                if query_id == "aq-1" && query == "weave storage internals"
        ),
        "query text drawn from the reply's reasoning: {:?}",
        refused.effect
    );
    assert!(
        matches!(
            &refused.outcome,
            DriverEffectOutcome::Failed { message }
                if message.contains("no knowledge buckets")
        ),
        "environmental refusal journals Failed: {:?}",
        refused.outcome
    );
    assert_eq!(
        h.scripted_data(&weave_id).get("aq_pending"),
        None,
        "the same-activation query_failed cleared the pending marker"
    );

    h.respond_tool(&primary, "toolu-aq", "no images", &mut pending_io);
    assert!(
        matches!(
            h.internal_of(&primary),
            ThreadInternalState::AwaitingModel { .. }
        ),
        "tools_completed continued bare — retrieval was only ever opportunistic"
    );
    assert!(h.nudge_messages(&primary).is_empty());
}

/// The flagship autoquery arc: a query in flight holds the continue
/// (parked tools boundary), the completion stores the nudge, the
/// re-fired boundary continues with it — the nudge lands as a
/// journaled system entry ahead of the next model call — and a second
/// completion carrying an already-seen hit dedups to nothing.
#[tokio::test]
async fn titled_chat_autoquery_holds_the_continue_and_injects_the_nudge() {
    let mut h = harness().await;
    h.install_driver("titled_chat", TITLED_CHAT_DRIVER);
    let primary = h
        .create_scripted_thread_with_config(
            "titled_chat",
            [("autoquery".to_string(), serde_json::json!(true))]
                .into_iter()
                .collect(),
        )
        .unwrap();
    let weave_id = h.weave_of(&primary);
    let mut pending_io = FuturesUnordered::new();

    h.sched.send_user_message(
        &primary,
        "How do weaves persist?".into(),
        Vec::new(),
        &mut pending_io,
    );
    h.sched.step_until_blocked(&primary, &mut pending_io);
    h.respond_model(
        &primary,
        vec![
            thinking_block("weave storage internals"),
            tool_use_block("toolu-aq", "list_images"),
        ],
        &mut pending_io,
    );
    // Stand in for a successful launch (see fake_query_in_flight).
    let effect_id = h.fake_query_in_flight(&weave_id, "aq-2", "weave storage internals");

    h.respond_tool(&primary, "toolu-aq", "no images", &mut pending_io);
    assert!(
        matches!(
            h.internal_of(&primary),
            ThreadInternalState::ToolsBoundary { .. }
        ),
        "the continue holds while the query is in flight"
    );

    let long_chunk = "The effect journal is flushed before any lazy IO future polls.";
    h.sched.apply_scripted_query_completion(
        crate::runtime::io_dispatch::ScriptedQueryCompletion {
            weave_id: weave_id.clone(),
            query_id: "aq-2".into(),
            effect_id,
            query: "weave storage internals".into(),
            snippet_chars: 40,
            result: Ok(vec![
                wiki_hit("Paxos", 9, 0.83, long_chunk),
                wiki_hit("", 12, 0.31, "chunk-keyed hit"),
            ]),
        },
        &mut pending_io,
    );

    assert!(
        h.sched.scripted_queries_in_flight.is_empty(),
        "completion cleared the in-flight entry"
    );
    let record = h.sched.weaves[&weave_id]
        .effect_journal
        .records()
        .iter()
        .find(|r| r.id == effect_id)
        .unwrap();
    assert_eq!(record.outcome, DriverEffectOutcome::Completed);
    assert!(
        matches!(
            h.internal_of(&primary),
            ThreadInternalState::AwaitingModel { .. }
        ),
        "the re-fired boundary continued into the next model call"
    );
    let nudges = h.nudge_messages(&primary);
    assert_eq!(nudges.len(), 1, "exactly one nudge entry injected");
    assert!(
        nudges[0].contains(&format!(
            "[{}] Paxos (§2)",
            crate::knowledge::BucketId::server("wiki")
        )),
        "nudge names the hit's bucket and source: {}",
        nudges[0]
    );
    assert!(
        nudges[0].contains("The effect journal is flushed before any…"),
        "snippet clipped to the completion's snippet_chars: {}",
        nudges[0]
    );
    let nudge_entry = h.sched.weaves[&weave_id]
        .effect_journal
        .records()
        .iter()
        .find_map(|r| match &r.effect {
            PersistedDriverEffect::Continue {
                nudge_entry: Some(index),
                ..
            } => Some(*index),
            _ => None,
        })
        .expect("the continue journaled its injected entry");
    let entry = &h.sched.tasks[&primary].conversation.messages()[nudge_entry];
    assert_eq!(entry.role, whisper_agent_protocol::Role::System);
    let data = h.scripted_data(&weave_id);
    assert_eq!(data.get("aq_pending"), None);
    assert_eq!(
        data.get("aq_seen")
            .and_then(|seen| seen.as_object())
            .map(|seen| seen.len()),
        Some(2),
        "both hits recorded for dedup"
    );

    // Second round: the model tools again, another query completes with
    // an already-seen hit — dedup leaves nothing to inject and the
    // continue runs bare.
    h.respond_model(
        &primary,
        vec![tool_use_block("toolu-aq2", "list_images")],
        &mut pending_io,
    );
    let effect_id_2 = h.fake_query_in_flight(&weave_id, "aq-3", "weave storage internals");
    h.respond_tool(&primary, "toolu-aq2", "still none", &mut pending_io);
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::ToolsBoundary { .. }
    ));
    h.sched.apply_scripted_query_completion(
        crate::runtime::io_dispatch::ScriptedQueryCompletion {
            weave_id: weave_id.clone(),
            query_id: "aq-3".into(),
            effect_id: effect_id_2,
            query: "weave storage internals".into(),
            snippet_chars: 40,
            result: Ok(vec![wiki_hit("Paxos", 9, 0.9, long_chunk)]),
        },
        &mut pending_io,
    );
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::AwaitingModel { .. }
    ));
    assert_eq!(
        h.nudge_messages(&primary).len(),
        1,
        "the already-seen hit deduped to nothing — no second nudge"
    );
}

/// Re-using a correlation id while its query is still in flight is a
/// static authoring bug: the effect refuses as a driver fault (the
/// activation fails and the origin thread dies loudly), unlike the
/// environmental refusals that deliver `query_failed`.
#[tokio::test]
async fn scripted_query_duplicate_id_is_a_driver_fault() {
    let mut h = harness().await;
    h.install_driver(
        "dup_query",
        r#"
            function on_event(state, event)
              if event.kind == "turn_start" then
                return { effects = { { kind = "query_knowledge",
                  id = "dup", query = "anything" } }, state = state }
              end
              return { state = state }
            end
        "#,
    );
    let primary = h.create_scripted_thread("dup_query").unwrap();
    let weave_id = h.weave_of(&primary);
    h.sched
        .scripted_queries_in_flight
        .insert((weave_id.clone(), "dup".into()), 999);
    let mut pending_io = FuturesUnordered::new();
    h.sched
        .send_user_message(&primary, "go".into(), Vec::new(), &mut pending_io);
    h.sched.step_until_blocked(&primary, &mut pending_io);
    let detail = h.sched.tasks[&primary]
        .failure_detail()
        .expect("duplicate id fails the activation's origin thread");
    assert!(
        detail.contains("already in flight"),
        "failure names the duplicate: {detail}"
    );
    assert!(
        h.sched.weaves[&weave_id]
            .effect_journal
            .records()
            .iter()
            .any(|r| matches!(
                (&r.effect, &r.outcome),
                (
                    PersistedDriverEffect::QueryKnowledge { query_id, .. },
                    DriverEffectOutcome::Failed { message },
                ) if query_id == "dup" && message.contains("already in flight")
            )),
        "static faults journal like every refusal — auditable, not silent"
    );
}

/// A query still pending at load died with the process: the healing
/// pass queues a loss notice (leaving the journal record Pending — a
/// crash before delivery re-derives the notice next load), and the
/// weave's next activation drains `query_failed` ahead of the re-fired
/// boundary, failing the record at delivery — the parked continue
/// releases bare instead of wedging forever. Also pins the
/// load-order prerequisite: the shutdown interrupt exempts async
/// non-thread records, or this scan would find nothing.
#[tokio::test]
async fn pending_scripted_queries_heal_to_query_failed_at_load() {
    let mut h = harness().await;
    h.install_driver("titled_chat", TITLED_CHAT_DRIVER);
    let primary = h
        .create_scripted_thread_with_config(
            "titled_chat",
            [("autoquery".to_string(), serde_json::json!(true))]
                .into_iter()
                .collect(),
        )
        .unwrap();
    let weave_id = h.weave_of(&primary);
    let mut pending_io = FuturesUnordered::new();

    h.sched.send_user_message(
        &primary,
        "How do weaves persist?".into(),
        Vec::new(),
        &mut pending_io,
    );
    h.sched.step_until_blocked(&primary, &mut pending_io);
    h.respond_model(
        &primary,
        vec![
            thinking_block("weave storage internals"),
            tool_use_block("toolu-aq", "list_images"),
        ],
        &mut pending_io,
    );
    let effect_id = h.fake_query_in_flight(&weave_id, "aq-2", "weave storage internals");
    h.respond_tool(&primary, "toolu-aq", "no images", &mut pending_io);
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::ToolsBoundary { .. }
    ));

    // Restart: in-flight futures die with the process. The shutdown
    // interrupt (persist load path) must leave the async record
    // Pending for the heal scan to find — pinned here by running it.
    h.sched.scripted_queries_in_flight.clear();
    h.sched
        .weaves
        .get_mut(&weave_id)
        .unwrap()
        .interrupt_all_pending("task was in-flight at last shutdown");
    h.sched.heal_pending_scripted_queries();
    let record = h.sched.weaves[&weave_id]
        .effect_journal
        .records()
        .iter()
        .find(|r| r.id == effect_id)
        .unwrap();
    assert_eq!(
        record.outcome,
        DriverEffectOutcome::Pending,
        "the scan queues the notice but leaves the record Pending — \
         a crash before delivery re-derives it next load"
    );
    assert!(
        h.sched.scripted_query_notices.contains_key(&weave_id),
        "loss notice queued for first activation"
    );

    // First activation after the restart: the parked boundary re-fires
    // (here via an explicit step — production reaches it through any
    // stepping path) and the drain delivers the loss notice first,
    // failing the record at delivery.
    h.sched.step_until_blocked(&primary, &mut pending_io);
    let record = h.sched.weaves[&weave_id]
        .effect_journal
        .records()
        .iter()
        .find(|r| r.id == effect_id)
        .unwrap();
    assert!(
        matches!(
            &record.outcome,
            DriverEffectOutcome::Failed { message } if message.contains("lost to restart")
        ),
        "delivery resolved the lost record: {:?}",
        record.outcome
    );
    assert!(
        matches!(
            h.internal_of(&primary),
            ThreadInternalState::AwaitingModel { .. }
        ),
        "the held continue released bare after the loss notice"
    );
    assert!(
        !h.sched.scripted_query_notices.contains_key(&weave_id),
        "loss notice drained"
    );
    assert_eq!(h.scripted_data(&weave_id).get("aq_pending"), None);
    assert!(h.nudge_messages(&primary).is_empty());
}

/// The flagship async-dispatch arc (step 11 slice 4): the model
/// dispatches a child with sync=false, the ack rides the normal tool
/// path, the turn finishes, and the child's later completion arrives as
/// `dispatch_completed` — the driver appends an attributed notification
/// and runs a fresh turn (builtin injection expressed in driver code).
/// The builtin followup queue stays untouched: machine text never rides
/// the input path on a scripted weave.
#[tokio::test]
async fn titled_chat_async_dispatch_round_trip() {
    let mut h = harness().await;
    h.install_driver("titled_chat", TITLED_CHAT_DRIVER);
    let primary = h.create_scripted_thread("titled_chat").unwrap();
    let weave_id = h.weave_of(&primary);
    let mut pending_io = FuturesUnordered::new();

    h.sched.send_user_message(
        &primary,
        "Survey weave retention in the background".into(),
        Vec::new(),
        &mut pending_io,
    );
    h.sched.step_until_blocked(&primary, &mut pending_io);
    h.respond_model(
        &primary,
        vec![tool_use_block_with_input(
            "toolu-disp",
            "dispatch_thread",
            serde_json::json!({"prompt": "Survey weave retention rules", "sync": false}),
        )],
        &mut pending_io,
    );
    // The intercept ran inside the boundary step: child created,
    // callback journaled Pending, watcher armed.
    let child = h.dispatch_child_of(&weave_id);
    assert!(h.sched.scripted_dispatch_watchers.contains_key(&child));
    assert!(matches!(
        h.internal_of(&child),
        ThreadInternalState::AwaitingModel { .. }
    ));
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::AwaitingTools { .. }
    ));

    // Ack lands (harness stand-in for the immediate-ack future), the
    // turn continues and finishes while the child still works.
    h.respond_tool(&primary, "toolu-disp", "dispatched ack", &mut pending_io);
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::AwaitingModel { .. }
    ));
    h.respond_model(
        &primary,
        vec![text_block("Dispatched; I'll report when it lands.")],
        &mut pending_io,
    );
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::Completed
    ));
    assert!(h.dispatch_notifications(&primary).is_empty());

    // Child terminal → watcher fires → quiescent primary gets the
    // notification appended and a fresh turn.
    h.respond_model(
        &child,
        vec![text_block("Retention is weave-unit, terminal-only.")],
        &mut pending_io,
    );
    assert!(
        matches!(
            h.internal_of(&primary),
            ThreadInternalState::AwaitingModel { .. }
        ),
        "the notification kicked a fresh turn on the quiescent primary"
    );
    let notifications = h.dispatch_notifications(&primary);
    assert_eq!(notifications.len(), 1);
    assert!(
        notifications[0].contains(&format!("[dispatched thread {child} completed]"))
            && notifications[0].contains("Retention is weave-unit, terminal-only."),
        "notification carries the child's final text: {:?}",
        notifications[0]
    );
    let record = h.sched.weaves[&weave_id]
        .effect_journal
        .records()
        .iter()
        .find(|r| matches!(&r.effect, PersistedDriverEffect::DispatchCallback { .. }))
        .unwrap();
    assert_eq!(record.outcome, DriverEffectOutcome::Completed);
    assert!(!h.sched.scripted_dispatch_watchers.contains_key(&child));

    h.respond_model(
        &primary,
        vec![text_block("The survey says: weave-unit retention.")],
        &mut pending_io,
    );
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::Completed
    ));
}

/// A dispatch terminal landing mid-cycle stores instead of moving the
/// primary (the movement contract): the record resolves immediately,
/// but the notification waits in driver state until the cycle finishes,
/// then flushes as append + fresh turn.
#[tokio::test]
async fn titled_chat_async_dispatch_stores_while_busy() {
    let mut h = harness().await;
    h.install_driver("titled_chat", TITLED_CHAT_DRIVER);
    let primary = h.create_scripted_thread("titled_chat").unwrap();
    let weave_id = h.weave_of(&primary);
    let mut pending_io = FuturesUnordered::new();

    h.sched.send_user_message(
        &primary,
        "Kick off the background survey".into(),
        Vec::new(),
        &mut pending_io,
    );
    h.sched.step_until_blocked(&primary, &mut pending_io);
    h.respond_model(
        &primary,
        vec![tool_use_block_with_input(
            "toolu-disp",
            "dispatch_thread",
            serde_json::json!({"prompt": "survey", "sync": false}),
        )],
        &mut pending_io,
    );
    let child = h.dispatch_child_of(&weave_id);
    h.respond_tool(&primary, "toolu-disp", "dispatched ack", &mut pending_io);
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::AwaitingModel { .. }
    ));

    // Child completes while the primary is mid-turn: store-only.
    h.respond_model(&child, vec![text_block("Early result.")], &mut pending_io);
    assert!(
        matches!(
            h.internal_of(&primary),
            ThreadInternalState::AwaitingModel { .. }
        ),
        "a busy primary is never moved by the dispatch handler"
    );
    let record = h.sched.weaves[&weave_id]
        .effect_journal
        .records()
        .iter()
        .find(|r| matches!(&r.effect, PersistedDriverEffect::DispatchCallback { .. }))
        .unwrap();
    assert_eq!(
        record.outcome,
        DriverEffectOutcome::Completed,
        "the record resolves at live delivery even when the driver stores"
    );
    assert!(h.dispatch_notifications(&primary).is_empty());
    assert_eq!(
        h.scripted_data(&weave_id)["disp_pending"]
            .as_array()
            .map(Vec::len),
        Some(1)
    );

    // The cycle finishes → the stored notification flushes.
    h.respond_model(
        &primary,
        vec![text_block("Done thinking.")],
        &mut pending_io,
    );
    assert!(
        matches!(
            h.internal_of(&primary),
            ThreadInternalState::AwaitingModel { .. }
        ),
        "finish flushed the stored notification into a fresh turn"
    );
    let notifications = h.dispatch_notifications(&primary);
    assert_eq!(notifications.len(), 1);
    assert!(notifications[0].contains("Early result."));
    assert_eq!(h.scripted_data(&weave_id).get("disp_pending"), None);

    h.respond_model(
        &primary,
        vec![text_block("Reporting the early result.")],
        &mut pending_io,
    );
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::Completed
    ));
}

/// Cancelling the PARENT with a child in flight (review finding, slice
/// 4): the cascade cancels the child, whose `dispatch_failed` arrives
/// right behind the primary's death notice. The driver must store, not
/// flush — a `run_agent` against the corpse would fault the activation
/// and clobber the user's Cancelled state with a driver Failed. Input
/// then revives the primary and the stored notification flushes at the
/// revived cycle's finish.
#[tokio::test]
async fn titled_chat_parent_cancel_with_inflight_dispatch_stays_cancelled() {
    let mut h = harness().await;
    h.install_driver("titled_chat", TITLED_CHAT_DRIVER);
    let primary = h.create_scripted_thread("titled_chat").unwrap();
    let weave_id = h.weave_of(&primary);
    let mut pending_io = FuturesUnordered::new();

    h.sched.send_user_message(
        &primary,
        "Spawn the background job".into(),
        Vec::new(),
        &mut pending_io,
    );
    h.sched.step_until_blocked(&primary, &mut pending_io);
    h.respond_model(
        &primary,
        vec![tool_use_block_with_input(
            "toolu-disp",
            "dispatch_thread",
            serde_json::json!({"prompt": "job", "sync": false}),
        )],
        &mut pending_io,
    );
    let child = h.dispatch_child_of(&weave_id);
    h.respond_tool(&primary, "toolu-disp", "dispatched ack", &mut pending_io);
    h.respond_model(&primary, vec![text_block("Waiting.")], &mut pending_io);
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::Completed
    ));

    h.sched.execute_cancel_thread(&primary, &mut pending_io);
    assert!(
        matches!(h.internal_of(&primary), ThreadInternalState::Cancelled),
        "the cancel sticks — no driver-failure clobber: {:?}",
        h.internal_of(&primary)
    );
    assert!(matches!(
        h.internal_of(&child),
        ThreadInternalState::Cancelled
    ));
    assert!(
        h.dispatch_notifications(&primary).is_empty(),
        "no notification appended to a corpse"
    );
    let record = h.sched.weaves[&weave_id]
        .effect_journal
        .records()
        .iter()
        .find(|r| matches!(&r.effect, PersistedDriverEffect::DispatchCallback { .. }))
        .unwrap();
    assert!(matches!(
        &record.outcome,
        DriverEffectOutcome::Failed { message } if message.contains("cancelled")
    ));
    assert_eq!(h.scripted_data(&weave_id)["dead"], serde_json::json!(true));
    assert_eq!(
        h.scripted_data(&weave_id)["disp_pending"]
            .as_array()
            .map(Vec::len),
        Some(1)
    );

    // Input revives the head; the stored notification flushes at the
    // revived cycle's finish.
    h.sched.send_user_message(
        &primary,
        "Never mind, continue".into(),
        Vec::new(),
        &mut pending_io,
    );
    h.sched.step_until_blocked(&primary, &mut pending_io);
    h.respond_model(&primary, vec![text_block("Back online.")], &mut pending_io);
    assert!(
        matches!(
            h.internal_of(&primary),
            ThreadInternalState::AwaitingModel { .. }
        ),
        "revived finish flushed the stored notification into a fresh turn"
    );
    let notifications = h.dispatch_notifications(&primary);
    assert_eq!(notifications.len(), 1);
    assert!(notifications[0].contains("failed: child thread was cancelled"));
}

/// An externally cancelled child reports through `dispatch_failed` —
/// the driver hears the loss (message names the cancel) instead of
/// waiting forever, and the journal record fails.
#[tokio::test]
async fn titled_chat_async_dispatch_reports_a_cancelled_child() {
    let mut h = harness().await;
    h.install_driver("titled_chat", TITLED_CHAT_DRIVER);
    let primary = h.create_scripted_thread("titled_chat").unwrap();
    let weave_id = h.weave_of(&primary);
    let mut pending_io = FuturesUnordered::new();

    h.sched.send_user_message(
        &primary,
        "Start the background job".into(),
        Vec::new(),
        &mut pending_io,
    );
    h.sched.step_until_blocked(&primary, &mut pending_io);
    h.respond_model(
        &primary,
        vec![tool_use_block_with_input(
            "toolu-disp",
            "dispatch_thread",
            serde_json::json!({"prompt": "doomed job", "sync": false}),
        )],
        &mut pending_io,
    );
    let child = h.dispatch_child_of(&weave_id);
    h.respond_tool(&primary, "toolu-disp", "dispatched ack", &mut pending_io);
    h.respond_model(&primary, vec![text_block("Running.")], &mut pending_io);
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::Completed
    ));

    h.sched.execute_cancel_thread(&child, &mut pending_io);
    assert!(
        matches!(
            h.internal_of(&primary),
            ThreadInternalState::AwaitingModel { .. }
        ),
        "the failure notification kicked a fresh turn"
    );
    let notifications = h.dispatch_notifications(&primary);
    assert_eq!(notifications.len(), 1);
    assert!(
        notifications[0].contains("failed: child thread was cancelled"),
        "notification names the cancel: {:?}",
        notifications[0]
    );
    let record = h.sched.weaves[&weave_id]
        .effect_journal
        .records()
        .iter()
        .find(|r| matches!(&r.effect, PersistedDriverEffect::DispatchCallback { .. }))
        .unwrap();
    assert!(matches!(
        &record.outcome,
        DriverEffectOutcome::Failed { message } if message.contains("cancelled")
    ));
    assert!(!h.sched.scripted_dispatch_watchers.contains_key(&child));
}

/// The restart contract (slice 4): a callback whose delivery the crash
/// swallowed — child completed, watcher gone with the process — heals
/// at load into a notice built from the child's persisted final state.
/// The shutdown interrupt must exempt the async record (else the heal
/// scans Pending and finds nothing), the record stays Pending until
/// delivery, and the next activation appends the notification with the
/// REAL result — reconnect, not loss.
#[tokio::test]
async fn pending_dispatch_callbacks_reconnect_at_load() {
    let mut h = harness().await;
    h.install_driver("titled_chat", TITLED_CHAT_DRIVER);
    let primary = h.create_scripted_thread("titled_chat").unwrap();
    let weave_id = h.weave_of(&primary);
    let mut pending_io = FuturesUnordered::new();

    h.sched.send_user_message(
        &primary,
        "Long survey please".into(),
        Vec::new(),
        &mut pending_io,
    );
    h.sched.step_until_blocked(&primary, &mut pending_io);
    h.respond_model(
        &primary,
        vec![tool_use_block_with_input(
            "toolu-disp",
            "dispatch_thread",
            serde_json::json!({"prompt": "long survey", "sync": false}),
        )],
        &mut pending_io,
    );
    let child = h.dispatch_child_of(&weave_id);
    h.respond_tool(&primary, "toolu-disp", "dispatched ack", &mut pending_io);
    h.respond_model(&primary, vec![text_block("Waiting.")], &mut pending_io);

    // Crash window: the in-memory watcher dies with the process; the
    // child's completion then has nobody to tell (this also pins that
    // an unwatched terminal delivers nothing).
    h.sched.scripted_dispatch_watchers.clear();
    h.respond_model(
        &child,
        vec![text_block("Post-crash findings.")],
        &mut pending_io,
    );
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::Completed
    ));
    assert!(h.dispatch_notifications(&primary).is_empty());

    // Load path: the shutdown interrupt exempts the async record...
    h.sched
        .weaves
        .get_mut(&weave_id)
        .unwrap()
        .interrupt_all_pending("task was in-flight at last shutdown");
    let record = h.sched.weaves[&weave_id]
        .effect_journal
        .records()
        .iter()
        .find(|r| matches!(&r.effect, PersistedDriverEffect::DispatchCallback { .. }))
        .unwrap();
    assert_eq!(
        record.outcome,
        DriverEffectOutcome::Pending,
        "the shutdown interrupt leaves the async record for the heal"
    );
    // ...and the heal derives a completion notice from the child's
    // persisted final state, leaving the record Pending until delivery.
    h.sched.heal_pending_scripted_dispatches();
    assert!(h.sched.scripted_dispatch_watchers.is_empty());
    assert!(h.sched.scripted_dispatch_notices.contains_key(&weave_id));
    let record = h.sched.weaves[&weave_id]
        .effect_journal
        .records()
        .iter()
        .find(|r| matches!(&r.effect, PersistedDriverEffect::DispatchCallback { .. }))
        .unwrap();
    assert_eq!(record.outcome, DriverEffectOutcome::Pending);

    // First activation after the restart delivers the notice: record
    // resolves, notification appends with the real result, fresh turn.
    h.sched
        .send_user_message(&primary, "Any news?".into(), Vec::new(), &mut pending_io);
    h.sched.step_until_blocked(&primary, &mut pending_io);
    let record = h.sched.weaves[&weave_id]
        .effect_journal
        .records()
        .iter()
        .find(|r| matches!(&r.effect, PersistedDriverEffect::DispatchCallback { .. }))
        .unwrap();
    assert_eq!(record.outcome, DriverEffectOutcome::Completed);
    let notifications = h.dispatch_notifications(&primary);
    assert_eq!(notifications.len(), 1);
    assert!(
        notifications[0].contains("Post-crash findings."),
        "reconnect delivered the REAL result, not a loss notice: {:?}",
        notifications[0]
    );
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::AwaitingModel { .. }
    ));
    assert!(!h.sched.scripted_dispatch_notices.contains_key(&weave_id));
}

/// Drive a fresh titled_chat weave through its opening exchange and
/// close the title job, leaving the primary Completed and quiescent —
/// the launch state for the slice-5 compaction tests.
async fn settled_titled_chat(
    knobs: std::collections::BTreeMap<String, serde_json::Value>,
) -> (Harness, String, String) {
    let mut h = harness().await;
    h.install_driver("titled_chat", TITLED_CHAT_DRIVER);
    let primary = h
        .create_scripted_thread_with_config("titled_chat", knobs)
        .unwrap();
    let weave_id = h.weave_of(&primary);
    let mut pending_io = FuturesUnordered::new();
    h.sched.send_user_message(
        &primary,
        "Opening question".into(),
        Vec::new(),
        &mut pending_io,
    );
    h.sched.step_until_blocked(&primary, &mut pending_io);
    h.respond_model(
        &primary,
        vec![text_block("Opening reply.")],
        &mut pending_io,
    );
    let title_thread = h.title_thread_of(&weave_id);
    h.respond_model(&title_thread, vec![text_block("A Title")], &mut pending_io);
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::Completed
    ));
    (h, primary, weave_id)
}

impl Harness {
    fn title_thread_of(&self, weave_id: &str) -> String {
        self.sched.weaves[weave_id]
            .threads
            .iter()
            .find(|r| {
                r.relationship
                    .as_ref()
                    .is_some_and(|rel| rel.kind == "title")
            })
            .expect("title thread derived")
            .thread_id
            .clone()
    }

    /// Send the client's manual CompactThread message (the scripted
    /// branch routes it to the driver as `compaction_ready`).
    fn compact_message(&mut self, thread_id: &str) {
        let mut pending_io = FuturesUnordered::new();
        self.sched.apply_client_message(
            7,
            whisper_agent_protocol::ClientToServer::CompactThread {
                thread_id: thread_id.into(),
                correlation_id: None,
            },
            &mut pending_io,
        );
    }

    /// The weave ref carrying a `compaction` relationship — the
    /// continuation thread — with its recorded source.
    fn compaction_continuation(&self, weave_id: &str) -> (String, String) {
        let r = self.sched.weaves[weave_id]
            .threads
            .iter()
            .find(|r| {
                r.relationship
                    .as_ref()
                    .is_some_and(|rel| rel.kind == "compaction")
            })
            .expect("compaction continuation derived");
        (
            r.thread_id.clone(),
            r.relationship
                .as_ref()
                .and_then(|rel| rel.source.as_ref())
                .map(|s| s.thread_id.clone())
                .unwrap_or_default(),
        )
    }
}

/// The manual compaction path on a scripted weave (step 11 slice 5),
/// end to end: the client message routes to the driver as
/// `compaction_ready` instead of the builtin Function, the driver runs
/// the summary turn with the resolved prompt appended as an ordinary
/// user message, extracts with the configured regex, derives the
/// continuation inheriting the OLD head's setup verbatim (pod drift
/// must not leak across the roll), advances the head, and runs the
/// seeded first turn. The old head stays Completed as a dormant
/// auxiliary.
#[tokio::test]
async fn titled_chat_manual_compaction_rolls_the_weave() {
    let (mut h, primary, weave_id) = settled_titled_chat(Default::default()).await;
    let mut pending_io = FuturesUnordered::new();

    // Pod drift after the head launched: the continuation must NOT
    // pick this up — its setup copies the old head's verbatim.
    h.sched.pods.get_mut(TEST_POD).unwrap().system_prompt = "DRIFTED PROMPT".into();

    h.compact_message(&primary);
    assert!(
        matches!(
            h.internal_of(&primary),
            ThreadInternalState::AwaitingModel { .. }
        ),
        "the summary turn started on the quiescent head"
    );
    assert_eq!(h.scripted_data(&weave_id)["cp"], "prompted");
    let prompt_appended = h.sched.tasks[&primary]
        .conversation
        .messages()
        .iter()
        .any(|m| {
            m.role == whisper_agent_protocol::Role::User
                && m.content.iter().any(|b| {
                    matches!(
                        b,
                        ContentBlock::Text { text } if text.contains("Produce a compact summary")
                    )
                })
        });
    assert!(
        prompt_appended,
        "the resolved prompt rode in as a user message"
    );

    h.respond_model(
        &primary,
        vec![text_block("Here it is.\n<summary>ROLLED SUMMARY</summary>")],
        &mut pending_io,
    );

    let (continuation, source) = h.compaction_continuation(&weave_id);
    assert_eq!(
        source, primary,
        "the relationship edge points at the old head"
    );
    assert_eq!(
        h.sched.weaves[&weave_id].primary_thread_id(),
        Some(continuation.as_str()),
        "advance_head promoted the continuation"
    );
    let old_ref = h.sched.weaves[&weave_id]
        .threads
        .iter()
        .find(|r| r.thread_id == primary)
        .unwrap();
    assert!(!old_ref.ticks, "the old head is a dormant auxiliary");
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::Completed
    ));
    assert!(
        matches!(
            h.internal_of(&continuation),
            ThreadInternalState::AwaitingModel { .. }
        ),
        "the continuation's seeded first turn is running"
    );
    assert_eq!(
        h.sched.tasks[&continuation]
            .conversation
            .system_prompt_text(),
        "you are a test agent",
        "setup_from copied the old head's prefix — pod drift did not leak"
    );
    let seeded = h.sched.tasks[&continuation]
        .conversation
        .messages()
        .iter()
        .any(|m| {
            m.content.iter().any(|b| {
                matches!(
                    b,
                    ContentBlock::Text { text } if text.contains("ROLLED SUMMARY")
                        && text.contains("continues a previous conversation")
                )
            })
        });
    assert!(
        seeded,
        "the continuation template carried the extracted summary"
    );
    let data = h.scripted_data(&weave_id);
    assert_eq!(data["primary"], continuation.as_str());
    assert!(
        data["cp"].is_null(),
        "compaction state cleared after the roll"
    );

    h.respond_model(&continuation, vec![text_block("Resumed.")], &mut pending_io);
    assert!(matches!(
        h.internal_of(&continuation),
        ThreadInternalState::Completed
    ));
    assert_eq!(
        h.sched.weaves[&weave_id].threads.len(),
        3,
        "old head + title thread + continuation; no second title job"
    );
    assert!(
        h.sched.weaves[&weave_id]
            .effect_journal
            .records()
            .iter()
            .any(|r| matches!(
                &r.effect,
                PersistedDriverEffect::AdvanceHead { thread_id, .. } if thread_id == &continuation
            )),
        "the roll journaled its advance_head"
    );
}

/// Driver-owned auto-compaction (step 11 slice 5): the
/// `compaction.token_threshold` knob compares the cumulative usage the
/// `agent_completed` event now carries, issues `request_compaction`,
/// and the `compaction_ready` answer lands in the same activation's
/// drain — the summary turn starts immediately on the quiescent head.
#[tokio::test]
async fn titled_chat_threshold_crossing_requests_and_rolls() {
    let (mut h, primary, weave_id) = settled_titled_chat(
        [(
            "compaction.token_threshold".to_string(),
            serde_json::json!(50_000),
        )]
        .into_iter()
        .collect(),
    )
    .await;
    let mut pending_io = FuturesUnordered::new();
    assert!(
        !h.sched.weaves[&weave_id]
            .effect_journal
            .records()
            .iter()
            .any(|r| matches!(&r.effect, PersistedDriverEffect::RequestCompaction { .. })),
        "under the threshold no request fires"
    );

    h.sched
        .tasks
        .get_mut(&primary)
        .unwrap()
        .total_usage
        .input_tokens = 60_000;
    h.sched
        .send_user_message(&primary, "More.".into(), Vec::new(), &mut pending_io);
    h.sched.step_until_blocked(&primary, &mut pending_io);
    h.respond_model(&primary, vec![text_block("A long reply.")], &mut pending_io);

    assert!(
        matches!(
            h.internal_of(&primary),
            ThreadInternalState::AwaitingModel { .. }
        ),
        "the crossing close chained straight into the summary turn"
    );
    assert_eq!(h.scripted_data(&weave_id)["cp"], "prompted");
    assert!(
        h.sched.weaves[&weave_id]
            .effect_journal
            .records()
            .iter()
            .any(|r| matches!(
                (&r.effect, &r.outcome),
                (
                    PersistedDriverEffect::RequestCompaction { thread_id },
                    DriverEffectOutcome::Completed,
                ) if thread_id == &primary
            )),
        "the request journaled Completed"
    );

    h.respond_model(
        &primary,
        vec![text_block("<summary>AUTO SUMMARY</summary>")],
        &mut pending_io,
    );
    let (continuation, _) = h.compaction_continuation(&weave_id);
    assert_eq!(
        h.sched.weaves[&weave_id].primary_thread_id(),
        Some(continuation.as_str())
    );
    assert!(matches!(
        h.internal_of(&continuation),
        ThreadInternalState::AwaitingModel { .. }
    ));
    assert!(
        h.sched.tasks[&continuation]
            .conversation
            .messages()
            .iter()
            .any(|m| m.content.iter().any(|b| matches!(
                b,
                ContentBlock::Text { text } if text.contains("AUTO SUMMARY")
            ))),
        "the continuation carries the extracted summary"
    );
}

/// Extraction failure matches the builtin: the head stays Completed
/// and primary, no continuation spawns, the marker clears — and a
/// LATER manual compact is not wedged.
#[tokio::test]
async fn titled_chat_extraction_failure_leaves_the_head_standing() {
    let (mut h, primary, weave_id) = settled_titled_chat(Default::default()).await;
    let mut pending_io = FuturesUnordered::new();

    h.compact_message(&primary);
    h.respond_model(
        &primary,
        vec![text_block("I would rather not summarize.")],
        &mut pending_io,
    );
    assert_eq!(
        h.sched.weaves[&weave_id].primary_thread_id(),
        Some(primary.as_str()),
        "no roll happened"
    );
    assert_eq!(
        h.sched.weaves[&weave_id].threads.len(),
        2,
        "primary + title thread only — no continuation"
    );
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::Completed
    ));
    assert!(h.scripted_data(&weave_id)["cp"].is_null());

    // The failed attempt left nothing wedged: compact again, succeed.
    h.compact_message(&primary);
    assert_eq!(h.scripted_data(&weave_id)["cp"], "prompted");
    h.respond_model(
        &primary,
        vec![text_block("<summary>SECOND TRY</summary>")],
        &mut pending_io,
    );
    let (continuation, _) = h.compaction_continuation(&weave_id);
    assert_eq!(
        h.sched.weaves[&weave_id].primary_thread_id(),
        Some(continuation.as_str())
    );
}

/// Refusals on the scripted path: a compaction-disabled thread bounces
/// the manual message at the shared admission (no summary turn, no
/// driver state), the driver's own request gets `compaction_refused`
/// and clears its marker (journal Failed — auditable), and a
/// non-primary target refuses with the head named.
#[tokio::test]
async fn scripted_compaction_refusals_do_not_wedge() {
    let mut h = harness().await;
    h.install_driver("titled_chat", TITLED_CHAT_DRIVER);
    let mut pending_io = FuturesUnordered::new();
    let primary = h
        .sched
        .create_task(
            None,
            None,
            None,
            Some(ThreadConfigOverride {
                driver: Some(ThreadDriverConfig::Scripted {
                    name: "titled_chat".into(),
                    config: [(
                        "compaction.token_threshold".to_string(),
                        serde_json::json!(50_000),
                    )]
                    .into_iter()
                    .collect(),
                }),
                compaction: Some(whisper_agent_protocol::CompactionConfigOverride {
                    enabled: Some(false),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            None,
            None,
            None,
            None,
            None,
            None,
            &mut pending_io,
        )
        .unwrap();
    let weave_id = h.weave_of(&primary);
    h.sched
        .send_user_message(&primary, "Hello".into(), Vec::new(), &mut pending_io);
    h.sched.step_until_blocked(&primary, &mut pending_io);
    h.respond_model(&primary, vec![text_block("Hi.")], &mut pending_io);
    let title_thread = h.title_thread_of(&weave_id);
    h.respond_model(&title_thread, vec![text_block("Refusals")], &mut pending_io);

    // (a) Manual on a disabled thread: bounced at admission.
    h.compact_message(&primary);
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::Completed
    ));
    assert!(h.scripted_data(&weave_id)["cp"].is_null());

    // (b) The driver's own request refuses the same way and clears the
    // in-flight marker via compaction_refused.
    h.sched
        .tasks
        .get_mut(&primary)
        .unwrap()
        .total_usage
        .input_tokens = 60_000;
    h.sched
        .send_user_message(&primary, "More.".into(), Vec::new(), &mut pending_io);
    h.sched.step_until_blocked(&primary, &mut pending_io);
    h.respond_model(&primary, vec![text_block("Still here.")], &mut pending_io);
    assert!(
        matches!(h.internal_of(&primary), ThreadInternalState::Completed),
        "no summary turn — the request was refused"
    );
    assert!(h.scripted_data(&weave_id)["cp"].is_null());
    assert!(
        h.sched.weaves[&weave_id]
            .effect_journal
            .records()
            .iter()
            .any(|r| matches!(
                (&r.effect, &r.outcome),
                (
                    PersistedDriverEffect::RequestCompaction { .. },
                    DriverEffectOutcome::Failed { message },
                ) if message.contains("disabled")
            )),
        "the refusal journaled Failed"
    );

    // (c) A non-primary target refuses at the shared admission.
    let err = h
        .sched
        .deliver_manual_compaction(&weave_id, &title_thread, &mut pending_io)
        .unwrap_err();
    assert!(
        err.contains("not the weave's current primary"),
        "refusal names the head rule: {err}"
    );
}

/// Restart mid-summary-turn abandons the compaction (the builtin's
/// documented contract): the healed head's `thread_failed` clears the
/// driver's marker at the next activation, input revives the head, and
/// a fresh manual compact runs the full roll.
#[tokio::test]
async fn restart_mid_summary_turn_abandons_the_compaction() {
    let (mut h, primary, weave_id) = settled_titled_chat(Default::default()).await;
    h.compact_message(&primary);
    assert_eq!(h.scripted_data(&weave_id)["cp"], "prompted");
    let title_thread = h.title_thread_of(&weave_id);

    // Snapshot and restart: the persister heals the in-flight summary
    // turn to Failed before load.
    let weave = h.sched.weaves[&weave_id].clone();
    let mut threads = Vec::new();
    for tid in [&primary, &title_thread] {
        let mut task = h.sched.tasks[tid.as_str()].clone();
        if task.is_in_flight() {
            let mut events = Vec::new();
            task.heal_to_idle("task was in-flight at last shutdown", &mut events);
            task.fail("resume", "task was in-flight at last shutdown");
        }
        threads.push(task);
    }
    let mut fresh = harness().await;
    fresh.install_driver("titled_chat", TITLED_CHAT_DRIVER);
    fresh.sched.load_state(crate::pod::persist::LoadedState {
        pods: Vec::new(),
        threads,
        weaves: vec![weave],
    });

    // Input revives the head; the death notice cleared the marker
    // before the input event ran.
    let mut pending_io = FuturesUnordered::new();
    fresh
        .sched
        .send_user_message(&primary, "revive".into(), Vec::new(), &mut pending_io);
    fresh.sched.step_until_blocked(&primary, &mut pending_io);
    assert!(matches!(
        fresh.internal_of(&primary),
        ThreadInternalState::AwaitingModel { .. }
    ));
    let data = fresh.scripted_data(&weave_id);
    assert!(
        data["cp"].is_null(),
        "the abandoned compaction left no marker"
    );
    assert!(data["dead"].is_null(), "input revived the head");
    fresh.respond_model(&primary, vec![text_block("Back.")], &mut pending_io);
    assert!(matches!(
        fresh.internal_of(&primary),
        ThreadInternalState::Completed
    ));

    // The abandoned compaction does not poison a retry.
    fresh.compact_message(&primary);
    assert_eq!(fresh.scripted_data(&weave_id)["cp"], "prompted");
    fresh.respond_model(
        &primary,
        vec![text_block("<summary>AFTER RESTART</summary>")],
        &mut pending_io,
    );
    let (continuation, _) = fresh.compaction_continuation(&weave_id);
    assert_eq!(
        fresh.sched.weaves[&weave_id].primary_thread_id(),
        Some(continuation.as_str())
    );
}

/// `setup_from` is exclusive with explicit `system_prompt` /
/// `disable_tools`: the combination journals a Failed derive record
/// (auditable, not silent) and faults the activation.
#[tokio::test]
async fn derive_setup_from_is_exclusive_with_explicit_setup() {
    let mut h = harness().await;
    h.install_driver(
        "bad_inherit",
        r#"
            function on_event(state, event)
              if event.kind == "turn_start" then
                return { effects = { { kind = "derive_thread",
                  relationship = "x", setup_from = event.thread_id,
                  system_prompt = "boom" } }, state = state }
              end
              return { state = state }
            end
        "#,
    );
    let primary = h.create_scripted_thread("bad_inherit").unwrap();
    let weave_id = h.weave_of(&primary);
    let mut pending_io = FuturesUnordered::new();
    h.sched
        .send_user_message(&primary, "go".into(), Vec::new(), &mut pending_io);
    h.sched.step_until_blocked(&primary, &mut pending_io);
    let detail = h.sched.tasks[&primary]
        .failure_detail()
        .expect("the bad combination fails the activation's origin thread");
    assert!(detail.contains("exclusive"), "{detail}");
    assert!(
        h.sched.weaves[&weave_id]
            .effect_journal
            .records()
            .iter()
            .any(|r| matches!(
                (&r.effect, &r.outcome),
                (
                    PersistedDriverEffect::DeriveThread { thread_id: None, .. },
                    DriverEffectOutcome::Failed { message },
                ) if message.contains("exclusive")
            )),
        "static faults journal like every refusal"
    );
}

/// Review fix (slice 5): input supersedes an in-flight compaction.
/// Typing while the summary turn runs heals the in-flight call and
/// clears the driver's marker — the reply to the user's message is
/// NOT regex-tested as a summary (with the compaction prompt still in
/// context it could genuinely match and roll the weave, swallowing
/// the user's message), and a wedged marker from a faulted roll heals
/// the same way. A fresh manual compact afterwards runs normally.
#[tokio::test]
async fn input_during_summary_turn_supersedes_the_compaction() {
    let (mut h, primary, weave_id) = settled_titled_chat(Default::default()).await;
    let mut pending_io = FuturesUnordered::new();

    h.compact_message(&primary);
    assert_eq!(h.scripted_data(&weave_id)["cp"], "prompted");

    // The user keeps typing mid-summary-turn.
    h.sched.send_user_message(
        &primary,
        "actually, one more thing".into(),
        Vec::new(),
        &mut pending_io,
    );
    h.sched.step_until_blocked(&primary, &mut pending_io);
    assert!(
        h.scripted_data(&weave_id)["cp"].is_null(),
        "input cleared the in-flight compaction"
    );

    // Even a summary-shaped reply must not roll the weave now.
    h.respond_model(
        &primary,
        vec![text_block("<summary>NOT A SUMMARY</summary>")],
        &mut pending_io,
    );
    assert_eq!(
        h.sched.weaves[&weave_id].primary_thread_id(),
        Some(primary.as_str()),
        "no roll: the superseded compaction never finalizes"
    );
    assert!(matches!(
        h.internal_of(&primary),
        ThreadInternalState::Completed
    ));

    // The superseded compaction leaves nothing wedged.
    h.compact_message(&primary);
    assert_eq!(h.scripted_data(&weave_id)["cp"], "prompted");
    h.respond_model(
        &primary,
        vec![text_block("<summary>REAL SUMMARY</summary>")],
        &mut pending_io,
    );
    let (continuation, _) = h.compaction_continuation(&weave_id);
    assert_eq!(
        h.sched.weaves[&weave_id].primary_thread_id(),
        Some(continuation.as_str())
    );
}

// ---------- step 11 slice 6: the builtin conversion ----------

/// Every creation path resolves the scripted default (step 11 slice
/// 6): a plain create gets the embedded titled_chat, and an explicit
/// builtin override from a stale client maps to the pod default
/// instead of erroring — nothing constructs the tombstone at runtime.
#[tokio::test]
async fn new_threads_default_to_the_embedded_scripted_driver() {
    let mut h = harness().await;
    let plain = h.create_thread();
    let weave_id = h.weave_of(&plain);
    assert!(matches!(
        &h.sched.weaves[&weave_id].driver,
        ThreadDriverConfig::Scripted { name, .. } if name == "titled_chat"
    ));

    let mut pending_io = FuturesUnordered::new();
    let stale = h
        .sched
        .create_task(
            None,
            None,
            None,
            Some(ThreadConfigOverride {
                driver: Some(ThreadDriverConfig::BuiltinSingleAgentChat),
                ..Default::default()
            }),
            None,
            None,
            None,
            None,
            None,
            None,
            &mut pending_io,
        )
        .unwrap();
    let stale_weave = h.weave_of(&stale);
    assert!(
        matches!(
            &h.sched.weaves[&stale_weave].driver,
            ThreadDriverConfig::Scripted { name, .. } if name == "titled_chat"
        ),
        "explicit builtin override maps to the pod default"
    );

    // The embedded program resolves without any pod file; a pod file
    // with the same stem shadows it.
    let embedded = h
        .sched
        .load_driver_program(TEST_POD, "titled_chat")
        .expect("embedded default resolves in a pod with no drivers dir");
    assert!(embedded.contains("Titled chat"));
    h.install_driver(
        "titled_chat",
        "-- shadowed\nfunction on_event(s) return nil end",
    );
    let shadowed = h
        .sched
        .load_driver_program(TEST_POD, "titled_chat")
        .unwrap();
    assert!(shadowed.starts_with("-- shadowed"), "pod file wins");
    let listed = h.sched.list_driver_names(TEST_POD);
    assert_eq!(
        listed,
        vec![("titled_chat".to_string(), false)],
        "a shadowing pod file lists once, as the pod entry"
    );
}

/// Load-time migration (step 11 slice 6): a weave persisted with the
/// builtin driver — mid-compaction marker and all — converts to the
/// scripted default with knobs translated from its primary's legacy
/// config, the thread's own config rewrites the same way, and the
/// migrated weave actually DRIVES: input runs a turn with no title job
/// (existing titles stand), and a manual compact rolls the head.
#[tokio::test]
async fn legacy_builtin_weaves_migrate_and_drive() {
    let mut h = harness().await;
    let seed = h.create_thread();
    let mut task = h.sched.tasks[&seed].clone();
    task.config.driver = ThreadDriverConfig::BuiltinSingleAgentChat;
    task.config.autoquery.enabled = true;
    task.config.compaction.token_threshold = Some(50_000);
    task.title = Some("Legacy thread".into());
    let mut weave = crate::runtime::weave::Weave::singleton_for_thread(
        task.id.clone(),
        task.pod_id.clone(),
        ThreadDriverConfig::BuiltinSingleAgentChat,
    );
    weave.driver_state = DriverState::BuiltinSingleAgentChat {
        cycles_started: 4,
        turns_in_cycle: 1,
        compacting: Some(task.id.clone()),
    };

    let mut fresh = harness().await;
    fresh.sched.load_state(crate::pod::persist::LoadedState {
        pods: Vec::new(),
        threads: vec![task],
        weaves: vec![weave],
    });

    let weave = &fresh.sched.weaves[&seed];
    match &weave.driver {
        ThreadDriverConfig::Scripted { name, config } => {
            assert_eq!(name, "titled_chat");
            assert_eq!(config.get("autoquery"), Some(&serde_json::json!(true)));
            assert_eq!(
                config.get("compaction.token_threshold"),
                Some(&serde_json::json!(50_000)),
                "legacy auto-compact threshold carries into the knob"
            );
        }
        other => panic!("weave did not migrate: {other:?}"),
    }
    match &weave.driver_state {
        DriverState::Scripted { data, .. } => {
            assert_eq!(data["primary"], seed.as_str());
            assert_eq!(
                data["title_requested"], true,
                "existing threads are not retro-titled"
            );
        }
        other => panic!("driver state did not migrate: {other:?}"),
    }
    assert!(
        matches!(
            &fresh.sched.tasks[&seed].config.driver,
            ThreadDriverConfig::Scripted { name, .. } if name == "titled_chat"
        ),
        "the thread's own config rewrites so forks inherit a live driver"
    );

    // The migrated weave drives: input runs a turn, the reply closes
    // the cycle, and no title thread derives.
    let mut pending_io = FuturesUnordered::new();
    fresh
        .sched
        .send_user_message(&seed, "still here?".into(), Vec::new(), &mut pending_io);
    fresh.sched.step_until_blocked(&seed, &mut pending_io);
    assert!(matches!(
        fresh.internal_of(&seed),
        ThreadInternalState::AwaitingModel { .. }
    ));
    fresh.respond_model(&seed, vec![text_block("Still here.")], &mut pending_io);
    assert!(matches!(
        fresh.internal_of(&seed),
        ThreadInternalState::Completed
    ));
    assert_eq!(
        fresh.sched.weaves[&seed].threads.len(),
        1,
        "no title job on a migrated thread"
    );
    assert_eq!(
        fresh.sched.tasks[&seed].title.as_deref(),
        Some("Legacy thread")
    );

    // Manual compaction routes through the driver on the migrated
    // weave — the builtin Function machinery is gone.
    fresh.compact_message(&seed);
    assert_eq!(fresh.scripted_data(&seed)["cp"], "prompted");
    fresh.respond_model(
        &seed,
        vec![text_block("<summary>MIGRATED SUMMARY</summary>")],
        &mut pending_io,
    );
    let (continuation, _) = fresh.compaction_continuation(&seed);
    assert_eq!(
        fresh.sched.weaves[&seed].primary_thread_id(),
        Some(continuation.as_str())
    );
}

/// Dispatch children inherit the scripted default too (step 11 slice
/// 6): a titled_chat parent's async dispatch spawns a child whose own
/// singleton weave runs titled_chat — the intermediate shape until
/// derived work moves inside the parent weave.
#[tokio::test]
async fn dispatch_children_get_the_scripted_default() {
    let mut h = harness().await;
    h.install_driver("titled_chat", TITLED_CHAT_DRIVER);
    let primary = h.create_scripted_thread("titled_chat").unwrap();
    let weave_id = h.weave_of(&primary);
    let mut pending_io = FuturesUnordered::new();
    h.sched
        .send_user_message(&primary, "spawn".into(), Vec::new(), &mut pending_io);
    h.sched.step_until_blocked(&primary, &mut pending_io);
    h.respond_model(
        &primary,
        vec![tool_use_block_with_input(
            "toolu-child",
            "dispatch_thread",
            serde_json::json!({"prompt": "child work", "sync": false}),
        )],
        &mut pending_io,
    );
    let child = h.dispatch_child_of(&weave_id);
    let child_weave = h.weave_of(&child);
    assert!(matches!(
        &h.sched.weaves[&child_weave].driver,
        ThreadDriverConfig::Scripted { name, .. } if name == "titled_chat"
    ));
}
