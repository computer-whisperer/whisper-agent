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
use crate::runtime::driver::{
    DriverEffectOutcome, EntryRef, PersistedDriverEffect, ThreadRelationship,
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

// ---------- dormant queue preservation ----------

#[tokio::test]
async fn dormant_thread_keeps_nudges_and_followups_queued() {
    let mut h = harness().await;
    let alpha = h.create_thread();
    let weave_id = h.weave_of(&alpha);
    h.sched
        .weave_release_ticker(&weave_id, &alpha)
        .expect("release at idle");

    {
        let task = h.sched.tasks.get_mut(&alpha).unwrap();
        task.pending_knowledge_nudges.push("nudge text".into());
        task.pending_tool_result_followups
            .push("<dispatched-thread-notification>done</dispatched-thread-notification>".into());
    }

    let mut pending_io = FuturesUnordered::new();
    h.sched.step_until_blocked(&alpha, &mut pending_io);

    let task = &h.sched.tasks[&alpha];
    assert!(task.is_idle(), "dormant thread did not fail or run");
    assert_eq!(
        task.pending_knowledge_nudges.len(),
        1,
        "nudge stays queued until a weave adopts the thread"
    );
    assert_eq!(
        task.pending_tool_result_followups.len(),
        1,
        "follow-up stays queued until a weave adopts the thread"
    );
}

// ---------- step 8: compaction as a weave head-advance ----------

/// Full builtin compaction lifecycle on weave machinery: the summary
/// turn runs under the weave's compacting marker; on cycle finish the
/// scheduler derives a continuation along a `compaction` edge, advances
/// the head (journaled), demotes the old thread to a dormant auxiliary,
/// and seeds the continuation from the extracted summary.
#[tokio::test]
async fn builtin_compaction_rolls_the_weave_head() {
    use crate::functions::{CallerLink, Function};
    let mut h = harness().await;
    let t1 = h.create_thread();
    let weave_id = h.weave_of(&t1);
    let mut pending_io = FuturesUnordered::new();

    // An ordinary turn first, so the thread has history worth rolling.
    h.sched
        .send_user_message(&t1, "hello there".into(), Vec::new(), &mut pending_io);
    h.sched.step_until_blocked(&t1, &mut pending_io);
    h.respond_model(&t1, vec![text_block("hi — done")], &mut pending_io);

    // Launch the compaction Function with the weave as caller (the
    // auto-trigger's shape).
    let fn_id = h
        .sched
        .register_function(
            Function::CompactThread {
                thread_id: t1.clone(),
            },
            CallerLink::Weave {
                weave_id: weave_id.clone(),
            },
        )
        .expect("primary idle builtin thread admits compaction");
    h.sched.launch_function(fn_id, &mut pending_io);

    // The summary turn is in flight and the weave carries the marker.
    assert!(matches!(
        h.internal_of(&t1),
        ThreadInternalState::AwaitingModel { .. }
    ));
    assert!(matches!(
        &h.sched.weaves[&weave_id].driver_state,
        crate::runtime::driver::DriverState::BuiltinSingleAgentChat {
            compacting: Some(t),
            ..
        } if t == &t1
    ));
    // Forks are refused mid-compaction at the scheduler level.
    assert!(
        h.sched
            .fork_task(None, None, &t1, 2, false, &mut pending_io)
            .is_err(),
        "fork during compaction must be refused"
    );
    // A second compaction is refused while one is running.
    assert!(matches!(
        h.sched.register_function(
            Function::CompactThread {
                thread_id: t1.clone(),
            },
            CallerLink::Weave {
                weave_id: weave_id.clone(),
            },
        ),
        Err(crate::functions::RejectReason::PreconditionFailed { .. })
    ));

    // The model returns the summary; the finalize rolls the head.
    h.respond_model(
        &t1,
        vec![text_block("<summary>\nthe distilled past\n</summary>")],
        &mut pending_io,
    );

    let weave = &h.sched.weaves[&weave_id];
    assert_eq!(weave.threads.len(), 2, "old head + continuation");
    let old_ref = weave
        .threads
        .iter()
        .find(|r| r.thread_id == t1)
        .expect("old head stays referenced");
    assert_eq!(old_ref.role, WeaveThreadRole::Auxiliary);
    assert!(!old_ref.ticks, "old head is dormant — frozen history");
    let new_ref = weave
        .threads
        .iter()
        .find(|r| r.thread_id != t1)
        .expect("continuation referenced");
    let t2 = new_ref.thread_id.clone();
    assert_eq!(new_ref.role, WeaveThreadRole::Primary);
    assert!(new_ref.ticks);
    let rel = new_ref
        .relationship
        .as_ref()
        .expect("promoted head keeps its lineage edge");
    assert_eq!(rel.kind, "compaction");
    assert_eq!(
        rel.source,
        Some(EntryRef {
            thread_id: t1.clone(),
            entry_index: None
        })
    );

    // Ticker index followed the head.
    assert!(!h.sched.thread_ticker.contains_key(&t1));
    assert_eq!(h.sched.thread_ticker.get(&t2), Some(&weave_id));

    // The marker cleared and the Function completed.
    assert!(matches!(
        &h.sched.weaves[&weave_id].driver_state,
        crate::runtime::driver::DriverState::BuiltinSingleAgentChat {
            compacting: None,
            ..
        }
    ));
    assert!(
        !h.sched.active_functions.contains_key(&fn_id),
        "CompactThread Function reached its terminal"
    );

    // Journal explains the roll: a derive along the compaction edge,
    // then the head-advance naming both threads.
    let records = h.sched.weaves[&weave_id].effect_journal.records();
    assert!(records.iter().any(|r| matches!(
        &r.effect,
        PersistedDriverEffect::DeriveThread {
            thread_id: Some(t),
            relationship,
            ..
        } if t == &t2 && relationship.kind == "compaction"
    )));
    assert!(
        records
            .iter()
            .any(|r| r.outcome == DriverEffectOutcome::Completed
                && matches!(
                    &r.effect,
                    PersistedDriverEffect::AdvanceHead {
                        thread_id,
                        previous: Some(prev)
                    } if thread_id == &t2 && prev == &t1
                ))
    );

    // The continuation inherited the parent's setup and got the summary
    // seed; its first model turn is already in flight.
    let t2_task = &h.sched.tasks[&t2];
    assert!(
        t2_task
            .conversation
            .messages()
            .iter()
            .any(|m| m.content.iter().any(|b| matches!(
                b,
                ContentBlock::Text { text } if text.contains("the distilled past")
            ))),
        "continuation seeded with the extracted summary"
    );
    assert!(matches!(
        h.internal_of(&t2),
        ThreadInternalState::AwaitingModel { .. }
    ));

    // Summaries stay coherent: the dormant old head is tagged from the
    // weave ref (sidebar nesting + compose-box gating), the new head is
    // primary.
    let old_summary = h.sched.decorate_summary(h.sched.tasks[&t1].summary());
    assert_eq!(old_summary.weave_id.as_deref(), Some(weave_id.as_str()));
    assert_eq!(
        old_summary.weave_role,
        Some(whisper_agent_protocol::weave::WeaveThreadRole::Auxiliary)
    );
    let new_summary = h.sched.decorate_summary(h.sched.tasks[&t2].summary());
    assert_eq!(
        new_summary.weave_role,
        Some(whisper_agent_protocol::weave::WeaveThreadRole::Primary)
    );

    // Input to the demoted head is refused (dormant), and it can never
    // compact again (not primary).
    let before = h.sched.tasks[&t1].conversation.messages().len();
    h.sched
        .send_user_message(&t1, "keep talking?".into(), Vec::new(), &mut pending_io);
    assert_eq!(
        h.sched.tasks[&t1].conversation.messages().len(),
        before,
        "dormant old head rejects input"
    );
    assert!(
        h.sched
            .register_function(
                Function::CompactThread {
                    thread_id: t1.clone(),
                },
                CallerLink::Weave {
                    weave_id: weave_id.clone(),
                },
            )
            .is_err()
    );

    // The degenerate presentation now shows the new head with the old
    // context in the drill-down list.
    assert_eq!(
        h.sched.weaves[&weave_id].wire_snapshot().presentation,
        vec![
            PresentationBlock::PrimaryTranscript {
                thread_id: t2.clone()
            },
            PresentationBlock::ThreadList {
                thread_ids: vec![t1.clone()]
            }
        ]
    );
}

/// A summary turn that yields no `<summary>` block clears the marker,
/// completes the Function as an error, and leaves the weave unrolled.
#[tokio::test]
async fn compaction_without_summary_errors_and_leaves_head_in_place() {
    use crate::functions::{CallerLink, Function};
    let mut h = harness().await;
    let t1 = h.create_thread();
    let weave_id = h.weave_of(&t1);
    let mut pending_io = FuturesUnordered::new();

    h.sched
        .send_user_message(&t1, "hello".into(), Vec::new(), &mut pending_io);
    h.sched.step_until_blocked(&t1, &mut pending_io);
    h.respond_model(&t1, vec![text_block("hi")], &mut pending_io);

    let fn_id = h
        .sched
        .register_function(
            Function::CompactThread {
                thread_id: t1.clone(),
            },
            CallerLink::Weave {
                weave_id: weave_id.clone(),
            },
        )
        .unwrap();
    h.sched.launch_function(fn_id, &mut pending_io);
    h.respond_model(
        &t1,
        vec![text_block("I would rather chat than summarize.")],
        &mut pending_io,
    );

    let weave = &h.sched.weaves[&weave_id];
    assert_eq!(weave.threads.len(), 1, "no continuation was derived");
    assert_eq!(weave.primary_thread_id(), Some(t1.as_str()));
    assert!(matches!(
        &weave.driver_state,
        crate::runtime::driver::DriverState::BuiltinSingleAgentChat {
            compacting: None,
            ..
        }
    ));
    assert!(!h.sched.active_functions.contains_key(&fn_id));
    assert_eq!(h.sched.thread_ticker.get(&t1), Some(&weave_id));
}

/// Cancelling the thread mid-summary-turn abandons the compaction
/// precisely: marker cleared, Function resolved, no continuation — and
/// the thread is a normal cancelled thread afterwards.
#[tokio::test]
async fn cancel_mid_compaction_clears_marker_and_resolves_function() {
    use crate::functions::{CallerLink, Function};
    let mut h = harness().await;
    let t1 = h.create_thread();
    let weave_id = h.weave_of(&t1);
    let mut pending_io = FuturesUnordered::new();

    h.sched
        .send_user_message(&t1, "hello".into(), Vec::new(), &mut pending_io);
    h.sched.step_until_blocked(&t1, &mut pending_io);
    h.respond_model(&t1, vec![text_block("hi")], &mut pending_io);

    let fn_id = h
        .sched
        .register_function(
            Function::CompactThread {
                thread_id: t1.clone(),
            },
            CallerLink::Weave {
                weave_id: weave_id.clone(),
            },
        )
        .unwrap();
    h.sched.launch_function(fn_id, &mut pending_io);
    assert!(matches!(
        h.internal_of(&t1),
        ThreadInternalState::AwaitingModel { .. }
    ));

    h.sched.execute_cancel_thread(&t1, &mut pending_io);

    assert!(matches!(
        &h.sched.weaves[&weave_id].driver_state,
        crate::runtime::driver::DriverState::BuiltinSingleAgentChat {
            compacting: None,
            ..
        }
    ));
    assert!(
        !h.sched.active_functions.contains_key(&fn_id),
        "abandoned CompactThread Function resolved as Cancelled"
    );
    assert_eq!(
        h.sched.weaves[&weave_id].threads.len(),
        1,
        "no continuation from a cancelled summary turn"
    );
}

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

/// A summary turn that fails (provider error) releases the weave's
/// compacting marker and errors the Function — the weave doesn't stay
/// "busy" until some unrelated Completed turn trips the finalize.
#[tokio::test]
async fn failed_summary_turn_releases_the_compacting_marker() {
    use crate::functions::{CallerLink, Function};
    let mut h = harness().await;
    let t1 = h.create_thread();
    let weave_id = h.weave_of(&t1);
    let mut pending_io = FuturesUnordered::new();

    h.sched
        .send_user_message(&t1, "hello".into(), Vec::new(), &mut pending_io);
    h.sched.step_until_blocked(&t1, &mut pending_io);
    h.respond_model(&t1, vec![text_block("hi")], &mut pending_io);

    let fn_id = h
        .sched
        .register_function(
            Function::CompactThread {
                thread_id: t1.clone(),
            },
            CallerLink::Weave {
                weave_id: weave_id.clone(),
            },
        )
        .unwrap();
    h.sched.launch_function(fn_id, &mut pending_io);

    // Provider dies mid-summary-turn.
    let op_id = match h.internal_of(&t1) {
        ThreadInternalState::AwaitingModel { op_id, .. } => *op_id,
        other => panic!("expected summary turn in flight, got {other:?}"),
    };
    let mut events = Vec::new();
    h.sched.tasks.get_mut(&t1).unwrap().apply_io_result(
        op_id,
        IoResult::ModelCall(Err("provider exploded".into())),
        &mut events,
    );
    h.sched.step_until_blocked(&t1, &mut pending_io);

    assert!(matches!(
        &h.sched.weaves[&weave_id].driver_state,
        crate::runtime::driver::DriverState::BuiltinSingleAgentChat {
            compacting: None,
            ..
        }
    ));
    assert!(
        !h.sched.active_functions.contains_key(&fn_id),
        "Function resolved as execution error"
    );
    assert_eq!(h.sched.weaves[&weave_id].threads.len(), 1);
    assert_eq!(
        h.sched.weaves[&weave_id].primary_thread_id(),
        Some(t1.as_str())
    );
}

/// A compacting marker persisted at shutdown is cleared at load: the
/// persister heals the in-flight summary turn to Failed and the
/// Function registry is in-memory only, so a surviving marker would
/// wedge admission and mis-trigger the finalize on the next ordinary
/// Completed turn.
#[tokio::test]
async fn load_clears_a_persisted_compacting_marker() {
    let mut h = harness().await;
    let t1 = h.create_thread();
    let weave_id = h.weave_of(&t1);

    // Simulate the previous process: marker set on the persisted weave.
    let mut weave = h.sched.weaves[&weave_id].clone();
    if let crate::runtime::driver::DriverState::BuiltinSingleAgentChat { compacting, .. } =
        &mut weave.driver_state
    {
        *compacting = Some(t1.clone());
    }

    let mut fresh = harness().await;
    fresh.sched.load_state(crate::pod::persist::LoadedState {
        pods: Vec::new(),
        threads: Vec::new(),
        weaves: vec![weave],
    });
    assert!(matches!(
        &fresh.sched.weaves[&weave_id].driver_state,
        crate::runtime::driver::DriverState::BuiltinSingleAgentChat {
            compacting: None,
            ..
        }
    ));
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

    // Journal: 2 derives, 2 voice turns, 3 finishes (2 voices + the
    // primary's round close), 6 appends (input to 2 voices, then each
    // reply to minutes + the other voice) — all Completed.
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
        2
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
        3,
        "second round derives no new threads"
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
        4,
        "replacement derived, dead voice still referenced"
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
    let mut pending_io2 = FuturesUnordered::new();
    fresh
        .sched
        .send_user_message(&primary, "round three".into(), Vec::new(), &mut pending_io2);
    fresh.sched.step_until_blocked(&primary, &mut pending_io2);
    assert_eq!(
        fresh.sched.weaves[&weave_id].threads.len(),
        4,
        "no re-derivation on the second post-restart round"
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
