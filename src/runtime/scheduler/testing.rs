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
use crate::runtime::weave::WeaveThreadRole;
use std::sync::atomic::{AtomicU64, Ordering};
use whisper_agent_protocol::{
    AllowMap, GenerationContext, Message, PodAllow, PodConfig, PodLimits, PodModifyCap,
    ThreadDefaultCaps, ThreadDefaults,
};

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
            backends: vec!["anthropic".into()],
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
            &mut pending_io,
        )
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
