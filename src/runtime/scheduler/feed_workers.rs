//! Per-tracked-bucket [`FeedWorker`](crate::knowledge::FeedWorker)
//! spawning + lifecycle. Sibling of [`buckets`](super::buckets).
//!
//! `Scheduler::new` calls [`spawn_for_registry`] once at construction
//! time to launch one worker per tracked-source bucket, storing the
//! resulting [`CancellationToken`]s on the scheduler so a
//! `DeleteBucket` can cleanly stop the worker.
//!
//! Today's worker is scaffolding-only — it ticks on cadence, logs,
//! and stops on cancellation. The actual delta-polling and delta-
//! application slices plug into the loop's NoOp branch in
//! `crate::knowledge::feed_worker` without changing the spawn-side
//! shape here.

use std::collections::HashMap;
use std::sync::Arc;

use tokio_util::sync::CancellationToken;
use tracing::info;

use crate::knowledge::config::{SourceConfig, TrackedDriver};
use crate::knowledge::source::feed::driver_for;
use crate::knowledge::{
    BucketRegistry, FeedWorker, NoopObserver, WorkerObserver, cadence_to_duration,
};

/// Walk the registry, spawn one [`FeedWorker`] per tracked bucket,
/// and return their cancellation tokens keyed by bucket id. The
/// scheduler stores the result and pulls a token out on
/// `DeleteBucket` to stop the corresponding worker.
///
/// Stored / linked / managed buckets are skipped — only `tracked`
/// buckets get a worker.
pub fn spawn_for_registry(registry: &BucketRegistry) -> HashMap<String, CancellationToken> {
    spawn_for_registry_with_observer(registry, Arc::new(NoopObserver))
}

/// Variant that lets callers (tests) supply a custom observer for
/// per-tick telemetry. Production wiring uses [`NoopObserver`] via
/// [`spawn_for_registry`].
pub fn spawn_for_registry_with_observer(
    registry: &BucketRegistry,
    observer: Arc<dyn WorkerObserver>,
) -> HashMap<String, CancellationToken> {
    let mut tokens: HashMap<String, CancellationToken> = HashMap::new();
    for (id, entry) in &registry.buckets {
        let SourceConfig::Tracked {
            driver: driver_cfg,
            delta_cadence,
            ..
        } = &entry.config.source
        else {
            continue;
        };
        let cancel = CancellationToken::new();
        let driver = driver_for(driver_cfg);
        let interval = cadence_to_duration(*delta_cadence);
        let driver_name = match driver_cfg {
            TrackedDriver::Wikipedia { language, .. } => format!("wikipedia:{language}"),
        };
        info!(
            bucket_id = %id,
            driver = %driver_name,
            interval_secs = interval.map(|d| d.as_secs()).unwrap_or(0),
            "spawning feed worker for tracked bucket",
        );
        let worker = FeedWorker::new(
            id.clone(),
            entry.dir.clone(),
            driver,
            interval,
            cancel.clone(),
            observer.clone(),
        );
        tokio::spawn(worker.run());
        tokens.insert(id.clone(), cancel);
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::sync::Mutex;

    use crate::knowledge::TickOutcome;

    /// Test helper: write a bucket.toml under `<root>/<id>/` and return
    /// the registry that loads from `<root>`.
    async fn make_registry_with_buckets(root: &Path, buckets: &[(&str, &str)]) -> BucketRegistry {
        for (id, toml) in buckets {
            let dir = root.join(id);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("bucket.toml"), toml).unwrap();
        }
        BucketRegistry::load(root.to_path_buf()).await.unwrap()
    }

    fn tracked_wikipedia_toml(name: &str, language: &str, cadence: &str) -> String {
        format!(
            r#"
name = "{name}"
created_at = "2026-04-26T00:00:00Z"

[source]
kind = "tracked"
driver = "wikipedia"
language = "{language}"
delta_cadence = "{cadence}"

[defaults]
embedder = "test_embedder"
"#
        )
    }

    fn linked_markdown_toml(name: &str, path: &Path) -> String {
        format!(
            r#"
name = "{name}"
created_at = "2026-04-26T00:00:00Z"

[source]
kind = "linked"
adapter = "markdown_dir"
path = "{}"

[defaults]
embedder = "test_embedder"
"#,
            path.display()
        )
    }

    #[derive(Default)]
    struct Recording {
        ticks: Mutex<Vec<(String, u64, TickOutcome)>>,
    }

    impl Recording {
        fn record(&self, bucket_id: &str, idx: u64, outcome: TickOutcome) {
            self.ticks
                .lock()
                .unwrap()
                .push((bucket_id.to_string(), idx, outcome));
        }
    }

    /// `WorkerObserver` doesn't carry the bucket id today (it lives on
    /// the worker, not on the trait method) — so this observer just
    /// records ticks anonymously and the test asserts on count + the
    /// shape of outcomes.
    struct ObserverBox(Arc<Recording>);
    impl WorkerObserver for ObserverBox {
        fn on_tick(&self, idx: u64, outcome: TickOutcome) {
            self.0.record("(anonymous)", idx, outcome);
        }
    }

    #[tokio::test]
    async fn skips_non_tracked_buckets() {
        let tmp = tempfile::tempdir().unwrap();
        let linked_path = tmp.path().join("notes");
        std::fs::create_dir_all(&linked_path).unwrap();
        let registry = make_registry_with_buckets(
            tmp.path(),
            &[
                (
                    "wiki_simple",
                    &tracked_wikipedia_toml("Simple Wiki", "simple", "manual"),
                ),
                ("my_notes", &linked_markdown_toml("My Notes", &linked_path)),
            ],
        )
        .await;

        let tokens = spawn_for_registry(&registry);

        assert_eq!(tokens.len(), 1, "only the tracked bucket gets a worker");
        assert!(tokens.contains_key("wiki_simple"));
        assert!(!tokens.contains_key("my_notes"));

        // Clean up — cancel the worker so the spawned task winds down
        // cleanly and the test process doesn't leave it hanging.
        for token in tokens.values() {
            token.cancel();
        }
    }

    #[tokio::test]
    async fn manual_cadence_worker_emits_idle_tick() {
        let tmp = tempfile::tempdir().unwrap();
        let registry = make_registry_with_buckets(
            tmp.path(),
            &[(
                "wiki_simple",
                &tracked_wikipedia_toml("Simple Wiki", "simple", "manual"),
            )],
        )
        .await;

        let recording = Arc::new(Recording::default());
        let observer = Arc::new(ObserverBox(recording.clone()));
        let tokens = spawn_for_registry_with_observer(&registry, observer);
        assert_eq!(tokens.len(), 1);

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        for token in tokens.values() {
            token.cancel();
        }
        // Yield enough for the worker tasks to finish their idle-then-
        // cancel handshake.
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }

        let ticks = recording.ticks.lock().unwrap().clone();
        assert!(
            ticks.iter().any(|(_, _, o)| *o == TickOutcome::Idle),
            "expected at least one Idle tick from manual-cadence worker, got {ticks:?}"
        );
    }
}
