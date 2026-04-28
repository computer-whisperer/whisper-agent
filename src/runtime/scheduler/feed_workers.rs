//! Per-tracked-bucket [`FeedWorker`](crate::knowledge::FeedWorker)
//! spawning + lifecycle. Sibling of [`buckets`](super::buckets).
//!
//! `Scheduler::new` calls [`spawn_for_registry`] once at construction
//! time to launch one worker per tracked-source bucket, storing the
//! resulting [`CancellationToken`]s on the scheduler so a
//! `DeleteBucket` can cleanly stop the worker.
//!
//! Each worker grabs the configured embedding provider for its bucket
//! at spawn time — the same provider the build pipeline used. If the
//! provider isn't in the scheduler's catalog, the worker is skipped
//! (logged) rather than spawned in a doomed state. Reconfiguring a
//! bucket's embedder rebuilds the worker; per-tick re-lookup isn't
//! plumbed through.

use std::collections::HashMap;
use std::sync::Arc;

use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::knowledge::config::{SourceConfig, TrackedDriver};
use crate::knowledge::source::feed::driver_for;
use crate::knowledge::{
    BucketRegistry, FeedWorker, NoopObserver, WorkerObserver, cadence_to_duration,
};
use crate::runtime::scheduler::EmbeddingProviderEntry;

/// Walk the registry, spawn one [`FeedWorker`] per tracked bucket,
/// and return their cancellation tokens keyed by bucket id. The
/// scheduler stores the result and pulls a token out on
/// `DeleteBucket` to stop the corresponding worker.
///
/// Stored / linked / managed buckets are skipped — only `tracked`
/// buckets get a worker. Tracked buckets whose configured embedder
/// isn't present in `embedding_providers` are also skipped (logged):
/// the worker would error every tick on `set_embedder`, so it's
/// cleaner to not spawn it at all.
pub fn spawn_for_registry(
    registry: &BucketRegistry,
    embedding_providers: &HashMap<String, EmbeddingProviderEntry>,
) -> HashMap<String, CancellationToken> {
    spawn_for_registry_with_observer(registry, embedding_providers, Arc::new(NoopObserver))
}

/// Variant that lets callers (tests) supply a custom observer for
/// per-tick telemetry. Production wiring uses [`NoopObserver`] via
/// [`spawn_for_registry`].
pub fn spawn_for_registry_with_observer(
    registry: &BucketRegistry,
    embedding_providers: &HashMap<String, EmbeddingProviderEntry>,
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
        let embedder_name = &entry.config.defaults.embedder;
        let Some(provider_entry) = embedding_providers.get(embedder_name) else {
            warn!(
                bucket_id = %id,
                embedder = %embedder_name,
                "tracked bucket's configured embedder is not in the provider catalog; \
                 skipping feed worker spawn",
            );
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
            embedder = %embedder_name,
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
            registry.clone(),
            provider_entry.provider.clone(),
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

    /// Build a one-entry embedding-providers map keyed by `name` whose
    /// provider returns deterministic 8-dim vectors. Tests only need
    /// the embedder Arc to exist — actual `embed` calls happen during
    /// `apply_delta` which the spawn-side tests don't exercise.
    fn embedders_with_test_entry(name: &str) -> HashMap<String, EmbeddingProviderEntry> {
        let provider: Arc<dyn crate::providers::embedding::EmbeddingProvider> =
            Arc::new(test_embedder::TestEmbedder);
        let mut map = HashMap::new();
        map.insert(
            name.to_string(),
            EmbeddingProviderEntry {
                provider,
                kind: "test".to_string(),
                auth_mode: None,
                source: crate::pod::config::EmbeddingProviderConfig::Tei {
                    endpoint: "http://test.invalid".to_string(),
                    auth: None,
                },
            },
        );
        map
    }

    /// Lightweight `EmbeddingProvider` for spawn-side tests. Never
    /// actually called — the apply path requires a built bucket which
    /// these tests don't construct.
    mod test_embedder {
        use crate::providers::embedding::{
            BoxFuture, EmbedRequest, EmbeddingError, EmbeddingModelInfo, EmbeddingProvider,
            EmbeddingResponse,
        };
        use tokio_util::sync::CancellationToken;

        pub struct TestEmbedder;
        impl EmbeddingProvider for TestEmbedder {
            fn embed<'a>(
                &'a self,
                _req: &'a EmbedRequest<'a>,
                _cancel: &'a CancellationToken,
            ) -> BoxFuture<'a, Result<EmbeddingResponse, EmbeddingError>> {
                Box::pin(async {
                    Err(EmbeddingError::Transport(
                        "TestEmbedder is spawn-only; not callable from apply path".into(),
                    ))
                })
            }
            fn list_models<'a>(
                &'a self,
            ) -> BoxFuture<'a, Result<Vec<EmbeddingModelInfo>, EmbeddingError>> {
                Box::pin(async {
                    Ok(vec![EmbeddingModelInfo {
                        id: "test-embedder".to_string(),
                        dimension: 8,
                        max_input_tokens: None,
                    }])
                })
            }
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

        let providers = embedders_with_test_entry("test_embedder");
        let tokens = spawn_for_registry(&registry, &providers);

        assert_eq!(tokens.len(), 1, "only the tracked bucket gets a worker");
        assert!(tokens.contains_key("wiki_simple"));
        assert!(!tokens.contains_key("my_notes"));

        // Clean up — cancel the worker so the spawned task winds down
        // cleanly and the test process doesn't leave it hanging.
        for token in tokens.values() {
            token.cancel();
        }
    }

    /// A tracked bucket whose `[defaults] embedder` isn't in the
    /// catalog gets skipped entirely. The worker would otherwise hit
    /// every tick with a `set_embedder` failure — cleaner to not
    /// spawn one at all and surface the misconfiguration via the
    /// startup log.
    #[tokio::test]
    async fn skips_tracked_bucket_when_embedder_not_in_catalog() {
        let tmp = tempfile::tempdir().unwrap();
        let registry = make_registry_with_buckets(
            tmp.path(),
            &[(
                "wiki_simple",
                &tracked_wikipedia_toml("Simple Wiki", "simple", "manual"),
            )],
        )
        .await;
        // Empty catalog — bucket's `embedder = "test_embedder"` won't
        // resolve.
        let providers = HashMap::new();
        let tokens = spawn_for_registry(&registry, &providers);
        assert!(
            tokens.is_empty(),
            "no workers should spawn when the configured embedder is missing",
        );
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
        let providers = embedders_with_test_entry("test_embedder");
        let tokens = spawn_for_registry_with_observer(&registry, &providers, observer);
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
