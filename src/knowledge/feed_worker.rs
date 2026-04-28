//! Per-bucket feed worker — the long-running task that polls a
//! [`FeedDriver`](super::source::FeedDriver) on the configured cadence
//! and applies new deltas to a tracked source-kind bucket.
//!
//! One [`FeedWorker`] per tracked bucket. Spawned at registry-load time
//! by the scheduler (see `runtime::scheduler::feed_workers`); held
//! behind its [`CancellationToken`] so a `DeleteBucket` /
//! reconfiguration cleanly tears the worker down.
//!
//! ## What lives here today
//!
//! - The cadence loop with cancellation discipline.
//! - A [`WorkerObserver`] hook that the scheduler can plug into for
//!   test/observability — production wiring uses [`NoopObserver`].
//! - The polling pipeline: each tick reads `feed-state.toml`, calls
//!   the driver's `list_deltas_since`, downloads any new deltas into
//!   `<bucket>/source-cache/deltas/<id>/`, and advances
//!   `last_applied_delta_id`.
//!
//! ## What does NOT live here yet
//!
//! - **Manual triggers.** The "Poll now" / "Resync now" wire path
//!   isn't wired; manual cadence currently emits one Idle tick and
//!   blocks on cancellation.
//! - **Background resync.** Per-month rebuild against a fresh base
//!   lives in another follow-up.
//!
//! ## Why this is its own module
//!
//! - **Lifecycle** for the worker is bucket-scoped: the spawn /
//!   cancellation policy belongs to the scheduler, but the run loop is
//!   pure knowledge-layer business (driver + bucket + feed-state). Same
//!   layering as `disk_bucket::build_slot` versus
//!   `runtime::scheduler::buckets::run_build`.
//! - **Testable in isolation** — exercise the loop with a synthetic
//!   driver and a short real-time interval without spinning up a
//!   scheduler. The constructor takes an `Option<Duration>` rather
//!   than a [`TrackedCadence`] so tests can pass intervals as small
//!   as they need.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::providers::embedding::EmbeddingProvider;

use super::DeltaApplyStats;
use super::config::TrackedCadence;
use super::feed_state::{FeedState, delta_cache_dir};
use super::registry::BucketRegistry;
#[cfg(test)]
use super::source::DeltaId;
use super::source::FeedDriver;
use super::source::MediaWikiXml;

/// Map a configured cadence to a polling-interval duration. `Manual`
/// returns `None` — the worker still runs but only ticks on explicit
/// trigger (a "Poll now" / "Resync now" UI button, when those land).
///
/// Driver-publication cadences are calendar-aligned in practice
/// (Wikipedia incrementals land roughly daily UTC), but for v1 the
/// worker uses a simple "interval since last successful tick" model.
/// A user-perceptible drift of up to one cadence period is acceptable
/// — driver `list_deltas_since` is the source of truth for what's
/// missing, so a wakeup that happens 4 hours late just means the
/// catch-up batch is one delta longer.
pub fn cadence_to_duration(cadence: TrackedCadence) -> Option<Duration> {
    match cadence {
        TrackedCadence::Daily => Some(Duration::from_secs(60 * 60 * 24)),
        TrackedCadence::Weekly => Some(Duration::from_secs(60 * 60 * 24 * 7)),
        TrackedCadence::Monthly => Some(Duration::from_secs(60 * 60 * 24 * 30)),
        TrackedCadence::Quarterly => Some(Duration::from_secs(60 * 60 * 24 * 90)),
        TrackedCadence::Manual => None,
    }
}

/// What a single tick of the feed-worker loop did. Surfaced through
/// [`WorkerObserver`] so tests / scheduler-side telemetry can observe
/// the loop without parsing logs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TickOutcome {
    /// Cadence is `Manual`; the loop is idle until an external trigger
    /// fires.
    Idle,
    /// Polled the driver, listed deltas, downloaded the new ones, and
    /// applied them against the active slot. Counters carry through
    /// from each step:
    /// - `listed` — what the driver returned;
    /// - `fetched` — how many actually transferred bytes (the
    ///   driver's idempotent `fetch_delta` short-circuits when a
    ///   file is already cached);
    /// - `applied` — sum of [`DeltaApplyStats`] across the deltas
    ///   processed in this tick (pages applied / unchanged / chunks
    ///   tombstoned + inserted).
    Polled {
        listed: usize,
        fetched: usize,
        applied: DeltaApplyStats,
    },
    /// The bucket has no `current_base_snapshot_id` recorded yet — the
    /// initial-build path hasn't run. The worker skips polling
    /// because there's no anchor to apply deltas against.
    SkippedNoBase,
    /// The driver / state-file step failed. The error string is
    /// truncated for log/UI surfacing; the worker continues running
    /// and will retry on the next tick.
    Error(String),
}

/// Hook for tests and scheduler-side telemetry. The scheduler's
/// production wiring will plug in something that broadcasts wire
/// events; tests record into a `Vec`.
pub trait WorkerObserver: Send + Sync {
    /// Called after each cadence tick (or skipped tick on `Manual`).
    /// `tick_index` starts at 0 for the first tick observed by this
    /// worker instance and increments monotonically.
    fn on_tick(&self, tick_index: u64, outcome: TickOutcome);
}

/// No-op observer — used by the scheduler when it doesn't need to
/// observe per-tick events (the run loop's `tracing` output is
/// sufficient for routine operation).
pub struct NoopObserver;

impl WorkerObserver for NoopObserver {
    fn on_tick(&self, _tick_index: u64, _outcome: TickOutcome) {}
}

/// Long-running per-bucket task. Owns the driver, the bucket directory
/// path, the polling interval, and a cancellation token; spawned via
/// [`FeedWorker::run`].
///
/// The constructor takes an already-resolved `Option<Duration>` rather
/// than a [`TrackedCadence`] enum — production code calls
/// [`cadence_to_duration`] once at spawn time; tests pass any
/// `Duration` they want without the cadence-enum's day-or-larger
/// values forcing paused-time gymnastics.
pub struct FeedWorker {
    bucket_id: String,
    /// Owning pod for pod-scope tracked buckets; `None` for
    /// server-scope. Drives whether `apply_one_delta` resolves through
    /// `loaded_bucket` (server) or `loaded_bucket_pod` (pod), and
    /// rides along on resync-request payloads so the scheduler can
    /// dispatch back to the correct sub-registry.
    pod_id: Option<String>,
    bucket_dir: PathBuf,
    driver: Box<dyn FeedDriver>,
    interval: Option<Duration>,
    cancel: CancellationToken,
    observer: Arc<dyn WorkerObserver>,
    /// Registry handle for resolving the bucket on first apply. The
    /// worker doesn't pre-load — first-tick HNSW open at boot would
    /// be expensive at enwiki scale — so we lean on the registry's
    /// lazy `loaded_bucket` / `loaded_bucket_pod`. The clone shares
    /// the registry's internal `Arc<Mutex>` cache, so one bucket is
    /// opened once across the worker, scheduler queries, and any
    /// other consumer.
    registry: BucketRegistry,
    /// Embedder snapshot for this bucket, captured at spawn time.
    /// Reconfiguring the bucket's `[embedding_providers.<name>]`
    /// rebuilds the worker; per-tick re-lookup isn't required for v1.
    /// The slot's manifest carries the canonical `model_id`, so we
    /// don't store one here — `apply_one_delta` reads it from the
    /// loaded bucket when calling `set_embedder`.
    embedder: Arc<dyn EmbeddingProvider>,
    /// Manual-trigger receiver. The scheduler holds the matching
    /// sender and `try_send`s a unit on the "Poll now" wire message;
    /// the worker treats it as another wakeup source alongside the
    /// cadence timer. Bounded at capacity 1 so multiple rapid
    /// triggers coalesce — there's no value in queuing them since
    /// the worker rolls all available deltas into one tick anyway.
    trigger: mpsc::Receiver<()>,
    /// Cadence between scheduled resyncs (`None` = `TrackedCadence::Manual`,
    /// no scheduled fire — manual "Resync now" still works through
    /// the wire path). Worker computes the first fire-time from
    /// `feed_state.last_resync_at + resync_interval` so a server
    /// restart doesn't reset the resync clock.
    resync_interval: Option<Duration>,
    /// Outbound channel to the scheduler. Each message is a
    /// `(pod_id, bucket_id)` pair requesting a scheduled resync; the
    /// scheduler drains the channel and dispatches
    /// `handle_resync_bucket` with no requester. `pod_id == None`
    /// means server-scope; `Some(pod)` is pod-scope and routes through
    /// the pod sub-registry. Unbounded: a stuck scheduler can't be
    /// unblocked by dropping resync requests, and the channel has at
    /// most one outstanding message per worker (we don't retry until
    /// next cadence fire).
    resync_request_tx: mpsc::UnboundedSender<(Option<String>, String)>,
}

/// One iteration of the run-loop's `select!` picks among several
/// wake-up sources. The arm that fires drives different post-wakeup
/// work — pulled into an enum so the existing `select!` block stays a
/// clean exhaustive match.
enum WakeReason {
    Cancelled,
    DeltaPoll,
    ResyncDue,
}

/// Spawn-side handle returned alongside a fresh [`FeedWorker`]. The
/// scheduler holds these to drive lifecycle (`cancel.cancel()` on
/// `DeleteBucket`) and manual triggering (`trigger.try_send(())` on
/// `PollFeedNow`).
#[derive(Debug, Clone)]
pub struct FeedWorkerControl {
    pub cancel: CancellationToken,
    pub trigger: mpsc::Sender<()>,
}

impl FeedWorker {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        bucket_id: impl Into<String>,
        pod_id: Option<String>,
        bucket_dir: PathBuf,
        driver: Box<dyn FeedDriver>,
        interval: Option<Duration>,
        cancel: CancellationToken,
        observer: Arc<dyn WorkerObserver>,
        registry: BucketRegistry,
        embedder: Arc<dyn EmbeddingProvider>,
        trigger: mpsc::Receiver<()>,
        resync_interval: Option<Duration>,
        resync_request_tx: mpsc::UnboundedSender<(Option<String>, String)>,
    ) -> Self {
        Self {
            bucket_id: bucket_id.into(),
            pod_id,
            bucket_dir,
            driver,
            interval,
            cancel,
            observer,
            registry,
            embedder,
            trigger,
            resync_interval,
            resync_request_tx,
        }
    }

    /// Compute the duration to wait before the first scheduled resync
    /// fire. Reads `feed_state.last_resync_at` so a server restart
    /// doesn't reset the resync clock — important for monthly
    /// cadences against deploy strategies that recreate the pod
    /// regularly.
    ///
    /// `None` last_resync_at maps to a *full interval* of sleep, not
    /// zero. "Never resynced" is the steady state for a freshly-built
    /// tracked bucket (build path doesn't stamp `last_resync_at` —
    /// only an explicit Resync run does), so treating `None` as
    /// "infinitely overdue" would auto-fire a multi-hour rebuild on
    /// every server boot. Operators who *want* an immediate resync
    /// click the manual button.
    fn initial_resync_sleep(
        state: &FeedState,
        interval: Duration,
        now: chrono::DateTime<chrono::Utc>,
    ) -> Duration {
        match state.last_resync_at {
            None => interval,
            Some(last) => {
                let span = match chrono::Duration::from_std(interval) {
                    Ok(d) => d,
                    // Cadence overflow chrono — not realistic for any
                    // configured value, but be defensive.
                    Err(_) => return interval,
                };
                let due = last + span;
                if due <= now {
                    Duration::ZERO
                } else {
                    (due - now).to_std().unwrap_or(Duration::ZERO)
                }
            }
        }
    }

    /// Run the cadence loop until cancelled. On `interval = None`
    /// (manual cadence) the loop blocks on cancellation indefinitely
    /// (future "Poll now" trigger will land alongside the
    /// trigger-channel work).
    ///
    /// Returns when the cancellation token fires. The caller (the
    /// scheduler) decides whether to spawn this on a runtime task or
    /// hold the future directly.
    pub async fn run(mut self) {
        info!(
            bucket_id = %self.bucket_id,
            interval_secs = self.interval.map(|d| d.as_secs()).unwrap_or(0),
            resync_interval_secs = self.resync_interval.map(|d| d.as_secs()).unwrap_or(0),
            "feed worker started",
        );

        let mut tick_index: u64 = 0;

        // Manual cadence still emits one Idle tick at startup so
        // observers see the worker is alive even before any trigger
        // fires. Cadence-driven workers don't need this — their first
        // real tick lands within `interval`.
        if self.interval.is_none() {
            self.observer.on_tick(tick_index, TickOutcome::Idle);
            tick_index += 1;
        }

        // Scheduled resync sleep is one-shot, re-armed after each
        // fire. Initial duration is computed from feed-state so a
        // server restart resumes the schedule rather than resetting.
        // Held in a Box<Pin> so the select! arm can borrow it
        // mutably across iterations.
        let initial_resync = match self.resync_interval {
            Some(d) => match FeedState::load(&self.bucket_dir) {
                Ok(state) => Self::initial_resync_sleep(&state, d, chrono::Utc::now()),
                Err(e) => {
                    // Fall back to "fire after one full interval" if
                    // we can't read the state — better than firing
                    // immediately on every restart.
                    warn!(
                        bucket_id = %self.bucket_id,
                        error = %e,
                        "feed-state load for resync schedule failed; using full interval as fallback",
                    );
                    d
                }
            },
            None => Duration::ZERO, // unused — arm is gated by resync_interval.is_some()
        };
        let mut next_resync = Box::pin(tokio::time::sleep(initial_resync));

        loop {
            // Wakeup sources, in priority order:
            //  1. cancellation (always wins)
            //  2. cadence timer (delta poll, when `interval` is Some)
            //  3. manual "Poll now" trigger (always live)
            //  4. resync cadence timer (when `resync_interval` is Some)
            let interval = self.interval;
            let has_resync = self.resync_interval.is_some();
            let reason = tokio::select! {
                biased;
                _ = self.cancel.cancelled() => WakeReason::Cancelled,
                _ = async move {
                    match interval {
                        Some(d) => tokio::time::sleep(d).await,
                        None => std::future::pending::<()>().await,
                    }
                } => WakeReason::DeltaPoll,
                _ = self.trigger.recv() => WakeReason::DeltaPoll,
                _ = &mut next_resync, if has_resync => WakeReason::ResyncDue,
            };

            match reason {
                WakeReason::Cancelled => break,
                WakeReason::DeltaPoll => {
                    let outcome = self.poll_once().await;
                    self.observer.on_tick(tick_index, outcome.clone());
                    tick_index += 1;
                    debug!(
                        bucket_id = %self.bucket_id,
                        ?outcome,
                        "feed worker tick complete",
                    );
                }
                WakeReason::ResyncDue => {
                    // Fire-and-forget — scheduler owns concurrency
                    // (rejects if a build is already in flight) and
                    // logs failures. We re-arm for another full
                    // interval regardless: if the resync runs and
                    // succeeds, `last_resync_at` advances; if it
                    // can't run (e.g. embedder unavailable), the
                    // next tick tries again.
                    let payload = (self.pod_id.clone(), self.bucket_id.clone());
                    if let Err(e) = self.resync_request_tx.send(payload) {
                        warn!(
                            bucket_id = %self.bucket_id,
                            pod_id = ?self.pod_id,
                            error = %e,
                            "resync_request channel send failed (scheduler stopped?)",
                        );
                    } else {
                        info!(
                            bucket_id = %self.bucket_id,
                            pod_id = ?self.pod_id,
                            "scheduled resync requested",
                        );
                    }
                    let interval = self
                        .resync_interval
                        .expect("ResyncDue fired with resync_interval=None");
                    next_resync = Box::pin(tokio::time::sleep(interval));
                }
            }
        }

        info!(bucket_id = %self.bucket_id, "feed worker stopped");
    }

    /// One round of "list deltas since last cursor; for each new
    /// delta, download then apply against the active slot, then
    /// advance the cursor." Pulled out of `run` so the future "Poll
    /// now" trigger can drive the same code path on demand.
    ///
    /// Cursor semantics: `last_applied_delta_id` advances after the
    /// delta has been *applied* to the bucket index — not just
    /// downloaded. Since `apply_delta` is idempotent (re-running
    /// against the same delta is a clean no-op) and `fetch_delta` is
    /// idempotent (skips already-cached files), retry on next tick
    /// is safe at any failure point.
    async fn poll_once(&self) -> TickOutcome {
        // Read the on-disk cursor. On any IO/parse error, surface as a
        // tick error so the worker can retry on its next cadence.
        let mut state = match FeedState::load(&self.bucket_dir) {
            Ok(s) => s,
            Err(e) => return TickOutcome::Error(format!("feed-state load: {e}")),
        };

        if state.current_base().is_none() {
            return TickOutcome::SkippedNoBase;
        }

        let since = state.last_applied_delta();
        let listing = match self
            .driver
            .list_deltas_since(since.as_ref(), &self.cancel)
            .await
        {
            Ok(v) => v,
            Err(e) => return TickOutcome::Error(format!("list_deltas_since: {e}")),
        };

        if listing.is_empty() {
            return TickOutcome::Polled {
                listed: 0,
                fetched: 0,
                applied: DeltaApplyStats::default(),
            };
        }

        let mut fetched = 0usize;
        let mut applied_total = DeltaApplyStats::default();
        for delta_id in &listing {
            if self.cancel.is_cancelled() {
                return TickOutcome::Error("cancelled".to_string());
            }

            let dest_dir = delta_cache_dir(&self.bucket_dir, delta_id);
            let dest = dest_dir.join(self.driver.delta_filename(delta_id));

            // Step 1: download (idempotent — skips already-cached
            // files at the driver's discretion). On failure leave the
            // cursor where it is so the next tick re-tries.
            let pre_existed = std::fs::metadata(&dest)
                .map(|m| m.is_file() && m.len() > 0)
                .unwrap_or(false);
            match self.driver.fetch_delta(delta_id, &dest, &self.cancel).await {
                Ok(()) => {
                    if !pre_existed {
                        fetched += 1;
                    }
                }
                Err(e) => {
                    warn!(
                        bucket_id = %self.bucket_id,
                        delta = %delta_id,
                        error = %e,
                        "fetch_delta failed; will retry on next tick",
                    );
                    return TickOutcome::Error(format!("fetch_delta {delta_id}: {e}"));
                }
            }

            // Step 2: apply against the active slot. Bucket lookup
            // goes through the registry's lazy `loaded_bucket` so the
            // HNSW reload cost lands on first-apply, not on server
            // boot. After first call the registry caches the Arc.
            let apply_result = self.apply_one_delta(&dest).await;
            match apply_result {
                Ok(stats) => {
                    debug!(
                        bucket_id = %self.bucket_id,
                        delta = %delta_id,
                        ?stats,
                        "delta applied",
                    );
                    applied_total = sum_stats(applied_total, stats);
                }
                Err(e) => {
                    warn!(
                        bucket_id = %self.bucket_id,
                        delta = %delta_id,
                        error = %e,
                        "apply_delta failed; will retry on next tick",
                    );
                    return TickOutcome::Error(format!("apply_delta {delta_id}: {e}"));
                }
            }

            // Step 3: advance cursor only after both download and
            // apply succeeded. Persist immediately so a crash here
            // doesn't reapply the same delta on the next tick (idempotent
            // anyway, but cheaper to skip the re-application).
            state.set_last_applied_delta(delta_id.clone());
            if let Err(e) = state.save_atomic(&self.bucket_dir) {
                return TickOutcome::Error(format!("feed-state save: {e}"));
            }
        }

        TickOutcome::Polled {
            listed: listing.len(),
            fetched,
            applied: applied_total,
        }
    }

    /// Resolve the bucket through the registry, ensure its embedder
    /// is wired (idempotent — `set_embedder` short-circuits when the
    /// model_id matches), construct the right
    /// [`SourceAdapter`](super::source::SourceAdapter) for the
    /// driver's archive format, and call
    /// [`DiskBucket::apply_delta`](super::disk_bucket::DiskBucket::apply_delta).
    ///
    /// Adapter dispatch is by the driver's `parse_adapter()` id —
    /// `"mediawiki_xml"` is the only valid value today. Future
    /// drivers (Wikidata, arxiv-feed) plug in here as new arms.
    async fn apply_one_delta(
        &self,
        delta_path: &Path,
    ) -> Result<DeltaApplyStats, super::types::BucketError> {
        let bucket = match self.pod_id.as_deref() {
            Some(pid) => {
                self.registry
                    .loaded_bucket_pod(pid, &self.bucket_id)
                    .await?
            }
            None => self.registry.loaded_bucket(&self.bucket_id).await?,
        };
        let model_id = bucket.active_embedder_model_id().ok_or_else(|| {
            super::types::BucketError::Other(
                "active slot has no recorded embedder; bucket needs an initial build before \
                 deltas can be applied"
                    .to_string(),
            )
        })?;
        bucket.set_embedder(self.embedder.clone(), &model_id)?;

        match self.driver.parse_adapter() {
            "mediawiki_xml" => {
                let adapter = MediaWikiXml::new(delta_path);
                bucket.apply_delta(&adapter, &self.cancel).await
            }
            other => Err(super::types::BucketError::Other(format!(
                "feed driver reported unknown parse_adapter `{other}`",
            ))),
        }
    }
}

/// Element-wise sum of two [`DeltaApplyStats`]. Used to roll up
/// per-delta stats into the tick-level total surfaced on `Polled`.
fn sum_stats(a: DeltaApplyStats, b: DeltaApplyStats) -> DeltaApplyStats {
    DeltaApplyStats {
        pages_applied: a.pages_applied + b.pages_applied,
        pages_unchanged: a.pages_unchanged + b.pages_unchanged,
        pages_skipped_empty: a.pages_skipped_empty + b.pages_skipped_empty,
        chunks_tombstoned: a.chunks_tombstoned + b.chunks_tombstoned,
        chunks_inserted: a.chunks_inserted + b.chunks_inserted,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::sync::Mutex;

    use crate::providers::embedding::{
        BoxFuture as ProviderBoxFuture, EmbedRequest, EmbeddingError, EmbeddingModelInfo,
        EmbeddingResponse,
    };

    use super::super::source::feed::BoxFuture;
    use super::super::source::{FeedError, SnapshotId};

    /// Minimal `EmbeddingProvider` used by the tests in this module —
    /// the constructor needs *something*, but apply isn't actually
    /// exercised here (the registry is empty so `loaded_bucket` fails
    /// before any embed call). For end-to-end download+apply
    /// coverage, see the live-simplewiki run; bucket-level apply
    /// behavior is exercised in `disk_bucket::tests::apply_delta_*`.
    struct StubEmbedder;
    impl EmbeddingProvider for StubEmbedder {
        fn embed<'a>(
            &'a self,
            _req: &'a EmbedRequest<'a>,
            _cancel: &'a CancellationToken,
        ) -> ProviderBoxFuture<'a, Result<EmbeddingResponse, EmbeddingError>> {
            Box::pin(async {
                Err(EmbeddingError::Transport(
                    "StubEmbedder not callable in feed_worker unit tests".into(),
                ))
            })
        }
        fn list_models<'a>(
            &'a self,
        ) -> ProviderBoxFuture<'a, Result<Vec<EmbeddingModelInfo>, EmbeddingError>> {
            Box::pin(async {
                Ok(vec![EmbeddingModelInfo {
                    id: "stub".to_string(),
                    dimension: 8,
                    max_input_tokens: None,
                }])
            })
        }
    }

    fn stub_embedder() -> Arc<dyn EmbeddingProvider> {
        Arc::new(StubEmbedder)
    }

    /// Make a fresh trigger channel; the test holds onto the sender
    /// for the duration of the worker's run so the channel stays
    /// open. A sender that's been dropped causes `recv()` to return
    /// `None` immediately — the select! arm would then fire on every
    /// loop iteration, defeating the whole purpose of waiting on the
    /// timer or cancel arm.
    fn fresh_trigger() -> (mpsc::Sender<()>, mpsc::Receiver<()>) {
        mpsc::channel::<()>(1)
    }

    /// Resync-request sender for tests that don't exercise the
    /// scheduled-resync arm. Receiver is dropped immediately — any
    /// `send()` on it returns `Err`, which the worker logs but doesn't
    /// fail on. Tests that DO exercise the resync arm should hold
    /// onto the receiver.
    fn fresh_resync_request_tx() -> mpsc::UnboundedSender<(Option<String>, String)> {
        let (tx, _rx) = mpsc::unbounded_channel::<(Option<String>, String)>();
        tx
    }

    #[derive(Default)]
    struct RecordingObserver {
        ticks: Mutex<Vec<(u64, TickOutcome)>>,
    }

    impl WorkerObserver for RecordingObserver {
        fn on_tick(&self, tick_index: u64, outcome: TickOutcome) {
            self.ticks.lock().unwrap().push((tick_index, outcome));
        }
    }

    /// Synthetic driver returning caller-supplied delta listings and
    /// caller-supplied payload bytes for fetch_delta. Records each
    /// call so tests can assert on call ordering / arguments.
    struct ScriptedDriver {
        deltas: Vec<DeltaId>,
        payload: Vec<u8>,
        log: std::sync::Arc<Mutex<Vec<String>>>,
    }

    impl ScriptedDriver {
        fn new(
            deltas: Vec<DeltaId>,
            payload: Vec<u8>,
        ) -> (Self, std::sync::Arc<Mutex<Vec<String>>>) {
            let log = std::sync::Arc::new(Mutex::new(Vec::new()));
            (
                Self {
                    deltas,
                    payload,
                    log: log.clone(),
                },
                log,
            )
        }
    }

    impl FeedDriver for ScriptedDriver {
        fn latest_base<'a>(
            &'a self,
            _cancel: &'a CancellationToken,
        ) -> BoxFuture<'a, Result<SnapshotId, FeedError>> {
            Box::pin(async { unreachable!("worker does not call latest_base") })
        }
        fn fetch_base<'a>(
            &'a self,
            _id: &'a SnapshotId,
            _dest: &'a Path,
            _cancel: &'a CancellationToken,
        ) -> BoxFuture<'a, Result<(), FeedError>> {
            Box::pin(async { unreachable!("worker does not call fetch_base") })
        }
        fn list_deltas_since<'a>(
            &'a self,
            since: Option<&'a DeltaId>,
            _cancel: &'a CancellationToken,
        ) -> BoxFuture<'a, Result<Vec<DeltaId>, FeedError>> {
            self.log.lock().unwrap().push(format!(
                "list_deltas_since({:?})",
                since.map(|d| d.as_str())
            ));
            let cutoff = since.map(|d| d.as_str().to_string());
            let out: Vec<DeltaId> = self
                .deltas
                .iter()
                .filter(|d| match &cutoff {
                    Some(c) => d.as_str() > c.as_str(),
                    None => true,
                })
                .cloned()
                .collect();
            Box::pin(async move { Ok(out) })
        }
        fn fetch_delta<'a>(
            &'a self,
            id: &'a DeltaId,
            dest: &'a Path,
            _cancel: &'a CancellationToken,
        ) -> BoxFuture<'a, Result<(), FeedError>> {
            self.log.lock().unwrap().push(format!(
                "fetch_delta({},{})",
                id.as_str(),
                dest.display()
            ));
            let payload = self.payload.clone();
            Box::pin(async move {
                if let Some(parent) = dest.parent() {
                    tokio::fs::create_dir_all(parent).await.unwrap();
                }
                tokio::fs::write(dest, &payload).await.unwrap();
                Ok(())
            })
        }
        fn parse_adapter(&self) -> &'static str {
            "mediawiki_xml"
        }
        fn base_filename(&self, id: &SnapshotId) -> String {
            format!("base-{}.bin", id.as_str())
        }
        fn delta_filename(&self, id: &DeltaId) -> String {
            format!("delta-{}.bin", id.as_str())
        }
    }

    fn write_feed_state_with_base(bucket_dir: &Path, snapshot: &str) {
        let mut state = FeedState::default();
        state.set_current_base(SnapshotId::new(snapshot.to_string()));
        state.save_atomic(bucket_dir).unwrap();
    }

    #[test]
    fn initial_resync_sleep_defers_a_full_interval_when_never_resynced() {
        // Regression: `None` last_resync_at used to map to ZERO and
        // auto-fire a multi-hour rebuild on every server boot for any
        // bucket built before the resync feature shipped. Now it
        // defers a full interval — operators click the manual Resync
        // button when they want an immediate one.
        let state = FeedState::default();
        let interval = Duration::from_secs(60 * 60 * 24);
        let sleep = FeedWorker::initial_resync_sleep(&state, interval, chrono::Utc::now());
        assert_eq!(
            sleep, interval,
            "no last_resync_at should defer first auto-resync by one full interval"
        );
    }

    #[test]
    fn initial_resync_sleep_is_zero_when_overdue() {
        // Last resync was 10 days ago, cadence is daily — overdue.
        let state = FeedState {
            last_resync_at: Some(chrono::Utc::now() - chrono::Duration::days(10)),
            ..FeedState::default()
        };
        let sleep = FeedWorker::initial_resync_sleep(
            &state,
            Duration::from_secs(60 * 60 * 24),
            chrono::Utc::now(),
        );
        assert_eq!(sleep, Duration::ZERO, "overdue should fire immediately");
    }

    #[test]
    fn initial_resync_sleep_is_remaining_window_when_recent() {
        let now = chrono::Utc::now();
        // Last resync was 2 hours ago, cadence is daily — sleep ~22h.
        let state = FeedState {
            last_resync_at: Some(now - chrono::Duration::hours(2)),
            ..FeedState::default()
        };
        let sleep =
            FeedWorker::initial_resync_sleep(&state, Duration::from_secs(60 * 60 * 24), now);
        // 22h ± a small slop for chrono rounding.
        let expected = Duration::from_secs(22 * 60 * 60);
        let diff = sleep.abs_diff(expected);
        assert!(
            diff < Duration::from_secs(5),
            "expected ~22h, got {sleep:?} (diff {diff:?})"
        );
    }

    #[test]
    fn cadence_to_duration_maps_each_variant() {
        assert_eq!(
            cadence_to_duration(TrackedCadence::Daily),
            Some(Duration::from_secs(86_400))
        );
        assert_eq!(
            cadence_to_duration(TrackedCadence::Weekly),
            Some(Duration::from_secs(604_800))
        );
        assert_eq!(
            cadence_to_duration(TrackedCadence::Monthly),
            Some(Duration::from_secs(2_592_000))
        );
        assert_eq!(
            cadence_to_duration(TrackedCadence::Quarterly),
            Some(Duration::from_secs(7_776_000))
        );
        assert_eq!(cadence_to_duration(TrackedCadence::Manual), None);
    }

    #[tokio::test]
    async fn empty_listing_yields_polled_zero_zero() {
        let tmp = tempfile::tempdir().unwrap();
        write_feed_state_with_base(tmp.path(), "20260401");

        let (driver, _log) = ScriptedDriver::new(vec![], vec![]);
        let observer = Arc::new(RecordingObserver::default());
        let cancel = CancellationToken::new();
        let (_trigger_tx, trigger_rx) = fresh_trigger();
        let worker = FeedWorker::new(
            "test_bucket",
            None,
            tmp.path().to_path_buf(),
            Box::new(driver),
            Some(Duration::from_millis(25)),
            cancel.clone(),
            observer.clone(),
            BucketRegistry::default(),
            stub_embedder(),
            trigger_rx,
            None,
            fresh_resync_request_tx(),
        );
        let handle = tokio::spawn(worker.run());

        tokio::time::sleep(Duration::from_millis(200)).await;
        cancel.cancel();
        handle.await.unwrap();

        let ticks = observer.ticks.lock().unwrap().clone();
        assert!(
            ticks.len() >= 3,
            "expected at least 3 ticks at 25 ms cadence in 200 ms; got {ticks:?}",
        );
        for (_, outcome) in &ticks {
            assert!(
                matches!(
                    outcome,
                    TickOutcome::Polled {
                        listed: 0,
                        fetched: 0,
                        ..
                    }
                ),
                "every tick should be Polled{{0,0,..}} on an empty listing; got {outcome:?}"
            );
        }
    }

    #[tokio::test]
    async fn manual_cadence_emits_idle_then_blocks_until_cancel() {
        let tmp = tempfile::tempdir().unwrap();
        let (driver, _log) = ScriptedDriver::new(vec![], vec![]);
        let observer = Arc::new(RecordingObserver::default());
        let cancel = CancellationToken::new();
        let (_trigger_tx, trigger_rx) = fresh_trigger();
        let worker = FeedWorker::new(
            "manual_bucket",
            None,
            tmp.path().to_path_buf(),
            Box::new(driver),
            None,
            cancel.clone(),
            observer.clone(),
            BucketRegistry::default(),
            stub_embedder(),
            trigger_rx,
            None,
            fresh_resync_request_tx(),
        );
        let handle = tokio::spawn(worker.run());

        tokio::time::sleep(Duration::from_millis(50)).await;
        cancel.cancel();
        handle.await.unwrap();

        let ticks = observer.ticks.lock().unwrap().clone();
        assert_eq!(ticks.len(), 1, "manual cadence should emit one Idle tick");
        assert_eq!(ticks[0], (0, TickOutcome::Idle));
    }

    #[tokio::test]
    async fn cancel_before_first_tick_returns_immediately() {
        let tmp = tempfile::tempdir().unwrap();
        let (driver, _log) = ScriptedDriver::new(vec![], vec![]);
        let observer = Arc::new(RecordingObserver::default());
        let cancel = CancellationToken::new();
        cancel.cancel();
        let (_trigger_tx, trigger_rx) = fresh_trigger();
        let worker = FeedWorker::new(
            "early_cancel",
            None,
            tmp.path().to_path_buf(),
            Box::new(driver),
            Some(Duration::from_secs(86_400)),
            cancel,
            observer.clone(),
            BucketRegistry::default(),
            stub_embedder(),
            trigger_rx,
            None,
            fresh_resync_request_tx(),
        );
        worker.run().await;
        assert!(
            observer.ticks.lock().unwrap().is_empty(),
            "no ticks should fire before cancellation"
        );
    }

    /// Verify that the worker's download stage drives the driver
    /// correctly — the listing is consulted, deltas land in the
    /// per-bucket cache, and the cursor *only* advances if apply
    /// also succeeds. With an empty registry (no bucket entry),
    /// `apply_one_delta` errors before any fetch happens — see
    /// `loaded_bucket` returning `BucketError::Other("unknown bucket id")`
    /// — so the first download lands but apply blows up, the tick is
    /// `TickOutcome::Error`, and the cursor stays at None. The next
    /// tick re-lists, re-downloads (idempotent — files already on
    /// disk), errors again. This shape matches what we want for the
    /// "feed-state has a base but the bucket initial-build hasn't
    /// finished yet" edge case in production.
    ///
    /// End-to-end download+apply with a real bucket is verified by
    /// the live simplewiki run; bucket-level apply behavior is
    /// covered by `disk_bucket::tests::apply_delta_*`.
    #[tokio::test]
    async fn poll_downloads_then_errors_when_bucket_not_in_registry() {
        let tmp = tempfile::tempdir().unwrap();
        write_feed_state_with_base(tmp.path(), "20260401");

        let payload = b"\x42\x5a\x68\x39 fake delta payload".to_vec();
        let (driver, log) = ScriptedDriver::new(
            vec![
                DeltaId::new("20260420"),
                DeltaId::new("20260421"),
                DeltaId::new("20260422"),
            ],
            payload.clone(),
        );
        let observer = Arc::new(RecordingObserver::default());
        let cancel = CancellationToken::new();
        let (_trigger_tx, trigger_rx) = fresh_trigger();
        let worker = FeedWorker::new(
            "test_bucket",
            None,
            tmp.path().to_path_buf(),
            Box::new(driver),
            Some(Duration::from_millis(50)),
            cancel.clone(),
            observer.clone(),
            BucketRegistry::default(),
            stub_embedder(),
            trigger_rx,
            None,
            fresh_resync_request_tx(),
        );
        let handle = tokio::spawn(worker.run());

        tokio::time::sleep(Duration::from_millis(150)).await;
        cancel.cancel();
        handle.await.unwrap();

        let ticks = observer.ticks.lock().unwrap().clone();
        // Every tick errors out at apply time; the first delta
        // gets downloaded before apply fails, but the cursor stays
        // at None so subsequent ticks see the same listing.
        assert!(
            ticks
                .iter()
                .all(|(_, o)| matches!(o, TickOutcome::Error(_))),
            "every tick should be Error (apply fails on missing bucket): {ticks:?}",
        );

        // First delta landed on disk before apply errored. The
        // worker bails out of the per-tick loop after the apply
        // failure, so deltas 2 and 3 don't get downloaded *this
        // tick* — they will on subsequent ticks (still failing at
        // apply, but the file exists after fetch). Verify at least
        // one of them landed.
        let landed_first = tmp
            .path()
            .join("source-cache/deltas/20260420/delta-20260420.bin");
        assert!(
            landed_first.exists(),
            "expected first delta at {}",
            landed_first.display()
        );
        assert_eq!(std::fs::read(&landed_first).unwrap(), payload);

        // Cursor must NOT have advanced — apply never succeeded.
        let state = FeedState::load(tmp.path()).unwrap();
        assert_eq!(state.last_applied_delta(), None);

        let calls = log.lock().unwrap().clone();
        assert_eq!(
            calls
                .iter()
                .filter(|c| c.starts_with("list_deltas_since"))
                .count(),
            ticks.len(),
            "one list_deltas_since per tick",
        );
        assert!(
            calls.iter().any(|c| c.starts_with("fetch_delta(20260420")),
            "first delta should have been fetched at least once: {calls:?}",
        );
    }

    #[tokio::test]
    async fn poll_skips_when_no_base_recorded() {
        let tmp = tempfile::tempdir().unwrap();
        let (driver, log) =
            ScriptedDriver::new(vec![DeltaId::new("20260420")], b"never read".to_vec());
        let observer = Arc::new(RecordingObserver::default());
        let cancel = CancellationToken::new();
        let (_trigger_tx, trigger_rx) = fresh_trigger();
        let worker = FeedWorker::new(
            "test_bucket",
            None,
            tmp.path().to_path_buf(),
            Box::new(driver),
            Some(Duration::from_millis(25)),
            cancel.clone(),
            observer.clone(),
            BucketRegistry::default(),
            stub_embedder(),
            trigger_rx,
            None,
            fresh_resync_request_tx(),
        );
        let handle = tokio::spawn(worker.run());

        tokio::time::sleep(Duration::from_millis(75)).await;
        cancel.cancel();
        handle.await.unwrap();

        let ticks = observer.ticks.lock().unwrap().clone();
        assert!(
            ticks.iter().all(|(_, o)| *o == TickOutcome::SkippedNoBase),
            "expected all ticks to be SkippedNoBase before initial build runs; got {ticks:?}",
        );
        assert!(
            log.lock().unwrap().is_empty(),
            "driver should not be called when no base is recorded",
        );
    }

    /// Manual "Poll now" trigger short-circuits the cadence wait. Set
    /// the cadence to a value that won't fire within the test window
    /// (1 hour) and verify a `try_send` on the trigger surfaces a tick
    /// promptly. Empty driver listing → `Polled{0,0,..}` per tick;
    /// each trigger produces one such tick.
    #[tokio::test]
    async fn manual_trigger_wakes_worker_before_cadence_elapses() {
        let tmp = tempfile::tempdir().unwrap();
        write_feed_state_with_base(tmp.path(), "20260401");

        let (driver, _log) = ScriptedDriver::new(vec![], vec![]);
        let observer = Arc::new(RecordingObserver::default());
        let cancel = CancellationToken::new();
        let (trigger_tx, trigger_rx) = fresh_trigger();
        let worker = FeedWorker::new(
            "test_bucket",
            None,
            tmp.path().to_path_buf(),
            Box::new(driver),
            Some(Duration::from_secs(3600)), // long enough that cadence won't fire
            cancel.clone(),
            observer.clone(),
            BucketRegistry::default(),
            stub_embedder(),
            trigger_rx,
            None,
            fresh_resync_request_tx(),
        );
        let handle = tokio::spawn(worker.run());

        // Give the loop one yield to enter the select! arm before
        // sending the trigger.
        tokio::task::yield_now().await;
        trigger_tx.try_send(()).expect("trigger send");
        // Yield a few times for the worker to drain the trigger,
        // run poll_once, and call observer.on_tick.
        for _ in 0..50 {
            tokio::task::yield_now().await;
            if !observer.ticks.lock().unwrap().is_empty() {
                break;
            }
        }
        cancel.cancel();
        handle.await.unwrap();

        let ticks = observer.ticks.lock().unwrap().clone();
        assert!(
            !ticks.is_empty(),
            "trigger should produce at least one tick before cadence",
        );
        assert!(
            matches!(
                ticks[0].1,
                TickOutcome::Polled {
                    listed: 0,
                    fetched: 0,
                    ..
                }
            ),
            "trigger-driven tick should be Polled{{0,0,..}} on empty driver: {ticks:?}",
        );
    }

    /// Buffer-full triggers coalesce — `try_send` returns `Err`
    /// without blocking. The semantics for "Poll now" is "either a
    /// poll is in flight or one is queued"; multiple rapid clicks
    /// shouldn't queue indefinitely.
    #[tokio::test]
    async fn manual_trigger_coalesces_rapid_sends() {
        let (tx, _rx) = fresh_trigger();
        // First send fills the bounded-1 buffer.
        tx.try_send(()).expect("first send");
        // Subsequent sends fail-fast with TrySendError::Full.
        let result = tx.try_send(());
        assert!(
            matches!(result, Err(mpsc::error::TrySendError::Full(()))),
            "second send on a full buffer should be Err(Full), got {result:?}",
        );
    }
}
