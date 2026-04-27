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
//! Scaffolding only — the cadence loop, cancellation, and a
//! [`WorkerObserver`] hook that the scheduler can plug into for
//! test/observability. The actual delta-polling logic
//! (`list_deltas_since` + `fetch_delta`) lands in a follow-up slice
//! along with the page-level `tombstone` + `insert` application that
//! propagates a delta into the active slot.
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

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

use super::config::TrackedCadence;
use super::source::FeedDriver;

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
    /// Tick fired but the polling logic is still scaffolding-only —
    /// no driver work performed yet. Goes away when the polling slice
    /// lands.
    NoOp,
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
///
/// `driver` and `bucket_dir` are deliberately unused in this slice —
/// they become load-bearing in the polling slice that lands next, and
/// keeping them on the type here (rather than threading them through
/// `run`'s parameter list later) avoids re-shaping the public
/// constructor when the polling work lands.
#[allow(dead_code)]
pub struct FeedWorker {
    bucket_id: String,
    bucket_dir: PathBuf,
    driver: Box<dyn FeedDriver>,
    interval: Option<Duration>,
    cancel: CancellationToken,
    observer: Arc<dyn WorkerObserver>,
}

impl FeedWorker {
    pub fn new(
        bucket_id: impl Into<String>,
        bucket_dir: PathBuf,
        driver: Box<dyn FeedDriver>,
        interval: Option<Duration>,
        cancel: CancellationToken,
        observer: Arc<dyn WorkerObserver>,
    ) -> Self {
        Self {
            bucket_id: bucket_id.into(),
            bucket_dir,
            driver,
            interval,
            cancel,
            observer,
        }
    }

    /// Run the cadence loop until cancelled. On `interval = None`
    /// (manual cadence) the loop blocks on cancellation indefinitely
    /// (future "Poll now" trigger will land alongside the actual
    /// polling logic).
    ///
    /// Returns when the cancellation token fires. The caller (the
    /// scheduler) decides whether to spawn this on a runtime task or
    /// hold the future directly.
    pub async fn run(self) {
        info!(
            bucket_id = %self.bucket_id,
            interval_secs = self.interval.map(|d| d.as_secs()).unwrap_or(0),
            "feed worker started",
        );

        let mut tick_index: u64 = 0;
        loop {
            let outcome = match self.interval {
                Some(d) => {
                    tokio::select! {
                        biased;
                        _ = self.cancel.cancelled() => break,
                        _ = tokio::time::sleep(d) => TickOutcome::NoOp,
                    }
                }
                None => {
                    // Manual cadence: emit one Idle tick and then block
                    // on cancellation. Future slice will replace the
                    // pending future with a wake-on-trigger channel —
                    // until then `tick_index` doesn't advance because
                    // the loop terminates here.
                    self.observer.on_tick(tick_index, TickOutcome::Idle);
                    self.cancel.cancelled().await;
                    break;
                }
            };

            self.observer.on_tick(tick_index, outcome.clone());
            tick_index += 1;

            // Stub for the actual polling pipeline — landed in a
            // follow-up slice. Today the loop just demonstrates that
            // cancellation works under load and that the observer
            // hook fires correctly.
            debug!(
                bucket_id = %self.bucket_id,
                ?outcome,
                "feed worker tick (polling stub)",
            );
        }

        info!(bucket_id = %self.bucket_id, "feed worker stopped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::sync::Mutex;

    use super::super::source::feed::BoxFuture;
    use super::super::source::{DeltaId, FeedError, SnapshotId};

    #[derive(Default)]
    struct RecordingObserver {
        ticks: Mutex<Vec<(u64, TickOutcome)>>,
    }

    impl WorkerObserver for RecordingObserver {
        fn on_tick(&self, tick_index: u64, outcome: TickOutcome) {
            self.ticks.lock().unwrap().push((tick_index, outcome));
        }
    }

    /// Synthetic driver — none of its methods are invoked by the
    /// scaffolding-only worker. Returning `unreachable!()` makes
    /// accidental usage in this slice a loud failure rather than a
    /// silent hang.
    struct StubDriver;

    impl FeedDriver for StubDriver {
        fn latest_base<'a>(
            &'a self,
            _cancel: &'a CancellationToken,
        ) -> BoxFuture<'a, Result<SnapshotId, FeedError>> {
            Box::pin(async { unreachable!("scaffolding does not call latest_base") })
        }
        fn fetch_base<'a>(
            &'a self,
            _id: &'a SnapshotId,
            _dest: &'a Path,
            _cancel: &'a CancellationToken,
        ) -> BoxFuture<'a, Result<(), FeedError>> {
            Box::pin(async { unreachable!("scaffolding does not call fetch_base") })
        }
        fn list_deltas_since<'a>(
            &'a self,
            _since: Option<&'a DeltaId>,
            _cancel: &'a CancellationToken,
        ) -> BoxFuture<'a, Result<Vec<DeltaId>, FeedError>> {
            Box::pin(async { unreachable!() })
        }
        fn fetch_delta<'a>(
            &'a self,
            _id: &'a DeltaId,
            _dest: &'a Path,
            _cancel: &'a CancellationToken,
        ) -> BoxFuture<'a, Result<(), FeedError>> {
            Box::pin(async { unreachable!() })
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
    async fn run_loop_ticks_on_cadence_and_stops_on_cancel() {
        // Short real-time interval — checks the loop semantics without
        // paused-time fiddling. Three ticks at 25 ms = ~75 ms minimum
        // wall-clock; we sleep 200 ms to give comfortable headroom on
        // slow CI.
        let interval = Duration::from_millis(25);
        let observer = Arc::new(RecordingObserver::default());
        let cancel = CancellationToken::new();
        let worker = FeedWorker::new(
            "test_bucket",
            std::path::PathBuf::from("/tmp/whisper-agent-test-bucket"),
            Box::new(StubDriver),
            Some(interval),
            cancel.clone(),
            observer.clone(),
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
        assert!(
            ticks
                .iter()
                .all(|(_, outcome)| *outcome == TickOutcome::NoOp),
            "expected all ticks to be NoOp, got {ticks:?}"
        );
        // Tick indices monotonic from 0.
        for (i, (tick_index, _)) in ticks.iter().enumerate() {
            assert_eq!(*tick_index, i as u64);
        }
    }

    #[tokio::test]
    async fn manual_cadence_emits_idle_then_blocks_until_cancel() {
        let observer = Arc::new(RecordingObserver::default());
        let cancel = CancellationToken::new();
        let worker = FeedWorker::new(
            "manual_bucket",
            std::path::PathBuf::from("/tmp/whisper-agent-test-bucket"),
            Box::new(StubDriver),
            None, // manual cadence
            cancel.clone(),
            observer.clone(),
        );
        let handle = tokio::spawn(worker.run());

        // Wait briefly for the worker to enter its idle wait. Real time
        // because the test isn't paused — using a small actual sleep to
        // give the spawned task a chance to run and emit the Idle tick.
        tokio::time::sleep(Duration::from_millis(50)).await;
        cancel.cancel();
        handle.await.unwrap();

        let ticks = observer.ticks.lock().unwrap().clone();
        assert_eq!(ticks.len(), 1, "manual cadence should emit one Idle tick");
        assert_eq!(ticks[0], (0, TickOutcome::Idle));
    }

    #[tokio::test]
    async fn cancel_before_first_tick_returns_immediately() {
        let observer = Arc::new(RecordingObserver::default());
        let cancel = CancellationToken::new();
        cancel.cancel();
        let worker = FeedWorker::new(
            "early_cancel",
            std::path::PathBuf::from("/tmp/whisper-agent-test-bucket"),
            Box::new(StubDriver),
            Some(Duration::from_secs(86_400)),
            cancel,
            observer.clone(),
        );
        worker.run().await;
        assert!(
            observer.ticks.lock().unwrap().is_empty(),
            "no ticks should fire before cancellation"
        );
    }
}
