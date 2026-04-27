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
//! - **Page-level mutation.** Today's worker advances the cursor as
//!   soon as a delta has been *downloaded*; page-level `tombstone`
//!   and `insert` against the active slot's index land in a follow-up
//!   slice. Until then the field name `last_applied_delta_id` is
//!   slightly aspirational — its real semantics in this slice is
//!   "downloaded; application pending".
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

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use super::config::TrackedCadence;
use super::feed_state::{FeedState, delta_cache_dir};
#[cfg(test)]
use super::source::DeltaId;
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
    /// Polled the driver, listed deltas, and downloaded all that were
    /// new. `listed` is what the driver returned; `fetched` is how
    /// many actually transferred bytes (the driver's idempotent
    /// `fetch_delta` short-circuits when a file is already cached).
    Polled { listed: usize, fetched: usize },
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
    /// (future "Poll now" trigger will land alongside the
    /// trigger-channel work).
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
            match self.interval {
                Some(d) => {
                    tokio::select! {
                        biased;
                        _ = self.cancel.cancelled() => break,
                        _ = tokio::time::sleep(d) => {}
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
            }

            let outcome = self.poll_once().await;

            self.observer.on_tick(tick_index, outcome.clone());
            tick_index += 1;

            debug!(
                bucket_id = %self.bucket_id,
                ?outcome,
                "feed worker tick complete",
            );
        }

        info!(bucket_id = %self.bucket_id, "feed worker stopped");
    }

    /// One round of "list deltas since last cursor + download each new
    /// one." Pulled out of `run` so the future "Poll now" trigger can
    /// drive the same code path on demand.
    ///
    /// Today the cursor advances as soon as a delta is downloaded
    /// (`last_applied_delta_id` doubles as the
    /// most-recently-downloaded id during this slice). The cursor
    /// keeps the driver's directory listing bounded; the actual
    /// page-level tombstone+insert against the active slot lands in
    /// the next slice.
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
            };
        }

        let mut fetched = 0usize;
        for delta_id in &listing {
            if self.cancel.is_cancelled() {
                return TickOutcome::Error("cancelled".to_string());
            }

            let dest_dir = delta_cache_dir(&self.bucket_dir, delta_id);
            let dest = dest_dir.join(self.driver.delta_filename(delta_id));

            // The driver's `fetch_delta` is idempotent — if the file
            // already exists with non-zero size it returns Ok without
            // transferring bytes. Whether or not we fetched, we
            // advance the cursor so the next listing skips this id.
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

            // Persist the cursor after each successful fetch so a
            // crash partway through a long catch-up batch doesn't
            // re-download what we already have.
            state.set_last_applied_delta(delta_id.clone());
            if let Err(e) = state.save_atomic(&self.bucket_dir) {
                return TickOutcome::Error(format!("feed-state save: {e}"));
            }
        }

        TickOutcome::Polled {
            listed: listing.len(),
            fetched,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::sync::Mutex;

    use super::super::source::feed::BoxFuture;
    use super::super::source::{FeedError, SnapshotId};

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
        let worker = FeedWorker::new(
            "test_bucket",
            tmp.path().to_path_buf(),
            Box::new(driver),
            Some(Duration::from_millis(25)),
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
        for (_, outcome) in &ticks {
            assert!(
                matches!(
                    outcome,
                    TickOutcome::Polled {
                        listed: 0,
                        fetched: 0
                    }
                ),
                "every tick should be Polled{{0,0}} on an empty listing; got {outcome:?}"
            );
        }
    }

    #[tokio::test]
    async fn manual_cadence_emits_idle_then_blocks_until_cancel() {
        let tmp = tempfile::tempdir().unwrap();
        let (driver, _log) = ScriptedDriver::new(vec![], vec![]);
        let observer = Arc::new(RecordingObserver::default());
        let cancel = CancellationToken::new();
        let worker = FeedWorker::new(
            "manual_bucket",
            tmp.path().to_path_buf(),
            Box::new(driver),
            None,
            cancel.clone(),
            observer.clone(),
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
        let worker = FeedWorker::new(
            "early_cancel",
            tmp.path().to_path_buf(),
            Box::new(driver),
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

    #[tokio::test]
    async fn poll_downloads_new_deltas_and_advances_cursor() {
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
        let worker = FeedWorker::new(
            "test_bucket",
            tmp.path().to_path_buf(),
            Box::new(driver),
            Some(Duration::from_millis(50)),
            cancel.clone(),
            observer.clone(),
        );
        let handle = tokio::spawn(worker.run());

        tokio::time::sleep(Duration::from_millis(150)).await;
        cancel.cancel();
        handle.await.unwrap();

        let ticks = observer.ticks.lock().unwrap().clone();
        let first = ticks.first().expect("at least one tick");
        assert_eq!(
            first.1,
            TickOutcome::Polled {
                listed: 3,
                fetched: 3
            },
            "first tick should download all 3 deltas, got {first:?}"
        );

        for date in ["20260420", "20260421", "20260422"] {
            let dest = tmp
                .path()
                .join(format!("source-cache/deltas/{date}/delta-{date}.bin"));
            assert!(dest.exists(), "expected delta file at {}", dest.display());
            assert_eq!(std::fs::read(&dest).unwrap(), payload);
        }

        let state = FeedState::load(tmp.path()).unwrap();
        assert_eq!(
            state.last_applied_delta(),
            Some(DeltaId::new("20260422".to_string())),
        );

        let calls = log.lock().unwrap().clone();
        assert_eq!(
            calls
                .iter()
                .filter(|c| c.starts_with("list_deltas_since"))
                .count(),
            ticks.len(),
            "one list_deltas_since per tick",
        );
        assert_eq!(
            calls
                .iter()
                .filter(|c| c.starts_with("fetch_delta"))
                .count(),
            3,
            "exactly 3 fetch_delta calls (cursor advances after first tick): {calls:?}",
        );
    }

    #[tokio::test]
    async fn poll_skips_when_no_base_recorded() {
        let tmp = tempfile::tempdir().unwrap();
        let (driver, log) =
            ScriptedDriver::new(vec![DeltaId::new("20260420")], b"never read".to_vec());
        let observer = Arc::new(RecordingObserver::default());
        let cancel = CancellationToken::new();
        let worker = FeedWorker::new(
            "test_bucket",
            tmp.path().to_path_buf(),
            Box::new(driver),
            Some(Duration::from_millis(25)),
            cancel.clone(),
            observer.clone(),
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
}
