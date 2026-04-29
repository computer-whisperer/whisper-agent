//! Knowledge-bucket lifecycle handlers — `CreateBucket`, `DeleteBucket`,
//! `StartBucketBuild`, `CancelBucketBuild`. Each is a `pub(super) fn`
//! on `Scheduler` so the giant match in `client_messages` dispatches by
//! name.
//!
//! Builds run as detached tokio tasks. The whole `DiskBucket::build_slot`
//! call sits inside `tokio::task::spawn_blocking`, with
//! `Handle::current().block_on` driving its async body — so neither the
//! scheduler thread nor the runtime workers are blocked by the (multi-
//! minute) HNSW build phase. The `DenseIndex::build` step itself is
//! also `spawn_blocking`'d inside `build_slot` for symmetry.
//!
//! Cross-task communication: detached build tasks can't borrow `&mut
//! Scheduler`, so they push `BucketTaskUpdate`s through a channel that
//! the scheduler's run loop drains in its `select!`. The loop then
//! mutates registry state (refresh after build) and broadcasts events
//! to all clients. This keeps build-time mutations linearizable with
//! every other scheduler operation.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::time::Duration;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::warn;
use whisper_agent_protocol::{
    BucketBuildOutcome, BucketBuildPhase, BucketCreateInput, BucketSourceInput, QuantizationInput,
    ServerToClient, TrackedCadenceInput, TrackedDriverInput,
};

use super::{ConnId, Scheduler};
use crate::knowledge::{BucketError, BucketId};
use crate::knowledge::{
    BuildObserver, BuildPhase, Chunker, DeltaId, DiskBucket, FeedDriver, FeedState, MarkdownDir,
    MediaWikiXml, SnapshotId, SourceAdapter, base_cache_dir, registry::entry_to_summary,
    resolve_chunker, source::feed::driver_for,
};
use crate::providers::embedding::EmbeddingProvider;
use crate::server::thread_router::ThreadEventRouter;

/// Throttle interval for `BucketBuildProgress` events. The build's
/// `on_progress` callback fires per chunk-batch (every 128 chunks);
/// throttling client-side to 1Hz keeps the websocket from drowning
/// the UI on dense source dumps without losing the "feels live"
/// rhythm.
const PROGRESS_THROTTLE: Duration = Duration::from_millis(1000);

/// Scope-aware key for the in-flight-build maps and per-build progress
/// state. Server-scope buckets share a global namespace
/// (`pod_id = None`); pod-scope buckets are keyed by their owning pod
/// so two pods naming a bucket `memory` don't collide.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct BucketKey {
    pub pod_id: Option<String>,
    pub name: String,
}

impl BucketKey {
    pub fn from_optional(pod_id: Option<String>, name: impl Into<String>) -> Self {
        Self {
            pod_id,
            name: name.into(),
        }
    }

    #[cfg(test)]
    pub fn server(name: impl Into<String>) -> Self {
        Self {
            pod_id: None,
            name: name.into(),
        }
    }
}

/// Updates a detached build task pushes back into the scheduler loop.
/// The loop drains these from `bucket_task_rx` and applies them under
/// `&mut Scheduler` so registry mutations stay linearizable with the
/// rest of the scheduler.
///
/// `Progress` ticks are channeled (rather than fanned out via a frozen
/// client snapshot inside the build task) so the scheduler can broadcast
/// each tick to the *currently-connected* clients via the live router.
/// That's what lets a client that joins mid-build see ongoing progress
/// instead of silence until the terminal `BuildEnded`.
pub enum BucketTaskUpdate {
    /// Per-tick progress sample. Throttled by the build task's emitter
    /// loop; the scheduler turns each into a `BucketBuildProgress`
    /// broadcast via `router.broadcast_task_list` so it reaches every
    /// client connected at the moment of the tick.
    Progress {
        bucket_id: String,
        /// Owning pod for pod-scope builds; `None` for server-scope.
        /// Carried through so the broadcast event can label the tick
        /// correctly and the scheduler can key into
        /// `active_bucket_progress` for follow-up lookups.
        pod_id: Option<String>,
        snapshot: ProgressSnapshot,
    },
    /// Build completed (success / error / cancel). The loop refreshes
    /// the registry entry on success, broadcasts `BucketBuildEnded`
    /// with the fresh summary, echoes the terminal event to the
    /// requester with their `correlation_id`, and unregisters the
    /// build from `active_bucket_builds` and `active_bucket_progress`.
    BuildEnded {
        bucket_id: String,
        pod_id: Option<String>,
        slot_id: String,
        outcome: BucketBuildOutcome,
        requester_conn: Option<ConnId>,
        correlation_id: Option<String>,
    },
}

impl Scheduler {
    pub(super) fn handle_create_bucket(
        &mut self,
        conn_id: ConnId,
        correlation_id: Option<String>,
        id: String,
        pod_id: Option<String>,
        config: BucketCreateInput,
    ) {
        if let Err(e) = validate_bucket_id(&id) {
            self.send_bucket_error(conn_id, correlation_id, "create_bucket", e);
            return;
        }
        if config.name.trim().is_empty() {
            self.send_bucket_error(
                conn_id,
                correlation_id,
                "create_bucket",
                "name must not be empty".into(),
            );
            return;
        }
        if !self.embedding_providers.contains_key(&config.embedder) {
            self.send_bucket_error(
                conn_id,
                correlation_id,
                "create_bucket",
                format!(
                    "embedder `{}` is not configured under [embedding_providers.*]",
                    config.embedder,
                ),
            );
            return;
        }
        if let Err(e) = validate_source(&config.source) {
            self.send_bucket_error(conn_id, correlation_id, "create_bucket", e);
            return;
        }
        if config.chunk_tokens == 0 {
            self.send_bucket_error(
                conn_id,
                correlation_id,
                "create_bucket",
                "chunk_tokens must be > 0".into(),
            );
            return;
        }

        // Resolve the bucket's parent dir from scope. Server-scope
        // wants `<buckets_root>/<id>/`; pod-scope wants
        // `<pods_root>/<pod_id>/buckets/<id>/`. Either source of
        // truth must already be initialized for creation to succeed.
        let (scope, bucket_dir) = match pod_id.as_deref() {
            None => {
                let Some(root) = self.bucket_registry.root.clone() else {
                    self.send_bucket_error(
                        conn_id,
                        correlation_id,
                        "create_bucket",
                        "server was started without --buckets-root; cannot create server-scope buckets"
                            .into(),
                    );
                    return;
                };
                (crate::knowledge::BucketScope::Server, root.join(&id))
            }
            Some(pid) => {
                let Some(pod) = self.pods.get(pid) else {
                    self.send_bucket_error(
                        conn_id,
                        correlation_id,
                        "create_bucket",
                        format!("unknown pod id `{pid}`"),
                    );
                    return;
                };
                (
                    crate::knowledge::BucketScope::Pod,
                    pod.dir.join("buckets").join(&id),
                )
            }
        };
        if bucket_dir.exists() {
            self.send_bucket_error(
                conn_id,
                correlation_id,
                "create_bucket",
                format!("directory already exists at {}", bucket_dir.display()),
            );
            return;
        }

        let toml_text = synthesize_bucket_toml(&config);
        if let Err(e) = std::fs::create_dir_all(&bucket_dir) {
            self.send_bucket_error(
                conn_id,
                correlation_id,
                "create_bucket",
                format!("mkdir failed: {e}"),
            );
            return;
        }
        if let Err(e) = std::fs::write(bucket_dir.join("bucket.toml"), &toml_text) {
            // Roll back the (now-empty) directory so a retry isn't blocked.
            let _ = std::fs::remove_dir_all(&bucket_dir);
            self.send_bucket_error(
                conn_id,
                correlation_id,
                "create_bucket",
                format!("write bucket.toml failed: {e}"),
            );
            return;
        }

        // Parse what we just wrote, the same way registry::load reads
        // existing buckets at startup. If this fails we have a bug in
        // synthesize_bucket_toml — surface it loudly and roll back.
        let parsed_config = match crate::knowledge::BucketConfig::from_toml_str(&toml_text) {
            Ok(c) => c,
            Err(e) => {
                let _ = std::fs::remove_dir_all(&bucket_dir);
                self.send_bucket_error(
                    conn_id,
                    correlation_id,
                    "create_bucket",
                    format!("synthesized bucket.toml failed to parse (bug): {e}"),
                );
                return;
            }
        };
        let entry = crate::knowledge::BucketEntry {
            id: id.clone(),
            scope,
            pod_id: pod_id.clone(),
            dir: bucket_dir.clone(),
            config: parsed_config,
            raw_toml: toml_text,
            active_slot: None,
            active_slot_disk_size_bytes: None,
        };
        let insert_result = match scope {
            crate::knowledge::BucketScope::Server => {
                self.bucket_registry.insert_entry(entry.clone())
            }
            crate::knowledge::BucketScope::Pod => {
                self.bucket_registry.insert_pod_entry(entry.clone())
            }
        };
        if let Err(e) = insert_result {
            // Roll back the on-disk dir so the operator isn't stuck
            // with a half-created bucket the registry doesn't know
            // about.
            let _ = std::fs::remove_dir_all(&bucket_dir);
            self.send_bucket_error(
                conn_id,
                correlation_id,
                "create_bucket",
                format!("registry insert failed: {e}"),
            );
            return;
        }

        let summary = entry_to_summary(&entry);
        let ev = ServerToClient::BucketCreated {
            correlation_id: None,
            summary: summary.clone(),
        };
        self.router.broadcast_task_list_except(ev, conn_id);
        self.router.send_to_client(
            conn_id,
            ServerToClient::BucketCreated {
                correlation_id,
                summary,
            },
        );
    }

    pub(super) fn handle_delete_bucket(
        &mut self,
        conn_id: ConnId,
        correlation_id: Option<String>,
        id: String,
        pod_id: Option<String>,
    ) {
        if let Err(e) = validate_bucket_id(&id) {
            self.send_bucket_error(conn_id, correlation_id, "delete_bucket", e);
            return;
        }
        let dir = match pod_id.as_deref() {
            None => match self.bucket_registry.buckets.get(&id) {
                Some(e) => e.dir.clone(),
                None => {
                    self.send_bucket_error(
                        conn_id,
                        correlation_id,
                        "delete_bucket",
                        format!("unknown bucket id `{id}`"),
                    );
                    return;
                }
            },
            Some(pid) => match self
                .bucket_registry
                .pod_buckets
                .get(pid)
                .and_then(|m| m.get(&id))
            {
                Some(e) => e.dir.clone(),
                None => {
                    self.send_bucket_error(
                        conn_id,
                        correlation_id,
                        "delete_bucket",
                        format!("unknown pod-scope bucket `{pid}`/`{id}`"),
                    );
                    return;
                }
            },
        };

        // Cancel any in-flight build first. The build task observes
        // the token and emits its own `BucketBuildEnded { Cancelled }`.
        let key = BucketKey::from_optional(pod_id.clone(), id.clone());
        if let Some(token) = self.active_bucket_builds.remove(&key) {
            token.cancel();
        }

        // Stop the per-tracked-bucket feed worker, if any. Looked up
        // by scope-aware `BucketKey` so a server-scope worker named
        // the same as a pod's bucket isn't mistakenly stopped.
        let worker_key = BucketKey::from_optional(pod_id.clone(), id.clone());
        if let Some(control) = self.active_feed_workers.remove(&worker_key) {
            control.cancel.cancel();
        }

        // Drop the in-memory entry + cached DiskBucket. Sync — both
        // operations are O(1) under a short-held mutex.
        match pod_id.as_deref() {
            None => {
                self.bucket_registry.remove_entry(&id);
            }
            Some(pid) => {
                self.bucket_registry.remove_pod_entry(pid, &id);
            }
        }

        // rmdir runs detached so the response isn't gated on it.
        let dir_for_rm = dir.clone();
        let id_for_log = id.clone();
        let pod_id_for_log = pod_id.clone();
        tokio::spawn(async move {
            if let Err(e) = tokio::fs::remove_dir_all(&dir_for_rm).await {
                warn!(
                    pod = ?pod_id_for_log,
                    bucket = %id_for_log,
                    path = %dir_for_rm.display(),
                    error = %e,
                    "delete_bucket: rmdir failed; orphan directory left on disk",
                );
            }
        });

        let ev = ServerToClient::BucketDeleted {
            correlation_id: None,
            id: id.clone(),
            pod_id: pod_id.clone(),
        };
        self.router.broadcast_task_list_except(ev, conn_id);
        self.router.send_to_client(
            conn_id,
            ServerToClient::BucketDeleted {
                correlation_id,
                id,
                pod_id,
            },
        );
    }

    pub(super) fn handle_start_bucket_build(
        &mut self,
        conn_id: ConnId,
        correlation_id: Option<String>,
        id: String,
        pod_id: Option<String>,
    ) {
        if let Err(e) = validate_bucket_id(&id) {
            self.send_bucket_error(conn_id, correlation_id, "start_bucket_build", e);
            return;
        }
        let key = BucketKey::from_optional(pod_id.clone(), id.clone());
        if self.active_bucket_builds.contains_key(&key) {
            self.send_bucket_error(
                conn_id,
                correlation_id,
                "start_bucket_build",
                format!("build already in flight for `{id}`"),
            );
            return;
        }
        let entry = match pod_id.as_deref() {
            None => match self.bucket_registry.buckets.get(&id) {
                Some(e) => e.clone(),
                None => {
                    self.send_bucket_error(
                        conn_id,
                        correlation_id,
                        "start_bucket_build",
                        format!("unknown bucket id `{id}`"),
                    );
                    return;
                }
            },
            Some(pid) => match self
                .bucket_registry
                .pod_buckets
                .get(pid)
                .and_then(|m| m.get(&id))
            {
                Some(e) => e.clone(),
                None => {
                    self.send_bucket_error(
                        conn_id,
                        correlation_id,
                        "start_bucket_build",
                        format!("unknown pod-scope bucket `{pid}`/`{id}`"),
                    );
                    return;
                }
            },
        };
        let embedder = match self
            .embedding_providers
            .get(&entry.config.defaults.embedder)
        {
            Some(p) => p.provider.clone(),
            None => {
                self.send_bucket_error(
                    conn_id,
                    correlation_id,
                    "start_bucket_build",
                    format!(
                        "embedder `{}` not configured",
                        entry.config.defaults.embedder
                    ),
                );
                return;
            }
        };
        let source = match build_source(&entry.config.source) {
            Ok(s) => s,
            Err(e) => {
                self.send_bucket_error(conn_id, correlation_id, "start_bucket_build", e);
                return;
            }
        };
        // `DiskBucket::open` (and the resumable-slot scan that follows
        // it) used to run here on the scheduler task. On a multi-GiB
        // bucket the HNSW load that happens inside `open` blocked the
        // scheduler for ~12s after a click on Build. Both calls now
        // live inside `run_build`'s `spawn_blocking` block — the
        // scheduler tick is freed immediately and the click → ack
        // round-trip is sub-ms.

        let cancel = CancellationToken::new();
        self.active_bucket_builds
            .insert(key.clone(), cancel.clone());

        // Shared progress state. The build task ticks the atomics; the
        // scheduler reads them to (a) replay current state to a client
        // that joins mid-build, and (b) build each `BucketBuildProgress`
        // broadcast from the `Progress` channel update. Keyed by the
        // scoped `BucketKey` so a fresh `RegisterClient` can walk
        // in-flight builds without going through the build task.
        let progress = Arc::new(ProgressShared::default());
        let started_at = progress.started_at().to_rfc3339();
        self.active_bucket_progress
            .insert(key.clone(), progress.clone());

        // Synchronous BuildStarted ack. slot_id is unknown until the
        // build's first phase callback; carry an empty slot_id until
        // the BuildEnded event lands the real one. UI ignores this
        // field on Started — only bucket_id matters for the row state.
        let started = ServerToClient::BucketBuildStarted {
            correlation_id: None,
            bucket_id: id.clone(),
            pod_id: pod_id.clone(),
            slot_id: String::new(),
            started_at: Some(started_at.clone()),
        };
        self.router.broadcast_task_list_except(started, conn_id);
        self.router.send_to_client(
            conn_id,
            ServerToClient::BucketBuildStarted {
                correlation_id: correlation_id.clone(),
                bucket_id: id.clone(),
                pod_id: pod_id.clone(),
                slot_id: String::new(),
                started_at: Some(started_at),
            },
        );

        let task_tx = self.bucket_task_sender();
        let chunker_config = entry.config.chunker.clone();
        let bucket_dir = entry.dir.clone();
        tokio::spawn(run_build(
            embedder,
            source,
            bucket_dir,
            chunker_config,
            cancel,
            id,
            pod_id,
            progress,
            task_tx,
            Some(conn_id),
            correlation_id,
            BuildIntent::Normal,
        ));
    }

    pub(super) fn handle_cancel_bucket_build(
        &mut self,
        conn_id: ConnId,
        correlation_id: Option<String>,
        id: String,
        pod_id: Option<String>,
    ) {
        let key = BucketKey::from_optional(pod_id.clone(), id.clone());
        match self.active_bucket_builds.get(&key) {
            Some(token) => {
                // Cancellation is observed at the next chunk-batch
                // boundary or HNSW build entry; the build task emits
                // its own `BucketBuildEnded { Cancelled }` on exit.
                // No separate ack — the UI watches BuildEnded.
                token.cancel();
            }
            None => {
                let label = match pod_id.as_deref() {
                    None => format!("`{id}`"),
                    Some(pid) => format!("`{pid}`/`{id}`"),
                };
                self.send_bucket_error(
                    conn_id,
                    correlation_id,
                    "cancel_bucket_build",
                    format!("no in-flight build for {label}"),
                );
            }
        }
    }

    /// Manually trigger a resync — same plumbing as
    /// `handle_start_bucket_build` but with [`BuildIntent::Resync`],
    /// gated to tracked buckets only. Rejection paths (unknown id,
    /// non-tracked, in-flight build, missing embedder) all surface as
    /// `Error` synchronously; acceptance broadcasts `BucketBuildStarted`
    /// the same way an initial build does, so the UI's existing
    /// progress wiring lights up unchanged.
    ///
    /// `requester` distinguishes a wire-driven dispatch (from
    /// `ResyncBucket`) from a scheduler-driven dispatch (from a
    /// FeedWorker resync-cadence tick). For scheduled triggers,
    /// rejections log instead of surfacing as `Error` events, and the
    /// `BucketBuildStarted` ack-to-requester path is skipped — the
    /// broadcast still goes to all clients.
    pub(super) fn handle_resync_bucket(
        &mut self,
        requester: Option<ConnId>,
        correlation_id: Option<String>,
        id: String,
        pod_id: Option<String>,
    ) {
        let pod_id_for_log = pod_id.clone();
        let refuse = |scheduler: &mut Scheduler, msg: String| match requester {
            Some(conn) => {
                scheduler.send_bucket_error(conn, correlation_id.clone(), "resync_bucket", msg);
            }
            None => {
                warn!(
                    bucket_id = %id,
                    pod_id = ?pod_id_for_log,
                    error = %msg,
                    "scheduled resync rejected",
                );
            }
        };
        if let Err(e) = validate_bucket_id(&id) {
            refuse(self, e);
            return;
        }
        let scope = match pod_id {
            Some(_) => crate::knowledge::BucketScope::Pod,
            None => crate::knowledge::BucketScope::Server,
        };
        let key = BucketKey::from_optional(pod_id.clone(), id.clone());
        if self.active_bucket_builds.contains_key(&key) {
            refuse(
                self,
                match &pod_id {
                    Some(pid) => format!("build already in flight for `{id}` (pod `{pid}`)"),
                    None => format!("build already in flight for `{id}`"),
                },
            );
            return;
        }
        let entry = match self
            .bucket_registry
            .find_entry(scope, pod_id.as_deref(), &id)
        {
            Some(e) => e.clone(),
            None => {
                refuse(
                    self,
                    match &pod_id {
                        Some(pid) => format!("unknown bucket id `{id}` (pod `{pid}`)"),
                        None => format!("unknown bucket id `{id}`"),
                    },
                );
                return;
            }
        };
        // Resync only makes sense for tracked sources — they're the
        // only kind with a driver capable of advancing to a fresh
        // base. Reject non-tracked at the wire layer rather than
        // letting `run_build` produce a confusing error mid-task.
        if !matches!(
            entry.config.source,
            crate::knowledge::SourceConfig::Tracked { .. }
        ) {
            refuse(
                self,
                format!(
                    "bucket `{id}` is not tracked (source kind: {}) — resync requires a tracked driver",
                    entry.config.source.kind(),
                ),
            );
            return;
        }
        let embedder = match self
            .embedding_providers
            .get(&entry.config.defaults.embedder)
        {
            Some(p) => p.provider.clone(),
            None => {
                refuse(
                    self,
                    format!(
                        "embedder `{}` not configured",
                        entry.config.defaults.embedder
                    ),
                );
                return;
            }
        };
        let source = match build_source(&entry.config.source) {
            Ok(s) => s,
            Err(e) => {
                refuse(self, e);
                return;
            }
        };
        // `DiskBucket::open` is deferred into `run_build`'s
        // `spawn_blocking` — the HNSW load can be ~12s on a multi-GiB
        // bucket and we don't want it on the scheduler task. Resync
        // intentionally does *not* reuse a resumable slot (a fresh
        // base demands a fresh pipeline; any leftover Planning/Building
        // slot was built off the old base), so the resume-slot scan is
        // skipped inside `run_build` for `BuildIntent::Resync`.

        let cancel = CancellationToken::new();
        self.active_bucket_builds
            .insert(key.clone(), cancel.clone());

        let progress = Arc::new(ProgressShared::default());
        let started_at = progress.started_at().to_rfc3339();
        self.active_bucket_progress
            .insert(key.clone(), progress.clone());

        // Broadcast `BucketBuildStarted` to all subscribers so the UI
        // lights up regardless of who triggered. For wire-driven
        // dispatch, also send a correlation-id-bearing ack back to the
        // requesting client; scheduled dispatch skips that step (no
        // requester to correlate with).
        match requester {
            Some(conn) => {
                let started = ServerToClient::BucketBuildStarted {
                    correlation_id: None,
                    bucket_id: id.clone(),
                    pod_id: pod_id.clone(),
                    slot_id: String::new(),
                    started_at: Some(started_at.clone()),
                };
                self.router.broadcast_task_list_except(started, conn);
                self.router.send_to_client(
                    conn,
                    ServerToClient::BucketBuildStarted {
                        correlation_id: correlation_id.clone(),
                        bucket_id: id.clone(),
                        pod_id: pod_id.clone(),
                        slot_id: String::new(),
                        started_at: Some(started_at),
                    },
                );
            }
            None => {
                let started = ServerToClient::BucketBuildStarted {
                    correlation_id: None,
                    bucket_id: id.clone(),
                    pod_id: pod_id.clone(),
                    slot_id: String::new(),
                    started_at: Some(started_at),
                };
                broadcast_all(&self.router, started);
                tracing::info!(
                    bucket_id = %id,
                    pod_id = ?pod_id,
                    "scheduled resync started",
                );
            }
        }

        let task_tx = self.bucket_task_sender();
        let chunker_config = entry.config.chunker.clone();
        let bucket_dir = entry.dir.clone();
        tokio::spawn(run_build(
            embedder,
            source,
            bucket_dir,
            chunker_config,
            cancel,
            id,
            pod_id,
            progress,
            task_tx,
            requester,
            correlation_id,
            BuildIntent::Resync,
        ));
    }

    pub(super) fn handle_poll_feed_now(
        &mut self,
        conn_id: ConnId,
        correlation_id: Option<String>,
        id: String,
        pod_id: Option<String>,
    ) {
        let key = BucketKey::from_optional(pod_id.clone(), id.clone());
        let Some(control) = self.active_feed_workers.get(&key) else {
            // No feed worker — either an unknown bucket id or the
            // bucket isn't tracked. Same error surface in either
            // case; the UI button is only shown for tracked buckets,
            // so this hitting in production means a stale UI.
            self.send_bucket_error(
                conn_id,
                correlation_id,
                "poll_feed_now",
                match &pod_id {
                    Some(pid) => {
                        format!("no feed worker for `{id}` (pod `{pid}`, not a tracked bucket?)")
                    }
                    None => format!("no feed worker for `{id}` (not a tracked bucket?)"),
                },
            );
            return;
        };
        // `try_send` semantics: if the trigger buffer (capacity 1) is
        // already full, a poll is already pending or in flight — the
        // user's click coalesces silently with the prior trigger.
        // We still ack `Accepted` because the user's intent ("a poll
        // happens soon") will be served by that pending trigger.
        match control.trigger.try_send(()) {
            Ok(()) => {}
            Err(tokio::sync::mpsc::error::TrySendError::Full(())) => {
                // Already pending; treat as accepted.
            }
            Err(tokio::sync::mpsc::error::TrySendError::Closed(())) => {
                // The worker task has exited (panic or shutdown).
                // This shouldn't normally happen — DeleteBucket
                // removes the entry before the channel closes.
                self.send_bucket_error(
                    conn_id,
                    correlation_id.clone(),
                    "poll_feed_now",
                    format!("feed worker for `{id}` has stopped"),
                );
                return;
            }
        }
        self.router.send_to_client(
            conn_id,
            ServerToClient::FeedPollAccepted {
                correlation_id,
                bucket_id: id,
                pod_id,
            },
        );
    }

    /// Replay current state for every in-flight build to one client
    /// (the one that just connected). Without this, a fresh client
    /// sees `active_slot.state = Building` from `ListBuckets` but no
    /// phase / counters until the *next* terminal `BucketBuildEnded`
    /// — `Started` was missed because the build began before the
    /// client connected, and `Progress` ticks already-broadcast went
    /// to other clients.
    pub(crate) fn replay_active_builds_to_client(&self, conn_id: ConnId) {
        for (key, progress) in &self.active_bucket_progress {
            let started_at = progress.started_at().to_rfc3339();
            self.router.send_to_client(
                conn_id,
                ServerToClient::BucketBuildStarted {
                    correlation_id: None,
                    bucket_id: key.name.clone(),
                    pod_id: key.pod_id.clone(),
                    slot_id: String::new(),
                    started_at: Some(started_at.clone()),
                },
            );
            let snap = progress.snapshot();
            self.router.send_to_client(
                conn_id,
                ServerToClient::BucketBuildProgress {
                    bucket_id: key.name.clone(),
                    pod_id: key.pod_id.clone(),
                    slot_id: String::new(),
                    phase: snap.phase,
                    source_records: snap.source_records,
                    chunks: snap.chunks,
                    started_at: Some(started_at),
                },
            );
        }
    }

    /// Drain a `BucketTaskUpdate` from the channel. Called from the
    /// scheduler loop's `select!` arm.
    pub(crate) async fn apply_bucket_task_update(&mut self, update: BucketTaskUpdate) {
        match update {
            BucketTaskUpdate::Progress {
                bucket_id,
                pod_id,
                snapshot,
            } => {
                let event = ServerToClient::BucketBuildProgress {
                    bucket_id,
                    pod_id,
                    slot_id: String::new(), // slot_id surfaces on Ended; stays empty here
                    phase: snapshot.phase,
                    source_records: snapshot.source_records,
                    chunks: snapshot.chunks,
                    started_at: Some(snapshot.started_at.to_rfc3339()),
                };
                self.router.broadcast_task_list(event);
            }
            BucketTaskUpdate::BuildEnded {
                bucket_id,
                pod_id,
                slot_id,
                outcome,
                requester_conn,
                correlation_id,
            } => {
                let key = BucketKey::from_optional(pod_id.clone(), bucket_id.clone());
                self.active_bucket_builds.remove(&key);
                self.active_bucket_progress.remove(&key);
                let summary = if matches!(outcome, BucketBuildOutcome::Success) {
                    match pod_id.as_deref() {
                        None => match self.bucket_registry.refresh_entry(&bucket_id).await {
                            Ok(()) => self
                                .bucket_registry
                                .buckets
                                .get(&bucket_id)
                                .map(entry_to_summary),
                            Err(e) => {
                                warn!(
                                    bucket = %bucket_id,
                                    error = %e,
                                    "post-build refresh failed; ListBuckets will be stale until next refresh",
                                );
                                None
                            }
                        },
                        Some(pid) => match self
                            .bucket_registry
                            .refresh_pod_entry(pid, &bucket_id)
                            .await
                        {
                            Ok(()) => self
                                .bucket_registry
                                .pod_buckets
                                .get(pid)
                                .and_then(|m| m.get(&bucket_id))
                                .map(entry_to_summary),
                            Err(e) => {
                                warn!(
                                    pod = %pid,
                                    bucket = %bucket_id,
                                    error = %e,
                                    "post-build refresh failed; ListBuckets will be stale until next refresh",
                                );
                                None
                            }
                        },
                    }
                } else {
                    None
                };

                let ended = ServerToClient::BucketBuildEnded {
                    correlation_id: None,
                    bucket_id: bucket_id.clone(),
                    pod_id: pod_id.clone(),
                    slot_id: slot_id.clone(),
                    outcome: outcome.clone(),
                    summary: summary.clone(),
                };
                broadcast_all(&self.router, ended);

                if let Some(conn) = requester_conn {
                    self.router.send_to_client(
                        conn,
                        ServerToClient::BucketBuildEnded {
                            correlation_id,
                            bucket_id,
                            pod_id,
                            slot_id,
                            outcome,
                            summary,
                        },
                    );
                }
            }
        }
    }

    fn send_bucket_error(
        &self,
        conn_id: ConnId,
        correlation_id: Option<String>,
        op: &str,
        msg: String,
    ) {
        self.router.send_to_client(
            conn_id,
            ServerToClient::Error {
                correlation_id,
                thread_id: None,
                message: format!("{op}: {msg}"),
            },
        );
    }
}

/// Best-effort fan-out to every currently-connected client. Failures
/// (disconnected since snapshot) are silently ignored.
fn broadcast_all(router: &ThreadEventRouter, event: ServerToClient) {
    for tx in router.clients_snapshot() {
        let _ = tx.send(event.clone());
    }
}

/// Distinguishes a normal (initial / rebuild / resume) build from a
/// resync. Resync uses [`resolve_resync_source`] (always calls
/// `latest_base()`) instead of [`resolve_source`] (reuses the recorded
/// snapshot id), short-circuits to `Success` when the recorded base is
/// already at latest, and on success calls [`reset_delta_cursor_to`]
/// so the FeedWorker's next delta tick lists deltas strictly after
/// the new base.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildIntent {
    /// Initial build, manual rebuild, or resume — uses the recorded
    /// `current_base_snapshot_id` for tracked sources.
    Normal,
    /// Manual or scheduled resync — advance to the driver's
    /// `latest_base()`.
    Resync,
}

#[allow(clippy::too_many_arguments)]
async fn run_build(
    embedder: Arc<dyn EmbeddingProvider>,
    source: BuildSource,
    bucket_dir: std::path::PathBuf,
    chunker_config: crate::knowledge::ChunkerConfig,
    cancel: CancellationToken,
    bucket_id: String,
    pod_id: Option<String>,
    progress: Arc<ProgressShared>,
    task_tx: mpsc::UnboundedSender<BucketTaskUpdate>,
    requester_conn: Option<ConnId>,
    correlation_id: Option<String>,
    intent: BuildIntent,
) {
    let observer_arc: Arc<dyn BuildObserver> = Arc::new(ProgressObserver {
        shared: progress.clone(),
    });

    // Throttle / emit task: reads `progress` every PROGRESS_THROTTLE
    // and pushes a `Progress` update through `task_tx`. The scheduler
    // loop turns each into a `BucketBuildProgress` broadcast against
    // the *currently-connected* clients — so a client that joins
    // mid-build picks up subsequent ticks without any special wiring
    // here. Stops when the build flips `done`.
    let bucket_id_for_emit = bucket_id.clone();
    let pod_id_for_emit = pod_id.clone();
    let progress_for_emit = progress.clone();
    let task_tx_for_emit = task_tx.clone();
    let emitter = tokio::spawn(async move {
        let mut interval = tokio::time::interval(PROGRESS_THROTTLE);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            if progress_for_emit.done.load(Ordering::Acquire) {
                break;
            }
            let snap = progress_for_emit.snapshot();
            let _ = task_tx_for_emit.send(BucketTaskUpdate::Progress {
                bucket_id: bucket_id_for_emit.clone(),
                pod_id: pod_id_for_emit.clone(),
                snapshot: snap,
            });
        }
    });

    // Run the build inside spawn_blocking + block_on so the runtime
    // workers stay free during the multi-minute HNSW build phase. The
    // observer is dyn-dispatched; the cost is negligible per call.
    //
    // Chunker resolution lives inside the blocking section: hf-hub uses
    // sync I/O for tokenizer.json fetch, and we want it on a worker
    // thread (not the scheduler tick). list_models is async though, so
    // we await it first to get the embedder's reported model id; the
    // result feeds `resolve_chunker` for the `tokenizer = "auto"` case.
    let embedder_model_id = match embedder.list_models().await {
        Ok(models) => models.first().map(|m| m.id.clone()),
        Err(e) => {
            // Best-effort — falling back to heuristic when the embedder
            // can't be queried is fine; build still proceeds.
            tracing::warn!(
                bucket_id = %bucket_id,
                error = %e,
                "list_models failed at build start; chunker will fall back to heuristic if `tokenizer = \"auto\"`",
            );
            None
        }
    };

    // Source resolution. For stored / linked, this is a no-op handoff.
    // For tracked Normal builds, the driver fetches the base snapshot
    // into the bucket's source-cache and persists the snapshot id to
    // feed-state.toml — Downloading phase fires here, before any
    // chunking or embedding starts.
    //
    // For Resync intent on a tracked bucket, `resolve_resync_source`
    // *always* consults `latest_base()` and may short-circuit to a
    // no-op success when the recorded snapshot is already current.
    // Stored / linked sources can't be resync'd (they have no driver);
    // we surface that as an error rather than silently treating it as
    // a Normal rebuild — the wire layer should have caught it before
    // dispatching here.
    let (adapter, resync_new_snapshot_id) = match (intent, source) {
        (BuildIntent::Resync, BuildSource::Tracked { driver }) => {
            match resolve_resync_source(driver, &bucket_dir, observer_arc.as_ref(), &cancel).await {
                Ok(ResyncResolution::AlreadyAtLatest { snapshot_id }) => {
                    // No work to do. Stop the emitter and emit a
                    // synthetic Success with empty slot_id so the UI's
                    // BuildEnded handler closes the in-flight row.
                    progress.done.store(true, Ordering::Release);
                    let _ = emitter.await;
                    tracing::info!(
                        bucket_id = %bucket_id,
                        snapshot_id = %snapshot_id,
                        "resync no-op: bucket already at latest base",
                    );
                    let _ = task_tx.send(BucketTaskUpdate::BuildEnded {
                        bucket_id,
                        pod_id,
                        slot_id: String::new(),
                        outcome: BucketBuildOutcome::Success,
                        requester_conn,
                        correlation_id,
                    });
                    return;
                }
                Ok(ResyncResolution::Build {
                    adapter,
                    new_snapshot_id,
                }) => (adapter, Some(new_snapshot_id)),
                Err(e) => {
                    progress.done.store(true, Ordering::Release);
                    let _ = emitter.await;
                    let outcome = match e {
                        BucketError::Cancelled => BucketBuildOutcome::Cancelled,
                        other => BucketBuildOutcome::Error {
                            message: other.to_string(),
                        },
                    };
                    let _ = task_tx.send(BucketTaskUpdate::BuildEnded {
                        bucket_id,
                        pod_id,
                        slot_id: String::new(),
                        outcome,
                        requester_conn,
                        correlation_id,
                    });
                    return;
                }
            }
        }
        (BuildIntent::Resync, source) => {
            // Caller dispatched a resync against a non-tracked bucket
            // — this is a wire-layer bug; surface it loudly.
            progress.done.store(true, Ordering::Release);
            let _ = emitter.await;
            let _ = task_tx.send(BucketTaskUpdate::BuildEnded {
                bucket_id,
                pod_id,
                slot_id: String::new(),
                outcome: BucketBuildOutcome::Error {
                    message: format!(
                        "resync requested for non-tracked bucket (source kind: {})",
                        match source {
                            BuildSource::Ready(_) => "stored/linked",
                            BuildSource::Tracked { .. } => unreachable!(),
                        }
                    ),
                },
                requester_conn,
                correlation_id,
            });
            return;
        }
        (BuildIntent::Normal, source) => {
            match resolve_source(source, &bucket_dir, observer_arc.as_ref(), &cancel).await {
                Ok(a) => (a, None),
                Err(e) => {
                    progress.done.store(true, Ordering::Release);
                    let _ = emitter.await;
                    let outcome = match e {
                        BucketError::Cancelled => BucketBuildOutcome::Cancelled,
                        other => BucketBuildOutcome::Error {
                            message: other.to_string(),
                        },
                    };
                    let _ = task_tx.send(BucketTaskUpdate::BuildEnded {
                        bucket_id,
                        pod_id,
                        slot_id: String::new(),
                        outcome,
                        requester_conn,
                        correlation_id,
                    });
                    return;
                }
            }
        }
    };

    let observer_for_build = observer_arc.clone();
    let cancel_for_build = cancel.clone();
    let handle = tokio::runtime::Handle::current();
    let bucket_dir_for_build = bucket_dir.clone();
    let bucket_id_for_build = bucket_id.clone();
    let typed_id = match pod_id.as_deref() {
        None => BucketId::server(bucket_id_for_build),
        Some(_) => BucketId::pod(bucket_id_for_build),
    };
    let result = tokio::task::spawn_blocking(move || {
        // Open the DiskBucket inside `spawn_blocking` so the HNSW
        // load — which can run ~12s on a multi-GiB bucket — never
        // touches the scheduler task. Re-opens an existing active
        // slot for rebuild flows; cheap when no active slot exists
        // (the common first-build case).
        let bucket = match DiskBucket::open(bucket_dir_for_build, typed_id) {
            Ok(b) => Arc::new(b),
            Err(e) => return Err(BucketError::Other(format!("open failed: {e}"))),
        };

        // Resume detection runs only for Normal intent. Resync
        // builds always start a fresh slot — any Planning/Building
        // slot left over from a prior interrupted build was built
        // off the *old* base and would be inconsistent with the
        // new one.
        let resume_slot_id = if matches!(intent, BuildIntent::Normal) {
            match bucket.find_resumable_slot() {
                Ok(opt) => opt,
                Err(e) => {
                    return Err(BucketError::Other(format!(
                        "resumable-slot scan failed: {e}"
                    )));
                }
            }
        } else {
            None
        };

        let resolved = match resolve_chunker(&chunker_config, embedder_model_id.as_deref()) {
            Ok(r) => r,
            Err(e) => return Err(BucketError::Other(format!("chunker resolution: {e}"))),
        };
        let chunker_box = resolved.chunker;
        let chunker_snapshot = resolved.snapshot;
        let chunker_ref: &(dyn Chunker + Send + Sync) = chunker_box.as_ref();

        handle.block_on(async move {
            match resume_slot_id {
                Some(slot_id) => {
                    bucket
                        .resume_slot(
                            &slot_id,
                            adapter.as_ref(),
                            chunker_ref,
                            chunker_snapshot,
                            embedder,
                            Some(observer_for_build.as_ref()),
                            &cancel_for_build,
                        )
                        .await
                }
                None => {
                    bucket
                        .build_slot(
                            adapter.as_ref(),
                            chunker_ref,
                            chunker_snapshot,
                            embedder,
                            Some(observer_for_build.as_ref()),
                            &cancel_for_build,
                        )
                        .await
                }
            }
        })
    })
    .await;

    // Stop the emitter before terminal events, so a stray Progress
    // doesn't land after Ended.
    progress.done.store(true, Ordering::Release);
    let _ = emitter.await;

    let (mut outcome, slot_id) = match result {
        Ok(Ok(slot_id)) => (BucketBuildOutcome::Success, slot_id),
        Ok(Err(BucketError::Cancelled)) => (BucketBuildOutcome::Cancelled, String::new()),
        Ok(Err(e)) => (
            BucketBuildOutcome::Error {
                message: e.to_string(),
            },
            String::new(),
        ),
        Err(join_err) => (
            BucketBuildOutcome::Error {
                message: format!("build task panicked: {join_err}"),
            },
            String::new(),
        ),
    };

    // Resync post-success: reset the delta cursor to the new base
    // snapshot id so the next FeedWorker tick lists deltas strictly
    // after the new base. Both ids share `YYYYMMDD` lexicographic
    // ordering with `list_deltas_since`'s `since` arg, so reuse is
    // exact, not approximate. If the cursor reset itself fails we
    // demote the outcome to Error — silently leaving the cursor on
    // its old value would cause the next poll to re-apply every
    // delta since the *old* base, which is more confusing than
    // surfacing the error and letting the operator retry.
    if matches!(outcome, BucketBuildOutcome::Success)
        && let Some(new_id) = resync_new_snapshot_id.as_ref()
    {
        if let Err(e) = reset_delta_cursor_to(&bucket_dir, new_id) {
            outcome = BucketBuildOutcome::Error {
                message: format!("resync built new slot but cursor reset failed: {e}"),
            };
        } else if let Err(e) = stamp_last_resync_at(&bucket_dir, chrono::Utc::now()) {
            outcome = BucketBuildOutcome::Error {
                message: format!(
                    "resync built new slot and reset cursor but timestamp write failed: {e}"
                ),
            };
        } else {
            tracing::info!(
                bucket_id = %bucket_id,
                snapshot_id = %new_id,
                "resync complete: new slot active, delta cursor reset, last_resync_at stamped",
            );
        }
    }

    let _ = task_tx.send(BucketTaskUpdate::BuildEnded {
        bucket_id,
        pod_id,
        slot_id,
        outcome,
        requester_conn,
        correlation_id,
    });
}

pub struct ProgressShared {
    source_records: AtomicU64,
    chunks: AtomicU64,
    /// Phase encoded as a u8 (0=Indexing, 1=BuildingDense, 2=Finalizing)
    /// so the throttled emitter can read without a lock.
    phase: AtomicU8,
    done: AtomicBool,
    /// Wall-clock time the build was dispatched. Set once at
    /// construction; the UI uses it to render an elapsed-time stopwatch
    /// and an ETA later. Replayed verbatim to clients that join
    /// mid-build via `replay_active_builds_to_client` so a fresh
    /// connection sees the *original* started_at, not the moment its
    /// connection landed.
    started_at: chrono::DateTime<chrono::Utc>,
}

impl Default for ProgressShared {
    fn default() -> Self {
        Self {
            source_records: AtomicU64::new(0),
            chunks: AtomicU64::new(0),
            phase: AtomicU8::new(0),
            done: AtomicBool::new(false),
            started_at: chrono::Utc::now(),
        }
    }
}

impl ProgressShared {
    pub fn snapshot(&self) -> ProgressSnapshot {
        ProgressSnapshot {
            source_records: self.source_records.load(Ordering::Acquire),
            chunks: self.chunks.load(Ordering::Acquire),
            phase: phase_from_u8(self.phase.load(Ordering::Acquire)),
            started_at: self.started_at,
        }
    }

    pub fn started_at(&self) -> chrono::DateTime<chrono::Utc> {
        self.started_at
    }
}

pub struct ProgressSnapshot {
    pub source_records: u64,
    pub chunks: u64,
    pub phase: BucketBuildPhase,
    pub started_at: chrono::DateTime<chrono::Utc>,
}

// Phase ↔ u8 mapping. Planning is encoded as 0 so the AtomicU8's
// `default()` zero-init lands on Planning — the right "we haven't
// emitted anything yet" state for stored / linked / managed builds
// where Downloading doesn't apply. Downloading lives at a higher
// number; tracked builds publish it explicitly from the run_build
// prelude before any other phase fires.
fn phase_to_u8(p: BuildPhase) -> u8 {
    match p {
        BuildPhase::Planning => 0,
        BuildPhase::Indexing => 1,
        BuildPhase::BuildingDense => 2,
        BuildPhase::Finalizing => 3,
        BuildPhase::Downloading => 4,
    }
}

fn phase_from_u8(b: u8) -> BucketBuildPhase {
    match b {
        1 => BucketBuildPhase::Indexing,
        2 => BucketBuildPhase::BuildingDense,
        3 => BucketBuildPhase::Finalizing,
        4 => BucketBuildPhase::Downloading,
        _ => BucketBuildPhase::Planning,
    }
}

struct ProgressObserver {
    shared: Arc<ProgressShared>,
}

impl BuildObserver for ProgressObserver {
    fn on_phase(&self, phase: BuildPhase) {
        self.shared
            .phase
            .store(phase_to_u8(phase), Ordering::Release);
    }
    fn on_progress(&self, source_records: u64, chunks: u64) {
        self.shared
            .source_records
            .store(source_records, Ordering::Release);
        self.shared.chunks.store(chunks, Ordering::Release);
    }
}

fn validate_bucket_id(id: &str) -> Result<(), String> {
    if id.is_empty() {
        return Err("bucket id must not be empty".into());
    }
    if id.starts_with('.') {
        return Err("bucket id must not start with '.' (reserved for hidden dirs)".into());
    }
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
    {
        return Err(format!(
            "bucket id `{id}` contains invalid chars; allowed: ASCII alphanumeric, _ - ."
        ));
    }
    Ok(())
}

fn validate_source(source: &BucketSourceInput) -> Result<(), String> {
    match source {
        BucketSourceInput::Stored {
            adapter,
            archive_path,
        } => {
            if adapter != "mediawiki_xml" {
                return Err(format!(
                    "source.adapter `{adapter}` not supported for kind=stored \
                     (only `mediawiki_xml` today)"
                ));
            }
            if archive_path.trim().is_empty() {
                return Err("source.archive_path must not be empty for kind=stored".into());
            }
            if !Path::new(archive_path).exists() {
                return Err(format!(
                    "source.archive_path `{archive_path}` does not exist on the server"
                ));
            }
        }
        BucketSourceInput::Linked { adapter, path } => {
            if adapter != "markdown_dir" {
                return Err(format!(
                    "source.adapter `{adapter}` not supported for kind=linked \
                     (only `markdown_dir` today)"
                ));
            }
            if path.trim().is_empty() {
                return Err("source.path must not be empty for kind=linked".into());
            }
            if !Path::new(path).exists() {
                return Err(format!("source.path `{path}` does not exist on the server"));
            }
        }
        BucketSourceInput::Managed {} => {}
        BucketSourceInput::Tracked { driver, .. } => match driver {
            TrackedDriverInput::Wikipedia { language, mirror } => {
                if language.trim().is_empty() {
                    return Err("source.language must not be empty for driver=wikipedia".into());
                }
                if let Some(m) = mirror.as_deref()
                    && !m.is_empty()
                    && !(m.starts_with("http://") || m.starts_with("https://"))
                {
                    return Err(format!(
                        "source.mirror `{m}` must be an http(s):// URL when set"
                    ));
                }
            }
        },
    }
    Ok(())
}

fn synthesize_bucket_toml(config: &BucketCreateInput) -> String {
    let now = chrono::Utc::now().to_rfc3339();
    let description_line = match &config.description {
        Some(d) if !d.is_empty() => format!("description = {}\n", toml_quote(d)),
        _ => String::new(),
    };
    let source_block = match &config.source {
        BucketSourceInput::Stored {
            adapter,
            archive_path,
        } => format!(
            "[source]\nkind = \"stored\"\nadapter = {}\narchive_path = {}\n",
            toml_quote(adapter),
            toml_quote(archive_path),
        ),
        BucketSourceInput::Linked { adapter, path } => format!(
            "[source]\nkind = \"linked\"\nadapter = {}\npath = {}\n",
            toml_quote(adapter),
            toml_quote(path),
        ),
        BucketSourceInput::Managed {} => "[source]\nkind = \"managed\"\n".to_string(),
        BucketSourceInput::Tracked {
            driver,
            delta_cadence,
            resync_cadence,
        } => render_tracked_source_block(driver, *delta_cadence, *resync_cadence),
    };
    let quantization = match config.quantization.unwrap_or_default() {
        QuantizationInput::F32 => "f32",
        QuantizationInput::F16 => "f16",
        QuantizationInput::Int8 => "int8",
    };
    format!(
        "name = {name}\n{description}created_at = \"{now}\"\n\n\
         {source}\n\
         [chunker]\n\
         strategy = \"token_based\"\n\
         chunk_tokens = {chunk_tokens}\n\
         overlap_tokens = {overlap_tokens}\n\n\
         [search_paths.dense]\n\
         enabled = {dense}\n\
         [search_paths.sparse]\n\
         enabled = {sparse}\n\
         tokenizer = \"default\"\n\n\
         [defaults]\n\
         embedder = {embedder}\n\
         serving_mode = \"ram\"\n\
         quantization = \"{quantization}\"\n",
        name = toml_quote(&config.name),
        description = description_line,
        source = source_block,
        chunk_tokens = config.chunk_tokens,
        overlap_tokens = config.overlap_tokens,
        dense = config.dense_enabled,
        sparse = config.sparse_enabled,
        embedder = toml_quote(&config.embedder),
    )
}

/// Render the `[source]` block for a tracked-kind bucket. Mirrors
/// the TOML shape that `crate::knowledge::SourceConfig::Tracked`
/// parses: `kind`, then the driver-specific fields flattened
/// alongside (`driver = "wikipedia"`, `language`, optional `mirror`),
/// then the cadence fields.
fn render_tracked_source_block(
    driver: &TrackedDriverInput,
    delta_cadence: TrackedCadenceInput,
    resync_cadence: TrackedCadenceInput,
) -> String {
    let mut s = String::from("[source]\nkind = \"tracked\"\n");
    match driver {
        TrackedDriverInput::Wikipedia { language, mirror } => {
            s.push_str("driver = \"wikipedia\"\n");
            s.push_str(&format!("language = {}\n", toml_quote(language)));
            if let Some(m) = mirror.as_deref().filter(|m| !m.is_empty()) {
                s.push_str(&format!("mirror = {}\n", toml_quote(m)));
            }
        }
    }
    s.push_str(&format!(
        "delta_cadence = \"{}\"\n",
        tracked_cadence_str(delta_cadence),
    ));
    s.push_str(&format!(
        "resync_cadence = \"{}\"\n",
        tracked_cadence_str(resync_cadence),
    ));
    s
}

/// Wire-cadence → TOML literal. The string values match
/// `#[serde(rename_all = "snake_case")]` on
/// `crate::knowledge::TrackedCadence`, so the synthesized TOML
/// round-trips through `BucketConfig::from_toml_str`.
fn tracked_cadence_str(c: TrackedCadenceInput) -> &'static str {
    match c {
        TrackedCadenceInput::Daily => "daily",
        TrackedCadenceInput::Weekly => "weekly",
        TrackedCadenceInput::Monthly => "monthly",
        TrackedCadenceInput::Quarterly => "quarterly",
        TrackedCadenceInput::Manual => "manual",
    }
}

/// Quote a string for TOML basic-string inclusion. Bucket fields are
/// short identifiers in practice — this handles `\` / `"` and replaces
/// raw newlines (which TOML basic strings forbid) with spaces.
fn toml_quote(s: &str) -> String {
    let escaped = s
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace(['\n', '\r'], " ");
    format!("\"{escaped}\"")
}

/// Build-time view of a bucket's source. `Ready` carries an already-
/// constructed `SourceAdapter` (the stored / linked path); `Tracked`
/// carries the feed driver so the run_build prelude can download the
/// base snapshot first, then construct a `MediaWikiXml` adapter
/// against the cached file.
enum BuildSource {
    Ready(Box<dyn SourceAdapter + Send + Sync>),
    Tracked { driver: Box<dyn FeedDriver> },
}

/// Async preamble to the build pipeline. For stored / linked, the
/// adapter is already constructed — this is a no-op handoff. For
/// tracked, the driver is consulted: if `feed-state.toml` already
/// records a `current_base_snapshot_id`, we reuse it (subsequent
/// rebuilds operate on the same base; only an explicit Resync advances
/// to a newer one); otherwise we ask the driver for `latest_base()`.
/// Either way we then fetch the base into `source-cache/base/<id>/`
/// and write the snapshot id back to `feed-state.toml` before
/// returning a `MediaWikiXml` adapter pointing at the downloaded file.
///
/// Phase=Downloading is published via the observer for the duration of
/// the fetch so the UI can distinguish download from indexing.
async fn resolve_source(
    source: BuildSource,
    bucket_dir: &Path,
    observer: &dyn BuildObserver,
    cancel: &CancellationToken,
) -> Result<Box<dyn SourceAdapter + Send + Sync>, BucketError> {
    match source {
        BuildSource::Ready(adapter) => Ok(adapter),
        BuildSource::Tracked { driver } => {
            observer.on_phase(BuildPhase::Downloading);

            let mut state = FeedState::load(bucket_dir)
                .map_err(|e| BucketError::Other(format!("feed-state load: {e}")))?;

            let snapshot_id: SnapshotId = match state.current_base() {
                Some(id) => id,
                None => driver
                    .latest_base(cancel)
                    .await
                    .map_err(|e| BucketError::Other(format!("feed driver latest_base: {e}")))?,
            };

            let dest_dir = base_cache_dir(bucket_dir, &snapshot_id);
            let dest = dest_dir.join(driver.base_filename(&snapshot_id));
            driver
                .fetch_base(&snapshot_id, &dest, cancel)
                .await
                .map_err(|e| match e {
                    crate::knowledge::FeedError::Cancelled => BucketError::Cancelled,
                    other => BucketError::Other(format!("feed driver fetch_base: {other}")),
                })?;

            // Persist the snapshot id only after the file is in place —
            // a partial write should never produce a state file
            // claiming a base we don't actually have.
            state.set_current_base(snapshot_id.clone());
            state
                .save_atomic(bucket_dir)
                .map_err(|e| BucketError::Other(format!("feed-state save: {e}")))?;

            // The parser is selected by the driver. Today every driver
            // points at "mediawiki_xml"; the lookup keeps the door open
            // for future drivers (e.g. wikidata-rdf) without forcing a
            // hardcoded match here.
            match driver.parse_adapter() {
                "mediawiki_xml" => Ok(Box::new(MediaWikiXml::new(&dest))),
                other => Err(BucketError::Other(format!(
                    "tracked driver `{}` reports parse_adapter `{other}`, \
                     which has no matching SourceAdapter binding",
                    driver.parse_adapter(),
                ))),
            }
        }
    }
}

fn build_source(source: &crate::knowledge::SourceConfig) -> Result<BuildSource, String> {
    use crate::knowledge::SourceConfig;
    match source {
        SourceConfig::Stored {
            adapter,
            archive_path,
        } => {
            if adapter != "mediawiki_xml" {
                return Err(format!("unsupported stored adapter `{adapter}`"));
            }
            Ok(BuildSource::Ready(Box::new(MediaWikiXml::new(
                archive_path,
            ))))
        }
        SourceConfig::Linked { adapter, path, .. } => {
            if adapter != "markdown_dir" {
                return Err(format!("unsupported linked adapter `{adapter}`"));
            }
            Ok(BuildSource::Ready(Box::new(MarkdownDir::new(path))))
        }
        SourceConfig::Managed {} => {
            // A managed bucket's "build" produces an empty slot — no
            // base chunks. Subsequent `knowledge_modify` inserts land
            // in that slot's delta layer just like any other mutation.
            // Without this StartBuild path, the bucket would have no
            // active slot for `Bucket::insert` to bind to.
            Ok(BuildSource::Ready(Box::new(
                crate::knowledge::source::EmptySource,
            )))
        }
        SourceConfig::Tracked { driver, .. } => {
            // The cadence fields (delta_cadence, resync_cadence) are
            // consumed by the per-bucket worker, not the initial-build
            // path — only the driver itself matters here.
            Ok(BuildSource::Tracked {
                driver: driver_for(driver),
            })
        }
    }
}

/// Outcome of resolving a tracked bucket's source for a *resync* — i.e.
/// when the caller wants to advance to the latest base regardless of
/// what's recorded in `feed-state.toml`.
///
/// `AlreadyAtLatest` is returned when the driver's `latest_base()`
/// matches the recorded `current_base_snapshot_id`; the caller can
/// skip the rebuild without touching disk. `Build` carries a fresh
/// adapter pointing at the just-downloaded base file plus the new
/// snapshot id (so the caller can advance the delta cursor on success).
enum ResyncResolution {
    AlreadyAtLatest {
        snapshot_id: SnapshotId,
    },
    Build {
        adapter: Box<dyn SourceAdapter + Send + Sync>,
        new_snapshot_id: SnapshotId,
    },
}

/// Async preamble to a *resync* build pipeline. Counterpart to
/// [`resolve_source`] for the case where the caller wants to advance
/// to the driver's current `latest_base()` regardless of any snapshot
/// id already recorded in `feed-state.toml`.
///
/// Returns [`ResyncResolution::AlreadyAtLatest`] when there's nothing
/// to do — the recorded `current_base_snapshot_id` matches the driver's
/// latest. Otherwise downloads the new base into
/// `source-cache/base/<new_id>/`, persists the new snapshot id, and
/// returns a [`SourceAdapter`] pointing at the freshly-cached file
/// alongside the new id (so the caller can reset the delta cursor on
/// successful slot rebuild via [`reset_delta_cursor_to`]).
///
/// Phase=Downloading is published via the observer for the duration of
/// the fetch — same UI affordance as the initial-build path.
async fn resolve_resync_source(
    driver: Box<dyn FeedDriver>,
    bucket_dir: &Path,
    observer: &dyn BuildObserver,
    cancel: &CancellationToken,
) -> Result<ResyncResolution, BucketError> {
    let latest = driver
        .latest_base(cancel)
        .await
        .map_err(|e| BucketError::Other(format!("feed driver latest_base: {e}")))?;

    let mut state = FeedState::load(bucket_dir)
        .map_err(|e| BucketError::Other(format!("feed-state load: {e}")))?;

    if state.current_base().as_ref() == Some(&latest) {
        return Ok(ResyncResolution::AlreadyAtLatest {
            snapshot_id: latest,
        });
    }

    observer.on_phase(BuildPhase::Downloading);

    let dest_dir = base_cache_dir(bucket_dir, &latest);
    let dest = dest_dir.join(driver.base_filename(&latest));
    driver
        .fetch_base(&latest, &dest, cancel)
        .await
        .map_err(|e| match e {
            crate::knowledge::FeedError::Cancelled => BucketError::Cancelled,
            other => BucketError::Other(format!("feed driver fetch_base: {other}")),
        })?;

    // Persist the new snapshot id only after the file is in place — a
    // partial write should never produce a state file claiming a base
    // we don't actually have. Mirrors `resolve_source`.
    state.set_current_base(latest.clone());
    state
        .save_atomic(bucket_dir)
        .map_err(|e| BucketError::Other(format!("feed-state save: {e}")))?;

    let adapter: Box<dyn SourceAdapter + Send + Sync> = match driver.parse_adapter() {
        "mediawiki_xml" => Box::new(MediaWikiXml::new(&dest)),
        other => {
            return Err(BucketError::Other(format!(
                "tracked driver `{}` reports parse_adapter `{other}`, \
                 which has no matching SourceAdapter binding",
                driver.parse_adapter(),
            )));
        }
    };

    Ok(ResyncResolution::Build {
        adapter,
        new_snapshot_id: latest,
    })
}

/// After a successful resync slot rebuild against `new_snapshot_id`,
/// reset `feed-state.toml`'s `last_applied_delta_id` so the next
/// FeedWorker tick lists deltas strictly after the new base. Both ids
/// share the same `YYYYMMDD` lexicographic order that
/// `WikipediaDriver::list_deltas_since` uses, so reusing the snapshot
/// id as a delta cursor is exact, not an approximation.
///
/// This is the correctness reset for the rare case where deltas dated
/// after the old base were applied to the old slot during the resync
/// build: re-applying them to the new slot is idempotent (apply_delta's
/// "first chunk_id already live" check), and skipping ones already
/// folded into the new base is a no-op.
fn reset_delta_cursor_to(
    bucket_dir: &Path,
    new_snapshot_id: &SnapshotId,
) -> Result<(), BucketError> {
    let mut state = FeedState::load(bucket_dir)
        .map_err(|e| BucketError::Other(format!("feed-state load: {e}")))?;
    state.set_last_applied_delta(DeltaId::new(new_snapshot_id.as_str()));
    state
        .save_atomic(bucket_dir)
        .map_err(|e| BucketError::Other(format!("feed-state save: {e}")))?;
    Ok(())
}

/// Stamp `feed-state.toml`'s `last_resync_at` so the FeedWorker's
/// resync-cadence scheduling survives server restarts. Called only
/// after a successful Resync slot rebuild + delta-cursor reset.
fn stamp_last_resync_at(
    bucket_dir: &Path,
    at: chrono::DateTime<chrono::Utc>,
) -> Result<(), BucketError> {
    let mut state = FeedState::load(bucket_dir)
        .map_err(|e| BucketError::Other(format!("feed-state load: {e}")))?;
    state.set_last_resync_at(at);
    state
        .save_atomic(bucket_dir)
        .map_err(|e| BucketError::Other(format!("feed-state save: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use crate::knowledge::source::feed::BoxFuture;

    /// Shared call log so a test holding an `Arc` can inspect calls
    /// after the driver has been moved into a `BuildSource`.
    type CallLog = Arc<Mutex<Vec<String>>>;

    /// Synthetic `FeedDriver` for resolve_source tests. Records every
    /// method call into a caller-shared log so assertions can verify
    /// call ordering / arguments without spinning up a real HTTP
    /// server.
    struct FakeDriver {
        latest: SnapshotId,
        payload: Vec<u8>,
        log: CallLog,
    }

    impl FakeDriver {
        fn new(latest: &str, payload: &[u8]) -> (Self, CallLog) {
            let log: CallLog = Arc::new(Mutex::new(Vec::new()));
            (
                Self {
                    latest: SnapshotId::new(latest),
                    payload: payload.to_vec(),
                    log: log.clone(),
                },
                log,
            )
        }
    }

    impl FeedDriver for FakeDriver {
        fn latest_base<'a>(
            &'a self,
            _cancel: &'a CancellationToken,
        ) -> BoxFuture<'a, Result<SnapshotId, crate::knowledge::FeedError>> {
            Box::pin(async move {
                self.log.lock().unwrap().push("latest_base".into());
                Ok(self.latest.clone())
            })
        }

        fn fetch_base<'a>(
            &'a self,
            id: &'a SnapshotId,
            dest: &'a Path,
            _cancel: &'a CancellationToken,
        ) -> BoxFuture<'a, Result<(), crate::knowledge::FeedError>> {
            let payload = self.payload.clone();
            Box::pin(async move {
                self.log.lock().unwrap().push(format!(
                    "fetch_base({},{})",
                    id.as_str(),
                    dest.display()
                ));
                if let Some(parent) = dest.parent() {
                    tokio::fs::create_dir_all(parent).await.unwrap();
                }
                tokio::fs::write(dest, &payload).await.unwrap();
                Ok(())
            })
        }

        fn list_deltas_since<'a>(
            &'a self,
            _since: Option<&'a DeltaId>,
            _cancel: &'a CancellationToken,
        ) -> BoxFuture<'a, Result<Vec<DeltaId>, crate::knowledge::FeedError>> {
            Box::pin(async move { Ok(Vec::new()) })
        }

        fn fetch_delta<'a>(
            &'a self,
            _id: &'a DeltaId,
            _dest: &'a Path,
            _cancel: &'a CancellationToken,
        ) -> BoxFuture<'a, Result<(), crate::knowledge::FeedError>> {
            Box::pin(async move { Ok(()) })
        }

        fn parse_adapter(&self) -> &'static str {
            "mediawiki_xml"
        }

        fn base_filename(&self, id: &SnapshotId) -> String {
            format!("fake-{}.xml.bz2", id.as_str())
        }

        fn delta_filename(&self, id: &DeltaId) -> String {
            format!("fake-delta-{}.xml.bz2", id.as_str())
        }
    }

    #[derive(Default)]
    struct PhaseRecorder {
        phases: Mutex<Vec<BuildPhase>>,
    }

    impl BuildObserver for PhaseRecorder {
        fn on_phase(&self, phase: BuildPhase) {
            self.phases.lock().unwrap().push(phase);
        }
        fn on_progress(&self, _source_records: u64, _chunks: u64) {}
    }

    #[tokio::test]
    async fn resolve_source_ready_passes_adapter_through_unchanged() {
        let tmp = tempfile::tempdir().unwrap();
        // Use a markdown_dir adapter as a stand-in — the assertion is
        // about pass-through, not about what kind of adapter rides
        // along.
        let adapter: Box<dyn SourceAdapter + Send + Sync> = Box::new(MarkdownDir::new(tmp.path()));
        let observer = PhaseRecorder::default();
        let cancel = CancellationToken::new();
        let _adapter = resolve_source(BuildSource::Ready(adapter), tmp.path(), &observer, &cancel)
            .await
            .unwrap();
        // Ready path doesn't emit a Downloading phase.
        assert!(observer.phases.lock().unwrap().is_empty());
        // No feed-state file should have been created.
        assert!(!tmp.path().join("feed-state.toml").exists());
    }

    #[tokio::test]
    async fn resolve_source_tracked_downloads_writes_feed_state_and_returns_adapter() {
        let tmp = tempfile::tempdir().unwrap();
        let payload = b"<?xml version=\"1.0\"?><mediawiki></mediawiki>";
        let (driver, log) = FakeDriver::new("20260401", payload);
        let driver: Box<dyn FeedDriver> = Box::new(driver);
        let observer = PhaseRecorder::default();
        let cancel = CancellationToken::new();

        let _adapter = resolve_source(
            BuildSource::Tracked { driver },
            tmp.path(),
            &observer,
            &cancel,
        )
        .await
        .unwrap();

        // Phase=Downloading was emitted exactly once.
        assert_eq!(
            observer.phases.lock().unwrap().as_slice(),
            &[BuildPhase::Downloading]
        );

        // Empty feed-state caused latest_base to be consulted, then fetch_base.
        let calls = log.lock().unwrap().clone();
        assert_eq!(calls.first().map(|s| s.as_str()), Some("latest_base"));
        assert!(
            calls.iter().any(|c| c.starts_with("fetch_base(20260401")),
            "expected fetch_base(20260401, ...) in {calls:?}"
        );

        // feed-state.toml was written with the chosen snapshot id.
        let state = FeedState::load(tmp.path()).unwrap();
        assert_eq!(state.current_base(), Some(SnapshotId::new("20260401")));

        // The base file landed in the right cache directory under the
        // driver's chosen filename.
        let expected_path = tmp
            .path()
            .join("source-cache/base/20260401/fake-20260401.xml.bz2");
        assert!(
            expected_path.exists(),
            "missing {}",
            expected_path.display()
        );
        assert_eq!(std::fs::read(&expected_path).unwrap(), payload);
    }

    #[tokio::test]
    async fn resolve_source_tracked_reuses_existing_snapshot_id_when_recorded() {
        let tmp = tempfile::tempdir().unwrap();
        // Pre-record a snapshot id — resolve_source should reuse it
        // rather than calling latest_base.
        let mut state = FeedState::default();
        state.set_current_base(SnapshotId::new("20260301"));
        state.save_atomic(tmp.path()).unwrap();

        let (driver, log) = FakeDriver::new("20260401", b"payload");
        let driver: Box<dyn FeedDriver> = Box::new(driver);
        let observer = PhaseRecorder::default();
        let cancel = CancellationToken::new();

        let _adapter = resolve_source(
            BuildSource::Tracked { driver },
            tmp.path(),
            &observer,
            &cancel,
        )
        .await
        .unwrap();

        let calls = log.lock().unwrap().clone();
        assert!(
            !calls.iter().any(|c| c == "latest_base"),
            "latest_base should not be called when state has a snapshot id; got {calls:?}"
        );
        assert!(calls.iter().any(|c| c.starts_with("fetch_base(20260301")));

        // State unchanged — still pointing at the recorded snapshot.
        let after = FeedState::load(tmp.path()).unwrap();
        assert_eq!(after.current_base(), Some(SnapshotId::new("20260301")));
    }

    #[tokio::test]
    async fn resolve_resync_source_returns_already_at_latest_when_current_matches() {
        let tmp = tempfile::tempdir().unwrap();
        let mut state = FeedState::default();
        state.set_current_base(SnapshotId::new("20260401"));
        state.save_atomic(tmp.path()).unwrap();

        let (driver, log) = FakeDriver::new("20260401", b"unused");
        let driver: Box<dyn FeedDriver> = Box::new(driver);
        let observer = PhaseRecorder::default();
        let cancel = CancellationToken::new();

        let res = resolve_resync_source(driver, tmp.path(), &observer, &cancel)
            .await
            .unwrap();

        match res {
            ResyncResolution::AlreadyAtLatest { snapshot_id } => {
                assert_eq!(snapshot_id, SnapshotId::new("20260401"));
            }
            ResyncResolution::Build { .. } => panic!("expected AlreadyAtLatest"),
        }

        let calls = log.lock().unwrap().clone();
        assert_eq!(calls.first().map(|s| s.as_str()), Some("latest_base"));
        assert!(
            !calls.iter().any(|c| c.starts_with("fetch_base")),
            "fetch_base must not run when already at latest; got {calls:?}"
        );
        assert!(
            observer.phases.lock().unwrap().is_empty(),
            "no Downloading phase should fire on the no-op path"
        );
    }

    #[tokio::test]
    async fn resolve_resync_source_fetches_new_base_when_latest_differs() {
        let tmp = tempfile::tempdir().unwrap();
        let mut state = FeedState::default();
        state.set_current_base(SnapshotId::new("20260301"));
        state.set_last_applied_delta(DeltaId::new("20260315"));
        state.save_atomic(tmp.path()).unwrap();

        let payload = b"<?xml version=\"1.0\"?><mediawiki></mediawiki>";
        let (driver, log) = FakeDriver::new("20260401", payload);
        let driver: Box<dyn FeedDriver> = Box::new(driver);
        let observer = PhaseRecorder::default();
        let cancel = CancellationToken::new();

        let res = resolve_resync_source(driver, tmp.path(), &observer, &cancel)
            .await
            .unwrap();

        let new_id = match res {
            ResyncResolution::Build {
                new_snapshot_id, ..
            } => new_snapshot_id,
            ResyncResolution::AlreadyAtLatest { .. } => panic!("expected Build"),
        };
        assert_eq!(new_id, SnapshotId::new("20260401"));

        // Phase=Downloading was emitted exactly once.
        assert_eq!(
            observer.phases.lock().unwrap().as_slice(),
            &[BuildPhase::Downloading]
        );

        let calls = log.lock().unwrap().clone();
        assert_eq!(calls.first().map(|s| s.as_str()), Some("latest_base"));
        assert!(calls.iter().any(|c| c.starts_with("fetch_base(20260401")));

        // current_base advanced to the new id.
        let after = FeedState::load(tmp.path()).unwrap();
        assert_eq!(after.current_base(), Some(SnapshotId::new("20260401")));
        // last_applied_delta_id is *unchanged* by resolve — the cursor
        // reset is a separate step (`reset_delta_cursor_to`) that the
        // orchestrator runs only after a successful rebuild.
        assert_eq!(
            after.last_applied_delta(),
            Some(DeltaId::new("20260315")),
            "resolve_resync_source must not touch the delta cursor",
        );

        // The base file landed.
        let expected_path = tmp
            .path()
            .join("source-cache/base/20260401/fake-20260401.xml.bz2");
        assert!(expected_path.exists());
    }

    #[tokio::test]
    async fn resolve_resync_source_handles_unset_current_base() {
        // Edge case: a tracked bucket whose feed-state was never
        // written (e.g. resync triggered before the initial build
        // somehow). `latest_base()` still applies; we just have
        // nothing to compare against, so the resync proceeds.
        let tmp = tempfile::tempdir().unwrap();

        let (driver, _log) = FakeDriver::new("20260401", b"payload");
        let driver: Box<dyn FeedDriver> = Box::new(driver);
        let observer = PhaseRecorder::default();
        let cancel = CancellationToken::new();

        let res = resolve_resync_source(driver, tmp.path(), &observer, &cancel)
            .await
            .unwrap();
        assert!(matches!(res, ResyncResolution::Build { .. }));

        let after = FeedState::load(tmp.path()).unwrap();
        assert_eq!(after.current_base(), Some(SnapshotId::new("20260401")));
    }

    /// Verifies the synthesized TOML for a `Tracked` bucket round-trips
    /// through `BucketConfig::from_toml_str` — same parser the registry
    /// uses at startup. Catches a class of bugs where the wire enum's
    /// rename (`snake_case`) drifts from what the TOML grammar expects.
    #[test]
    fn synthesize_tracked_bucket_toml_round_trips() {
        let cfg = BucketCreateInput {
            name: "Wikipedia (simple)".to_string(),
            description: Some("simple english wiki".to_string()),
            source: BucketSourceInput::Tracked {
                driver: TrackedDriverInput::Wikipedia {
                    language: "simple".to_string(),
                    mirror: Some("https://example.com/dumps".to_string()),
                },
                delta_cadence: TrackedCadenceInput::Daily,
                resync_cadence: TrackedCadenceInput::Monthly,
            },
            embedder: "qwen3_embed_0_6b".to_string(),
            chunk_tokens: 500,
            overlap_tokens: 50,
            dense_enabled: true,
            sparse_enabled: true,
            quantization: None,
        };
        let toml_text = synthesize_bucket_toml(&cfg);
        let parsed = crate::knowledge::BucketConfig::from_toml_str(&toml_text)
            .expect("synthesized TOML must parse");
        let crate::knowledge::SourceConfig::Tracked {
            driver,
            delta_cadence,
            resync_cadence,
        } = &parsed.source
        else {
            panic!("expected Tracked source, got {:?}", parsed.source);
        };
        let crate::knowledge::TrackedDriver::Wikipedia { language, mirror } = driver;
        assert_eq!(language, "simple");
        assert_eq!(mirror.as_deref(), Some("https://example.com/dumps"));
        assert_eq!(*delta_cadence, crate::knowledge::TrackedCadence::Daily);
        assert_eq!(*resync_cadence, crate::knowledge::TrackedCadence::Monthly);
    }

    /// `mirror = None` should omit the field entirely so the TOML
    /// parser falls back to the driver's default
    /// (`https://dumps.wikimedia.org`). Empty-string is treated as
    /// "unset" by the form, so we should never emit `mirror = ""`.
    #[test]
    fn synthesize_tracked_omits_mirror_when_unset() {
        let cfg = BucketCreateInput {
            name: "Wikipedia (en)".to_string(),
            description: None,
            source: BucketSourceInput::Tracked {
                driver: TrackedDriverInput::Wikipedia {
                    language: "en".to_string(),
                    mirror: None,
                },
                delta_cadence: TrackedCadenceInput::Manual,
                resync_cadence: TrackedCadenceInput::Manual,
            },
            embedder: "qwen3_embed_0_6b".to_string(),
            chunk_tokens: 500,
            overlap_tokens: 50,
            dense_enabled: true,
            sparse_enabled: true,
            quantization: None,
        };
        let toml_text = synthesize_bucket_toml(&cfg);
        assert!(
            !toml_text.contains("mirror"),
            "mirror should be omitted when None; got:\n{toml_text}"
        );
        let parsed = crate::knowledge::BucketConfig::from_toml_str(&toml_text)
            .expect("synthesized TOML must parse");
        let crate::knowledge::SourceConfig::Tracked { driver, .. } = &parsed.source else {
            panic!("expected Tracked source");
        };
        let crate::knowledge::TrackedDriver::Wikipedia { mirror, .. } = driver;
        assert!(mirror.is_none(), "parsed mirror must be None");
    }

    #[test]
    fn reset_delta_cursor_to_overwrites_last_applied_delta() {
        let tmp = tempfile::tempdir().unwrap();
        let mut state = FeedState::default();
        state.set_current_base(SnapshotId::new("20260401"));
        state.set_last_applied_delta(DeltaId::new("20260415"));
        state.save_atomic(tmp.path()).unwrap();

        reset_delta_cursor_to(tmp.path(), &SnapshotId::new("20260501")).unwrap();

        let after = FeedState::load(tmp.path()).unwrap();
        // current_base is left alone — that's resolve_resync_source's job.
        assert_eq!(after.current_base(), Some(SnapshotId::new("20260401")));
        // The cursor is now the new snapshot id verbatim, so the next
        // list_deltas_since(Some("20260501")) returns deltas strictly
        // after that date.
        assert_eq!(after.last_applied_delta(), Some(DeltaId::new("20260501")),);
    }
}
