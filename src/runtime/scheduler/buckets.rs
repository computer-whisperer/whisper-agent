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
    BucketBuildOutcome, BucketBuildPhase, BucketCreateInput, BucketSourceInput, ServerToClient,
};

use super::{ConnId, Scheduler};
use crate::knowledge::{BucketError, BucketId, ChunkerConfig};
use crate::knowledge::{
    BuildObserver, BuildPhase, Chunker, DiskBucket, MarkdownDir, MediaWikiXml, SourceAdapter,
    TokenBasedChunker, registry::entry_to_summary,
};
use crate::providers::embedding::EmbeddingProvider;
use crate::server::thread_router::ThreadEventRouter;

/// Throttle interval for `BucketBuildProgress` events. The build's
/// `on_progress` callback fires per chunk-batch (every 128 chunks);
/// throttling client-side to 1Hz keeps the websocket from drowning
/// the UI on dense source dumps without losing the "feels live"
/// rhythm.
const PROGRESS_THROTTLE: Duration = Duration::from_millis(1000);

/// Updates a detached build task pushes back into the scheduler loop.
/// The loop drains these from `bucket_task_rx` and applies them under
/// `&mut Scheduler` so registry mutations stay linearizable with the
/// rest of the scheduler.
///
/// Currently a single-variant enum because per-tick `BucketBuildProgress`
/// events fan out via the client-snapshot path (taken at task start) —
/// only the terminal event needs the scheduler-loop's `&mut self` to
/// refresh the registry. Kept as an enum so adding more task→loop
/// signals later doesn't churn the channel type.
pub enum BucketTaskUpdate {
    /// Build completed (success / error / cancel). The loop refreshes
    /// the registry entry on success, broadcasts `BucketBuildEnded`
    /// with the fresh summary, echoes the terminal event to the
    /// requester with their `correlation_id`, and unregisters the
    /// build from `active_bucket_builds`.
    BuildEnded {
        bucket_id: String,
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

        let Some(root) = self.bucket_registry.root.clone() else {
            self.send_bucket_error(
                conn_id,
                correlation_id,
                "create_bucket",
                "server was started without --buckets-root; cannot create buckets".into(),
            );
            return;
        };
        let bucket_dir = root.join(&id);
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
            dir: bucket_dir,
            config: parsed_config,
            raw_toml: toml_text,
            active_slot: None,
            active_slot_disk_size_bytes: None,
        };
        if let Err(e) = self.bucket_registry.insert_entry(entry.clone()) {
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
    ) {
        if let Err(e) = validate_bucket_id(&id) {
            self.send_bucket_error(conn_id, correlation_id, "delete_bucket", e);
            return;
        }
        let dir = match self.bucket_registry.buckets.get(&id) {
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
        };

        // Cancel any in-flight build first. The build task observes
        // the token and emits its own `BucketBuildEnded { Cancelled }`.
        if let Some(token) = self.active_bucket_builds.remove(&id) {
            token.cancel();
        }

        // Drop the in-memory entry + cached DiskBucket. Sync — both
        // operations are O(1) under a short-held mutex.
        self.bucket_registry.remove_entry(&id);

        // rmdir runs detached so the response isn't gated on it.
        let dir_for_rm = dir.clone();
        let id_for_log = id.clone();
        tokio::spawn(async move {
            if let Err(e) = tokio::fs::remove_dir_all(&dir_for_rm).await {
                warn!(
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
        };
        self.router.broadcast_task_list_except(ev, conn_id);
        self.router.send_to_client(
            conn_id,
            ServerToClient::BucketDeleted { correlation_id, id },
        );
    }

    pub(super) fn handle_start_bucket_build(
        &mut self,
        conn_id: ConnId,
        correlation_id: Option<String>,
        id: String,
    ) {
        if let Err(e) = validate_bucket_id(&id) {
            self.send_bucket_error(conn_id, correlation_id, "start_bucket_build", e);
            return;
        }
        if self.active_bucket_builds.contains_key(&id) {
            self.send_bucket_error(
                conn_id,
                correlation_id,
                "start_bucket_build",
                format!("build already in flight for `{id}`"),
            );
            return;
        }
        let entry = match self.bucket_registry.buckets.get(&id) {
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
        let adapter = match build_adapter(&entry.config.source) {
            Ok(a) => a,
            Err(e) => {
                self.send_bucket_error(conn_id, correlation_id, "start_bucket_build", e);
                return;
            }
        };
        let chunker = build_chunker(&entry.config.chunker);

        // Open the DiskBucket once — cheap when no active slot exists
        // (the common case for first-build); re-opens an existing
        // active slot for rebuild flows. Don't pay the HNSW load cost
        // if we can avoid it: a rebuild with no existing slot is fine,
        // but a real slot-load happens only on the next query path.
        let bucket = match DiskBucket::open(entry.dir.clone(), BucketId::server(id.clone())) {
            Ok(b) => Arc::new(b),
            Err(e) => {
                self.send_bucket_error(
                    conn_id,
                    correlation_id,
                    "start_bucket_build",
                    format!("open failed: {e}"),
                );
                return;
            }
        };

        // Resume detection: if a prior build was interrupted (Pause
        // or crash), its slot directory is still on disk in
        // Planning/Building state. Pick that up rather than starting
        // a parallel slot.
        let resume_slot_id = match bucket.find_resumable_slot() {
            Ok(opt) => opt,
            Err(e) => {
                self.send_bucket_error(
                    conn_id,
                    correlation_id,
                    "start_bucket_build",
                    format!("resumable-slot scan failed: {e}"),
                );
                return;
            }
        };

        let cancel = CancellationToken::new();
        self.active_bucket_builds.insert(id.clone(), cancel.clone());

        // Synchronous BuildStarted ack. slot_id is unknown until the
        // build's first phase callback; carry an empty slot_id until
        // the BuildEnded event lands the real one. UI ignores this
        // field on Started — only bucket_id matters for the row state.
        let started = ServerToClient::BucketBuildStarted {
            correlation_id: None,
            bucket_id: id.clone(),
            slot_id: String::new(),
        };
        self.router.broadcast_task_list_except(started, conn_id);
        self.router.send_to_client(
            conn_id,
            ServerToClient::BucketBuildStarted {
                correlation_id: correlation_id.clone(),
                bucket_id: id.clone(),
                slot_id: String::new(),
            },
        );

        let snapshot = self.router.clients_snapshot();
        let task_tx = self.bucket_task_sender();
        tokio::spawn(run_build(
            bucket,
            embedder,
            adapter,
            chunker,
            cancel,
            id,
            snapshot,
            task_tx,
            conn_id,
            correlation_id,
            resume_slot_id,
        ));
    }

    pub(super) fn handle_cancel_bucket_build(
        &mut self,
        conn_id: ConnId,
        correlation_id: Option<String>,
        id: String,
    ) {
        match self.active_bucket_builds.get(&id) {
            Some(token) => {
                // Cancellation is observed at the next chunk-batch
                // boundary or HNSW build entry; the build task emits
                // its own `BucketBuildEnded { Cancelled }` on exit.
                // No separate ack — the UI watches BuildEnded.
                token.cancel();
            }
            None => {
                self.send_bucket_error(
                    conn_id,
                    correlation_id,
                    "cancel_bucket_build",
                    format!("no in-flight build for `{id}`"),
                );
            }
        }
    }

    /// Drain a `BucketTaskUpdate` from the channel. Called from the
    /// scheduler loop's `select!` arm.
    pub(crate) async fn apply_bucket_task_update(&mut self, update: BucketTaskUpdate) {
        match update {
            BucketTaskUpdate::BuildEnded {
                bucket_id,
                slot_id,
                outcome,
                requester_conn,
                correlation_id,
            } => {
                self.active_bucket_builds.remove(&bucket_id);
                let summary = if matches!(outcome, BucketBuildOutcome::Success) {
                    match self.bucket_registry.refresh_entry(&bucket_id).await {
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
                    }
                } else {
                    None
                };

                let ended = ServerToClient::BucketBuildEnded {
                    correlation_id: None,
                    bucket_id: bucket_id.clone(),
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

#[allow(clippy::too_many_arguments)]
async fn run_build(
    bucket: Arc<DiskBucket>,
    embedder: Arc<dyn EmbeddingProvider>,
    adapter: Box<dyn SourceAdapter + Send + Sync>,
    chunker: Box<dyn Chunker + Send + Sync>,
    cancel: CancellationToken,
    bucket_id: String,
    snapshot: Vec<mpsc::UnboundedSender<ServerToClient>>,
    task_tx: mpsc::UnboundedSender<BucketTaskUpdate>,
    requester_conn: ConnId,
    correlation_id: Option<String>,
    resume_slot_id: Option<String>,
) {
    let progress = Arc::new(ProgressShared::default());
    let observer_arc: Arc<dyn BuildObserver> = Arc::new(ProgressObserver {
        shared: progress.clone(),
    });

    // Throttle / emit task: reads `progress` every PROGRESS_THROTTLE
    // and dispatches a BucketBuildProgress through the snapshot. Stops
    // when the build flips `done`.
    let snapshot_for_emit = snapshot.clone();
    let bucket_id_for_emit = bucket_id.clone();
    let progress_for_emit = progress.clone();
    let emitter = tokio::spawn(async move {
        let mut interval = tokio::time::interval(PROGRESS_THROTTLE);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            if progress_for_emit.done.load(Ordering::Acquire) {
                break;
            }
            let snap = progress_for_emit.snapshot();
            let event = ServerToClient::BucketBuildProgress {
                bucket_id: bucket_id_for_emit.clone(),
                slot_id: String::new(), // slot_id not surfaced during build; carried on Ended
                phase: snap.phase,
                source_records: snap.source_records,
                chunks: snap.chunks,
            };
            for tx in &snapshot_for_emit {
                let _ = tx.send(event.clone());
            }
        }
    });

    // Run the build inside spawn_blocking + block_on so the runtime
    // workers stay free during the multi-minute HNSW build phase. The
    // observer is dyn-dispatched; the cost is negligible per call.
    //
    // Dispatches to either the fresh `build_slot` path or
    // `resume_slot` based on whether the scheduler found an
    // in-progress slot to pick up.
    let bucket_for_build = bucket.clone();
    let observer_for_build = observer_arc.clone();
    let cancel_for_build = cancel.clone();
    let handle = tokio::runtime::Handle::current();
    let result = tokio::task::spawn_blocking(move || {
        handle.block_on(async move {
            match resume_slot_id {
                Some(slot_id) => {
                    bucket_for_build
                        .resume_slot(
                            &slot_id,
                            adapter.as_ref(),
                            chunker.as_ref(),
                            embedder,
                            Some(observer_for_build.as_ref()),
                            &cancel_for_build,
                        )
                        .await
                }
                None => {
                    bucket_for_build
                        .build_slot(
                            adapter.as_ref(),
                            chunker.as_ref(),
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

    let (outcome, slot_id) = match result {
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

    let _ = task_tx.send(BucketTaskUpdate::BuildEnded {
        bucket_id,
        slot_id,
        outcome,
        requester_conn: Some(requester_conn),
        correlation_id,
    });
}

#[derive(Default)]
struct ProgressShared {
    source_records: AtomicU64,
    chunks: AtomicU64,
    /// Phase encoded as a u8 (0=Indexing, 1=BuildingDense, 2=Finalizing)
    /// so the throttled emitter can read without a lock.
    phase: AtomicU8,
    done: AtomicBool,
}

impl ProgressShared {
    fn snapshot(&self) -> ProgressSnapshot {
        ProgressSnapshot {
            source_records: self.source_records.load(Ordering::Acquire),
            chunks: self.chunks.load(Ordering::Acquire),
            phase: phase_from_u8(self.phase.load(Ordering::Acquire)),
        }
    }
}

struct ProgressSnapshot {
    source_records: u64,
    chunks: u64,
    phase: BucketBuildPhase,
}

fn phase_to_u8(p: BuildPhase) -> u8 {
    match p {
        BuildPhase::Planning => 0,
        BuildPhase::Indexing => 1,
        BuildPhase::BuildingDense => 2,
        BuildPhase::Finalizing => 3,
    }
}

fn phase_from_u8(b: u8) -> BucketBuildPhase {
    match b {
        1 => BucketBuildPhase::Indexing,
        2 => BucketBuildPhase::BuildingDense,
        3 => BucketBuildPhase::Finalizing,
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
         quantization = \"f32\"\n",
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

fn build_adapter(
    source: &crate::knowledge::SourceConfig,
) -> Result<Box<dyn SourceAdapter + Send + Sync>, String> {
    use crate::knowledge::SourceConfig;
    match source {
        SourceConfig::Stored {
            adapter,
            archive_path,
        } => {
            if adapter != "mediawiki_xml" {
                return Err(format!("unsupported stored adapter `{adapter}`"));
            }
            Ok(Box::new(MediaWikiXml::new(archive_path)))
        }
        SourceConfig::Linked { adapter, path, .. } => {
            if adapter != "markdown_dir" {
                return Err(format!("unsupported linked adapter `{adapter}`"));
            }
            Ok(Box::new(MarkdownDir::new(path)))
        }
        SourceConfig::Managed {} => {
            Err("managed buckets have no source adapter to build from".into())
        }
    }
}

fn build_chunker(config: &ChunkerConfig) -> Box<dyn Chunker + Send + Sync> {
    match config {
        ChunkerConfig::TokenBased {
            chunk_tokens,
            overlap_tokens,
        } => Box::new(TokenBasedChunker::new(*chunk_tokens, *overlap_tokens)),
    }
}
