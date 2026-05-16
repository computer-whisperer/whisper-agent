//! Daemon-side credential publisher.
//!
//! For each `[[publish_credential]]` entry in the daemon's TOML, the
//! daemon spawns one [`publish_task`]. The task owns the on-disk
//! credential file: it (a) refreshes the access token before expiry
//! using the OAuth refresh-token flow, persisting rotated tokens
//! back to the file; (b) watches the parent directory via inotify
//! and reloads on external rewrites (e.g. the user re-running
//! `codex login`); and (c) after every change ships the on-disk
//! contents to the scheduler as a
//! [`Frame::PublishCredential`].
//!
//! The publisher is per-connection: it lives only while a host-env
//! link is up, and re-spawns on reconnect. State is reloaded from
//! disk on each spawn so a daemon restart and a reconnect look the
//! same to the scheduler. Eager-on-connect: the first frame the
//! task sends is the *current* on-disk credential, not a refreshed
//! one — that bounds how stale the scheduler's view can be after
//! either side restarts.
//!
//! Why parent-dir watching: Codex (and most credential writers)
//! save atomically by writing a tempfile and renaming over the
//! target. Watching the file by inode loses the watch on the
//! original inode at rename time. Watching the parent dir and
//! filtering by basename in the event callback covers both
//! in-place edits and atomic-rename saves. Mirrors the imagegen
//! daemon's config watcher.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use chrono::Utc;
use notify::{Event, EventKind, RecursiveMode, Watcher, recommended_watcher};
use reqwest::Client;
use tokio::sync::Mutex;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};
use whisper_agent_auth::CodexAuth;
use whisper_agent_host_proto::{CredentialPayload, Frame};

use crate::config::{PublishCredentialConfig, PublishCredentialKind};

/// How long before the access token's `exp` the publisher attempts a
/// refresh. The scheduler's per-request safety margin is 60 s; we
/// refresh well ahead of that so a rotated token has time to flow
/// daemon → scheduler → next request without ever bumping into the
/// JIT-refresh path.
const REFRESH_LEAD: Duration = Duration::from_secs(300);

/// Fallback wait when the next-refresh time can't be computed (JWT's
/// `exp` won't parse, exp is in the past and refresh keeps failing,
/// etc.). Used both as the retry backoff after a failed refresh and
/// as the minimum sleep duration in the publisher loop — without a
/// floor, an unparseable / past-exp credential would tight-loop the
/// task on `tokio::time::sleep(Duration::ZERO)`.
const REFRESH_RETRY_FALLBACK: Duration = Duration::from_secs(60);

/// Debounce window for filesystem events. Editors and credential
/// writers fire bursts (CREATE + MODIFY + CHMOD) per save; collapse
/// them to one reload.
const WATCH_DEBOUNCE: Duration = Duration::from_millis(300);

/// Spawn one publisher per config entry. Returns immediately; the
/// returned [`PublisherHandles`] can be dropped once the connection
/// is torn down to stop every task at the next await point.
pub fn spawn_all(
    entries: &[PublishCredentialConfig],
    outbound: mpsc::Sender<Frame>,
    http: Client,
) -> PublisherHandles {
    let mut handles = Vec::new();
    for entry in entries {
        let cfg = entry.clone();
        let outbound = outbound.clone();
        let http = http.clone();
        let backend = cfg.backend.clone();
        let join = tokio::spawn(async move {
            if let Err(e) = publish_task(cfg, outbound, http).await {
                error!(
                    backend = %backend,
                    error = %e,
                    "credential publisher exited with error",
                );
            } else {
                info!(backend = %backend, "credential publisher exited cleanly");
            }
        });
        handles.push(join);
    }
    PublisherHandles { joins: handles }
}

/// Owns the spawned task `JoinHandle`s. Drop aborts them — used by
/// the connection event loop to tear down publishers when the link
/// closes. Tasks also exit naturally when the outbound channel is
/// closed, so abort is just a fast-path.
#[derive(Default)]
pub struct PublisherHandles {
    joins: Vec<tokio::task::JoinHandle<()>>,
}

impl Drop for PublisherHandles {
    fn drop(&mut self) {
        for j in &self.joins {
            j.abort();
        }
    }
}

async fn publish_task(
    cfg: PublishCredentialConfig,
    outbound: mpsc::Sender<Frame>,
    http: Client,
) -> Result<()> {
    match cfg.kind {
        PublishCredentialKind::Codex => {
            publish_codex(&cfg.backend, &cfg.path, outbound, http).await
        }
    }
}

async fn publish_codex(
    backend: &str,
    path: &Path,
    outbound: mpsc::Sender<Frame>,
    http: Client,
) -> Result<()> {
    // One async read serves both the in-memory parse (state) and the
    // eager-on-connect publish (bytes). The earlier version did this
    // in two passes — `CodexAuth::load` (sync) plus a re-read inside
    // `publish_current` — which is just wasted I/O on startup.
    let initial = tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("initial read of {}", path.display()))?;
    let auth = CodexAuth::from_contents(path.to_path_buf(), &initial)
        .with_context(|| format!("initial parse of {}", path.display()))?;
    let state = Arc::new(Mutex::new(auth));

    // Eager-on-connect: ship the just-read bytes so the scheduler
    // sees the latest tokens after either side restarted, even
    // before any refresh runs.
    publish_bytes(backend, initial, &outbound).await?;

    // File watcher channel — small bound, the consumer drains
    // immediately and we debounce inside the consumer.
    let (fs_tx, mut fs_rx) = mpsc::unbounded_channel::<()>();
    let _watcher = arm_watcher(path, fs_tx).context("arm credential watcher")?;

    info!(backend, path = %path.display(), "credential publisher armed");

    loop {
        // Compute the next wake from the *current* token's expiry.
        // Reload after each refresh / external rewrite, since both
        // mutate `exp`.
        let sleep_dur = {
            let guard = state.lock().await;
            let lead = chrono::Duration::from_std(REFRESH_LEAD).unwrap();
            guard
                .next_refresh_at(lead)
                .and_then(|at| (at - Utc::now()).to_std().ok())
                .unwrap_or(REFRESH_RETRY_FALLBACK)
                .max(REFRESH_RETRY_FALLBACK)
        };
        debug!(
            backend,
            sleep_secs = sleep_dur.as_secs(),
            "next refresh wake"
        );

        tokio::select! {
            biased;
            // Channel closed by the connection event loop → time to
            // exit. Drop the watcher with us.
            _ = outbound.closed() => {
                debug!(backend, "outbound channel closed — publisher exiting");
                return Ok(());
            }
            // External file change → reload + publish. Debounce a
            // burst (one save fires several notify events).
            Some(()) = fs_rx.recv() => {
                drain_debounce(&mut fs_rx).await;
                match reload(path, &state).await {
                    Ok(true) => {
                        info!(backend, path = %path.display(), "credential file rewritten externally — republishing");
                        if let Err(e) = publish_current(backend, path, &outbound).await {
                            warn!(backend, error = %e, "publish after external rewrite failed");
                        }
                    }
                    Ok(false) => {
                        // Spurious event (we wrote the file ourselves
                        // during refresh, and it took notify a beat
                        // to surface it). No-op.
                    }
                    Err(e) => {
                        warn!(backend, error = %e, "reloading credential file failed; will retry on next change");
                    }
                }
            }
            // Refresh timer fired → call ensure_fresh, persist, publish.
            () = tokio::time::sleep(sleep_dur) => {
                match refresh_and_persist(&state, &http).await {
                    Ok(true) => {
                        info!(backend, "refreshed access token, publishing to scheduler");
                        if let Err(e) = publish_current(backend, path, &outbound).await {
                            warn!(backend, error = %e, "publish after refresh failed");
                        }
                    }
                    Ok(false) => {
                        // Was already fresh — somehow we woke early. Log and continue.
                        debug!(backend, "refresh timer fired but token still fresh");
                    }
                    Err(e) => {
                        warn!(
                            backend,
                            error = %e,
                            retry_in_secs = REFRESH_RETRY_FALLBACK.as_secs(),
                            "refresh failed; will retry"
                        );
                        tokio::time::sleep(REFRESH_RETRY_FALLBACK).await;
                    }
                }
            }
        }
    }
}

/// Inotify-watch the parent dir, filter events by basename. Returns
/// the watcher itself — drop it to stop watching.
fn arm_watcher(path: &Path, tx: mpsc::UnboundedSender<()>) -> Result<notify::RecommendedWatcher> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let target = path
        .file_name()
        .ok_or_else(|| anyhow!("credential path has no file name"))?
        .to_os_string();
    let mut watcher = recommended_watcher(move |res: notify::Result<Event>| match res {
        Ok(event) => {
            // Skip pure reads — our own persist() during refresh
            // fires IN_ACCESS on real filesystems, which would close
            // a feedback loop through the debounce timer.
            if matches!(event.kind, EventKind::Access(_)) {
                return;
            }
            if event
                .paths
                .iter()
                .any(|p| p.file_name() == Some(target.as_os_str()))
            {
                let _ = tx.send(());
            }
        }
        Err(e) => warn!(error = %e, "credential watcher error"),
    })
    .context("create credential watcher")?;
    watcher
        .watch(&parent, RecursiveMode::NonRecursive)
        .with_context(|| format!("watch {}", parent.display()))?;
    Ok(watcher)
}

/// Drain pending events arriving within [`WATCH_DEBOUNCE`] of the
/// first one — collapses a save burst into one reload.
async fn drain_debounce(rx: &mut mpsc::UnboundedReceiver<()>) {
    let deadline = tokio::time::Instant::now() + WATCH_DEBOUNCE;
    loop {
        tokio::select! {
            biased;
            _ = tokio::time::sleep_until(deadline) => return,
            maybe = rx.recv() => {
                if maybe.is_none() {
                    return;
                }
            }
        }
    }
}

/// `Ok(true)` if the file's access token differs from the cached
/// state — i.e. an external rewrite landed and we should republish.
async fn reload(path: &Path, state: &Arc<Mutex<CodexAuth>>) -> Result<bool> {
    let fresh = CodexAuth::load(Some(path.to_path_buf()))
        .with_context(|| format!("reload {}", path.display()))?;
    let mut guard = state.lock().await;
    let changed = guard.access_token() != fresh.access_token();
    *guard = fresh;
    Ok(changed)
}

/// `Ok(true)` if `ensure_fresh` produced a new access token,
/// `Ok(false)` if the existing one was still inside its lifetime.
async fn refresh_and_persist(state: &Arc<Mutex<CodexAuth>>, http: &Client) -> Result<bool> {
    let mut guard = state.lock().await;
    let before = guard.access_token().to_string();
    guard.ensure_fresh(http).await.context("ensure_fresh")?;
    Ok(guard.access_token() != before)
}

/// Ship the literal on-disk bytes (rather than re-serializing the
/// in-memory state) so any extra fields Codex writes that the
/// `AuthDotJson` schema doesn't model survive the round-trip.
async fn publish_current(backend: &str, path: &Path, outbound: &mpsc::Sender<Frame>) -> Result<()> {
    let contents = tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("read for publish: {}", path.display()))?;
    publish_bytes(backend, contents, outbound).await
}

async fn publish_bytes(
    backend: &str,
    contents: String,
    outbound: &mpsc::Sender<Frame>,
) -> Result<()> {
    outbound
        .send(Frame::PublishCredential {
            backend: backend.to_string(),
            payload: CredentialPayload::Codex { contents },
        })
        .await
        .map_err(|_| anyhow!("outbound channel closed"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// rustls 0.23 panics on the first TLS use unless a provider is
    /// installed as the process default. The binary does this in
    /// `main`; tests have to install it themselves before any
    /// `reqwest::Client::new()`. Idempotent — `install_default`
    /// returns `Err` if someone else already installed, which is
    /// fine in the test runner where many tests share a process.
    fn install_crypto_provider() {
        let _ = rustls::crypto::ring::default_provider().install_default();
    }

    /// Build a fake unsigned JWT carrying `exp = now + secs`. The
    /// daemon-side refresher only cares about the `exp` claim — no
    /// signature verification — so this is enough to drive the
    /// publish loop.
    fn jwt_with_exp(secs_from_now: i64) -> String {
        use base64::Engine;
        let exp = Utc::now().timestamp() + secs_from_now;
        let payload = serde_json::json!({ "exp": exp });
        let h = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b"{}");
        let p = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_string(&payload).unwrap().as_bytes());
        let s = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b"sig");
        format!("{h}.{p}.{s}")
    }

    fn write_codex_auth(path: &Path, access_token: &str, refresh_token: &str, exp_secs: i64) {
        let blob = serde_json::json!({
            "tokens": {
                "id_token": jwt_with_exp(exp_secs),
                "access_token": jwt_with_exp(exp_secs),
                "refresh_token": refresh_token,
            },
            "_test_marker": access_token,
        });
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(blob.to_string().as_bytes()).unwrap();
    }

    #[tokio::test]
    async fn eager_publish_on_spawn() {
        // The publisher should fire one PublishCredential immediately
        // after starting, so the scheduler sees the current on-disk
        // state without waiting for a refresh tick.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("auth.json");
        write_codex_auth(&path, "first", "rt-1", 3_600);

        let (tx, mut rx) = mpsc::channel::<Frame>(4);
        install_crypto_provider();
        let http = reqwest::Client::new();
        let entry = PublishCredentialConfig {
            backend: "openai-sub".into(),
            kind: PublishCredentialKind::Codex,
            path: path.clone(),
        };
        let _handles = spawn_all(std::slice::from_ref(&entry), tx, http);

        let frame = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
            .await
            .expect("publisher should send eager publish")
            .expect("channel must yield a frame");
        match frame {
            Frame::PublishCredential { backend, payload } => {
                assert_eq!(backend, "openai-sub");
                match payload {
                    CredentialPayload::Codex { contents } => {
                        assert!(contents.contains("\"_test_marker\":\"first\""));
                    }
                }
            }
            other => panic!("expected PublishCredential, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn external_rewrite_triggers_republish() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("auth.json");
        write_codex_auth(&path, "before", "rt", 3_600);

        let (tx, mut rx) = mpsc::channel::<Frame>(4);
        install_crypto_provider();
        let http = reqwest::Client::new();
        let entry = PublishCredentialConfig {
            backend: "openai-sub".into(),
            kind: PublishCredentialKind::Codex,
            path: path.clone(),
        };
        let _handles = spawn_all(std::slice::from_ref(&entry), tx, http);

        // Eat the eager-on-spawn publish.
        let _ = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
            .await
            .expect("eager publish");

        // Rewrite with a *different* exp so the embedded access-token
        // JWT differs. `reload` compares access tokens to filter out
        // spurious notify events for writes we did ourselves; a
        // same-second second write to the same exp_secs value would
        // produce a byte-identical JWT and `reload` would (correctly)
        // report no change. Short sleep ensures the parent-dir
        // watcher has armed before we touch the file.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        write_codex_auth(&path, "after", "rt", 7_200);

        let frame = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
            .await
            .expect("rewrite should trigger republish")
            .expect("channel must yield a frame");
        match frame {
            Frame::PublishCredential { payload, .. } => match payload {
                CredentialPayload::Codex { contents } => {
                    assert!(
                        contents.contains("\"_test_marker\":\"after\""),
                        "expected updated contents, got: {contents}"
                    );
                }
            },
            other => panic!("expected PublishCredential, got {other:?}"),
        }
    }

    #[test]
    fn drop_aborts_spawned_tasks_smoke() {
        // Empty publisher list — drop is trivially fine, no panics.
        let handles = PublisherHandles::default();
        drop(handles);
    }
}
