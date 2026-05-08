//! Live config-file watcher. Notifies the daemon when `whisper-agent.toml`
//! changes so credential rotations and backend swaps land without a restart.
//!
//! We watch the *parent directory* (non-recursively) rather than the file
//! itself: many editors save via `tmpfile + rename`, which destroys the
//! inotify watch on the original inode. Filtering by basename in the
//! callback gives us reliable coverage of both in-place edits and atomic
//! saves.
//!
//! Notify fires multiple events per save (e.g. CREATE + MODIFY + CHMOD).
//! We debounce 300ms before reloading so a burst settles into one reload.
//! Reload errors are logged and the previous resolved config stays live.
//!
//! IN_ACCESS events are filtered out: our own re-read of the config inside
//! the reload path triggers them on real filesystems, which would close a
//! self-feedback loop (read → access event → debounce → read → …).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use arc_swap::ArcSwap;
use notify::{Event, EventKind, RecursiveMode, Watcher, recommended_watcher};
use tokio::sync::mpsc;
use tracing::{error, info, warn};

use crate::config::{self, Resolved};

const DEBOUNCE: Duration = Duration::from_millis(300);

/// Watch `config_path` for changes; on each settled change, re-resolve the
/// named backend and swap it into `resolved`. Runs forever; returns only if
/// the watcher itself shuts down (parent dir vanished, channel closed, etc.).
pub async fn watch(
    config_path: PathBuf,
    backend_name: String,
    resolved: Arc<ArcSwap<Resolved>>,
) -> Result<()> {
    let parent = config_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let target_basename = config_path
        .file_name()
        .context("config path has no file name component")?
        .to_os_string();

    let (tx, mut rx) = mpsc::unbounded_channel::<()>();
    let target_basename_cb = target_basename.clone();
    let mut watcher = recommended_watcher(move |res: notify::Result<Event>| match res {
        Ok(event) => {
            // Skip pure-read events. Without this, our own re-read of the
            // config inside the reload path fires inotify IN_ACCESS, which
            // closes a feedback loop with the debounce timer firing every
            // ~300ms forever.
            if matches!(event.kind, EventKind::Access(_)) {
                return;
            }
            let touched = event
                .paths
                .iter()
                .any(|p| p.file_name() == Some(target_basename_cb.as_os_str()));
            if touched {
                let _ = tx.send(());
            }
        }
        Err(e) => warn!(error = %e, "config watcher error"),
    })
    .context("create file watcher")?;
    watcher
        .watch(&parent, RecursiveMode::NonRecursive)
        .with_context(|| format!("watch {}", parent.display()))?;
    info!(
        path = %config_path.display(),
        watched_dir = %parent.display(),
        "config watcher armed"
    );

    loop {
        // Block until the first event of a burst.
        if rx.recv().await.is_none() {
            return Ok(());
        }
        // Drain any further events that arrive within the debounce window.
        loop {
            tokio::select! {
                _ = tokio::time::sleep(DEBOUNCE) => break,
                msg = rx.recv() => {
                    if msg.is_none() {
                        return Ok(());
                    }
                    // keep draining
                }
            }
        }
        match config::resolve(&config_path, &backend_name) {
            Ok(new) => {
                let auth_mode = match &new.auth {
                    whisper_agent_auth::ClientAuth::ApiKey(_) => "api_key",
                    whisper_agent_auth::ClientAuth::Codex(_) => "chatgpt_subscription",
                };
                info!(
                    auth_mode,
                    api_base = %new.api_base,
                    image_model = %new.image_model,
                    chat_model = %new.chat_model,
                    "config reloaded"
                );
                resolved.store(Arc::new(new));
            }
            Err(e) => {
                error!(error = %e, "config reload failed; keeping previous resolved config");
            }
        }
    }
}
