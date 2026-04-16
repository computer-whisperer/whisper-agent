//! JSON-per-task persistence at `<state-dir>/<task_id>.json`.
//!
//! Writes happen synchronously on every scheduler-loop boundary for tasks that were
//! modified that iteration — see `Scheduler::flush_dirty`. The per-task-one-file shape
//! is deliberately naive; we'll upgrade to SQLite when cross-task queries become
//! useful or write volume makes one-file-per-task unwieldy (see `design_task_scheduler.md`).
//!
//! On startup, `load_all` scans the directory and any task that was in an in-flight
//! internal state at shutdown is transitioned to `Failed { at_phase: "resume" }` — we
//! have no way to cleanly pick up a half-finished HTTP call, so the user can restart
//! the turn by sending a fresh user message.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tokio::fs;
use tracing::{info, warn};

use crate::task::{Task, TaskInternalState};

pub struct Persister {
    dir: PathBuf,
}

impl Persister {
    pub async fn new(dir: PathBuf) -> Result<Self> {
        fs::create_dir_all(&dir)
            .await
            .with_context(|| format!("create state dir {}", dir.display()))?;
        Ok(Self { dir })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn task_path(&self, task_id: &str) -> PathBuf {
        self.dir.join(format!("{task_id}.json"))
    }

    /// Write task JSON. Overwrites any existing file.
    pub async fn flush(&self, task: &Task) -> Result<()> {
        let path = self.task_path(&task.id);
        let json = serde_json::to_vec_pretty(task)?;
        fs::write(&path, json)
            .await
            .with_context(|| format!("write {}", path.display()))?;
        Ok(())
    }

    /// Load every task from the state directory. Tasks in an in-flight internal state
    /// are transitioned to `Failed { at_phase: "resume" }` before being returned.
    pub async fn load_all(&self) -> Result<Vec<Task>> {
        let mut tasks = Vec::new();
        let mut entries = match fs::read_dir(&self.dir).await {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(tasks),
            Err(e) => return Err(e).with_context(|| format!("read_dir {}", self.dir.display())),
        };
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            match load_one(&path).await {
                Ok(task) => tasks.push(task),
                Err(e) => {
                    warn!(path = %path.display(), error = %e, "skipping unreadable task file")
                }
            }
        }
        info!(count = tasks.len(), "loaded persisted tasks");
        Ok(tasks)
    }
}

async fn load_one(path: &Path) -> Result<Task> {
    let bytes = fs::read(path)
        .await
        .with_context(|| format!("read {}", path.display()))?;
    let mut task: Task =
        serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
    if is_in_flight(&task.internal) {
        task.fail("resume", "task was in-flight at last shutdown");
    }
    Ok(task)
}

fn is_in_flight(state: &TaskInternalState) -> bool {
    matches!(
        state,
        TaskInternalState::NeedsMcpConnect
            | TaskInternalState::AwaitingMcpConnect { .. }
            | TaskInternalState::NeedsListTools
            | TaskInternalState::AwaitingListTools { .. }
            | TaskInternalState::NeedsModelCall
            | TaskInternalState::AwaitingModel { .. }
            | TaskInternalState::AwaitingTools { .. }
    )
}
