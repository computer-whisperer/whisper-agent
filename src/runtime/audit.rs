//! Append-only JSONL audit log.
//!
//! One line per tool-call completion. Scope-widening grants and other
//! permission-changing events will land here as distinct entry shapes
//! when the `request_escalation` family is wired up — see
//! `docs/design_permissions_rework.md`.

use std::path::PathBuf;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::Serialize;
use serde_json::Value;
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

#[derive(Serialize, Debug)]
pub struct ToolCallEntry<'a> {
    pub timestamp: DateTime<Utc>,
    pub thread_id: &'a str,
    pub host_id: &'a str,
    pub tool_name: &'a str,
    pub args: Value,
    pub outcome: ToolCallOutcome<'a>,
}

#[derive(Serialize, Debug)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolCallOutcome<'a> {
    Ok { is_error: bool },
    Failed { message: &'a str },
}

#[derive(Clone)]
pub struct AuditLog {
    inner: Arc<Mutex<tokio::fs::File>>,
    path: PathBuf,
}

impl AuditLog {
    pub async fn open(path: PathBuf) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await?;
        Ok(Self {
            inner: Arc::new(Mutex::new(file)),
            path,
        })
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    pub async fn write(&self, entry: &ToolCallEntry<'_>) -> std::io::Result<()> {
        let mut line = serde_json::to_vec(entry)?;
        line.push(b'\n');
        let mut guard = self.inner.lock().await;
        guard.write_all(&line).await?;
        guard.flush().await?;
        Ok(())
    }
}
