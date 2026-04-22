//! Append-only JSONL audit log.
//!
//! Two entry shapes land here today:
//!
//!   - `ToolCallEntry` — one line per tool-call completion.
//!   - `SudoDecisionEntry` — one line per user sudo decision
//!     (approve_once / approve_remember / reject). For approves, the
//!     entry is written after the inner tool completes so its
//!     outcome is part of the record.
//!
//! Each record carries a top-level `"event"` tag so downstream
//! auditors can filter by kind without key-sniffing.

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
    pub event: ToolCallEventTag,
    pub timestamp: DateTime<Utc>,
    pub thread_id: &'a str,
    pub host_id: &'a str,
    pub tool_name: &'a str,
    pub args: Value,
    pub outcome: ToolCallOutcome<'a>,
}

#[derive(Serialize, Debug, Default, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallEventTag {
    #[default]
    ToolCall,
}

#[derive(Serialize, Debug)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolCallOutcome<'a> {
    Ok { is_error: bool },
    Failed { message: &'a str },
}

/// One user sudo decision — written after approves complete their
/// inner call (so the record carries the outcome) or immediately on
/// reject. Inner tool name + args are captured so an auditor can see
/// exactly what was approved, without having to cross-reference the
/// tool-call stream.
#[derive(Serialize, Debug)]
pub struct SudoDecisionEntry<'a> {
    pub event: SudoDecisionEventTag,
    pub timestamp: DateTime<Utc>,
    pub thread_id: &'a str,
    pub host_id: &'a str,
    /// The wrapped tool's name (what the model asked sudo to run).
    pub tool_name: &'a str,
    /// The wrapped tool's args. JSON blob — structure depends on the
    /// inner tool.
    pub args: Value,
    /// Caller-supplied justification from the model's sudo call —
    /// "why does this need elevated permission." Always present.
    pub reason: &'a str,
    pub decision: SudoDecisionTag,
    /// User-typed text on reject (may be empty even when the decision
    /// is Reject). `None` for approves.
    pub reject_reason: Option<&'a str>,
    /// Inner tool's outcome for approves; `None` for rejects — no
    /// inner call ran.
    pub inner_outcome: Option<ToolCallOutcome<'a>>,
}

#[derive(Serialize, Debug, Default, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum SudoDecisionEventTag {
    #[default]
    SudoDecision,
}

#[derive(Serialize, Debug, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum SudoDecisionTag {
    ApproveOnce,
    ApproveRemember,
    Reject,
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
        self.write_line(serde_json::to_vec(entry)?).await
    }

    pub async fn write_sudo(&self, entry: &SudoDecisionEntry<'_>) -> std::io::Result<()> {
        self.write_line(serde_json::to_vec(entry)?).await
    }

    async fn write_line(&self, mut line: Vec<u8>) -> std::io::Result<()> {
        line.push(b'\n');
        let mut guard = self.inner.lock().await;
        guard.write_all(&line).await?;
        guard.flush().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_call_entry_serializes_with_event_tag() {
        let entry = ToolCallEntry {
            event: ToolCallEventTag::ToolCall,
            timestamp: DateTime::parse_from_rfc3339("2026-04-21T00:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            thread_id: "t-1",
            host_id: "h",
            tool_name: "write_file",
            args: serde_json::json!({"path": "/tmp/x"}),
            outcome: ToolCallOutcome::Ok { is_error: false },
        };
        let v: serde_json::Value =
            serde_json::from_slice(&serde_json::to_vec(&entry).unwrap()).unwrap();
        assert_eq!(v["event"], "tool_call");
        assert_eq!(v["tool_name"], "write_file");
        assert_eq!(v["outcome"]["kind"], "ok");
    }

    #[test]
    fn sudo_decision_entry_serializes_with_event_tag() {
        let entry = SudoDecisionEntry {
            event: SudoDecisionEventTag::SudoDecision,
            timestamp: DateTime::parse_from_rfc3339("2026-04-21T00:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            thread_id: "t-1",
            host_id: "h",
            tool_name: "pod_write_file",
            args: serde_json::json!({"filename": "pod.toml", "content": "..."}),
            reason: "need to update allowlist",
            decision: SudoDecisionTag::ApproveRemember,
            reject_reason: None,
            inner_outcome: Some(ToolCallOutcome::Ok { is_error: false }),
        };
        let v: serde_json::Value =
            serde_json::from_slice(&serde_json::to_vec(&entry).unwrap()).unwrap();
        assert_eq!(v["event"], "sudo_decision");
        assert_eq!(v["decision"], "approve_remember");
        assert_eq!(v["reason"], "need to update allowlist");
        assert_eq!(v["inner_outcome"]["kind"], "ok");
        assert!(v["reject_reason"].is_null());
    }

    #[test]
    fn sudo_decision_reject_entry_has_null_inner_outcome() {
        let entry = SudoDecisionEntry {
            event: SudoDecisionEventTag::SudoDecision,
            timestamp: Utc::now(),
            thread_id: "t-1",
            host_id: "h",
            tool_name: "bash",
            args: serde_json::json!({"cmd": "rm -rf /"}),
            reason: "cleanup",
            decision: SudoDecisionTag::Reject,
            reject_reason: Some("absolutely not"),
            inner_outcome: None,
        };
        let v: serde_json::Value =
            serde_json::from_slice(&serde_json::to_vec(&entry).unwrap()).unwrap();
        assert_eq!(v["decision"], "reject");
        assert_eq!(v["reject_reason"], "absolutely not");
        assert!(v["inner_outcome"].is_null());
    }

    #[tokio::test]
    async fn audit_log_round_trips_sudo_decision_entry() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "wa-sudo-audit-test-{}-{n}.jsonl",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let log = AuditLog::open(path.clone()).await.unwrap();

        let entry = SudoDecisionEntry {
            event: SudoDecisionEventTag::SudoDecision,
            timestamp: Utc::now(),
            thread_id: "t-1",
            host_id: "h",
            tool_name: "pod_write_file",
            args: serde_json::json!({"filename": "pod.toml"}),
            reason: "widening allowlist",
            decision: SudoDecisionTag::ApproveOnce,
            reject_reason: None,
            inner_outcome: Some(ToolCallOutcome::Ok { is_error: false }),
        };
        log.write_sudo(&entry).await.unwrap();

        let body = std::fs::read_to_string(&path).unwrap();
        let l0: serde_json::Value = serde_json::from_str(body.lines().next().unwrap()).unwrap();
        assert_eq!(l0["event"], "sudo_decision");
        assert_eq!(l0["decision"], "approve_once");
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn audit_log_round_trips_tool_call_entry() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("wa-audit-test-{}-{n}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let log = AuditLog::open(path.clone()).await.unwrap();

        let tc = ToolCallEntry {
            event: ToolCallEventTag::ToolCall,
            timestamp: Utc::now(),
            thread_id: "t-1",
            host_id: "h",
            tool_name: "read_file",
            args: serde_json::json!({"path": "/tmp/a"}),
            outcome: ToolCallOutcome::Ok { is_error: false },
        };
        log.write(&tc).await.unwrap();

        let body = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 1, "one line per entry; body was: {body}");
        let l0: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(l0["event"], "tool_call");
        let _ = std::fs::remove_file(&path);
    }
}
