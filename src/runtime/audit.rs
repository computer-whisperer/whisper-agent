//! Append-only JSONL audit log.
//!
//! Two entry shapes currently land here:
//!
//!   - `ToolCallEntry` — one line per tool-call completion.
//!   - `EscalationEntry` — one line per `request_escalation` resolution,
//!     covering every exit path (approve-success, approve-recheck-fail,
//!     reject, channel-dropped revocation). A scope widening is never
//!     invisible.
//!
//! Each record carries a top-level `"event"` tag so downstream
//! auditors can filter without key-sniffing.

use std::path::PathBuf;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::Serialize;
use serde_json::Value;
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

use whisper_agent_protocol::permission::{EscalationDecision, EscalationRequest};

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

/// Recorded for every terminal transition of a `request_escalation`
/// Function — the widening was applied, rejected, failed re-check, or
/// the escalation channel dropped with the request still pending.
/// `detail` carries the failure reason on `recheck_failed` and
/// `channel_dropped` paths (empty otherwise).
#[derive(Serialize, Debug)]
pub struct EscalationEntry<'a> {
    pub event: EscalationEventTag,
    pub timestamp: DateTime<Utc>,
    pub thread_id: &'a str,
    pub host_id: &'a str,
    pub request: &'a EscalationRequest,
    pub resolution: EscalationResolution<'a>,
}

#[derive(Serialize, Debug, Default, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum EscalationEventTag {
    #[default]
    Escalation,
}

/// Outcome of a single escalation request, as written to the audit log.
/// `Approved` is the only path that actually widens scope; the other
/// three are the "nothing widened, here's why" shapes.
#[derive(Serialize, Debug)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EscalationResolution<'a> {
    Approved,
    Rejected { reason: Option<&'a str> },
    RecheckFailed { detail: &'a str },
    ChannelDropped,
}

impl<'a> EscalationResolution<'a> {
    pub fn from_decision(decision: EscalationDecision, reason: Option<&'a str>) -> Self {
        match decision {
            EscalationDecision::Approve => Self::Approved,
            EscalationDecision::Reject => Self::Rejected { reason },
        }
    }
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

    pub async fn write_escalation(&self, entry: &EscalationEntry<'_>) -> std::io::Result<()> {
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
    use whisper_agent_protocol::permission::{EscalationRequest, PodModifyCap};

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
    fn escalation_entry_approved_serializes() {
        let req = EscalationRequest::AddTool {
            name: "write_file".into(),
        };
        let entry = EscalationEntry {
            event: EscalationEventTag::Escalation,
            timestamp: DateTime::parse_from_rfc3339("2026-04-21T00:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            thread_id: "t-1",
            host_id: "h",
            request: &req,
            resolution: EscalationResolution::Approved,
        };
        let v: serde_json::Value =
            serde_json::from_slice(&serde_json::to_vec(&entry).unwrap()).unwrap();
        assert_eq!(v["event"], "escalation");
        assert_eq!(v["request"]["variant"], "add_tool");
        assert_eq!(v["request"]["name"], "write_file");
        assert_eq!(v["resolution"]["kind"], "approved");
    }

    #[test]
    fn escalation_entry_rejected_carries_reason() {
        let req = EscalationRequest::RaisePodModify {
            target: PodModifyCap::ModifyAllow,
        };
        let reason = "not now".to_string();
        let entry = EscalationEntry {
            event: EscalationEventTag::Escalation,
            timestamp: DateTime::parse_from_rfc3339("2026-04-21T00:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            thread_id: "t-1",
            host_id: "h",
            request: &req,
            resolution: EscalationResolution::Rejected {
                reason: Some(&reason),
            },
        };
        let v: serde_json::Value =
            serde_json::from_slice(&serde_json::to_vec(&entry).unwrap()).unwrap();
        assert_eq!(v["resolution"]["kind"], "rejected");
        assert_eq!(v["resolution"]["reason"], "not now");
    }

    #[test]
    fn escalation_entry_recheck_failed_and_channel_dropped_serialize() {
        let req = EscalationRequest::AddTool {
            name: "exec".into(),
        };
        let entry_recheck = EscalationEntry {
            event: EscalationEventTag::Escalation,
            timestamp: Utc::now(),
            thread_id: "t-1",
            host_id: "h",
            request: &req,
            resolution: EscalationResolution::RecheckFailed {
                detail: "would exceed pod ceiling",
            },
        };
        let v: serde_json::Value =
            serde_json::from_slice(&serde_json::to_vec(&entry_recheck).unwrap()).unwrap();
        assert_eq!(v["resolution"]["kind"], "recheck_failed");
        assert_eq!(v["resolution"]["detail"], "would exceed pod ceiling");

        let entry_drop = EscalationEntry {
            event: EscalationEventTag::Escalation,
            timestamp: Utc::now(),
            thread_id: "t-1",
            host_id: "h",
            request: &req,
            resolution: EscalationResolution::ChannelDropped,
        };
        let v: serde_json::Value =
            serde_json::from_slice(&serde_json::to_vec(&entry_drop).unwrap()).unwrap();
        assert_eq!(v["resolution"]["kind"], "channel_dropped");
    }

    #[tokio::test]
    async fn audit_log_round_trips_tool_call_and_escalation_entries() {
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

        let req = EscalationRequest::AddTool {
            name: "exec".into(),
        };
        let esc = EscalationEntry {
            event: EscalationEventTag::Escalation,
            timestamp: Utc::now(),
            thread_id: "t-1",
            host_id: "h",
            request: &req,
            resolution: EscalationResolution::Approved,
        };
        log.write_escalation(&esc).await.unwrap();

        let body = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 2, "one line per entry; body was: {body}");
        let l0: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        let l1: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(l0["event"], "tool_call");
        assert_eq!(l1["event"], "escalation");
        assert_eq!(l1["resolution"]["kind"], "approved");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn from_decision_maps_approve_and_reject() {
        assert!(matches!(
            EscalationResolution::from_decision(EscalationDecision::Approve, None),
            EscalationResolution::Approved
        ));
        let reason = "nope".to_string();
        let r = EscalationResolution::from_decision(EscalationDecision::Reject, Some(&reason));
        match r {
            EscalationResolution::Rejected { reason: Some(r) } => assert_eq!(r, "nope"),
            other => panic!("expected Rejected with Some reason, got {other:?}"),
        }
    }
}
