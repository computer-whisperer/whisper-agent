//! Background-task types — identifier, lifecycle state, and the
//! summary entries returned in [`crate::Frame::BackgroundTaskList`].

use serde::{Deserialize, Serialize};

/// Background task identifier. Daemon-assigned, opaque to the
/// scheduler — the scheduler treats it as a token.
///
/// Daemons may pick whatever shape suits their bookkeeping (UUID,
/// counter, prefixed string, …). The scheduler only ever round-trips
/// it.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub struct BackgroundTaskId(pub String);

impl BackgroundTaskId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

impl std::fmt::Display for BackgroundTaskId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Lifecycle state of a background task. Transitions are
/// `Running → Exited | Killed`; once terminal, the task may still
/// appear in [`crate::Frame::BackgroundTaskList`] until the daemon
/// garbage-collects it.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BackgroundTaskState {
    /// Worker process is still running.
    Running,
    /// Worker process exited on its own. `code` is the OS exit code,
    /// `signal` the signal number if it was terminated by signal.
    /// Both fields are independently optional because not every host
    /// surfaces both reliably.
    Exited {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        code: Option<i32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signal: Option<i32>,
    },
    /// Daemon killed the task.
    Killed { reason: KilledReason },
}

/// Why the daemon killed a background task.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum KilledReason {
    /// Owning session was closed (either via
    /// [`crate::Frame::CloseSession`] or because the worker itself
    /// died).
    SessionClosed,
    /// Model invoked `bash_kill` (or equivalent tool).
    ExplicitKill,
    /// Daemon was shutting down.
    DaemonShutdown,
}

/// Summary entry returned in [`crate::Frame::BackgroundTaskList`]
/// (and structurally what the scheduler tracks per session). Carries
/// enough for the UI to render the task without separately querying.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct BackgroundTaskSummary {
    pub task_id: BackgroundTaskId,
    pub state: BackgroundTaskState,
    /// The original command string (or a daemon-chosen description if
    /// the tool wasn't a literal shell command).
    pub command: String,
    /// RFC 3339 timestamp.
    pub started_at: String,
    /// Cumulative bytes of buffered output the daemon has retained
    /// for this task. Reading them goes through the model-visible
    /// tool layer (e.g. `bash_check`), not the protocol.
    pub bytes_available: u64,
    /// RFC 3339 timestamp of the most recent output. Equal to
    /// [`Self::started_at`] if the task has produced no output yet.
    pub last_output_at: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip<T>(value: T)
    where
        T: Serialize + serde::de::DeserializeOwned + std::fmt::Debug + PartialEq,
    {
        let mut bytes = Vec::new();
        ciborium::into_writer(&value, &mut bytes).unwrap();
        let back: T = ciborium::from_reader(bytes.as_slice()).unwrap();
        assert_eq!(value, back);
    }

    #[test]
    fn background_task_id_round_trips() {
        round_trip(BackgroundTaskId::new("bg-42"));
    }

    #[test]
    fn state_running_round_trips() {
        round_trip(BackgroundTaskState::Running);
    }

    #[test]
    fn state_exited_with_code_round_trips() {
        round_trip(BackgroundTaskState::Exited {
            code: Some(0),
            signal: None,
        });
    }

    #[test]
    fn state_exited_with_signal_round_trips() {
        round_trip(BackgroundTaskState::Exited {
            code: None,
            signal: Some(15),
        });
    }

    #[test]
    fn state_exited_empty_round_trips() {
        // Both fields None should still survive — captures "we saw
        // the task end but couldn't pin down why."
        round_trip(BackgroundTaskState::Exited {
            code: None,
            signal: None,
        });
    }

    #[test]
    fn state_killed_round_trips_each_reason() {
        for reason in [
            KilledReason::SessionClosed,
            KilledReason::ExplicitKill,
            KilledReason::DaemonShutdown,
        ] {
            round_trip(BackgroundTaskState::Killed { reason });
        }
    }

    #[test]
    fn summary_round_trips() {
        round_trip(BackgroundTaskSummary {
            task_id: BackgroundTaskId::new("bg-1"),
            state: BackgroundTaskState::Running,
            command: "tail -f /var/log/syslog".into(),
            started_at: "2026-05-02T10:00:00Z".into(),
            bytes_available: 1024,
            last_output_at: "2026-05-02T10:01:30Z".into(),
        });
    }

    #[test]
    fn killed_reason_serializes_snake_case() {
        let mut bytes = Vec::new();
        ciborium::into_writer(&KilledReason::SessionClosed, &mut bytes).unwrap();
        let v: ciborium::Value = ciborium::from_reader(bytes.as_slice()).unwrap();
        assert_eq!(v.as_text(), Some("session_closed"));
    }
}
