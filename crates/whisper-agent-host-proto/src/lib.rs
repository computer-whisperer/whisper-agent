//! Host-env protocol — wire types for whisper-agent ↔ host-daemon.
//!
//! Defines the [`Frame`] enum sent in both directions over the
//! WebSocket the daemon opens at `/v1/host_env_link`. This crate ships
//! types only; the dialer (daemon side) and the server (scheduler
//! side) live in the binaries that consume it.
//!
//! See `docs/design_host_env_protocol.md` for the architecture.
//!
//! # Connection lifecycle
//!
//! 1. Daemon dials whisper-agent over HTTPS, presenting
//!    `Authorization: Bearer <daemon-token>` on the WebSocket
//!    upgrade. Whisper-agent matches the token against
//!    `[[auth.daemons]]` and resolves a daemon name.
//! 2. Daemon sends [`Frame::Hello`] as the first frame after upgrade.
//!    Carries [`DaemonCapabilities`] (advertised tool catalog,
//!    supported sandbox spec kinds, background-task support).
//! 3. Scheduler responds with [`Frame::Welcome`]. Both sides verify
//!    the other's [`PROTOCOL_VERSION`]; mismatch → [`Frame::Goodbye`]
//!    with [`GoodbyeReason::ProtocolMismatch`].
//! 4. Scheduler issues [`Frame::OpenSession`] / [`Frame::InvokeTool`]
//!    / etc. as threads bind to the daemon. Daemon emits
//!    [`Frame::SessionReady`] / [`Frame::ToolChunk`] / [`Frame::ToolFinal`]
//!    / [`Frame::BackgroundTaskUpdate`] in response.
//! 5. Disconnect (either side closes, or WS-level Pong timeout): the
//!    scheduler applies the per-thread
//!    [`crate::session::*`] disconnect policy to threads holding
//!    sessions on this daemon. Background tasks survive the drop on
//!    the daemon side; on reconnect, the scheduler issues
//!    [`Frame::ListBackgroundTasks`] per session to resync.
//!
//! # Wire format
//!
//! Frames are CBOR (via [`ciborium`]) inside WebSocket binary data
//! frames. Heartbeats use WS-level Ping/Pong (RFC 6455 §5.5.2/3) — no
//! application-layer heartbeat frame.

mod background;
mod call;
mod connection;
mod hook;
mod session;

pub use background::*;
pub use call::*;
pub use connection::*;
pub use hook::*;
pub use session::*;

// Re-exported for convenience: every consumer that opens a session
// also constructs / parses a `HostEnvSpec`.
pub use whisper_agent_protocol::HostEnvSpec;

use serde::{Deserialize, Serialize};

/// Wire-format version. Bumped on incompatible changes; daemon and
/// scheduler exchange in [`Frame::Hello`] / [`Frame::Welcome`] and
/// hard-fail on mismatch via [`GoodbyeReason::ProtocolMismatch`].
///
/// Bump rules:
/// - **Add a frame variant** → no bump (recipients ignore unknown
///   `kind` values; new senders must check `Welcome.protocol_version`
///   before relying on the new variant).
/// - **Add a field** with `#[serde(default)]` → no bump.
/// - **Remove or repurpose a field, change a tag, change semantics**
///   → bump.
pub const PROTOCOL_VERSION: u32 = 1;

/// Top-level frame sent in either direction over the WebSocket.
///
/// Direction is documented per variant. We keep a single union (not
/// separate `ToDaemon` / `ToScheduler` enums) because the asymmetry is
/// small and a single union is easier to log, route, and evolve.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Frame {
    // ─── Connection ────────────────────────────────────────────
    /// **D→S**, first frame after WS upgrade.
    Hello {
        daemon_version: String,
        protocol_version: u32,
        capabilities: DaemonCapabilities,
    },
    /// **S→D**, response to [`Frame::Hello`].
    Welcome {
        scheduler_version: String,
        protocol_version: u32,
    },
    /// Either direction. The sender will close the socket immediately
    /// after writing this frame.
    Goodbye {
        reason: GoodbyeReason,
        message: String,
    },

    // ─── Session lifecycle ────────────────────────────────────
    /// **S→D**. Scheduler asks the daemon to provision a sandbox for
    /// a thread. The daemon responds with either [`Frame::SessionReady`]
    /// or [`Frame::SessionFailed`].
    OpenSession {
        session_id: SessionId,
        thread_id: String,
        spec: HostEnvSpec,
        context: ThreadContext,
    },
    /// **S→D**. Apply a [`ThreadContextDelta`] to a live session. The
    /// daemon applies the delta before dispatching the next tool call;
    /// in-flight calls keep their pre-update context.
    UpdateSession {
        session_id: SessionId,
        context_delta: ThreadContextDelta,
    },
    /// **S→D**. Tear down the session. Daemon kills the worker, releases
    /// resources, terminates all owned background tasks, and replies
    /// with [`Frame::SessionClosed`].
    CloseSession { session_id: SessionId },

    /// **D→S**. Provision succeeded; session is live.
    SessionReady { session_id: SessionId },
    /// **D→S**. Provision failed at `phase`. The session id is final —
    /// the scheduler treats this session as dead and reprovisions
    /// (with a fresh id) if the thread should retry.
    SessionFailed {
        session_id: SessionId,
        phase: ProvisionPhase,
        message: String,
    },
    /// **D→S**. Session ended. Either acknowledges a [`Frame::CloseSession`]
    /// or surfaces an unsolicited end (worker crashed, daemon shutting
    /// down).
    SessionClosed {
        session_id: SessionId,
        reason: SessionEndReason,
    },

    // ─── Per-call ─────────────────────────────────────────────
    /// **S→D**. Invoke a tool inside a live session.
    InvokeTool {
        session_id: SessionId,
        call_id: CallId,
        tool_name: String,
        arguments: serde_json::Value,
    },
    /// **S→D**. Cancel an in-flight call. Daemon may take a moment to
    /// produce the terminal [`Frame::ToolFinal`] — the scheduler treats
    /// the eventual `ToolFinal` as authoritative.
    CancelCall {
        session_id: SessionId,
        call_id: CallId,
    },

    /// **D→S**. Streaming output, zero or more per call. Only emitted
    /// by tools that stream (e.g. `bash`); one-shot tools yield only
    /// [`Frame::ToolFinal`].
    ToolChunk {
        session_id: SessionId,
        call_id: CallId,
        block: ContentBlock,
    },
    /// **D→S**. Terminal frame for a call, exactly one per `call_id`.
    ToolFinal {
        session_id: SessionId,
        call_id: CallId,
        result: CallToolResult,
    },

    // ─── Background tasks ─────────────────────────────────────
    /// **D→S**. State change or new output available on a background
    /// task. Rate-limited (not one frame per stdout byte; coalesced
    /// to ~500 ms windows by the daemon). The actual output bytes
    /// flow through model-visible tool calls (e.g. `bash_check`),
    /// not through this frame.
    BackgroundTaskUpdate {
        session_id: SessionId,
        task_id: BackgroundTaskId,
        state: BackgroundTaskState,
        bytes_available: u64,
        /// RFC 3339 timestamp of the most recent output (or task
        /// start, if none yet).
        last_output_at: String,
    },
    /// **S→D**. Resync request. Used on reconnect to re-fetch the
    /// daemon's view of the session's background tasks.
    ListBackgroundTasks { session_id: SessionId },
    /// **D→S**. Response to [`Frame::ListBackgroundTasks`].
    BackgroundTaskList {
        session_id: SessionId,
        tasks: Vec<BackgroundTaskSummary>,
    },

    // ─── Future extension point ──────────────────────────────
    /// **D→S**. Out-of-band event from the worker — audit, behavior
    /// hook, resource warning. Schema is intentionally open in v1;
    /// consumers ignore unknown [`HookKind`]s.
    ///
    /// Field name is `hook_kind` rather than `kind` because the
    /// outer enum reserves `kind` for the variant discriminator.
    HookEvent {
        session_id: SessionId,
        hook_kind: HookKind,
        payload: serde_json::Value,
    },
}

/// Errors from CBOR encoding / decoding helpers on [`Frame`].
#[derive(Debug)]
pub enum CodecError {
    Encode(ciborium::ser::Error<std::io::Error>),
    Decode(ciborium::de::Error<std::io::Error>),
}

impl std::fmt::Display for CodecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodecError::Encode(e) => write!(f, "encode: {e}"),
            CodecError::Decode(e) => write!(f, "decode: {e}"),
        }
    }
}

impl std::error::Error for CodecError {}

impl Frame {
    /// Encode this frame as CBOR bytes.
    pub fn encode_cbor(&self) -> Result<Vec<u8>, CodecError> {
        let mut buf = Vec::new();
        ciborium::into_writer(self, &mut buf).map_err(CodecError::Encode)?;
        Ok(buf)
    }

    /// Decode a frame from CBOR bytes.
    pub fn decode_cbor(bytes: &[u8]) -> Result<Self, CodecError> {
        ciborium::from_reader(bytes).map_err(CodecError::Decode)
    }
}

#[cfg(test)]
mod tests {
    //! Round-trip every Frame variant through CBOR. The point is to
    //! catch serde shape errors (variant tag, field rename, missing
    //! `#[serde(default)]` on optional fields) before any wire code
    //! is written against this crate.
    //!
    //! Per-module tests in `connection`/`session`/`call`/`background`
    //! cover the supporting types in more depth (default elision,
    //! `Option<Option<T>>` semantics in deltas, etc.); these tests
    //! verify the types compose correctly at the [`Frame`] envelope.

    use super::*;
    use std::collections::{BTreeMap, BTreeSet};
    use whisper_agent_protocol::sandbox::{NetworkPolicy, PathAccess};

    fn assert_round_trip(frame: Frame) {
        let bytes = frame.encode_cbor().expect("encode");
        let back = Frame::decode_cbor(&bytes).expect("decode");
        assert_eq!(frame, back);
    }

    fn sample_session_id() -> SessionId {
        SessionId::new("01HXYZ-test-session")
    }

    #[test]
    fn hello_round_trips() {
        assert_round_trip(Frame::Hello {
            daemon_version: "0.3.13".into(),
            protocol_version: PROTOCOL_VERSION,
            capabilities: DaemonCapabilities {
                tools: vec![ToolDescriptor {
                    name: "bash".into(),
                    description: "run a shell command".into(),
                    input_schema: serde_json::json!({"type":"object"}),
                    annotations: ToolAnnotations::default(),
                }],
                spec_kinds: vec![HostEnvSpecKind::Landlock],
                max_concurrent_sessions: Some(8),
                supports_background_tasks: true,
            },
        });
    }

    #[test]
    fn welcome_round_trips() {
        assert_round_trip(Frame::Welcome {
            scheduler_version: "0.3.13".into(),
            protocol_version: PROTOCOL_VERSION,
        });
    }

    #[test]
    fn goodbye_round_trips_each_reason() {
        for reason in [
            GoodbyeReason::ProtocolMismatch,
            GoodbyeReason::Unauthorized,
            GoodbyeReason::NameAlreadyConnected,
            GoodbyeReason::ServerShutdown,
            GoodbyeReason::DaemonShutdown,
            GoodbyeReason::Other,
        ] {
            assert_round_trip(Frame::Goodbye {
                reason,
                message: "see you".into(),
            });
        }
    }

    #[test]
    fn open_session_round_trips_landlock() {
        assert_round_trip(Frame::OpenSession {
            session_id: sample_session_id(),
            thread_id: "thread-abc".into(),
            spec: HostEnvSpec::Landlock {
                allowed_paths: vec![
                    PathAccess::read_write("/home/me/project"),
                    PathAccess::read_only("/usr"),
                ],
                network: NetworkPolicy::Isolated,
            },
            context: ThreadContext {
                env: BTreeMap::from([("RUST_LOG".into(), "info".into())]),
                runas: Some("worker".into()),
                tool_denylist: BTreeSet::from(["view_pdf".into()]),
                ..ThreadContext::default()
            },
        });
    }

    #[test]
    fn open_session_round_trips_container() {
        assert_round_trip(Frame::OpenSession {
            session_id: sample_session_id(),
            thread_id: "thread-abc".into(),
            spec: HostEnvSpec::Container {
                image: "ghcr.io/whisper-agent/dev:latest".into(),
                mounts: vec![],
                network: NetworkPolicy::Unrestricted,
                limits: None,
                env: BTreeMap::new(),
            },
            context: ThreadContext::default(),
        });
    }

    #[test]
    fn update_session_round_trips() {
        assert_round_trip(Frame::UpdateSession {
            session_id: sample_session_id(),
            context_delta: ThreadContextDelta {
                env_set: BTreeMap::from([("EXTRA".into(), "1".into())]),
                runas: Some(None), // explicit clear
                ..ThreadContextDelta::default()
            },
        });
    }

    #[test]
    fn close_session_round_trips() {
        assert_round_trip(Frame::CloseSession {
            session_id: sample_session_id(),
        });
    }

    #[test]
    fn session_ready_round_trips() {
        assert_round_trip(Frame::SessionReady {
            session_id: sample_session_id(),
        });
    }

    #[test]
    fn session_failed_round_trips_each_phase() {
        for phase in [
            ProvisionPhase::PreExec,
            ProvisionPhase::Landlock,
            ProvisionPhase::WorkerSpawn,
            ProvisionPhase::WorkerHandshake,
        ] {
            assert_round_trip(Frame::SessionFailed {
                session_id: sample_session_id(),
                phase,
                message: "boom".into(),
            });
        }
    }

    #[test]
    fn session_closed_round_trips_each_reason() {
        for reason in [
            SessionEndReason::RequestedByScheduler,
            SessionEndReason::WorkerExited { code: Some(137) },
            SessionEndReason::WorkerExited { code: None },
            SessionEndReason::DaemonShutdown,
        ] {
            assert_round_trip(Frame::SessionClosed {
                session_id: sample_session_id(),
                reason,
            });
        }
    }

    #[test]
    fn invoke_tool_round_trips() {
        assert_round_trip(Frame::InvokeTool {
            session_id: sample_session_id(),
            call_id: CallId(42),
            tool_name: "read_file".into(),
            arguments: serde_json::json!({ "path": "/tmp/x", "max_lines": 50 }),
        });
    }

    #[test]
    fn cancel_call_round_trips() {
        assert_round_trip(Frame::CancelCall {
            session_id: sample_session_id(),
            call_id: CallId(42),
        });
    }

    #[test]
    fn tool_chunk_round_trips() {
        assert_round_trip(Frame::ToolChunk {
            session_id: sample_session_id(),
            call_id: CallId(42),
            block: ContentBlock::Text {
                text: "partial output\n".into(),
            },
        });
    }

    #[test]
    fn tool_final_round_trips() {
        assert_round_trip(Frame::ToolFinal {
            session_id: sample_session_id(),
            call_id: CallId(42),
            result: CallToolResult {
                content: vec![ContentBlock::Text {
                    text: "done".into(),
                }],
                is_error: None,
            },
        });
    }

    #[test]
    fn background_task_update_round_trips() {
        assert_round_trip(Frame::BackgroundTaskUpdate {
            session_id: sample_session_id(),
            task_id: BackgroundTaskId::new("bg-1"),
            state: BackgroundTaskState::Running,
            bytes_available: 1024,
            last_output_at: "2026-05-02T10:00:00Z".into(),
        });
    }

    #[test]
    fn list_background_tasks_round_trips() {
        assert_round_trip(Frame::ListBackgroundTasks {
            session_id: sample_session_id(),
        });
    }

    #[test]
    fn background_task_list_round_trips() {
        assert_round_trip(Frame::BackgroundTaskList {
            session_id: sample_session_id(),
            tasks: vec![BackgroundTaskSummary {
                task_id: BackgroundTaskId::new("bg-1"),
                state: BackgroundTaskState::Exited {
                    code: Some(0),
                    signal: None,
                },
                command: "sleep 10".into(),
                started_at: "2026-05-02T10:00:00Z".into(),
                bytes_available: 0,
                last_output_at: "2026-05-02T10:00:00Z".into(),
            }],
        });
    }

    #[test]
    fn hook_event_round_trips() {
        assert_round_trip(Frame::HookEvent {
            session_id: sample_session_id(),
            hook_kind: HookKind::new("tool_started"),
            payload: serde_json::json!({"tool": "bash"}),
        });
    }

    #[test]
    fn cbor_carries_frame_kind_discriminator() {
        // Sanity-check that internal-tag serialization is actually
        // happening: encode a frame, decode as a generic CBOR value,
        // confirm `kind` is present at the top level. If we ever
        // accidentally drop the `#[serde(tag = "kind")]` attribute
        // this test will catch it before any wire code does.
        let frame = Frame::SessionReady {
            session_id: sample_session_id(),
        };
        let bytes = frame.encode_cbor().unwrap();
        let value: ciborium::Value = ciborium::from_reader(bytes.as_slice()).unwrap();
        let map = value.as_map().expect("frame must encode as a CBOR map");
        let kind = map
            .iter()
            .find(|(k, _)| k.as_text() == Some("kind"))
            .expect("frame must carry a `kind` discriminator")
            .1
            .as_text()
            .expect("`kind` must be a text string");
        assert_eq!(kind, "session_ready");
    }
}
