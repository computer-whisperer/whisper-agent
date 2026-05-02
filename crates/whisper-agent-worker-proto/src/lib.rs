//! Daemon ↔ worker IPC wire types.
//!
//! The daemon (`whisper-agent-host-daemon`) spawns the worker
//! (`whisper-agent-mcp-host`, soon-to-be-renamed) per session inside a
//! landlock sandbox. The two communicate over a Unix `socketpair` —
//! the daemon hands one end to the worker as FD 3 at spawn, keeps the
//! other. Frames are length-prefixed CBOR (4-byte big-endian length
//! header, then the encoded [`WorkerFrame`]).
//!
//! # Why a separate proto crate
//!
//! The daemon ↔ worker hop used to be HTTP + JSON-RPC `tools/call`
//! (the v1 MCP machinery the worker was originally built for). That
//! wire bought us an interop story we never used — no external clients
//! ever pointed at our workers — at the cost of HTTP+SSE+bearer
//! plumbing on a strictly daemon-private subprocess. Replacing it with
//! a dedicated CBOR socket lets us add bidirectional features
//! (background-task event push, in-flight cancellation) cleanly,
//! without bending an HTTP-shaped wire to do them.
//!
//! # Layering
//!
//! `whisper-agent-host-proto` defines the scheduler ↔ daemon protocol.
//! Several supporting types ([`CallId`], [`ToolDescriptor`],
//! [`ContentBlock`], [`CallToolResult`]) appear on both wires
//! unchanged — this crate re-exports them rather than duplicating, so
//! a `ContentBlock` from a worker's `ToolChunk` can flow through the
//! daemon into a scheduler-bound [`whisper_agent_host_proto::Frame::ToolChunk`]
//! without a copy.
//!
//! # Connection lifecycle
//!
//! 1. Daemon creates a `socketpair(AF_UNIX, SOCK_STREAM)`, dups one
//!    end to FD 3 in the child, spawns
//!    `whisper-agent-mcp-host --workspace-root <path>`.
//! 2. Worker reads FD 3, sends [`WorkerFrame::Hello`] as the first
//!    frame. Carries the worker version, [`PROTOCOL_VERSION`], and
//!    the tool catalog (probed once at startup).
//! 3. Daemon issues [`WorkerFrame::InvokeTool`] / [`WorkerFrame::CancelCall`];
//!    worker responds with zero or more [`WorkerFrame::ToolChunk`]s
//!    followed by exactly one [`WorkerFrame::ToolFinal`] per call_id.
//! 4. Either side dropping the socket closes the connection. Daemon
//!    drop ⇒ worker exits when its read returns EOF; worker exit ⇒
//!    daemon's read returns EOF and the session is torn down.

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use whisper_agent_host_proto::{
    CallId, CallToolResult, ContentBlock, EmbeddedResource, ToolAnnotations, ToolDescriptor,
};

/// Wire-format version. Bumped on incompatible changes; daemon and
/// worker exchange in [`WorkerFrame::Hello`] and the daemon hard-fails
/// the spawn on mismatch.
///
/// Bump rules mirror `whisper-agent-host-proto::PROTOCOL_VERSION`:
/// adding a frame variant or a `#[serde(default)]` field is non-
/// breaking; removing or repurposing a field, changing a tag, or
/// changing semantics is breaking.
pub const PROTOCOL_VERSION: u32 = 1;

/// Top-level frame on the daemon ↔ worker socket.
///
/// Direction is documented per variant. We keep a single union (not
/// separate `ToWorker` / `ToDaemon` enums) because the asymmetry is
/// small and a single union is easier to log, route, and evolve —
/// the same call we made for [`whisper_agent_host_proto::Frame`].
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WorkerFrame {
    /// **W→D**, first frame after spawn. Replaces the v1 worker's
    /// `listening <addr>` stdout handshake plus the daemon's HTTP
    /// `tools/list` probe — the worker advertises its catalog
    /// directly. Daemon hard-fails the spawn on `protocol_version`
    /// mismatch.
    Hello {
        worker_version: String,
        protocol_version: u32,
        tools: Vec<ToolDescriptor>,
    },

    /// **D→W**. Run a tool. The worker responds with zero or more
    /// [`Self::ToolChunk`] frames (only for streaming tools) followed
    /// by exactly one [`Self::ToolFinal`] for this `call_id`.
    InvokeTool {
        call_id: CallId,
        tool_name: String,
        arguments: Value,
    },

    /// **D→W**. Cancel an in-flight call. The worker may take a
    /// moment to produce the terminal [`Self::ToolFinal`] — the daemon
    /// treats the eventual `ToolFinal` as authoritative, mirroring
    /// the [`whisper_agent_host_proto::Frame::CancelCall`] semantics.
    CancelCall { call_id: CallId },

    /// **W→D**. Streaming output, zero or more per call. Only emitted
    /// by tools that stream (today: `bash`); one-shot tools yield
    /// only [`Self::ToolFinal`].
    ToolChunk {
        call_id: CallId,
        block: ContentBlock,
    },

    /// **W→D**. Terminal frame for a call, exactly one per `call_id`.
    /// Tool-level errors fold into a [`CallToolResult`] with
    /// `is_error = Some(true)` rather than a separate frame variant —
    /// the wire only has the one terminal.
    ToolFinal {
        call_id: CallId,
        result: CallToolResult,
    },
}

/// Errors from CBOR encoding / decoding helpers on [`WorkerFrame`].
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

impl WorkerFrame {
    /// Encode this frame as CBOR bytes. Length-prefix framing for
    /// the stream socket lives in the consumers (daemon + worker
    /// connection code) — this method emits the inner payload only,
    /// keeping the proto crate sync and tokio-free.
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
    //! Round-trip every `WorkerFrame` variant through CBOR. Catches
    //! serde shape errors (variant tag, field rename, missing
    //! `#[serde(default)]`) before any wire code is written against
    //! this crate.

    use super::*;

    fn assert_round_trip(frame: WorkerFrame) {
        let bytes = frame.encode_cbor().expect("encode");
        let back = WorkerFrame::decode_cbor(&bytes).expect("decode");
        assert_eq!(frame, back);
    }

    #[test]
    fn hello_round_trips() {
        assert_round_trip(WorkerFrame::Hello {
            worker_version: "0.3.13".into(),
            protocol_version: PROTOCOL_VERSION,
            tools: vec![ToolDescriptor {
                name: "read_file".into(),
                description: "read a file".into(),
                input_schema: serde_json::json!({"type":"object"}),
                annotations: ToolAnnotations::default(),
            }],
        });
    }

    #[test]
    fn hello_with_empty_catalog_round_trips() {
        // A worker that exposes no tools is pathological but valid;
        // the daemon should be able to hand the empty catalog up to
        // the scheduler without choking.
        assert_round_trip(WorkerFrame::Hello {
            worker_version: "0.3.13".into(),
            protocol_version: PROTOCOL_VERSION,
            tools: vec![],
        });
    }

    #[test]
    fn invoke_tool_round_trips() {
        assert_round_trip(WorkerFrame::InvokeTool {
            call_id: CallId(42),
            tool_name: "read_file".into(),
            arguments: serde_json::json!({ "path": "/tmp/x" }),
        });
    }

    #[test]
    fn invoke_tool_with_null_arguments_round_trips() {
        // Some tools take no arguments. JSON-null on the wire is the
        // canonical "no args" — make sure the codec doesn't fold it
        // into a missing field or reject it.
        assert_round_trip(WorkerFrame::InvokeTool {
            call_id: CallId(1),
            tool_name: "list_dir".into(),
            arguments: serde_json::Value::Null,
        });
    }

    #[test]
    fn cancel_call_round_trips() {
        assert_round_trip(WorkerFrame::CancelCall {
            call_id: CallId(42),
        });
    }

    #[test]
    fn tool_chunk_text_round_trips() {
        assert_round_trip(WorkerFrame::ToolChunk {
            call_id: CallId(42),
            block: ContentBlock::Text {
                text: "partial output\n".into(),
            },
        });
    }

    #[test]
    fn tool_chunk_image_round_trips() {
        assert_round_trip(WorkerFrame::ToolChunk {
            call_id: CallId(42),
            block: ContentBlock::Image {
                data: "AAAA".into(),
                mime_type: "image/png".into(),
            },
        });
    }

    #[test]
    fn tool_final_success_round_trips() {
        assert_round_trip(WorkerFrame::ToolFinal {
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
    fn tool_final_error_round_trips() {
        assert_round_trip(WorkerFrame::ToolFinal {
            call_id: CallId(42),
            result: CallToolResult::error_text("boom"),
        });
    }

    #[test]
    fn tool_final_with_resource_blob_round_trips() {
        assert_round_trip(WorkerFrame::ToolFinal {
            call_id: CallId(42),
            result: CallToolResult {
                content: vec![ContentBlock::Resource {
                    resource: EmbeddedResource {
                        uri: "file:///tmp/x.pdf".into(),
                        mime_type: Some("application/pdf".into()),
                        text: None,
                        blob: Some("BBBB".into()),
                    },
                }],
                is_error: None,
            },
        });
    }

    #[test]
    fn cbor_carries_kind_discriminator() {
        // Sanity-check that internal-tag serialization is happening:
        // encode a frame, decode as a generic CBOR value, confirm
        // `kind` is present at the top level. If we ever drop the
        // `#[serde(tag = "kind")]` attribute this test catches it
        // before any wire code does.
        let frame = WorkerFrame::CancelCall { call_id: CallId(1) };
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
        assert_eq!(kind, "cancel_call");
    }

    #[test]
    fn variant_tags_are_snake_case() {
        // Pin the on-the-wire spelling for each variant. If a future
        // refactor renames them, the daemon and worker stay in sync
        // (they share this enum) but anyone reading frame-level logs
        // or pcaps from a previous build would see a mismatch — flag
        // it explicitly.
        let cases = [
            (
                WorkerFrame::Hello {
                    worker_version: "x".into(),
                    protocol_version: PROTOCOL_VERSION,
                    tools: vec![],
                },
                "hello",
            ),
            (
                WorkerFrame::InvokeTool {
                    call_id: CallId(1),
                    tool_name: "x".into(),
                    arguments: Value::Null,
                },
                "invoke_tool",
            ),
            (
                WorkerFrame::CancelCall { call_id: CallId(1) },
                "cancel_call",
            ),
            (
                WorkerFrame::ToolChunk {
                    call_id: CallId(1),
                    block: ContentBlock::Text { text: "x".into() },
                },
                "tool_chunk",
            ),
            (
                WorkerFrame::ToolFinal {
                    call_id: CallId(1),
                    result: CallToolResult::text("x"),
                },
                "tool_final",
            ),
        ];
        for (frame, expected) in cases {
            let bytes = frame.encode_cbor().unwrap();
            let v: ciborium::Value = ciborium::from_reader(bytes.as_slice()).unwrap();
            let map = v.as_map().expect("map");
            let kind = map
                .iter()
                .find(|(k, _)| k.as_text() == Some("kind"))
                .expect("kind")
                .1
                .as_text()
                .expect("text");
            assert_eq!(kind, expected, "variant tag mismatch for {frame:?}");
        }
    }
}
