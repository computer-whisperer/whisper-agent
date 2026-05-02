//! Tool-call types — call identifier, descriptor shape, and the
//! content blocks shared between [`crate::Frame::ToolChunk`] and
//! [`crate::Frame::ToolFinal`].
//!
//! These mirror the MCP 2025-06-18 shape with two intentional
//! deviations: snake_case throughout (we're not on the MCP wire,
//! we're internal CBOR) and full `Deserialize` impls (the MCP types
//! in `whisper-agent-mcp-proto` are server-side, mostly Serialize-
//! only; this protocol is bidirectional so both sides need both
//! directions).

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Per-session monotonic call identifier. Scheduler-assigned, starts
/// at 1 for each fresh session, never reused within a session.
///
/// `u64` is plenty — even at 1000 calls/sec it's ~600 million years
/// of headroom.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CallId(pub u64);

impl std::fmt::Display for CallId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

/// A tool the daemon's worker exposes. Sent in
/// [`crate::DaemonCapabilities::tools`] at connect.
///
/// The shape matches what the scheduler already consumes for MCP
/// tools — the same allowlist filtering and model-presentation paths
/// reuse without translation.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ToolDescriptor {
    pub name: String,
    pub description: String,
    /// JSON-Schema for the `arguments` field of [`crate::Frame::InvokeTool`].
    /// The wire transports this as opaque structured data; the
    /// scheduler validates against it before sending.
    pub input_schema: Value,
    /// Optional safety / capability hints. Defaults to all-`None`,
    /// which the scheduler treats as "no hint provided."
    #[serde(default)]
    pub annotations: ToolAnnotations,
}

/// Tool-annotation fields per MCP 2025-06-18. All fields are tri-state
/// (`Option<bool>`) so daemons can be explicit about defaults vs
/// "no opinion."
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolAnnotations {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub read_only_hint: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub destructive_hint: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotent_hint: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open_world_hint: Option<bool>,
}

/// One content block in a tool result or chunk. Same conceptual shape
/// as MCP's `ContentBlock`; see also `whisper-agent-mcp-proto`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    /// Image content. `data` is base64-encoded bytes; `mime_type` is
    /// the IANA media type (`image/png`, `image/jpeg`, ...).
    Image {
        data: String,
        mime_type: String,
    },
    /// Embedded resource — typically a binary blob (e.g. PDF) too
    /// large or non-image to fit into [`Self::Image`].
    Resource {
        resource: EmbeddedResource,
    },
}

/// An embedded resource. Mirrors MCP's shape: `text` and `blob` are
/// mutually exclusive — one or the other depending on the resource
/// kind.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct EmbeddedResource {
    pub uri: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Inline text content. Set when the resource is text; mutually
    /// exclusive with [`Self::blob`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Base64-encoded binary content. Set when the resource is
    /// binary; mutually exclusive with [`Self::text`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blob: Option<String>,
}

/// Final result of a tool call, terminal frame for a call_id.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct CallToolResult {
    pub content: Vec<ContentBlock>,
    /// `Some(true)` if the tool returned an error result. `None` /
    /// `Some(false)` for success — tools that have no concept of
    /// "error vs success result" (most read-only operations) leave
    /// this `None` to keep the wire compact.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

impl CallToolResult {
    /// Convenience: a successful single-text-block result.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::Text { text: text.into() }],
            is_error: None,
        }
    }

    /// Convenience: an error single-text-block result.
    pub fn error_text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::Text { text: text.into() }],
            is_error: Some(true),
        }
    }

    /// Convenience: a successful single-image result. `data` is
    /// base64-encoded bytes; `mime_type` is the IANA media type
    /// (`image/png`, `image/jpeg`, ...).
    pub fn image(data: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::Image {
                data: data.into(),
                mime_type: mime_type.into(),
            }],
            is_error: None,
        }
    }

    /// Convenience: a successful single embedded-resource result with a
    /// base64-encoded binary blob (PDFs, etc.). `uri` is a stable
    /// identifier (typically `file:///<path>`); `mime_type` is the IANA
    /// media type.
    pub fn resource_blob(
        uri: impl Into<String>,
        mime_type: impl Into<String>,
        blob: impl Into<String>,
    ) -> Self {
        Self {
            content: vec![ContentBlock::Resource {
                resource: EmbeddedResource {
                    uri: uri.into(),
                    mime_type: Some(mime_type.into()),
                    text: None,
                    blob: Some(blob.into()),
                },
            }],
            is_error: None,
        }
    }
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
    fn call_id_round_trips() {
        round_trip(CallId(0));
        round_trip(CallId(u64::MAX));
    }

    #[test]
    fn tool_descriptor_default_annotations_round_trip() {
        round_trip(ToolDescriptor {
            name: "read_file".into(),
            description: "read a file".into(),
            input_schema: serde_json::json!({"type":"object"}),
            annotations: ToolAnnotations::default(),
        });
    }

    #[test]
    fn tool_descriptor_with_annotations_round_trips() {
        round_trip(ToolDescriptor {
            name: "bash".into(),
            description: "run a shell command".into(),
            input_schema: serde_json::json!({"type":"object"}),
            annotations: ToolAnnotations {
                title: Some("Shell".into()),
                read_only_hint: Some(false),
                destructive_hint: Some(true),
                idempotent_hint: Some(false),
                open_world_hint: Some(true),
            },
        });
    }

    #[test]
    fn content_block_text_round_trips() {
        round_trip(ContentBlock::Text {
            text: "hello world".into(),
        });
    }

    #[test]
    fn content_block_image_round_trips() {
        round_trip(ContentBlock::Image {
            data: "AAAA".into(),
            mime_type: "image/png".into(),
        });
    }

    #[test]
    fn content_block_resource_text_round_trips() {
        round_trip(ContentBlock::Resource {
            resource: EmbeddedResource {
                uri: "file:///tmp/a.txt".into(),
                mime_type: Some("text/plain".into()),
                text: Some("body".into()),
                blob: None,
            },
        });
    }

    #[test]
    fn content_block_resource_blob_round_trips() {
        round_trip(ContentBlock::Resource {
            resource: EmbeddedResource {
                uri: "file:///tmp/a.pdf".into(),
                mime_type: Some("application/pdf".into()),
                text: None,
                blob: Some("BBBB".into()),
            },
        });
    }

    #[test]
    fn call_tool_result_helpers() {
        let ok = CallToolResult::text("done");
        assert_eq!(ok.is_error, None);
        let err = CallToolResult::error_text("boom");
        assert_eq!(err.is_error, Some(true));
        round_trip(ok);
        round_trip(err);
    }

    #[test]
    fn call_tool_result_is_error_elides_when_none() {
        let ok = CallToolResult::text("done");
        let mut bytes = Vec::new();
        ciborium::into_writer(&ok, &mut bytes).unwrap();
        let v: ciborium::Value = ciborium::from_reader(bytes.as_slice()).unwrap();
        let map = v.as_map().expect("map");
        assert!(
            !map.iter().any(|(k, _)| k.as_text() == Some("is_error")),
            "is_error must be elided when None"
        );
    }

    #[test]
    fn content_block_carries_type_discriminator() {
        let block = ContentBlock::Text { text: "x".into() };
        let mut bytes = Vec::new();
        ciborium::into_writer(&block, &mut bytes).unwrap();
        let v: ciborium::Value = ciborium::from_reader(bytes.as_slice()).unwrap();
        let map = v.as_map().expect("map");
        let ty = map
            .iter()
            .find(|(k, _)| k.as_text() == Some("type"))
            .expect("`type` discriminator")
            .1
            .as_text()
            .expect("text");
        assert_eq!(ty, "text");
    }
}
