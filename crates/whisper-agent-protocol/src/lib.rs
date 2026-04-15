//! Wire protocol between the whisper-agent server and its webui.
//!
//! Both directions are CBOR-encoded enums — see the helper functions at the bottom for
//! the (de)serialization entry points. The shape is deliberately small and explicit:
//! each variant maps to one logical event the harness wants the UI to render or one
//! action the user can request.

use serde::{Deserialize, Serialize};

/// Messages the client (webui) sends to the server (whisper-agent).
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientToServer {
    /// User submitted a new prompt.
    UserMessage { text: String },
    // Cancel deferred to v0.2.
}

/// Messages the server sends to the client.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerToClient {
    /// Connection-level status changes ("connected", "ready", "error", etc.).
    Status { state: SessionState, detail: Option<String> },
    /// A new assistant turn is starting.
    AssistantBegin,
    /// Text emitted by the model in this turn. For MVP (non-streaming Anthropic) this
    /// arrives as one chunk per turn; once streaming is added, multiple deltas may arrive.
    AssistantText { text: String },
    /// The model emitted a tool_use block; we are about to invoke it via MCP.
    ToolCallBegin { id: String, name: String, args_preview: String },
    /// Tool returned. `result_preview` is the (possibly truncated) text result.
    ToolCallEnd { id: String, result_preview: String, is_error: bool },
    /// Turn ended (either tool_use → next turn, or end_turn → loop done).
    AssistantEnd { stop_reason: Option<String>, usage: Usage },
    /// The agent loop finished (model returned a non-tool stop_reason or hit max_turns).
    LoopComplete,
    /// Something went wrong inside the harness.
    Error { message: String },
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SessionState {
    Connected,
    Ready,
    Working,
    Error,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, Default)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_read_input_tokens: u32,
    pub cache_creation_input_tokens: u32,
}

// ---------- (de)serialization ----------

#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    #[error("encode: {0}")]
    Encode(String),
    #[error("decode: {0}")]
    Decode(String),
}

pub fn encode_to_server(msg: &ClientToServer) -> Result<Vec<u8>, CodecError> {
    let mut buf = Vec::new();
    ciborium::ser::into_writer(msg, &mut buf).map_err(|e| CodecError::Encode(e.to_string()))?;
    Ok(buf)
}

pub fn decode_from_client(bytes: &[u8]) -> Result<ClientToServer, CodecError> {
    ciborium::de::from_reader(bytes).map_err(|e| CodecError::Decode(e.to_string()))
}

pub fn encode_to_client(msg: &ServerToClient) -> Result<Vec<u8>, CodecError> {
    let mut buf = Vec::new();
    ciborium::ser::into_writer(msg, &mut buf).map_err(|e| CodecError::Encode(e.to_string()))?;
    Ok(buf)
}

pub fn decode_from_server(bytes: &[u8]) -> Result<ServerToClient, CodecError> {
    ciborium::de::from_reader(bytes).map_err(|e| CodecError::Decode(e.to_string()))
}
