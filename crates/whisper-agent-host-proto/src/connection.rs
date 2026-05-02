//! Connection-establishment frames and the daemon's advertised
//! capabilities.

use serde::{Deserialize, Serialize};

use crate::call::ToolDescriptor;

/// Daemon-advertised capabilities, sent in [`crate::Frame::Hello`].
///
/// The contents are the source of truth for "what this daemon can do":
/// the scheduler should not assume a tool exists, a spec kind is
/// supported, or background tasks work just because the protocol
/// version matches. This is what lets daemon and scheduler ship at
/// independent versions — the wire is locked, the surface is
/// advertised.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct DaemonCapabilities {
    /// Tool descriptors the daemon's worker exposes. Same shape the
    /// scheduler already consumes for MCP tools so existing surface
    /// (allowlist filtering, model presentation) reuses without
    /// translation.
    pub tools: Vec<ToolDescriptor>,

    /// Sandbox spec kinds this daemon can fulfill. Today this is
    /// typically `[Landlock]`; container support adds [`HostEnvSpecKind::Container`].
    pub spec_kinds: Vec<HostEnvSpecKind>,

    /// Optional cap on concurrent sessions. `None` means the daemon
    /// does not enforce a ceiling — the scheduler still applies its
    /// own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_concurrent_sessions: Option<u32>,

    /// Whether the daemon supports the [`crate::Frame::BackgroundTaskUpdate`]
    /// / [`crate::Frame::ListBackgroundTasks`] frames. Until this is
    /// `true`, the scheduler hides background-mode tools from threads
    /// bound to this daemon.
    pub supports_background_tasks: bool,
}

/// The discriminator on [`crate::HostEnvSpec`]. Daemons advertise the
/// set they can fulfill in [`DaemonCapabilities::spec_kinds`].
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HostEnvSpecKind {
    Landlock,
    Container,
}

/// Reason carried in [`crate::Frame::Goodbye`]. Free-form detail goes
/// in `Goodbye.message`; the variant here is what the recipient
/// switches on to decide whether to retry, escalate, etc.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GoodbyeReason {
    /// `protocol_version` mismatch in `Hello` / `Welcome`. The sender
    /// refuses to continue. Daemons should not retry on this — the
    /// version is compiled in.
    ProtocolMismatch,
    /// Bearer authentication failed at the WS upgrade. Daemons should
    /// not retry without an updated token.
    Unauthorized,
    /// A daemon with this name is already connected. Sent by the
    /// scheduler to a freshly-arrived second connection. The arriving
    /// daemon should report this loudly to its operator rather than
    /// reconnecting in a tight loop.
    NameAlreadyConnected,
    /// Server is shutting down. Daemons should reconnect with backoff.
    ServerShutdown,
    /// Daemon is shutting down. The scheduler tears down sessions and
    /// marks the daemon offline; reconnect is up to the daemon's
    /// operator.
    DaemonShutdown,
    /// Catch-all. Detail in `Goodbye.message`. Recipient should log
    /// loudly but not assume any particular retry semantics.
    Other,
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
    fn capabilities_with_minimal_fields() {
        round_trip(DaemonCapabilities {
            tools: vec![],
            spec_kinds: vec![HostEnvSpecKind::Landlock],
            max_concurrent_sessions: None,
            supports_background_tasks: false,
        });
    }

    #[test]
    fn capabilities_with_all_fields() {
        round_trip(DaemonCapabilities {
            tools: vec![ToolDescriptor {
                name: "ls".into(),
                description: "list".into(),
                input_schema: serde_json::json!({}),
                annotations: Default::default(),
            }],
            spec_kinds: vec![HostEnvSpecKind::Landlock, HostEnvSpecKind::Container],
            max_concurrent_sessions: Some(16),
            supports_background_tasks: true,
        });
    }

    #[test]
    fn host_env_spec_kind_serializes_snake_case() {
        let mut bytes = Vec::new();
        ciborium::into_writer(&HostEnvSpecKind::Landlock, &mut bytes).unwrap();
        let v: ciborium::Value = ciborium::from_reader(bytes.as_slice()).unwrap();
        assert_eq!(v.as_text(), Some("landlock"));
    }

    #[test]
    fn goodbye_reason_serializes_snake_case() {
        let mut bytes = Vec::new();
        ciborium::into_writer(&GoodbyeReason::NameAlreadyConnected, &mut bytes).unwrap();
        let v: ciborium::Value = ciborium::from_reader(bytes.as_slice()).unwrap();
        assert_eq!(v.as_text(), Some("name_already_connected"));
    }
}
