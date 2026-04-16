//! Per-task execution environment specification.
//!
//! Each task carries a [`SandboxSpec`] that declares how its MCP tools should be
//! isolated. UI presets stamp a default spec at task-creation time — the task's
//! spec is the source of truth at runtime.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Execution environment for a task's MCP tools.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SandboxSpec {
    /// No isolation — tools execute directly on the host with workspace-level
    /// path clamping only. Matches pre-sandbox behavior.
    #[default]
    None,
    /// OCI container isolation (podman/docker). The scheduler provisions a
    /// container from `image`, bind-mounts directories, and starts the MCP
    /// host inside it.
    Container {
        image: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        mounts: Vec<Mount>,
        #[serde(default)]
        network: NetworkPolicy,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        limits: Option<ResourceLimits>,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        env: BTreeMap<String, String>,
    },
    /// Lightweight Linux-native isolation via landlock filesystem rules and
    /// optional user namespaces. No container image required — the MCP host
    /// runs as a regular process with restricted syscall-level access.
    Landlock {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        allowed_paths: Vec<PathAccess>,
        #[serde(default)]
        network: NetworkPolicy,
    },
}

/// A bind-mount from the host into a container.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Mount {
    /// Absolute path on the host.
    pub host: String,
    /// Path inside the container.
    pub guest: String,
    #[serde(default)]
    pub mode: AccessMode,
}

/// A filesystem access rule for landlock — an allowed host path with an
/// access mode. Paths not listed are denied by default.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct PathAccess {
    pub path: String,
    #[serde(default)]
    pub mode: AccessMode,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum AccessMode {
    #[default]
    ReadOnly,
    ReadWrite,
}

/// Network access policy inside the sandbox.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
#[serde(tag = "policy", rename_all = "snake_case")]
pub enum NetworkPolicy {
    /// Full network access. Default — matches unsandboxed behavior.
    #[default]
    Unrestricted,
    /// No network access at all.
    Isolated,
    /// Allow connections to specific hosts only (e.g. `["crates.io", "github.com"]`).
    AllowList { hosts: Vec<String> },
}

/// Resource limits for the sandbox. All fields optional — backends apply
/// what they support and ignore the rest.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ResourceLimits {
    /// CPU count (fractional). Maps to `--cpus` in podman.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpus: Option<u32>,
    /// Memory cap in MiB. Maps to `--memory` in podman.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_mb: Option<u32>,
    /// Wall-clock timeout for the entire sandbox session, in seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_s: Option<u32>,
}

// ---------- Daemon API ----------

/// Request from whisper-agent to the sandbox daemon to provision an
/// execution environment for a task.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ProvisionRequest {
    pub task_id: String,
    pub spec: SandboxSpec,
}

/// Successful provision response. The daemon has created the environment and
/// started an MCP host inside it — `mcp_url` is where the scheduler should
/// connect.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ProvisionResponse {
    pub session_id: String,
    pub mcp_url: String,
}

/// Request to tear down a previously provisioned sandbox.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TeardownRequest {
    pub session_id: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_is_default() {
        let spec: SandboxSpec = Default::default();
        assert_eq!(spec, SandboxSpec::None);
    }

    #[test]
    fn container_spec_round_trips_through_json() {
        let spec = SandboxSpec::Container {
            image: "ghcr.io/whisper-agent/dev-rust:latest".into(),
            mounts: vec![Mount {
                host: "/home/me/project".into(),
                guest: "/workspace".into(),
                mode: AccessMode::ReadWrite,
            }],
            network: NetworkPolicy::AllowList {
                hosts: vec!["crates.io".into()],
            },
            limits: Some(ResourceLimits {
                cpus: Some(2),
                memory_mb: Some(4096),
                timeout_s: None,
            }),
            env: BTreeMap::from([("CARGO_HOME".into(), "/workspace/.cargo".into())]),
        };
        let json = serde_json::to_string(&spec).unwrap();
        let back: SandboxSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(spec, back);
    }

    #[test]
    fn landlock_spec_round_trips_through_json() {
        let spec = SandboxSpec::Landlock {
            allowed_paths: vec![
                PathAccess {
                    path: "/home/me/project".into(),
                    mode: AccessMode::ReadWrite,
                },
                PathAccess {
                    path: "/usr".into(),
                    mode: AccessMode::ReadOnly,
                },
            ],
            network: NetworkPolicy::Isolated,
        };
        let json = serde_json::to_string(&spec).unwrap();
        let back: SandboxSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(spec, back);
    }

    #[test]
    fn none_deserializes_from_legacy_missing_field() {
        // Existing persisted tasks have no `sandbox` field — serde(default)
        // on TaskConfig must produce SandboxSpec::None.
        let spec: SandboxSpec = serde_json::from_str(r#"{"type":"none"}"#).unwrap();
        assert_eq!(spec, SandboxSpec::None);
    }

    #[test]
    fn access_mode_defaults_to_read_only() {
        let m: Mount = serde_json::from_str(r#"{"host":"/a","guest":"/b"}"#).unwrap();
        assert_eq!(m.mode, AccessMode::ReadOnly);
    }
}
