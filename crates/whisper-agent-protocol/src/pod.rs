//! Pod config wire types — shared between the server and any client that
//! wants to render or edit a pod's configuration. Owned in this crate (not in
//! `whisper-agent::pod`) because both sides need the same structural shape;
//! TOML parsing and validation stay in the server.
//!
//! `created_at` is an RFC-3339 string rather than a `chrono::DateTime` for
//! the same reason `ThreadSummary.created_at` is — the protocol crate
//! deliberately does not depend on chrono.

use serde::{Deserialize, Serialize};

use crate::sandbox::HostEnvSpec;
use crate::ApprovalPolicy;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct PodConfig {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// RFC-3339 timestamp.
    pub created_at: String,

    pub allow: PodAllow,
    pub thread_defaults: ThreadDefaults,
    #[serde(default)]
    pub limits: PodLimits,
    // [[hooks]] reserved for future Lua integration; not parsed yet.
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
pub struct PodAllow {
    #[serde(default)]
    pub backends: Vec<String>,
    #[serde(default)]
    pub mcp_hosts: Vec<String>,
    #[serde(default)]
    pub host_env: Vec<NamedHostEnv>,
}

/// One pod-level "host env" entry — a named (provider, spec) pair the
/// pod's threads are allowed to bind to. The provider name resolves
/// against the server-level catalog in `whisper-agent.toml`. Every
/// `HostEnvSpec` variant refers to a real provisioned environment; a
/// pod that wants "no isolation" for its threads just declares fewer
/// `[[allow.host_env]]` entries (or zero) — threads then run with no
/// host-env MCP connection.
///
/// TOML form:
/// ```toml
/// [[allow.host_env]]
/// name = "rust-dev"
/// provider = "landlock-laptop"
/// type = "landlock"        # HostEnvSpec discriminator
/// allowed_paths = ["/home/me/project:rw", "/:ro"]
/// network = "isolated"
/// ```
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct NamedHostEnv {
    pub name: String,
    pub provider: String,
    #[serde(flatten)]
    pub spec: HostEnvSpec,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ThreadDefaults {
    pub backend: String,
    pub model: String,
    /// Path relative to the pod directory.
    pub system_prompt_file: String,
    pub max_tokens: u32,
    pub max_turns: u32,
    #[serde(default)]
    pub approval_policy: ApprovalPolicy,
    /// Name of one of the `[[allow.host_env]]` entries. Empty allowed
    /// only when `[allow].host_env` is empty.
    #[serde(default)]
    pub host_env: String,
    #[serde(default)]
    pub mcp_hosts: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct PodLimits {
    #[serde(default = "default_max_concurrent_threads")]
    pub max_concurrent_threads: u32,
}

fn default_max_concurrent_threads() -> u32 {
    10
}

impl Default for PodLimits {
    fn default() -> Self {
        Self {
            max_concurrent_threads: default_max_concurrent_threads(),
        }
    }
}

/// Lightweight per-pod entry used by `PodList` responses.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct PodSummary {
    pub pod_id: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub created_at: String,
    pub thread_count: u32,
    pub archived: bool,
}

/// Full pod state delivered in response to `GetPod`. Carries both the parsed
/// config (for structured form rendering) and the raw TOML text (for the raw
/// editor view), so clients pick whichever fits the surface they're showing.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PodSnapshot {
    pub pod_id: String,
    pub config: PodConfig,
    pub toml_text: String,
    pub threads: Vec<crate::ThreadSummary>,
    pub archived: bool,
}
