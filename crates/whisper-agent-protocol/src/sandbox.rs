//! Host environment specification — describes the kind of isolated
//! execution environment a thread runs inside (the "host env" the
//! scheduler asks a `HostEnvProvider` daemon to provision).
//!
//! Each `[[allow.host_env]]` entry on a pod carries a `HostEnvSpec`
//! plus the `provider` name that will be asked to provision it. The
//! spec is the wire-level input the daemon receives in
//! `ProvisionRequest`; it does not itself know which provider will
//! fulfill it (that's resolved server-side against the catalog in
//! `whisper-agent.toml`).
//!
//! There is no "none" / "bare" spec. A thread that has no host env
//! bound simply has no host-env MCP connection — its tool catalog is
//! whatever shared MCP hosts it binds to, and nothing more. "No
//! isolation" is a future spec variant the sandbox daemon can grow
//! (it would still provision a real MCP host, just without the
//! landlock wrapper).

use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Execution environment a `HostEnvProvider` daemon can provision. A
/// thread's host_env binding resolves to one of these via its pod's
/// `[[allow.host_env]]` table. Provider responsibilities are clear:
///   * `Landlock` — spawn an MCP host inside a landlock jail.
///   * `Container` — spawn an MCP host inside an OCI container.
///
/// Specs never encode "no environment" — that's the absence of a
/// binding, not a flavor of binding.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HostEnvSpec {
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

/// A filesystem access rule for landlock. Serializes as a compact
/// `"<path>:<ro|rw>"` string — keeps the TOML layout of
/// `[[allow.host_env]]` entries flat and self-contained instead of
/// spawning a nested `[[allow.host_env.allowed_paths]]` block that
/// visually reads as a top-level sibling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathAccess {
    pub path: String,
    pub mode: AccessMode,
}

impl PathAccess {
    pub fn read_only(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            mode: AccessMode::ReadOnly,
        }
    }
    pub fn read_write(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            mode: AccessMode::ReadWrite,
        }
    }
}

impl Serialize for PathAccess {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let suffix = match self.mode {
            AccessMode::ReadOnly => "ro",
            AccessMode::ReadWrite => "rw",
        };
        s.collect_str(&format_args!("{}:{}", self.path, suffix))
    }
}

impl<'de> Deserialize<'de> for PathAccess {
    /// Accepts the canonical compact form `"/path:rw"` as well as the
    /// legacy object form `{ path = "...", mode = "read_only" }` that
    /// existed before the compact serializer landed. Objects without
    /// a `mode` field default to `ReadOnly`.
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::Error;

        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Wire {
            Compact(String),
            Struct {
                path: String,
                #[serde(default)]
                mode: AccessMode,
            },
        }

        match Wire::deserialize(d)? {
            Wire::Compact(raw) => {
                let (path, mode) = raw
                    .rsplit_once(':')
                    .ok_or_else(|| D::Error::custom("missing mode suffix (`:ro` or `:rw`)"))?;
                let mode = match mode {
                    "ro" => AccessMode::ReadOnly,
                    "rw" => AccessMode::ReadWrite,
                    other => {
                        return Err(D::Error::custom(format!(
                            "unknown path mode `{other}` (expected `ro` or `rw`)",
                        )));
                    }
                };
                if path.is_empty() {
                    return Err(D::Error::custom("path is empty"));
                }
                Ok(Self {
                    path: path.to_string(),
                    mode,
                })
            }
            Wire::Struct { path, mode } => {
                if path.is_empty() {
                    return Err(D::Error::custom("path is empty"));
                }
                Ok(Self { path, mode })
            }
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum AccessMode {
    #[default]
    ReadOnly,
    ReadWrite,
}

/// Network access policy inside the sandbox. Simple variants
/// serialize as bare strings (`"unrestricted"`, `"isolated"`) so a
/// pod's TOML reads `network = "unrestricted"` instead of
/// `network = { policy = "unrestricted" }`. Payload-bearing variants
/// serialize as objects.
///
/// Deserialize also accepts the legacy internally-tagged form
/// (`{ policy = "unrestricted" }`) — existing pod.toml files from
/// before the untagging migrate without edits.
#[derive(Serialize, Debug, Clone, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum NetworkPolicy {
    /// Full network access. Default — matches unsandboxed behavior.
    #[default]
    Unrestricted,
    /// No network access at all.
    Isolated,
    /// Allow connections to specific hosts only (e.g. `["crates.io", "github.com"]`).
    AllowList { hosts: Vec<String> },
}

impl<'de> Deserialize<'de> for NetworkPolicy {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::Error;

        #[derive(Deserialize)]
        struct AllowListPayload {
            hosts: Vec<String>,
        }

        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Wire {
            /// Canonical: `"unrestricted"` / `"isolated"` as bare strings.
            Bare(String),
            /// Canonical externally-tagged payload variant:
            /// `{ "allow_list": { "hosts": [...] } }`.
            ExternallyTagged { allow_list: AllowListPayload },
            /// Legacy internally-tagged form from before the untag
            /// (`{ policy = "unrestricted" }` or
            /// `{ policy = "allow_list", hosts = [...] }`). Pre-refactor
            /// pod.toml files use this shape.
            Legacy {
                policy: String,
                #[serde(default)]
                hosts: Option<Vec<String>>,
            },
        }

        match Wire::deserialize(d)? {
            Wire::Bare(s) => match s.as_str() {
                "unrestricted" => Ok(Self::Unrestricted),
                "isolated" => Ok(Self::Isolated),
                other => Err(D::Error::custom(format!(
                    "unknown network policy `{other}`",
                ))),
            },
            Wire::ExternallyTagged { allow_list } => Ok(Self::AllowList {
                hosts: allow_list.hosts,
            }),
            Wire::Legacy { policy, hosts } => match policy.as_str() {
                "unrestricted" => Ok(Self::Unrestricted),
                "isolated" => Ok(Self::Isolated),
                "allow_list" => Ok(Self::AllowList {
                    hosts: hosts.unwrap_or_default(),
                }),
                other => Err(D::Error::custom(format!(
                    "unknown network policy `{other}`",
                ))),
            },
        }
    }
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
    pub thread_id: String,
    pub spec: HostEnvSpec,
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
    fn container_spec_round_trips_through_json() {
        let spec = HostEnvSpec::Container {
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
        let back: HostEnvSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(spec, back);
    }

    #[test]
    fn landlock_spec_round_trips_through_json() {
        let spec = HostEnvSpec::Landlock {
            allowed_paths: vec![
                PathAccess::read_write("/home/me/project"),
                PathAccess::read_only("/usr"),
            ],
            network: NetworkPolicy::Isolated,
        };
        let json = serde_json::to_string(&spec).unwrap();
        let back: HostEnvSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(spec, back);
    }

    #[test]
    fn path_access_serializes_as_compact_string() {
        let pa = PathAccess::read_write("/tmp/workspace");
        let json = serde_json::to_string(&pa).unwrap();
        assert_eq!(json, r#""/tmp/workspace:rw""#);
        let back: PathAccess = serde_json::from_str(&json).unwrap();
        assert_eq!(back, pa);
    }

    #[test]
    fn path_access_rejects_missing_mode() {
        let err = serde_json::from_str::<PathAccess>(r#""/tmp/foo""#).unwrap_err();
        assert!(err.to_string().contains("missing mode suffix"));
    }

    #[test]
    fn path_access_rejects_unknown_mode() {
        let err = serde_json::from_str::<PathAccess>(r#""/tmp/foo:xx""#).unwrap_err();
        assert!(err.to_string().contains("unknown path mode"));
    }

    #[test]
    fn network_policy_serializes_as_bare_string_for_unit_variants() {
        let p = NetworkPolicy::Unrestricted;
        assert_eq!(serde_json::to_string(&p).unwrap(), r#""unrestricted""#);
        let back: NetworkPolicy = serde_json::from_str(r#""isolated""#).unwrap();
        assert_eq!(back, NetworkPolicy::Isolated);
    }

    #[test]
    fn access_mode_defaults_to_read_only() {
        let m: Mount = serde_json::from_str(r#"{"host":"/a","guest":"/b"}"#).unwrap();
        assert_eq!(m.mode, AccessMode::ReadOnly);
    }
}
