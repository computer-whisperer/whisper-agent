//! Pod data model and `pod.toml` (de)serialization.
//!
//! A pod is a directory on disk owning a `pod.toml` config plus per-thread
//! JSON files. This module owns the in-memory shape of the parsed config and
//! the round-trip with TOML; directory walking, file I/O, and the broader
//! `Scheduler` integration land in `crate::pod_persist` (Phase 2b) and the
//! scheduler refactor (Phase 2c).
//!
//! See `docs/design_pod_thread_scheduler.md` for the full design.
//!
//! Validation discipline: `parse` runs structural validation (every
//! `thread_defaults` reference resolves into the surrounding `[allow]` set,
//! sandbox names are unique). Pods that fail validation can't be loaded — a
//! pod-level Errored state in the registry is the wire-side answer (Phase 2b).

use std::collections::BTreeSet;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use whisper_agent_protocol::{ApprovalPolicy, SandboxSpec};

pub type PodId = String;

/// Filename of the per-pod config inside the pod's directory.
pub const POD_TOML: &str = "pod.toml";
/// Subdirectory under the pod that holds per-thread JSON files.
pub const THREADS_DIR: &str = "threads";

#[derive(Debug, Error)]
pub enum PodConfigError {
    #[error("toml parse: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("toml encode: {0}")]
    Encode(#[from] toml::ser::Error),
    #[error("duplicate sandbox name `{0}` in [[allow.sandbox]]")]
    DuplicateSandboxName(String),
    #[error(
        "thread_defaults.{field} = `{value}` does not match any entry in [allow] (valid: [{valid}])"
    )]
    UnknownThreadDefault {
        field: &'static str,
        value: String,
        valid: String,
    },
}

/// Parsed `pod.toml`. The struct is the pod's truth — anything not present in
/// `[allow]` is unreachable to threads in this pod.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct PodConfig {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub created_at: DateTime<Utc>,

    pub allow: PodAllow,
    pub thread_defaults: ThreadDefaults,
    #[serde(default)]
    pub limits: PodLimits,
    // [[hooks]] reserved for future Lua integration; not parsed yet.
}

/// The capability boundary for threads in this pod. Backends / MCP hosts
/// reference the server catalog by name; sandboxes are inline specs (pods are
/// self-contained for sandbox definitions).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
pub struct PodAllow {
    #[serde(default)]
    pub backends: Vec<String>,
    #[serde(default)]
    pub mcp_hosts: Vec<String>,
    #[serde(default)]
    pub sandbox: Vec<NamedSandboxSpec>,
}

/// An inline sandbox spec with a name. The name is the handle threads use to
/// pick a sandbox (`thread_defaults.sandbox = "..."` or
/// `bindings.sandbox = "..."` on a `CreateThread` request).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct NamedSandboxSpec {
    pub name: String,
    /// `SandboxSpec` is `tag = "type"` so the inner discriminant lifts to a
    /// sibling key here. TOML form:
    /// ```toml
    /// [[allow.sandbox]]
    /// name = "landlock-rw"
    /// type = "landlock"
    /// allowed_paths = [...]
    /// network = { policy = "unrestricted" }
    /// ```
    #[serde(flatten)]
    pub spec: SandboxSpec,
}

/// Defaults applied to every fresh thread in this pod. Each field is
/// overridable per-thread; any reference must resolve into the surrounding
/// `[allow]` set or `parse` rejects the pod.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ThreadDefaults {
    pub backend: String,
    pub model: String,
    /// Path relative to the pod directory. The persistence layer reads the
    /// referenced file and treats its contents as the system prompt; this
    /// keeps long prompts editable as standalone files.
    pub system_prompt_file: String,
    pub max_tokens: u32,
    pub max_turns: u32,
    #[serde(default)]
    pub approval_policy: ApprovalPolicy,
    /// Name of one of the `[[allow.sandbox]]` entries. Empty string is allowed
    /// only if `[allow].sandbox` is also empty (no-sandbox pod).
    #[serde(default)]
    pub sandbox: String,
    #[serde(default)]
    pub mcp_hosts: Vec<String>,
}

/// Pod-level limits. Only `max_concurrent_threads` is wired today; token /
/// cost budgets land when there's a concrete enforcement path.
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

impl PodConfig {
    /// Parse from TOML text and run structural validation.
    pub fn parse(text: &str) -> Result<Self, PodConfigError> {
        let config: PodConfig = toml::from_str(text)?;
        config.validate()?;
        Ok(config)
    }

    /// Serialize to TOML text. Round-trips with `parse`.
    pub fn to_toml(&self) -> Result<String, PodConfigError> {
        Ok(toml::to_string_pretty(self)?)
    }

    /// Verify every reference in `thread_defaults` resolves into `[allow]`
    /// and that sandbox names are unique. Called by `parse`; safe to call
    /// after in-memory mutations too.
    pub fn validate(&self) -> Result<(), PodConfigError> {
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        for nss in &self.allow.sandbox {
            if !seen.insert(nss.name.as_str()) {
                return Err(PodConfigError::DuplicateSandboxName(nss.name.clone()));
            }
        }

        if !self.allow.backends.iter().any(|b| b == &self.thread_defaults.backend) {
            return Err(PodConfigError::UnknownThreadDefault {
                field: "backend",
                value: self.thread_defaults.backend.clone(),
                valid: self.allow.backends.join(", "),
            });
        }

        // Sandbox: empty is valid only if pod has no allowed sandboxes (degenerate
        // no-isolation pod). Otherwise must name an entry in [[allow.sandbox]].
        if !self.thread_defaults.sandbox.is_empty()
            && !self
                .allow
                .sandbox
                .iter()
                .any(|nss| nss.name == self.thread_defaults.sandbox)
        {
            let valid = self
                .allow
                .sandbox
                .iter()
                .map(|nss| nss.name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(PodConfigError::UnknownThreadDefault {
                field: "sandbox",
                value: self.thread_defaults.sandbox.clone(),
                valid,
            });
        }

        for host in &self.thread_defaults.mcp_hosts {
            if !self.allow.mcp_hosts.iter().any(|h| h == host) {
                return Err(PodConfigError::UnknownThreadDefault {
                    field: "mcp_hosts",
                    value: host.clone(),
                    valid: self.allow.mcp_hosts.join(", "),
                });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use whisper_agent_protocol::sandbox::{AccessMode, NetworkPolicy, PathAccess};

    fn sample_config() -> PodConfig {
        PodConfig {
            name: "whisper-agent dev".into(),
            description: Some("Working on whisper-agent itself".into()),
            created_at: "2026-04-16T10:23:11Z".parse().unwrap(),
            allow: PodAllow {
                backends: vec!["anthropic".into(), "openai-compat".into()],
                mcp_hosts: vec!["fetch".into(), "search".into()],
                sandbox: vec![
                    NamedSandboxSpec {
                        name: "landlock-rw".into(),
                        spec: SandboxSpec::Landlock {
                            allowed_paths: vec![PathAccess {
                                path: "/home/me/project".into(),
                                mode: AccessMode::ReadWrite,
                            }],
                            network: NetworkPolicy::Unrestricted,
                        },
                    },
                    NamedSandboxSpec {
                        name: "no-isolation".into(),
                        spec: SandboxSpec::None,
                    },
                ],
            },
            thread_defaults: ThreadDefaults {
                backend: "anthropic".into(),
                model: "claude-opus-4-7".into(),
                system_prompt_file: "system_prompt.md".into(),
                max_tokens: 32000,
                max_turns: 100,
                approval_policy: ApprovalPolicy::PromptDestructive,
                sandbox: "landlock-rw".into(),
                mcp_hosts: vec!["fetch".into(), "search".into()],
            },
            limits: PodLimits {
                max_concurrent_threads: 10,
            },
        }
    }

    #[test]
    fn round_trips_through_toml() {
        let cfg = sample_config();
        let text = cfg.to_toml().unwrap();
        let back = PodConfig::parse(&text).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn detects_duplicate_sandbox_name() {
        let mut cfg = sample_config();
        cfg.allow.sandbox.push(NamedSandboxSpec {
            name: "landlock-rw".into(), // duplicate
            spec: SandboxSpec::None,
        });
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, PodConfigError::DuplicateSandboxName(_)));
    }

    #[test]
    fn rejects_unknown_default_backend() {
        let mut cfg = sample_config();
        cfg.thread_defaults.backend = "not-a-backend".into();
        let err = cfg.validate().unwrap_err();
        match err {
            PodConfigError::UnknownThreadDefault { field, value, .. } => {
                assert_eq!(field, "backend");
                assert_eq!(value, "not-a-backend");
            }
            other => panic!("expected UnknownThreadDefault, got {other:?}"),
        }
    }

    #[test]
    fn rejects_unknown_default_sandbox() {
        let mut cfg = sample_config();
        cfg.thread_defaults.sandbox = "phantom".into();
        let err = cfg.validate().unwrap_err();
        match err {
            PodConfigError::UnknownThreadDefault { field, .. } => assert_eq!(field, "sandbox"),
            other => panic!("expected UnknownThreadDefault, got {other:?}"),
        }
    }

    #[test]
    fn empty_default_sandbox_ok_when_allow_is_empty() {
        let mut cfg = sample_config();
        cfg.allow.sandbox.clear();
        cfg.thread_defaults.sandbox = String::new();
        cfg.validate().unwrap();
    }

    #[test]
    fn rejects_unknown_default_mcp_host() {
        let mut cfg = sample_config();
        cfg.thread_defaults.mcp_hosts.push("phantom".into());
        let err = cfg.validate().unwrap_err();
        match err {
            PodConfigError::UnknownThreadDefault { field, value, .. } => {
                assert_eq!(field, "mcp_hosts");
                assert_eq!(value, "phantom");
            }
            other => panic!("expected UnknownThreadDefault, got {other:?}"),
        }
    }

    #[test]
    fn parses_minimal_pod() {
        let text = r#"
name = "minimal"
created_at = "2026-04-16T10:00:00Z"

[allow]
backends = ["anthropic"]
mcp_hosts = []
sandbox = []

[thread_defaults]
backend = "anthropic"
model = "claude-opus-4-7"
system_prompt_file = "system_prompt.md"
max_tokens = 8000
max_turns = 50
"#;
        let cfg = PodConfig::parse(text).unwrap();
        assert_eq!(cfg.name, "minimal");
        assert_eq!(cfg.thread_defaults.max_tokens, 8000);
        // limits omitted → default 10
        assert_eq!(cfg.limits.max_concurrent_threads, 10);
    }

    #[test]
    fn unparseable_toml_returns_parse_error() {
        let err = PodConfig::parse("not = valid {toml").unwrap_err();
        assert!(matches!(err, PodConfigError::Parse(_)));
    }
}
