//! Pod TOML parsing, validation, and on-disk constants.
//!
//! The data types themselves (`PodConfig`, `PodAllow`, `NamedHostEnv`,
//! `ThreadDefaults`, `PodLimits`) live in the protocol crate so the webui
//! can render them directly from wire snapshots. This module owns the
//! server-side concerns: filename constants, the TOML round-trip, and
//! structural validation.
//!
//! See `docs/design_pod_thread_scheduler.md` for the full design.
//!
//! Submodules:
//! - [`config`] — server-side catalog config (the TOML that lists model
//!   backends the server is willing to drive).
//! - [`persist`] — pod/thread on-disk I/O (read, write, list, rename).
//! - [`resources`] — registry of sandboxes, MCP hosts, and model backends
//!   as first-class entities with lifecycles independent of tasks.

pub mod config;
pub mod persist;
pub mod resources;

use std::collections::BTreeSet;
use std::path::PathBuf;

use thiserror::Error;
use whisper_agent_protocol::PodConfig;

pub type PodId = String;
pub type ThreadId = String;

/// Filename of the per-pod config inside the pod's directory.
pub const POD_TOML: &str = "pod.toml";
/// Subdirectory under the pod that holds per-thread JSON files.
pub const THREADS_DIR: &str = "threads";

/// In-memory representation of a pod the scheduler is currently tracking.
///
/// One per pod directory under `<pods_root>/`. The `config` is the parsed
/// `pod.toml`; `raw_toml` is the on-disk text (kept verbatim so the wire
/// can return it for raw editing without re-serializing). `threads` is the
/// scheduler's view of which threads currently belong to this pod —
/// authoritative at runtime; the on-disk truth is `<pod>/threads/*.json`.
///
/// `system_prompt` is the text of the file referenced by
/// `config.thread_defaults.system_prompt_file`, loaded eagerly so the
/// scheduler can build a fresh `ThreadConfig` without blocking on I/O.
/// Empty string when the file is absent or `system_prompt_file` is empty.
#[derive(Debug, Clone)]
pub struct Pod {
    pub id: PodId,
    pub dir: PathBuf,
    pub config: PodConfig,
    pub raw_toml: String,
    pub system_prompt: String,
    pub threads: BTreeSet<ThreadId>,
    pub archived: bool,
}

impl Pod {
    pub fn new(
        id: PodId,
        dir: PathBuf,
        config: PodConfig,
        raw_toml: String,
        system_prompt: String,
    ) -> Self {
        Self {
            id,
            dir,
            config,
            raw_toml,
            system_prompt,
            threads: BTreeSet::new(),
            archived: false,
        }
    }
}

#[derive(Debug, Error)]
pub enum PodConfigError {
    #[error("toml parse: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("toml encode: {0}")]
    Encode(#[from] toml::ser::Error),
    #[error("duplicate sandbox name `{0}` in [[allow.host_env]]")]
    DuplicateSandboxName(String),
    #[error(
        "thread_defaults.{field} = `{value}` does not match any entry in [allow] (valid: [{valid}])"
    )]
    UnknownThreadDefault {
        field: &'static str,
        value: String,
        valid: String,
    },
    #[error(
        "thread_defaults.host_env is empty but [[allow.host_env]] has entries — pick one (valid: [{valid}])"
    )]
    HostEnvDefaultRequired { valid: String },
}

/// Parse pod.toml text, then run structural validation.
pub fn parse_toml(text: &str) -> Result<PodConfig, PodConfigError> {
    let config: PodConfig = toml::from_str(text)?;
    validate(&config)?;
    Ok(config)
}

pub fn to_toml(config: &PodConfig) -> Result<String, PodConfigError> {
    Ok(toml::to_string_pretty(config)?)
}

/// Verify every reference in `thread_defaults` resolves into `[allow]` and
/// that sandbox names are unique. Safe to call after in-memory mutations
/// (e.g. when applying an `UpdatePodConfig` patch).
pub fn validate(config: &PodConfig) -> Result<(), PodConfigError> {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for nss in &config.allow.host_env {
        if !seen.insert(nss.name.as_str()) {
            return Err(PodConfigError::DuplicateSandboxName(nss.name.clone()));
        }
    }

    if !config
        .allow
        .backends
        .iter()
        .any(|b| b == &config.thread_defaults.backend)
    {
        return Err(PodConfigError::UnknownThreadDefault {
            field: "backend",
            value: config.thread_defaults.backend.clone(),
            valid: config.allow.backends.join(", "),
        });
    }

    // Host env validation. The pod's `allow.host_env` is authoritative;
    // `thread_defaults.host_env` must either be empty (for pods with
    // zero entries — threads there run with shared MCPs only) or name
    // an entry in the allow list. An empty default on a pod that has
    // entries is rejected to prevent a silent "no host env" fallback.
    //   * empty defaults + empty allow => shared-only pod, OK.
    //   * empty defaults + non-empty allow => reject.
    //   * non-empty defaults => must reference an allow entry.
    let valid_names: Vec<&str> = config
        .allow
        .host_env
        .iter()
        .map(|nh| nh.name.as_str())
        .collect();
    if config.thread_defaults.host_env.is_empty() {
        if !valid_names.is_empty() {
            return Err(PodConfigError::HostEnvDefaultRequired {
                valid: valid_names.join(", "),
            });
        }
    } else if !valid_names.contains(&config.thread_defaults.host_env.as_str()) {
        return Err(PodConfigError::UnknownThreadDefault {
            field: "host_env",
            value: config.thread_defaults.host_env.clone(),
            valid: valid_names.join(", "),
        });
    }

    for host in &config.thread_defaults.mcp_hosts {
        if !config.allow.mcp_hosts.iter().any(|h| h == host) {
            return Err(PodConfigError::UnknownThreadDefault {
                field: "mcp_hosts",
                value: host.clone(),
                valid: config.allow.mcp_hosts.join(", "),
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use whisper_agent_protocol::sandbox::{NetworkPolicy, PathAccess};
    use whisper_agent_protocol::{
        ApprovalPolicy, HostEnvSpec, NamedHostEnv, PodAllow, PodLimits, ThreadDefaults,
    };

    fn sample_config() -> PodConfig {
        PodConfig {
            name: "whisper-agent dev".into(),
            description: Some("Working on whisper-agent itself".into()),
            created_at: "2026-04-16T10:23:11Z".into(),
            allow: PodAllow {
                backends: vec!["anthropic".into(), "openai-compat".into()],
                mcp_hosts: vec!["fetch".into(), "search".into()],
                host_env: vec![
                    NamedHostEnv {
                        name: "landlock-rw".into(),
                        provider: "landlock-laptop".into(),
                        spec: HostEnvSpec::Landlock {
                            allowed_paths: vec![PathAccess::read_write("/home/me/project")],
                            network: NetworkPolicy::Unrestricted,
                        },
                    },
                    NamedHostEnv {
                        name: "landlock-ro".into(),
                        provider: "landlock-laptop".into(),
                        spec: HostEnvSpec::Landlock {
                            allowed_paths: vec![PathAccess::read_only("/home/me/project")],
                            network: NetworkPolicy::Isolated,
                        },
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
                host_env: "landlock-rw".into(),
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
        let text = to_toml(&cfg).unwrap();
        let back = parse_toml(&text).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn detects_duplicate_sandbox_name() {
        let mut cfg = sample_config();
        cfg.allow.host_env.push(NamedHostEnv {
            name: "landlock-rw".into(),
            provider: "landlock-laptop".into(),
            spec: HostEnvSpec::Landlock {
                allowed_paths: vec![],
                network: NetworkPolicy::Unrestricted,
            },
        });
        let err = validate(&cfg).unwrap_err();
        assert!(matches!(err, PodConfigError::DuplicateSandboxName(_)));
    }

    #[test]
    fn rejects_unknown_default_backend() {
        let mut cfg = sample_config();
        cfg.thread_defaults.backend = "not-a-backend".into();
        let err = validate(&cfg).unwrap_err();
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
        cfg.thread_defaults.host_env = "phantom".into();
        let err = validate(&cfg).unwrap_err();
        match err {
            PodConfigError::UnknownThreadDefault { field, .. } => assert_eq!(field, "host_env"),
            other => panic!("expected UnknownThreadDefault, got {other:?}"),
        }
    }

    #[test]
    fn empty_default_sandbox_ok_when_allow_is_empty() {
        let mut cfg = sample_config();
        cfg.allow.host_env.clear();
        cfg.thread_defaults.host_env = String::new();
        validate(&cfg).unwrap();
    }

    #[test]
    fn rejects_empty_default_when_allow_has_entries() {
        let mut cfg = sample_config();
        // sample_config has entries in allow.host_env; empty defaults
        // is rejected so threads can't silently land on "no host env"
        // when the pod actually offers some.
        cfg.thread_defaults.host_env = String::new();
        let err = validate(&cfg).unwrap_err();
        assert!(
            matches!(err, PodConfigError::HostEnvDefaultRequired { .. }),
            "expected HostEnvDefaultRequired, got {err:?}"
        );
    }

    #[test]
    fn rejects_unknown_default_mcp_host() {
        let mut cfg = sample_config();
        cfg.thread_defaults.mcp_hosts.push("phantom".into());
        let err = validate(&cfg).unwrap_err();
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
        // A pod with no host_env entries is valid — threads inside it
        // run with no host-env MCP, only shared MCP hosts.
        let text = r#"
name = "minimal"
created_at = "2026-04-16T10:00:00Z"

[allow]
backends = ["anthropic"]
mcp_hosts = []
host_env = []

[thread_defaults]
backend = "anthropic"
model = "claude-opus-4-7"
system_prompt_file = "system_prompt.md"
max_tokens = 8000
max_turns = 50
"#;
        let cfg = parse_toml(text).unwrap();
        assert_eq!(cfg.name, "minimal");
        assert_eq!(cfg.thread_defaults.max_tokens, 8000);
        // limits omitted → default 10
        assert_eq!(cfg.limits.max_concurrent_threads, 10);
    }

    #[test]
    fn unparseable_toml_returns_parse_error() {
        let err = parse_toml("not = valid {toml").unwrap_err();
        assert!(matches!(err, PodConfigError::Parse(_)));
    }
}
