//! Pod TOML parsing, validation, and on-disk constants.
//!
//! The data types themselves (`PodConfig`, `PodAllow`, `NamedSandboxSpec`,
//! `ThreadDefaults`, `PodLimits`) live in the protocol crate so the webui
//! can render them directly from wire snapshots. This module owns the
//! server-side concerns: filename constants, the TOML round-trip, and
//! structural validation.
//!
//! See `docs/design_pod_thread_scheduler.md` for the full design.

use std::collections::BTreeSet;

use thiserror::Error;
use whisper_agent_protocol::PodConfig;

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
    for nss in &config.allow.sandbox {
        if !seen.insert(nss.name.as_str()) {
            return Err(PodConfigError::DuplicateSandboxName(nss.name.clone()));
        }
    }

    if !config.allow.backends.iter().any(|b| b == &config.thread_defaults.backend) {
        return Err(PodConfigError::UnknownThreadDefault {
            field: "backend",
            value: config.thread_defaults.backend.clone(),
            valid: config.allow.backends.join(", "),
        });
    }

    // Sandbox: empty is valid only if pod has no allowed sandboxes (degenerate
    // no-isolation pod). Otherwise must name an entry in [[allow.sandbox]].
    if !config.thread_defaults.sandbox.is_empty()
        && !config
            .allow
            .sandbox
            .iter()
            .any(|nss| nss.name == config.thread_defaults.sandbox)
    {
        let valid = config
            .allow
            .sandbox
            .iter()
            .map(|nss| nss.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(PodConfigError::UnknownThreadDefault {
            field: "sandbox",
            value: config.thread_defaults.sandbox.clone(),
            valid,
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
    use whisper_agent_protocol::sandbox::{AccessMode, NetworkPolicy, PathAccess};
    use whisper_agent_protocol::{
        ApprovalPolicy, NamedSandboxSpec, PodAllow, PodLimits, SandboxSpec, ThreadDefaults,
    };

    fn sample_config() -> PodConfig {
        PodConfig {
            name: "whisper-agent dev".into(),
            description: Some("Working on whisper-agent itself".into()),
            created_at: "2026-04-16T10:23:11Z".into(),
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
        let text = to_toml(&cfg).unwrap();
        let back = parse_toml(&text).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn detects_duplicate_sandbox_name() {
        let mut cfg = sample_config();
        cfg.allow.sandbox.push(NamedSandboxSpec {
            name: "landlock-rw".into(),
            spec: SandboxSpec::None,
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
        cfg.thread_defaults.sandbox = "phantom".into();
        let err = validate(&cfg).unwrap_err();
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
        validate(&cfg).unwrap();
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
