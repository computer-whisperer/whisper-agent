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
//! - [`behaviors`] — per-pod autonomous-behavior catalog: parsing,
//!   validation, on-disk loader (see `docs/design_behaviors.md`).
//! - [`config`] — server-side catalog config (the TOML that lists model
//!   backends the server is willing to drive).
//! - [`persist`] — pod/thread on-disk I/O (read, write, list, rename).
//! - [`resources`] — registry of sandboxes, MCP hosts, and model backends
//!   as first-class entities with lifecycles independent of tasks.

pub mod behaviors;
pub mod config;
pub mod fs;
pub mod persist;
pub mod resources;

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use thiserror::Error;
use whisper_agent_protocol::{PodConfig, PodState};

use crate::pod::behaviors::{Behavior, BehaviorId};

pub type PodId = String;
pub type ThreadId = String;

/// Filename of the per-pod config inside the pod's directory.
pub const POD_TOML: &str = "pod.toml";
/// Subdirectory under the pod that holds per-thread JSON files.
pub const THREADS_DIR: &str = "threads";
/// Filename of the per-pod operational state (pause flag, etc.).
/// Kept out of `pod.toml` so pause/resume doesn't touch
/// version-controlled config.
pub const POD_STATE_JSON: &str = "pod_state.json";

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
    /// Behavior catalog loaded from `<pod>/behaviors/`. Empty when the pod
    /// has no behaviors subdirectory. Keyed by behavior id (== directory
    /// name); sorted iteration matches on-disk lexical order.
    pub behaviors: BTreeMap<BehaviorId, Behavior>,
    /// Operational state persisted to `<pod>/pod_state.json`. Defaults
    /// to enabled when the file is absent.
    pub state: PodState,
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
            behaviors: BTreeMap::new(),
            state: PodState::default(),
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
    #[error("invalid [[allow.host_env]] name `{name}`: {reason}")]
    InvalidHostEnvName { name: String, reason: String },
    #[error("thread_defaults.caps.{field} = {default:?} exceeds allow.caps.{field} = {ceiling:?}")]
    CapsDefaultExceedsCeiling {
        field: &'static str,
        default: String,
        ceiling: String,
    },
}

/// Validate a `[[allow.host_env]]` entry's name. Names are used as tool-
/// name prefixes presented to the model (`{env_name}_{tool_name}`), so
/// they must (a) stay ambiguous-free — no embedded underscore, since
/// the separator is a single underscore — and (b) not shadow a builtin
/// tool's own prefix (see [`builtin_tools::reserved_env_name_prefixes`]).
pub fn validate_host_env_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("name must be non-empty".into());
    }
    if name.len() > 32 {
        return Err(format!("name is {} chars (max 32)", name.len()));
    }
    let first = name.chars().next().expect("non-empty");
    if !(first.is_ascii_lowercase() || first.is_ascii_digit()) {
        return Err("name must start with a lowercase ASCII letter or digit".into());
    }
    for ch in name.chars() {
        if !(ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-') {
            return Err(format!(
                "name contains `{ch}`; only lowercase a-z, 0-9, and `-` are allowed \
                 (underscore is the prefix separator in `{{name}}_{{tool}}` — bare names cannot contain it)"
            ));
        }
    }
    let reserved = crate::tools::builtin_tools::reserved_env_name_prefixes();
    if reserved.contains(&name) {
        return Err(format!(
            "name `{name}` collides with a builtin tool prefix (reserved: [{}])",
            reserved.join(", ")
        ));
    }
    Ok(())
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
        if let Err(reason) = validate_host_env_name(&nss.name) {
            return Err(PodConfigError::InvalidHostEnvName {
                name: nss.name.clone(),
                reason,
            });
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
    } else {
        for name in &config.thread_defaults.host_env {
            if !valid_names.contains(&name.as_str()) {
                return Err(PodConfigError::UnknownThreadDefault {
                    field: "host_env",
                    value: name.clone(),
                    valid: valid_names.join(", "),
                });
            }
        }
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

    // thread_defaults.caps must not exceed allow.caps. Each cap enum
    // is totally ordered; `default <= ceiling` is the check.
    let td = &config.thread_defaults.caps;
    let ac = &config.allow.caps;
    if td.pod_modify > ac.pod_modify {
        return Err(PodConfigError::CapsDefaultExceedsCeiling {
            field: "pod_modify",
            default: format!("{:?}", td.pod_modify),
            ceiling: format!("{:?}", ac.pod_modify),
        });
    }
    if td.dispatch > ac.dispatch {
        return Err(PodConfigError::CapsDefaultExceedsCeiling {
            field: "dispatch",
            default: format!("{:?}", td.dispatch),
            ceiling: format!("{:?}", ac.dispatch),
        });
    }
    if td.behaviors > ac.behaviors {
        return Err(PodConfigError::CapsDefaultExceedsCeiling {
            field: "behaviors",
            default: format!("{:?}", td.behaviors),
            ceiling: format!("{:?}", ac.behaviors),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use whisper_agent_protocol::sandbox::{NetworkPolicy, PathAccess};
    use whisper_agent_protocol::{
        AllowMap, BehaviorOpsCap, DispatchCap, Disposition, HostEnvSpec, NamedHostEnv, PodAllow,
        PodAllowCaps, PodLimits, PodModifyCap, ThreadDefaultCaps, ThreadDefaults,
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
                tools: AllowMap {
                    default: Disposition::Allow,
                    overrides: [("bash".to_string(), Disposition::Deny)]
                        .into_iter()
                        .collect(),
                },
                caps: PodAllowCaps::default(),
            },
            thread_defaults: ThreadDefaults {
                backend: "anthropic".into(),
                model: "claude-opus-4-7".into(),
                system_prompt_file: "system_prompt.md".into(),
                max_tokens: 32000,
                max_turns: 100,
                host_env: vec!["landlock-rw".into()],
                mcp_hosts: vec!["fetch".into(), "search".into()],
                compaction: Default::default(),
                caps: ThreadDefaultCaps::default(),
                tool_surface: Default::default(),
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
        cfg.thread_defaults.host_env = vec!["phantom".into()];
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
        cfg.thread_defaults.host_env.clear();
        validate(&cfg).unwrap();
    }

    #[test]
    fn rejects_empty_default_when_allow_has_entries() {
        let mut cfg = sample_config();
        // sample_config has entries in allow.host_env; empty defaults
        // is rejected so threads can't silently land on "no host env"
        // when the pod actually offers some.
        cfg.thread_defaults.host_env.clear();
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
        // Caps sections omitted → permissive ceiling + baseline defaults.
        assert_eq!(cfg.allow.caps.pod_modify, PodModifyCap::ModifyAllow);
        assert_eq!(cfg.allow.caps.dispatch, DispatchCap::WithinScope);
        assert_eq!(cfg.allow.caps.behaviors, BehaviorOpsCap::AuthorAny);
        assert_eq!(cfg.thread_defaults.caps.pod_modify, PodModifyCap::Memories);
        assert_eq!(cfg.thread_defaults.caps.dispatch, DispatchCap::WithinScope);
        assert_eq!(cfg.thread_defaults.caps.behaviors, BehaviorOpsCap::Read);
    }

    #[test]
    fn parses_pod_with_explicit_caps() {
        let text = r#"
name = "tight"
created_at = "2026-04-21T10:00:00Z"

[allow]
backends = ["anthropic"]
mcp_hosts = []
host_env = []

[allow.caps]
pod_modify = "content"
dispatch = "within_scope"
behaviors = "author_narrower"

[thread_defaults]
backend = "anthropic"
model = "claude-opus-4-7"
system_prompt_file = "system_prompt.md"
max_tokens = 8000
max_turns = 50

[thread_defaults.caps]
pod_modify = "none"
dispatch = "none"
behaviors = "none"
"#;
        let cfg = parse_toml(text).unwrap();
        assert_eq!(cfg.allow.caps.pod_modify, PodModifyCap::Content);
        assert_eq!(cfg.allow.caps.behaviors, BehaviorOpsCap::AuthorNarrower);
        assert_eq!(cfg.thread_defaults.caps.pod_modify, PodModifyCap::None);
        assert_eq!(cfg.thread_defaults.caps.dispatch, DispatchCap::None);
    }

    #[test]
    fn rejects_thread_defaults_caps_above_ceiling() {
        let mut cfg = sample_config();
        cfg.allow.caps.pod_modify = PodModifyCap::Memories;
        cfg.thread_defaults.caps.pod_modify = PodModifyCap::Content;
        let err = validate(&cfg).unwrap_err();
        match err {
            PodConfigError::CapsDefaultExceedsCeiling { field, .. } => {
                assert_eq!(field, "pod_modify");
            }
            other => panic!("expected CapsDefaultExceedsCeiling, got {other:?}"),
        }
    }

    #[test]
    fn legacy_singular_host_env_default_parses_as_one_entry_vec() {
        // Pre-multi-env pod.tomls set `host_env` as a bare string under
        // `[thread_defaults]`. The deserializer must accept that form
        // and wrap it in a one-entry vec so old files load without
        // needing a manual edit.
        let mut cfg = sample_config();
        let text = to_toml(&cfg).unwrap();
        // Swap the new list-shape back to the legacy bare-string form
        // to simulate a file written by the old code.
        let legacy_text =
            text.replace("host_env = [\"landlock-rw\"]", "host_env = \"landlock-rw\"");
        assert_ne!(text, legacy_text, "replacement must have landed");
        let parsed = parse_toml(&legacy_text).unwrap();
        cfg.thread_defaults.host_env = vec!["landlock-rw".into()];
        assert_eq!(
            parsed.thread_defaults.host_env,
            cfg.thread_defaults.host_env
        );
    }

    #[test]
    fn unparseable_toml_returns_parse_error() {
        let err = parse_toml("not = valid {toml").unwrap_err();
        assert!(matches!(err, PodConfigError::Parse(_)));
    }

    #[test]
    fn host_env_names_validate_on_pod_save() {
        let mut cfg = sample_config();
        // Underscore is the tool-name prefix separator — reject so
        // `{env_name}_{tool}` parses unambiguously.
        cfg.allow.host_env[0].name = "my_env".into();
        let err = validate(&cfg).unwrap_err();
        assert!(
            matches!(err, PodConfigError::InvalidHostEnvName { .. }),
            "underscore should be rejected: {err:?}"
        );
        // Uppercase letters are not allowed (lowercase-only convention
        // keeps prefixes terminal-friendly and visually consistent).
        cfg.allow.host_env[0].name = "MyEnv".into();
        assert!(matches!(
            validate(&cfg).unwrap_err(),
            PodConfigError::InvalidHostEnvName { .. }
        ));
        // `pod` is a reserved prefix — collides with the builtin
        // `pod_*` tool namespace.
        cfg.allow.host_env[0].name = "pod".into();
        assert!(matches!(
            validate(&cfg).unwrap_err(),
            PodConfigError::InvalidHostEnvName { .. }
        ));
        // Short lowercase-alphanum-dash is fine.
        cfg.allow.host_env[0].name = "c-dtop".into();
        cfg.thread_defaults.host_env = vec!["c-dtop".into()];
        cfg.allow.host_env[1].name = "c-srv3".into();
        validate(&cfg).unwrap();
    }

    #[test]
    fn reserved_env_name_prefixes_include_pod_and_dispatch() {
        let reserved = crate::tools::builtin_tools::reserved_env_name_prefixes();
        assert!(
            reserved.contains(&"pod"),
            "pod must be reserved: {reserved:?}"
        );
        assert!(
            reserved.contains(&"dispatch"),
            "dispatch must be reserved: {reserved:?}"
        );
    }
}
