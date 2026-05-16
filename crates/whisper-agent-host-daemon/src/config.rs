//! Host-daemon TOML config.
//!
//! Optional file. Today its only role is declaring which credential
//! files the daemon should manage on behalf of named scheduler-side
//! backends — see [`PublishCredentialConfig`]. A daemon without a
//! config file (or with an empty `publish_credential` list) just
//! provides host-env sessions and never sends a
//! [`whisper_agent_host_proto::Frame::PublishCredential`].
//!
//! Path resolution: the binary's `--config` flag (or
//! `WHISPER_AGENT_HOST_DAEMON_CONFIG` env) names the file.
//! Default is `/etc/whisper-agent/host-daemon.toml`, matching the
//! AUR package layout where the systemd unit also reads
//! `/etc/whisper-agent/host-daemon.env`. Missing default file is a
//! soft no-op so the unconfigured daemon keeps working; an
//! explicitly-named path that doesn't exist is an error.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;

/// Top-level daemon config. Every field is optional so future
/// additions don't break existing config files.
#[derive(Deserialize, Debug, Default, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DaemonConfig {
    /// Credential files this daemon will keep fresh and publish to
    /// the matching scheduler-side backend. Empty / absent → daemon
    /// publishes nothing.
    #[serde(default)]
    pub publish_credential: Vec<PublishCredentialConfig>,
}

/// One credential the daemon is responsible for.
///
/// The daemon watches `path` for external rewrites, runs the
/// kind-appropriate refresh loop against the file, and after every
/// change ships the on-disk contents to the scheduler tagged with
/// `backend`. The scheduler matches `backend` against its own
/// `[backends.X]` table and rejects publishes for backends whose
/// declared `auth.manager` doesn't equal the connected daemon's name
/// — pairing is bilateral, so a misconfiguration on either side
/// fails closed rather than silently misrouting tokens.
#[derive(Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PublishCredentialConfig {
    /// Scheduler-side backend name this entry feeds. Must match the
    /// `[backends.<name>]` key on the server *and* that backend's
    /// `auth.manager` must equal this daemon's name.
    pub backend: String,
    /// Credential family. Today only `codex` is supported — the
    /// Codex CLI's `auth.json` shape.
    pub kind: PublishCredentialKind,
    /// On-disk location of the credential file the daemon manages.
    /// Must be readable AND writable by the daemon process — the
    /// refresh loop persists rotated tokens back to this file.
    pub path: PathBuf,
}

#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PublishCredentialKind {
    /// The `~/.codex/auth.json` shape — id_token / access_token /
    /// refresh_token. Refresh is the OAuth refresh-token flow against
    /// `auth.openai.com`; payload sent on the wire is the raw file
    /// contents.
    Codex,
}

impl DaemonConfig {
    /// Load from `path`. Returns `Ok(Default)` if `path` does not
    /// exist and `allow_missing` is set (the default-path case);
    /// otherwise propagates the IO error so an explicitly-named
    /// missing file is surfaced.
    pub fn load(path: &Path, allow_missing: bool) -> Result<Self> {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound && allow_missing => {
                tracing::debug!(path = %path.display(), "no host-daemon config — using defaults");
                return Ok(Self::default());
            }
            Err(e) => {
                return Err(anyhow!(e))
                    .with_context(|| format!("read host-daemon config {}", path.display()));
            }
        };
        let config: Self = toml::from_str(&text)
            .with_context(|| format!("parse host-daemon config {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    /// Reject duplicate `backend` entries — only one publish path
    /// per scheduler-side backend makes sense, and silent
    /// last-wins is the kind of misconfiguration that produces an
    /// authentication mystery weeks later.
    pub fn validate(&self) -> Result<()> {
        let mut seen = std::collections::HashSet::new();
        for entry in &self.publish_credential {
            if !seen.insert(entry.backend.as_str()) {
                return Err(anyhow!(
                    "publish_credential: backend `{}` appears more than once",
                    entry.backend
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_config_parses() {
        let cfg: DaemonConfig = toml::from_str("").unwrap();
        assert!(cfg.publish_credential.is_empty());
    }

    #[test]
    fn single_codex_entry_parses() {
        let cfg: DaemonConfig = toml::from_str(
            r#"
            [[publish_credential]]
            backend = "openai-sub"
            kind = "codex"
            path = "/home/me/.codex/auth.json"
            "#,
        )
        .unwrap();
        assert_eq!(cfg.publish_credential.len(), 1);
        let e = &cfg.publish_credential[0];
        assert_eq!(e.backend, "openai-sub");
        assert_eq!(e.kind, PublishCredentialKind::Codex);
        assert_eq!(e.path, PathBuf::from("/home/me/.codex/auth.json"));
    }

    #[test]
    fn duplicate_backend_is_rejected() {
        let cfg: DaemonConfig = toml::from_str(
            r#"
            [[publish_credential]]
            backend = "openai-sub"
            kind = "codex"
            path = "/a"

            [[publish_credential]]
            backend = "openai-sub"
            kind = "codex"
            path = "/b"
            "#,
        )
        .unwrap();
        let err = cfg.validate().expect_err("dup must reject");
        assert!(
            err.to_string().contains("appears more than once"),
            "got: {err}"
        );
    }

    #[test]
    fn unknown_field_at_top_level_is_rejected() {
        // deny_unknown_fields is load-bearing: catches typos like
        // `publish_credentials` (plural) at parse time rather than
        // silently producing an empty publish list.
        let err = toml::from_str::<DaemonConfig>(
            r#"
            publish_credentials = []
            "#,
        )
        .expect_err("typo must reject");
        assert!(
            err.to_string().contains("publish_credentials"),
            "got: {err}"
        );
    }

    #[test]
    fn missing_default_path_is_ok() {
        let cfg = DaemonConfig::load(Path::new("/definitely/does/not/exist.toml"), true).unwrap();
        assert!(cfg.publish_credential.is_empty());
    }

    #[test]
    fn missing_explicit_path_is_error() {
        let err = DaemonConfig::load(Path::new("/definitely/does/not/exist.toml"), false)
            .expect_err("explicit missing must error");
        assert!(err.to_string().contains("host-daemon config"), "got: {err}");
    }
}
