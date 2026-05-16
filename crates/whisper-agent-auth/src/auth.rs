//! Static auth shape from `[backends.X].auth` in `whisper-agent.toml`.

use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;

/// Backend auth configuration. Tagged by `mode`; providers pick the variants
/// they accept and error at build time on incompatible combinations (e.g.
/// Anthropic rejects `chatgpt_subscription`).
///
/// Examples:
/// ```toml
/// auth = { mode = "api_key", value = "sk-..." }
/// auth = { mode = "api_key", env  = "ANTHROPIC_API_KEY" }
/// auth = { mode = "chatgpt_subscription", source = "codex" }
/// auth = { mode = "chatgpt_subscription", source = "codex", path = "/custom/auth.json" }
/// auth = { mode = "chatgpt_subscription", source = "codex", manager = "laptop-daemon" }
/// ```
#[derive(Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum Auth {
    ApiKey {
        #[serde(default)]
        value: Option<String>,
        #[serde(default)]
        env: Option<String>,
    },
    /// Use a ChatGPT subscription's OAuth tokens instead of an API key. Today
    /// the only `source` is `codex` — we read the credentials file that the
    /// Codex CLI maintains at `~/.codex/auth.json`.
    ///
    /// When `manager` names a v2 host-env daemon (one admitted via
    /// `[[auth.daemons]]`), this backend's credential file is owned by
    /// that daemon: the daemon keeps it fresh and pushes updates over
    /// the host-env link. The server starts even if the file is absent
    /// at boot — requests against the backend fail with a clear
    /// "awaiting publication" message until the first publish arrives.
    /// Locally-administered backends omit `manager` and behave as
    /// before (file required at startup; admin-paste rotation only).
    ChatgptSubscription {
        source: ChatgptSubscriptionSource,
        /// Override the default file location for the chosen `source`.
        #[serde(default)]
        path: Option<PathBuf>,
        /// Daemon allowed to publish refreshes for this backend's
        /// credential file. Must match a name in `[[auth.daemons]]`.
        #[serde(default)]
        manager: Option<String>,
    },
    /// Use Google OAuth tokens minted by a companion tool. Today the only
    /// `source` is `gemini_cli` — we read `~/.gemini/oauth_creds.json`.
    GoogleOauth {
        source: GoogleOauthSource,
        #[serde(default)]
        path: Option<PathBuf>,
    },
}

#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ChatgptSubscriptionSource {
    /// `~/.codex/auth.json`, maintained by `codex login`.
    Codex,
}

#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GoogleOauthSource {
    /// `~/.gemini/oauth_creds.json`, maintained by `gemini` interactive login.
    GeminiCli,
}

impl Auth {
    /// Stable tag matching the serde `mode` string. Used by the settings
    /// panel to display which credential slot a backend uses without
    /// touching the credential itself.
    pub fn mode_name(&self) -> &'static str {
        match self {
            Auth::ApiKey { .. } => "api_key",
            Auth::ChatgptSubscription { .. } => "chatgpt_subscription",
            Auth::GoogleOauth { .. } => "google_oauth",
        }
    }

    /// Name of the daemon authorized to manage this credential, if
    /// any. Today only the `chatgpt_subscription` variant carries a
    /// manager; future credential families (Google OAuth, etc.) will
    /// extend this with their own fields.
    pub fn credential_manager(&self) -> Option<&str> {
        match self {
            Auth::ChatgptSubscription { manager, .. } => manager.as_deref(),
            Auth::ApiKey { .. } | Auth::GoogleOauth { .. } => None,
        }
    }

    /// Resolve to the raw API key string, reading env vars as needed. Errors if the
    /// auth entry is malformed (neither or both of `value`/`env` set, env var unset),
    /// or if the auth mode isn't `api_key`.
    pub fn resolve_api_key(&self) -> Result<String> {
        match self {
            Auth::ApiKey { value, env } => match (value.as_deref(), env.as_deref()) {
                (Some(v), None) => Ok(v.to_string()),
                (None, Some(var)) => {
                    std::env::var(var).with_context(|| format!("env var `{var}` not set"))
                }
                (Some(_), Some(_)) => {
                    Err(anyhow!("auth.api_key: set exactly one of `value` or `env`"))
                }
                (None, None) => Err(anyhow!("auth.api_key: missing `value` or `env`")),
            },
            Auth::ChatgptSubscription { .. } | Auth::GoogleOauth { .. } => {
                Err(anyhow!("this backend requires `auth.mode = \"api_key\"`"))
            }
        }
    }
}
