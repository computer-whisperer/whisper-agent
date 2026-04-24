//! Server-side configuration loaded from a TOML file.
//!
//! The file describes the catalog of model backends the server is willing to drive.
//! Each entry is a named alias (e.g. `anthropic-prod`, `local-ollama`) pointing at a
//! backend kind plus any credentials / endpoint details that kind needs. Tasks refer
//! to entries by alias; the scheduler never touches backend-specific config directly.
//!
//! Credentials live in a structured `auth` field per entry. The shape is a tagged
//! union keyed by `mode`; today the only mode is `api_key`, which accepts either an
//! inline `value` or an `env` var name:
//!
//! ```toml
//! [backends.anthropic]
//! kind = "anthropic"
//! default_model = "claude-sonnet-4-6"
//! auth = { mode = "api_key", value = "sk-ant-..." }
//!
//! [backends.local-ollama]
//! kind = "openai_chat"
//! base_url = "http://localhost:11434/v1"
//! default_model = "llama3.2"
//! # no `auth` — local endpoints don't require credentials
//!
//! [backends.openai]
//! kind = "openai_chat"
//! base_url = "https://api.openai.com/v1"
//! default_model = "gpt-4o"
//! auth = { mode = "api_key", env = "OPENAI_API_KEY" }
//! ```

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;

use std::path::PathBuf;

use crate::providers::anthropic::AnthropicClient;
use crate::providers::codex_auth::CodexAuth;
use crate::providers::gemini::{GEMINI_API_BASE, GEMINI_CODE_ASSIST_BASE, GeminiClient};
use crate::providers::gemini_auth::GeminiAuth;
use crate::providers::llamacpp::LlamaCppClient;
use crate::providers::model::ModelProvider;
use crate::providers::openai_chat::OpenAiChatClient;
use crate::providers::openai_responses::{
    CHATGPT_CODEX_BASE, OPENAI_API_BASE, OpenAiResponsesClient,
};

#[derive(Deserialize, Debug, Clone)]
pub struct Config {
    #[serde(default)]
    pub backends: BTreeMap<String, BackendConfig>,
    /// Optional `[shared_mcp_hosts]` table mapping name → url. CLI-provided
    /// `--shared-mcp-host` flags are merged with these (CLI wins on name
    /// conflict).
    #[serde(default)]
    pub shared_mcp_hosts: BTreeMap<String, String>,
    /// Optional `[[host_env_providers]]` table — list of named daemon
    /// URLs the scheduler can dispatch host-env provisioning to.
    /// Empty is valid (no host envs provisionable → threads run with
    /// only shared MCP tools). CLI `--host-env-provider name=url`
    /// flags are merged with these.
    #[serde(default)]
    pub host_env_providers: Vec<HostEnvProviderConfig>,
    /// Ancillary secrets consumed by sibling daemons (whisper-agent-mcp-search,
    /// etc.) rather than the server process itself. Keys are env-var names,
    /// values are the secret strings; `whisper-agent config env` emits these
    /// as shell `export` lines for dev scripts to source.
    #[serde(default)]
    pub secrets: BTreeMap<String, String>,
    /// Per-client bearer tokens for non-loopback HTTP/WS access. Loopback
    /// connections always bypass auth. When this list is empty, non-loopback
    /// connections are rejected outright.
    #[serde(default)]
    pub auth: AuthConfig,
}

/// Client-auth section. Tokens are stored in cleartext alongside other
/// credentials in `whisper-agent.toml` (e.g. provider API keys). The server
/// presents these to bearer-token comparisons (`Authorization: Bearer <tok>`
/// for native clients, `wa_session=<tok>` cookie for browser).
///
/// Two token lists: `[[auth.clients]]` for everyday chat access, and
/// `[[auth.admins]]` for settings mutations (rotate provider
/// credentials, etc.). An admin token also grants client access —
/// the handlers check "is admin" as a capability on top of the base
/// auth match.
#[derive(Deserialize, Debug, Clone, Default, PartialEq, Eq)]
pub struct AuthConfig {
    #[serde(default)]
    pub clients: Vec<AuthClient>,
    #[serde(default)]
    pub admins: Vec<AuthClient>,
}

/// One entry from `[[auth.clients]]` or `[[auth.admins]]`. `name` is
/// descriptive (used to identify a device for revocation by editing
/// the file); `token` is the opaque bearer string. Suggested
/// generator: `openssl rand -base64 32`.
#[derive(Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct AuthClient {
    pub name: String,
    pub token: String,
}

/// One catalog entry from `[[host_env_providers]]`. `name` is what
/// pod entries reference; `url` is the daemon endpoint; `token_file`
/// (optional) points at a file whose single-line contents are the
/// pre-shared bearer the daemon's `--control-token-file` loaded.
/// Future security fields (`tls = { ca = "...", client_cert = "..." }`,
/// etc.) plug in here without breaking compat — TOML deserialize
/// ignores unknown keys by default.
#[derive(Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct HostEnvProviderConfig {
    pub name: String,
    pub url: String,
    #[serde(default)]
    pub token_file: Option<std::path::PathBuf>,
}

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
    ChatgptSubscription {
        source: ChatgptSubscriptionSource,
        /// Override the default file location for the chosen `source`.
        #[serde(default)]
        path: Option<PathBuf>,
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

#[derive(Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BackendConfig {
    Anthropic {
        auth: Auth,
        #[serde(default)]
        default_model: Option<String>,
    },
    // serde's snake_case would default this to "open_ai_chat" — override so
    // the config file uses the tighter form.
    #[serde(rename = "openai_chat")]
    OpenAiChat {
        base_url: String,
        /// Optional — local endpoints (Ollama, LM Studio, llama.cpp) skip auth entirely.
        #[serde(default)]
        auth: Option<Auth>,
        #[serde(default)]
        default_model: Option<String>,
    },
    #[serde(rename = "openai_responses")]
    OpenAiResponses {
        /// Optional override; defaults to `https://api.openai.com/v1`. Lets
        /// follow-up stages point the same provider at
        /// `https://chatgpt.com/backend-api/codex` for subscription auth, or
        /// at an internal proxy.
        #[serde(default)]
        base_url: Option<String>,
        auth: Auth,
        #[serde(default)]
        default_model: Option<String>,
    },
    /// Google Gemini via the native `generateContent` API. API-key auth today;
    /// gemini-cli OAuth auth follows.
    Gemini {
        /// Optional override; defaults to `https://generativelanguage.googleapis.com/v1beta`.
        #[serde(default)]
        base_url: Option<String>,
        auth: Auth,
        #[serde(default)]
        default_model: Option<String>,
    },
    /// llama.cpp's native HTTP server. Uses the same `/v1/chat/completions`
    /// wire shape as the generic `openai_chat` kind for the actual request,
    /// but additionally polls llama.cpp's `/slots` endpoint during prefill
    /// to emit real prefill-progress events — the only backend here that
    /// can give UIs an honest "X of Y tokens ingested" progress bar.
    /// Room for llama.cpp-specific extensions down the road
    /// (GBNF-constrained tool calls, cache_prompt tuning, /tokenize, slot
    /// save/restore) without polluting `openai_chat`.
    #[serde(rename = "llamacpp")]
    LlamaCpp {
        /// Server origin, e.g. `http://localhost:8080`. The driver appends
        /// `/v1/chat/completions` and `/slots` itself — do NOT include a
        /// `/v1` suffix here.
        base_url: String,
        #[serde(default)]
        default_model: Option<String>,
    },
}

impl BackendConfig {
    pub fn kind(&self) -> &'static str {
        match self {
            BackendConfig::Anthropic { .. } => "anthropic",
            BackendConfig::OpenAiChat { .. } => "openai_chat",
            BackendConfig::OpenAiResponses { .. } => "openai_responses",
            BackendConfig::Gemini { .. } => "gemini",
            BackendConfig::LlamaCpp { .. } => "llamacpp",
        }
    }

    /// The backend's default model, if one is configured. Local llama.cpp endpoints
    /// typically only serve one model and ignore the field, so the TOML can omit it.
    pub fn default_model(&self) -> Option<&str> {
        match self {
            BackendConfig::Anthropic { default_model, .. }
            | BackendConfig::OpenAiChat { default_model, .. }
            | BackendConfig::OpenAiResponses { default_model, .. }
            | BackendConfig::Gemini { default_model, .. }
            | BackendConfig::LlamaCpp { default_model, .. } => default_model.as_deref(),
        }
    }

    /// Auth mode tag for this backend, or `None` if the backend is
    /// configured without auth (e.g. a local openai-compatible endpoint).
    pub fn auth_mode(&self) -> Option<&'static str> {
        match self {
            BackendConfig::Anthropic { auth, .. }
            | BackendConfig::OpenAiResponses { auth, .. }
            | BackendConfig::Gemini { auth, .. } => Some(auth.mode_name()),
            BackendConfig::OpenAiChat { auth, .. } => auth.as_ref().map(Auth::mode_name),
            BackendConfig::LlamaCpp { .. } => None,
        }
    }

    /// Resolve auth material and construct the corresponding provider.
    pub fn build(&self) -> Result<Arc<dyn ModelProvider>> {
        match self {
            BackendConfig::Anthropic { auth, .. } => {
                let key = auth.resolve_api_key().context("anthropic auth")?;
                Ok(Arc::new(AnthropicClient::new(key)))
            }
            BackendConfig::OpenAiChat { base_url, auth, .. } => {
                let key = match auth {
                    Some(a) => Some(a.resolve_api_key().context("openai_chat auth")?),
                    None => None,
                };
                Ok(Arc::new(OpenAiChatClient::new(base_url.clone(), key)))
            }
            BackendConfig::OpenAiResponses { base_url, auth, .. } => {
                build_openai_responses(base_url.as_deref(), auth)
            }
            BackendConfig::Gemini { base_url, auth, .. } => build_gemini(base_url.as_deref(), auth),
            BackendConfig::LlamaCpp { base_url, .. } => {
                Ok(Arc::new(LlamaCppClient::new(base_url.clone())))
            }
        }
    }
}

fn build_gemini(base_url: Option<&str>, auth: &Auth) -> Result<Arc<dyn ModelProvider>> {
    match auth {
        Auth::ApiKey { .. } => {
            let key = auth.resolve_api_key().context("gemini auth")?;
            let url = base_url
                .map(|s| s.to_string())
                .unwrap_or_else(|| GEMINI_API_BASE.to_string());
            Ok(Arc::new(GeminiClient::with_api_key(url, key)))
        }
        Auth::GoogleOauth { source, path } => {
            let GoogleOauthSource::GeminiCli = source;
            let gemini_auth = GeminiAuth::load(path.clone())
                .context("load gemini oauth_creds.json for gemini-cli auth")?;
            let url = base_url
                .map(|s| s.to_string())
                .unwrap_or_else(|| GEMINI_CODE_ASSIST_BASE.to_string());
            Ok(Arc::new(GeminiClient::with_gemini_cli_auth(
                url,
                gemini_auth,
            )))
        }
        Auth::ChatgptSubscription { .. } => Err(anyhow!(
            "gemini: auth.mode `chatgpt_subscription` not supported"
        )),
    }
}

fn build_openai_responses(base_url: Option<&str>, auth: &Auth) -> Result<Arc<dyn ModelProvider>> {
    match auth {
        Auth::ApiKey { .. } => {
            let key = auth.resolve_api_key().context("openai_responses auth")?;
            let url = base_url
                .map(|s| s.to_string())
                .unwrap_or_else(|| OPENAI_API_BASE.to_string());
            Ok(Arc::new(OpenAiResponsesClient::with_api_key(url, key)))
        }
        Auth::ChatgptSubscription { source, path } => {
            let ChatgptSubscriptionSource::Codex = source;
            let codex = CodexAuth::load(path.clone())
                .context("load codex auth.json for openai_responses subscription auth")?;
            let url = base_url
                .map(|s| s.to_string())
                .unwrap_or_else(|| CHATGPT_CODEX_BASE.to_string());
            Ok(Arc::new(OpenAiResponsesClient::with_codex_auth(url, codex)))
        }
        Auth::GoogleOauth { .. } => Err(anyhow!(
            "openai_responses: auth.mode `google_oauth` not supported"
        )),
    }
}

impl Config {
    pub async fn load(path: &Path) -> Result<Self> {
        let text = tokio::fs::read_to_string(path)
            .await
            .with_context(|| format!("read config {}", path.display()))?;
        let config: Config =
            toml::from_str(&text).with_context(|| format!("parse config {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.backends.is_empty() {
            return Err(anyhow!("config: no backends defined"));
        }
        let mut names: std::collections::HashSet<&str> = std::collections::HashSet::new();
        let mut tokens: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for client in &self.auth.clients {
            let name = client.name.trim();
            if name.is_empty() {
                return Err(anyhow!("config: auth.clients entry has empty name"));
            }
            if client.token.is_empty() {
                return Err(anyhow!(
                    "config: auth.clients entry `{name}` has empty token"
                ));
            }
            if !names.insert(name) {
                return Err(anyhow!("config: auth.clients has duplicate name `{name}`"));
            }
            if !tokens.insert(&client.token) {
                return Err(anyhow!(
                    "config: auth.clients entry `{name}` reuses a token from another entry"
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
    fn parses_api_key_variants() {
        let text = r#"
[backends.cloud]
kind = "anthropic"
default_model = "claude-sonnet-4-6"
auth = { mode = "api_key", value = "sk-test" }

[backends.cloud-env]
kind = "openai_chat"
base_url = "https://api.openai.com/v1"
auth = { mode = "api_key", env = "OPENAI_API_KEY" }

[backends.local]
kind = "openai_chat"
base_url = "http://localhost:11434/v1"

[backends.local-llamacpp]
kind = "llamacpp"
base_url = "http://localhost:8080"
default_model = "qwen2.5-coder-7b"
"#;
        let cfg: Config = toml::from_str(text).unwrap();
        cfg.validate().unwrap();

        match cfg.backends.get("cloud").unwrap() {
            BackendConfig::Anthropic {
                auth: Auth::ApiKey { value, env },
                ..
            } => {
                assert_eq!(value.as_deref(), Some("sk-test"));
                assert!(env.is_none());
            }
            _ => panic!("wrong variant"),
        }

        match cfg.backends.get("cloud-env").unwrap() {
            BackendConfig::OpenAiChat {
                auth: Some(Auth::ApiKey { env, .. }),
                ..
            } => assert_eq!(env.as_deref(), Some("OPENAI_API_KEY")),
            _ => panic!("wrong variant"),
        }

        match cfg.backends.get("local").unwrap() {
            BackendConfig::OpenAiChat { auth, .. } => assert!(auth.is_none()),
            _ => panic!("wrong variant"),
        }

        match cfg.backends.get("local-llamacpp").unwrap() {
            BackendConfig::LlamaCpp {
                base_url,
                default_model,
            } => {
                assert_eq!(base_url, "http://localhost:8080");
                assert_eq!(default_model.as_deref(), Some("qwen2.5-coder-7b"));
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn parses_auth_clients_section() {
        let text = r#"
[backends.cloud]
kind = "anthropic"
auth = { mode = "api_key", value = "sk-test" }

[[auth.clients]]
name = "desktop-laptop"
token = "wsk_aaa"

[[auth.clients]]
name = "phone"
token = "wsk_bbb"
"#;
        let cfg: Config = toml::from_str(text).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.auth.clients.len(), 2);
        assert_eq!(cfg.auth.clients[0].name, "desktop-laptop");
        assert_eq!(cfg.auth.clients[1].token, "wsk_bbb");
    }

    #[test]
    fn auth_clients_section_is_optional() {
        let text = r#"
[backends.cloud]
kind = "anthropic"
auth = { mode = "api_key", value = "sk-test" }
"#;
        let cfg: Config = toml::from_str(text).unwrap();
        cfg.validate().unwrap();
        assert!(cfg.auth.clients.is_empty());
    }

    #[test]
    fn auth_clients_validation_rejects_collisions() {
        let base = r#"
[backends.cloud]
kind = "anthropic"
auth = { mode = "api_key", value = "sk-test" }
"#;

        let dup_name = format!(
            "{base}\n[[auth.clients]]\nname = \"x\"\ntoken = \"a\"\n[[auth.clients]]\nname = \"x\"\ntoken = \"b\"\n"
        );
        assert!(
            toml::from_str::<Config>(&dup_name)
                .unwrap()
                .validate()
                .is_err()
        );

        let dup_token = format!(
            "{base}\n[[auth.clients]]\nname = \"x\"\ntoken = \"a\"\n[[auth.clients]]\nname = \"y\"\ntoken = \"a\"\n"
        );
        assert!(
            toml::from_str::<Config>(&dup_token)
                .unwrap()
                .validate()
                .is_err()
        );

        let empty_name = format!("{base}\n[[auth.clients]]\nname = \"\"\ntoken = \"a\"\n");
        assert!(
            toml::from_str::<Config>(&empty_name)
                .unwrap()
                .validate()
                .is_err()
        );

        let empty_token = format!("{base}\n[[auth.clients]]\nname = \"x\"\ntoken = \"\"\n");
        assert!(
            toml::from_str::<Config>(&empty_token)
                .unwrap()
                .validate()
                .is_err()
        );
    }

    #[test]
    fn api_key_requires_exactly_one_source() {
        // value only
        Auth::ApiKey {
            value: Some("x".into()),
            env: None,
        }
        .resolve_api_key()
        .unwrap();
        // both set → error
        assert!(
            Auth::ApiKey {
                value: Some("x".into()),
                env: Some("Y".into()),
            }
            .resolve_api_key()
            .is_err()
        );
        // neither set → error
        assert!(
            Auth::ApiKey {
                value: None,
                env: None,
            }
            .resolve_api_key()
            .is_err()
        );
    }
}
