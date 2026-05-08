//! Minimal view of `whisper-agent.toml` — we only deserialize the parts we
//! need: the named backend's auth, base URL, and default model. Other
//! sections (`[shared_mcp_hosts]`, `[secrets]`, `[auth]`, etc.) are silently
//! ignored.
//!
//! Defining this here rather than depending on `whisper-agent` as a library
//! keeps the dep graph small — pulling in the full main crate would drag
//! anthropic / gemini / llama.cpp / scheduler with us.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use reqwest::Client;
use serde::Deserialize;
use tracing::warn;
use whisper_agent_auth::{Auth, ClientAuth};

use crate::codex;

/// Top-level subset of `whisper-agent.toml`.
#[derive(Deserialize, Debug)]
struct ConfigView {
    #[serde(default)]
    backends: BTreeMap<String, BackendView>,
}

/// One `[backends.X]` entry. We don't model the full `BackendConfig` enum
/// — we accept any kind on parse and only assert `openai_responses` at lookup
/// time. `auth` is `Option` because some backend kinds (local llama.cpp,
/// unauth'd Ollama) omit it.
#[derive(Deserialize, Debug)]
struct BackendView {
    kind: String,
    #[serde(default)]
    base_url: Option<String>,
    #[serde(default)]
    auth: Option<Auth>,
    #[serde(default)]
    default_model: Option<String>,
}

/// Material derived from the named backend, ready for use in image-gen
/// requests. Hot-swapped wholesale on config reload — an in-flight request
/// holds the old `Arc<Resolved>` and finishes against it; the next request
/// picks up the swapped-in successor.
pub struct Resolved {
    pub auth: ClientAuth,
    pub api_base: String,
    /// Image model name passed to the api_key path's
    /// `/v1/images/generations` request body. Hardcoded default since
    /// the backend's `default_model` is the chat model for ChatGPT
    /// flows, not an image model.
    pub image_model: String,
    /// Chat model used to drive the Codex path's `/responses` request
    /// when the model invokes the `image_generation` built-in tool.
    /// Sourced from the backend's `default_model`, with a fallback for
    /// older configs that don't set one.
    pub chat_model: String,
}

/// Default base URL when the backend doesn't override `base_url` and
/// auth is `api_key`.
const OPENAI_API_BASE: &str = "https://api.openai.com/v1";

/// Default base URL when the backend doesn't override `base_url` and
/// auth is `chatgpt_subscription` (Codex). Mirrors the main daemon's
/// behavior — the ChatGPT route serves its own host.
const CHATGPT_CODEX_BASE: &str = "https://chatgpt.com/backend-api/codex";

/// Image model used by the api_key path's `/v1/images/generations` request
/// when the tool call doesn't override it. `gpt-image-2` is OpenAI's latest
/// as of April 2026.
const DEFAULT_IMAGE_MODEL: &str = "gpt-image-2";

/// Chat model used by the Codex path's `/responses` request when the
/// backend's `default_model` is unset. Picked to be a current model that
/// supports the `image_generation` built-in tool.
const FALLBACK_CHAT_MODEL: &str = "gpt-5";

/// Read the config file from disk and resolve the named backend into
/// runtime auth material. Errors if the file is missing/unparseable, the
/// backend isn't found, the kind isn't `openai_responses`, or auth resolution
/// fails (e.g. unset env var, missing `~/.codex/auth.json`).
pub fn resolve(config_path: &Path, backend_name: &str) -> Result<Resolved> {
    let text = std::fs::read_to_string(config_path)
        .with_context(|| format!("read config {}", config_path.display()))?;
    let view: ConfigView =
        toml::from_str(&text).with_context(|| format!("parse config {}", config_path.display()))?;
    let backend = view.backends.get(backend_name).ok_or_else(|| {
        let available = if view.backends.is_empty() {
            "(none)".to_string()
        } else {
            view.backends.keys().cloned().collect::<Vec<_>>().join(", ")
        };
        anyhow!(
            "backend `{backend_name}` not found in {}; available: {available}",
            config_path.display()
        )
    })?;
    if backend.kind != "openai_responses" {
        return Err(anyhow!(
            "backend `{backend_name}` has kind `{}`; image-gen requires kind `openai_responses`",
            backend.kind
        ));
    }
    let auth = backend
        .auth
        .as_ref()
        .ok_or_else(|| anyhow!("backend `{backend_name}` has no `auth` block"))?;
    let client_auth =
        ClientAuth::from_config(auth).with_context(|| format!("backend `{backend_name}` auth"))?;
    let api_base = backend
        .base_url
        .clone()
        .unwrap_or_else(|| match auth {
            Auth::ChatgptSubscription { .. } => CHATGPT_CODEX_BASE.to_string(),
            _ => OPENAI_API_BASE.to_string(),
        })
        .trim_end_matches('/')
        .to_string();
    let chat_model = backend
        .default_model
        .clone()
        .unwrap_or_else(|| FALLBACK_CHAT_MODEL.to_string());
    Ok(Resolved {
        auth: client_auth,
        api_base,
        image_model: DEFAULT_IMAGE_MODEL.to_string(),
        chat_model,
    })
}

/// On the Codex auth path, replace `resolved.chat_model` with whatever the
/// upstream `/models` endpoint considers current — `gpt-5` (and friends)
/// rotate quickly, and a stale default is the most common cause of HTTP
/// 400 from the subscription host. Discovery failures are logged and the
/// resolve-time value is left in place so the daemon stays up.
pub async fn refresh_chat_model(http: &Client, resolved: &mut Resolved) {
    if !matches!(resolved.auth, ClientAuth::Codex(_)) {
        return;
    }
    match codex::discover_chat_model(http, &resolved.auth, &resolved.api_base).await {
        Ok(slug) => {
            tracing::info!(model = %slug, "discovered chat model from /models endpoint");
            resolved.chat_model = slug;
        }
        Err(e) => {
            warn!(
                error = %e,
                fallback = %resolved.chat_model,
                "model discovery failed; using fallback"
            );
        }
    }
}

/// Discover the config path the same way the main `whisper-agent` daemon
/// does: explicit `--config` wins; otherwise check `$XDG_CONFIG_HOME`,
/// then `$HOME/.config`, then `./whisper-agent.toml`. Returns the first
/// path that exists.
pub fn discover_config_path(explicit: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        if !p.exists() {
            return Err(anyhow!("config file not found: {}", p.display()));
        }
        return Ok(p);
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
        candidates.push(PathBuf::from(xdg).join("whisper-agent/whisper-agent.toml"));
    }
    if let Ok(home) = std::env::var("HOME") {
        candidates.push(PathBuf::from(home).join(".config/whisper-agent/whisper-agent.toml"));
    }
    candidates.push(PathBuf::from("whisper-agent.toml"));

    for path in &candidates {
        if path.exists() {
            return Ok(path.clone());
        }
    }
    Err(anyhow!(
        "could not locate whisper-agent.toml (searched: {})",
        candidates
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    ))
}
