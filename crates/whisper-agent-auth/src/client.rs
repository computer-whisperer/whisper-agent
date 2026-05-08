//! Live runtime auth: built once per backend session, prepares the bearer
//! token (refreshing first if needed) and any extra HTTP headers (e.g.
//! `chatgpt-account-id` for the Codex subscription path).

use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use tokio::sync::Mutex;

use crate::{Auth, CodexAuth};

/// Runtime auth material for an OpenAI-compatible client. The wire client
/// itself is cheap and shared; `ClientAuth` is what differs between an
/// API-key session and a ChatGPT-subscription session.
pub enum ClientAuth {
    ApiKey(String),
    /// ChatGPT OAuth tokens, typically loaded from `~/.codex/auth.json`.
    /// Wrapped in a mutex because refresh mutates the tokens, and the client
    /// may be called from multiple tasks concurrently.
    Codex(Arc<Mutex<CodexAuth>>),
}

impl ClientAuth {
    /// Build live auth material from a config `Auth` entry. Loads CodexAuth
    /// from disk if needed; resolves `value`/`env` for API keys.
    pub fn from_config(auth: &Auth) -> Result<Self> {
        match auth {
            Auth::ApiKey { .. } => Ok(ClientAuth::ApiKey(
                auth.resolve_api_key().context("resolve api_key")?,
            )),
            Auth::ChatgptSubscription { source: _, path } => {
                let codex = CodexAuth::load(path.clone())
                    .context("load codex auth.json (chatgpt_subscription)")?;
                Ok(ClientAuth::Codex(Arc::new(Mutex::new(codex))))
            }
            Auth::GoogleOauth { .. } => Err(anyhow!(
                "google_oauth auth mode is not supported by OpenAI clients"
            )),
        }
    }

    /// Returns `(bearer_token, extra_headers)` for the next outbound request.
    /// For the Codex variant this refreshes the access token if it's near
    /// expiry — callers don't need to do anything special.
    pub async fn prepare_headers(
        &self,
        http: &reqwest::Client,
    ) -> Result<(String, Vec<(&'static str, String)>)> {
        match self {
            ClientAuth::ApiKey(key) => Ok((key.clone(), Vec::new())),
            ClientAuth::Codex(m) => {
                let mut guard = m.lock().await;
                guard
                    .ensure_fresh(http)
                    .await
                    .context("codex auth refresh")?;
                let mut extras = Vec::new();
                if let Some(acc) = guard.chatgpt_account_id() {
                    extras.push(("chatgpt-account-id", acc.to_string()));
                }
                Ok((guard.access_token().to_string(), extras))
            }
        }
    }
}
