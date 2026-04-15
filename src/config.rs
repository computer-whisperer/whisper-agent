//! Server-side configuration loaded from a TOML file.
//!
//! The file describes the catalog of model backends the server is willing to drive.
//! Each entry is a named alias (e.g. `anthropic-prod`, `local-ollama`) pointing at a
//! backend kind plus any credentials / endpoint details that kind needs. Tasks refer
//! to entries by alias; the scheduler never touches backend-specific config directly.
//!
//! Example:
//!
//! ```toml
//! default_backend = "anthropic"
//!
//! [backends.anthropic]
//! kind = "anthropic"
//! api_key_env = "ANTHROPIC_API_KEY"
//! default_model = "claude-sonnet-4-6"
//!
//! [backends.local-ollama]
//! kind = "openai_chat"
//! base_url = "http://localhost:11434/v1"
//! default_model = "llama3.2"
//!
//! [backends.openai]
//! kind = "openai_chat"
//! base_url = "https://api.openai.com/v1"
//! api_key_env = "OPENAI_API_KEY"
//! default_model = "gpt-4o"
//! ```

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;

use crate::anthropic::AnthropicClient;
use crate::model::ModelProvider;
use crate::openai_chat::OpenAiChatClient;

#[derive(Deserialize, Debug, Clone)]
pub struct Config {
    pub default_backend: String,
    #[serde(default)]
    pub backends: BTreeMap<String, BackendConfig>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BackendConfig {
    Anthropic {
        #[serde(default)]
        api_key_env: Option<String>,
        default_model: String,
    },
    OpenAiChat {
        base_url: String,
        #[serde(default)]
        api_key_env: Option<String>,
        default_model: String,
    },
}

impl BackendConfig {
    pub fn kind(&self) -> &'static str {
        match self {
            BackendConfig::Anthropic { .. } => "anthropic",
            BackendConfig::OpenAiChat { .. } => "openai_chat",
        }
    }

    pub fn default_model(&self) -> &str {
        match self {
            BackendConfig::Anthropic { default_model, .. } => default_model,
            BackendConfig::OpenAiChat { default_model, .. } => default_model,
        }
    }

    /// Resolve any `api_key_env` lookup and construct the corresponding provider.
    /// Returns an error if a required env var is unset.
    pub fn build(&self) -> Result<Arc<dyn ModelProvider>> {
        match self {
            BackendConfig::Anthropic {
                api_key_env,
                default_model: _,
            } => {
                let var = api_key_env.as_deref().unwrap_or("ANTHROPIC_API_KEY");
                let key = std::env::var(var)
                    .with_context(|| format!("env var `{var}` not set"))?;
                Ok(Arc::new(AnthropicClient::new(key)))
            }
            BackendConfig::OpenAiChat {
                base_url,
                api_key_env,
                default_model: _,
            } => {
                let key = match api_key_env {
                    Some(var) => Some(
                        std::env::var(var)
                            .with_context(|| format!("env var `{var}` not set"))?,
                    ),
                    None => None,
                };
                Ok(Arc::new(OpenAiChatClient::new(base_url.clone(), key)))
            }
        }
    }
}

impl Config {
    pub async fn load(path: &Path) -> Result<Self> {
        let text = tokio::fs::read_to_string(path)
            .await
            .with_context(|| format!("read config {}", path.display()))?;
        let config: Config = toml::from_str(&text)
            .with_context(|| format!("parse config {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        if self.backends.is_empty() {
            return Err(anyhow!("config: no backends defined"));
        }
        if !self.backends.contains_key(&self.default_backend) {
            return Err(anyhow!(
                "config: default_backend `{}` not in backends",
                self.default_backend
            ));
        }
        Ok(())
    }
}
