//! llama.cpp-specific backend.
//!
//! The chat request itself goes out on llama.cpp's OpenAI-compat endpoint
//! (`POST /v1/chat/completions`), which we reuse by delegating to an
//! internal [`OpenAiChatClient`] — reasoning-tag extraction, tool-call
//! parsing, and the streaming state machine are already battle-tested
//! there and have no llama.cpp-specific quirks we'd need to override.
//!
//! Driver-specific bits today are limited to two things: setting
//! `return_progress: true` on streaming requests so chunks carry inline
//! `prompt_progress` data during prefill (that recognition lives in the
//! generic `OpenAiChatClient::consume` path so it can lift them into
//! [`ModelEvent::PrefillProgress`]), and reading
//! `default_generation_settings.n_ctx` from `/props` so the model list
//! advertises the actual loaded context window. Both are llama.cpp's
//! native niceties; everything else flows through the OpenAI-shape
//! pipeline unchanged.
//!
//! Future llama.cpp-specific knobs (GBNF-constrained tool calls,
//! `cache_prompt` tuning, `/tokenize`, explicit slot save/restore for
//! fork-without-reprefill) will layer on top of this.

use serde::Deserialize;
use tokio_util::sync::CancellationToken;

use crate::providers::model::{
    BoxFuture, BoxStream, ModelError, ModelEvent, ModelInfo, ModelProvider, ModelRequest,
    ModelResponse,
};
use crate::providers::openai_chat::OpenAiChatClient;

pub struct LlamaCppClient {
    /// Server origin — e.g. `http://localhost:8080`, without a `/v1`
    /// suffix. Used for the `/props` side-channel (context window
    /// discovery) only; chat requests go through `inner`.
    origin: String,
    /// Delegate for the actual chat request. Configured with
    /// `{origin}/v1` so it reaches `POST /v1/chat/completions` the same
    /// way any OpenAI-compat endpoint is called. The
    /// `with_prompt_progress` flag flips on llama.cpp's inline prefill-
    /// progress SSE field.
    inner: OpenAiChatClient,
    /// Side-channel HTTP client for `/props`. Distinct from the chat
    /// connection pool so a slow chat request doesn't gate a model-list
    /// refresh.
    http: reqwest::Client,
}

impl LlamaCppClient {
    pub fn new(base_url: String) -> Self {
        let origin = base_url.trim_end_matches('/').to_string();
        let openai_base = format!("{origin}/v1");
        Self {
            inner: OpenAiChatClient::new(openai_base, None).with_prompt_progress(),
            http: reqwest::Client::new(),
            origin,
        }
    }
}

impl ModelProvider for LlamaCppClient {
    fn create_message<'a>(
        &'a self,
        req: &'a ModelRequest<'a>,
        cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Result<ModelResponse, ModelError>> {
        self.inner.create_message(req, cancel)
    }

    fn create_message_streaming<'a>(
        &'a self,
        req: &'a ModelRequest<'a>,
        cancel: &'a CancellationToken,
    ) -> BoxStream<'a, Result<ModelEvent, ModelError>> {
        // `inner` was built with `with_prompt_progress()`, so the SSE
        // chunks carry `prompt_progress` during prefill and are lifted
        // into `ModelEvent::PrefillProgress` by the generic consume path.
        self.inner.create_message_streaming(req, cancel)
    }

    fn capabilities_for(&self, model_id: &str) -> whisper_agent_protocol::ContentCapabilities {
        // Delegate to the wrapped OpenAI-shaped client. Today this
        // returns text-only for every llama.cpp model id (the OpenAI
        // heuristic doesn't match `llama-3`, `qwen-vl`, etc.) so
        // vision-capable local models won't see `view_image`. Refining
        // this is a config-side concern — we'll need the user to
        // declare per-model capabilities in the backend config.
        self.inner.capabilities_for(model_id)
    }

    fn list_models<'a>(&'a self) -> BoxFuture<'a, Result<Vec<ModelInfo>, ModelError>> {
        Box::pin(async move {
            let mut models = self.inner.list_models().await?;
            // llama.cpp serves one loaded model per server, so the same
            // context window applies to every entry the /v1/models
            // endpoint reports. /props exposes it as
            // `default_generation_settings.n_ctx`.
            if let Some(n_ctx) = self.fetch_n_ctx().await {
                for m in &mut models {
                    m.context_window = Some(n_ctx);
                }
            }
            Ok(models)
        })
    }
}

impl LlamaCppClient {
    /// Fetch `default_generation_settings.n_ctx` from llama.cpp's
    /// `/props` endpoint. Returns `None` on any failure (transport,
    /// non-200, unparseable) — callers treat missing context window as
    /// "unknown" rather than surfacing the error, since the model list
    /// itself already succeeded.
    async fn fetch_n_ctx(&self) -> Option<u32> {
        let url = format!("{}/props", self.origin);
        let resp = self.http.get(&url).send().await.ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let parsed: PropsResponse = resp.json().await.ok()?;
        parsed.default_generation_settings.n_ctx
    }
}

/// Subset of `/props` — llama.cpp returns many more fields (build info,
/// model metadata, tokenizer config) that we skip via serde's default
/// ignore-extra-keys behavior.
#[derive(Deserialize, Debug, Default)]
struct PropsResponse {
    #[serde(default)]
    default_generation_settings: PropsGenSettings,
}

#[derive(Deserialize, Debug, Default)]
struct PropsGenSettings {
    #[serde(default)]
    n_ctx: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_n_ctx_from_real_props_payload() {
        // Trimmed /props response shape — llama.cpp surfaces many
        // more keys (build_info, model_path, chat_template, …) that
        // we rely on serde to ignore. The load-bearing parse is
        // default_generation_settings.n_ctx.
        let json = r#"{
            "default_generation_settings": {
                "n_ctx": 32768,
                "n_predict": -1,
                "temperature": 0.8
            },
            "total_slots": 1,
            "chat_template": "{% for m in messages %}..."
        }"#;
        let parsed: PropsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.default_generation_settings.n_ctx, Some(32768));
    }

    #[test]
    fn props_without_n_ctx_yields_none() {
        let parsed: PropsResponse = serde_json::from_str("{}").unwrap();
        assert_eq!(parsed.default_generation_settings.n_ctx, None);
    }
}
