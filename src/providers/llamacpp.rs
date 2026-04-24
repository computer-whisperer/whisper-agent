//! llama.cpp-specific backend.
//!
//! The chat request itself goes out on llama.cpp's OpenAI-compat endpoint
//! (`POST /v1/chat/completions`), which we reuse by delegating to an
//! internal [`OpenAiChatClient`] — reasoning-tag extraction, tool-call
//! parsing, and the streaming state machine are already battle-tested
//! there and have no llama.cpp-specific quirks we'd need to override.
//!
//! The reason to have a distinct driver at all is llama.cpp's
//! **`GET /slots` endpoint**, which reports per-slot state including
//! `n_past` (prompt tokens ingested so far) and `n_prompt_tokens` (total
//! prompt length). Polling that endpoint in parallel with the chat stream
//! gives us real prefill-progress events during the multi-second-to-minutes
//! wait for the first output token on long prompts — the only backend in
//! this repo that can honestly drive a percentage-based progress bar
//! rather than a spinner.
//!
//! This module stays minimal today; future llama.cpp-specific knobs
//! (GBNF-constrained tool calls, `cache_prompt` tuning, `/tokenize`,
//! explicit slot save/restore for fork-without-reprefill) will layer on
//! top without disturbing the generic `openai_chat` driver.

use std::time::Duration;

use async_stream::try_stream;
use futures::StreamExt;
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

use crate::providers::model::{
    BoxFuture, BoxStream, ModelError, ModelEvent, ModelInfo, ModelProvider, ModelRequest,
    ModelResponse,
};
use crate::providers::openai_chat::OpenAiChatClient;

/// Polling cadence for `/slots`. Long enough that we don't hammer the
/// server (which is busy prefilling anyway); short enough that a ~20k
/// token prefill shows meaningful progress granularity.
const SLOTS_POLL_INTERVAL: Duration = Duration::from_millis(500);

pub struct LlamaCppClient {
    /// Server origin — e.g. `http://localhost:8080`, without a `/v1`
    /// suffix. We append endpoint paths ourselves.
    origin: String,
    /// Delegate for the actual chat request. Configured with
    /// `{origin}/v1` so it reaches `POST /v1/chat/completions` the same
    /// way any OpenAI-compat endpoint is called.
    inner: OpenAiChatClient,
    /// Separate client instance for `/slots` polls so an in-flight chat
    /// request's connection pool doesn't serialize our side-channel.
    http: reqwest::Client,
}

impl LlamaCppClient {
    pub fn new(base_url: String) -> Self {
        let origin = base_url.trim_end_matches('/').to_string();
        let openai_base = format!("{origin}/v1");
        Self {
            inner: OpenAiChatClient::new(openai_base, None),
            http: reqwest::Client::new(),
            origin,
        }
    }

    /// Poll `/slots` once and return `(tokens_processed, tokens_total)`
    /// when we can identify a single busy slot with enough fields to
    /// drive a percentage. Returns `Ok(None)` for transient ambiguous
    /// states (zero or multiple busy slots, missing fields) — those
    /// keep the poller running because the next tick may produce a
    /// usable reading. Returns `Err(())` for unrecoverable situations
    /// (endpoint 404s, response isn't parseable as llama.cpp's schema)
    /// — the caller uses that as a signal to stop polling for the rest
    /// of this request.
    async fn poll_slots_once(&self) -> Result<Option<(u32, u32)>, ()> {
        let url = format!("{}/slots", self.origin);
        let resp = self.http.get(&url).send().await.map_err(|_| ())?;
        if !resp.status().is_success() {
            return Err(());
        }
        let slots: Vec<SlotEntry> = resp.json().await.map_err(|_| ())?;
        Ok(find_active_progress(&slots))
    }
}

/// Outcome of one iteration of the merge loop in
/// [`LlamaCppClient::create_message_streaming`]. Lifts `tokio::select!`
/// branches out of their implicit async blocks so the body can use `?`
/// on the inner stream's errors without the macro reinterpreting it in
/// the wrong context.
enum Branch {
    Inner(Option<Result<ModelEvent, ModelError>>),
    Tick,
}

/// Subset of the `/slots` response shape. llama.cpp carries a dozen more
/// fields per slot (params, cache stats, grammar state, …) that we skip
/// via serde's default behaviour of ignoring extra keys. Every field is
/// `Option`-wrapped because the exact set varies between llama.cpp
/// versions — we emit progress only when both counters are present.
#[derive(Deserialize, Debug)]
struct SlotEntry {
    #[serde(default)]
    is_processing: bool,
    #[serde(default)]
    n_past: Option<u32>,
    #[serde(default)]
    n_prompt_tokens: Option<u32>,
}

/// Extract a `(processed, total)` pair from a `/slots` response when
/// exactly one slot is processing and carries both counters. Pure function
/// so it's easy to unit-test against recorded llama.cpp responses.
fn find_active_progress(slots: &[SlotEntry]) -> Option<(u32, u32)> {
    let mut active = slots.iter().filter(|s| s.is_processing);
    let first = active.next()?;
    // Multiple busy slots — we can't tell which one is ours, so bail
    // rather than report a stranger's progress.
    if active.next().is_some() {
        return None;
    }
    let past = first.n_past?;
    let total = first.n_prompt_tokens?;
    if total == 0 || past > total {
        return None;
    }
    Some((past, total))
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
        Box::pin(try_stream! {
            let mut inner = self.inner.create_message_streaming(req, cancel);
            // We poll `/slots` until either (a) the inner stream emits
            // its first output-bearing event — prefill is over — or
            // (b) a poll fails in a way that tells us the server
            // doesn't actually expose `/slots` (old build, disabled,
            // wrong endpoint). In case (b) we fall silent for the rest
            // of the request rather than retry every 500ms.
            let mut first_output_seen = false;
            let mut polling_disabled = false;
            let mut last_emitted: Option<(u32, u32)> = None;
            loop {
                // Route each select branch to a plain `Branch` value
                // and handle `?` at the try_stream! level — `?` inside
                // a select arm lands in the arm's inferred async block,
                // which returns `()` not `Result`.
                let branch = tokio::select! {
                    // `biased` forces top-down branch selection so a
                    // ready inner event always wins over a ready tick:
                    // we'd rather forward the first delta and stop
                    // polling than emit a stale progress number right
                    // as output begins.
                    biased;
                    next = inner.next() => Branch::Inner(next),
                    _ = tokio::time::sleep(SLOTS_POLL_INTERVAL),
                        if !first_output_seen && !polling_disabled => Branch::Tick,
                };
                match branch {
                    Branch::Inner(None) => break,
                    Branch::Inner(Some(result)) => {
                        let event = result?;
                        if matches!(
                            &event,
                            ModelEvent::TextDelta { .. }
                                | ModelEvent::ThinkingDelta { .. }
                                | ModelEvent::ToolCall { .. }
                        ) {
                            first_output_seen = true;
                        }
                        let is_terminal = matches!(&event, ModelEvent::Completed { .. });
                        yield event;
                        if is_terminal {
                            break;
                        }
                    }
                    Branch::Tick => match self.poll_slots_once().await {
                        Ok(Some(progress)) => {
                            // Dedup identical consecutive readings.
                            // llama.cpp's `n_past` advances in chunk
                            // batches rather than per-token, so
                            // multiple consecutive polls often see the
                            // same value — no point putting a
                            // heartbeat on the wire when nothing
                            // changed.
                            if last_emitted != Some(progress) {
                                last_emitted = Some(progress);
                                yield ModelEvent::PrefillProgress {
                                    tokens_processed: progress.0,
                                    tokens_total: progress.1,
                                };
                            }
                        }
                        Ok(None) => {
                            // Ambiguous but recoverable — try again
                            // next tick.
                        }
                        Err(()) => {
                            polling_disabled = true;
                        }
                    },
                }
            }
        })
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

    fn slot(is_processing: bool, past: Option<u32>, total: Option<u32>) -> SlotEntry {
        SlotEntry {
            is_processing,
            n_past: past,
            n_prompt_tokens: total,
        }
    }

    #[test]
    fn returns_progress_for_single_busy_slot() {
        let slots = vec![slot(true, Some(1234), Some(5000))];
        assert_eq!(find_active_progress(&slots), Some((1234, 5000)));
    }

    #[test]
    fn returns_none_when_no_slot_is_busy() {
        let slots = vec![slot(false, Some(100), Some(500))];
        assert_eq!(find_active_progress(&slots), None);
    }

    #[test]
    fn returns_none_when_multiple_slots_are_busy() {
        let slots = vec![
            slot(true, Some(100), Some(500)),
            slot(true, Some(200), Some(500)),
        ];
        assert_eq!(find_active_progress(&slots), None);
    }

    #[test]
    fn returns_none_when_counters_are_missing() {
        assert_eq!(find_active_progress(&[slot(true, None, Some(500))]), None);
        assert_eq!(find_active_progress(&[slot(true, Some(100), None)]), None);
    }

    #[test]
    fn returns_none_when_total_is_zero_or_past_exceeds_total() {
        assert_eq!(find_active_progress(&[slot(true, Some(0), Some(0))]), None);
        assert_eq!(
            find_active_progress(&[slot(true, Some(600), Some(500))]),
            None
        );
    }

    #[test]
    fn parses_minimal_slot_json_shape() {
        // Captured from a recent llama.cpp server; most fields elided
        // via serde's unknown-key skipping. We're checking that a real
        // wire payload doesn't trip the deserializer.
        let json = r#"[{
            "id": 0,
            "id_task": 42,
            "is_processing": true,
            "n_past": 1024,
            "n_prompt_tokens": 5120,
            "params": {"temperature": 0.2},
            "prompt": "..."
        }]"#;
        let slots: Vec<SlotEntry> = serde_json::from_str(json).unwrap();
        assert_eq!(slots.len(), 1);
        assert!(slots[0].is_processing);
        assert_eq!(slots[0].n_past, Some(1024));
        assert_eq!(slots[0].n_prompt_tokens, Some(5120));
    }

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
