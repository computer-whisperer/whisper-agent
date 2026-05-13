//! llama.cpp-specific backend.
//!
//! The chat request itself goes out on llama.cpp's OpenAI-compat endpoint
//! (`POST /v1/chat/completions`), which we reuse by delegating to an
//! internal [`OpenAiChatClient`] — reasoning-tag extraction, tool-call
//! parsing, and the streaming state machine are already battle-tested
//! there and have no llama.cpp-specific quirks we'd need to override.
//!
//! Driver-specific bits today are:
//!
//! - `return_progress: true` on streaming requests so chunks carry
//!   inline `prompt_progress` data during prefill — the generic
//!   `OpenAiChatClient::consume` path lifts those into
//!   [`ModelEvent::PrefillProgress`].
//! - Reading `default_generation_settings.n_ctx` from `/props` so the
//!   model list advertises the actual loaded context window.
//! - Polling `/slots` during the decode phase to surface cumulative
//!   `n_decoded` as [`ModelEvent::OutputTokensProgress`]. The
//!   OpenAI-compat SSE only carries `usage` on its final chunk, so
//!   without the side-channel poll the UI's live tokens-per-second
//!   chip has nothing to chew on for llama.cpp during the run — only a
//!   single value right before stream-end. Polling starts on the first
//!   text/reasoning delta (during prefill `n_decoded` is 0 anyway) and
//!   stops when the inner stream terminates.
//!
//! Future llama.cpp-specific knobs (GBNF-constrained tool calls,
//! `cache_prompt` tuning, `/tokenize`, explicit slot save/restore for
//! fork-without-reprefill) will layer on top of this.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
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

/// How often to poll llama.cpp's `/slots` side-channel during decode.
/// 2 Hz is smooth enough for the live tokens-per-second chip to feel
/// responsive without flooding the local server — each tick is one
/// small GET, so even at sub-ms /slots latencies the overhead is
/// negligible. The chip snaps to the authoritative `usage.output_tokens`
/// on stream end anyway, so this only affects the live indicator's
/// granularity, not the final accuracy.
const SLOTS_POLL_INTERVAL_MS: u64 = 500;

pub struct LlamaCppClient {
    /// Server origin — e.g. `http://localhost:8080`, without a `/v1`
    /// suffix. Used for the `/props` + `/slots` side-channels (context
    /// window discovery, live decode-token polling); chat requests go
    /// through `inner`.
    origin: String,
    /// Delegate for the actual chat request. Configured with
    /// `{origin}/v1` so it reaches `POST /v1/chat/completions` the same
    /// way any OpenAI-compat endpoint is called. The
    /// `with_prompt_progress` flag flips on llama.cpp's inline prefill-
    /// progress SSE field.
    inner: OpenAiChatClient,
    /// Side-channel HTTP client for `/props` + `/slots`. Distinct from
    /// the chat connection pool so a slow chat request doesn't gate
    /// model-list refresh or per-tick decode polls.
    http: reqwest::Client,
    /// One-shot guard for the "multiple is_processing slots — picking
    /// first" warning. llama.cpp's OpenAI-compat path doesn't echo
    /// which slot served our request, so on a contended multi-user
    /// server the live tok/s chip can pin to the wrong slot's
    /// `n_decoded`. Warn once per process so the operator sees the
    /// caveat without flooding the log on every poll.
    multi_slot_warned: Arc<AtomicBool>,
}

impl LlamaCppClient {
    pub fn new(base_url: String) -> Self {
        crate::ensure_default_crypto_provider();
        let origin = base_url.trim_end_matches('/').to_string();
        let openai_base = format!("{origin}/v1");
        Self {
            inner: OpenAiChatClient::new(openai_base, None).with_prompt_progress(),
            http: reqwest::Client::new(),
            origin,
            multi_slot_warned: Arc::new(AtomicBool::new(false)),
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
        // Inner stream covers chat content, tool calls, and inline
        // `prompt_progress`-derived [`ModelEvent::PrefillProgress`]
        // events. We layer a `/slots` poller on top to surface
        // cumulative `n_decoded` as [`ModelEvent::OutputTokensProgress`]
        // — the OpenAI-compat SSE itself only carries `usage` once at
        // the very end, so without this side-channel the live tok/s
        // chip would have nothing to track during the run.
        //
        // **Concurrency:** the poller runs on its own `tokio::spawn`d
        // task and pushes values through an mpsc channel. An earlier
        // attempt put both the inner-stream pull and the `tick.tick()`
        // in a single `tokio::select!` inside the main loop. That
        // starved the tick branch under realistic conditions —
        // reqwest's bytes_stream is heavily buffered, so
        // `inner.next()` returns Ready in micros most iterations, and
        // even without `biased;` the tick can wait far longer than
        // 500 ms to win the race. Separate tasks decouple them
        // entirely: the SSE loop never blocks the poller.
        let inner_stream = self.inner.create_message_streaming(req, cancel);
        let http = self.http.clone();
        let origin = self.origin.clone();
        let multi_slot_warned = self.multi_slot_warned.clone();
        let cancel_for_poller = cancel.clone();
        let (progress_tx, mut progress_rx) = tokio::sync::mpsc::unbounded_channel::<u32>();
        // Detached task: exits when (a) cancel fires, or (b) the
        // receiver is dropped (stream ended naturally — next send
        // fails and we break out). Worst-case leak is one extra tick
        // + one /slots HTTP call after stream end; acceptable.
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_millis(SLOTS_POLL_INTERVAL_MS));
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            // Swallow the immediate first tick — interval fires once at
            // t=0 by default and we don't want to poll before the
            // server has even routed our request.
            tick.tick().await;
            loop {
                tokio::select! {
                    _ = tick.tick() => {
                        if let Some(n) = fetch_slot_n_decoded(
                            &http, &origin, &multi_slot_warned,
                        ).await
                            && progress_tx.send(n).is_err()
                        {
                            break;
                        }
                    }
                    _ = cancel_for_poller.cancelled() => break,
                }
            }
        });

        Box::pin(try_stream! {
            let mut inner = inner_stream;
            let mut last_n_decoded: u32 = 0;
            loop {
                // Bind via a local so `?` on the inner result lives
                // outside the tokio::select! branch — async-stream's
                // `try_stream!` doesn't let `?` propagate through the
                // per-branch async blocks the macro expands into.
                #[derive(Debug)]
                enum Pulled {
                    Inner(Option<Result<ModelEvent, ModelError>>),
                    Progress(Option<u32>),
                }
                let pulled = tokio::select! {
                    next = inner.next() => Pulled::Inner(next),
                    n = progress_rx.recv() => Pulled::Progress(n),
                };
                match pulled {
                    Pulled::Inner(None) => break,
                    Pulled::Inner(Some(result)) => {
                        let ev = result?;
                        yield ev;
                    }
                    // Filter 0 (prefill window — `n_decoded` hasn't
                    // grown yet) and any non-monotonic value
                    // (defensive against a slot-id mix-up where the
                    // observed slot reset).
                    Pulled::Progress(Some(n)) => {
                        if n > last_n_decoded {
                            last_n_decoded = n;
                            yield ModelEvent::OutputTokensProgress { output_tokens: n };
                        }
                    }
                    // Poller exited (cancel or some unexpected
                    // failure). The stream still completes through
                    // the inner branch — we just lose live updates
                    // from here on. The post-turn snap from
                    // `ThreadAssistantEnd` still fires.
                    Pulled::Progress(None) => {}
                }
            }
        })
    }

    fn capabilities_for(&self, model_id: &str) -> whisper_agent_protocol::ContentCapabilities {
        llamacpp_capabilities_for(model_id)
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
            for m in &mut models {
                m.capabilities = llamacpp_capabilities_for(&m.id);
            }
            Ok(models)
        })
    }
}

/// Best-effort capability lookup for llama.cpp model ids.
///
/// llama.cpp serves multimodal input through the OpenAI-compatible
/// `/v1/chat/completions` shape when the loaded model has a matching
/// multimodal projector. Its `/v1/models` payload is still just a
/// model id, so the scheduler has no reliable in-band capability bit
/// to consult while assembling the tool list. Keep OpenAI aliases
/// working first, then recognize common GGUF vision-family names so
/// local VLMs receive media tools such as `view_image`.
fn llamacpp_capabilities_for(model_id: &str) -> whisper_agent_protocol::ContentCapabilities {
    let openai_caps = crate::providers::model::openai_vision_capabilities(model_id);
    if !openai_caps.input.image.is_empty() || !openai_caps.input.document.is_empty() {
        return llamacpp_vision_capabilities();
    }

    if looks_like_llamacpp_vision_model(model_id) {
        llamacpp_vision_capabilities()
    } else {
        whisper_agent_protocol::ContentCapabilities::default()
    }
}

fn llamacpp_vision_capabilities() -> whisper_agent_protocol::ContentCapabilities {
    whisper_agent_protocol::ContentCapabilities {
        input: whisper_agent_protocol::MediaSupport::standard_image_input(),
        output: whisper_agent_protocol::MediaSupport::default(),
    }
}

fn looks_like_llamacpp_vision_model(model_id: &str) -> bool {
    let lower = model_id.to_ascii_lowercase();
    let normalized: String = lower
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    let compact: String = lower
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect();

    // Gemma 4 is natively multimodal. Gemma 3 vision checkpoints are
    // the 4B+ family; the 1B checkpoint is text-only, so avoid a broad
    // `gemma-3` match.
    if normalized.contains("gemma-4")
        || compact.contains("gemma4")
        || normalized.contains("gemma-3n")
        || compact.contains("gemma3n")
        || normalized.contains("gemma-3-4b")
        || normalized.contains("gemma-3-12b")
        || normalized.contains("gemma-3-27b")
        || compact.contains("gemma34b")
        || compact.contains("gemma312b")
        || compact.contains("gemma327b")
    {
        return true;
    }

    const NEEDLES: &[&str] = &[
        "llava",
        "bakllava",
        "moondream",
        "minicpm-v",
        "minicpmv",
        "internvl",
        "pixtral",
        "smolvlm",
        "idefics",
        "cogvlm",
        "deepseek-vl",
        "deepseekvl",
        "glm-4v",
        "glm4v",
        "qwen-vl",
        "qwenvl",
        "qwen2-vl",
        "qwen2vl",
        "qwen2-5-vl",
        "qwen25vl",
        "qwen3-vl",
        "qwen3vl",
        "qwen2-5-omni",
        "qwen25omni",
        "llama-3-2-vision",
        "llama32vision",
        "llama-4-scout",
        "llama4scout",
        "llama-4-maverick",
        "llama4maverick",
        "phi-3-vision",
        "phi3vision",
        "phi-3-5-vision",
        "phi35vision",
        "yi-vl",
        "yivl",
        "lfm2-vl",
        "lfm2vl",
    ];

    NEEDLES
        .iter()
        .any(|needle| normalized.contains(needle) || compact.contains(needle))
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

/// One entry in llama.cpp's `/slots` response array. Only the fields
/// we read are deserialized; the real payload carries many more (slot
/// id, prompt, params, …) that we let serde discard.
#[derive(Deserialize, Debug, Default)]
struct SlotInfo {
    #[serde(default)]
    is_processing: bool,
    #[serde(default)]
    n_decoded: u32,
}

/// Pick the `n_decoded` of the first `is_processing` slot from a
/// `/slots` response. Returns `None` if no slot is processing — that
/// can happen during the brief window between the inner stream's
/// last chunk and `is_processing` flipping back to false on the
/// server. On the first poll where we see 2+ processing slots, flip
/// `multi_slot_warned` and emit one `tracing::warn` — picking the
/// first processing slot will be wrong on a contended multi-user
/// server but is the best we can do without a server-side hook to
/// correlate the slot to our request.
fn pick_n_decoded(slots: &[SlotInfo], multi_slot_warned: &AtomicBool) -> Option<u32> {
    let processing: Vec<&SlotInfo> = slots.iter().filter(|s| s.is_processing).collect();
    if processing.len() > 1 && !multi_slot_warned.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            count = processing.len(),
            "llama.cpp /slots reports {} concurrently-processing slots; \
            tokens-per-second indicator will track the first one and may \
            be wrong on contended multi-user servers",
            processing.len(),
        );
    }
    processing.first().map(|s| s.n_decoded)
}

/// Fetch `/slots` and return the active slot's cumulative
/// `n_decoded`. Any failure (transport, non-200, parse) yields `None`
/// — the chip simply doesn't update this tick. Logging every miss
/// would spam the server log on disconnect, and the next tick will
/// retry naturally.
async fn fetch_slot_n_decoded(
    http: &reqwest::Client,
    origin: &str,
    multi_slot_warned: &AtomicBool,
) -> Option<u32> {
    let url = format!("{origin}/slots");
    let resp = http.get(&url).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let slots: Vec<SlotInfo> = resp.json().await.ok()?;
    pick_n_decoded(&slots, multi_slot_warned)
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

    #[test]
    fn pick_n_decoded_returns_processing_slot_value() {
        // Trimmed /slots payload — llama.cpp's real response carries
        // many more fields per slot (id, prompt, params, …) which we
        // rely on serde to ignore.
        let raw = r#"[
            {"id":0,"is_processing":true,"n_decoded":42},
            {"id":1,"is_processing":false,"n_decoded":0}
        ]"#;
        let slots: Vec<SlotInfo> = serde_json::from_str(raw).unwrap();
        let warned = AtomicBool::new(false);
        assert_eq!(pick_n_decoded(&slots, &warned), Some(42));
        assert!(
            !warned.load(Ordering::Relaxed),
            "single processing slot must not warn"
        );
    }

    #[test]
    fn pick_n_decoded_with_no_processing_slot_returns_none() {
        let raw = r#"[
            {"id":0,"is_processing":false,"n_decoded":99}
        ]"#;
        let slots: Vec<SlotInfo> = serde_json::from_str(raw).unwrap();
        let warned = AtomicBool::new(false);
        assert_eq!(pick_n_decoded(&slots, &warned), None);
    }

    #[test]
    fn pick_n_decoded_with_multiple_processing_slots_picks_first_and_warns_once() {
        let raw = r#"[
            {"id":0,"is_processing":true,"n_decoded":17},
            {"id":1,"is_processing":true,"n_decoded":91}
        ]"#;
        let slots: Vec<SlotInfo> = serde_json::from_str(raw).unwrap();
        let warned = AtomicBool::new(false);
        // First call: picks index-0 slot, flips the warn flag.
        assert_eq!(pick_n_decoded(&slots, &warned), Some(17));
        assert!(
            warned.load(Ordering::Relaxed),
            "multi-processing must set warned"
        );
        // Subsequent calls don't re-warn — guard is one-shot.
        assert_eq!(pick_n_decoded(&slots, &warned), Some(17));
        assert!(warned.load(Ordering::Relaxed));
    }

    #[test]
    fn pick_n_decoded_handles_empty_slots_array() {
        let slots: Vec<SlotInfo> = serde_json::from_str("[]").unwrap();
        let warned = AtomicBool::new(false);
        assert_eq!(pick_n_decoded(&slots, &warned), None);
    }

    #[test]
    fn capabilities_mark_gemma_4_as_vision() {
        let caps = llamacpp_capabilities_for("ggml-org/gemma-4-E4B-it-GGUF");
        assert!(!caps.input.image.is_empty());
        assert!(caps.input.document.is_empty());
    }

    #[test]
    fn capabilities_do_not_mark_text_only_local_ids_as_vision() {
        let caps = llamacpp_capabilities_for("qwen2.5-coder-7b");
        assert!(caps.input.image.is_empty());
    }

    #[test]
    fn capabilities_keep_openai_vision_aliases_working() {
        let caps = llamacpp_capabilities_for("gpt-4o");
        assert!(!caps.input.image.is_empty());
    }
}
