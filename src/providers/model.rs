//! Provider-agnostic model interface.
//!
//! [`ModelProvider`] is the object-safe trait every backend (Anthropic, OpenAI-compatible
//! Chat Completions, …) implements. The scheduler holds `Arc<dyn ModelProvider>` and
//! never touches a provider-specific type. Each impl converts between its own wire
//! shape and the normalized types here at the boundary.
//!
//! Normalized shapes intentionally re-use our canonical [`ContentBlock`] / [`Message`]
//! types, which are Anthropic-content-block-shaped. Translating OpenAI-style messages
//! into / out of this shape is the OpenAI backend's job, not the scheduler's.

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use futures::stream::{Stream, StreamExt};
use serde_json::Value;
use thiserror::Error;
use tokio_util::sync::CancellationToken;
use whisper_agent_protocol::{
    BackendUsage, ContentBlock, ContentCapabilities, ImageMime, MediaSupport, Message, ParamSpec,
    ToolKind, ToolSchema, TunableSpec, TunableValue, Usage,
};

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
pub type BoxStream<'a, T> = Pin<Box<dyn Stream<Item = T> + Send + 'a>>;

pub struct ModelRequest<'a> {
    pub model: &'a str,
    /// Per-request output cap — the maximum number of tokens the model
    /// may generate in this single response. This is a ceiling on the
    /// *response*, not the model's total input context window (for
    /// that, see [`ModelInfo::context_window`]). Anthropic requires
    /// this field; OpenAI's Responses API ignores it; others pass it
    /// through as `max_output_tokens` / `max_completion_tokens`.
    pub max_tokens: u32,
    pub system_prompt: &'a str,
    pub tools: &'a [ToolSpec],
    pub messages: &'a [Message],
    /// Positions at which the backend should mark prompt-cache checkpoints.
    /// Providers that don't support caching ignore this list. Anthropic caps the
    /// list at 4 markers per request — the scheduler is responsible for staying
    /// under that limit.
    pub cache_breakpoints: &'a [CacheBreakpoint],
    /// Backend-specific execution knobs, keyed by the spec keys the
    /// provider advertised via `tunables_for(model_id)`. Each provider
    /// only consumes keys it advertised; unknown keys are ignored so
    /// the wire stays stable as new tunables land. Empty when the
    /// thread carries no per-knob choices (either because the model
    /// advertises nothing, or because the user accepted every
    /// advertised default and the UI sent nothing back).
    pub tunables: &'a BTreeMap<String, TunableValue>,
    /// Stable per-thread routing key. Forwarded as OpenAI's
    /// `prompt_cache_key` body field so successive requests for the
    /// same thread land on the same cache shard — without it the
    /// load balancer routes by prefix-hash alone and can shuffle
    /// adjacent same-prefix calls across cache-cold nodes. Providers
    /// without a routing-key concept ignore this field.
    pub request_cache_key: Option<&'a str>,
}

/// Logical positions at which a cache checkpoint can be attached. Translated into
/// provider-specific markers by each [`ModelProvider`] implementation — or ignored
/// by providers without cache support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheBreakpoint {
    /// Cache the system prompt (everything up to and including `system`).
    AfterSystem,
    /// Cache the tool declarations (everything up to and including `tools`).
    AfterTools,
    /// Cache through `messages[index]` (inclusive).
    AfterMessage(usize),
}

#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// Typed parameter list. Adapters that need the JSON Schema
    /// envelope on the wire build it via [`Self::input_schema_value`]
    /// at the boundary — every provider currently consumes
    /// JSON-Schema-shaped input, so this is a thin wrapper around
    /// [`ToolSchema::input_schema_value`].
    pub params: Vec<ParamSpec>,
    /// Function-call (default, agent-side dispatch) vs ProviderBuiltin
    /// (provider runs the tool inline; no tool_use → tool_result
    /// roundtrip). Adapters branch on this in their `tools` wire
    /// shaping (e.g. OpenAI's `image_generation` is `{type: "image_generation"}`,
    /// not `{type: "function", function: {...}}`).
    pub kind: ToolKind,
}

impl ToolSpec {
    /// JSON Schema view of `params` — `{type:"object", properties:...,
    /// required:[...]}` — for provider adapters and any other site
    /// that still consumes the wire schema directly.
    pub fn input_schema_value(&self) -> Value {
        ToolSchema {
            name: self.name.clone(),
            description: self.description.clone(),
            params: self.params.clone(),
            kind: self.kind,
        }
        .input_schema_value()
    }
}

#[derive(Debug)]
pub struct ModelResponse {
    pub content: Vec<ContentBlock>,
    pub stop_reason: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub display_name: Option<String>,
    /// Maximum input context the model will accept, in tokens. `None`
    /// when the upstream `/models` endpoint doesn't publish it —
    /// today Anthropic and both OpenAI routes don't, so their entries
    /// land as `None`. Gemini and llama.cpp report the real number.
    pub context_window: Option<u32>,
    /// Provider-declared ceiling on output tokens per request. `None`
    /// when not exposed by the upstream API. This is the hard upper
    /// bound; [`ModelRequest::max_tokens`] is the caller's per-call
    /// choice within that bound.
    pub max_output_tokens: Option<u32>,
    /// Multimodal capability declaration — which MIME types the model
    /// accepts as input and emits as output. Adapters fill this in
    /// from a per-model table (vision support varies across each
    /// provider's catalog). Default is empty = text-only.
    pub capabilities: ContentCapabilities,
    /// Backend-specific execution knobs this model exposes (thinking
    /// on/off, reasoning effort, fast mode, …). Providers populate
    /// this from their own per-model knowledge; empty means "no
    /// advertised tunables." The scheduler copies this list onto
    /// `ModelSummary.tunables` so UIs can render matching controls
    /// in the thread-creation flow.
    pub tunables: Vec<TunableSpec>,
}

/// Structured fields parsed out of a provider's native error envelope.
/// Populated by each adapter at the point of failure so io_dispatch can
/// log them as tracing fields and the forensic dump can record them
/// verbatim. All fields are optional because each provider populates a
/// different subset:
///
/// - OpenAI Responses (`response.failed`, HTTP non-2xx): `code`
///   (`response.error.code` or top-level `error.code`), `kind`
///   (`error.type`), `response_id` (`response.id`), `message`.
/// - OpenAI Chat: `code`, `kind` (= `error.type`), `message`.
/// - Anthropic: `code` (= `error.type`, e.g. `"overloaded_error"`),
///   `message`.
/// - Gemini: `code` (= `error.status`, e.g. `"RESOURCE_EXHAUSTED"`),
///   `kind` (HTTP-status numeric mapping), `message`.
///
/// All-None means the body wasn't parseable in any provider-recognised
/// shape; the raw body still lives in [`ModelError::Api::body`].
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct ProviderErrorDetail {
    /// Provider-specific machine-readable code — the most useful field
    /// for grepping and for deciding retry-vs-give-up. Examples:
    /// `context_length_exceeded`, `overloaded_error`,
    /// `RESOURCE_EXHAUSTED`, `server_error`.
    pub code: Option<String>,
    /// Coarser categorisation when the provider exposes one separately
    /// from `code`. OpenAI's `error.type` lives here
    /// (`invalid_request_error`, `server_error`, …); Anthropic uses its
    /// `error.type` as `code` so this stays None there.
    pub kind: Option<String>,
    /// Provider-side response identifier so an operator can
    /// cross-reference the failure with the provider's own logs
    /// (OpenAI request id, Responses `response.id`, etc.). None when
    /// the wire shape doesn't carry one.
    pub response_id: Option<String>,
    /// Human-readable detail, verbatim from the provider when present.
    /// Often the same as [`ModelError::Api::body`] when no nested
    /// envelope exists, but distinct when the message is dug out of a
    /// nested `error.message` field.
    pub message: Option<String>,
}

/// Captured request + response bytes from a single model call, for
/// post-mortem forensic dump when something goes wrong. Wrapped in an
/// `Arc` on [`ModelError::Api::forensic`] so cloning the error stays
/// cheap. Adapters populate this at the point of failure; io_dispatch
/// decides whether to flush it to disk.
///
/// `response_excerpt` is capped (the adapter holds a rolling window of
/// the last `FORENSIC_RESPONSE_CAP_BYTES` bytes) to bound memory on
/// long-running streams. `response_truncated` records whether the cap
/// was hit so a reader of the dump knows the bytes aren't the full
/// stream.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ForensicArtifact {
    /// Backend name as registered with the scheduler — used to label
    /// the dump and to slice the forensic directory by backend.
    pub backend: String,
    /// Model id as actually sent on the wire.
    pub model: String,
    /// HTTP request body the adapter POSTed. Almost always JSON; the
    /// content type is recorded separately for the rare exception.
    pub request_body: String,
    /// Content type of `request_body`.
    pub request_content_type: &'static str,
    /// Response bytes captured up to the point of failure. For
    /// streaming responses this is the rolling tail of the SSE stream;
    /// for HTTP non-2xx responses this is the full body (or capped if
    /// huge). UTF-8 in practice — kept as `Vec<u8>` to avoid losing
    /// bytes when a stream truncates mid-codepoint.
    pub response_excerpt: Vec<u8>,
    /// True when `response_excerpt` was capped by the rolling window
    /// and earlier bytes were dropped. Always false for non-streaming
    /// HTTP error captures.
    pub response_truncated: bool,
}

/// Soft cap on `ForensicArtifact::response_excerpt`. Big enough to
/// retain the full final SSE event in nearly every realistic failure;
/// small enough that an occasional dump doesn't balloon a multi-MB
/// JSON file. Exposed so adapters and tests share one constant.
pub const FORENSIC_RESPONSE_CAP_BYTES: usize = 256 * 1024;

/// Per-call forensic context built up by the adapter before it ships
/// any bytes. Captures the request body once so error paths can attach
/// it without re-serializing. Pair with a [`ForensicBuf`] that
/// accumulates response bytes; on failure call
/// [`Self::into_artifact_with_response`] (streaming) or
/// [`Self::into_http_error_artifact`] (pre-stream non-2xx) to mint the
/// final [`ForensicArtifact`].
#[derive(Debug, Clone)]
pub struct ForensicContext {
    /// Stable provider-shaped id ("openai_responses", "anthropic", …).
    /// Distinct from the user-configured backend label which io_dispatch
    /// tacks onto the dump filename — adapters don't know the latter.
    pub backend: String,
    pub model: String,
    pub request_body: String,
    pub request_content_type: &'static str,
}

impl ForensicContext {
    pub fn into_artifact_with_response(self, buf: ForensicBuf) -> ForensicArtifact {
        ForensicArtifact {
            backend: self.backend,
            model: self.model,
            request_body: self.request_body,
            request_content_type: self.request_content_type,
            response_excerpt: buf.bytes,
            response_truncated: buf.truncated,
        }
    }

    /// Build an artifact for a pre-stream HTTP-error capture — the
    /// entire error body is the response and was read whole, so the
    /// truncated flag is always false.
    pub fn into_http_error_artifact(self, body: impl Into<Vec<u8>>) -> ForensicArtifact {
        ForensicArtifact {
            backend: self.backend,
            model: self.model,
            request_body: self.request_body,
            request_content_type: self.request_content_type,
            response_excerpt: body.into(),
            response_truncated: false,
        }
    }
}

/// Rolling response-bytes window used during streaming consumption.
/// Keeps the most-recent [`FORENSIC_RESPONSE_CAP_BYTES`] bytes so that
/// a terminal failure mid-stream still captures the events leading up
/// to it without unbounded memory growth on multi-MB streams.
#[derive(Debug, Default)]
pub struct ForensicBuf {
    pub bytes: Vec<u8>,
    pub truncated: bool,
}

impl ForensicBuf {
    pub fn push(&mut self, chunk: &[u8]) {
        self.bytes.extend_from_slice(chunk);
        if self.bytes.len() > FORENSIC_RESPONSE_CAP_BYTES {
            let drop = self.bytes.len() - FORENSIC_RESPONSE_CAP_BYTES;
            self.bytes.drain(..drop);
            self.truncated = true;
        }
    }
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("transport error: {0}")]
    Transport(String),
    #[error("api error {status}: {body}")]
    Api {
        status: u16,
        body: String,
        /// Structured fields extracted from the provider's error
        /// envelope, when the adapter recognised the shape. `None`
        /// when nothing was parseable. Drives the structured tracing
        /// fields in io_dispatch and the `detail` block in the
        /// forensic dump. Boxed to keep the `ModelError` enum small
        /// enough for the `Result<_, ModelError>` clippy size lint
        /// (the detail struct is ~100 B on its own).
        #[allow(dead_code)]
        detail: Option<Box<ProviderErrorDetail>>,
        /// Captured request + response bytes for forensic dump. `None`
        /// when the failure happened before a request was built (auth
        /// refresh failed pre-send, etc.) or when the adapter declined
        /// to capture for a specific class of error.
        #[allow(dead_code)]
        forensic: Option<std::sync::Arc<ForensicArtifact>>,
    },
    /// Server signalled a short-term capacity / quota exhaustion (typically HTTP
    /// 429). Carries the server-suggested wait before retrying along with the
    /// raw body so the caller can show a useful message. The dispatch layer may
    /// retry once transparently when `retry_after` is small; longer waits
    /// surface as failures with the wait in the message.
    #[error("rate limited: retry after {retry_after:?}: {body}")]
    RateLimited { retry_after: Duration, body: String },
    /// The per-thread `CancellationToken` fired mid-request. Providers
    /// return this from both pre-request and mid-stream cancel paths so
    /// the scheduler can tell a cancel apart from a transport failure —
    /// cancels never retry and never surface as user-visible errors.
    #[error("cancelled")]
    Cancelled,
}

impl ModelError {
    /// Construct an `Api` error with no structured detail and no
    /// forensic capture — the cheapest call-site form for tests and
    /// for adapters that haven't been wired through the forensic
    /// machinery yet. Production adapters should populate `detail` and
    /// `forensic` directly via the struct form.
    pub fn api(status: u16, body: impl Into<String>) -> Self {
        ModelError::Api {
            status,
            body: body.into(),
            detail: None,
            forensic: None,
        }
    }

    /// True when this error is a transient infrastructure fault worth
    /// retrying transparently — network glitches, request timeouts,
    /// and 5xx server errors (including Anthropic's 529 Overloaded).
    /// Client-side 4xx (malformed request, auth failure, context
    /// exhausted) and `RateLimited` are NOT transient here:
    /// `RateLimited` has its own retry path driven by the
    /// server-suggested `retry_after`, and 4xx won't fix itself on
    /// retry.
    pub fn is_transient(&self) -> bool {
        match self {
            ModelError::Transport(_) => true,
            ModelError::Api { status, .. } => matches!(*status, 408 | 500..=599),
            ModelError::RateLimited { .. } | ModelError::Cancelled => false,
        }
    }

    /// Borrow the structured detail from an `Api` error, if any. Used
    /// by io_dispatch to attach tracing fields without reaching into
    /// the variant directly at every log site.
    pub fn provider_detail(&self) -> Option<&ProviderErrorDetail> {
        match self {
            ModelError::Api {
                detail: Some(d), ..
            } => Some(d.as_ref()),
            _ => None,
        }
    }

    /// Borrow the forensic artifact from an `Api` error, if captured.
    /// Used by io_dispatch's terminal-failure path to flush a dump.
    pub fn forensic(&self) -> Option<&ForensicArtifact> {
        match self {
            ModelError::Api { forensic, .. } => forensic.as_deref(),
            _ => None,
        }
    }
}

/// Incremental event emitted from a streaming model call. Intended for live UI
/// updates; the canonical, assembled turn content lands in the terminal
/// [`ModelEvent::Completed`] event.
///
/// Streams are `Stream<Item = Result<ModelEvent, ModelError>>`. A terminal
/// error ends the stream without a `Completed`. On success, `Completed` is
/// always the last event — the scheduler persists from its `content`, not
/// from reassembling deltas.
///
/// Tool calls arrive as a single `ToolCall` event with fully-assembled args,
/// not a begin/delta/end trio — live partial-JSON rendering isn't worth the
/// webui complexity. Text and thinking *are* emitted as deltas because
/// character-by-character streaming is the user-visible win.
#[derive(Debug, Clone)]
pub enum ModelEvent {
    /// Append to the current assistant text block (open a new one if the last
    /// emitted event wasn't also a `TextDelta`).
    TextDelta { text: String },
    /// Append to the current assistant thinking block.
    ThinkingDelta { text: String },
    /// A fully-formed tool call. Emitted once per call, after the model has
    /// streamed its full argument JSON and the adapter has parsed it.
    ToolCall {
        id: String,
        name: String,
        input: Value,
    },
    /// In-progress tool call. Providers that can observe the tool name
    /// before the arguments finish streaming emit this so clients can
    /// show a placeholder chip (tool name + spinner + char count) during
    /// the buffering window, instead of dead silence while the model
    /// writes a large args blob. Throttled by senders — clients should
    /// expect only periodic updates, not one per token.
    ///
    /// Emitted at least once when the tool-use block opens (with
    /// `args_chars = 0`) and again periodically as args grow. The
    /// subsequent `ToolCall` event carries the fully-parsed input and
    /// supersedes every streaming update for the same `id`.
    ToolCallStreaming {
        id: String,
        name: String,
        /// Cumulative length of the raw args-JSON buffered so far. Not
        /// a token count — providers don't expose those mid-stream.
        args_chars: u32,
    },
    /// A fully-formed assistant image block. Emitted once per
    /// inline-data part the streaming parser sees. Image bytes don't
    /// stream incrementally on any provider we target today — the
    /// payload arrives whole — so a single event with the assembled
    /// `ImageSource` is enough. Synthesizing path emits one for each
    /// `ContentBlock::Image` in a non-streaming response so live and
    /// snapshot-rebuild paths match.
    ImageBlock {
        source: whisper_agent_protocol::ImageSource,
    },
    /// Mid-prefill progress heartbeat. Emitted by providers that can observe
    /// how many prompt tokens have been ingested before the first output
    /// token arrives — today only the llama.cpp driver does, via the
    /// inline `prompt_progress` field on SSE chunks that the
    /// `return_progress: true` request flag enables. Ephemeral: not
    /// persisted, just broadcast to connected clients so long prefills
    /// show a progress bar instead of dead air. Providers stop emitting
    /// once the first `TextDelta` / `ThinkingDelta` / `ToolCall` has
    /// been sent on the same stream.
    PrefillProgress {
        tokens_processed: u32,
        tokens_total: u32,
    },
    /// Mid-decode running output-token count. Emitted by providers that
    /// expose cumulative `output_tokens` during streaming so clients can
    /// drive a live tokens-per-second indicator without estimating from
    /// delta character counts. Cumulative (not incremental), monotonically
    /// non-decreasing across the stream, terminal value matches
    /// `Completed.usage.output_tokens`. Emitter coverage today:
    ///
    /// - Gemini — per-chunk `usageMetadata.candidatesTokenCount`,
    ///   continuous live updates.
    /// - llama.cpp — `/slots` side-channel polled at ~2 Hz during
    ///   decode (post-prefill), continuous live updates.
    /// - Anthropic — single emission near end-of-stream when
    ///   `message_delta` lands its terminal `usage`. Not a "live"
    ///   stream of values; clients shouldn't expect a smooth chip.
    /// - OpenAI Responses — silent. The SSE has no per-chunk usage;
    ///   the rate can only be derived from `Completed.usage` /
    ///   elapsed once it lands.
    ///
    /// Ephemeral: not persisted.
    OutputTokensProgress { output_tokens: u32 },
    /// Backend-level account/quota snapshot extracted from response
    /// headers or other provider side-channels at the start of a turn.
    /// Routed by the scheduler into the backend's registry entry rather
    /// than the thread state — the snapshot is *per backend account*,
    /// not per turn. Emitted at most once per stream by the providers
    /// that expose this data (today: OpenAI Responses on the Codex
    /// route); silent for all others.
    ProviderUsage { snapshot: BackendUsage },
    /// Non-fatal degradation signal from the provider — the call
    /// succeeded (a `Completed` will still arrive) but the model's
    /// output was constrained or partially withheld for a reason the
    /// caller cares about. Examples: Gemini `finishReason = SAFETY`
    /// or `RECITATION`, OpenAI Chat `finish_reason = content_filter`,
    /// OpenAI Responses `status = incomplete` with a non-token reason.
    /// io_dispatch logs these as structured warnings; the UI may
    /// surface them as a row badge in future. Ephemeral: not persisted
    /// into the thread state.
    ProviderWarning {
        /// Provider-shaped machine code so log queries can slice by
        /// category — `"SAFETY"`, `"RECITATION"`, `"content_filter"`,
        /// `"incomplete:context_length"`. Free-form per provider.
        code: String,
        /// Human-readable detail, verbatim when the provider supplies
        /// one. Empty string when there's nothing more to say beyond
        /// `code`.
        message: String,
    },
    /// Terminal event. `content` is the assistant-turn content block list the
    /// scheduler should persist. Deltas emitted before this are strictly for
    /// broadcast; the canonical state comes from here.
    Completed {
        content: Vec<ContentBlock>,
        stop_reason: Option<String>,
        usage: Usage,
    },
}

/// Provider-agnostic model backend. Object-safe via explicit pinned-boxed futures
/// (native `async fn in trait` isn't dyn-safe with `Send` bound without extra crates).
///
/// ## Cancellation
///
/// Every `create_message*` call takes a [`CancellationToken`]. When the
/// user cancels the owning thread, the scheduler fires the token and the
/// provider must abort: drop the in-flight reqwest future (which aborts
/// at TCP level), stop consuming the response body, and return
/// [`ModelError::Cancelled`]. Either pre-request or mid-stream is fine;
/// the scheduler handles both the same way — the `Cancelled` outcome is
/// swallowed silently because the thread is already in a terminal
/// `Cancelled` state by the time the future resolves.
pub trait ModelProvider: Send + Sync {
    fn create_message<'a>(
        &'a self,
        req: &'a ModelRequest<'a>,
        cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Result<ModelResponse, ModelError>>;

    /// Streaming variant. Returns a stream of [`ModelEvent`]s ending in a
    /// single `Completed` event. The default implementation buffers on top of
    /// `create_message` and synthesizes one delta per text/thinking block
    /// plus one `ToolCall` per tool_use, so consumers only need to handle
    /// the streaming shape. Adapters override to emit real SSE deltas as
    /// bytes arrive.
    fn create_message_streaming<'a>(
        &'a self,
        req: &'a ModelRequest<'a>,
        cancel: &'a CancellationToken,
    ) -> BoxStream<'a, Result<ModelEvent, ModelError>> {
        Box::pin(async_stream::try_stream! {
            let resp = self.create_message(req, cancel).await?;
            // Synthesize per-block events so a non-streaming backend
            // flows through the same consumer code path as a streaming
            // one. Callers that only care about Completed still get the
            // canonical assembled content there.
            for block in &resp.content {
                match block {
                    ContentBlock::Text { text } if !text.is_empty() => {
                        yield ModelEvent::TextDelta { text: text.clone() };
                    }
                    ContentBlock::Thinking { thinking, .. } if !thinking.is_empty() => {
                        yield ModelEvent::ThinkingDelta { text: thinking.clone() };
                    }
                    ContentBlock::ToolUse { id, name, input, .. } => {
                        yield ModelEvent::ToolCall {
                            id: id.clone(),
                            name: name.clone(),
                            input: input.clone(),
                        };
                    }
                    ContentBlock::Image { source, .. } => {
                        yield ModelEvent::ImageBlock { source: source.clone() };
                    }
                    _ => {}
                }
            }
            yield ModelEvent::Completed {
                content: resp.content,
                stop_reason: resp.stop_reason,
                usage: resp.usage,
            };
        })
    }

    fn list_models<'a>(&'a self) -> BoxFuture<'a, Result<Vec<ModelInfo>, ModelError>>;

    /// Synchronous best-effort capability lookup for a known model id.
    /// Used at tool-list assembly time to decide whether to advertise
    /// image-producing tools (`view_image`, etc.) to the model.
    ///
    /// Default returns `ContentCapabilities::default()` — text-only.
    /// Providers with per-model vision tables override (Anthropic
    /// claims standard vision; OpenAI / Gemini consult their per-id
    /// tables; llama.cpp delegates to its inner OpenAI-shaped client,
    /// which today won't recognize any vision models — safe-but-
    /// conservative default).
    fn capabilities_for(&self, _model_id: &str) -> ContentCapabilities {
        ContentCapabilities::default()
    }

    /// Synchronous lookup of the backend-specific tunable schema for a
    /// known model id. Used by the thread-creation flow to decide what
    /// controls to render, and to validate that values landing on
    /// [`crate::runtime::scheduler::Scheduler`] correspond to keys this
    /// model actually advertises.
    ///
    /// Default returns an empty vec — backends opt in per-model when
    /// they have knobs worth exposing. Should agree with whatever
    /// `list_models()` returns for the same id (the scheduler treats
    /// `list_models` as the catalog; this method is a fast path for
    /// callers that already know the id).
    fn tunables_for(&self, _model_id: &str) -> Vec<TunableSpec> {
        Vec::new()
    }

    /// Explicitly poll the backend's account/quota endpoint for a
    /// fresh usage snapshot. Triggered by the UI's "refresh" affordance
    /// in the per-backend usage dropdown — distinct from the implicit
    /// header-scrape on `create_message_streaming` paths, which carries
    /// only the bits the upstream attaches to every response (window
    /// utilizations). The poll path carries the richer bits the
    /// header path doesn't include (plan type, credit balance on the
    /// Codex backend).
    ///
    /// Default returns `Ok(None)` — backends that don't expose a
    /// dedicated usage endpoint opt out by not overriding. `Ok(None)`
    /// is also the right return when the upstream returns an empty
    /// payload (e.g. enterprise accounts opted out of the metered
    /// surface) — distinct from `Err(_)` which signals network / auth
    /// failure the UI should surface.
    fn fetch_usage<'a>(
        &'a self,
    ) -> BoxFuture<'a, Result<Option<whisper_agent_protocol::BackendUsage>, ModelError>> {
        Box::pin(async { Ok(None) })
    }

    /// Replace the Codex `auth.json` contents for this provider, writing
    /// the new bytes to the same on-disk path the provider was built
    /// with and swapping the in-memory auth state so subsequent
    /// requests use the new tokens.
    ///
    /// Async because providers whose in-memory auth is guarded by a
    /// `tokio::sync::Mutex` (the Codex flow) need to await the lock
    /// on a runtime thread that might currently be mid-refresh.
    /// Returns `Err(UpdateAuthError::NotSupported)` by default — only
    /// `OpenAiResponsesClient` configured with `ClientAuth::Codex(_)`
    /// overrides.
    fn update_codex_auth<'a>(
        &'a self,
        _contents: &'a str,
    ) -> BoxFuture<'a, Result<(), UpdateAuthError>> {
        Box::pin(async { Err(UpdateAuthError::NotSupported) })
    }
}

/// Failure shape for [`ModelProvider::update_codex_auth`]. Kept as a
/// narrow enum rather than `anyhow::Error` so handlers can tell "not
/// supported" (client error) from "disk write failed" (server error)
/// and map to the right surface.
#[derive(Debug, Error)]
pub enum UpdateAuthError {
    #[error("this backend does not support codex auth rotation")]
    NotSupported,
    #[error("invalid codex auth.json: {0}")]
    Invalid(String),
    #[error("failed to persist codex auth.json: {0}")]
    Io(String),
}

/// Standard JPEG/PNG/WebP/GIF + PDF vision set shared across
/// Anthropic and OpenAI (Chat Completions + Responses). Gemini adds
/// HEIC/HEIF on top of this. PDF input piggybacks on vision: every
/// vision-capable model on these three providers accepts PDFs (the
/// providers do server-side text extraction + page rasterization
/// before the model sees the request), so we group them under one
/// helper.
pub fn standard_vision_capabilities() -> ContentCapabilities {
    let mut input = MediaSupport::standard_image_input();
    input
        .document
        .push(whisper_agent_protocol::DocumentMime::Pdf);
    ContentCapabilities {
        input,
        output: MediaSupport::default(),
    }
}

/// Vision capability lookup for an OpenAI model id. Covers both the
/// Chat Completions and Responses catalogs — they share the same
/// model namespace. Heuristic by id prefix because OpenAI doesn't
/// publish per-model capability flags on `/v1/models`; refine later
/// when the upstream adds one. Conservative when unsure — an unknown
/// id lands as text-only rather than claiming vision that isn't there.
pub fn openai_vision_capabilities(model_id: &str) -> ContentCapabilities {
    let lower = model_id.to_ascii_lowercase();
    let vision = lower.starts_with("gpt-4o")
        || lower.starts_with("gpt-4.1")
        || lower.starts_with("gpt-4-turbo")
        || lower.starts_with("gpt-4-vision")
        || lower.starts_with("gpt-5")
        || lower.starts_with("o1")
        || lower.starts_with("o3")
        || lower.starts_with("o4")
        || lower == "chatgpt-4o-latest";
    if vision {
        standard_vision_capabilities()
    } else {
        ContentCapabilities::default()
    }
}

/// Gemini vision set — adds HEIC/HEIF on top of the standard four
/// since Gemini's inline_data accepts them and we'd otherwise reject
/// valid uploads at the scheduler edge.
pub fn gemini_vision_capabilities() -> ContentCapabilities {
    let mut caps = standard_vision_capabilities();
    caps.input.image.push(ImageMime::Heic);
    caps.input.image.push(ImageMime::Heif);
    caps
}

/// Per-model Gemini capabilities. Always carries the Gemini vision
/// input set; additionally advertises PNG output for the image-
/// generation variants (`gemini-2.5-flash-image*`,
/// `gemini-2.0-flash-preview-image-generation`) so the UI knows the
/// model can return images and the scheduler can populate
/// `responseModalities` correctly.
pub fn gemini_capabilities_for(model_id: &str) -> ContentCapabilities {
    let mut caps = gemini_vision_capabilities();
    if crate::providers::gemini::model_emits_images(model_id) {
        // Gemini's image-output models emit PNG (image/png in the
        // inline_data MIME). They don't currently emit any other
        // image MIMEs in the `image` modality response.
        caps.output.image.push(ImageMime::Png);
    }
    caps
}

/// Drain a `create_message_streaming` stream into a single [`ModelResponse`]
/// — use this when a caller needs batch semantics on top of a streaming
/// provider. Returns the first error if the stream terminates early.
pub async fn buffer_stream<'a>(
    mut stream: BoxStream<'a, Result<ModelEvent, ModelError>>,
) -> Result<ModelResponse, ModelError> {
    while let Some(ev) = stream.next().await {
        match ev? {
            ModelEvent::Completed {
                content,
                stop_reason,
                usage,
            } => {
                return Ok(ModelResponse {
                    content,
                    stop_reason,
                    usage,
                });
            }
            // Deltas are discarded — the canonical content arrives in
            // `Completed`. A caller that wants them should consume the
            // stream directly instead of using this helper.
            _ => continue,
        }
    }
    Err(ModelError::Transport(
        "model stream ended without Completed event".into(),
    ))
}

/// Default cache policy — mirrors Claude Code's observed wire behavior: one
/// breakpoint on the system prompt (stable) plus one rolling breakpoint on the
/// most recent user-side message (where fresh input lands each turn). Tools sit
/// between the two and ride the prefix implicitly — the server extends the cached
/// prefix forward on each turn as long as tool declarations are byte-stable.
///
/// Keeping the total at two markers is intentional: extra breakpoints each
/// create separate cache entries with their own write multipliers (1.25× for 5m,
/// 2× for 1h), and the redundancy only pays off if the longer entry TTLs out
/// mid-session — unlikely at 1h. Two markers is also well under Anthropic's 4-cap.
///
/// "User-side" is `Role::User` or `Role::ToolResult` — anything representing
/// fresh input the model consumes. `Role::Assistant` and `Role::System` are
/// skipped: assistant output is where reasoning continues, not where input
/// lands; mid-conversation system injections are harness guidance whose
/// content can churn independently of user turns and shouldn't hijack the
/// rolling breakpoint away from the stable user-anchored prefix.
pub fn default_cache_policy(messages: &[Message]) -> Vec<CacheBreakpoint> {
    use whisper_agent_protocol::Role;
    let mut out = vec![CacheBreakpoint::AfterSystem];
    if let Some(last) = messages
        .iter()
        .enumerate()
        .rfind(|(_, m)| !matches!(m.role, Role::Assistant | Role::System))
        .map(|(i, _)| i)
    {
        out.push(CacheBreakpoint::AfterMessage(last));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use whisper_agent_protocol::{ContentBlock, Message, ToolResultContent};

    fn tr(id: &str, text: &str) -> Message {
        Message::tool_result_blocks(vec![ContentBlock::ToolResult {
            tool_use_id: id.into(),
            content: ToolResultContent::Text(text.into()),
            is_error: false,
        }])
    }

    #[test]
    fn rolling_breakpoint_tracks_last_tool_result() {
        // Agentic loop: one user prompt followed by several assistant/tool_result
        // cycles. The rolling breakpoint must land on the last user-side message
        // (the trailing tool_result at index 6), not stall on the initial user
        // text at index 0.
        let msgs = vec![
            Message::user_text("start"),
            Message::assistant_blocks(vec![ContentBlock::Text { text: "a".into() }]),
            tr("t1", "r1"),
            Message::assistant_blocks(vec![ContentBlock::Text { text: "b".into() }]),
            tr("t2", "r2"),
            Message::assistant_blocks(vec![ContentBlock::Text { text: "c".into() }]),
            tr("t3", "r3"),
        ];
        let bps = default_cache_policy(&msgs);
        assert_eq!(
            bps,
            vec![
                CacheBreakpoint::AfterSystem,
                CacheBreakpoint::AfterMessage(6)
            ]
        );
    }

    #[test]
    fn empty_conversation_yields_system_marker_only() {
        // No messages (ScheduleWakeup from a fresh thread, seeded systems, etc.) —
        // nothing to anchor a rolling breakpoint on, so only the system marker.
        let bps = default_cache_policy(&[]);
        assert_eq!(bps, vec![CacheBreakpoint::AfterSystem]);
    }

    #[test]
    fn injected_system_message_is_skipped_for_rolling_breakpoint() {
        // A mid-conversation Role::System (e.g. a memory-index injection
        // placed just before the next send) should not capture the
        // rolling breakpoint — the anchor belongs on the real user turn
        // so cache lookups stay stable across injections whose content
        // may churn.
        let msgs = vec![
            Message::user_text("hi"),
            Message::assistant_blocks(vec![ContentBlock::Text { text: "hey".into() }]),
            Message::user_text("do X"),
            Message::system_text("<harness note>"),
        ];
        let bps = default_cache_policy(&msgs);
        assert_eq!(
            bps,
            vec![
                CacheBreakpoint::AfterSystem,
                CacheBreakpoint::AfterMessage(2)
            ]
        );
    }

    #[test]
    fn is_transient_classification() {
        assert!(ModelError::Transport("read timeout".into()).is_transient());
        // 5xx server errors — retry.
        assert!(ModelError::api(500, "").is_transient());
        assert!(ModelError::api(502, "").is_transient());
        assert!(ModelError::api(503, "").is_transient());
        assert!(ModelError::api(529, "").is_transient());
        // 408 Request Timeout — retry.
        assert!(ModelError::api(408, "").is_transient());
        // Client-side 4xx — don't retry (auth, bad request, context exhausted).
        assert!(!ModelError::api(400, "").is_transient());
        assert!(!ModelError::api(401, "").is_transient());
        assert!(!ModelError::api(403, "").is_transient());
        assert!(!ModelError::api(404, "").is_transient());
        // 429 is carried by RateLimited, not Api — but even if it showed up as Api,
        // rate-limit retry is driven by the dedicated variant's retry_after path.
        assert!(
            !ModelError::RateLimited {
                retry_after: Duration::from_secs(2),
                body: String::new()
            }
            .is_transient()
        );
    }

    #[test]
    fn trailing_assistant_message_is_not_a_breakpoint() {
        // Trailing assistant message (shouldn't normally happen at dispatch
        // time, but the policy must handle it gracefully): walk back past it
        // to the nearest user-side anchor.
        let msgs = vec![
            Message::user_text("hi"),
            Message::assistant_blocks(vec![ContentBlock::Text { text: "hey".into() }]),
        ];
        let bps = default_cache_policy(&msgs);
        assert_eq!(
            bps,
            vec![
                CacheBreakpoint::AfterSystem,
                CacheBreakpoint::AfterMessage(0)
            ]
        );
    }
}
