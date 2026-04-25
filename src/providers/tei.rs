//! Hugging Face Text Embeddings Inference (TEI) provider.
//!
//! TEI is a single-model HTTP server: each running instance serves either an
//! embedding model **or** a reranker, never both. So we expose two distinct
//! client types — [`TeiEmbeddingClient`] implements
//! [`crate::providers::embedding::EmbeddingProvider`], and
//! [`TeiRerankClient`] implements
//! [`crate::providers::rerank::RerankProvider`]. The two TEI endpoints in a
//! deployment are registered separately in
//! `whisper-agent.toml` and the scheduler holds them in symmetric registries.
//!
//! Native TEI wire shape (not the OpenAI-compat alias):
//! - `POST /embed` — `{"inputs": [..], "truncate": bool, "normalize": bool}` → `[[f32], ...]`
//! - `POST /rerank` — `{"query": "..", "texts": [..], "top_n": ?, "return_text": false}`
//!   returns `[{"index": u32, "score": f32}, ...]`
//! - `GET /info` — model card (`model_id`, `max_input_length`, ...)
//!
//! Embedding dimension is determined by probing `/embed` with a single
//! trivial input the first time `list_models` is called. TEI's `/info`
//! endpoint reports `hidden_size` for embedding deployments, but the field
//! shape has shifted across TEI versions, so the probe is the authoritative
//! source. The result is cached on the client.

use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::sync::OnceCell;
use tokio_util::sync::CancellationToken;

use crate::providers::embedding::{
    BoxFuture as EmbedBoxFuture, EmbedRequest, EmbeddingError, EmbeddingModelInfo,
    EmbeddingProvider, EmbeddingResponse,
};
use crate::providers::rerank::{
    BoxFuture as RerankBoxFuture, RerankError, RerankModelInfo, RerankProvider, RerankRequest,
    RerankResponse, RerankResult,
};

/// Shared low-level HTTP wrapper. The two trait impls go through this for
/// `/info` lookups and request building so auth + cancellation handling
/// stays in one place.
struct TeiHttp {
    /// Server origin without trailing slash, e.g. `http://localhost:8080`.
    /// Endpoint paths are appended directly (`{origin}/embed`).
    origin: String,
    /// Optional bearer token. `None` when the deployment was started
    /// without `--api-key`. TEI compares this against the
    /// `Authorization: Bearer <key>` header on every request.
    api_key: Option<String>,
    http: reqwest::Client,
}

impl TeiHttp {
    fn new(endpoint: String, api_key: Option<String>) -> Self {
        Self {
            origin: endpoint.trim_end_matches('/').to_string(),
            api_key,
            http: reqwest::Client::new(),
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.origin, path)
    }

    fn apply_auth(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.api_key {
            Some(k) => builder.bearer_auth(k),
            None => builder,
        }
    }

    /// Send a JSON request body to `path` and parse the JSON response. The
    /// `Err` shape is generic so call sites can map into their own provider
    /// error type. `429` is split out as a dedicated outcome so callers
    /// can build the right error variant (`RateLimited` carries
    /// `retry_after`, which we extract from the `Retry-After` header here).
    async fn post_json<Req, Resp>(
        &self,
        path: &str,
        body: &Req,
        cancel: &CancellationToken,
    ) -> Result<Resp, TeiHttpError>
    where
        Req: Serialize,
        Resp: for<'de> Deserialize<'de>,
    {
        let url = self.url(path);
        let builder = self.http.post(&url).json(body);
        let send = self.apply_auth(builder).send();
        let resp = tokio::select! {
            r = send => r.map_err(|e| TeiHttpError::Transport(e.to_string()))?,
            _ = cancel.cancelled() => return Err(TeiHttpError::Cancelled),
        };
        let status = resp.status();
        if status.as_u16() == 429 {
            let retry_after = parse_retry_after(resp.headers());
            let body = tokio::select! {
                b = resp.text() => b.unwrap_or_default(),
                _ = cancel.cancelled() => return Err(TeiHttpError::Cancelled),
            };
            return Err(TeiHttpError::RateLimited { retry_after, body });
        }
        if !status.is_success() {
            let body = tokio::select! {
                b = resp.text() => b.unwrap_or_default(),
                _ = cancel.cancelled() => return Err(TeiHttpError::Cancelled),
            };
            return Err(TeiHttpError::Api {
                status: status.as_u16(),
                body,
            });
        }
        let parsed = tokio::select! {
            r = resp.json::<Resp>() => r.map_err(|e| TeiHttpError::Transport(e.to_string()))?,
            _ = cancel.cancelled() => return Err(TeiHttpError::Cancelled),
        };
        Ok(parsed)
    }

    /// Fetch and parse `/info`. Returns `None` on any failure — `/info` is
    /// best-effort metadata used to populate `EmbeddingModelInfo` /
    /// `RerankModelInfo`, so we'd rather report partial info than fail
    /// `list_models`. Cancellation aware so a slow `/info` doesn't pin a
    /// caller that's already given up.
    async fn fetch_info(&self, cancel: &CancellationToken) -> Option<TeiInfo> {
        let url = self.url("/info");
        let builder = self.http.get(&url);
        let send = self.apply_auth(builder).send();
        let resp = tokio::select! {
            r = send => r.ok()?,
            _ = cancel.cancelled() => return None,
        };
        if !resp.status().is_success() {
            return None;
        }
        tokio::select! {
            r = resp.json::<TeiInfo>() => r.ok(),
            _ = cancel.cancelled() => None,
        }
    }
}

/// Internal error type covering every wire-level failure mode of a TEI
/// HTTP call. The two trait impls (`EmbeddingProvider` /
/// `RerankProvider`) translate this into their own error enums.
enum TeiHttpError {
    Transport(String),
    Api { status: u16, body: String },
    RateLimited { retry_after: Duration, body: String },
    Cancelled,
}

impl TeiHttpError {
    fn into_embedding(self) -> EmbeddingError {
        match self {
            TeiHttpError::Transport(s) => EmbeddingError::Transport(s),
            TeiHttpError::Api { status, body } => EmbeddingError::Api { status, body },
            TeiHttpError::RateLimited { retry_after, body } => {
                EmbeddingError::RateLimited { retry_after, body }
            }
            TeiHttpError::Cancelled => EmbeddingError::Cancelled,
        }
    }

    fn into_rerank(self) -> RerankError {
        match self {
            TeiHttpError::Transport(s) => RerankError::Transport(s),
            TeiHttpError::Api { status, body } => RerankError::Api { status, body },
            TeiHttpError::RateLimited { retry_after, body } => {
                RerankError::RateLimited { retry_after, body }
            }
            TeiHttpError::Cancelled => RerankError::Cancelled,
        }
    }
}

/// Parse the `Retry-After` HTTP header. Two formats per RFC 7231: a
/// delta-seconds integer or an HTTP-date. We accept the integer form and
/// fall back to a default when the value is missing or unparseable —
/// HTTP-date in 429 responses is rare in practice for ML inference
/// servers.
fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Duration {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or(Duration::from_secs(1))
}

/// Subset of TEI's `/info` we read. Every field is optional because the
/// shape has churned across TEI versions; missing fields land as `None`
/// in the model-info struct rather than failing the call.
#[derive(Deserialize, Debug, Default)]
struct TeiInfo {
    #[serde(default)]
    model_id: Option<String>,
    #[serde(default)]
    max_input_length: Option<u32>,
}

// ---------------------------------------------------------------------
// Embedding client
// ---------------------------------------------------------------------

pub struct TeiEmbeddingClient {
    http: Arc<TeiHttp>,
    /// Lazily-resolved output dimension. Populated on the first
    /// `list_models` call by issuing a one-input probe to `/embed`. Once
    /// resolved it's stable for the lifetime of the client (TEI doesn't
    /// hot-swap models — a model change requires restarting the TEI
    /// server, which the scheduler observes as a config edit).
    dimension: OnceCell<u32>,
}

impl TeiEmbeddingClient {
    pub fn new(endpoint: String, api_key: Option<String>) -> Self {
        Self {
            http: Arc::new(TeiHttp::new(endpoint, api_key)),
            dimension: OnceCell::new(),
        }
    }

    /// Probe the embedding dimension by submitting a single-element batch
    /// of trivial input and reading the result vector's length. Cached
    /// in `self.dimension` after the first successful probe.
    async fn resolve_dimension(&self, cancel: &CancellationToken) -> Result<u32, EmbeddingError> {
        if let Some(d) = self.dimension.get() {
            return Ok(*d);
        }
        let req = TeiEmbedRequest {
            inputs: vec![" ".to_string()],
            truncate: true,
            normalize: false,
        };
        let resp: Vec<Vec<f32>> = self
            .http
            .post_json("/embed", &req, cancel)
            .await
            .map_err(TeiHttpError::into_embedding)?;
        let first = resp
            .into_iter()
            .next()
            .ok_or_else(|| EmbeddingError::Transport("/embed probe returned no vectors".into()))?;
        let dim = u32::try_from(first.len()).map_err(|_| {
            EmbeddingError::Transport(format!(
                "/embed probe returned vector with {} dims (exceeds u32)",
                first.len()
            ))
        })?;
        if dim == 0 {
            return Err(EmbeddingError::Transport(
                "/embed probe returned a zero-length vector".into(),
            ));
        }
        let _ = self.dimension.set(dim);
        Ok(dim)
    }
}

#[derive(Serialize)]
struct TeiEmbedRequest {
    inputs: Vec<String>,
    truncate: bool,
    /// L2-normalize embeddings server-side. Normalized vectors make
    /// cosine similarity equivalent to dot product, which is the
    /// distance metric most ANN indexes default to. We always ask TEI
    /// for normalized output so downstream index code doesn't have to
    /// renormalize.
    normalize: bool,
}

impl EmbeddingProvider for TeiEmbeddingClient {
    fn embed<'a>(
        &'a self,
        req: &'a EmbedRequest<'a>,
        cancel: &'a CancellationToken,
    ) -> EmbedBoxFuture<'a, Result<EmbeddingResponse, EmbeddingError>> {
        Box::pin(async move {
            if req.inputs.is_empty() {
                return Ok(EmbeddingResponse {
                    embeddings: Vec::new(),
                    usage: None,
                });
            }
            let body = TeiEmbedRequest {
                inputs: req.inputs.to_vec(),
                truncate: true,
                normalize: true,
            };
            let vectors: Vec<Vec<f32>> = self
                .http
                .post_json("/embed", &body, cancel)
                .await
                .map_err(TeiHttpError::into_embedding)?;
            if vectors.len() != req.inputs.len() {
                return Err(EmbeddingError::Api {
                    status: 0,
                    body: format!(
                        "/embed returned {} vectors for {} inputs",
                        vectors.len(),
                        req.inputs.len()
                    ),
                });
            }
            // Validate dimensional uniformity. A ragged response would
            // silently mix vectors of different shapes into the bucket
            // index — much harder to debug downstream than a clean
            // failure here.
            if let Some(first) = vectors.first() {
                let dim = first.len();
                if let Some(bad) = vectors.iter().find(|v| v.len() != dim) {
                    return Err(EmbeddingError::Api {
                        status: 0,
                        body: format!(
                            "/embed returned ragged vectors: first={}, found={}",
                            dim,
                            bad.len()
                        ),
                    });
                }
                let dim_u32 = u32::try_from(dim)
                    .map_err(|_| EmbeddingError::Transport("vector dim exceeds u32".into()))?;
                let _ = self.dimension.set(dim_u32);
            }
            Ok(EmbeddingResponse {
                embeddings: vectors,
                usage: None,
            })
        })
    }

    fn list_models<'a>(
        &'a self,
    ) -> EmbedBoxFuture<'a, Result<Vec<EmbeddingModelInfo>, EmbeddingError>> {
        Box::pin(async move {
            // Use a fresh cancellation token — `list_models` is called
            // from settings handlers, not from a thread context, so
            // there's no per-thread cancel to propagate. A future
            // scheduler-level shutdown signal can replace this if we
            // need to abort metadata fetches on stop.
            let cancel = CancellationToken::new();
            let info = self.http.fetch_info(&cancel).await.unwrap_or_default();
            let dimension = self.resolve_dimension(&cancel).await?;
            let id = info.model_id.unwrap_or_else(|| "tei".to_string());
            Ok(vec![EmbeddingModelInfo {
                id,
                dimension,
                max_input_tokens: info.max_input_length,
            }])
        })
    }
}

// ---------------------------------------------------------------------
// Rerank client
// ---------------------------------------------------------------------

pub struct TeiRerankClient {
    http: Arc<TeiHttp>,
}

impl TeiRerankClient {
    pub fn new(endpoint: String, api_key: Option<String>) -> Self {
        Self {
            http: Arc::new(TeiHttp::new(endpoint, api_key)),
        }
    }
}

#[derive(Serialize)]
struct TeiRerankRequest<'a> {
    query: &'a str,
    texts: Vec<String>,
    /// Ask the server not to echo document text back in the response —
    /// we have it locally already and the round-trip just bloats latency.
    return_text: bool,
    /// `None` ⇒ omit, server returns scores for every document.
    #[serde(skip_serializing_if = "Option::is_none")]
    top_n: Option<u32>,
}

#[derive(Deserialize, Debug)]
struct TeiRerankResult {
    index: u32,
    score: f32,
}

impl RerankProvider for TeiRerankClient {
    fn rerank<'a>(
        &'a self,
        req: &'a RerankRequest<'a>,
        cancel: &'a CancellationToken,
    ) -> RerankBoxFuture<'a, Result<RerankResponse, RerankError>> {
        Box::pin(async move {
            if req.documents.is_empty() {
                return Ok(RerankResponse {
                    results: Vec::new(),
                });
            }
            let body = TeiRerankRequest {
                query: req.query,
                texts: req.documents.to_vec(),
                return_text: false,
                top_n: req.top_n,
            };
            let raw: Vec<TeiRerankResult> = self
                .http
                .post_json("/rerank", &body, cancel)
                .await
                .map_err(TeiHttpError::into_rerank)?;
            // TEI returns results sorted by score descending already;
            // re-sort defensively in case a future version changes that.
            let mut results: Vec<RerankResult> = raw
                .into_iter()
                .map(|r| RerankResult {
                    index: r.index,
                    score: r.score,
                })
                .collect();
            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            Ok(RerankResponse { results })
        })
    }

    fn list_models<'a>(&'a self) -> RerankBoxFuture<'a, Result<Vec<RerankModelInfo>, RerankError>> {
        Box::pin(async move {
            let cancel = CancellationToken::new();
            let info = self.http.fetch_info(&cancel).await.unwrap_or_default();
            let id = info.model_id.unwrap_or_else(|| "tei".to_string());
            Ok(vec![RerankModelInfo {
                id,
                max_input_tokens: info.max_input_length,
            }])
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_retry_after_integer() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::RETRY_AFTER,
            reqwest::header::HeaderValue::from_static("17"),
        );
        assert_eq!(parse_retry_after(&headers), Duration::from_secs(17));
    }

    #[test]
    fn parse_retry_after_missing_falls_back() {
        let headers = reqwest::header::HeaderMap::new();
        assert_eq!(parse_retry_after(&headers), Duration::from_secs(1));
    }

    #[test]
    fn parse_retry_after_unparseable_falls_back() {
        // HTTP-date form — RFC-legal but we don't parse it; the
        // fallback default is acceptable for our use.
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::RETRY_AFTER,
            reqwest::header::HeaderValue::from_static("Wed, 21 Oct 2026 07:28:00 GMT"),
        );
        assert_eq!(parse_retry_after(&headers), Duration::from_secs(1));
    }

    #[test]
    fn tei_info_default_is_all_none() {
        let info = TeiInfo::default();
        assert!(info.model_id.is_none());
        assert!(info.max_input_length.is_none());
    }

    #[test]
    fn parses_real_info_payload() {
        // Trimmed real TEI /info shape; fields beyond what we read are
        // ignored by serde default.
        let json = r#"{
            "model_id": "BAAI/bge-base-en-v1.5",
            "model_dtype": "float32",
            "max_input_length": 512,
            "max_concurrent_requests": 512,
            "max_batch_tokens": 16384,
            "version": "1.5.0"
        }"#;
        let info: TeiInfo = serde_json::from_str(json).unwrap();
        assert_eq!(info.model_id.as_deref(), Some("BAAI/bge-base-en-v1.5"));
        assert_eq!(info.max_input_length, Some(512));
    }

    #[test]
    fn parses_rerank_results_payload() {
        let json = r#"[
            {"index": 2, "score": 0.987},
            {"index": 0, "score": 0.654},
            {"index": 1, "score": 0.123}
        ]"#;
        let parsed: Vec<TeiRerankResult> = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[0].index, 2);
        assert!((parsed[0].score - 0.987).abs() < 1e-6);
    }

    #[test]
    fn http_error_into_embedding_preserves_variants() {
        match TeiHttpError::Transport("nope".into()).into_embedding() {
            EmbeddingError::Transport(s) => assert_eq!(s, "nope"),
            _ => panic!("wrong variant"),
        }
        match (TeiHttpError::Api {
            status: 502,
            body: "x".into(),
        })
        .into_embedding()
        {
            EmbeddingError::Api { status, body } => {
                assert_eq!(status, 502);
                assert_eq!(body, "x");
            }
            _ => panic!("wrong variant"),
        }
        match TeiHttpError::Cancelled.into_embedding() {
            EmbeddingError::Cancelled => {}
            _ => panic!("wrong variant"),
        }
    }
}
