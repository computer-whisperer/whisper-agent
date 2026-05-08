//! Model providers: one submodule per backend, plus the shared trait.
//!
//! [`model::ModelProvider`] is the object-safe interface every chat backend
//! implements; each sibling module (`anthropic`, `openai_chat`,
//! `openai_responses`, `gemini`, `llamacpp`) speaks its own wire shape and
//! adapts to the shared [`model::ModelRequest`] / [`model::ModelResponse`]
//! types.
//!
//! [`embedding::EmbeddingProvider`] and [`rerank::RerankProvider`] mirror
//! the same pattern for retrieval-side workloads. Today only [`tei`]
//! implements them; OpenAI / Cohere / Voyage hookups plug in alongside.
//!
//! `gemini_auth` is a reader for the OAuth token store the Gemini CLI drops
//! on disk, so we can piggyback on an already-authenticated user session
//! without running our own OAuth flow. The corresponding Codex (ChatGPT
//! subscription) reader lives in the shared `whisper-agent-auth` crate so
//! sibling MCP daemons can use it too.

pub mod anthropic;
pub mod embedding;
pub mod gemini;
pub mod gemini_auth;
pub mod llamacpp;
pub mod model;
pub mod openai_chat;
pub mod openai_responses;
pub mod rerank;
pub mod tei;
