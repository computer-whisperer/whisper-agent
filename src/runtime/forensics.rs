//! Post-mortem disk capture for terminal model-call failures.
//!
//! When io_dispatch exhausts the retry budget on a model call, it
//! hands the resulting [`crate::providers::model::ModelError`] to a
//! [`ForensicSink`]. If the error carries a captured
//! [`crate::providers::model::ForensicArtifact`], the sink serializes
//! the artifact + the structured detail + identifying metadata into a
//! single JSON file under the configured directory. An operator can
//! later replay the failed request against the provider verbatim from
//! that file.
//!
//! Disabled when the configured directory is empty — useful for
//! single-binary dev runs that don't want disk side effects. Structured
//! tracing in io_dispatch still fires either way.
//!
//! File naming: `<timestamp>-<backend>-<thread-id>-<short>.json`.
//! `<short>` is a four-byte hex suffix per dump so simultaneous
//! failures from two threads don't collide; we don't otherwise rotate
//! or prune. Operators are expected to wire log shipping / retention
//! externally (the dump directory mirrors the audit-log lifecycle).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::Utc;
use serde::Serialize;
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;
use tracing::warn;

use crate::providers::model::{ForensicArtifact, ModelError, ProviderErrorDetail};

/// Disk dump writer for terminal model-call failures. Cheap to clone
/// (Arc'd path); a `None` inner path means "disabled — drop every
/// dump on the floor."
#[derive(Clone, Debug)]
pub struct ForensicSink {
    dir: Option<Arc<PathBuf>>,
}

impl ForensicSink {
    /// Construct a sink rooted at `dir`. The directory is created on
    /// first use; permissions errors there will be logged but not
    /// surfaced (the model-call path shouldn't fail because we
    /// couldn't write a dump).
    pub fn new(dir: PathBuf) -> Self {
        Self {
            dir: Some(Arc::new(dir)),
        }
    }

    /// Construct a no-op sink. Used for tests and for dev runs where
    /// `--forensics-dir ""` opts out of disk capture.
    pub fn disabled() -> Self {
        Self { dir: None }
    }

    pub fn is_enabled(&self) -> bool {
        self.dir.is_some()
    }

    /// Write a dump for a terminal model-call failure, if the sink is
    /// enabled and the error carries a forensic artifact. Logs the
    /// resulting path at WARN so an operator scanning logs can find
    /// the file directly. Never errors — disk failures are best-effort.
    pub async fn record_terminal_failure(
        &self,
        thread_id: &str,
        backend_name: &str,
        op_id: u64,
        error: &ModelError,
    ) {
        let Some(dir) = self.dir.as_deref() else {
            return;
        };
        let Some(artifact) = error.forensic() else {
            return;
        };
        let detail = error.provider_detail().cloned().unwrap_or_default();
        let envelope =
            ForensicEnvelope::build(thread_id, backend_name, op_id, error, artifact, &detail);
        let path = match envelope_path(dir, &envelope).await {
            Ok(p) => p,
            Err(e) => {
                warn!(
                    %thread_id, backend = %backend_name,
                    error = %e,
                    "could not prepare forensic dump directory; skipping"
                );
                return;
            }
        };
        let bytes = match serde_json::to_vec_pretty(&envelope) {
            Ok(b) => b,
            Err(e) => {
                warn!(
                    %thread_id, backend = %backend_name,
                    error = %e,
                    "could not serialize forensic envelope; skipping"
                );
                return;
            }
        };
        match write_atomic(&path, &bytes).await {
            Ok(()) => {
                warn!(
                    %thread_id, backend = %backend_name,
                    forensic_path = %path.display(),
                    "forensic dump written for terminal model-call failure"
                );
            }
            Err(e) => {
                warn!(
                    %thread_id, backend = %backend_name,
                    error = %e,
                    "could not write forensic dump; skipping"
                );
            }
        }
    }
}

/// Top-level JSON shape written to a forensic dump file.
///
/// `request_body` is `String` (typically JSON) and lands inline; a
/// reader can `jq .request_body | jq .` to inspect. `response_excerpt`
/// is captured as UTF-8 when possible — adapters keep it as bytes only
/// to survive stream truncation mid-codepoint; the actual provider
/// wire bytes are virtually always UTF-8.
#[derive(Serialize)]
struct ForensicEnvelope {
    schema: &'static str,
    timestamp: String,
    thread_id: String,
    op_id: u64,
    backend_name: String,
    provider_backend: String,
    model: String,
    error: ErrorBlock,
    request_content_type: String,
    request_body: String,
    response_excerpt: String,
    response_excerpt_was_lossy_utf8: bool,
    response_truncated: bool,
}

#[derive(Serialize)]
struct ErrorBlock {
    /// `ModelError::to_string()` — the user-visible message.
    summary: String,
    detail: ProviderErrorDetail,
}

impl ForensicEnvelope {
    fn build(
        thread_id: &str,
        backend_name: &str,
        op_id: u64,
        error: &ModelError,
        artifact: &ForensicArtifact,
        detail: &ProviderErrorDetail,
    ) -> Self {
        let (excerpt, lossy) = decode_excerpt(&artifact.response_excerpt);
        Self {
            schema: "whisper-agent.forensic.v1",
            timestamp: Utc::now().to_rfc3339(),
            thread_id: thread_id.to_string(),
            op_id,
            backend_name: backend_name.to_string(),
            provider_backend: artifact.backend.clone(),
            model: artifact.model.clone(),
            error: ErrorBlock {
                summary: error.to_string(),
                detail: detail.clone(),
            },
            request_content_type: artifact.request_content_type.to_string(),
            request_body: artifact.request_body.clone(),
            response_excerpt: excerpt,
            response_excerpt_was_lossy_utf8: lossy,
            response_truncated: artifact.response_truncated,
        }
    }
}

fn decode_excerpt(bytes: &[u8]) -> (String, bool) {
    match std::str::from_utf8(bytes) {
        Ok(s) => (s.to_string(), false),
        Err(_) => (String::from_utf8_lossy(bytes).into_owned(), true),
    }
}

async fn envelope_path(dir: &Path, env: &ForensicEnvelope) -> std::io::Result<PathBuf> {
    tokio::fs::create_dir_all(dir).await?;
    // Filesystem-safe variant of the timestamp (avoid colons in
    // `12:34:56` on case-insensitive filesystems / Windows shares).
    let stamp = Utc::now().format("%Y%m%dT%H%M%S%.3fZ").to_string();
    // Sanitize the thread id and backend name so a colon, slash, or
    // other surprise character can't escape the dump dir. Cap each
    // to avoid pathologically long filenames from a stray header.
    let backend = sanitize(&env.backend_name)
        .chars()
        .take(32)
        .collect::<String>();
    let thread = sanitize(&env.thread_id)
        .chars()
        .take(40)
        .collect::<String>();
    // Short random suffix so two simultaneous failures from the same
    // thread/backend don't collide on the same millisecond.
    let suffix = format!("{:08x}", rand::random::<u32>());
    let name = format!("{stamp}-{backend}-{thread}-{suffix}.json");
    Ok(dir.join(name))
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' => c,
            _ => '_',
        })
        .collect()
}

/// Write `bytes` to `path` via a `.tmp` sibling + rename so a
/// half-written dump never appears under its final name. Best-effort
/// fsync of the tempfile before rename — we want the bytes durable
/// before an operator might cite the path in an incident review.
async fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("json.tmp");
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&tmp)
        .await?;
    file.write_all(bytes).await?;
    file.flush().await?;
    file.sync_all().await.ok();
    drop(file);
    tokio::fs::rename(&tmp, path).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::model::ForensicContext;

    fn sample_error_with_artifact() -> ModelError {
        let ctx = ForensicContext {
            backend: "openai_responses".into(),
            model: "gpt-5".into(),
            request_body: r#"{"model":"gpt-5","messages":[]}"#.into(),
            request_content_type: "application/json",
        };
        let artifact = ctx.into_http_error_artifact(b"server boom".to_vec());
        ModelError::Api {
            status: 500,
            body: "boom".into(),
            detail: Some(Box::new(ProviderErrorDetail {
                code: Some("server_error".into()),
                kind: Some("server_error".into()),
                response_id: Some("resp_abc".into()),
                message: Some("Sorry, something went wrong".into()),
            })),
            forensic: Some(Arc::new(artifact)),
        }
    }

    #[tokio::test]
    async fn disabled_sink_drops_dumps_silently() {
        let sink = ForensicSink::disabled();
        let err = sample_error_with_artifact();
        // Must not panic, must not write anywhere.
        sink.record_terminal_failure("task-1", "openai-sub", 42, &err)
            .await;
        assert!(!sink.is_enabled());
    }

    #[tokio::test]
    async fn enabled_sink_writes_envelope_with_all_fields_populated() {
        let dir = tempfile::tempdir().unwrap();
        let sink = ForensicSink::new(dir.path().to_path_buf());
        let err = sample_error_with_artifact();
        sink.record_terminal_failure("task-18b03cdfe997bd9b", "openai-subscription", 469, &err)
            .await;

        // Exactly one .json file in the dump dir.
        let mut entries = tokio::fs::read_dir(dir.path()).await.unwrap();
        let mut files = Vec::new();
        while let Some(e) = entries.next_entry().await.unwrap() {
            files.push(e.path());
        }
        assert_eq!(files.len(), 1, "expected one dump, got {files:?}");
        let body = tokio::fs::read_to_string(&files[0]).await.unwrap();
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["schema"], "whisper-agent.forensic.v1");
        assert_eq!(v["thread_id"], "task-18b03cdfe997bd9b");
        assert_eq!(v["op_id"], 469);
        assert_eq!(v["backend_name"], "openai-subscription");
        assert_eq!(v["provider_backend"], "openai_responses");
        assert_eq!(v["model"], "gpt-5");
        assert_eq!(v["error"]["detail"]["code"], "server_error");
        assert_eq!(v["error"]["detail"]["response_id"], "resp_abc");
        assert!(v["request_body"].as_str().unwrap().contains("gpt-5"));
        assert!(
            v["response_excerpt"]
                .as_str()
                .unwrap()
                .contains("server boom")
        );
        assert_eq!(v["response_excerpt_was_lossy_utf8"], false);
        assert_eq!(v["response_truncated"], false);
    }

    #[tokio::test]
    async fn enabled_sink_skips_errors_without_forensic_artifact() {
        let dir = tempfile::tempdir().unwrap();
        let sink = ForensicSink::new(dir.path().to_path_buf());
        // Transport error has no forensic — sink should write nothing.
        let err = ModelError::Transport("dns failure".into());
        sink.record_terminal_failure("task-1", "openai", 1, &err)
            .await;
        let mut entries = tokio::fs::read_dir(dir.path()).await.unwrap();
        assert!(entries.next_entry().await.unwrap().is_none());
    }

    #[test]
    fn sanitize_strips_path_separators_and_unicode() {
        assert_eq!(sanitize("task-18b03"), "task-18b03");
        assert_eq!(sanitize("foo/bar"), "foo_bar");
        assert_eq!(sanitize("a:b\\c"), "a_b_c");
        // `sanitize` walks Unicode `char`s (not bytes), so a single
        // emoji collapses to a single underscore.
        assert_eq!(sanitize("emo-ji-🦀"), "emo-ji-_");
    }
}
