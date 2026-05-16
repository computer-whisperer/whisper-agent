//! Live runtime auth: built once per backend session, prepares the bearer
//! token (refreshing first if needed) and any extra HTTP headers (e.g.
//! `chatgpt-account-id` for the Codex subscription path).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use tokio::sync::Mutex;

use crate::{Auth, CodexAuth};

/// Why a [`CodexAuthSlot::update_from_contents`] call failed.
/// Separating the validation failure from the persistence failure
/// lets consumers (notably the provider's `update_codex_auth` trait
/// impl) map cleanly into their own error enums without inspecting
/// message text.
#[derive(Debug, thiserror::Error)]
pub enum SlotUpdateError {
    #[error("invalid codex auth payload: {0}")]
    Invalid(String),
    #[error("persist codex auth: {0}")]
    Io(String),
}

/// Runtime auth material for an OpenAI-compatible client. The wire client
/// itself is cheap and shared; `ClientAuth` is what differs between an
/// API-key session and a ChatGPT-subscription session.
pub enum ClientAuth {
    ApiKey(String),
    /// ChatGPT OAuth tokens, typically loaded from `~/.codex/auth.json`.
    /// Wrapped in a [`CodexAuthSlot`] so the credential can be empty at
    /// startup — that's the legitimate transient state for a backend
    /// whose tokens are managed by a remote daemon and hasn't seen its
    /// first publication yet.
    Codex(Arc<CodexAuthSlot>),
}

/// On-disk-aware container for a Codex credential.
///
/// Holds a fixed `path` (the credential file the consumer is supposed
/// to read/write) plus an `Option<CodexAuth>` for the in-memory live
/// view. The two are designed to stay in sync:
///
/// - On rotation (admin paste or daemon publication), [`update_from_contents`]
///   parses, persists to disk, then swaps the in-memory state in one
///   critical section. A disk-write failure leaves the in-memory state
///   untouched, so a successful caller can rely on "if Some, the file
///   matches" without an extra read.
/// - When `state` is `None` and a request lands, [`prepare`] errors
///   with a clear "credential not yet available" diagnostic that
///   names the path — the operator sees exactly which file the
///   backend is waiting for.
pub struct CodexAuthSlot {
    path: PathBuf,
    state: Mutex<Option<CodexAuth>>,
}

impl CodexAuthSlot {
    /// Slot already populated from disk. The slot's path matches the
    /// `auth`'s recorded path, so a later [`update_from_contents`]
    /// rewrites the same file the load came from.
    pub fn loaded(auth: CodexAuth) -> Self {
        let path = auth.path().to_path_buf();
        Self {
            path,
            state: Mutex::new(Some(auth)),
        }
    }

    /// Slot with no credential yet — the path is recorded so the
    /// first publication can land in the right place. Requests
    /// against the empty slot fail with a clear diagnostic until
    /// populated.
    pub fn empty(path: PathBuf) -> Self {
        Self {
            path,
            state: Mutex::new(None),
        }
    }

    /// On-disk path the slot reads and writes.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns `(bearer_token, extra_headers)` for the next outbound
    /// request. Refreshes the access token if it's near expiry —
    /// callers don't need to do anything special.
    pub async fn prepare(
        &self,
        http: &reqwest::Client,
    ) -> Result<(String, Vec<(&'static str, String)>)> {
        let mut guard = self.state.lock().await;
        let auth = guard.as_mut().ok_or_else(|| {
            anyhow!(
                "codex auth credential not yet available — awaiting publication at {}",
                self.path.display()
            )
        })?;
        auth.ensure_fresh(http)
            .await
            .context("codex auth refresh")?;
        let mut extras = Vec::new();
        if let Some(acc) = auth.chatgpt_account_id() {
            extras.push(("chatgpt-account-id", acc.to_string()));
        }
        Ok((auth.access_token().to_string(), extras))
    }

    /// Validate raw `contents` (the literal `auth.json` blob),
    /// persist to `self.path`, and swap the in-memory state. Errors
    /// from either step leave the previous state untouched.
    ///
    /// Persists *before* swapping the in-memory state: if the disk
    /// write fails, the slot keeps serving the old in-memory tokens
    /// — better than reporting fresh tokens that the next process
    /// start won't see.
    pub async fn update_from_contents(&self, contents: &str) -> Result<(), SlotUpdateError> {
        let fresh = CodexAuth::from_contents(self.path.clone(), contents)
            .map_err(|e| SlotUpdateError::Invalid(format!("{e:#}")))?;
        if let Some(parent) = self.path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).map_err(|e| {
                SlotUpdateError::Io(format!("create credential dir {}: {e}", parent.display()))
            })?;
        }
        fresh
            .persist()
            .map_err(|e| SlotUpdateError::Io(format!("{e:#}")))?;
        let mut guard = self.state.lock().await;
        *guard = Some(fresh);
        Ok(())
    }

    /// Snapshot the chatgpt_account_id without taking ownership.
    /// Returns `None` if no credential is loaded or the JWT didn't
    /// carry the claim. Read-only diagnostic — never used on the hot
    /// path.
    pub async fn chatgpt_account_id(&self) -> Option<String> {
        let guard = self.state.lock().await;
        guard
            .as_ref()
            .and_then(|a| a.chatgpt_account_id().map(str::to_string))
    }

    /// `true` if a credential is currently loaded. Read-only — useful
    /// for surfacing "awaiting publication" in admin diagnostics.
    pub async fn is_loaded(&self) -> bool {
        self.state.lock().await.is_some()
    }
}

impl ClientAuth {
    /// Build live auth material from a config `Auth` entry. Loads CodexAuth
    /// from disk if needed; resolves `value`/`env` for API keys.
    ///
    /// For `ChatgptSubscription`, the file is required: this path is
    /// used by consumers (notably `whisper-agent-mcp-imagegen`) that
    /// have no fallback for an empty credential. The main daemon
    /// constructs its Codex backends through [`pod::config`], which
    /// chooses between [`CodexAuthSlot::loaded`] and
    /// [`CodexAuthSlot::empty`] based on the `manager` field.
    pub fn from_config(auth: &Auth) -> Result<Self> {
        match auth {
            Auth::ApiKey { .. } => Ok(ClientAuth::ApiKey(
                auth.resolve_api_key().context("resolve api_key")?,
            )),
            Auth::ChatgptSubscription {
                source: _, path, ..
            } => {
                let codex = CodexAuth::load(path.clone())
                    .context("load codex auth.json (chatgpt_subscription)")?;
                Ok(ClientAuth::Codex(Arc::new(CodexAuthSlot::loaded(codex))))
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
            ClientAuth::Codex(slot) => slot.prepare(http).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;

    fn jwt_with_exp(secs_from_now: i64) -> String {
        let exp = chrono::Utc::now().timestamp() + secs_from_now;
        let payload = serde_json::json!({ "exp": exp });
        let h = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b"{}");
        let p = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_string(&payload).unwrap().as_bytes());
        let s = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b"sig");
        format!("{h}.{p}.{s}")
    }

    fn well_formed_auth_blob() -> String {
        serde_json::json!({
            "tokens": {
                "id_token": jwt_with_exp(3600),
                "access_token": jwt_with_exp(3600),
                "refresh_token": "rt",
            }
        })
        .to_string()
    }

    fn install_crypto_provider() {
        let _ = rustls::crypto::ring::default_provider().install_default();
    }

    #[tokio::test]
    async fn empty_slot_errors_with_path_in_message() {
        install_crypto_provider();
        let tmp = std::env::temp_dir().join("test-codex-empty-slot-msg.json");
        let slot = CodexAuthSlot::empty(tmp.clone());
        let http = reqwest::Client::new();
        let err = slot.prepare(&http).await.expect_err("empty slot must fail");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("awaiting publication"),
            "expected awaiting-publication diagnostic, got: {msg}"
        );
        assert!(
            msg.contains(tmp.to_string_lossy().as_ref()),
            "expected path `{}` in diagnostic, got: {msg}",
            tmp.display()
        );
    }

    #[tokio::test]
    async fn update_from_contents_populates_and_persists() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("subdir/auth.json");
        let slot = CodexAuthSlot::empty(path.clone());
        assert!(!slot.is_loaded().await);

        slot.update_from_contents(&well_formed_auth_blob())
            .await
            .unwrap();
        assert!(slot.is_loaded().await);
        assert!(path.exists(), "file must be written through to disk");
        let on_disk = std::fs::read_to_string(&path).unwrap();
        // Round-trips through serde so formatting differs, but the
        // tokens block must survive.
        assert!(on_disk.contains("access_token"), "got: {on_disk}");
    }

    #[tokio::test]
    async fn update_from_contents_rejects_garbage() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.json");
        let slot = CodexAuthSlot::empty(path.clone());
        let err = slot
            .update_from_contents("not json")
            .await
            .expect_err("garbage must reject");
        assert!(
            matches!(err, SlotUpdateError::Invalid(_)),
            "expected Invalid variant, got: {err:?}"
        );
        // Failed update must leave the slot empty.
        assert!(!slot.is_loaded().await);
        assert!(!path.exists(), "file must not be written on parse failure");
    }
}
