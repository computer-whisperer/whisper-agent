//! Host-env provisioning and lifecycle.
//!
//! [`HostEnvProvider`] translates a [`HostEnvSpec`] into a running
//! execution environment with an MCP host inside it. The scheduler
//! holds a registry of providers loaded from `[[host_env_providers]]`
//! in `whisper-agent.toml`; each pod's `[[allow.host_env]]` entry
//! names one by name, and threads bind by name into the pod's allow
//! list.
//!
//! There is no built-in provider. A thread that does not bind a host
//! env simply has no host-env MCP connection — its tool set is
//! whatever shared MCP hosts it binds to, and nothing else.
//!
//! Today's only provider implementation is [`DaemonClient`] — talks
//! to the existing `whisper-agent-sandbox` daemon over its custom
//! HTTP API. Auth on the control plane is a pre-shared bearer per
//! dispatcher: the catalog entry's `token_file` points at a file
//! holding the token the daemon's `--control-token-file` loaded, and
//! every `/provision` / `/teardown` request carries it as
//! `Authorization: Bearer <token>`. Each provision response also
//! carries a fresh per-sandbox bearer the scheduler then uses on the
//! MCP wire, so a leak of an MCP token only exposes one live sandbox.

use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use thiserror::Error;
use tracing::{info, warn};
use whisper_agent_protocol::HostEnvSpec;
use whisper_agent_protocol::sandbox::{ProvisionRequest, ProvisionResponse, TeardownRequest};

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[derive(Debug, Error)]
pub enum HostEnvError {
    #[error("provision failed: {0}")]
    Provision(String),
    #[error("teardown failed: {0}")]
    Teardown(String),
    #[error("unknown provider `{0}` (not in host_env_providers catalog)")]
    UnknownProvider(String),
}

/// One entry in the server's host-env-provider catalog. Loaded from
/// `[[host_env_providers]]` in `whisper-agent.toml`. `token_file`
/// points at a file whose single-line contents are the pre-shared
/// bearer the daemon expects on its control plane. Leaving it unset
/// is allowed (dev / `--no-auth` daemons), but the registry logs a
/// loud warning — production dispatchers should always have a token.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct HostEnvProviderEntry {
    pub name: String,
    pub url: String,
    #[serde(default)]
    pub token_file: Option<PathBuf>,
}

/// Provisions host environments. The scheduler holds a registry of
/// these keyed by name; each thread's bindings reference one by name
/// at provision time.
pub trait HostEnvProvider: Send + Sync {
    /// Create a new host env for the given thread. The handle must be
    /// kept alive for the env's lifetime — dropping it may tear down
    /// the underlying resources.
    fn provision<'a>(
        &'a self,
        thread_id: &'a str,
        spec: &'a HostEnvSpec,
    ) -> BoxFuture<'a, Result<Box<dyn HostEnvHandle>, HostEnvError>>;
}

/// A running host env. Held by the scheduler for the env's lifetime.
pub trait HostEnvHandle: Send + Sync {
    /// MCP URL of the host inside this env. Always populated for real
    /// providers — a host env without an MCP host would be pointless
    /// (its whole purpose is to expose tools).
    fn mcp_url(&self) -> &str;

    /// Per-sandbox bearer the MCP host requires on every request.
    /// `None` means the provider intentionally did not set one (e.g. a
    /// future provider that proves authenticity some other way); the
    /// scheduler should attach it to `McpSession` when present.
    fn mcp_token(&self) -> Option<&str>;

    /// Tear down the env (stop the daemon-side session, release
    /// namespaces, etc.). Called when the env is GC'd or the bound
    /// thread is removed.
    fn teardown(&mut self) -> BoxFuture<'_, Result<(), HostEnvError>>;
}

// ---------- DaemonClient ----------

/// HTTP client for one configured `whisper-agent-sandbox` daemon. The
/// catalog can hold many of these (each with its own URL); the
/// scheduler dispatches to one by name at provision time.
pub struct DaemonClient {
    name: String,
    http: reqwest::Client,
    daemon_url: String,
    /// Control-plane bearer loaded from `token_file` at registry
    /// construction. `None` when the operator explicitly didn't
    /// configure one — requests then go out unauthenticated (will 401
    /// against any daemon running with a real token).
    control_token: Option<String>,
}

impl DaemonClient {
    pub fn new(name: String, daemon_url: String, control_token: Option<String>) -> Self {
        Self {
            name,
            http: reqwest::Client::new(),
            daemon_url: daemon_url.trim_end_matches('/').to_string(),
            control_token,
        }
    }
}

impl HostEnvProvider for DaemonClient {
    fn provision<'a>(
        &'a self,
        thread_id: &'a str,
        spec: &'a HostEnvSpec,
    ) -> BoxFuture<'a, Result<Box<dyn HostEnvHandle>, HostEnvError>> {
        Box::pin(async move {
            info!(thread_id, provider = %self.name, "requesting host env from daemon");
            let req = ProvisionRequest {
                thread_id: thread_id.to_string(),
                spec: spec.clone(),
            };
            let url = format!("{}/provision", self.daemon_url);
            let mut builder = self.http.post(&url).json(&req);
            if let Some(tok) = &self.control_token {
                builder = builder.bearer_auth(tok);
            }
            let resp = builder
                .send()
                .await
                .map_err(|e| HostEnvError::Provision(format!("daemon unreachable: {e}")))?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                return Err(HostEnvError::Provision(format!(
                    "daemon returned {status}: {body}",
                )));
            }

            let prov: ProvisionResponse = resp
                .json()
                .await
                .map_err(|e| HostEnvError::Provision(format!("bad response: {e}")))?;

            info!(
                thread_id,
                provider = %self.name,
                session_id = %prov.session_id,
                mcp_url = %prov.mcp_url,
                "host env provisioned"
            );

            Ok(Box::new(DaemonHandle {
                http: self.http.clone(),
                daemon_url: self.daemon_url.clone(),
                control_token: self.control_token.clone(),
                session_id: prov.session_id,
                mcp_url: prov.mcp_url,
                mcp_token: prov.mcp_token,
            }) as Box<dyn HostEnvHandle>)
        })
    }
}

struct DaemonHandle {
    http: reqwest::Client,
    daemon_url: String,
    control_token: Option<String>,
    session_id: String,
    mcp_url: String,
    mcp_token: String,
}

impl HostEnvHandle for DaemonHandle {
    fn mcp_url(&self) -> &str {
        &self.mcp_url
    }

    fn mcp_token(&self) -> Option<&str> {
        Some(&self.mcp_token)
    }

    fn teardown(&mut self) -> BoxFuture<'_, Result<(), HostEnvError>> {
        Box::pin(async {
            let url = format!("{}/teardown", self.daemon_url);
            let req = TeardownRequest {
                session_id: self.session_id.clone(),
            };
            let mut builder = self.http.post(&url).json(&req);
            if let Some(tok) = &self.control_token {
                builder = builder.bearer_auth(tok);
            }
            let resp = builder
                .send()
                .await
                .map_err(|e| HostEnvError::Teardown(format!("daemon unreachable: {e}")))?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                return Err(HostEnvError::Teardown(format!(
                    "daemon returned {status}: {body}",
                )));
            }
            Ok(())
        })
    }
}

// ---------- Registry ----------

/// Server-held registry of host-env providers, keyed by catalog name.
/// Entries are hydrated on startup from the durable catalog store (and
/// any CLI overlay), then mutated at runtime via scheduler commands.
/// A server running with zero providers simply can't provision host
/// envs; its pods can still host threads, those threads just have no
/// host-env MCP connection.
#[derive(Clone, Default)]
pub struct HostEnvRegistry {
    providers: HashMap<String, RegistryEntry>,
}

/// Per-provider live state the registry tracks. `url` and `token` are
/// surfaced alongside the provider handle so `ListHostEnvProviders`
/// can report them without dereffing into the trait object.
#[derive(Clone)]
pub struct RegistryEntry {
    pub url: String,
    /// `None` means the provider intentionally runs anonymous —
    /// requests will 401 against any daemon that expects a control
    /// token, which the caller is responsible for being aware of.
    pub token: Option<String>,
    pub provider: Arc<dyn HostEnvProvider>,
}

impl HostEnvRegistry {
    /// Empty registry. Callers seed it with `insert` / `insert_or_replace`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a new provider. Errors if the name is already present —
    /// callers that want last-write-wins use `insert_or_replace`.
    pub fn insert(
        &mut self,
        name: String,
        url: String,
        token: Option<String>,
    ) -> anyhow::Result<()> {
        if self.providers.contains_key(&name) {
            anyhow::bail!("host-env provider `{name}` already registered");
        }
        self.providers
            .insert(name.clone(), build_entry(name, url, token));
        Ok(())
    }

    /// Insert a provider, replacing any existing entry with the same
    /// name. Used by the CLI-overlay path (where operators explicitly
    /// want their flag to win over a catalog entry) and by
    /// `update_provider`.
    pub fn insert_or_replace(&mut self, name: String, url: String, token: Option<String>) {
        self.providers
            .insert(name.clone(), build_entry(name, url, token));
    }

    /// Remove a provider by name. Returns whether the name was known.
    /// Existing `Arc<dyn HostEnvProvider>` clones keep working — removal
    /// only prevents future `get` lookups from finding this name. The
    /// live `DaemonHandle`s that already hold their own `http` client +
    /// URL + session id can still `teardown` after removal.
    pub fn remove(&mut self, name: &str) -> bool {
        self.providers.remove(name).is_some()
    }

    pub fn get(&self, name: &str) -> Option<&Arc<dyn HostEnvProvider>> {
        self.providers.get(name).map(|e| &e.provider)
    }

    pub fn entry(&self, name: &str) -> Option<&RegistryEntry> {
        self.providers.get(name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.providers.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.providers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }

    /// Names in sorted order. Stable output keeps logs and protocol
    /// snapshots deterministic.
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.providers.keys().cloned().collect();
        names.sort();
        names
    }

    /// Catalog entries in deterministic order (sorted by name). Used
    /// by `ListHostEnvProviders` responses. The caller supplies an
    /// `origin_for` closure that maps a provider name to its origin —
    /// the registry itself doesn't know whether an entry came from the
    /// catalog (`Seeded` / `Manual`) or a CLI overlay
    /// (`RuntimeOverlay`), so the scheduler threads that in.
    pub fn snapshot<F>(&self, origin_for: F) -> Vec<whisper_agent_protocol::HostEnvProviderInfo>
    where
        F: Fn(&str) -> whisper_agent_protocol::HostEnvProviderOrigin,
    {
        use whisper_agent_protocol::HostEnvProviderInfo;
        self.names()
            .into_iter()
            .map(|name| {
                let entry = self
                    .providers
                    .get(&name)
                    .expect("name came from self.names()");
                HostEnvProviderInfo {
                    origin: origin_for(&name),
                    has_token: entry.token.is_some(),
                    url: entry.url.clone(),
                    name,
                }
            })
            .collect()
    }
}

fn build_entry(name: String, url: String, token: Option<String>) -> RegistryEntry {
    if token.is_none() {
        warn!(
            provider = %name,
            "host-env provider registered without a control token — requests will be anonymous \
             and will be rejected by any daemon running with --control-token-file"
        );
    }
    RegistryEntry {
        url: url.clone(),
        token: token.clone(),
        provider: Arc::new(DaemonClient::new(name, url, token)) as Arc<dyn HostEnvProvider>,
    }
}

/// Read a token from disk: trim, reject empty. Extracted so CLI-overlay
/// and TOML-seed paths share one code path for the `token_file` shape.
pub fn read_token_file(provider_name: &str, path: &std::path::Path) -> anyhow::Result<String> {
    let raw = std::fs::read_to_string(path).map_err(|e| {
        anyhow::anyhow!(
            "host-env provider `{provider_name}`: reading token_file {}: {e}",
            path.display()
        )
    })?;
    let tok = raw.trim().to_string();
    if tok.is_empty() {
        anyhow::bail!(
            "host-env provider `{provider_name}`: token_file {} is empty after trim",
            path.display()
        );
    }
    Ok(tok)
}

#[cfg(test)]
mod tests {
    use super::*;
    use whisper_agent_protocol::HostEnvProviderOrigin;

    #[test]
    fn empty_registry_is_empty() {
        let r = HostEnvRegistry::new();
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
        assert!(!r.contains("x"));
        assert!(r.get("x").is_none());
    }

    #[test]
    fn insert_rejects_duplicate() {
        let mut r = HostEnvRegistry::new();
        r.insert("a".into(), "http://a".into(), Some("t".into()))
            .unwrap();
        let err = r.insert("a".into(), "http://a2".into(), None).unwrap_err();
        assert!(format!("{err}").contains("already registered"));
    }

    #[test]
    fn insert_or_replace_overwrites() {
        let mut r = HostEnvRegistry::new();
        r.insert("a".into(), "http://a".into(), Some("t1".into()))
            .unwrap();
        r.insert_or_replace("a".into(), "http://a2".into(), Some("t2".into()));
        let entry = r.entry("a").unwrap();
        assert_eq!(entry.url, "http://a2");
        assert_eq!(entry.token.as_deref(), Some("t2"));
    }

    #[test]
    fn remove_is_idempotent() {
        let mut r = HostEnvRegistry::new();
        r.insert("a".into(), "http://a".into(), None).unwrap();
        assert!(r.remove("a"));
        assert!(!r.remove("a"));
        assert!(!r.contains("a"));
    }

    #[test]
    fn names_sorted() {
        let mut r = HostEnvRegistry::new();
        r.insert("c".into(), "http://c".into(), None).unwrap();
        r.insert("a".into(), "http://a".into(), None).unwrap();
        r.insert("b".into(), "http://b".into(), None).unwrap();
        assert_eq!(r.names(), vec!["a", "b", "c"]);
    }

    #[test]
    fn snapshot_classifies_origins_via_closure() {
        let mut r = HostEnvRegistry::new();
        r.insert("seeded".into(), "http://s".into(), Some("t".into()))
            .unwrap();
        r.insert("manual".into(), "http://m".into(), None).unwrap();
        r.insert("overlay".into(), "http://o".into(), None).unwrap();
        let snap = r.snapshot(|name| match name {
            "seeded" => HostEnvProviderOrigin::Seeded,
            "manual" => HostEnvProviderOrigin::Manual,
            _ => HostEnvProviderOrigin::RuntimeOverlay,
        });
        assert_eq!(snap.len(), 3);
        let by_name: std::collections::HashMap<_, _> =
            snap.iter().map(|p| (p.name.as_str(), p)).collect();
        assert_eq!(by_name["seeded"].origin, HostEnvProviderOrigin::Seeded);
        assert!(by_name["seeded"].has_token);
        assert_eq!(by_name["manual"].origin, HostEnvProviderOrigin::Manual);
        assert!(!by_name["manual"].has_token);
        assert_eq!(
            by_name["overlay"].origin,
            HostEnvProviderOrigin::RuntimeOverlay
        );
        assert_eq!(by_name["overlay"].url, "http://o");
    }

    #[test]
    fn read_token_file_rejects_empty_after_trim() {
        let dir = std::env::temp_dir().join(format!("wa-token-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty");
        std::fs::write(&path, "   \n\n").unwrap();
        let err = read_token_file("x", &path).unwrap_err();
        assert!(format!("{err}").contains("empty after trim"));
    }

    #[test]
    fn read_token_file_trims() {
        let dir = std::env::temp_dir().join(format!("wa-token2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("good");
        std::fs::write(&path, "  secret\n").unwrap();
        let tok = read_token_file("x", &path).unwrap();
        assert_eq!(tok, "secret");
    }
}
