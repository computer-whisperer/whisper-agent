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
/// All entries come from `[[host_env_providers]]` in
/// `whisper-agent.toml` — there are no built-ins. A server running
/// with zero providers simply can't provision host envs; its pods can
/// still host threads, those threads just have no host-env MCP
/// connection.
#[derive(Clone, Default)]
pub struct HostEnvRegistry {
    providers: HashMap<String, Arc<dyn HostEnvProvider>>,
}

impl HostEnvRegistry {
    pub fn new(catalog: Vec<HostEnvProviderEntry>) -> anyhow::Result<Self> {
        let mut providers: HashMap<String, Arc<dyn HostEnvProvider>> = HashMap::new();
        for entry in catalog {
            if providers.contains_key(&entry.name) {
                anyhow::bail!("duplicate host_env_provider name `{}`", entry.name);
            }
            let control_token = match &entry.token_file {
                Some(path) => {
                    let raw = std::fs::read_to_string(path).map_err(|e| {
                        anyhow::anyhow!(
                            "host_env_provider `{}`: reading token_file {}: {e}",
                            entry.name,
                            path.display()
                        )
                    })?;
                    let tok = raw.trim().to_string();
                    if tok.is_empty() {
                        anyhow::bail!(
                            "host_env_provider `{}`: token_file {} is empty after trim",
                            entry.name,
                            path.display()
                        );
                    }
                    Some(tok)
                }
                None => {
                    warn!(
                        provider = %entry.name,
                        "host_env_provider has no token_file — requests will be anonymous \
                         and will be rejected by any daemon running with --control-token-file"
                    );
                    None
                }
            };
            providers.insert(
                entry.name.clone(),
                Arc::new(DaemonClient::new(entry.name, entry.url, control_token))
                    as Arc<dyn HostEnvProvider>,
            );
        }
        Ok(Self { providers })
    }

    pub fn get(&self, name: &str) -> Option<&Arc<dyn HostEnvProvider>> {
        self.providers.get(name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.providers.contains_key(name)
    }

    /// Catalog entries in deterministic order (sorted by name). Used
    /// by `ListHostEnvProviders` responses.
    pub fn snapshot(&self) -> Vec<whisper_agent_protocol::HostEnvProviderInfo> {
        use whisper_agent_protocol::HostEnvProviderInfo;
        let mut names: Vec<&String> = self.providers.keys().collect();
        names.sort();
        names
            .into_iter()
            .map(|name| HostEnvProviderInfo { name: name.clone() })
            .collect()
    }
}
