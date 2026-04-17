//! Host-env provisioning and lifecycle.
//!
//! [`HostEnvProvider`] translates a [`HostEnvSpec`] into a running
//! execution environment with an MCP host inside it. The scheduler holds
//! a *registry* of providers keyed by name (catalog entries from
//! `whisper-agent.toml` plus the always-built-in `"bare"` provider).
//! Each thread's bindings name which provider to use; the scheduler
//! dispatches to that provider's `provision`.
//!
//! Today's only network-attached provider implementation is
//! [`DaemonClient`] — talks to the existing `whisper-agent-sandbox`
//! daemon over its custom HTTP API. Authentication on that wire is
//! intentionally absent for now (dev environment); the catalog entry
//! struct is left extensible so adding mTLS / shared-secret auth later
//! is purely additive.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use thiserror::Error;
use tracing::info;
use whisper_agent_protocol::HostEnvSpec;
use whisper_agent_protocol::sandbox::{ProvisionRequest, ProvisionResponse, TeardownRequest};

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Always-registered name of the in-process noop provider. Pod entries
/// pairing this provider with `HostEnvSpec::None` mean "no isolation,
/// thread runs bare-metal."
pub const BARE_PROVIDER_NAME: &str = "bare";

#[derive(Debug, Error)]
pub enum HostEnvError {
    #[error("provision failed: {0}")]
    Provision(String),
    #[error("teardown failed: {0}")]
    Teardown(String),
    #[error("provider `{provider}` does not support spec type `{spec_type}`")]
    UnsupportedSpec {
        provider: String,
        spec_type: &'static str,
    },
    #[error("unknown provider `{0}` (not in host_env_providers catalog)")]
    UnknownProvider(String),
}

/// One entry in the server's host-env-provider catalog. Loaded from
/// `[[host_env_providers]]` in `whisper-agent.toml`. Only `name` and
/// `url` are required today — additional fields (`auth`, etc.) plug in
/// without breaking the wire because TOML deserialization ignores
/// unknown keys by default.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct HostEnvProviderEntry {
    pub name: String,
    pub url: String,
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
    /// MCP URL of the host inside this env, if the provider spawned
    /// one. Returns `None` for the `bare` provider — the scheduler
    /// then falls back to the server's `default_mcp_host_url`.
    fn mcp_url(&self) -> Option<&str>;

    /// Tear down the env (stop the daemon-side session, release
    /// namespaces, etc.). Called when the env is GC'd or the bound
    /// thread is removed.
    fn teardown(&mut self) -> BoxFuture<'_, Result<(), HostEnvError>>;
}

// ---------- Bare (in-process noop) ----------

/// Always-registered built-in. Accepts only `HostEnvSpec::None`; any
/// other spec is rejected with `UnsupportedSpec`. The handle holds no
/// state; teardown is a no-op.
pub struct BareProvider;

impl HostEnvProvider for BareProvider {
    fn provision<'a>(
        &'a self,
        _thread_id: &'a str,
        spec: &'a HostEnvSpec,
    ) -> BoxFuture<'a, Result<Box<dyn HostEnvHandle>, HostEnvError>> {
        Box::pin(async move {
            match spec {
                HostEnvSpec::None => Ok(Box::new(BareHandle) as Box<dyn HostEnvHandle>),
                HostEnvSpec::Container { .. } => Err(HostEnvError::UnsupportedSpec {
                    provider: BARE_PROVIDER_NAME.into(),
                    spec_type: "container",
                }),
                HostEnvSpec::Landlock { .. } => Err(HostEnvError::UnsupportedSpec {
                    provider: BARE_PROVIDER_NAME.into(),
                    spec_type: "landlock",
                }),
            }
        })
    }
}

struct BareHandle;

impl HostEnvHandle for BareHandle {
    fn mcp_url(&self) -> Option<&str> {
        None
    }

    fn teardown(&mut self) -> BoxFuture<'_, Result<(), HostEnvError>> {
        Box::pin(async { Ok(()) })
    }
}

// ---------- DaemonClient ----------

/// HTTP client for one configured `whisper-agent-sandbox` daemon. The
/// catalog can hold many of these (each with its own URL); the
/// scheduler dispatches to one by name at provision time. Today only
/// `Container` and `Landlock` specs are valid — the daemon would
/// reject `None`.
pub struct DaemonClient {
    name: String,
    http: reqwest::Client,
    daemon_url: String,
}

impl DaemonClient {
    pub fn new(name: String, daemon_url: String) -> Self {
        Self {
            name,
            http: reqwest::Client::new(),
            daemon_url: daemon_url.trim_end_matches('/').to_string(),
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
            if matches!(spec, HostEnvSpec::None) {
                return Err(HostEnvError::UnsupportedSpec {
                    provider: self.name.clone(),
                    spec_type: "none",
                });
            }

            info!(thread_id, provider = %self.name, "requesting host env from daemon");
            let req = ProvisionRequest {
                thread_id: thread_id.to_string(),
                spec: spec.clone(),
            };
            let url = format!("{}/provision", self.daemon_url);
            let resp = self
                .http
                .post(&url)
                .json(&req)
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
                session_id: prov.session_id,
                mcp_url: prov.mcp_url,
            }) as Box<dyn HostEnvHandle>)
        })
    }
}

struct DaemonHandle {
    http: reqwest::Client,
    daemon_url: String,
    session_id: String,
    mcp_url: String,
}

impl HostEnvHandle for DaemonHandle {
    fn mcp_url(&self) -> Option<&str> {
        Some(&self.mcp_url)
    }

    fn teardown(&mut self) -> BoxFuture<'_, Result<(), HostEnvError>> {
        Box::pin(async {
            let url = format!("{}/teardown", self.daemon_url);
            let req = TeardownRequest {
                session_id: self.session_id.clone(),
            };
            let resp = self
                .http
                .post(&url)
                .json(&req)
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
/// `bare` is always present; other entries come from
/// `[[host_env_providers]]` in `whisper-agent.toml`.
#[derive(Clone)]
pub struct HostEnvRegistry {
    providers: HashMap<String, Arc<dyn HostEnvProvider>>,
}

impl HostEnvRegistry {
    /// Build a registry from the configured catalog. Always includes a
    /// `bare` entry; refuses if any catalog entry tries to shadow that
    /// reserved name.
    pub fn new(catalog: Vec<HostEnvProviderEntry>) -> anyhow::Result<Self> {
        let mut providers: HashMap<String, Arc<dyn HostEnvProvider>> = HashMap::new();
        providers.insert(
            BARE_PROVIDER_NAME.to_string(),
            Arc::new(BareProvider) as Arc<dyn HostEnvProvider>,
        );
        for entry in catalog {
            if entry.name == BARE_PROVIDER_NAME {
                anyhow::bail!(
                    "host_env_provider name `{BARE_PROVIDER_NAME}` is reserved for the built-in"
                );
            }
            if providers.contains_key(&entry.name) {
                anyhow::bail!("duplicate host_env_provider name `{}`", entry.name);
            }
            providers.insert(
                entry.name.clone(),
                Arc::new(DaemonClient::new(entry.name, entry.url))
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

    /// Catalog entries in deterministic order — `bare` first, then the
    /// configured ones sorted by name. Used by `ListHostEnvProviders`
    /// responses.
    pub fn snapshot(&self) -> Vec<whisper_agent_protocol::HostEnvProviderInfo> {
        use whisper_agent_protocol::HostEnvProviderInfo;
        let mut others: Vec<&String> = self
            .providers
            .keys()
            .filter(|k| k.as_str() != BARE_PROVIDER_NAME)
            .collect();
        others.sort();
        let mut out = Vec::with_capacity(self.providers.len());
        out.push(HostEnvProviderInfo {
            name: BARE_PROVIDER_NAME.to_string(),
            builtin: true,
        });
        for name in others {
            out.push(HostEnvProviderInfo {
                name: name.clone(),
                builtin: false,
            });
        }
        out
    }
}
