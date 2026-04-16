//! Sandbox provisioning and lifecycle.
//!
//! [`SandboxProvider`] translates a [`SandboxSpec`] (from the task config) into a
//! running execution environment. The scheduler holds a provider and calls
//! [`provision`] when a task enters its MCP-connect phase; the returned
//! [`SandboxHandle`] lives for the task's lifetime and is torn down when the
//! task completes or is cancelled.

use std::future::Future;
use std::pin::Pin;

use thiserror::Error;
use tracing::info;
use whisper_agent_protocol::SandboxSpec;
use whisper_agent_protocol::sandbox::{ProvisionRequest, ProvisionResponse, TeardownRequest};

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[derive(Debug, Error)]
pub enum SandboxError {
    #[error("provision failed: {0}")]
    Provision(String),
    #[error("teardown failed: {0}")]
    Teardown(String),
    #[error("unsupported sandbox type for this provider: {0}")]
    Unsupported(String),
}

/// Provisions sandboxed execution environments from a [`SandboxSpec`].
///
/// The scheduler holds one provider (chosen at startup based on host
/// capabilities). Each task gets its own [`SandboxHandle`].
pub trait SandboxProvider: Send + Sync {
    /// Create a new sandbox for the given task. The handle must be kept alive
    /// for the task's duration — dropping it may tear down resources.
    fn provision<'a>(
        &'a self,
        task_id: &'a str,
        spec: &'a SandboxSpec,
    ) -> BoxFuture<'a, Result<Box<dyn SandboxHandle>, SandboxError>>;
}

/// A running sandbox. Held by the scheduler for the task's lifetime.
pub trait SandboxHandle: Send + Sync {
    /// MCP host URL inside this sandbox, if the sandbox provisions its own.
    /// Returns `None` for pass-through backends — the scheduler falls back to
    /// the task's configured `mcp_host_url`.
    fn mcp_url(&self) -> Option<&str>;

    /// Tear down the sandbox (stop container, release namespaces, etc.).
    /// Called when the task completes, fails, or is cancelled.
    fn teardown(&mut self) -> BoxFuture<'_, Result<(), SandboxError>>;
}

// ---------- BareMetal (no-op) ----------

/// Pass-through provider: no isolation, tools run directly on the host.
/// Used when `SandboxSpec::None` or when the host has no container runtime.
pub struct BareMetal;

impl SandboxProvider for BareMetal {
    fn provision<'a>(
        &'a self,
        _task_id: &'a str,
        spec: &'a SandboxSpec,
    ) -> BoxFuture<'a, Result<Box<dyn SandboxHandle>, SandboxError>> {
        Box::pin(async move {
            match spec {
                SandboxSpec::None => Ok(Box::new(BareMetalHandle) as Box<dyn SandboxHandle>),
                SandboxSpec::Container { .. } => Err(SandboxError::Unsupported(
                    "BareMetal provider cannot provision containers".into(),
                )),
                SandboxSpec::Landlock { .. } => Err(SandboxError::Unsupported(
                    "BareMetal provider cannot apply landlock rules".into(),
                )),
            }
        })
    }
}

struct BareMetalHandle;

impl SandboxHandle for BareMetalHandle {
    fn mcp_url(&self) -> Option<&str> {
        None
    }

    fn teardown(&mut self) -> BoxFuture<'_, Result<(), SandboxError>> {
        Box::pin(async { Ok(()) })
    }
}

// ---------- DaemonClient ----------

/// Dispatching provider: handles `None` locally (no daemon call), delegates
/// `Container` and `Landlock` to a remote [`whisper-agent-sandbox`] daemon.
pub struct DaemonClient {
    http: reqwest::Client,
    daemon_url: String,
}

impl DaemonClient {
    pub fn new(daemon_url: String) -> Self {
        Self {
            http: reqwest::Client::new(),
            daemon_url: daemon_url.trim_end_matches('/').to_string(),
        }
    }
}

impl SandboxProvider for DaemonClient {
    fn provision<'a>(
        &'a self,
        task_id: &'a str,
        spec: &'a SandboxSpec,
    ) -> BoxFuture<'a, Result<Box<dyn SandboxHandle>, SandboxError>> {
        Box::pin(async move {
            if matches!(spec, SandboxSpec::None) {
                return Ok(Box::new(BareMetalHandle) as Box<dyn SandboxHandle>);
            }

            info!(task_id, "requesting sandbox from daemon");
            let req = ProvisionRequest {
                task_id: task_id.to_string(),
                spec: spec.clone(),
            };
            let url = format!("{}/provision", self.daemon_url);
            let resp = self
                .http
                .post(&url)
                .json(&req)
                .send()
                .await
                .map_err(|e| SandboxError::Provision(format!("daemon unreachable: {e}")))?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                return Err(SandboxError::Provision(format!(
                    "daemon returned {status}: {body}",
                )));
            }

            let prov: ProvisionResponse = resp
                .json()
                .await
                .map_err(|e| SandboxError::Provision(format!("bad response: {e}")))?;

            info!(
                task_id,
                session_id = %prov.session_id,
                mcp_url = %prov.mcp_url,
                "sandbox provisioned"
            );

            Ok(Box::new(DaemonHandle {
                http: self.http.clone(),
                daemon_url: self.daemon_url.clone(),
                session_id: prov.session_id,
                mcp_url: prov.mcp_url,
            }) as Box<dyn SandboxHandle>)
        })
    }
}

struct DaemonHandle {
    http: reqwest::Client,
    daemon_url: String,
    session_id: String,
    mcp_url: String,
}

impl SandboxHandle for DaemonHandle {
    fn mcp_url(&self) -> Option<&str> {
        Some(&self.mcp_url)
    }

    fn teardown(&mut self) -> BoxFuture<'_, Result<(), SandboxError>> {
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
                .map_err(|e| SandboxError::Teardown(format!("daemon unreachable: {e}")))?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                return Err(SandboxError::Teardown(format!(
                    "daemon returned {status}: {body}",
                )));
            }
            Ok(())
        })
    }
}
