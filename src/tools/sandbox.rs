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
use std::time::Duration;

use thiserror::Error;
use tracing::{info, warn};
use whisper_agent_protocol::HostEnvSpec;
use whisper_agent_protocol::sandbox::{ProvisionRequest, ProvisionResponse, TeardownRequest};

/// TCP connect timeout for every daemon request. A powered-off daemon
/// host would otherwise hang on Linux's default SYN retry for minutes;
/// 5s is generous for any real LAN/loopback daemon and fails fast on
/// everything else. Applies to probe, provision, and teardown alike.
pub const DAEMON_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// Overall request timeout for provision / teardown. The daemon may do
/// real work on provision (spawn MCP host, set up cgroups / namespaces)
/// so we can't make this too tight. 30s covers comfortable worst cases
/// without letting a stuck daemon wedge the scheduler indefinitely.
pub const DAEMON_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Overall request timeout for reachability probes. /health on a live
/// daemon returns in single-digit ms, so anything over a few seconds
/// means something is wedged — report unreachable and let the next
/// probe tick try again.
pub const DAEMON_PROBE_TIMEOUT: Duration = Duration::from_secs(5);

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Replace the host portion of a URL returned by the sandbox daemon's
/// `/provision` with the host we already used to reach that daemon.
///
/// The daemon builds `mcp_url` from the IP its MCP host child is bound
/// to. If the daemon binds `[::]` or is running on a remote LAN host
/// (we reach it over an IPv6 ULA), the child's literal bind IP isn't
/// useful to us — either it's the wildcard literal or it's the
/// daemon's local view of its own interface. Either way, our known-
/// reachable handle on the daemon is the URL we used to POST
/// `/provision`; re-using that host with the child's port is correct.
///
/// Scheme and path are preserved from the child URL (it may not be
/// http the daemon itself speaks, and `/mcp` etc. are the child's
/// concern). Port is preserved as given by the daemon.
fn rewrite_mcp_host(daemon_url: &str, mcp_url: &str) -> Result<String, HostEnvError> {
    let daemon = url::Url::parse(daemon_url)
        .map_err(|e| HostEnvError::Provision(format!("daemon URL unparseable: {e}")))?;
    let mut mcp = url::Url::parse(mcp_url).map_err(|e| {
        HostEnvError::Provision(format!("daemon returned unparseable mcp_url: {e}"))
    })?;
    let host = daemon
        .host_str()
        .ok_or_else(|| HostEnvError::Provision("daemon URL has no host".into()))?;
    mcp.set_host(Some(host))
        .map_err(|e| HostEnvError::Provision(format!("host swap on mcp_url failed: {e}")))?;
    Ok(mcp.into())
}

#[derive(Debug, Error)]
pub enum HostEnvError {
    #[error("provision failed: {0}")]
    Provision(String),
    #[error("teardown failed: {0}")]
    Teardown(String),
    #[error("unknown provider `{0}` (not in host_env_providers catalog)")]
    UnknownProvider(String),
    /// Reachability probe failed. Distinct from `Provision` so the
    /// scheduler can update the registry entry's reachability without
    /// confusing it with a real provision attempt.
    #[error("probe failed: {0}")]
    Probe(String),
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

    /// Lightweight liveness probe. The scheduler runs this periodically
    /// and opportunistically to update the provider's `Reachability`.
    /// Must finish quickly (single-digit seconds at most) because the
    /// probe ticker fires every N seconds for every provider — a slow
    /// probe drags the scheduler.
    fn probe(&self) -> BoxFuture<'_, Result<(), HostEnvError>>;
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
        let http = reqwest::Client::builder()
            .connect_timeout(DAEMON_CONNECT_TIMEOUT)
            .timeout(DAEMON_REQUEST_TIMEOUT)
            .build()
            .unwrap_or_else(|e| {
                // Falls back to the default client rather than panicking:
                // a misconfigured rustls/native-tls provider is the only
                // plausible failure here, and we'd rather the next
                // request surface the real error than fail at registry
                // construction.
                warn!(error = %e, "reqwest client builder failed; falling back to defaults");
                reqwest::Client::new()
            });
        Self {
            name,
            http,
            daemon_url: daemon_url.trim_end_matches('/').to_string(),
            control_token,
        }
    }

    async fn probe_inner(&self) -> Result<(), HostEnvError> {
        let url = format!("{}/health", self.daemon_url);
        let resp = self
            .http
            .get(&url)
            .timeout(DAEMON_PROBE_TIMEOUT)
            .send()
            .await
            .map_err(|e| HostEnvError::Probe(format!("daemon unreachable: {e}")))?;
        if !resp.status().is_success() {
            return Err(HostEnvError::Probe(format!(
                "daemon /health returned {}",
                resp.status()
            )));
        }
        Ok(())
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

            // The daemon returns a URL whose host is whatever IP it
            // bound the MCP host child on — fine for a local daemon, but
            // if the daemon bound `[::]` / `0.0.0.0` or we reached it
            // over a LAN address, the returned host is unreachable from
            // here. Swap it for the host we already used to reach the
            // daemon control plane, which is reachable by construction.
            let mcp_url = rewrite_mcp_host(&self.daemon_url, &prov.mcp_url)?;

            info!(
                thread_id,
                provider = %self.name,
                session_id = %prov.session_id,
                daemon_mcp_url = %prov.mcp_url,
                %mcp_url,
                "host env provisioned"
            );

            Ok(Box::new(DaemonHandle {
                http: self.http.clone(),
                daemon_url: self.daemon_url.clone(),
                control_token: self.control_token.clone(),
                session_id: prov.session_id,
                mcp_url,
                mcp_token: prov.mcp_token,
            }) as Box<dyn HostEnvHandle>)
        })
    }

    fn probe(&self) -> BoxFuture<'_, Result<(), HostEnvError>> {
        Box::pin(self.probe_inner())
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
    /// Last observed reachability — updated by periodic probes and
    /// opportunistically by provision success / connect-fail. Starts
    /// `Unknown` at registration and moves to `Reachable` or
    /// `Unreachable` as signals arrive.
    pub reachability: whisper_agent_protocol::HostEnvReachability,
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

    /// Mark `name` reachable at the given observation time. Returns
    /// whether the **variant** actually changed — a repeat reachable
    /// probe refreshes `at` silently and returns `false`, so callers
    /// don't log / broadcast a "recovered" edge on every 30s tick for
    /// a steadily-up daemon.
    pub fn mark_reachable(&mut self, name: &str, at: chrono::DateTime<chrono::Utc>) -> bool {
        use whisper_agent_protocol::HostEnvReachability;
        let Some(entry) = self.providers.get_mut(name) else {
            return false;
        };
        let was_reachable = matches!(entry.reachability, HostEnvReachability::Reachable { .. });
        entry.reachability = HostEnvReachability::Reachable {
            at: at.to_rfc3339(),
        };
        !was_reachable
    }

    /// Mark `name` unreachable. Preserves `since` across successive
    /// failures so operators see how long the outage has lasted —
    /// only the first failure after a `Reachable` / `Unknown` resets
    /// it. Returns whether the state changed.
    pub fn mark_unreachable(
        &mut self,
        name: &str,
        at: chrono::DateTime<chrono::Utc>,
        error: String,
    ) -> bool {
        use whisper_agent_protocol::HostEnvReachability;
        let Some(entry) = self.providers.get_mut(name) else {
            return false;
        };
        let at_str = at.to_rfc3339();
        let new_state = match &entry.reachability {
            HostEnvReachability::Unreachable { since, .. } => HostEnvReachability::Unreachable {
                since: since.clone(),
                last_error: error,
            },
            // Unknown or Reachable → start a fresh outage window.
            _ => HostEnvReachability::Unreachable {
                since: at_str,
                last_error: error,
            },
        };
        if entry.reachability == new_state {
            return false;
        }
        entry.reachability = new_state;
        true
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
                    reachability: entry.reachability.clone(),
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
        reachability: whisper_agent_protocol::HostEnvReachability::Unknown,
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
    fn rewrite_mcp_host_swaps_ipv4_loopback_for_lan_host() {
        let out = rewrite_mcp_host(
            "http://[fdb7:1860:52b3:2600::1]:9820",
            "http://127.0.0.1:9901/mcp",
        )
        .unwrap();
        assert_eq!(out, "http://[fdb7:1860:52b3:2600::1]:9901/mcp");
    }

    #[test]
    fn rewrite_mcp_host_swaps_ipv6_unspecified_for_dns_name() {
        let out = rewrite_mcp_host("http://sandbox.lan:9820", "http://[::]:9901/mcp").unwrap();
        assert_eq!(out, "http://sandbox.lan:9901/mcp");
    }

    #[test]
    fn rewrite_mcp_host_preserves_port_and_path() {
        let out =
            rewrite_mcp_host("http://10.0.0.5:9820/", "http://127.0.0.1:12345/mcp/v2").unwrap();
        assert_eq!(out, "http://10.0.0.5:12345/mcp/v2");
    }

    #[test]
    fn rewrite_mcp_host_rejects_unparseable_inputs() {
        assert!(rewrite_mcp_host("not a url", "http://127.0.0.1:9901/mcp").is_err());
        assert!(rewrite_mcp_host("http://a:9820", "::::").is_err());
    }

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
    fn reachability_starts_unknown() {
        let mut r = HostEnvRegistry::new();
        r.insert("a".into(), "http://a".into(), None).unwrap();
        let e = r.entry("a").unwrap();
        assert!(matches!(
            e.reachability,
            whisper_agent_protocol::HostEnvReachability::Unknown
        ));
    }

    #[test]
    fn mark_reachable_transitions_and_dedups() {
        use whisper_agent_protocol::HostEnvReachability;
        let mut r = HostEnvRegistry::new();
        r.insert("a".into(), "http://a".into(), None).unwrap();
        let t1 = chrono::DateTime::parse_from_rfc3339("2026-04-20T12:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        // Unknown → Reachable is a real transition.
        assert!(r.mark_reachable("a", t1));
        // Same state, same timestamp → no change.
        assert!(!r.mark_reachable("a", t1));
        // Same variant, different timestamp → still a repeat; the
        // 30s probe tick shouldn't log "recovered" every iteration.
        let t2 = t1 + chrono::Duration::seconds(30);
        assert!(!r.mark_reachable("a", t2));
        // But the stored `at` refreshes so freshness is observable.
        match &r.entry("a").unwrap().reachability {
            HostEnvReachability::Reachable { at } => {
                assert_eq!(at, &t2.to_rfc3339());
            }
            other => panic!("expected Reachable, got {other:?}"),
        }
    }

    #[test]
    fn mark_unreachable_preserves_since_across_failures() {
        use whisper_agent_protocol::HostEnvReachability;
        let mut r = HostEnvRegistry::new();
        r.insert("a".into(), "http://a".into(), None).unwrap();
        let t1 = chrono::DateTime::parse_from_rfc3339("2026-04-20T12:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        r.mark_unreachable("a", t1, "connect refused".into());
        let t2 = t1 + chrono::Duration::seconds(30);
        r.mark_unreachable("a", t2, "connect timeout".into());
        match &r.entry("a").unwrap().reachability {
            HostEnvReachability::Unreachable { since, last_error } => {
                assert_eq!(
                    since,
                    &t1.to_rfc3339(),
                    "since must be pinned to first failure"
                );
                assert_eq!(last_error, "connect timeout");
            }
            other => panic!("expected Unreachable, got {other:?}"),
        }
    }

    #[test]
    fn reachable_to_unreachable_resets_since() {
        use whisper_agent_protocol::HostEnvReachability;
        let mut r = HostEnvRegistry::new();
        r.insert("a".into(), "http://a".into(), None).unwrap();
        let t1 = chrono::DateTime::parse_from_rfc3339("2026-04-20T12:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        r.mark_reachable("a", t1);
        let t2 = t1 + chrono::Duration::seconds(30);
        r.mark_unreachable("a", t2, "oops".into());
        match &r.entry("a").unwrap().reachability {
            HostEnvReachability::Unreachable { since, .. } => {
                // Since must be t2 (first failure after reachable), not t1.
                assert_eq!(since, &t2.to_rfc3339());
            }
            other => panic!("expected Unreachable, got {other:?}"),
        }
    }

    #[test]
    fn insert_or_replace_resets_reachability() {
        use whisper_agent_protocol::HostEnvReachability;
        let mut r = HostEnvRegistry::new();
        r.insert("a".into(), "http://a".into(), None).unwrap();
        let t1 = chrono::DateTime::parse_from_rfc3339("2026-04-20T12:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        r.mark_reachable("a", t1);
        // Update flows through insert_or_replace; it rebuilds the entry
        // so reachability goes back to Unknown — makes sense because
        // the URL may now point at a different daemon.
        r.insert_or_replace("a".into(), "http://a2".into(), Some("tok".into()));
        assert!(matches!(
            r.entry("a").unwrap().reachability,
            HostEnvReachability::Unknown
        ));
    }

    #[test]
    fn mark_on_missing_name_is_noop() {
        let mut r = HostEnvRegistry::new();
        let t1 = chrono::Utc::now();
        assert!(!r.mark_reachable("nope", t1));
        assert!(!r.mark_unreachable("nope", t1, "x".into()));
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
