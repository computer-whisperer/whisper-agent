//! Resource registries — sandboxes, MCP hosts, and model backends as first-class
//! entities with lifecycles independent of the tasks/threads that reference them.
//!
//! See `docs/design_pod_thread_scheduler.md` for the end-state design. This
//! module is the Phase 1a foundation: types and a passive registry that mirrors
//! the existing per-task resource ownership without changing behavior. Later
//! phases promote this from a shadow registry to the authoritative store, drop
//! the per-task `mcp_pools`/`sandbox_handles` fields, and add wire/UI surface.
//!
//! Keys throughout this module are `String` rather than wrapped task/thread IDs
//! because Phase 1a still routes everything through `thread_id`. When Phase 2
//! introduces `ThreadId`, the `users: BTreeSet<String>` field swaps for
//! `BTreeSet<ThreadId>` and the rest of the module stays put.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use whisper_agent_protocol::{HostEnvSpec, ResourceSnapshot, ResourceStateLabel};

use crate::tools::mcp::{McpSession, ToolAnnotations, ToolDescriptor};
use crate::tools::sandbox::HostEnvHandle;

#[derive(Serialize, Deserialize, Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct HostEnvId(pub String);

#[derive(Serialize, Deserialize, Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct McpHostId(pub String);

#[derive(Serialize, Deserialize, Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct BackendId(pub String);

impl HostEnvId {
    /// Derive a stable id from the (provider, spec) pair. Identical
    /// pairs hash to identical ids — this is how dedup works: when a
    /// second thread requests a host env with matching provider+spec,
    /// `pre_register_host_env` finds the existing entry by id and
    /// adds the thread as a user instead of inserting a duplicate.
    /// Two different providers handling identical-shaped specs get
    /// distinct ids (they're independently provisioned).
    ///
    /// Hash is `DefaultHasher` over (provider, spec_json). Not
    /// cryptographic; collision risk on a few-hundred-pod scale is
    /// negligible. The 16-hex-char width keeps ids readable in logs.
    pub fn for_provider_spec(provider: &str, spec: &HostEnvSpec) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let json = serde_json::to_string(spec).unwrap_or_default();
        let mut hasher = DefaultHasher::new();
        provider.hash(&mut hasher);
        json.hash(&mut hasher);
        Self(format!("he-{:016x}", hasher.finish()))
    }
}

impl McpHostId {
    /// Registry id for a shared MCP host. `name` is the catalog name
    /// pod entries and thread bindings reference.
    pub fn shared(name: &str) -> Self {
        Self(format!("mcp-shared-{name}"))
    }
    /// Registry id for an MCP host provisioned by a host-env
    /// provider. One-to-one with the owning `HostEnvId` — distinct
    /// host envs produce distinct MCPs; a second thread with the
    /// same binding dedups onto the same entry.
    pub fn for_host_env(id: &HostEnvId) -> Self {
        Self(format!("mcp-{}", id.0))
    }
    /// Whether this id points at a host-env-backed MCP (vs a shared
    /// MCP). Used by the tool-call path to decide whether a transport
    /// failure means "the sandbox daemon lost our session" (host-env
    /// case, triggers Lost state) or "the shared MCP host is
    /// unreachable" (shared case, surfaces as a plain tool error).
    pub fn is_host_env(&self) -> bool {
        self.0.starts_with("mcp-he-")
    }
    /// Recover the owning `HostEnvId` from a host-env-backed MCP id.
    /// Returns `None` for shared MCPs. Uses the `mcp-` prefix from
    /// `for_host_env`.
    pub fn host_env_id(&self) -> Option<HostEnvId> {
        let rest = self.0.strip_prefix("mcp-")?;
        // Host-env form is `mcp-he-<hex>`; shared form is
        // `mcp-shared-<name>` — guard against stripping `shared-`
        // by accident.
        if rest.starts_with("he-") {
            Some(HostEnvId(rest.to_string()))
        } else {
            None
        }
    }
}

impl BackendId {
    pub fn for_name(name: &str) -> Self {
        Self(format!("backend-{name}"))
    }
}

/// Lifecycle state common to every resource kind. Wire-collapsed to a smaller
/// `ResourceStateLabel` in Phase 1b — internal distinctions like provisioning's
/// op_id are not exposed.
#[derive(Debug, Clone)]
pub enum ResourceState {
    /// Provisioning I/O is in flight. `op_id` is the scheduler's op id for the
    /// provisioning future so the registry can correlate completions.
    Provisioning {
        op_id: u64,
    },
    Ready,
    Errored {
        message: String,
    },
    /// Previously Ready, but the daemon session became unusable
    /// (typical cause: daemon host powered off, daemon restarted and
    /// forgot our session id). Distinct from `Errored` so the UI can
    /// distinguish "never worked" from "was working and went away".
    /// Reset to Provisioning on the next thread arrival with the
    /// same `(provider, spec)` key, same as `Errored`.
    Lost {
        message: String,
    },
    /// Resource has been torn down. Entry lingers briefly for inspection then
    /// is reaped on the next GC pass.
    TornDown,
}

impl ResourceState {
    pub fn is_ready(&self) -> bool {
        matches!(self, ResourceState::Ready)
    }
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            ResourceState::Errored { .. } | ResourceState::Lost { .. } | ResourceState::TornDown
        )
    }
    pub fn label(&self) -> ResourceStateLabel {
        match self {
            ResourceState::Provisioning { .. } => ResourceStateLabel::Provisioning,
            ResourceState::Ready => ResourceStateLabel::Ready,
            ResourceState::Errored { .. } => ResourceStateLabel::Errored,
            ResourceState::Lost { .. } => ResourceStateLabel::Lost,
            ResourceState::TornDown => ResourceStateLabel::TornDown,
        }
    }
}

pub struct HostEnvEntry {
    pub id: HostEnvId,
    /// Catalog name of the provider that provisioned this entry.
    pub provider: String,
    pub spec: HostEnvSpec,
    pub state: ResourceState,
    /// Set of thread_ids currently using this host env.
    pub users: BTreeSet<String>,
    pub pinned: bool,
    pub created_at: DateTime<Utc>,
    pub last_used: DateTime<Utc>,
    /// The provisioned env's handle. Owned by the registry; taken out
    /// at teardown via `take_host_env_handle` so the scheduler can run
    /// the async teardown without holding a borrow.
    pub handle: Option<Box<dyn HostEnvHandle>>,
    /// The MCP URL the env advertises. Cached so dedup hits don't have
    /// to consult the live handle, and so the URL stays correct even
    /// after the handle has been taken for teardown.
    pub mcp_url: Option<String>,
    /// Per-sandbox bearer the provider issued at provision time.
    /// Cached for the same reason as `mcp_url` — dedup hits need it
    /// to open their own `McpSession` against the cached URL.
    pub mcp_token: Option<String>,
}

impl std::fmt::Debug for HostEnvEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostEnvEntry")
            .field("id", &self.id)
            .field("provider", &self.provider)
            .field("spec", &self.spec)
            .field("state", &self.state)
            .field("users", &self.users)
            .field("pinned", &self.pinned)
            .field("created_at", &self.created_at)
            .field("last_used", &self.last_used)
            .field("handle", &self.handle.as_ref().map(|_| "<host_env>"))
            .field("mcp_url", &self.mcp_url)
            .finish()
    }
}

/// Outcome of `ResourceRegistry::complete_host_env_provisioning`. Distinguishes
/// the happy path (entry was waiting for us to fill in the handle) from the
/// dedup race (someone else already completed it) so the caller knows
/// whether to spawn teardown of the now-redundant handle it provisioned.
pub enum CompleteHostEnvOutcome {
    /// We attached the handle to a pre-registered (Provisioning) entry and
    /// transitioned it to Ready. Registry owns the handle.
    Completed,
    /// Entry was already Ready when we arrived — handle is unused, caller
    /// should tear it down. Happens when two concurrent `mcp_connect`
    /// futures provision against the same HostEnvId before either finishes.
    AlreadyCompleted {
        unused_handle: Option<Box<dyn HostEnvHandle>>,
    },
    /// Entry didn't exist at all — `pre_register_host_env` wasn't called for
    /// this id. Treated as a programmer error by callers; the unused handle
    /// is returned so it can be torn down rather than leaked.
    NotPreRegistered {
        unused_handle: Option<Box<dyn HostEnvHandle>>,
    },
}

#[derive(Debug, Clone)]
pub struct McpHostSpec {
    /// Endpoint URL. For per-task primaries this is whatever the sandbox
    /// reported (or fallback `mcp_host_url`); for shared hosts it's the
    /// configured URL.
    pub url: String,
    /// Stable display name. For shared hosts this is the catalog name; for
    /// per-task primaries it's `"primary:<thread_id>"`.
    pub label: String,
    /// True for the per-task primary (its lifecycle is bound to a task today).
    /// False for shared singletons.
    pub per_task: bool,
}

#[derive(Debug)]
pub struct McpHostEntry {
    pub id: McpHostId,
    pub spec: McpHostSpec,
    pub state: ResourceState,
    /// `Some` once the underlying `McpSession` is connected. Cloned cheaply via
    /// `Arc` when callers need to invoke the host.
    pub session: Option<Arc<McpSession>>,
    /// Populated after the first successful `tools/list`.
    pub tools: Vec<ToolDescriptor>,
    pub annotations: HashMap<String, ToolAnnotations>,
    pub users: BTreeSet<String>,
    pub pinned: bool,
    pub created_at: DateTime<Utc>,
    pub last_used: DateTime<Utc>,
}

#[derive(Debug)]
pub struct BackendEntry {
    pub id: BackendId,
    /// Catalog name (`"anthropic"`, `"openai-compat"`, ...). Mirrors the key
    /// used by `scheduler::BackendEntry`.
    pub name: String,
    /// Provider kind string from the live catalog entry; useful for display.
    pub kind: String,
    pub default_model: Option<String>,
    pub state: ResourceState,
    pub users: BTreeSet<String>,
    /// Backends are pinned by default — they're cheap (no live socket, just a
    /// trait object) and never torn down.
    pub pinned: bool,
    pub created_at: DateTime<Utc>,
    pub last_used: DateTime<Utc>,
}

/// Three flat registries owned by the scheduler. Methods are deliberately
/// minimal in Phase 1a — most callers mutate fields directly (the scheduler is
/// the only writer). Higher-level helpers (auto-provision, GC) come in later
/// phases when policy logic actually has work to do.
#[derive(Default, Debug)]
pub struct ResourceRegistry {
    pub host_envs: HashMap<HostEnvId, HostEnvEntry>,
    pub mcp_hosts: HashMap<McpHostId, McpHostEntry>,
    pub backends: HashMap<BackendId, BackendEntry>,
}

impl HostEnvEntry {
    pub fn to_snapshot(&self) -> ResourceSnapshot {
        ResourceSnapshot::HostEnv {
            id: self.id.0.clone(),
            provider: self.provider.clone(),
            spec: self.spec.clone(),
            state: self.state.label(),
            users: self.users.iter().cloned().collect(),
            pinned: self.pinned,
            created_at: self.created_at.to_rfc3339(),
            last_used: self.last_used.to_rfc3339(),
        }
    }
}

impl McpHostEntry {
    pub fn to_snapshot(&self) -> ResourceSnapshot {
        ResourceSnapshot::McpHost {
            id: self.id.0.clone(),
            url: self.spec.url.clone(),
            label: self.spec.label.clone(),
            per_task: self.spec.per_task,
            state: self.state.label(),
            tools: self.tools.iter().map(|t| t.name.clone()).collect(),
            users: self.users.iter().cloned().collect(),
            pinned: self.pinned,
            created_at: self.created_at.to_rfc3339(),
            last_used: self.last_used.to_rfc3339(),
        }
    }
}

impl BackendEntry {
    pub fn to_snapshot(&self) -> ResourceSnapshot {
        ResourceSnapshot::Backend {
            id: self.id.0.clone(),
            name: self.name.clone(),
            backend_kind: self.kind.clone(),
            default_model: self.default_model.clone(),
            state: self.state.label(),
            users: self.users.iter().cloned().collect(),
            pinned: self.pinned,
            created_at: self.created_at.to_rfc3339(),
            last_used: self.last_used.to_rfc3339(),
        }
    }
}

impl ResourceRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot every entry for `ListResources` responses. Order is sandboxes,
    /// then mcp hosts, then backends — stable and human-readable.
    pub fn snapshot_all(&self) -> Vec<ResourceSnapshot> {
        let mut out: Vec<ResourceSnapshot> =
            Vec::with_capacity(self.host_envs.len() + self.mcp_hosts.len() + self.backends.len());
        let mut sandboxes: Vec<&HostEnvEntry> = self.host_envs.values().collect();
        sandboxes.sort_by(|a, b| a.id.cmp(&b.id));
        out.extend(sandboxes.into_iter().map(HostEnvEntry::to_snapshot));
        let mut mcps: Vec<&McpHostEntry> = self.mcp_hosts.values().collect();
        mcps.sort_by(|a, b| a.id.cmp(&b.id));
        out.extend(mcps.into_iter().map(McpHostEntry::to_snapshot));
        let mut backends: Vec<&BackendEntry> = self.backends.values().collect();
        backends.sort_by(|a, b| a.id.cmp(&b.id));
        out.extend(backends.into_iter().map(BackendEntry::to_snapshot));
        out
    }

    pub fn snapshot_host_env(&self, id: &HostEnvId) -> Option<ResourceSnapshot> {
        self.host_envs.get(id).map(HostEnvEntry::to_snapshot)
    }
    pub fn snapshot_mcp_host(&self, id: &McpHostId) -> Option<ResourceSnapshot> {
        self.mcp_hosts.get(id).map(McpHostEntry::to_snapshot)
    }
    pub fn snapshot_backend(&self, id: &BackendId) -> Option<ResourceSnapshot> {
        self.backends.get(id).map(BackendEntry::to_snapshot)
    }

    // ---------- Construction helpers ----------

    /// Insert a Ready backend entry. Idempotent — re-insert overwrites.
    pub fn insert_backend(
        &mut self,
        name: String,
        kind: String,
        default_model: Option<String>,
    ) -> BackendId {
        let id = BackendId::for_name(&name);
        let now = Utc::now();
        self.backends.insert(
            id.clone(),
            BackendEntry {
                id: id.clone(),
                name,
                kind,
                default_model,
                state: ResourceState::Ready,
                users: BTreeSet::new(),
                pinned: true,
                created_at: now,
                last_used: now,
            },
        );
        id
    }

    /// Insert a Ready shared MCP host entry. Idempotent on `id`.
    pub fn insert_shared_mcp_host(
        &mut self,
        name: String,
        url: String,
        session: Arc<McpSession>,
    ) -> McpHostId {
        let id = McpHostId::shared(&name);
        let now = Utc::now();
        self.mcp_hosts.insert(
            id.clone(),
            McpHostEntry {
                id: id.clone(),
                spec: McpHostSpec {
                    url,
                    label: name,
                    per_task: false,
                },
                state: ResourceState::Ready,
                session: Some(session),
                tools: Vec::new(),
                annotations: HashMap::new(),
                users: BTreeSet::new(),
                pinned: true,
                created_at: now,
                last_used: now,
            },
        );
        id
    }

    /// Swap the `Arc<McpSession>` on an existing shared-MCP entry
    /// without rebuilding the entry from scratch. Preserves `users`,
    /// `tools`, `annotations`, `pinned`, and `created_at`; advances
    /// `last_used`. Used by the OAuth refresh path: a rotated
    /// access_token spawns a fresh `McpSession`, which this method
    /// installs in-place so the entry's thread-attachment count
    /// survives the rotation.
    ///
    /// In-flight tool calls that captured the old `Arc` before the
    /// swap keep running against the old session (and may 401 on the
    /// server if their bearer has already expired). New calls dispatch
    /// through the new session. Returns false when the entry is
    /// missing — caller should treat as a lost-entry error.
    pub fn replace_shared_mcp_session(
        &mut self,
        name: &str,
        url: String,
        session: Arc<McpSession>,
    ) -> bool {
        let id = McpHostId::shared(name);
        let Some(entry) = self.mcp_hosts.get_mut(&id) else {
            return false;
        };
        entry.spec.url = url;
        entry.session = Some(session);
        entry.state = ResourceState::Ready;
        entry.last_used = Utc::now();
        true
    }

    /// Register a shared MCP host as Errored — used when a startup
    /// connect fails but the catalog still knows about the entry.
    /// Surfaces the connect error on `ListSharedMcpHosts` so the
    /// operator doesn't have to dig in server logs to learn why the
    /// entry shows as disconnected.
    pub fn insert_shared_mcp_host_errored(
        &mut self,
        name: String,
        url: String,
        message: String,
    ) -> McpHostId {
        let id = McpHostId::shared(&name);
        let now = Utc::now();
        self.mcp_hosts.insert(
            id.clone(),
            McpHostEntry {
                id: id.clone(),
                spec: McpHostSpec {
                    url,
                    label: name,
                    per_task: false,
                },
                state: ResourceState::Errored { message },
                session: None,
                tools: Vec::new(),
                annotations: HashMap::new(),
                users: BTreeSet::new(),
                pinned: true,
                created_at: now,
                last_used: now,
            },
        );
        id
    }

    /// Eagerly register the MCP host belonging to a host env, as
    /// `Provisioning`. One-to-one with the [`HostEnvId`] — distinct
    /// host envs get distinct MCP hosts, while a second thread sharing
    /// the same binding dedups onto the existing entry (matching the
    /// host_env dedup). Idempotent on id.
    pub fn pre_register_host_env_mcp(
        &mut self,
        host_env_id: &HostEnvId,
        thread_id: &str,
    ) -> McpHostId {
        let id = McpHostId::for_host_env(host_env_id);
        let now = Utc::now();
        if let Some(entry) = self.mcp_hosts.get_mut(&id) {
            // If terminal (Errored from a prior failure, or TornDown
            // after a rebind), reset back to Provisioning so the next
            // provision future can complete cleanly.
            if entry.state.is_terminal() {
                entry.state = ResourceState::Provisioning { op_id: 0 };
                entry.session = None;
                entry.tools.clear();
                entry.annotations.clear();
                entry.spec.url.clear();
            }
            entry.users.insert(thread_id.to_string());
            entry.last_used = now;
            return id;
        }
        self.mcp_hosts.insert(
            id.clone(),
            McpHostEntry {
                id: id.clone(),
                spec: McpHostSpec {
                    url: String::new(),
                    label: format!("host_env:{}", host_env_id.0),
                    per_task: false,
                },
                state: ResourceState::Provisioning { op_id: 0 },
                session: None,
                tools: Vec::new(),
                annotations: HashMap::new(),
                users: BTreeSet::from([thread_id.to_string()]),
                pinned: false,
                created_at: now,
                last_used: now,
            },
        );
        id
    }

    /// Attach a connected `McpSession` to a previously pre-registered
    /// host-env-MCP entry, transitioning it to Ready. Returns false if
    /// the entry is missing or already Ready (caller can drop the
    /// session and avoid double-attach).
    pub fn complete_host_env_mcp(
        &mut self,
        id: &McpHostId,
        url: String,
        session: Arc<McpSession>,
    ) -> bool {
        let Some(entry) = self.mcp_hosts.get_mut(id) else {
            return false;
        };
        if entry.state.is_ready() {
            return false;
        }
        entry.spec.url = url;
        entry.session = Some(session);
        entry.state = ResourceState::Ready;
        entry.last_used = Utc::now();
        true
    }

    /// Mark a primary MCP host as Errored — provisioning failed somewhere
    /// in the chain (sandbox, MCP connect, or initial tools/list).
    pub fn mark_mcp_errored(&mut self, id: &McpHostId, message: String) {
        if let Some(entry) = self.mcp_hosts.get_mut(id) {
            entry.state = ResourceState::Errored { message };
            entry.last_used = Utc::now();
        }
    }

    /// Mark a sandbox as Errored.
    pub fn mark_host_env_errored(&mut self, id: &HostEnvId, message: String) {
        if let Some(entry) = self.host_envs.get_mut(id) {
            entry.state = ResourceState::Errored { message };
            entry.last_used = Utc::now();
        }
    }

    /// Mark a sandbox as Lost — it was provisioned successfully but
    /// the daemon session is no longer usable (daemon gone, restarted,
    /// or session forgotten). Drops the cached handle so GC doesn't
    /// try to RPC a teardown against a daemon that can't honor it.
    /// Returns whether the entry existed and transitioned.
    pub fn mark_host_env_lost(&mut self, id: &HostEnvId, message: String) -> bool {
        let Some(entry) = self.host_envs.get_mut(id) else {
            return false;
        };
        entry.state = ResourceState::Lost { message };
        entry.last_used = Utc::now();
        // Drop the handle without calling teardown — the daemon either
        // doesn't remember the session (restart) or can't be reached
        // (powered off). Either way the teardown RPC is useless.
        entry.handle = None;
        true
    }

    /// Mark a primary MCP host as Lost. Same semantics as Errored but
    /// specifically for the "was Ready, became unusable" transition.
    pub fn mark_mcp_lost(&mut self, id: &McpHostId, message: String) -> bool {
        let Some(entry) = self.mcp_hosts.get_mut(id) else {
            return false;
        };
        entry.state = ResourceState::Lost { message };
        entry.last_used = Utc::now();
        // Drop the cached session so future tool calls can't accidentally
        // hit the dead endpoint — they'll see the Lost state and the
        // scheduler will re-provision on the next thread's arrival.
        entry.session = None;
        true
    }

    /// Eagerly register a sandbox spec at thread-creation time, before any
    /// provisioning future runs. Returns the deterministic
    /// `HostEnvId::for_provider_spec(provider, spec)`; idempotent on the id (calling twice for
    /// the same spec just adds the user). Inserted entries land in
    /// `Provisioning` so the dispatcher can later either complete them
    /// (via [`Self::complete_host_env_provisioning`]) or, if they're already
    /// Ready from a previous thread's completion, skip the provision call
    /// entirely.
    ///
    /// The `op_id` field on `Provisioning` is set to 0 — there's no in-flight
    /// future yet at pre-register time. Phase 3d.ii's resolver promotes
    /// this to a real op id when it dispatches the provision.
    pub fn pre_register_host_env(
        &mut self,
        thread_id: &str,
        provider: String,
        spec: HostEnvSpec,
    ) -> HostEnvId {
        let id = HostEnvId::for_provider_spec(&provider, &spec);
        let now = Utc::now();
        match self.host_envs.get_mut(&id) {
            Some(entry) => {
                // If terminal (Errored from a prior failure, Lost from
                // a daemon-gone classification, or TornDown after the
                // last user left), reset back to Provisioning so the
                // next provision future re-establishes a fresh
                // sandbox. Handle / mcp_url / mcp_token are already
                // cleared on Lost; clear them here as a belt-and-
                // suspenders for Errored / TornDown.
                if entry.state.is_terminal() {
                    entry.state = ResourceState::Provisioning { op_id: 0 };
                    entry.handle = None;
                    entry.mcp_url = None;
                    entry.mcp_token = None;
                }
                entry.users.insert(thread_id.to_string());
                entry.last_used = now;
            }
            None => {
                self.host_envs.insert(
                    id.clone(),
                    HostEnvEntry {
                        id: id.clone(),
                        provider,
                        spec,
                        state: ResourceState::Provisioning { op_id: 0 },
                        users: BTreeSet::from([thread_id.to_string()]),
                        pinned: false,
                        created_at: now,
                        last_used: now,
                        handle: None,
                        mcp_url: None,
                        mcp_token: None,
                    },
                );
            }
        }
        id
    }

    /// Complete a previously pre-registered sandbox's provisioning by
    /// attaching the handle and url and transitioning to Ready. Returns an
    /// outcome that tells the caller whether the handle landed on the entry
    /// (happy path) or was redundant (dedup race / already torn down).
    pub fn complete_host_env_provisioning(
        &mut self,
        id: &HostEnvId,
        mcp_url: Option<String>,
        mcp_token: Option<String>,
        handle: Option<Box<dyn HostEnvHandle>>,
    ) -> CompleteHostEnvOutcome {
        let Some(entry) = self.host_envs.get_mut(id) else {
            return CompleteHostEnvOutcome::NotPreRegistered {
                unused_handle: handle,
            };
        };
        // Ready entries are already completed — a second completion
        // would either replace a live handle (leaking the old one) or
        // no-op pointlessly. Bounce the caller's handle back so it
        // gets torn down rather than leaked.
        if entry.state.is_ready() {
            return CompleteHostEnvOutcome::AlreadyCompleted {
                unused_handle: handle,
            };
        }
        entry.state = ResourceState::Ready;
        entry.handle = handle;
        entry.mcp_url = mcp_url;
        entry.mcp_token = mcp_token;
        entry.last_used = Utc::now();
        CompleteHostEnvOutcome::Completed
    }

    /// Drop `thread_id` from the sandbox's user set and return the new
    /// user count. Caller decides whether to tear down (count == 0 and
    /// not pinned).
    pub fn release_host_env_user(&mut self, id: &HostEnvId, thread_id: &str) -> usize {
        let Some(entry) = self.host_envs.get_mut(id) else {
            return 0;
        };
        entry.users.remove(thread_id);
        entry.last_used = Utc::now();
        entry.users.len()
    }

    /// Pull the sandbox handle out of the registry so the caller can spawn
    /// an async teardown without holding a borrow on the registry. The
    /// entry remains in the map (with `handle: None`) so inspectors can
    /// still see "this sandbox existed" — `mark_host_env_torn_down` is the
    /// follow-up call that flips its state.
    pub fn take_host_env_handle(&mut self, id: &HostEnvId) -> Option<Box<dyn HostEnvHandle>> {
        self.host_envs
            .get_mut(id)
            .and_then(|entry| entry.handle.take())
    }

    // ---------- Mutation helpers ----------

    pub fn populate_mcp_tools(
        &mut self,
        id: &McpHostId,
        tools: Vec<ToolDescriptor>,
        annotations: HashMap<String, ToolAnnotations>,
    ) {
        if let Some(entry) = self.mcp_hosts.get_mut(id) {
            entry.tools = tools;
            entry.annotations = annotations;
            entry.last_used = Utc::now();
        }
    }

    pub fn add_mcp_user(&mut self, id: &McpHostId, user: &str) {
        if let Some(entry) = self.mcp_hosts.get_mut(id) {
            entry.users.insert(user.to_string());
            entry.last_used = Utc::now();
        }
    }

    pub fn remove_mcp_user(&mut self, id: &McpHostId, user: &str) {
        if let Some(entry) = self.mcp_hosts.get_mut(id) {
            entry.users.remove(user);
            // Start the idle grace window from the moment the last
            // user actually leaves — otherwise `last_used` can carry
            // a stale timestamp from provisioning while the entry sits
            // unreferenced, and reap_idle would tear it down as soon
            // as the new in-use check passes.
            entry.last_used = Utc::now();
        }
    }

    pub fn add_host_env_user(&mut self, id: &HostEnvId, user: &str) {
        if let Some(entry) = self.host_envs.get_mut(id) {
            entry.users.insert(user.to_string());
            entry.last_used = Utc::now();
        }
    }

    pub fn remove_host_env_user(&mut self, id: &HostEnvId, user: &str) {
        if let Some(entry) = self.host_envs.get_mut(id) {
            entry.users.remove(user);
            entry.last_used = Utc::now();
        }
    }

    pub fn add_backend_user(&mut self, id: &BackendId, user: &str) {
        if let Some(entry) = self.backends.get_mut(id) {
            entry.users.insert(user.to_string());
            entry.last_used = Utc::now();
        }
    }

    pub fn remove_backend_user(&mut self, id: &BackendId, user: &str) {
        if let Some(entry) = self.backends.get_mut(id) {
            entry.users.remove(user);
        }
    }

    /// Mark a sandbox as torn down and drop its session/handle linkage. The
    /// entry itself stays in the map so inspectability tools can still see
    /// "this sandbox was used by task X." Phase 1b's GC pass reaps entries.
    pub fn mark_host_env_torn_down(&mut self, id: &HostEnvId) {
        if let Some(entry) = self.host_envs.get_mut(id) {
            entry.state = ResourceState::TornDown;
            entry.users.clear();
        }
    }

    pub fn mark_mcp_torn_down(&mut self, id: &McpHostId) {
        if let Some(entry) = self.mcp_hosts.get_mut(id) {
            entry.state = ResourceState::TornDown;
            entry.session = None;
            entry.users.clear();
        }
    }

    /// Single GC sweep over the registry. Two passes:
    ///
    /// 1. **Idle Ready entries** — sandboxes and MCP hosts that are
    ///    `Ready`, aren't pinned, aren't bound by any live thread
    ///    (per the caller-supplied `in_use_*` sets), and haven't been
    ///    touched since `now - idle_threshold`. These are torn down:
    ///    sandbox handles are pulled out (returned in the plan so the
    ///    caller can spawn `handle.teardown()`), MCP sessions are
    ///    dropped, and the entries flip to `TornDown`.
    ///
    ///    The "bound by a live thread" check is ground truth from the
    ///    scheduler's `tasks` map, not the registry's `users` refcount.
    ///    `users` is maintained for UI / broadcast but doesn't gate
    ///    GC — a thread parked on a model call or waiting on a
    ///    dispatched child doesn't touch the registry, so refcount-
    ///    only GC would race live work. As long as any thread still
    ///    has the resource in `bindings.host_env` (and isn't
    ///    Failed / Cancelled), the entry is pinned against reap.
    ///
    /// 2. **Stale terminal entries** — sandboxes and MCP hosts in
    ///    `Errored` or `TornDown` state whose `last_used` is older than
    ///    `now - terminal_retention`. These are removed from the map
    ///    entirely (the registry isn't a permanent ledger; we keep them
    ///    around for inspection, then evict).
    ///
    /// Backends are intentionally never GC'd — they're pinned by
    /// construction (cheap, no live socket).
    ///
    /// `now` is taken as a parameter rather than reading the clock
    /// internally so tests can drive deterministic timestamps.
    pub fn reap_idle(
        &mut self,
        now: DateTime<Utc>,
        idle_threshold: chrono::Duration,
        terminal_retention: chrono::Duration,
        in_use_host_envs: &HashSet<HostEnvId>,
        in_use_mcp_hosts: &HashSet<McpHostId>,
    ) -> ReapPlan {
        let mut plan = ReapPlan::default();

        // Pass 1a: idle Ready sandboxes -> tear down.
        let idle_sandboxes: Vec<HostEnvId> = self
            .host_envs
            .iter()
            .filter(|(id, e)| {
                e.state.is_ready()
                    && !e.pinned
                    && !in_use_host_envs.contains(id)
                    && now.signed_duration_since(e.last_used) >= idle_threshold
            })
            .map(|(id, _)| id.clone())
            .collect();
        for id in idle_sandboxes {
            let handle = self.host_envs.get_mut(&id).and_then(|e| e.handle.take());
            if let Some(entry) = self.host_envs.get_mut(&id) {
                entry.state = ResourceState::TornDown;
                entry.users.clear();
            }
            plan.torn_down_host_envs.push((id, handle));
        }

        // Pass 1b: idle Ready MCP hosts -> tear down (drop session).
        let idle_mcp: Vec<McpHostId> = self
            .mcp_hosts
            .iter()
            .filter(|(id, e)| {
                e.state.is_ready()
                    && !e.pinned
                    && !in_use_mcp_hosts.contains(id)
                    && now.signed_duration_since(e.last_used) >= idle_threshold
            })
            .map(|(id, _)| id.clone())
            .collect();
        for id in idle_mcp {
            if let Some(entry) = self.mcp_hosts.get_mut(&id) {
                entry.state = ResourceState::TornDown;
                entry.session = None;
                entry.users.clear();
            }
            plan.torn_down_mcp_hosts.push(id);
        }

        // Pass 2: stale terminal entries -> remove.
        let stale_sandboxes: Vec<HostEnvId> = self
            .host_envs
            .iter()
            .filter(|(_, e)| {
                e.state.is_terminal()
                    && now.signed_duration_since(e.last_used) >= terminal_retention
            })
            .map(|(id, _)| id.clone())
            .collect();
        for id in stale_sandboxes {
            self.host_envs.remove(&id);
            plan.destroyed_host_envs.push(id);
        }
        let stale_mcp: Vec<McpHostId> = self
            .mcp_hosts
            .iter()
            .filter(|(_, e)| {
                e.state.is_terminal()
                    && now.signed_duration_since(e.last_used) >= terminal_retention
            })
            .map(|(id, _)| id.clone())
            .collect();
        for id in stale_mcp {
            self.mcp_hosts.remove(&id);
            plan.destroyed_mcp_hosts.push(id);
        }

        plan
    }
}

/// Result of a single `ResourceRegistry::reap_idle` sweep. The scheduler
/// uses this to drive the side effects the registry can't do itself —
/// spawning async sandbox teardowns and broadcasting wire events.
#[derive(Default)]
pub struct ReapPlan {
    /// Idle sandbox entries that were marked `TornDown`. The handle is
    /// `Some` for entries the registry was holding a live handle for;
    /// the caller spawns `handle.teardown()`.
    pub torn_down_host_envs: Vec<(HostEnvId, Option<Box<dyn HostEnvHandle>>)>,
    /// Idle MCP host entries that were marked `TornDown`. Sessions were
    /// dropped in-place; nothing async for the caller to do, just emit
    /// the wire update.
    pub torn_down_mcp_hosts: Vec<McpHostId>,
    /// Stale terminal sandbox entries removed from the registry.
    pub destroyed_host_envs: Vec<HostEnvId>,
    /// Stale terminal MCP host entries removed from the registry.
    pub destroyed_mcp_hosts: Vec<McpHostId>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use whisper_agent_protocol::sandbox::PathAccess;

    #[test]
    fn id_constructors_are_stable() {
        // Spec-derived host-env ids are deterministic — same
        // (provider, spec), same id.
        let spec = HostEnvSpec::Landlock {
            allowed_paths: vec![],
            network: whisper_agent_protocol::sandbox::NetworkPolicy::Unrestricted,
        };
        let id1 = HostEnvId::for_provider_spec("local-landlock", &spec);
        let id2 = HostEnvId::for_provider_spec("local-landlock", &spec);
        assert_eq!(id1, id2);
        assert!(id1.0.starts_with("he-"));
        assert_eq!(id1.0.len(), "he-".len() + 16); // he- + 16 hex chars
        assert_eq!(McpHostId::shared("fetch").0, "mcp-shared-fetch");
        let mcp_id = McpHostId::for_host_env(&id1);
        assert_eq!(mcp_id.0, format!("mcp-{}", id1.0));
        assert_eq!(BackendId::for_name("anthropic").0, "backend-anthropic");
    }

    #[test]
    fn backend_user_refcount() {
        let mut reg = ResourceRegistry::new();
        let id = reg.insert_backend("anthropic".into(), "anthropic".into(), None);
        reg.add_backend_user(&id, "task-a");
        reg.add_backend_user(&id, "task-b");
        assert_eq!(reg.backends[&id].users.len(), 2);
        reg.remove_backend_user(&id, "task-a");
        assert_eq!(reg.backends[&id].users.len(), 1);
        assert!(reg.backends[&id].users.contains("task-b"));
    }

    #[test]
    fn sandbox_teardown_clears_users() {
        let mut reg = ResourceRegistry::new();
        let id = reg.pre_register_host_env(
            "t-1",
            "test-landlock".into(),
            HostEnvSpec::Landlock {
                allowed_paths: Vec::new(),
                network: Default::default(),
            },
        );
        let outcome = reg.complete_host_env_provisioning(&id, None, None, None);
        assert!(matches!(outcome, CompleteHostEnvOutcome::Completed));
        assert!(reg.host_envs[&id].state.is_ready());
        assert_eq!(reg.host_envs[&id].users.len(), 1);
        reg.mark_host_env_torn_down(&id);
        assert!(matches!(reg.host_envs[&id].state, ResourceState::TornDown));
        assert!(reg.host_envs[&id].users.is_empty());
    }

    #[test]
    fn pre_register_host_env_dedups_by_spec() {
        let mut reg = ResourceRegistry::new();
        let id_a = reg.pre_register_host_env(
            "t-1",
            "test-landlock".into(),
            HostEnvSpec::Landlock {
                allowed_paths: Vec::new(),
                network: Default::default(),
            },
        );
        let id_b = reg.pre_register_host_env(
            "t-2",
            "test-landlock".into(),
            HostEnvSpec::Landlock {
                allowed_paths: Vec::new(),
                network: Default::default(),
            },
        );
        assert_eq!(id_a, id_b);
        // Both threads share the same pre-registered entry.
        assert_eq!(reg.host_envs[&id_a].users.len(), 2);
        assert!(reg.host_envs[&id_a].users.contains("t-1"));
        assert!(reg.host_envs[&id_a].users.contains("t-2"));
        // State stays Provisioning until completion runs.
        assert!(matches!(
            reg.host_envs[&id_a].state,
            ResourceState::Provisioning { .. }
        ));
    }

    #[test]
    fn complete_host_env_provisioning_idempotent_on_race() {
        let mut reg = ResourceRegistry::new();
        let id = reg.pre_register_host_env(
            "t-1",
            "test-landlock".into(),
            HostEnvSpec::Landlock {
                allowed_paths: Vec::new(),
                network: Default::default(),
            },
        );
        let first = reg.complete_host_env_provisioning(&id, None, None, None);
        assert!(matches!(first, CompleteHostEnvOutcome::Completed));
        // Race-loser scenario: a second concurrent provision lands after
        // the entry already transitioned to Ready. The handle round-trips
        // back to the caller for teardown rather than silently leaking.
        let second = reg.complete_host_env_provisioning(&id, None, None, None);
        assert!(matches!(
            second,
            CompleteHostEnvOutcome::AlreadyCompleted { .. }
        ));
    }

    #[test]
    fn reap_idle_tears_down_idle_ready_and_removes_stale_terminal() {
        let mut reg = ResourceRegistry::new();
        // Idle Ready sandbox: provision, drop the user, backdate last_used.
        let idle_id = reg.pre_register_host_env(
            "t-idle",
            "test-landlock".into(),
            HostEnvSpec::Landlock {
                allowed_paths: Vec::new(),
                network: Default::default(),
            },
        );
        let _ = reg.complete_host_env_provisioning(&idle_id, None, None, None);
        let _ = reg.release_host_env_user(&idle_id, "t-idle");
        assert!(reg.host_envs[&idle_id].users.is_empty());

        let now = Utc::now();
        if let Some(e) = reg.host_envs.get_mut(&idle_id) {
            e.last_used = now - chrono::Duration::seconds(600);
        }

        // Pinned MCP host that's idle — must NOT be reaped. Inserted
        // directly because `insert_shared_mcp_host` wants a real
        // `Arc<McpSession>`.
        let pinned_id = McpHostId::shared("pinned");
        reg.mcp_hosts.insert(
            pinned_id.clone(),
            McpHostEntry {
                id: pinned_id.clone(),
                spec: McpHostSpec {
                    url: "http://x".into(),
                    label: "pinned".into(),
                    per_task: false,
                },
                state: ResourceState::Ready,
                session: None,
                tools: Vec::new(),
                annotations: HashMap::new(),
                users: BTreeSet::new(),
                pinned: true,
                created_at: now,
                last_used: now - chrono::Duration::seconds(9999),
            },
        );

        // Long-since-torn-down host env: a different spec so its id
        // differs from `idle_id`.
        let dead_id = reg.pre_register_host_env(
            "t-dead",
            "test-landlock".into(),
            HostEnvSpec::Landlock {
                allowed_paths: vec![PathAccess::read_only("/tmp/dead")],
                network: Default::default(),
            },
        );
        let _ = reg.complete_host_env_provisioning(&dead_id, None, None, None);
        reg.mark_host_env_torn_down(&dead_id);
        if let Some(e) = reg.host_envs.get_mut(&dead_id) {
            e.last_used = now - chrono::Duration::seconds(7200);
        }

        let plan = reg.reap_idle(
            now,
            chrono::Duration::seconds(300),
            chrono::Duration::seconds(3600),
            &HashSet::new(),
            &HashSet::new(),
        );
        assert_eq!(plan.torn_down_host_envs.len(), 1);
        assert_eq!(plan.torn_down_host_envs[0].0, idle_id);
        assert!(matches!(
            reg.host_envs[&idle_id].state,
            ResourceState::TornDown
        ));
        assert_eq!(plan.destroyed_host_envs, vec![dead_id.clone()]);
        assert!(!reg.host_envs.contains_key(&dead_id));
        assert!(plan.torn_down_mcp_hosts.is_empty());
        assert!(reg.mcp_hosts[&pinned_id].state.is_ready());
    }

    #[test]
    fn reap_idle_skips_recently_used() {
        let mut reg = ResourceRegistry::new();
        let id = reg.pre_register_host_env(
            "t-1",
            "test-landlock".into(),
            HostEnvSpec::Landlock {
                allowed_paths: Vec::new(),
                network: Default::default(),
            },
        );
        let _ = reg.complete_host_env_provisioning(&id, None, None, None);
        // last_used is `now`; should NOT reap on the first sweep even
        // with an empty in-use set — the idle threshold protects.
        let plan = reg.reap_idle(
            Utc::now(),
            chrono::Duration::seconds(300),
            chrono::Duration::seconds(3600),
            &HashSet::new(),
            &HashSet::new(),
        );
        assert!(plan.torn_down_host_envs.is_empty());
        assert!(reg.host_envs[&id].state.is_ready());
    }

    #[test]
    fn reap_idle_skips_entries_pinned_by_live_thread() {
        // Ground truth: a live thread binding this resource keeps
        // it alive regardless of the entry's `users` refcount or how
        // long it's been since `last_used` was bumped. Covers the
        // mavis-pod incident where a thread parked for 5+ minutes
        // between tool calls got its mcp-host reaped out from under
        // it mid-conversation.
        let mut reg = ResourceRegistry::new();
        let id = reg.pre_register_host_env(
            "t-parked",
            "test-landlock".into(),
            HostEnvSpec::Landlock {
                allowed_paths: Vec::new(),
                network: Default::default(),
            },
        );
        let _ = reg.complete_host_env_provisioning(&id, None, None, None);
        let mcp_id = McpHostId::for_host_env(&id);
        // Seed a Ready MCP entry so Pass 1b has a candidate. Simulate
        // the "refcount drifted empty but thread is still bound" race:
        // users is empty, last_used is ancient, idle threshold passed.
        reg.mcp_hosts.insert(
            mcp_id.clone(),
            McpHostEntry {
                id: mcp_id.clone(),
                spec: McpHostSpec {
                    url: "http://mcp".into(),
                    label: format!("host_env:{}", id.0),
                    per_task: false,
                },
                state: ResourceState::Ready,
                session: None,
                tools: Vec::new(),
                annotations: HashMap::new(),
                users: BTreeSet::new(),
                pinned: false,
                created_at: Utc::now(),
                last_used: Utc::now() - chrono::Duration::seconds(9999),
            },
        );
        if let Some(e) = reg.host_envs.get_mut(&id) {
            e.users.clear();
            e.last_used = Utc::now() - chrono::Duration::seconds(9999);
        }

        // Sweep with the thread listed as in-use: NOT reaped.
        let in_use_he: HashSet<HostEnvId> = [id.clone()].into();
        let in_use_mcp: HashSet<McpHostId> = [mcp_id.clone()].into();
        let plan = reg.reap_idle(
            Utc::now(),
            chrono::Duration::seconds(300),
            chrono::Duration::seconds(3600),
            &in_use_he,
            &in_use_mcp,
        );
        assert!(plan.torn_down_host_envs.is_empty());
        assert!(plan.torn_down_mcp_hosts.is_empty());
        assert!(reg.host_envs[&id].state.is_ready());
        assert!(reg.mcp_hosts[&mcp_id].state.is_ready());

        // Now the thread is gone (swept): same entries get reaped.
        let plan = reg.reap_idle(
            Utc::now(),
            chrono::Duration::seconds(300),
            chrono::Duration::seconds(3600),
            &HashSet::new(),
            &HashSet::new(),
        );
        assert_eq!(plan.torn_down_host_envs.len(), 1);
        assert_eq!(plan.torn_down_host_envs[0].0, id);
        assert_eq!(plan.torn_down_mcp_hosts, vec![mcp_id]);
    }

    #[test]
    fn release_host_env_user_decrements_count() {
        let mut reg = ResourceRegistry::new();
        let id = reg.pre_register_host_env(
            "t-1",
            "test-landlock".into(),
            HostEnvSpec::Landlock {
                allowed_paths: Vec::new(),
                network: Default::default(),
            },
        );
        reg.pre_register_host_env(
            "t-2",
            "test-landlock".into(),
            HostEnvSpec::Landlock {
                allowed_paths: Vec::new(),
                network: Default::default(),
            },
        );
        assert_eq!(reg.release_host_env_user(&id, "t-1"), 1);
        assert_eq!(reg.release_host_env_user(&id, "t-2"), 0);
    }

    #[test]
    fn mcp_tool_population() {
        use serde_json::json;
        let mut reg = ResourceRegistry::new();
        // Need an Arc<McpSession> — connect won't work in tests, so we skip
        // the per-task constructor and exercise the registry with a shared
        // host built around a never-used dummy session via unsafe? No — the
        // registry doesn't care about session contents for population. Use
        // `populate_mcp_tools` against an entry we manually insert.
        let id = McpHostId::shared("demo");
        reg.mcp_hosts.insert(
            id.clone(),
            McpHostEntry {
                id: id.clone(),
                spec: McpHostSpec {
                    url: "http://test".into(),
                    label: "demo".into(),
                    per_task: false,
                },
                state: ResourceState::Ready,
                session: None,
                tools: Vec::new(),
                annotations: HashMap::new(),
                users: BTreeSet::new(),
                pinned: true,
                created_at: Utc::now(),
                last_used: Utc::now(),
            },
        );
        let tool = ToolDescriptor {
            name: "echo".into(),
            description: "d".into(),
            input_schema: json!({}),
            annotations: ToolAnnotations::default(),
        };
        reg.populate_mcp_tools(&id, vec![tool], HashMap::new());
        assert_eq!(reg.mcp_hosts[&id].tools.len(), 1);
        assert_eq!(reg.mcp_hosts[&id].tools[0].name, "echo");
    }

    #[test]
    fn mark_host_env_lost_drops_handle_and_transitions() {
        let mut reg = ResourceRegistry::new();
        let id = reg.pre_register_host_env(
            "t-1",
            "p".into(),
            HostEnvSpec::Landlock {
                allowed_paths: Vec::new(),
                network: Default::default(),
            },
        );
        reg.complete_host_env_provisioning(
            &id,
            Some("http://mcp".into()),
            None,
            None, // no handle in the test
        );
        assert!(reg.host_envs[&id].state.is_ready());
        let transitioned = reg.mark_host_env_lost(&id, "daemon gone".into());
        assert!(transitioned);
        assert!(matches!(
            reg.host_envs[&id].state,
            ResourceState::Lost { .. }
        ));
        assert!(reg.host_envs[&id].state.is_terminal());
        assert!(reg.host_envs[&id].handle.is_none());
    }

    #[test]
    fn lost_host_env_resets_on_next_pre_register() {
        let mut reg = ResourceRegistry::new();
        let spec = HostEnvSpec::Landlock {
            allowed_paths: Vec::new(),
            network: Default::default(),
        };
        let id = reg.pre_register_host_env("t-1", "p".into(), spec.clone());
        reg.complete_host_env_provisioning(&id, Some("http://mcp".into()), None, None);
        reg.mark_host_env_lost(&id, "boom".into());
        // New thread arrives with same (provider, spec) → the entry
        // resets to Provisioning so provisioning can re-run.
        let id2 = reg.pre_register_host_env("t-2", "p".into(), spec);
        assert_eq!(id, id2);
        assert!(
            matches!(reg.host_envs[&id].state, ResourceState::Provisioning { .. }),
            "expected reset to Provisioning, got {:?}",
            reg.host_envs[&id].state
        );
        assert!(reg.host_envs[&id].mcp_url.is_none());
        assert!(reg.host_envs[&id].users.contains("t-2"));
    }

    #[test]
    fn mcp_host_id_distinguishes_shared_and_host_env() {
        let shared = McpHostId::shared("fetch");
        assert!(!shared.is_host_env());
        assert!(shared.host_env_id().is_none());

        let spec = HostEnvSpec::Landlock {
            allowed_paths: Vec::new(),
            network: Default::default(),
        };
        let he_id = HostEnvId::for_provider_spec("p", &spec);
        let he_mcp = McpHostId::for_host_env(&he_id);
        assert!(he_mcp.is_host_env());
        assert_eq!(he_mcp.host_env_id(), Some(he_id));
    }

    #[test]
    fn lost_terminal_state_label_maps_to_lost_wire() {
        let state = ResourceState::Lost {
            message: "x".into(),
        };
        assert_eq!(state.label(), ResourceStateLabel::Lost);
        assert!(state.is_terminal());
    }
}
