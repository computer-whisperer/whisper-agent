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
use whisper_agent_protocol::{BackendUsage, ResourceSnapshot, ResourceStateLabel};

use crate::tools::mcp::{McpSession, ToolAnnotations, ToolDescriptor};

#[derive(Serialize, Deserialize, Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct McpHostId(pub String);

#[derive(Serialize, Deserialize, Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct BackendId(pub String);

impl McpHostId {
    /// Registry id for a shared MCP host. `name` is the catalog name
    /// pod entries and thread bindings reference.
    pub fn shared(name: &str) -> Self {
        Self(format!("mcp-shared-{name}"))
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
    /// Most recent account/quota snapshot. `None` until either a
    /// model call comes back with usage headers (Codex path) or a
    /// client requests an explicit refresh. Backends that don't
    /// expose usage data stay `None` forever.
    pub usage: Option<BackendUsage>,
}

/// Two flat registries owned by the scheduler. v1 host envs are gone;
/// v2 daemon sessions are owned by the daemon and tracked in
/// [`crate::runtime::scheduler::V2SessionStore`], not here.
#[derive(Default, Debug)]
pub struct ResourceRegistry {
    pub mcp_hosts: HashMap<McpHostId, McpHostEntry>,
    pub backends: HashMap<BackendId, BackendEntry>,
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
            usage: self.usage.clone(),
        }
    }
}

impl ResourceRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot every entry for `ListResources` responses. Order is
    /// mcp hosts then backends — stable and human-readable.
    pub fn snapshot_all(&self) -> Vec<ResourceSnapshot> {
        let mut out: Vec<ResourceSnapshot> =
            Vec::with_capacity(self.mcp_hosts.len() + self.backends.len());
        let mut mcps: Vec<&McpHostEntry> = self.mcp_hosts.values().collect();
        mcps.sort_by(|a, b| a.id.cmp(&b.id));
        out.extend(mcps.into_iter().map(McpHostEntry::to_snapshot));
        let mut backends: Vec<&BackendEntry> = self.backends.values().collect();
        backends.sort_by(|a, b| a.id.cmp(&b.id));
        out.extend(backends.into_iter().map(BackendEntry::to_snapshot));
        out
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
                usage: None,
            },
        );
        id
    }

    /// Replace the cached usage snapshot for a backend. Used by the
    /// model-call path (header scrape) and the active refresh path
    /// (polling the backend's usage endpoint). No-op if the backend
    /// is unknown — a missing entry means the catalog was reloaded
    /// out from under an in-flight update, not a bug worth surfacing.
    pub fn set_backend_usage(&mut self, id: &BackendId, usage: BackendUsage) {
        if let Some(entry) = self.backends.get_mut(id) {
            entry.usage = Some(usage);
        }
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

    /// Mark a primary MCP host as Errored — provisioning failed somewhere
    /// in the chain (sandbox, MCP connect, or initial tools/list).
    pub fn mark_mcp_errored(&mut self, id: &McpHostId, message: String) {
        if let Some(entry) = self.mcp_hosts.get_mut(id) {
            entry.state = ResourceState::Errored { message };
            entry.last_used = Utc::now();
        }
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
        in_use_mcp_hosts: &HashSet<McpHostId>,
    ) -> ReapPlan {
        let mut plan = ReapPlan::default();

        // Pass 1: idle Ready MCP hosts -> tear down (drop session).
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

        // Pass 2: stale terminal MCP entries -> remove.
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
/// broadcasting wire events.
#[derive(Default)]
pub struct ReapPlan {
    /// Idle MCP host entries that were marked `TornDown`. Sessions were
    /// dropped in-place; nothing async for the caller to do, just emit
    /// the wire update.
    pub torn_down_mcp_hosts: Vec<McpHostId>,
    /// Stale terminal MCP host entries removed from the registry.
    pub destroyed_mcp_hosts: Vec<McpHostId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_constructors_are_stable() {
        assert_eq!(McpHostId::shared("fetch").0, "mcp-shared-fetch");
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
    fn lost_terminal_state_label_maps_to_lost_wire() {
        let state = ResourceState::Lost {
            message: "x".into(),
        };
        assert_eq!(state.label(), ResourceStateLabel::Lost);
        assert!(state.is_terminal());
    }
}
