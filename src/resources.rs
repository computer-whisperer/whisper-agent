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

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use whisper_agent_protocol::{ResourceSnapshot, ResourceStateLabel, SandboxSpec};

use crate::mcp::{McpSession, ToolAnnotations, ToolDescriptor};
use crate::sandbox::SandboxHandle;

#[derive(Serialize, Deserialize, Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct SandboxId(pub String);

#[derive(Serialize, Deserialize, Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct McpHostId(pub String);

#[derive(Serialize, Deserialize, Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct BackendId(pub String);

impl SandboxId {
    /// Derive a stable id from the sandbox spec. Identical specs hash to
    /// identical ids — this is how Phase 3b's dedup works: when a second
    /// thread requests a sandbox whose spec matches an existing entry,
    /// `register_sandbox` finds the entry by id and adds the thread as a
    /// user instead of inserting a duplicate.
    ///
    /// Hash is `DefaultHasher` over the spec's JSON serialization. Not
    /// cryptographic; collision risk on a few-hundred-pod scale is
    /// negligible. The 16-hex-char width keeps ids readable in logs.
    pub fn for_spec(spec: &SandboxSpec) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let json = serde_json::to_string(spec).unwrap_or_default();
        let mut hasher = DefaultHasher::new();
        json.hash(&mut hasher);
        Self(format!("sb-{:016x}", hasher.finish()))
    }
}

impl McpHostId {
    pub fn primary_for_task(thread_id: &str) -> Self {
        Self(format!("mcp-primary-{thread_id}"))
    }
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
    Provisioning { op_id: u64 },
    Ready,
    Errored { message: String },
    /// Resource has been torn down. Entry lingers briefly for inspection then
    /// is reaped on the next GC pass.
    TornDown,
}

impl ResourceState {
    pub fn is_ready(&self) -> bool {
        matches!(self, ResourceState::Ready)
    }
    pub fn is_terminal(&self) -> bool {
        matches!(self, ResourceState::Errored { .. } | ResourceState::TornDown)
    }
    pub fn label(&self) -> ResourceStateLabel {
        match self {
            ResourceState::Provisioning { .. } => ResourceStateLabel::Provisioning,
            ResourceState::Ready => ResourceStateLabel::Ready,
            ResourceState::Errored { .. } => ResourceStateLabel::Errored,
            ResourceState::TornDown => ResourceStateLabel::TornDown,
        }
    }
}

pub struct SandboxEntry {
    pub id: SandboxId,
    pub spec: SandboxSpec,
    pub state: ResourceState,
    /// Set of task_ids (later: ThreadIds) currently using this sandbox.
    pub users: BTreeSet<String>,
    pub pinned: bool,
    pub created_at: DateTime<Utc>,
    pub last_used: DateTime<Utc>,
    /// The provisioned sandbox itself. Owned by the registry (Phase 3a).
    /// Taken out of the entry by `take_sandbox_handle` at teardown so the
    /// scheduler can spawn the async teardown future without holding a
    /// borrow. Cleared on `mark_sandbox_torn_down` when the handle has
    /// already been retrieved.
    pub handle: Option<Box<dyn SandboxHandle>>,
    /// The MCP URL the sandbox advertises (or the fallback used to connect
    /// to it when the sandbox doesn't run its own MCP). Cached on the
    /// entry so dedup hits don't have to consult the live handle — and so
    /// the URL stays correct even after the handle is taken for teardown.
    pub mcp_url: Option<String>,
}

impl std::fmt::Debug for SandboxEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SandboxEntry")
            .field("id", &self.id)
            .field("spec", &self.spec)
            .field("state", &self.state)
            .field("users", &self.users)
            .field("pinned", &self.pinned)
            .field("created_at", &self.created_at)
            .field("last_used", &self.last_used)
            .field("handle", &self.handle.as_ref().map(|_| "<sandbox>"))
            .field("mcp_url", &self.mcp_url)
            .finish()
    }
}

/// Outcome of `ResourceRegistry::register_sandbox`. Distinguishes a fresh
/// insert from a dedup hit so the caller knows whether to spawn teardown
/// of the now-redundant handle it just provisioned.
pub enum RegisterSandboxOutcome {
    /// New entry inserted; the passed handle is now owned by the registry.
    Created(SandboxId),
    /// An entry with this spec already existed; the passed handle is
    /// returned to the caller (so it can spawn teardown). The thread has
    /// been added to the existing entry's user set.
    AlreadyExists {
        id: SandboxId,
        unused_handle: Option<Box<dyn SandboxHandle>>,
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
    pub sandboxes: HashMap<SandboxId, SandboxEntry>,
    pub mcp_hosts: HashMap<McpHostId, McpHostEntry>,
    pub backends: HashMap<BackendId, BackendEntry>,
}

impl SandboxEntry {
    pub fn to_snapshot(&self) -> ResourceSnapshot {
        ResourceSnapshot::Sandbox {
            id: self.id.0.clone(),
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
        let mut out: Vec<ResourceSnapshot> = Vec::with_capacity(
            self.sandboxes.len() + self.mcp_hosts.len() + self.backends.len(),
        );
        let mut sandboxes: Vec<&SandboxEntry> = self.sandboxes.values().collect();
        sandboxes.sort_by(|a, b| a.id.cmp(&b.id));
        out.extend(sandboxes.into_iter().map(SandboxEntry::to_snapshot));
        let mut mcps: Vec<&McpHostEntry> = self.mcp_hosts.values().collect();
        mcps.sort_by(|a, b| a.id.cmp(&b.id));
        out.extend(mcps.into_iter().map(McpHostEntry::to_snapshot));
        let mut backends: Vec<&BackendEntry> = self.backends.values().collect();
        backends.sort_by(|a, b| a.id.cmp(&b.id));
        out.extend(backends.into_iter().map(BackendEntry::to_snapshot));
        out
    }

    pub fn snapshot_sandbox(&self, id: &SandboxId) -> Option<ResourceSnapshot> {
        self.sandboxes.get(id).map(SandboxEntry::to_snapshot)
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

    /// Insert a Ready per-task primary MCP host entry. The session must already
    /// be connected; for primaries the registry never observes a Provisioning
    /// state in Phase 1a (that lives in the task state machine until Phase 3
    /// moves it here).
    pub fn insert_primary_mcp_host(
        &mut self,
        thread_id: &str,
        url: String,
        session: Arc<McpSession>,
    ) -> McpHostId {
        let id = McpHostId::primary_for_task(thread_id);
        let now = Utc::now();
        self.mcp_hosts.insert(
            id.clone(),
            McpHostEntry {
                id: id.clone(),
                spec: McpHostSpec {
                    url,
                    label: format!("primary:{thread_id}"),
                    per_task: true,
                },
                state: ResourceState::Ready,
                session: Some(session),
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

    /// Spec-based registration. If a Ready entry with the same spec id
    /// already exists, add this thread as a user and hand the unused
    /// handle back to the caller so it can spawn its teardown. Otherwise
    /// insert a fresh entry with the given handle.
    ///
    /// `mcp_url` is the URL the sandbox is reachable at (from
    /// `handle.mcp_url()` or the configured fallback) — cached on the
    /// entry so future dedup hits don't have to consult the handle.
    pub fn register_sandbox(
        &mut self,
        thread_id: &str,
        spec: SandboxSpec,
        mcp_url: Option<String>,
        handle: Option<Box<dyn SandboxHandle>>,
    ) -> RegisterSandboxOutcome {
        let id = SandboxId::for_spec(&spec);
        if let Some(entry) = self.sandboxes.get_mut(&id)
            && entry.state.is_ready()
        {
            entry.users.insert(thread_id.to_string());
            entry.last_used = Utc::now();
            return RegisterSandboxOutcome::AlreadyExists {
                id,
                unused_handle: handle,
            };
        }
        let now = Utc::now();
        self.sandboxes.insert(
            id.clone(),
            SandboxEntry {
                id: id.clone(),
                spec,
                state: ResourceState::Ready,
                users: BTreeSet::from([thread_id.to_string()]),
                pinned: false,
                created_at: now,
                last_used: now,
                handle,
                mcp_url,
            },
        );
        RegisterSandboxOutcome::Created(id)
    }

    /// Find a Ready sandbox whose spec matches; used by the dispatcher to
    /// skip provisioning when an existing sandbox can be reused. Only
    /// returns entries whose handle is still present — we don't want to
    /// dedup onto a sandbox whose handle was just taken for teardown.
    pub fn find_ready_sandbox_for_spec(
        &self,
        spec: &SandboxSpec,
    ) -> Option<&SandboxEntry> {
        let id = SandboxId::for_spec(spec);
        self.sandboxes
            .get(&id)
            .filter(|e| e.state.is_ready() && e.handle.is_some())
    }

    /// Find which sandbox a given thread is currently bound to. Linear
    /// scan over the registry — fine at the few-dozen-sandbox scale
    /// Phase 3 targets. Phase 3d's `ThreadBindings` makes this an O(1)
    /// thread-side lookup.
    pub fn find_sandbox_used_by(&self, thread_id: &str) -> Option<SandboxId> {
        self.sandboxes
            .values()
            .find(|e| e.users.contains(thread_id))
            .map(|e| e.id.clone())
    }

    /// Drop `thread_id` from the sandbox's user set and return the new
    /// user count. Caller decides whether to tear down (count == 0 and
    /// not pinned).
    pub fn release_sandbox_user(&mut self, id: &SandboxId, thread_id: &str) -> usize {
        let Some(entry) = self.sandboxes.get_mut(id) else {
            return 0;
        };
        entry.users.remove(thread_id);
        entry.last_used = Utc::now();
        entry.users.len()
    }

    /// Pull the sandbox handle out of the registry so the caller can spawn
    /// an async teardown without holding a borrow on the registry. The
    /// entry remains in the map (with `handle: None`) so inspectors can
    /// still see "this sandbox existed" — `mark_sandbox_torn_down` is the
    /// follow-up call that flips its state.
    pub fn take_sandbox_handle(&mut self, id: &SandboxId) -> Option<Box<dyn SandboxHandle>> {
        self.sandboxes
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
        }
    }

    pub fn add_sandbox_user(&mut self, id: &SandboxId, user: &str) {
        if let Some(entry) = self.sandboxes.get_mut(id) {
            entry.users.insert(user.to_string());
            entry.last_used = Utc::now();
        }
    }

    pub fn remove_sandbox_user(&mut self, id: &SandboxId, user: &str) {
        if let Some(entry) = self.sandboxes.get_mut(id) {
            entry.users.remove(user);
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
    pub fn mark_sandbox_torn_down(&mut self, id: &SandboxId) {
        if let Some(entry) = self.sandboxes.get_mut(id) {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_constructors_are_stable() {
        // Spec-derived sandbox ids are deterministic — same spec, same id.
        let id1 = SandboxId::for_spec(&SandboxSpec::None);
        let id2 = SandboxId::for_spec(&SandboxSpec::None);
        assert_eq!(id1, id2);
        assert!(id1.0.starts_with("sb-"));
        assert_eq!(id1.0.len(), "sb-".len() + 16); // sb- + 16 hex chars
        assert_eq!(
            McpHostId::primary_for_task("t-1").0,
            "mcp-primary-t-1"
        );
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
    fn sandbox_teardown_clears_users() {
        let mut reg = ResourceRegistry::new();
        let outcome = reg.register_sandbox("t-1", SandboxSpec::None, None, None);
        let RegisterSandboxOutcome::Created(id) = outcome else {
            panic!("first register should be Created");
        };
        assert!(reg.sandboxes[&id].state.is_ready());
        assert_eq!(reg.sandboxes[&id].users.len(), 1);
        reg.mark_sandbox_torn_down(&id);
        assert!(matches!(
            reg.sandboxes[&id].state,
            ResourceState::TornDown
        ));
        assert!(reg.sandboxes[&id].users.is_empty());
    }

    #[test]
    fn register_sandbox_dedups_by_spec() {
        let mut reg = ResourceRegistry::new();
        let first = reg.register_sandbox("t-1", SandboxSpec::None, None, None);
        let second = reg.register_sandbox("t-2", SandboxSpec::None, None, None);
        let RegisterSandboxOutcome::Created(id_a) = first else {
            panic!("first should be Created");
        };
        let RegisterSandboxOutcome::AlreadyExists { id: id_b, .. } = second else {
            panic!("second should be AlreadyExists");
        };
        assert_eq!(id_a, id_b);
        // Both threads now share the entry's user set.
        assert_eq!(reg.sandboxes[&id_a].users.len(), 2);
        assert!(reg.sandboxes[&id_a].users.contains("t-1"));
        assert!(reg.sandboxes[&id_a].users.contains("t-2"));
    }

    #[test]
    fn release_sandbox_user_decrements_count() {
        let mut reg = ResourceRegistry::new();
        let RegisterSandboxOutcome::Created(id) =
            reg.register_sandbox("t-1", SandboxSpec::None, None, None)
        else {
            panic!("created");
        };
        reg.register_sandbox("t-2", SandboxSpec::None, None, None);
        assert_eq!(reg.release_sandbox_user(&id, "t-1"), 1);
        assert_eq!(reg.release_sandbox_user(&id, "t-2"), 0);
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
}
