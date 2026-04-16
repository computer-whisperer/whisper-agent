//! Resource registries — sandboxes, MCP hosts, and model backends as first-class
//! entities with lifecycles independent of the tasks/threads that reference them.
//!
//! See `docs/design_session_thread_scheduler.md` for the end-state design. This
//! module is the Phase 1a foundation: types and a passive registry that mirrors
//! the existing per-task resource ownership without changing behavior. Later
//! phases promote this from a shadow registry to the authoritative store, drop
//! the per-task `mcp_pools`/`sandbox_handles` fields, and add wire/UI surface.
//!
//! Keys throughout this module are `String` rather than wrapped task/thread IDs
//! because Phase 1a still routes everything through `task_id`. When Phase 2
//! introduces `ThreadId`, the `users: BTreeSet<String>` field swaps for
//! `BTreeSet<ThreadId>` and the rest of the module stays put.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use whisper_agent_protocol::SandboxSpec;

use crate::mcp::{McpSession, ToolAnnotations, ToolDescriptor};

#[derive(Serialize, Deserialize, Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct SandboxId(pub String);

#[derive(Serialize, Deserialize, Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct McpHostId(pub String);

#[derive(Serialize, Deserialize, Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct BackendId(pub String);

impl SandboxId {
    pub fn for_task(task_id: &str) -> Self {
        Self(format!("sb-{task_id}"))
    }
}

impl McpHostId {
    pub fn primary_for_task(task_id: &str) -> Self {
        Self(format!("mcp-primary-{task_id}"))
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
}

#[derive(Debug)]
pub struct SandboxEntry {
    pub id: SandboxId,
    pub spec: SandboxSpec,
    pub state: ResourceState,
    /// Set of task_ids (later: ThreadIds) currently using this sandbox.
    pub users: BTreeSet<String>,
    pub pinned: bool,
    pub created_at: DateTime<Utc>,
    pub last_used: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct McpHostSpec {
    /// Endpoint URL. For per-task primaries this is whatever the sandbox
    /// reported (or fallback `mcp_host_url`); for shared hosts it's the
    /// configured URL.
    pub url: String,
    /// Stable display name. For shared hosts this is the catalog name; for
    /// per-task primaries it's `"primary:<task_id>"`.
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

impl ResourceRegistry {
    pub fn new() -> Self {
        Self::default()
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
        task_id: &str,
        url: String,
        session: Arc<McpSession>,
    ) -> McpHostId {
        let id = McpHostId::primary_for_task(task_id);
        let now = Utc::now();
        self.mcp_hosts.insert(
            id.clone(),
            McpHostEntry {
                id: id.clone(),
                spec: McpHostSpec {
                    url,
                    label: format!("primary:{task_id}"),
                    per_task: true,
                },
                state: ResourceState::Ready,
                session: Some(session),
                tools: Vec::new(),
                annotations: HashMap::new(),
                users: BTreeSet::from([task_id.to_string()]),
                pinned: false,
                created_at: now,
                last_used: now,
            },
        );
        id
    }

    /// Insert a Ready per-task sandbox entry.
    pub fn insert_sandbox_for_task(&mut self, task_id: &str, spec: SandboxSpec) -> SandboxId {
        let id = SandboxId::for_task(task_id);
        let now = Utc::now();
        self.sandboxes.insert(
            id.clone(),
            SandboxEntry {
                id: id.clone(),
                spec,
                state: ResourceState::Ready,
                users: BTreeSet::from([task_id.to_string()]),
                pinned: false,
                created_at: now,
                last_used: now,
            },
        );
        id
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
        assert_eq!(SandboxId::for_task("t-1").0, "sb-t-1");
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
        let id = reg.insert_sandbox_for_task("t-1", SandboxSpec::None);
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
