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
    /// `pre_register_sandbox` finds the entry by id and adds the thread as
    /// a user instead of inserting a duplicate.
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

/// Outcome of `ResourceRegistry::complete_sandbox_provisioning`. Distinguishes
/// the happy path (entry was waiting for us to fill in the handle) from the
/// dedup race (someone else already completed it) so the caller knows
/// whether to spawn teardown of the now-redundant handle it provisioned.
pub enum CompleteSandboxOutcome {
    /// We attached the handle to a pre-registered (Provisioning) entry and
    /// transitioned it to Ready. Registry owns the handle.
    Completed,
    /// Entry was already Ready when we arrived — handle is unused, caller
    /// should tear it down. Happens when two concurrent `mcp_connect`
    /// futures provision against the same SandboxId before either finishes.
    AlreadyCompleted {
        unused_handle: Option<Box<dyn SandboxHandle>>,
    },
    /// Entry didn't exist at all — `pre_register_sandbox` wasn't called for
    /// this id. Treated as a programmer error by callers; the unused handle
    /// is returned so it can be torn down rather than leaked.
    NotPreRegistered {
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

    /// Eagerly register the per-thread primary MCP host as `Provisioning`
    /// at thread-creation time, so the registry is the source of truth for
    /// "is this thread's MCP ready" from create onward. The
    /// `fallback_url` is the URL the host would connect to if its sandbox
    /// doesn't advertise its own — replaced when the sandbox provision
    /// completes and the actual session connects.
    ///
    /// Idempotent on id: re-registering for the same thread just touches
    /// `last_used` and adds the user (which was already there) — no
    /// state regression.
    pub fn pre_register_primary_mcp_host(
        &mut self,
        thread_id: &str,
        fallback_url: String,
    ) -> McpHostId {
        let id = McpHostId::primary_for_task(thread_id);
        let now = Utc::now();
        if let Some(entry) = self.mcp_hosts.get_mut(&id) {
            // If terminal (Errored from a prior failure, or TornDown
            // after a sandbox rebind), reset back to Provisioning so the
            // next `provision_primary_mcp` future can complete cleanly.
            if entry.state.is_terminal() {
                entry.state = ResourceState::Provisioning { op_id: 0 };
                entry.session = None;
                entry.tools.clear();
                entry.annotations.clear();
                entry.spec.url = fallback_url;
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
                    url: fallback_url,
                    label: format!("primary:{thread_id}"),
                    per_task: true,
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
    /// primary entry, transitioning it to Ready. The url is updated too —
    /// the actual sandbox-advertised URL may differ from the fallback we
    /// pre-registered with. Returns false if the entry is missing or
    /// already Ready (caller can drop the session and avoid double-attach).
    pub fn complete_primary_mcp_host(
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
    pub fn mark_sandbox_errored(&mut self, id: &SandboxId, message: String) {
        if let Some(entry) = self.sandboxes.get_mut(id) {
            entry.state = ResourceState::Errored { message };
            entry.last_used = Utc::now();
        }
    }

    /// Eagerly register a sandbox spec at thread-creation time, before any
    /// provisioning future runs. Returns the deterministic
    /// `SandboxId::for_spec(spec)`; idempotent on the id (calling twice for
    /// the same spec just adds the user). Inserted entries land in
    /// `Provisioning` so the dispatcher can later either complete them
    /// (via [`Self::complete_sandbox_provisioning`]) or, if they're already
    /// Ready from a previous thread's completion, skip the provision call
    /// entirely.
    ///
    /// The `op_id` field on `Provisioning` is set to 0 — there's no in-flight
    /// future yet at pre-register time. Phase 3d.ii's resolver promotes
    /// this to a real op id when it dispatches the provision.
    pub fn pre_register_sandbox(&mut self, thread_id: &str, spec: SandboxSpec) -> SandboxId {
        let id = SandboxId::for_spec(&spec);
        let now = Utc::now();
        match self.sandboxes.get_mut(&id) {
            Some(entry) => {
                entry.users.insert(thread_id.to_string());
                entry.last_used = now;
            }
            None => {
                self.sandboxes.insert(
                    id.clone(),
                    SandboxEntry {
                        id: id.clone(),
                        spec,
                        state: ResourceState::Provisioning { op_id: 0 },
                        users: BTreeSet::from([thread_id.to_string()]),
                        pinned: false,
                        created_at: now,
                        last_used: now,
                        handle: None,
                        mcp_url: None,
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
    pub fn complete_sandbox_provisioning(
        &mut self,
        id: &SandboxId,
        mcp_url: Option<String>,
        handle: Option<Box<dyn SandboxHandle>>,
    ) -> CompleteSandboxOutcome {
        let Some(entry) = self.sandboxes.get_mut(id) else {
            return CompleteSandboxOutcome::NotPreRegistered {
                unused_handle: handle,
            };
        };
        // Ready entries are already completed — even if their handle is
        // None (e.g. SandboxSpec::None never produces one), a second
        // completion would either replace a live handle (leaking the old
        // one) or no-op pointlessly. Bounce the caller's handle back so it
        // gets torn down rather than leaked.
        if entry.state.is_ready() {
            return CompleteSandboxOutcome::AlreadyCompleted {
                unused_handle: handle,
            };
        }
        entry.state = ResourceState::Ready;
        entry.handle = handle;
        entry.mcp_url = mcp_url;
        entry.last_used = Utc::now();
        CompleteSandboxOutcome::Completed
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

    /// Single GC sweep over the registry. Two passes:
    ///
    /// 1. **Idle Ready entries** — sandboxes and MCP hosts that are
    ///    `Ready`, have no users, aren't pinned, and haven't been
    ///    touched since `now - idle_threshold`. These are torn down:
    ///    sandbox handles are pulled out (returned in the plan so the
    ///    caller can spawn `handle.teardown()`), MCP sessions are
    ///    dropped, and the entries flip to `TornDown`.
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
    ) -> ReapPlan {
        let mut plan = ReapPlan::default();

        // Pass 1a: idle Ready sandboxes -> tear down.
        let idle_sandboxes: Vec<SandboxId> = self
            .sandboxes
            .iter()
            .filter(|(_, e)| {
                e.state.is_ready()
                    && !e.pinned
                    && e.users.is_empty()
                    && now.signed_duration_since(e.last_used) >= idle_threshold
            })
            .map(|(id, _)| id.clone())
            .collect();
        for id in idle_sandboxes {
            let handle = self
                .sandboxes
                .get_mut(&id)
                .and_then(|e| e.handle.take());
            if let Some(entry) = self.sandboxes.get_mut(&id) {
                entry.state = ResourceState::TornDown;
                entry.users.clear();
            }
            plan.torn_down_sandboxes.push((id, handle));
        }

        // Pass 1b: idle Ready MCP hosts -> tear down (drop session).
        let idle_mcp: Vec<McpHostId> = self
            .mcp_hosts
            .iter()
            .filter(|(_, e)| {
                e.state.is_ready()
                    && !e.pinned
                    && e.users.is_empty()
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
        let stale_sandboxes: Vec<SandboxId> = self
            .sandboxes
            .iter()
            .filter(|(_, e)| {
                e.state.is_terminal()
                    && now.signed_duration_since(e.last_used) >= terminal_retention
            })
            .map(|(id, _)| id.clone())
            .collect();
        for id in stale_sandboxes {
            self.sandboxes.remove(&id);
            plan.destroyed_sandboxes.push(id);
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
    pub torn_down_sandboxes: Vec<(SandboxId, Option<Box<dyn SandboxHandle>>)>,
    /// Idle MCP host entries that were marked `TornDown`. Sessions were
    /// dropped in-place; nothing async for the caller to do, just emit
    /// the wire update.
    pub torn_down_mcp_hosts: Vec<McpHostId>,
    /// Stale terminal sandbox entries removed from the registry.
    pub destroyed_sandboxes: Vec<SandboxId>,
    /// Stale terminal MCP host entries removed from the registry.
    pub destroyed_mcp_hosts: Vec<McpHostId>,
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
        let id = reg.pre_register_sandbox("t-1", SandboxSpec::None);
        let outcome = reg.complete_sandbox_provisioning(&id, None, None);
        assert!(matches!(outcome, CompleteSandboxOutcome::Completed));
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
    fn pre_register_sandbox_dedups_by_spec() {
        let mut reg = ResourceRegistry::new();
        let id_a = reg.pre_register_sandbox("t-1", SandboxSpec::None);
        let id_b = reg.pre_register_sandbox("t-2", SandboxSpec::None);
        assert_eq!(id_a, id_b);
        // Both threads share the same pre-registered entry.
        assert_eq!(reg.sandboxes[&id_a].users.len(), 2);
        assert!(reg.sandboxes[&id_a].users.contains("t-1"));
        assert!(reg.sandboxes[&id_a].users.contains("t-2"));
        // State stays Provisioning until completion runs.
        assert!(matches!(
            reg.sandboxes[&id_a].state,
            ResourceState::Provisioning { .. }
        ));
    }

    #[test]
    fn complete_sandbox_provisioning_idempotent_on_race() {
        let mut reg = ResourceRegistry::new();
        let id = reg.pre_register_sandbox("t-1", SandboxSpec::None);
        let first = reg.complete_sandbox_provisioning(&id, None, None);
        assert!(matches!(first, CompleteSandboxOutcome::Completed));
        // Race-loser scenario: a second concurrent provision lands after
        // the entry already transitioned to Ready. The handle round-trips
        // back to the caller for teardown rather than silently leaking.
        let second = reg.complete_sandbox_provisioning(&id, None, None);
        assert!(matches!(
            second,
            CompleteSandboxOutcome::AlreadyCompleted { .. }
        ));
    }

    #[test]
    fn reap_idle_tears_down_idle_ready_and_removes_stale_terminal() {
        let mut reg = ResourceRegistry::new();
        // Idle Ready sandbox: provision, drop the user, backdate last_used.
        let idle_id = reg.pre_register_sandbox("t-idle", SandboxSpec::None);
        let _ = reg.complete_sandbox_provisioning(&idle_id, None, None);
        let _ = reg.release_sandbox_user(&idle_id, "t-idle");
        assert!(reg.sandboxes[&idle_id].users.is_empty());

        let now = Utc::now();
        if let Some(e) = reg.sandboxes.get_mut(&idle_id) {
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

        // Long-since-torn-down sandbox: a different spec so its id
        // differs from `idle_id`.
        let dead_id = reg.pre_register_sandbox(
            "t-dead",
            SandboxSpec::Landlock {
                allowed_paths: Vec::new(),
                network: Default::default(),
            },
        );
        let _ = reg.complete_sandbox_provisioning(&dead_id, None, None);
        reg.mark_sandbox_torn_down(&dead_id);
        if let Some(e) = reg.sandboxes.get_mut(&dead_id) {
            e.last_used = now - chrono::Duration::seconds(7200);
        }

        let plan = reg.reap_idle(
            now,
            chrono::Duration::seconds(300),
            chrono::Duration::seconds(3600),
        );
        assert_eq!(plan.torn_down_sandboxes.len(), 1);
        assert_eq!(plan.torn_down_sandboxes[0].0, idle_id);
        assert!(matches!(
            reg.sandboxes[&idle_id].state,
            ResourceState::TornDown
        ));
        assert_eq!(plan.destroyed_sandboxes, vec![dead_id.clone()]);
        assert!(!reg.sandboxes.contains_key(&dead_id));
        assert!(plan.torn_down_mcp_hosts.is_empty());
        assert!(reg.mcp_hosts[&pinned_id].state.is_ready());
    }

    #[test]
    fn reap_idle_skips_recently_used_and_user_held() {
        let mut reg = ResourceRegistry::new();
        let id = reg.pre_register_sandbox("t-1", SandboxSpec::None);
        let _ = reg.complete_sandbox_provisioning(&id, None, None);
        // last_used is `now`; should NOT reap on the first sweep.
        let plan = reg.reap_idle(
            Utc::now(),
            chrono::Duration::seconds(300),
            chrono::Duration::seconds(3600),
        );
        assert!(plan.torn_down_sandboxes.is_empty());
        assert!(reg.sandboxes[&id].state.is_ready());
        // Still has its user.
        assert_eq!(reg.sandboxes[&id].users.len(), 1);
    }

    #[test]
    fn release_sandbox_user_decrements_count() {
        let mut reg = ResourceRegistry::new();
        let id = reg.pre_register_sandbox("t-1", SandboxSpec::None);
        reg.pre_register_sandbox("t-2", SandboxSpec::None);
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
