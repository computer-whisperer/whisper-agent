//! In-process tools that let an agent edit its own pod's configuration.
//!
//! Mirrors mcp-host's `read_file` / `write_file` / `edit_file` shape but
//! scoped to a whitelist of filenames derived from the pod's config:
//!
//!   * `pod.toml` — always allowed. Writes are parsed + validated by
//!     [`crate::pod::parse_toml`] before landing on disk; on validation
//!     failure the file is untouched and the tool returns an error.
//!   * The file named by `thread_defaults.system_prompt_file` — allowed
//!     whenever non-empty. No structural validation (plain text).
//!   * `behaviors/<id>/behavior.toml` — parsed + validated by
//!     [`crate::pod::behaviors::parse_toml`]. Writing to a new `<id>`
//!     creates the behavior directory and seeds `state.json` with a
//!     default `BehaviorState`.
//!   * `behaviors/<id>/prompt.md` — plain text; the initial user-message
//!     template fired threads run against. Only writable for ids that
//!     already have a `behavior.toml` (the agent must create the
//!     behavior first).
//!
//! Successful writes emit a [`PodUpdate`] so the scheduler can refresh
//! in-memory pod state and broadcast the change to subscribers in the
//! same step as the disk write. See
//! [`crate::runtime::scheduler::Scheduler::apply_pod_config_update`] and
//! [`crate::runtime::scheduler::Scheduler::apply_behavior_config_update`].

mod about;
mod behavior_control;
pub mod describe_tool;
pub mod dispatch_thread;
mod filesystem;
pub mod find_tool;
mod grep;
pub mod knowledge_modify;
pub mod knowledge_query;
pub mod list_host_env_providers;
pub mod list_llm_providers;
pub mod list_mcp_hosts;
mod list_threads;
mod pod_show_thread;
pub mod sudo;

use std::collections::HashMap;
use std::path::PathBuf;

use serde_json::Value;
use whisper_agent_protocol::{BehaviorConfig, PodConfig};

#[cfg(test)]
use crate::pod;
use crate::tools::mcp::{
    CallToolResult, McpContentBlock, ToolAnnotations, ToolDescriptor as McpTool,
};

fn text_result(text: String) -> CallToolResult {
    CallToolResult {
        content: vec![McpContentBlock::Text { text }],
        is_error: false,
    }
}

fn error_result(text: String) -> CallToolResult {
    CallToolResult {
        content: vec![McpContentBlock::Text { text }],
        is_error: true,
    }
}

// Test fixtures live at file level (gated on `cfg(test)`) so every
// submodule's `mod tests` can reach them via `super::temp_dir()` etc.
// without duplication or a pub(super) test_support module.
#[cfg(test)]
fn temp_dir() -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path =
        std::env::temp_dir().join(format!("wa-builtin-tools-test-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&path).unwrap();
    path
}

#[cfg(test)]
fn sample_config() -> PodConfig {
    let text = r#"
name = "t"
created_at = "2026-04-17T00:00:00Z"
[allow]
backends = ["anthropic"]
mcp_hosts = []
host_env = []
[thread_defaults]
backend = "anthropic"
model = "claude-opus-4-7"
system_prompt_file = "system_prompt.md"
max_tokens = 8000
max_turns = 50
"#;
    pod::parse_toml(text).unwrap()
}

#[cfg(test)]
fn join_blocks(blocks: &[McpContentBlock]) -> String {
    let mut s = String::new();
    for b in blocks {
        if let McpContentBlock::Text { text } = b {
            s.push_str(text);
        }
    }
    s
}

/// Shortcut for a tool response that didn't trigger a pod update: just
/// an error payload, no [`PodUpdate`], no scheduler command.
fn no_update_error(message: String) -> ToolOutcome {
    ToolOutcome {
        result: error_result(message),
        pod_update: None,
        scheduler_command: None,
    }
}

/// Shortcut for a tool response that didn't trigger a pod update: just
/// a text payload, no [`PodUpdate`], no scheduler command.
fn no_update_text(message: String) -> ToolOutcome {
    ToolOutcome {
        result: text_result(message),
        pod_update: None,
        scheduler_command: None,
    }
}

pub const POD_READ_FILE: &str = "pod_read_file";
pub const POD_WRITE_FILE: &str = "pod_write_file";
pub const POD_EDIT_FILE: &str = "pod_edit_file";
pub const POD_REMOVE_FILE: &str = "pod_remove_file";
pub const POD_LIST_FILES: &str = "pod_list_files";
pub const POD_GREP: &str = "pod_grep";
pub const POD_LIST_THREADS: &str = "pod_list_threads";
pub const POD_SHOW_THREAD: &str = "pod_show_thread";
pub const ABOUT: &str = "about";
pub const POD_RUN_BEHAVIOR: &str = "pod_run_behavior";
pub const POD_SET_BEHAVIOR_ENABLED: &str = "pod_set_behavior_enabled";
pub const DISPATCH_THREAD: &str = "dispatch_thread";
pub const SUDO: &str = "sudo";
pub const DESCRIBE_TOOL: &str = "describe_tool";
pub const FIND_TOOL: &str = "find_tool";
pub const LIST_LLM_PROVIDERS: &str = "list_llm_providers";
pub const LIST_MCP_HOSTS: &str = "list_mcp_hosts";
pub const LIST_HOST_ENV_PROVIDERS: &str = "list_host_env_providers";
pub const KNOWLEDGE_QUERY: &str = "knowledge_query";
pub const KNOWLEDGE_MODIFY: &str = "knowledge_modify";

/// True if `name` is a builtin pod tool. Used by the scheduler's router
/// to branch the tool-call dispatch path.
pub fn is_builtin(name: &str) -> bool {
    matches!(
        name,
        POD_READ_FILE
            | POD_WRITE_FILE
            | POD_EDIT_FILE
            | POD_REMOVE_FILE
            | POD_LIST_FILES
            | POD_GREP
            | POD_LIST_THREADS
            | POD_SHOW_THREAD
            | ABOUT
            | POD_RUN_BEHAVIOR
            | POD_SET_BEHAVIOR_ENABLED
            | DISPATCH_THREAD
            | SUDO
            | DESCRIBE_TOOL
            | FIND_TOOL
            | LIST_LLM_PROVIDERS
            | LIST_MCP_HOSTS
            | LIST_HOST_ENV_PROVIDERS
            | KNOWLEDGE_QUERY
            | KNOWLEDGE_MODIFY
    )
}

/// Env-name prefixes that would shadow a builtin tool after host-env
/// tool-name prefixing (`{env_name}_{tool_name}`). Returned as the
/// sorted, deduplicated set of everything up to the first underscore
/// in every builtin name — so adding a new builtin automatically
/// expands the reserved set without bespoke bookkeeping.
///
/// Used by the pod validator to reject `[[allow.host_env]]` entries
/// whose name would collide with a builtin prefix.
pub fn reserved_env_name_prefixes() -> Vec<&'static str> {
    const BUILTINS: &[&str] = &[
        POD_READ_FILE,
        POD_WRITE_FILE,
        POD_EDIT_FILE,
        POD_REMOVE_FILE,
        POD_LIST_FILES,
        POD_GREP,
        POD_LIST_THREADS,
        POD_SHOW_THREAD,
        ABOUT,
        POD_RUN_BEHAVIOR,
        POD_SET_BEHAVIOR_ENABLED,
        DISPATCH_THREAD,
        SUDO,
        DESCRIBE_TOOL,
        FIND_TOOL,
        LIST_LLM_PROVIDERS,
        LIST_MCP_HOSTS,
        LIST_HOST_ENV_PROVIDERS,
        KNOWLEDGE_QUERY,
        KNOWLEDGE_MODIFY,
    ];
    let mut out: Vec<&'static str> = BUILTINS
        .iter()
        .map(|n| n.split_once('_').map(|(p, _)| p).unwrap_or(*n))
        .collect();
    out.sort();
    out.dedup();
    out
}

/// Tool descriptors for the builtin set, shaped for direct inclusion in
/// the model's advertised catalog.
pub fn descriptors() -> Vec<McpTool> {
    vec![
        filesystem::list_descriptor(),
        filesystem::read_descriptor(),
        filesystem::write_descriptor(),
        filesystem::edit_descriptor(),
        filesystem::remove_descriptor(),
        grep::descriptor(),
        list_threads::descriptor(),
        pod_show_thread::descriptor(),
        about::descriptor(),
        behavior_control::run_behavior_descriptor(),
        behavior_control::set_behavior_enabled_descriptor(),
        dispatch_thread::descriptor(),
        sudo::descriptor(),
        describe_tool::descriptor(),
        find_tool::descriptor(),
        list_llm_providers::descriptor(),
        list_mcp_hosts::descriptor(),
        list_host_env_providers::descriptor(),
        knowledge_query::descriptor(),
        knowledge_modify::descriptor(),
    ]
}

/// Server-declared annotations for each builtin tool. Keyed by tool name
/// so `annotations_for` can merge them into the thread's overall
/// annotation map alongside MCP-sourced annotations.
pub fn annotations() -> HashMap<String, ToolAnnotations> {
    descriptors()
        .into_iter()
        .map(|t| (t.name, t.annotations))
        .collect()
}

/// Side effect to apply to in-memory pod state when a write succeeds.
/// Emitted by the tool handler and consumed by the scheduler.
#[derive(Debug, Clone)]
pub enum PodUpdate {
    /// A new `pod.toml` that has already been parsed and validated.
    Config {
        toml_text: String,
        parsed: Box<PodConfig>,
    },
    /// New system-prompt text.
    SystemPrompt { text: String },
    /// A new or updated `behaviors/<id>/behavior.toml`. Parsed and
    /// validated before emission. The scheduler decides whether this is
    /// a create or an update by looking at its in-memory pod state.
    BehaviorConfig {
        behavior_id: String,
        toml_text: String,
        parsed: Box<BehaviorConfig>,
    },
    /// New text for a behavior's `prompt.md`. Only valid when the
    /// behavior already exists in-memory (see module docs); the tool
    /// handler rejects prompt writes for unknown ids.
    BehaviorPrompt { behavior_id: String, text: String },
    /// A `pod_remove_file` on `behaviors/<id>/behavior.toml` removed
    /// the behavior's entire directory from disk. The scheduler
    /// unregisters the in-memory behavior and broadcasts
    /// `BehaviorDeleted`.
    BehaviorDeleted { behavior_id: String },
}

/// Runtime action a builtin tool asks the scheduler to perform AFTER
/// returning the tool result to the model. These bypass disk writes —
/// they're for flipping runtime state (pause/resume) or kicking a
/// behavior run. The scheduler applies one of these in the same loop
/// iteration that consumes the tool result, keeping broadcasts and the
/// ToolCallEnd in the same well-defined order.
#[derive(Debug, Clone)]
pub enum SchedulerCommand {
    /// Manually fire a behavior. Equivalent to the UI's Run button:
    /// bypasses the cron/paused gates (it's an explicit action). The
    /// optional payload is substituted into `prompt.md`'s `{{payload}}`
    /// the same way a webhook trigger's body is.
    RunBehavior {
        behavior_id: String,
        payload: Option<serde_json::Value>,
    },
    /// Pause or resume a single behavior. Wraps the scheduler's
    /// `SetBehaviorEnabled` handler — on pause drops any queued
    /// payload, on resume bumps the cron cursor to now so catch-up
    /// doesn't replay missed windows.
    SetBehaviorEnabled { behavior_id: String, enabled: bool },
}

/// What a builtin tool handler produces: the tool result the model sees,
/// plus optional state/command side effects the scheduler applies
/// atomically in the same completion step.
pub struct ToolOutcome {
    pub result: CallToolResult,
    pub pod_update: Option<PodUpdate>,
    pub scheduler_command: Option<SchedulerCommand>,
}

/// Dispatch a builtin tool call. `pod_dir` is the pod's on-disk directory;
/// `pod_config` is the current in-memory config; `behavior_ids` lists
/// the behaviors currently registered (used to derive the dynamic slice
/// of the allowlist and to distinguish create vs update on write).
/// `pod_modify` is the calling thread's scope-level pod-modify cap —
/// pod-file reads/writes are denied at this layer when the target path
/// isn't admitted by the cap, regardless of what the filename
/// allowlist says.
pub async fn dispatch(
    pod_dir: PathBuf,
    pod_config: PodConfig,
    behavior_ids: Vec<String>,
    pod_modify: crate::permission::PodModifyCap,
    behaviors_cap: crate::permission::BehaviorOpsCap,
    tool_name: &str,
    args: Value,
) -> ToolOutcome {
    let allowed = filesystem::allowed_filenames(&pod_config, &behavior_ids);
    // Pod-file *writes* are gated by `PodModifyCap::admits(path)` before
    // reaching the handler. Reads use the broader filename allowlist —
    // inspecting pod-internal state (`threads/*.json`, etc.) is not a
    // capability-raising action, so the cap hierarchy doesn't apply.
    // See `docs/design_permissions_rework.md`.
    if matches!(tool_name, POD_WRITE_FILE | POD_EDIT_FILE | POD_REMOVE_FILE)
        && let Some(filename) = args.get("filename").and_then(|v| v.as_str())
        && !pod_modify.admits(filename)
    {
        return no_update_error(format!(
            "`{tool_name}` on `{filename}` is denied by the thread's pod_modify capability \
             ({pod_modify:?}). Ask for scope widening if this action is needed."
        ));
    }
    // Removing a behavior (`behaviors/<id>/behavior.toml`) requires
    // `behaviors` cap ≥ author_narrower — the same tier authorized to
    // create one. None / Read can neither mint nor retire behaviors.
    // Memory removes and future plain-file removes pass through on
    // the pod_modify check above; non-behavior paths skip this gate.
    if tool_name == POD_REMOVE_FILE
        && let Some(filename) = args.get("filename").and_then(|v| v.as_str())
        && filename.starts_with(&format!("{}/", crate::pod::behaviors::BEHAVIORS_DIR))
        && filename.ends_with(&format!("/{}", crate::pod::behaviors::BEHAVIOR_TOML))
        && !matches!(
            behaviors_cap,
            crate::permission::BehaviorOpsCap::AuthorNarrower
                | crate::permission::BehaviorOpsCap::AuthorAny
        )
    {
        return no_update_error(format!(
            "`{tool_name}` on `{filename}` requires behaviors cap ≥ author_narrower \
             (have: {behaviors_cap:?}). Ask for scope widening if this action is needed."
        ));
    }
    // Behavior-subsystem tools are gated by the thread's `behaviors` cap.
    // `None` denies all access; `Read` and above admit run / pause-resume
    // (they're invocations/state-flips on an already-authored asset, not
    // authoring). Authoring — creating or modifying a behavior's config
    // via `pod_write_file` — is gated at the scheduler layer by
    // `check_behavior_authoring`, which also re-checks AuthorNarrower /
    // AuthorAny against the caller's scope.
    if matches!(tool_name, POD_RUN_BEHAVIOR | POD_SET_BEHAVIOR_ENABLED)
        && behaviors_cap == crate::permission::BehaviorOpsCap::None
    {
        return no_update_error(format!(
            "`{tool_name}` is denied by the thread's behaviors capability \
             ({behaviors_cap:?}). Ask for scope widening if this action is needed."
        ));
    }
    // Reads on behavior files (`behaviors/<id>/*`) require `behaviors`
    // cap ≥ Read. Autonomous / no-behavior threads can't inspect the
    // pod's behavior library. Per design: "Read = can list / read
    // behavior configs and prompts."
    if tool_name == POD_READ_FILE
        && let Some(filename) = args.get("filename").and_then(|v| v.as_str())
        && filename.starts_with(&format!("{}/", crate::pod::behaviors::BEHAVIORS_DIR))
        && behaviors_cap == crate::permission::BehaviorOpsCap::None
    {
        return no_update_error(format!(
            "reading `{filename}` requires behaviors cap ≥ read (have: {behaviors_cap:?}). \
             Ask for scope widening if this action is needed."
        ));
    }
    let system_prompt_file = pod_config.thread_defaults.system_prompt_file.as_str();
    match tool_name {
        POD_LIST_FILES => filesystem::list_files(&pod_dir, &allowed, &behavior_ids, args).await,
        POD_READ_FILE => filesystem::read_file(&pod_dir, &allowed, &behavior_ids, args).await,
        POD_WRITE_FILE => {
            filesystem::write_file(&pod_dir, &allowed, &behavior_ids, system_prompt_file, args)
                .await
        }
        POD_EDIT_FILE => {
            filesystem::edit_file(&pod_dir, &allowed, &behavior_ids, system_prompt_file, args).await
        }
        POD_REMOVE_FILE => filesystem::remove_file(&pod_dir, &allowed, &behavior_ids, args).await,
        POD_GREP => grep::run(&pod_dir, args).await,
        POD_LIST_THREADS => list_threads::run(&pod_dir, args).await,
        POD_SHOW_THREAD => pod_show_thread::run(&pod_dir, args).await,
        ABOUT => about::run(args),
        POD_RUN_BEHAVIOR => behavior_control::run_behavior(&behavior_ids, args),
        POD_SET_BEHAVIOR_ENABLED => behavior_control::set_behavior_enabled(&behavior_ids, args),
        DISPATCH_THREAD => no_update_error(
            "dispatch_thread must be intercepted at the scheduler layer \
             (io_dispatch::tool_call); reaching this arm is a bug"
                .into(),
        ),
        SUDO => no_update_error(
            "sudo must be intercepted at the scheduler layer \
             (register_sudo_tool); reaching this arm is a bug"
                .into(),
        ),
        DESCRIBE_TOOL => no_update_error(
            "describe_tool must be intercepted at the scheduler layer \
             (complete_describe_tool_call); reaching this arm is a bug"
                .into(),
        ),
        FIND_TOOL => no_update_error(
            "find_tool must be intercepted at the scheduler layer \
             (complete_find_tool_call); reaching this arm is a bug"
                .into(),
        ),
        LIST_LLM_PROVIDERS => no_update_error(
            "list_llm_providers must be intercepted at the scheduler layer \
             (complete_list_llm_providers_call); reaching this arm is a bug"
                .into(),
        ),
        LIST_MCP_HOSTS => no_update_error(
            "list_mcp_hosts must be intercepted at the scheduler layer \
             (complete_list_mcp_hosts_call); reaching this arm is a bug"
                .into(),
        ),
        LIST_HOST_ENV_PROVIDERS => no_update_error(
            "list_host_env_providers must be intercepted at the scheduler layer \
             (complete_list_host_env_providers_call); reaching this arm is a bug"
                .into(),
        ),
        KNOWLEDGE_QUERY => no_update_error(
            "knowledge_query must be intercepted at the scheduler layer \
             (complete_knowledge_query_call); reaching this arm is a bug"
                .into(),
        ),
        KNOWLEDGE_MODIFY => no_update_error(
            "knowledge_modify must be intercepted at the scheduler layer \
             (complete_knowledge_modify_call); reaching this arm is a bug"
                .into(),
        ),
        other => no_update_error(format!("unknown builtin tool: {other}")),
    }
}
