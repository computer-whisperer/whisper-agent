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
mod filesystem;
mod grep;
mod list_threads;

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
        let McpContentBlock::Text { text } = b;
        s.push_str(text);
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
pub const POD_LIST_FILES: &str = "pod_list_files";
pub const POD_GREP: &str = "pod_grep";
pub const POD_LIST_THREADS: &str = "pod_list_threads";
pub const POD_ABOUT: &str = "pod_about";
pub const POD_RUN_BEHAVIOR: &str = "pod_run_behavior";
pub const POD_SET_BEHAVIOR_ENABLED: &str = "pod_set_behavior_enabled";

/// True if `name` is a builtin pod tool. Used by the scheduler's router
/// to branch the tool-call dispatch path.
pub fn is_builtin(name: &str) -> bool {
    matches!(
        name,
        POD_READ_FILE
            | POD_WRITE_FILE
            | POD_EDIT_FILE
            | POD_LIST_FILES
            | POD_GREP
            | POD_LIST_THREADS
            | POD_ABOUT
            | POD_RUN_BEHAVIOR
            | POD_SET_BEHAVIOR_ENABLED
    )
}

/// True if `name` is a builtin tool that modifies the pod. Used by
/// `evaluate_policy` so `ApprovalPolicy::PromptPodModify` can gate these
/// specifically without prompting on every MCP tool.
pub fn is_pod_modify(name: &str) -> bool {
    matches!(name, POD_WRITE_FILE | POD_EDIT_FILE)
}

/// Tool descriptors for the builtin set, shaped for direct inclusion in
/// the model's advertised catalog.
pub fn descriptors() -> Vec<McpTool> {
    vec![
        filesystem::list_descriptor(),
        filesystem::read_descriptor(),
        filesystem::write_descriptor(),
        filesystem::edit_descriptor(),
        grep::descriptor(),
        list_threads::descriptor(),
        about::descriptor(),
        behavior_control::run_behavior_descriptor(),
        behavior_control::set_behavior_enabled_descriptor(),
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
pub async fn dispatch(
    pod_dir: PathBuf,
    pod_config: PodConfig,
    behavior_ids: Vec<String>,
    tool_name: &str,
    args: Value,
) -> ToolOutcome {
    let allowed = filesystem::allowed_filenames(&pod_config, &behavior_ids);
    match tool_name {
        POD_LIST_FILES => filesystem::list_files(&pod_dir, &allowed, args).await,
        POD_READ_FILE => filesystem::read_file(&pod_dir, &allowed, &behavior_ids, args).await,
        POD_WRITE_FILE => filesystem::write_file(&pod_dir, &allowed, &behavior_ids, args).await,
        POD_EDIT_FILE => filesystem::edit_file(&pod_dir, &allowed, &behavior_ids, args).await,
        POD_GREP => grep::run(&pod_dir, args).await,
        POD_LIST_THREADS => list_threads::run(&pod_dir, args).await,
        POD_ABOUT => about::run(args),
        POD_RUN_BEHAVIOR => behavior_control::run_behavior(&behavior_ids, args),
        POD_SET_BEHAVIOR_ENABLED => behavior_control::set_behavior_enabled(&behavior_ids, args),
        other => no_update_error(format!("unknown builtin tool: {other}")),
    }
}

