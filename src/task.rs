//! Task — the unit of long-lived agent work, modeled as tasks-as-data.
//!
//! A Task is a serializable state machine. The scheduler drives it via two methods:
//!
//! - [`Task::step`] advances the state synchronously. It returns a [`StepOutcome`]
//!   telling the scheduler whether to dispatch an I/O op, continue stepping, or pause
//!   until input arrives.
//! - [`Task::apply_io_result`] integrates the completion of a previously-dispatched
//!   I/O op back into the task.
//!
//! Both methods push [`TaskEvent`]s into an out-param so the scheduler can translate
//! them to wire-protocol events and broadcast to subscribers.
//!
//! The internal [`TaskInternalState`] has finer distinctions than the public
//! [`TaskStateLabel`] — the wire collapses them via [`Task::public_state`]. This
//! indirection is the point of having a state machine: we can split a phase into
//! sub-phases (e.g. add `AwaitingApproval` between tool dispatch and execution) without
//! touching the wire.

use std::collections::{BTreeSet, HashMap};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use whisper_agent_protocol::{
    ApprovalChoice, ApprovalPolicy, ContentBlock, Conversation, Message, TaskConfig, TaskSnapshot,
    TaskStateLabel, TaskSummary, ToolResultContent, Usage,
};

use crate::mcp::{CallToolResult, ToolAnnotations};
use crate::model::ModelResponse;

pub type OpId = u64;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Task {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub last_active: DateTime<Utc>,
    pub title: Option<String>,
    pub config: TaskConfig,
    pub conversation: Conversation,
    pub total_usage: Usage,
    #[serde(default)]
    pub archived: bool,
    /// Turns issued in the current user-message cycle; resets on user message.
    #[serde(default)]
    pub turns_in_cycle: u32,
    /// Per-task "always allow" set keyed by tool name. Calls to a name in this
    /// set bypass the approval prompt regardless of `approval_policy`. Persisted
    /// with the task, so the allowlist survives restart.
    #[serde(default)]
    pub tool_allowlist: BTreeSet<String>,
    pub internal: TaskInternalState,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TaskInternalState {
    /// No conversation yet / never run.
    Idle,
    /// Previous loop ended on `end_turn`; accepts follow-up user messages.
    Completed,
    /// A user message has been appended and we must connect MCP before calling the model.
    NeedsMcpConnect,
    AwaitingMcpConnect { op_id: OpId },
    /// MCP connected; need to list tools before issuing any model call.
    NeedsListTools,
    AwaitingListTools { op_id: OpId },
    /// Ready to dispatch a model call. The conversation already contains the latest
    /// user or tool_result message.
    NeedsModelCall,
    AwaitingModel { op_id: OpId, started_at: DateTime<Utc> },
    /// Model responded with tool_uses; at least one needs user approval before we can
    /// dispatch. `dispositions` is parallel to `tool_uses`. On every ApprovalDecision the
    /// corresponding entry transitions Pending → UserApproved / UserRejected; once none
    /// remain Pending the task moves to `AwaitingTools`.
    AwaitingApproval {
        tool_uses: Vec<ToolUseReq>,
        dispositions: Vec<ApprovalDisposition>,
    },
    /// Model responded with tool_uses. Each entry in `pending_dispatch` still needs to
    /// be fired at MCP; `pending_io` maps op_ids of in-flight tool calls to their
    /// tool_use_ids; `completed` accumulates ToolResult blocks as they return. A tool
    /// that was rejected during the approval phase is already synthesized into
    /// `completed` here with `is_error: true` so the model's next turn sees the denial.
    ///
    /// `approvals` carries the decision that led each tool to be dispatched so the audit
    /// log can record it when the tool completes (one line per call — see
    /// `design_permissions.md`). Keyed by tool_use_id.
    AwaitingTools {
        pending_dispatch: Vec<ToolUseReq>,
        pending_io: HashMap<OpId, String>,
        completed: Vec<ContentBlock>,
        #[serde(default)]
        approvals: HashMap<String, ApprovalRecord>,
    },
    /// Terminal: unrecoverable error.
    Failed { at_phase: String, message: String },
    /// Terminal: user-initiated cancellation. In-flight I/O may still complete; their
    /// results are discarded by the scheduler.
    Cancelled,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolUseReq {
    pub tool_use_id: String,
    pub name: String,
    pub input: serde_json::Value,
}

/// Per-tool-call approval state carried by [`TaskInternalState::AwaitingApproval`].
/// Parallel to the tool_uses vector; each entry is always in one of these four states.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ApprovalDisposition {
    /// Policy or allowlist auto-approved this call. `reason` becomes the
    /// `who_decided` field on the audit log entry — `policy:read_only`,
    /// `policy:auto_approve_all`, or `policy:allowlist`.
    AutoApproved {
        #[serde(default)]
        reason: String,
    },
    /// Awaiting a user decision keyed by `approval_id`.
    Pending {
        approval_id: String,
        destructive: bool,
        read_only: bool,
    },
    /// User approved.
    UserApproved {
        approval_id: String,
        decided_by_conn: Option<u64>,
    },
    /// User rejected. `message` is the text we'll surface to the model as the
    /// synthesized tool_result when we leave the approval phase.
    UserRejected {
        approval_id: String,
        decided_by_conn: Option<u64>,
        message: String,
    },
}

impl ApprovalDisposition {
    fn is_pending(&self) -> bool {
        matches!(self, ApprovalDisposition::Pending { .. })
    }
}

/// Decision record threaded from an approval outcome into the dispatch phase so the
/// audit log captures who said yes (or that policy auto-approved).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ApprovalRecord {
    pub decision: String,    // "auto" | "approve" | "reject"
    pub who_decided: String, // "policy:read_only" | "policy:auto_approve_all" | "user:{conn}"
}

/// Request for an I/O operation to be dispatched by the scheduler.
#[derive(Debug)]
pub enum IoRequest {
    McpConnect {
        op_id: OpId,
    },
    ListTools {
        op_id: OpId,
    },
    ModelCall {
        op_id: OpId,
    },
    ToolCall {
        op_id: OpId,
        tool_use_id: String,
        name: String,
        input: serde_json::Value,
    },
}

/// Successful result of an I/O op. Errors are carried in-band so apply_io_result can
/// transition the task to Failed with the right phase tag.
///
/// The `*Success` variants carry scheduler-owned resources (MCP session, tool list)
/// that get peeled off in [`crate::scheduler`] before the result reaches the task
/// state machine; the task itself only sees the slim `McpConnect` / `ListTools`
/// boolean variants.
pub enum IoResult {
    McpConnectSuccess {
        session: std::sync::Arc<crate::mcp::McpSession>,
        sandbox_handle: Option<Box<dyn crate::sandbox::SandboxHandle>>,
    },
    McpConnect(Result<(), String>),
    ListToolsSuccess {
        tools: Vec<crate::mcp::ToolDescriptor>,
        /// `tool_name → index into the task's pool sessions vec`. Scheduler-
        /// internal — peeled off in apply_io_completion before the task sees
        /// the result.
        routing: std::collections::HashMap<String, usize>,
    },
    ListTools(Result<(), String>),
    ModelCall(Result<ModelResponse, String>),
    ToolCall {
        tool_use_id: String,
        result: Result<CallToolResult, String>,
    },
}

#[derive(Debug)]
pub enum StepOutcome {
    /// Dispatch this I/O op, then wait for it.
    DispatchIo(IoRequest),
    /// State advanced synchronously; call step() again.
    Continue,
    /// No further progress possible until an I/O result or user input arrives.
    Paused,
}

/// Internal task event — translated by the scheduler into wire [`ServerToClient`] events.
#[derive(Debug, Clone)]
pub enum TaskEvent {
    AssistantBegin { turn: u32 },
    AssistantText { text: String },
    /// Chain-of-thought block. Emitted in the same order as text/tool-use
    /// blocks so the client can interleave them faithfully.
    AssistantReasoning { text: String },
    ToolCallBegin {
        tool_use_id: String,
        name: String,
        args_preview: String,
    },
    ToolCallEnd {
        tool_use_id: String,
        result_preview: String,
        is_error: bool,
    },
    PendingApproval {
        approval_id: String,
        tool_use_id: String,
        name: String,
        args_preview: String,
        destructive: bool,
        read_only: bool,
    },
    ApprovalResolved {
        approval_id: String,
        decision: ApprovalChoice,
        decided_by_conn: Option<u64>,
    },
    AssistantEnd {
        stop_reason: Option<String>,
        usage: Usage,
    },
    LoopComplete,
    StateChanged,
    Error { message: String },
    /// The task's `tool_allowlist` set was modified — either a tool was added
    /// (via an approve-and-remember decision) or removed (via the revoke UI).
    /// Carries the full new set so the scheduler can broadcast it without
    /// recomputing.
    AllowlistChanged { allowlist: Vec<String> },
    /// Tool call completed. Carried separately from the user-visible ToolCallEnd event
    /// so the scheduler can write an audit entry outside the task state machine.
    AuditToolCall {
        tool_name: String,
        args: serde_json::Value,
        is_error: bool,
        error_message: Option<String>,
        decision: String,
        who_decided: String,
    },
}

impl Task {
    pub fn new(id: String, config: TaskConfig) -> Self {
        let now = Utc::now();
        Self {
            id,
            created_at: now,
            last_active: now,
            title: None,
            config,
            conversation: Conversation::new(),
            total_usage: Usage::default(),
            archived: false,
            turns_in_cycle: 0,
            tool_allowlist: BTreeSet::new(),
            internal: TaskInternalState::Idle,
        }
    }

    pub fn touch(&mut self) {
        self.last_active = Utc::now();
    }

    pub fn public_state(&self) -> TaskStateLabel {
        match &self.internal {
            TaskInternalState::Idle => TaskStateLabel::Idle,
            TaskInternalState::Completed => TaskStateLabel::Completed,
            TaskInternalState::AwaitingApproval { .. } => TaskStateLabel::AwaitingApproval,
            TaskInternalState::NeedsMcpConnect
            | TaskInternalState::AwaitingMcpConnect { .. }
            | TaskInternalState::NeedsListTools
            | TaskInternalState::AwaitingListTools { .. }
            | TaskInternalState::NeedsModelCall
            | TaskInternalState::AwaitingModel { .. }
            | TaskInternalState::AwaitingTools { .. } => TaskStateLabel::Working,
            TaskInternalState::Failed { .. } => TaskStateLabel::Failed,
            TaskInternalState::Cancelled => TaskStateLabel::Cancelled,
        }
    }

    pub fn summary(&self) -> TaskSummary {
        TaskSummary {
            task_id: self.id.clone(),
            title: self.title.clone(),
            state: self.public_state(),
            created_at: self.created_at.to_rfc3339(),
            last_active: self.last_active.to_rfc3339(),
        }
    }

    pub fn snapshot(&self) -> TaskSnapshot {
        TaskSnapshot {
            task_id: self.id.clone(),
            title: self.title.clone(),
            config: self.config.clone(),
            state: self.public_state(),
            conversation: self.conversation.clone(),
            total_usage: self.total_usage,
            created_at: self.created_at.to_rfc3339(),
            last_active: self.last_active.to_rfc3339(),
            failure: self.failure_detail(),
            tool_allowlist: self.tool_allowlist.iter().cloned().collect(),
        }
    }

    /// If the task is in the Failed state, return a human-readable description of
    /// why (combining phase + message). Returns None otherwise.
    pub fn failure_detail(&self) -> Option<String> {
        match &self.internal {
            TaskInternalState::Failed { at_phase, message } => {
                Some(format!("{at_phase}: {message}"))
            }
            _ => None,
        }
    }

    /// Apply a user-submitted message. Appends to the conversation and starts the
    /// loop (or re-starts after a completed/failed cycle).
    pub fn submit_user_message(&mut self, text: String, mcp_connected: bool, tools_listed: bool) {
        self.conversation.push(Message::user_text(text));
        self.turns_in_cycle = 0;
        self.internal = if !mcp_connected {
            TaskInternalState::NeedsMcpConnect
        } else if !tools_listed {
            TaskInternalState::NeedsListTools
        } else {
            TaskInternalState::NeedsModelCall
        };
        self.touch();
    }

    pub fn cancel(&mut self) {
        self.internal = TaskInternalState::Cancelled;
        self.touch();
    }

    pub fn fail(&mut self, phase: impl Into<String>, message: impl Into<String>) {
        self.internal = TaskInternalState::Failed {
            at_phase: phase.into(),
            message: message.into(),
        };
        self.touch();
    }

    /// Drop a tool name from the allowlist. Returns true if the entry was
    /// present (so the caller knows whether to broadcast an update).
    pub fn remove_from_allowlist(&mut self, tool_name: &str) -> bool {
        let removed = self.tool_allowlist.remove(tool_name);
        if removed {
            self.touch();
        }
        removed
    }

    pub fn allowlist_snapshot(&self) -> Vec<String> {
        self.tool_allowlist.iter().cloned().collect()
    }

    /// Is the task accepting new user input right now?
    pub fn is_idle(&self) -> bool {
        matches!(
            self.internal,
            TaskInternalState::Idle
                | TaskInternalState::Completed
                | TaskInternalState::Failed { .. }
                | TaskInternalState::Cancelled
        )
    }

    /// Advance the state machine. Caller provides `next_op_id` for fresh I/O ops and
    /// `events` as the out-param for task events emitted during this step.
    pub fn step(
        &mut self,
        next_op_id: &mut OpId,
        events: &mut Vec<TaskEvent>,
    ) -> StepOutcome {
        let prev_public = self.public_state();
        let outcome = self.step_inner(next_op_id, events);
        let new_public = self.public_state();
        if prev_public != new_public {
            events.push(TaskEvent::StateChanged);
        }
        outcome
    }

    fn step_inner(
        &mut self,
        next_op_id: &mut OpId,
        events: &mut Vec<TaskEvent>,
    ) -> StepOutcome {
        // Break the state out so we can replace it.
        let current = std::mem::replace(&mut self.internal, TaskInternalState::Idle);
        match current {
            TaskInternalState::Idle
            | TaskInternalState::Completed
            | TaskInternalState::Cancelled
            | TaskInternalState::Failed { .. } => {
                // Restore and pause — caller shouldn't be stepping terminal states.
                self.internal = current;
                StepOutcome::Paused
            }
            TaskInternalState::NeedsMcpConnect => {
                let op_id = next_id(next_op_id);
                self.internal = TaskInternalState::AwaitingMcpConnect { op_id };
                self.touch();
                StepOutcome::DispatchIo(IoRequest::McpConnect { op_id })
            }
            TaskInternalState::NeedsListTools => {
                let op_id = next_id(next_op_id);
                self.internal = TaskInternalState::AwaitingListTools { op_id };
                self.touch();
                StepOutcome::DispatchIo(IoRequest::ListTools { op_id })
            }
            TaskInternalState::NeedsModelCall => {
                if self.turns_in_cycle >= self.config.max_turns {
                    tracing::warn!(max_turns = self.config.max_turns, task_id = %self.id, "max_turns reached");
                    events.push(TaskEvent::LoopComplete);
                    self.internal = TaskInternalState::Completed;
                    self.touch();
                    return StepOutcome::Continue;
                }
                self.turns_in_cycle += 1;
                events.push(TaskEvent::AssistantBegin { turn: self.turns_in_cycle });
                let op_id = next_id(next_op_id);
                self.internal = TaskInternalState::AwaitingModel {
                    op_id,
                    started_at: Utc::now(),
                };
                self.touch();
                StepOutcome::DispatchIo(IoRequest::ModelCall { op_id })
            }
            TaskInternalState::AwaitingModel { .. }
            | TaskInternalState::AwaitingMcpConnect { .. }
            | TaskInternalState::AwaitingListTools { .. }
            | TaskInternalState::AwaitingApproval { .. } => {
                // Not stepping these — waiting on I/O or user decision.
                self.internal = current;
                StepOutcome::Paused
            }
            TaskInternalState::AwaitingTools {
                mut pending_dispatch,
                mut pending_io,
                completed,
                approvals,
            } => {
                if let Some(next) = pending_dispatch.pop() {
                    let op_id = next_id(next_op_id);
                    pending_io.insert(op_id, next.tool_use_id.clone());
                    events.push(TaskEvent::ToolCallBegin {
                        tool_use_id: next.tool_use_id.clone(),
                        name: next.name.clone(),
                        args_preview: truncate(
                            serde_json::to_string(&next.input).unwrap_or_default(),
                            200,
                        ),
                    });
                    let dispatch = IoRequest::ToolCall {
                        op_id,
                        tool_use_id: next.tool_use_id.clone(),
                        name: next.name.clone(),
                        input: next.input.clone(),
                    };
                    self.internal = TaskInternalState::AwaitingTools {
                        pending_dispatch,
                        pending_io,
                        completed,
                        approvals,
                    };
                    self.touch();
                    StepOutcome::DispatchIo(dispatch)
                } else if pending_io.is_empty() {
                    // All tool calls done — append ToolResult blocks and loop back.
                    self.conversation.push(Message::user_blocks(completed));
                    self.internal = TaskInternalState::NeedsModelCall;
                    self.touch();
                    StepOutcome::Continue
                } else {
                    self.internal = TaskInternalState::AwaitingTools {
                        pending_dispatch,
                        pending_io,
                        completed,
                        approvals,
                    };
                    StepOutcome::Paused
                }
            }
        }
    }

    /// Apply an I/O completion. Pushes events describing the integration; the scheduler
    /// should call `step_until_blocked` afterward.
    ///
    /// `tool_annotations` is consulted when a model response arrives so the approval
    /// policy can be evaluated against each tool's hint flags.
    pub fn apply_io_result(
        &mut self,
        op_id: OpId,
        result: IoResult,
        tool_annotations: &HashMap<String, ToolAnnotations>,
        events: &mut Vec<TaskEvent>,
    ) {
        let prev_public = self.public_state();
        self.apply_io_result_inner(op_id, result, tool_annotations, events);
        let new_public = self.public_state();
        if prev_public != new_public {
            events.push(TaskEvent::StateChanged);
        }
    }

    fn apply_io_result_inner(
        &mut self,
        op_id: OpId,
        result: IoResult,
        tool_annotations: &HashMap<String, ToolAnnotations>,
        events: &mut Vec<TaskEvent>,
    ) {
        self.touch();
        // Cancelled task: drop the result on the floor.
        if matches!(self.internal, TaskInternalState::Cancelled) {
            return;
        }

        match (&self.internal, result) {
            (TaskInternalState::AwaitingMcpConnect { op_id: expected }, IoResult::McpConnect(res))
                if *expected == op_id =>
            {
                match res {
                    Ok(()) => {
                        self.internal = TaskInternalState::NeedsListTools;
                    }
                    Err(msg) => {
                        events.push(TaskEvent::Error { message: format!("mcp connect failed: {msg}") });
                        self.fail("mcp_connect", msg);
                    }
                }
            }
            (TaskInternalState::AwaitingListTools { op_id: expected }, IoResult::ListTools(res))
                if *expected == op_id =>
            {
                match res {
                    Ok(()) => {
                        self.internal = TaskInternalState::NeedsModelCall;
                    }
                    Err(msg) => {
                        events.push(TaskEvent::Error { message: format!("list_tools failed: {msg}") });
                        self.fail("list_tools", msg);
                    }
                }
            }
            (TaskInternalState::AwaitingModel { op_id: expected, .. }, IoResult::ModelCall(res))
                if *expected == op_id =>
            {
                match res {
                    Ok(response) => self.integrate_model_response(response, tool_annotations, events),
                    Err(msg) => {
                        events.push(TaskEvent::Error { message: format!("model call failed: {msg}") });
                        self.fail("model_call", msg);
                    }
                }
            }
            (
                TaskInternalState::AwaitingTools { .. },
                IoResult::ToolCall { tool_use_id, result },
            ) => self.integrate_tool_result(op_id, tool_use_id, result, events),
            (state, result) => {
                tracing::warn!(
                    task_id = %self.id,
                    op_id,
                    state = ?std::mem::discriminant(state),
                    result = ?std::mem::discriminant(&result),
                    "io result does not match current state — discarding"
                );
            }
        }
    }

    fn integrate_model_response(
        &mut self,
        response: ModelResponse,
        tool_annotations: &HashMap<String, ToolAnnotations>,
        events: &mut Vec<TaskEvent>,
    ) {
        let ModelResponse {
            content: assistant_blocks,
            stop_reason,
            usage,
        } = response;
        self.total_usage.add(&usage);
        for block in &assistant_blocks {
            match block {
                ContentBlock::Text { text } => {
                    events.push(TaskEvent::AssistantText { text: text.clone() });
                }
                ContentBlock::Thinking { thinking, .. } => {
                    events.push(TaskEvent::AssistantReasoning { text: thinking.clone() });
                }
                _ => {}
            }
        }
        let tool_uses: Vec<ToolUseReq> = assistant_blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolUse { id, name, input } => Some(ToolUseReq {
                    tool_use_id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                }),
                _ => None,
            })
            .collect();
        events.push(TaskEvent::AssistantEnd {
            stop_reason,
            usage,
        });
        self.conversation.push(Message::assistant_blocks(assistant_blocks));

        if tool_uses.is_empty() {
            events.push(TaskEvent::LoopComplete);
            self.internal = TaskInternalState::Completed;
            return;
        }

        // Evaluate approval policy for each tool_use.
        let mut dispositions: Vec<ApprovalDisposition> = Vec::with_capacity(tool_uses.len());
        let mut auto_reasons: HashMap<String, String> = HashMap::new();
        let mut any_pending = false;
        for tool_use in &tool_uses {
            let empty = ToolAnnotations::default();
            let ann = tool_annotations.get(&tool_use.name).unwrap_or(&empty);
            let allowlisted = self.tool_allowlist.contains(&tool_use.name);
            match evaluate_policy(self.config.approval_policy, ann, allowlisted) {
                ApprovalOutcome::Auto(reason) => {
                    auto_reasons.insert(tool_use.tool_use_id.clone(), reason.clone());
                    dispositions.push(ApprovalDisposition::AutoApproved { reason });
                }
                ApprovalOutcome::Prompt => {
                    any_pending = true;
                    let approval_id = format!("ap-{}", tool_use.tool_use_id);
                    events.push(TaskEvent::PendingApproval {
                        approval_id: approval_id.clone(),
                        tool_use_id: tool_use.tool_use_id.clone(),
                        name: tool_use.name.clone(),
                        args_preview: truncate(
                            serde_json::to_string(&tool_use.input).unwrap_or_default(),
                            200,
                        ),
                        destructive: ann.is_destructive(),
                        read_only: ann.is_read_only(),
                    });
                    dispositions.push(ApprovalDisposition::Pending {
                        approval_id,
                        destructive: ann.is_destructive(),
                        read_only: ann.is_read_only(),
                    });
                }
            }
        }

        if any_pending {
            self.internal = TaskInternalState::AwaitingApproval {
                tool_uses,
                dispositions,
            };
        } else {
            // Every tool auto-approved — record why so the audit log reflects it.
            let approvals: HashMap<String, ApprovalRecord> = tool_uses
                .iter()
                .map(|tu| {
                    let reason = auto_reasons
                        .remove(&tu.tool_use_id)
                        .unwrap_or_else(|| "policy:auto".into());
                    (
                        tu.tool_use_id.clone(),
                        ApprovalRecord {
                            decision: "auto".into(),
                            who_decided: reason,
                        },
                    )
                })
                .collect();
            self.internal = TaskInternalState::AwaitingTools {
                pending_dispatch: make_dispatch_order(tool_uses),
                pending_io: HashMap::new(),
                completed: Vec::new(),
                approvals,
            };
        }
    }

    fn integrate_tool_result(
        &mut self,
        op_id: OpId,
        tool_use_id: String,
        result: Result<CallToolResult, String>,
        events: &mut Vec<TaskEvent>,
    ) {
        let TaskInternalState::AwaitingTools {
            pending_dispatch,
            mut pending_io,
            mut completed,
            mut approvals,
        } = std::mem::replace(&mut self.internal, TaskInternalState::Idle)
        else {
            unreachable!("matched guard ensures we're in AwaitingTools")
        };

        // The op_id should be tracked. Use it to verify; tool_use_id is the canonical
        // key for the conversation block either way.
        pending_io.remove(&op_id);

        let approval = approvals.remove(&tool_use_id).unwrap_or_else(|| ApprovalRecord {
            decision: "dispatched".into(),
            who_decided: "policy".into(),
        });

        let (text, is_error, tool_name, args, err_msg) = match result {
            Ok(r) => {
                let text = join_mcp_text(&r.content);
                let tool_name = find_tool_name(&self.conversation, &tool_use_id);
                let args = find_tool_args(&self.conversation, &tool_use_id);
                (text, r.is_error, tool_name, args, None)
            }
            Err(msg) => {
                let tool_name = find_tool_name(&self.conversation, &tool_use_id);
                let args = find_tool_args(&self.conversation, &tool_use_id);
                (
                    format!("tool invocation failed: {msg}"),
                    true,
                    tool_name,
                    args,
                    Some(msg),
                )
            }
        };

        events.push(TaskEvent::ToolCallEnd {
            tool_use_id: tool_use_id.clone(),
            result_preview: truncate(text.clone(), 200),
            is_error,
        });
        events.push(TaskEvent::AuditToolCall {
            tool_name,
            args,
            is_error,
            error_message: err_msg,
            decision: approval.decision,
            who_decided: approval.who_decided,
        });

        completed.push(ContentBlock::ToolResult {
            tool_use_id,
            content: ToolResultContent::Text(text),
            is_error,
        });

        self.internal = TaskInternalState::AwaitingTools {
            pending_dispatch,
            pending_io,
            completed,
            approvals,
        };
    }

    /// Apply a user's approval decision. Finds the pending entry by `approval_id`,
    /// transitions it to `UserApproved`/`UserRejected`, and if every disposition is
    /// now resolved, moves the task into `AwaitingTools` (with synthesized denial
    /// tool_results for rejected calls).
    ///
    /// When `remember` is true and the decision is `Approve`, the resolved tool's
    /// name is added to the per-task allowlist. Any other still-pending approvals
    /// for the same tool name in the current batch are auto-resolved as approved
    /// (with reason `policy:allowlist`) and emit their own `ApprovalResolved`
    /// events so connected clients dismiss them.
    pub fn apply_approval_decision(
        &mut self,
        approval_id: &str,
        decision: ApprovalChoice,
        remember: bool,
        decided_by_conn: Option<u64>,
        events: &mut Vec<TaskEvent>,
    ) {
        let prev_public = self.public_state();
        self.apply_approval_decision_inner(
            approval_id,
            decision,
            remember,
            decided_by_conn,
            events,
        );
        let new_public = self.public_state();
        if prev_public != new_public {
            events.push(TaskEvent::StateChanged);
        }
    }

    fn apply_approval_decision_inner(
        &mut self,
        approval_id: &str,
        decision: ApprovalChoice,
        remember: bool,
        decided_by_conn: Option<u64>,
        events: &mut Vec<TaskEvent>,
    ) {
        self.touch();
        let TaskInternalState::AwaitingApproval { tool_uses, dispositions } = &mut self.internal
        else {
            tracing::warn!(
                task_id = %self.id,
                approval_id,
                "approval decision received but task not awaiting approval"
            );
            return;
        };

        let Some(idx) = dispositions.iter().position(|d| {
            matches!(d, ApprovalDisposition::Pending { approval_id: aid, .. } if aid == approval_id)
        }) else {
            tracing::warn!(
                task_id = %self.id,
                approval_id,
                "approval_id not pending — duplicate decision?"
            );
            return;
        };

        // If the user said "always allow", grow the allowlist before resolving the
        // cascade so the matching dispositions transition through the allowlist
        // path rather than a one-off user-approval.
        let resolved_tool_name = tool_uses[idx].name.clone();
        let mut allowlist_added = false;
        if remember && decision == ApprovalChoice::Approve {
            allowlist_added = self.tool_allowlist.insert(resolved_tool_name.clone());
        }

        dispositions[idx] = match decision {
            ApprovalChoice::Approve => ApprovalDisposition::UserApproved {
                approval_id: approval_id.to_string(),
                decided_by_conn,
            },
            ApprovalChoice::Reject => ApprovalDisposition::UserRejected {
                approval_id: approval_id.to_string(),
                decided_by_conn,
                message: "Denied by user.".to_string(),
            },
        };
        events.push(TaskEvent::ApprovalResolved {
            approval_id: approval_id.to_string(),
            decision,
            decided_by_conn,
        });

        // Cascade-resolve other pending approvals for the same tool now that it's
        // been allowlisted.
        if allowlist_added {
            for (i, tool_use) in tool_uses.iter().enumerate() {
                if tool_use.name != resolved_tool_name {
                    continue;
                }
                let ApprovalDisposition::Pending { approval_id: aid, .. } = &dispositions[i] else {
                    continue;
                };
                let aid = aid.clone();
                dispositions[i] = ApprovalDisposition::AutoApproved {
                    reason: "policy:allowlist".into(),
                };
                events.push(TaskEvent::ApprovalResolved {
                    approval_id: aid,
                    decision: ApprovalChoice::Approve,
                    decided_by_conn,
                });
            }
            events.push(TaskEvent::AllowlistChanged {
                allowlist: self.tool_allowlist.iter().cloned().collect(),
            });
        }

        if dispositions.iter().any(|d| d.is_pending()) {
            return; // more approvals outstanding
        }

        // Every disposition resolved — transition to AwaitingTools. Approved calls go
        // into pending_dispatch; rejected calls are synthesized as denial tool_results
        // in `completed` and also emit audit entries recording the rejection.
        let tool_uses = std::mem::take(tool_uses);
        let dispositions = std::mem::take(dispositions);
        let mut pending_dispatch = Vec::new();
        let mut completed = Vec::new();
        let mut approvals: HashMap<String, ApprovalRecord> = HashMap::new();
        for (tool_use, disposition) in tool_uses.into_iter().zip(dispositions) {
            match disposition {
                ApprovalDisposition::AutoApproved { reason } => {
                    approvals.insert(
                        tool_use.tool_use_id.clone(),
                        ApprovalRecord {
                            decision: "auto".into(),
                            who_decided: reason,
                        },
                    );
                    pending_dispatch.push(tool_use);
                }
                ApprovalDisposition::UserApproved { decided_by_conn: conn, .. } => {
                    approvals.insert(
                        tool_use.tool_use_id.clone(),
                        ApprovalRecord {
                            decision: "approve".into(),
                            who_decided: who_string(conn),
                        },
                    );
                    pending_dispatch.push(tool_use);
                }
                ApprovalDisposition::UserRejected { decided_by_conn: conn, message, .. } => {
                    // Emit synthetic ToolCallBegin/End so subscribed clients see the
                    // rejected call in their chat view alongside approved/executed ones.
                    events.push(TaskEvent::ToolCallBegin {
                        tool_use_id: tool_use.tool_use_id.clone(),
                        name: tool_use.name.clone(),
                        args_preview: truncate(
                            serde_json::to_string(&tool_use.input).unwrap_or_default(),
                            200,
                        ),
                    });
                    events.push(TaskEvent::ToolCallEnd {
                        tool_use_id: tool_use.tool_use_id.clone(),
                        result_preview: message.clone(),
                        is_error: true,
                    });
                    events.push(TaskEvent::AuditToolCall {
                        tool_name: tool_use.name.clone(),
                        args: tool_use.input.clone(),
                        is_error: true,
                        error_message: Some(message.clone()),
                        decision: "reject".into(),
                        who_decided: who_string(conn),
                    });
                    completed.push(ContentBlock::ToolResult {
                        tool_use_id: tool_use.tool_use_id,
                        content: ToolResultContent::Text(message),
                        is_error: true,
                    });
                }
                ApprovalDisposition::Pending { .. } => unreachable!("checked above"),
            }
        }
        self.internal = TaskInternalState::AwaitingTools {
            pending_dispatch: reverse_for_pop(pending_dispatch),
            pending_io: HashMap::new(),
            completed,
            approvals,
        };
    }
}

fn who_string(conn: Option<u64>) -> String {
    conn.map(|c| format!("user:{c}"))
        .unwrap_or_else(|| "user".into())
}

enum ApprovalOutcome {
    /// Auto-approved; the carried string becomes `who_decided` in the audit log.
    Auto(String),
    Prompt,
}

fn evaluate_policy(
    policy: ApprovalPolicy,
    annotations: &ToolAnnotations,
    allowlisted: bool,
) -> ApprovalOutcome {
    if allowlisted {
        return ApprovalOutcome::Auto("policy:allowlist".into());
    }
    match policy {
        ApprovalPolicy::AutoApproveAll => {
            ApprovalOutcome::Auto("policy:auto_approve_all".into())
        }
        ApprovalPolicy::PromptDestructive => {
            if annotations.is_read_only() {
                ApprovalOutcome::Auto("policy:read_only".into())
            } else {
                ApprovalOutcome::Prompt
            }
        }
    }
}

/// `step()` pops from the back; feed it in reverse so the original order is preserved.
fn make_dispatch_order(tool_uses: Vec<ToolUseReq>) -> Vec<ToolUseReq> {
    reverse_for_pop(tool_uses)
}

fn reverse_for_pop<T>(mut v: Vec<T>) -> Vec<T> {
    v.reverse();
    v
}

/// Derive a title from the user's initial message: trim, collapse internal whitespace,
/// truncate to ~50 chars (rounded to a char boundary) with a trailing ellipsis.
pub fn derive_title(initial_message: &str) -> String {
    let collapsed: String = initial_message.split_whitespace().collect::<Vec<_>>().join(" ");
    const MAX: usize = 50;
    if collapsed.chars().count() <= MAX {
        collapsed
    } else {
        let mut out: String = collapsed.chars().take(MAX).collect();
        out.push('…');
        out
    }
}

pub fn new_task_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("task-{:016x}", nanos as u64)
}

fn next_id(counter: &mut OpId) -> OpId {
    *counter += 1;
    *counter
}

fn join_mcp_text(blocks: &[crate::mcp::McpContentBlock]) -> String {
    let mut out = String::new();
    for b in blocks {
        match b {
            crate::mcp::McpContentBlock::Text { text } => out.push_str(text),
        }
    }
    out
}

fn find_tool_name(conv: &Conversation, tool_use_id: &str) -> String {
    for msg in conv.messages().iter().rev() {
        for block in &msg.content {
            if let ContentBlock::ToolUse { id, name, .. } = block {
                if id == tool_use_id {
                    return name.clone();
                }
            }
        }
    }
    String::new()
}

fn find_tool_args(conv: &Conversation, tool_use_id: &str) -> serde_json::Value {
    for msg in conv.messages().iter().rev() {
        for block in &msg.content {
            if let ContentBlock::ToolUse { id, input, .. } = block {
                if id == tool_use_id {
                    return input.clone();
                }
            }
        }
    }
    serde_json::Value::Null
}

fn truncate(mut s: String, max: usize) -> String {
    if s.len() > max {
        // `max` is a byte count but may land inside a multi-byte UTF-8 sequence;
        // walk back to the nearest char boundary before cutting.
        let mut cut = max;
        while cut > 0 && !s.is_char_boundary(cut) {
            cut -= 1;
        }
        s.truncate(cut);
        s.push('…');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use whisper_agent_protocol::ApprovalPolicy;

    #[test]
    fn truncate_handles_multibyte_boundary() {
        // `…` is 3 bytes (E2 80 A6). Position the char so `max` lands inside it —
        // the naive `String::truncate(max)` panics here.
        let s = format!("{}…tail", "a".repeat(198));
        let out = truncate(s, 200);
        assert!(out.ends_with('…'));
        // The cut falls back to byte 198 (end of the "a" run, on a boundary).
        assert_eq!(out, format!("{}…", "a".repeat(198)));
    }

    #[test]
    fn truncate_no_op_when_short() {
        assert_eq!(truncate("héllo".into(), 200), "héllo");
    }

    #[test]
    fn truncate_cuts_on_ascii_boundary() {
        assert_eq!(truncate("abcdefghij".into(), 4), "abcd…");
    }

    fn task_with_pending_batch(tool_names: &[&str]) -> Task {
        let cfg = TaskConfig {
            backend: String::new(),
            model: "test".into(),
            system_prompt: "test".into(),
            mcp_host_url: "http://test".into(),
            max_tokens: 100,
            max_turns: 10,
            approval_policy: ApprovalPolicy::PromptDestructive,
            sandbox: Default::default(),
            shared_mcp_hosts: Vec::new(),
        };
        let mut task = Task::new("t1".into(), cfg);
        let tool_uses: Vec<ToolUseReq> = tool_names
            .iter()
            .enumerate()
            .map(|(i, name)| ToolUseReq {
                tool_use_id: format!("tu-{i}"),
                name: (*name).into(),
                input: serde_json::Value::Null,
            })
            .collect();
        let dispositions = tool_uses
            .iter()
            .map(|tu| ApprovalDisposition::Pending {
                approval_id: format!("ap-{}", tu.tool_use_id),
                destructive: false,
                read_only: false,
            })
            .collect();
        task.internal = TaskInternalState::AwaitingApproval { tool_uses, dispositions };
        task
    }

    #[test]
    fn always_allow_adds_to_allowlist_and_emits_event() {
        let mut task = task_with_pending_batch(&["edit_file"]);
        let mut events = Vec::new();
        task.apply_approval_decision("ap-tu-0", ApprovalChoice::Approve, true, Some(7), &mut events);

        assert!(task.tool_allowlist.contains("edit_file"));
        assert!(events.iter().any(|e| matches!(e,
            TaskEvent::AllowlistChanged { allowlist } if allowlist == &vec!["edit_file".to_string()]
        )));
    }

    #[test]
    fn always_allow_cascades_other_pending_for_same_tool() {
        let mut task = task_with_pending_batch(&["edit_file", "edit_file", "read_file"]);
        let mut events = Vec::new();
        task.apply_approval_decision("ap-tu-0", ApprovalChoice::Approve, true, Some(1), &mut events);

        // edit_file batch (3 pending: 1 directly resolved + 1 cascaded; read_file still pending)
        let resolved: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                TaskEvent::ApprovalResolved { approval_id, .. } => Some(approval_id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(resolved, vec!["ap-tu-0", "ap-tu-1"]);

        // read_file should still be pending — task should not have transitioned out
        // of AwaitingApproval.
        assert!(matches!(task.internal, TaskInternalState::AwaitingApproval { .. }));
    }

    #[test]
    fn approve_without_remember_does_not_touch_allowlist() {
        let mut task = task_with_pending_batch(&["edit_file"]);
        let mut events = Vec::new();
        task.apply_approval_decision("ap-tu-0", ApprovalChoice::Approve, false, None, &mut events);

        assert!(task.tool_allowlist.is_empty());
        assert!(!events.iter().any(|e| matches!(e, TaskEvent::AllowlistChanged { .. })));
    }

    #[test]
    fn allowlist_short_circuits_evaluate_policy() {
        let read_only_ann = ToolAnnotations::default();
        let allowlisted = matches!(
            evaluate_policy(ApprovalPolicy::PromptDestructive, &read_only_ann, true),
            ApprovalOutcome::Auto(_)
        );
        let not_listed = matches!(
            evaluate_policy(ApprovalPolicy::PromptDestructive, &read_only_ann, false),
            ApprovalOutcome::Prompt
        );
        assert!(allowlisted, "allowlisted call must auto-resolve");
        assert!(not_listed, "non-allowlisted unannotated call must prompt");
    }

    #[test]
    fn remove_from_allowlist_returns_true_only_when_present() {
        let mut task = task_with_pending_batch(&["edit_file"]);
        task.tool_allowlist.insert("edit_file".into());
        assert!(task.remove_from_allowlist("edit_file"));
        assert!(!task.remove_from_allowlist("edit_file"));
        assert!(!task.remove_from_allowlist("never_added"));
    }
}
