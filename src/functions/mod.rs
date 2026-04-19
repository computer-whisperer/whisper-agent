//! Function model: server-owned in-flight operations, caller-agnostic.
//!
//! See `docs/design_functions.md` for the full design. In brief:
//!
//! - Every caller-initiated operation (create thread, compact, invoke tool,
//!   run behavior, etc.) is a variant of the `Function` enum.
//! - The scheduler owns an `active_functions` registry of `ActiveFunction`s,
//!   each with a `PermissionScope` (what's admissible), a `CallerLink` (who
//!   invoked and where the results go), and two streams (progress, terminal).
//! - `Function` implementations are opaque to the caller surface. Caller
//!   surfaces (WS client, model tool call, lua hook, scheduler-internal)
//!   differ only in `CallerLink` variant.
//!
//! **Status: Phase 1a — types only.** Declared here but not yet consumed by
//! the scheduler. Registry, routing, and operation migration follow in later
//! phases.

use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use whisper_agent_protocol::ThreadBindingsPatch;

use crate::permission::{HostName, PodId, ToolName};

// ---------------------------------------------------------------------------
// Identifiers and bookkeeping
// ---------------------------------------------------------------------------

/// Server-assigned monotonic id for an in-flight Function. Not persisted;
/// not stable across server restarts. Same shape as `ConnId`.
pub type FunctionId = u64;

/// Wire thread identifier (string) — matches the existing `Thread.id`
/// convention. Aliased here so Function specs don't need to import through
/// the pod module.
pub type ThreadId = String;

/// Model-supplied identifier for a tool invocation within a thread. Used
/// as the terminal-routing key for `ThreadToolCall` caller-links.
pub type ToolUseId = String;

pub type BehaviorId = String;

/// Correlation handle supplied by WS clients so they can match responses
/// to their original requests. `None` for fire-and-forget. Matches the
/// existing wire-protocol type in `whisper-agent-protocol`.
pub type CorrelationId = Option<String>;

pub type ConnId = u64;

/// Opaque delivery handle into a lua runtime — the runtime hands one to the
/// scheduler at hook invocation time, and the scheduler posts results to it
/// without knowing anything about coroutines or VMs.
pub type LuaChannelId = u64;

pub type HookId = String;

// ---------------------------------------------------------------------------
// Function specs
// ---------------------------------------------------------------------------

/// A caller-initiated operation. Each variant is the *spec* for that
/// operation — the arguments needed to invoke it.
///
/// Closed enum by design: the scheduler match-and-mutates without dynamic
/// dispatch, and every caller-visible operation is visible in one place.
/// MCP tools are the one open set, collapsed to `McpToolUse`.
#[derive(Debug, Clone, PartialEq)]
pub enum Function {
    CreateThread {
        pod_id: PodId,
        initial_message: Option<String>,
        parent: Option<ParentLink>,
        wait_mode: WaitMode,
        // Thread config / bindings carried as opaque serde_json::Value for now;
        // the typed shapes live in the protocol crate and will plug in when
        // CreateThread gets migrated (Phase 2).
        config_override: Option<serde_json::Value>,
        bindings_request: Option<serde_json::Value>,
    },
    CompactThread {
        thread_id: ThreadId,
    },
    RebindThread {
        thread_id: ThreadId,
        patch: ThreadBindingsPatch,
    },
    CancelThread {
        thread_id: ThreadId,
    },
    RunBehavior {
        pod_id: PodId,
        behavior_id: BehaviorId,
        payload: serde_json::Value,
    },
    BuiltinToolCall {
        name: ToolName,
        args: serde_json::Value,
    },
    McpToolUse {
        host: HostName,
        name: ToolName,
        args: serde_json::Value,
    },
}

impl Function {
    /// Thread id this Function primarily targets, if any. Used for
    /// error-event routing (the wire `Error` shape carries an optional
    /// `thread_id` so clients can scope failures). Returns `None` for
    /// variants whose work isn't bound to a single thread.
    pub fn primary_thread_id(&self) -> Option<&str> {
        match self {
            Self::CompactThread { thread_id }
            | Self::RebindThread { thread_id, .. }
            | Self::CancelThread { thread_id } => Some(thread_id),
            Self::CreateThread { .. }
            | Self::RunBehavior { .. }
            | Self::BuiltinToolCall { .. }
            | Self::McpToolUse { .. } => None,
        }
    }
}

/// Sync vs async dispatch for `CreateThread`. Meaningful primarily when
/// `parent` is set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitMode {
    /// Terminate as soon as the thread exists and is seeded. Used by
    /// interactive create, behavior-spawned, and async dispatch.
    ThreadCreated,
    /// Terminate only when the created thread itself reaches terminal.
    /// Used by sync dispatch — parent parks on this Function's terminal.
    ThreadTerminal,
}

/// Parent-thread linkage for dispatched child threads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParentLink {
    pub thread_id: ThreadId,
    pub tool_use_id: ToolUseId,
}

// ---------------------------------------------------------------------------
// Caller-link: who invoked, where results go
// ---------------------------------------------------------------------------

/// Describes *who invoked this Function* and *where the results go*.
///
/// Functions are opaque to the variant — the scheduler's router reads the
/// link and delivers to the right surface. The variant names are for
/// identity / audit; delivery details are implicit adjuncts the scheduler
/// looks up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallerLink {
    WsClient {
        conn_id: ConnId,
        correlation_id: CorrelationId,
    },
    ThreadToolCall {
        thread_id: ThreadId,
        tool_use_id: ToolUseId,
    },
    Lua {
        hook_id: HookId,
        delivery: LuaChannelId,
    },
    SchedulerInternal(InternalOriginator),
}

/// Origins for scheduler-internally-invoked Functions. **Closed enum by
/// design** — new entries are a review step, not a drive-by addition. This
/// is the primary mechanism for preventing "just add a callback" escape
/// hatches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InternalOriginator {
    /// Trigger-driven `RunBehavior` fire (cron, webhook, queued replay,
    /// or tool-call cascade). `source` captures which path originated
    /// the fire for audit / logging.
    BehaviorFire {
        pod_id: PodId,
        behavior_id: BehaviorId,
        source: TriggerSource,
    },
    AutoCompact {
        thread_id: ThreadId,
    },
}

/// Which trigger surface initiated a behavior fire. Stored on
/// `InternalOriginator::BehaviorFire` for audit/logging; doesn't affect
/// execution semantics (overlap policy is already applied at the
/// caller).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerSource {
    Cron,
    Webhook,
    /// QueueOne-queued payload replayed on the previous run's terminal.
    QueuedReplay,
    /// Builtin-tool cascade (e.g. `pod_run_behavior`).
    ToolCall,
}

impl TriggerSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cron => "cron",
            Self::Webhook => "webhook",
            Self::QueuedReplay => "queued_replay",
            Self::ToolCall => "tool_call",
        }
    }
}

impl CallerLink {
    /// Does this caller-link target the given thread? Used by the
    /// cancel-by-thread sweep (fired when a thread reaches terminal) —
    /// the sweep scans `active_functions` for entries whose link targets
    /// the affected thread and applies per-variant cancel-on-caller-gone
    /// policy.
    ///
    /// This must peer into `SchedulerInternal` payloads too — an
    /// `AutoCompact { thread_id: T }` Function logically targets T even
    /// though its direct caller-link variant is `SchedulerInternal`.
    pub fn targets_thread(&self, id: &str) -> bool {
        match self {
            Self::ThreadToolCall { thread_id, .. } => thread_id == id,
            Self::SchedulerInternal(orig) => match orig {
                InternalOriginator::AutoCompact { thread_id } => thread_id == id,
                InternalOriginator::BehaviorFire { .. } => false,
            },
            Self::WsClient { .. } | Self::Lua { .. } => false,
        }
    }

    /// Stable, log-friendly identifier for audit records. Keeps the audit
    /// log format in one place regardless of caller-link variant.
    pub fn audit_tag(&self) -> String {
        match self {
            Self::WsClient { conn_id, .. } => format!("ws:{conn_id}"),
            Self::ThreadToolCall {
                thread_id,
                tool_use_id,
            } => format!("thread:{thread_id}/tool:{tool_use_id}"),
            Self::Lua { hook_id, .. } => format!("lua:{hook_id}"),
            Self::SchedulerInternal(o) => match o {
                InternalOriginator::BehaviorFire {
                    pod_id,
                    behavior_id,
                    source,
                } => format!("behavior_fire:{}:{pod_id}/{behavior_id}", source.as_str()),
                InternalOriginator::AutoCompact { thread_id } => {
                    format!("auto_compact:{thread_id}")
                }
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Outcomes: terminals (per-variant) + overall result (Success/Error/Cancelled)
// ---------------------------------------------------------------------------

/// Per-variant successful terminal payload. Callers match on this for the
/// result of their invocation. Orthogonal to `FunctionOutcome`'s
/// Error/Cancelled overlays.
#[derive(Debug, Clone, PartialEq)]
pub enum FunctionTerminal {
    CreateThread(CreateThreadTerminal),
    CompactThread(CompactThreadTerminal),
    RebindThread(RebindThreadTerminal),
    CancelThread,
    RunBehavior(RunBehaviorTerminal),
    BuiltinToolCall(ToolResult),
    McpToolUse(ToolResult),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateThreadTerminal {
    pub thread_id: ThreadId,
    /// Populated only for `wait_mode = ThreadTerminal` (sync dispatch).
    /// `None` for interactive / behavior / async-dispatch creates.
    pub final_result: Option<ThreadTerminalSummary>,
}

/// Carried as part of `CreateThreadTerminal` when the parent waits for the
/// child to finish. Opaque `serde_json::Value` for now — Phase 2 migration
/// of CreateThread will formalize this.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThreadTerminalSummary {
    pub state: String,
    pub final_text: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactThreadTerminal {
    pub continuation_thread_id: ThreadId,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RebindThreadTerminal {
    // Empty for now; migration may add resolved bindings summary.
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunBehaviorTerminal {
    pub thread_id: ThreadId,
}

/// Generic tool result for builtin and MCP tools. Carries content blocks
/// (the streaming progress payloads already merged) plus an is_error flag
/// matching the MCP convention.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolResult {
    pub content: Vec<ContentBlock>,
    pub is_error: bool,
}

/// Multimodal content carrier. Re-exports the conversational
/// `ContentBlock` from the protocol crate — streaming tool output
/// (`ProgressEvent::Content`), persisted tool results, and model output
/// all use the same type so no translation is needed at the
/// caller-link / wire boundary.
pub type ContentBlock = whisper_agent_protocol::ContentBlock;

/// Overall Function result. Orthogonal to per-variant `FunctionTerminal`.
#[derive(Debug, Clone, PartialEq)]
pub enum FunctionOutcome {
    Success(FunctionTerminal),
    Error(FunctionError),
    Cancelled(CancelReason),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionError {
    pub kind: FunctionErrorKind,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FunctionErrorKind {
    /// Resource the Function needed (backend, MCP host, host-env) couldn't
    /// be reached at execution time. Distinct from `ScopeDenied` because
    /// the scope admitted the invocation — the physical resource failed.
    ResourceUnavailable,
    /// Input args failed variant-specific runtime validation after
    /// admission. (Static invalidity should surface as
    /// `RejectReason::InvalidSpec` at registration.)
    BadInput,
    /// The Function's work itself failed (MCP tool returned an error, model
    /// call errored, etc.).
    Execution,
    /// Catch-all for kinds not yet enumerated. Start narrow; extend as
    /// concrete needs surface.
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelReason {
    /// User denied the approval prompt for an `AllowWithPrompt` Function.
    UserDenied,
    /// Explicit cancel from the caller (or scheduler on behalf of an
    /// originator that dropped — e.g., thread cancel cascading to its
    /// tool-call Functions).
    ExplicitCancel,
    /// Caller-link went away before the Function terminated (per-variant
    /// cancel-on-caller-gone policy fired).
    CallerGone,
    /// `AllowWithPrompt` Function registered but no delivery surface
    /// exists to show the prompt to a human. Auto-denied after a bounded
    /// wait.
    NoApprovalPath,
}

// ---------------------------------------------------------------------------
// Registration: synchronous accept/deny
// ---------------------------------------------------------------------------

/// Failure path for `register(...)`. Sync-only — registration itself never
/// awaits I/O; deferred errors arrive via the Function's terminal channel
/// instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RejectReason {
    /// Scope did not admit the operation. The scope-level disposition for
    /// this invocation was `Deny`.
    ScopeDenied { detail: String },
    /// Precondition failed — e.g., thread is already compacting, target
    /// thread doesn't exist, rebind patch incompatible with pod allow.
    PreconditionFailed { detail: String },
    /// Spec itself is malformed or references an unknown resource by name.
    InvalidSpec { detail: String },
    /// Resource already in use in a way incompatible with this invocation
    /// (e.g., behavior already running with overlap=Skip).
    ResourceBusy { detail: String },
}

// ---------------------------------------------------------------------------
// Progress events
// ---------------------------------------------------------------------------

/// Streaming-decorative progress events emitted during Function execution.
///
/// Terminals are typed per-variant via `FunctionTerminal`; progress uses
/// this shared envelope so UI / audit / forwarding layers don't have to
/// special-case per Function variant.
#[derive(Debug, Clone, PartialEq)]
pub enum ProgressEvent {
    /// Content blocks — text, image, audio, structured. Reuses the existing
    /// `ContentBlock` protocol type (interim alias to `serde_json::Value`)
    /// so multimodal payloads ride the CBOR wire natively.
    Content(ContentBlock),
    /// Status transitions — "starting", "waiting_approval", "retrying", etc.
    Status(StatusKind),
    /// Approval request — produced by tool Functions whose scope-check
    /// disposition was `AllowWithPrompt`. The resolution routes back into
    /// the Function.
    ApprovalRequest(ApprovalRequest),
    /// Escape hatch for variant-specific progress that isn't worth naming
    /// in the common enum. `kind` is a stable string tag.
    Custom {
        kind: String,
        payload: serde_json::Value,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StatusKind {
    Starting,
    WaitingApproval,
    Retrying,
    /// Free-form status for variant-specific transitions. Keeps the common
    /// named variants small.
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovalRequest {
    /// Human-readable summary of what's being approved.
    pub message: String,
    /// Opaque context — e.g., serialized tool spec + args. Rendering is a
    /// caller-link-router concern.
    pub detail: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Thread in-flight flags
// ---------------------------------------------------------------------------

bitflags! {
    /// Per-thread "is an exclusivity-requiring Function currently running
    /// targeting this thread?" flags. Functions that need exclusivity set
    /// their bit on start and clear it on terminal; preconditions check by
    /// reading the bit rather than scanning the `active_functions` registry.
    ///
    /// Cleared on scheduler restart as part of the thread's resume-flip
    /// handling — Functions are non-persistent, so a bit set at crash time
    /// represents a Function that no longer exists.
    #[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct InFlightOps: u32 {
        const COMPACTING = 1 << 0;
        const REBINDING  = 1 << 1;
        // New bits as operations are migrated that need exclusivity.
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caller_link_targets_thread_direct() {
        let l = CallerLink::ThreadToolCall {
            thread_id: "t1".into(),
            tool_use_id: "u1".into(),
        };
        assert!(l.targets_thread("t1"));
        assert!(!l.targets_thread("t2"));
    }

    #[test]
    fn caller_link_targets_thread_through_auto_compact() {
        let l = CallerLink::SchedulerInternal(InternalOriginator::AutoCompact {
            thread_id: "t1".into(),
        });
        assert!(l.targets_thread("t1"));
        assert!(!l.targets_thread("t2"));
    }

    #[test]
    fn caller_link_targets_thread_ignores_unrelated_variants() {
        let ws = CallerLink::WsClient {
            conn_id: 42,
            correlation_id: Some("req-1".into()),
        };
        assert!(!ws.targets_thread("t1"));

        let cron = CallerLink::SchedulerInternal(InternalOriginator::BehaviorFire {
            pod_id: "p".into(),
            behavior_id: "b".into(),
            source: TriggerSource::Cron,
        });
        assert!(!cron.targets_thread("t1"));
    }

    #[test]
    fn caller_link_audit_tag_stable_per_variant() {
        let ws = CallerLink::WsClient {
            conn_id: 5,
            correlation_id: Some("req-7".into()),
        };
        assert_eq!(ws.audit_tag(), "ws:5");

        let thread = CallerLink::ThreadToolCall {
            thread_id: "t".into(),
            tool_use_id: "u".into(),
        };
        assert_eq!(thread.audit_tag(), "thread:t/tool:u");

        let ac = CallerLink::SchedulerInternal(InternalOriginator::AutoCompact {
            thread_id: "t".into(),
        });
        assert_eq!(ac.audit_tag(), "auto_compact:t");
    }

    #[test]
    fn in_flight_ops_set_and_check() {
        let mut ops = InFlightOps::empty();
        ops.insert(InFlightOps::COMPACTING);
        assert!(ops.contains(InFlightOps::COMPACTING));
        assert!(!ops.contains(InFlightOps::REBINDING));
        ops.remove(InFlightOps::COMPACTING);
        assert!(ops.is_empty());
    }

    // Silence unused-variant warnings on types that are declared here for
    // Phase 2 consumption but not exercised in Phase 1a tests.
    #[test]
    fn types_construct_without_warnings() {
        let _ = RejectReason::ScopeDenied {
            detail: "x".into(),
        };
        let _ = FunctionOutcome::Cancelled(CancelReason::UserDenied);
        let _ = ProgressEvent::Status(StatusKind::Starting);
        let _ = Function::CancelThread {
            thread_id: "t".into(),
        };
        let _ = FunctionTerminal::CancelThread;
    }
}
