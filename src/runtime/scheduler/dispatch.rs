//! Render helper for the `<dispatched-thread-notification>` envelope
//! used when an async `dispatch_thread` child terminates.
//!
//! The dispatch path itself lives in
//! `src/runtime/scheduler/functions.rs::register_dispatch_thread_tool`:
//! `dispatch_thread` is a tool-pool alias for `Function::CreateThread`,
//! handled by the single tool-dispatch entry point. This module only
//! owns the envelope rendering + its status enum so the async-followup
//! delivery path in `complete_function` can reuse them.

#[derive(Debug, Clone, Copy)]
pub(super) enum DispatchStatus {
    Completed,
    Failed,
    Cancelled,
}

impl DispatchStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Render the `<dispatched-thread-notification>` envelope for the
/// async delivery path. Shape mirrors Claude Code's `<task-notification>`:
/// stamps (thread-id + original parent tool_use_id) for correlation,
/// a short summary, the child's final `<result>` inline (XML-escaped),
/// and a `<usage>` block. The resulting string is injected as the text
/// content of a fresh user message on the parent's conversation.
pub(super) fn render_dispatch_notification(
    task: &crate::runtime::thread::Thread,
    parent_tool_use_id: &str,
    status: DispatchStatus,
    body: &str,
) -> String {
    let title = task.title.as_deref().unwrap_or(task.id.as_str());
    let summary = format!("dispatched thread \"{}\" {}", title, status.as_str());
    let total_tokens = task
        .total_usage
        .input_tokens
        .saturating_add(task.total_usage.output_tokens);
    let tool_uses = count_tool_uses(&task.conversation);
    let duration_ms = (task.last_active - task.created_at)
        .num_milliseconds()
        .max(0);
    let mut out = String::new();
    out.push_str("<dispatched-thread-notification>\n");
    out.push_str(&format!(
        "  <thread-id>{}</thread-id>\n",
        escape_xml(&task.id)
    ));
    out.push_str(&format!(
        "  <tool-use-id>{}</tool-use-id>\n",
        escape_xml(parent_tool_use_id)
    ));
    out.push_str(&format!("  <status>{}</status>\n", status.as_str()));
    out.push_str(&format!("  <summary>{}</summary>\n", escape_xml(&summary)));
    out.push_str(&format!("  <result>{}</result>\n", escape_xml(body)));
    out.push_str(&format!(
        "  <usage><total_tokens>{total_tokens}</total_tokens><tool_uses>{tool_uses}</tool_uses><duration_ms>{duration_ms}</duration_ms></usage>\n"
    ));
    out.push_str("</dispatched-thread-notification>");
    out
}

fn count_tool_uses(conv: &whisper_agent_protocol::Conversation) -> u32 {
    use whisper_agent_protocol::{ContentBlock, Role};
    let mut n: u32 = 0;
    for msg in conv.messages() {
        if msg.role != Role::Assistant {
            continue;
        }
        for block in &msg.content {
            if matches!(block, ContentBlock::ToolUse { .. }) {
                n = n.saturating_add(1);
            }
        }
    }
    n
}

fn escape_xml(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_xml_handles_angle_and_amp() {
        assert_eq!(escape_xml("<T>&"), "&lt;T&gt;&amp;");
        assert_eq!(escape_xml("plain"), "plain");
    }

    #[test]
    fn dispatch_status_str() {
        assert_eq!(DispatchStatus::Completed.as_str(), "completed");
        assert_eq!(DispatchStatus::Failed.as_str(), "failed");
        assert_eq!(DispatchStatus::Cancelled.as_str(), "cancelled");
    }
}
