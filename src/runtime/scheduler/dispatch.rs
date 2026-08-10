//! Dispatch helpers. `dispatch_thread` is a tool-pool alias for
//! `Function::CreateThread`, handled by the single tool-dispatch entry
//! point in `functions.rs::register_dispatch_thread_tool`; async
//! terminals reach the parent's coordinating driver as
//! `dispatch_completed`/`dispatch_failed` events (step 11 slice 4).
//! The pre-slice-6 `<dispatched-thread-notification>` envelope path
//! died with the builtin driver — only the usage accounting shared
//! with the scripted event payloads remains here.

pub(super) fn count_tool_uses(conv: &whisper_agent_protocol::Conversation) -> u32 {
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
