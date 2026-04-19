//! Canonical conversation state, modeled after Anthropic's content-block shape.
//!
//! Adapting *down* from this representation to the flatter OpenAI/Gemini shapes is
//! mechanical; adapting *up* from a flatter shape would require restructuring state.
//!
//! Field names use snake_case so serde serializes directly into Anthropic's request
//! shape — no manual translator needed for that path.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// User-typed input, or server-injected text addressed to the
    /// model (behavior-trigger prompts, compaction continuation
    /// seeds, `dispatch_thread` async notifications). From the
    /// model's perspective this is always "someone gave me new
    /// instructions or information."
    User,
    /// Model output — text, reasoning, tool-use requests.
    Assistant,
    /// Results for assistant-emitted tool calls. Kept distinct from
    /// `Role::User` so clients and the runtime can treat "tool
    /// finished" and "the user typed something" as different kinds of
    /// event without inspecting content blocks. Provider adapters
    /// translate this role back to each wire format (Anthropic:
    /// `user` with `tool_result` blocks; OpenAI chat: `tool` role;
    /// Gemini: function-response parts; etc.) on serialize.
    ToolResult,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

impl Message {
    pub fn user_text(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    pub fn user_blocks(blocks: Vec<ContentBlock>) -> Self {
        Self {
            role: Role::User,
            content: blocks,
        }
    }

    pub fn assistant_blocks(blocks: Vec<ContentBlock>) -> Self {
        Self {
            role: Role::Assistant,
            content: blocks,
        }
    }

    /// Construct a tool-result message. Typically one per turn,
    /// carrying all `ToolResult` content blocks for the turn's tool
    /// calls — matches the way Anthropic's wire format already bundles
    /// tool results, with adapters splitting/resplicing for providers
    /// that want separate messages per result.
    pub fn tool_result_blocks(blocks: Vec<ContentBlock>) -> Self {
        Self {
            role: Role::ToolResult,
            content: blocks,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: ToolResultContent,
        #[serde(default, skip_serializing_if = "is_false")]
        is_error: bool,
    },
    Thinking {
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
        thinking: String,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

fn is_false(b: &bool) -> bool {
    !b
}

/// Ordered list of messages. Owns the vector; exposes controlled push access so callers
/// don't splice arbitrarily into the middle (the Anthropic request shape requires a
/// well-formed alternation of user/assistant roles).
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(transparent)]
pub struct Conversation {
    messages: Vec<Message>,
}

impl Conversation {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, m: Message) {
        self.messages.push(m);
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// In-place migration for threads persisted before `Role::ToolResult`
    /// existed: tool results used to ride inside `Role::User` messages
    /// as `ContentBlock::ToolResult` blocks. After adding the new role
    /// we want them to live under it.
    ///
    /// Rules:
    /// - A user-role message whose content blocks are ALL `ToolResult`
    ///   is reassigned to `Role::ToolResult` in place.
    /// - A user-role message with mixed content (text + tool results,
    ///   hypothetical in current code but possible in edited JSON) is
    ///   split into two adjacent messages: a `Role::User` holding the
    ///   non-tool-result blocks, and a `Role::ToolResult` holding the
    ///   tool-result blocks.
    /// - Messages without any `ToolResult` blocks are untouched.
    ///
    /// Safe to run repeatedly (idempotent) — after a prior run, no
    /// user-role message contains `ToolResult` blocks so the walk
    /// makes no changes.
    pub fn normalize_legacy_tool_result_role(&mut self) {
        let mut out: Vec<Message> = Vec::with_capacity(self.messages.len());
        for msg in self.messages.drain(..) {
            if msg.role != Role::User {
                out.push(msg);
                continue;
            }
            let any_tool_result = msg
                .content
                .iter()
                .any(|b| matches!(b, ContentBlock::ToolResult { .. }));
            if !any_tool_result {
                out.push(msg);
                continue;
            }
            let all_tool_result = msg
                .content
                .iter()
                .all(|b| matches!(b, ContentBlock::ToolResult { .. }));
            if all_tool_result {
                out.push(Message {
                    role: Role::ToolResult,
                    content: msg.content,
                });
                continue;
            }
            // Mixed — partition while preserving original order within
            // each partition; emit the text-ish partition first under
            // Role::User and the tool-results partition second under
            // Role::ToolResult. Matches the temporal order the runtime
            // would have produced if the two concepts had been separate
            // from the start.
            let mut user_side = Vec::new();
            let mut tool_side = Vec::new();
            for block in msg.content {
                match block {
                    ContentBlock::ToolResult { .. } => tool_side.push(block),
                    _ => user_side.push(block),
                }
            }
            if !user_side.is_empty() {
                out.push(Message {
                    role: Role::User,
                    content: user_side,
                });
            }
            if !tool_side.is_empty() {
                out.push(Message {
                    role: Role::ToolResult,
                    content: tool_side,
                });
            }
        }
        self.messages = out;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_promotes_user_tool_result_to_tool_result_role() {
        let mut conv = Conversation::new();
        conv.push(Message::user_text("hi"));
        conv.push(Message::assistant_blocks(vec![ContentBlock::ToolUse {
            id: "u1".into(),
            name: "t".into(),
            input: serde_json::Value::Null,
        }]));
        // Legacy shape: tool result under Role::User.
        conv.push(Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "u1".into(),
                content: ToolResultContent::Text("ok".into()),
                is_error: false,
            }],
        });
        conv.normalize_legacy_tool_result_role();
        assert_eq!(conv.messages().len(), 3);
        assert_eq!(conv.messages()[0].role, Role::User);
        assert_eq!(conv.messages()[1].role, Role::Assistant);
        assert_eq!(conv.messages()[2].role, Role::ToolResult);
    }

    #[test]
    fn normalize_splits_mixed_user_message() {
        let mut conv = Conversation::new();
        conv.push(Message {
            role: Role::User,
            content: vec![
                ContentBlock::Text {
                    text: "hand-edited".into(),
                },
                ContentBlock::ToolResult {
                    tool_use_id: "u1".into(),
                    content: ToolResultContent::Text("r".into()),
                    is_error: false,
                },
            ],
        });
        conv.normalize_legacy_tool_result_role();
        assert_eq!(conv.messages().len(), 2);
        assert_eq!(conv.messages()[0].role, Role::User);
        assert_eq!(conv.messages()[1].role, Role::ToolResult);
    }

    #[test]
    fn normalize_is_idempotent() {
        let mut conv = Conversation::new();
        conv.push(Message::tool_result_blocks(vec![
            ContentBlock::ToolResult {
                tool_use_id: "u1".into(),
                content: ToolResultContent::Text("ok".into()),
                is_error: false,
            },
        ]));
        let before = serde_json::to_string(&conv).unwrap();
        conv.normalize_legacy_tool_result_role();
        conv.normalize_legacy_tool_result_role();
        let after = serde_json::to_string(&conv).unwrap();
        assert_eq!(before, after);
    }
}
