//! The canonical conversation state, modeled after Anthropic's content-block shape.
//!
//! Adapting *down* from this representation to the flatter OpenAI/Gemini shapes is
//! mechanical; adapting *up* from a flatter shape would require restructuring state.
//! See `docs/design_mvp.md` for the rationale.
//!
//! Field names use snake_case so serde serializes directly into Anthropic's request
//! shape — no manual translator needed for that path.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
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
        Self { role: Role::User, content: blocks }
    }

    pub fn assistant_blocks(blocks: Vec<ContentBlock>) -> Self {
        Self { role: Role::Assistant, content: blocks }
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

#[derive(Debug, Default)]
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
}
