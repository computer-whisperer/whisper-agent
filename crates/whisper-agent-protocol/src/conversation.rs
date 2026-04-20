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
    /// System-authored guidance the model should treat as harness
    /// instructions, not as user speech. Covers two distinct uses:
    ///
    /// 1. **Thread-prefix system prompt** — lives as `messages[0]`,
    ///    captured at thread creation so the log faithfully records
    ///    what instructions the model saw. Provider adapters lift
    ///    this into their wire-level `system` field (Anthropic) or
    ///    prefix `system`-role message (OpenAI chat / Responses) /
    ///    `systemInstruction` (Gemini).
    ///
    /// 2. **Mid-conversation injections** — later entries the harness
    ///    appends to steer or brief the model (e.g. a memory-index
    ///    block placed before the next user message). Providers that
    ///    accept a mid-conversation system role (OpenAI chat,
    ///    Responses) emit it as `role: "system"` in place; providers
    ///    that don't (Anthropic, Gemini) emit it as a `user` message
    ///    with content wrapped in `<system-reminder>...</system-reminder>`
    ///    so the model can distinguish harness guidance from real
    ///    user speech.
    System,
    /// Snapshot of the tool manifest the model was shown. Lives as
    /// a `Tools` message immediately after `System` at the top of
    /// the conversation — content is a list of
    /// `ContentBlock::ToolSchema` entries, one per tool advertised
    /// at the moment the thread was created (or the tool set was
    /// rebound). Stored in the conversation (rather than resolved
    /// at request time) because any change to the tool manifest is
    /// a prompt-cache buster: capturing the snapshot here keeps the
    /// conversation log identical to what the model actually saw,
    /// and makes mid-thread tool changes an explicit event rather
    /// than silent drift. Adapters extract these blocks into their
    /// native `tools` request field.
    Tools,
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

    /// Build the thread-prefix system message from the pod's resolved
    /// system prompt text. Lives at `conversation[0]` for every thread;
    /// provider adapters lift the text into their wire-level `system`
    /// field. Intentionally preserved even when the pod's
    /// `system_prompt.md` is empty (content is a single empty-string
    /// text block) so the index of subsequent messages is stable —
    /// adapters are responsible for skipping a wire `system` emission
    /// when the text is empty.
    pub fn system_text(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    /// Build the thread-prefix tools message from a resolved tool
    /// manifest. Each entry becomes a `ContentBlock::ToolSchema`;
    /// provider adapters lift these into their native `tools` field on
    /// outbound requests. Lives at `conversation[1]` for every thread,
    /// refreshed in place when bindings rebind the MCP host set.
    pub fn tools_manifest(tools: Vec<ContentBlock>) -> Self {
        debug_assert!(
            tools
                .iter()
                .all(|b| matches!(b, ContentBlock::ToolSchema { .. })),
            "Role::Tools content must be ContentBlock::ToolSchema entries"
        );
        Self {
            role: Role::Tools,
            content: tools,
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

    /// Construct a tool-output message carrying unstructured text —
    /// used for async tool-call callbacks that can't bind to the
    /// original `tool_use_id` (the initial synchronous ack already
    /// consumed it). The role still marks the message as tool output
    /// so clients can render it distinctly from user-typed text and
    /// the webui can reroute the payload back to the originating
    /// tool-call item. On the wire, provider adapters fold
    /// `Role::ToolResult` back to their native user-role shape
    /// (Anthropic: `user`; OpenAI chat / responses: `user` for text
    /// content), so the model sees it as a normal follow-up user
    /// turn.
    pub fn tool_result_text(text: impl Into<String>) -> Self {
        Self {
            role: Role::ToolResult,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
        /// Provider-tagged opaque blob carried forward for reasoning
        /// replay — e.g. Gemini attaches `thoughtSignature` to the
        /// `functionCall` part so chain-of-thought resumes across the
        /// tool_result boundary. `None` for providers that don't use it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        replay: Option<ProviderReplay>,
    },
    ToolResult {
        tool_use_id: String,
        content: ToolResultContent,
        #[serde(default, skip_serializing_if = "is_false")]
        is_error: bool,
    },
    Thinking {
        /// Provider-tagged opaque blob carried forward for reasoning
        /// replay — Anthropic's `signature`, OpenAI Responses'
        /// `encrypted_content` plus item id. `None` when the provider
        /// doesn't supply reasoning-continuation data (or when the block
        /// came from a provider that scopes it elsewhere, like Gemini,
        /// which attaches its signature to `ToolUse` instead).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        replay: Option<ProviderReplay>,
        thinking: String,
    },
    /// One entry in a tool manifest snapshot. Appears only as content
    /// on a `Role::Tools` message; provider adapters lift it into
    /// their native `tools` request field (Anthropic `tools[]`,
    /// OpenAI `tools[].function`, Gemini `FunctionDeclaration`). The
    /// fields mirror `ToolSpec` — name, free-form description, JSON
    /// Schema for the input — so translation is direct.
    ToolSchema {
        name: String,
        description: String,
        input_schema: Value,
    },
}

/// Opaque provider-specific data echoed back into later requests so the
/// server can resume its chain-of-thought across turns.
///
/// Cross-provider safety: adapters must strip (or drop) blocks whose
/// `provider` tag doesn't match the current backend — a signature minted
/// by Anthropic is meaningless (and potentially an error) to Gemini.
/// The scheduler keeps the block in conversation state so switching back
/// to the original backend later still has the replay data available.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ProviderReplay {
    /// Backend that produced this blob. Match against the adapter's own
    /// identifier (`"anthropic"`, `"openai_responses"`, `"gemini"`) on
    /// outbound to decide whether to echo or drop.
    pub provider: String,
    /// Provider-defined payload. Each adapter chooses its own shape —
    /// Anthropic uses `{"signature": "..."}`, Gemini uses
    /// `{"thought_signature": "..."}`, OpenAI Responses carries the
    /// whole `reasoning` item (`{"id": "...", "encrypted_content":
    /// "...", "summary": [...]}`) so the echoed-back item matches the
    /// server's expected shape exactly.
    pub data: Value,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
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

    /// Insert `m` at `at`, shifting later messages right by one.
    /// Callers use this when the setup prefix (`[System, Tools]`)
    /// needs to be completed after body messages already landed —
    /// specifically, when a host-env MCP's `tools/list` finishes
    /// after the thread has accepted its initial user message.
    /// Panics if `at > len()`, matching `Vec::insert` semantics.
    pub fn insert(&mut self, at: usize, m: Message) {
        self.messages.insert(at, m);
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

    /// Drop every message at or after `at`. Delegates to
    /// [`Vec::truncate`], so `at >= self.len()` is a no-op.
    pub fn truncate(&mut self, at: usize) {
        self.messages.truncate(at);
    }

    /// System-prompt text captured at thread creation. Looks at
    /// `messages[0]` and returns the first `ContentBlock::Text` body
    /// if (and only if) the role is `Role::System`. Returns `""` for
    /// conversations without a system-prefix (empty conversations,
    /// legacy fixtures, etc.) so callers can treat the empty case as
    /// "no system prompt."
    pub fn system_prompt_text(&self) -> &str {
        let Some(msg) = self.messages.first() else {
            return "";
        };
        if msg.role != Role::System {
            return "";
        }
        msg.content
            .iter()
            .find_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .unwrap_or("")
    }

    /// Mutable access to the `Role::Tools` manifest message — returns
    /// `None` if the conversation doesn't yet have one at the expected
    /// slot (`messages[1]` when `messages[0]` is `Role::System`, or
    /// `messages[0]` otherwise). Used by the scheduler's rebind path to
    /// refresh the snapshot in place when the MCP host set changes.
    pub fn tools_message_mut(&mut self) -> Option<&mut Message> {
        let idx = match self.messages.first() {
            Some(m) if m.role == Role::System => 1,
            _ => 0,
        };
        let msg = self.messages.get_mut(idx)?;
        if msg.role == Role::Tools {
            Some(msg)
        } else {
            None
        }
    }

    /// Iterator over the `Role::ToolSchema` entries stored in the
    /// thread's `Role::Tools` manifest message. Returns an empty iter
    /// if the manifest slot isn't populated — adapters interpret that
    /// as "this thread has no tools available."
    pub fn tool_schemas(&self) -> impl Iterator<Item = (&str, &str, &Value)> {
        let idx = match self.messages.first() {
            Some(m) if m.role == Role::System => 1,
            _ => 0,
        };
        let blocks = self
            .messages
            .get(idx)
            .filter(|m| m.role == Role::Tools)
            .map(|m| m.content.as_slice())
            .unwrap_or(&[]);
        blocks.iter().filter_map(|b| match b {
            ContentBlock::ToolSchema {
                name,
                description,
                input_schema,
            } => Some((name.as_str(), description.as_str(), input_schema)),
            _ => None,
        })
    }

    /// Index of the first message that is *not* part of the setup
    /// prefix (neither `Role::System` nor `Role::Tools`). Used by
    /// the model-request builder to slice off the setup messages
    /// before handing the conversation to a provider adapter, and by
    /// cache-policy code to skip setup when choosing the rolling
    /// breakpoint anchor.
    pub fn setup_prefix_end(&self) -> usize {
        self.messages
            .iter()
            .take_while(|m| matches!(m.role, Role::System | Role::Tools))
            .count()
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
            replay: None,
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
