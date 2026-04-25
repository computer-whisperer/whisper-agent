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

/// Image MIME types we carry through the protocol. Kept as a closed
/// enum (not a free-form string) so capability declarations
/// (`ContentCapabilities::input.image`) can be compared by value and
/// provider adapters get exhaustive matching when deciding how to
/// translate. Covers the union of what the four backends actually
/// accept today; anything outside this set gets rejected at the
/// scheduler edge.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ImageMime {
    Jpeg,
    Png,
    Gif,
    Webp,
    /// Apple's HEIC/HEIF — accepted by Gemini, rejected by
    /// Anthropic/OpenAI. Adapters that can't serve it must surface a
    /// clear error before dispatch rather than forwarding a 400 from
    /// upstream.
    Heic,
    Heif,
}

impl ImageMime {
    pub fn as_mime_str(self) -> &'static str {
        match self {
            ImageMime::Jpeg => "image/jpeg",
            ImageMime::Png => "image/png",
            ImageMime::Gif => "image/gif",
            ImageMime::Webp => "image/webp",
            ImageMime::Heic => "image/heic",
            ImageMime::Heif => "image/heif",
        }
    }

    /// Parse a MIME string into the closed enum. Accepts the common
    /// `image/jpg` alias as `Jpeg`. Returns `None` for anything we
    /// don't carry — callers reject at that boundary.
    pub fn from_mime_str(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "image/jpeg" | "image/jpg" => Some(ImageMime::Jpeg),
            "image/png" => Some(ImageMime::Png),
            "image/gif" => Some(ImageMime::Gif),
            "image/webp" => Some(ImageMime::Webp),
            "image/heic" => Some(ImageMime::Heic),
            "image/heif" => Some(ImageMime::Heif),
            _ => None,
        }
    }
}

/// Where the bytes of an image content block come from.
///
/// `Bytes` is the canonical form: every backend accepts inline image
/// data (base64 / data URL / `inline_data`), so the scheduler can
/// always dispatch a thread containing these. `Url` is a convenience
/// for user-supplied web URLs — Anthropic and OpenAI pass it through
/// natively; the Gemini adapter has to fetch-and-inline because
/// Gemini doesn't accept URL image sources.
///
/// Data lives as raw bytes in memory. Adapters encode per-provider at
/// send time (base64 for Anthropic / OpenAI Chat data URLs / OpenAI
/// Responses, raw bytes then base64 for Gemini `inline_data`). We
/// don't keep a base64 string around because the hot path is the
/// outbound encode, not in-memory inspection.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    Bytes {
        media_type: ImageMime,
        #[serde(with = "image_bytes_serde")]
        data: Vec<u8>,
    },
    Url {
        url: String,
    },
}

/// Document MIME types the protocol carries. Today only PDF — every
/// provider accepts PDFs natively (text + page images). Audio and
/// other document formats stay as a future extension.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum DocumentMime {
    Pdf,
}

impl DocumentMime {
    pub fn as_mime_str(self) -> &'static str {
        match self {
            DocumentMime::Pdf => "application/pdf",
        }
    }

    /// Parse a MIME string into the closed enum. Returns `None` for
    /// anything we don't carry — callers reject at that boundary.
    pub fn from_mime_str(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "application/pdf" => Some(DocumentMime::Pdf),
            _ => None,
        }
    }
}

/// Where the bytes of a document content block come from. Mirrors
/// [`ImageSource`] — `Bytes` is the canonical inline form, `Url` is
/// used for provider-native URL passthrough (Anthropic, OpenAI
/// Responses; the OpenAI Chat and Gemini adapters fetch-and-inline).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DocumentSource {
    Bytes {
        media_type: DocumentMime,
        // The byte serializer is provider-agnostic and shared with
        // images: base64 in human-readable formats, native bytes in
        // CBOR. Reuse rather than duplicate.
        #[serde(with = "image_bytes_serde")]
        data: Vec<u8>,
    },
    Url {
        url: String,
    },
}

/// Format-aware (de)serialization for image byte buffers. Human-
/// readable formats (disk JSON, debug dumps, YAML test fixtures) get
/// a base64 string so a 2 MB image doesn't balloon to 8 MB of
/// `[1, 2, 3, ...]`. Binary formats (CBOR on the wire) get native
/// byte-string encoding — zero-copy on the wire, no encode overhead.
mod image_bytes_serde {
    use base64::Engine;
    use base64::engine::general_purpose::STANDARD;
    use serde::de::{self, Visitor};
    use serde::{Deserializer, Serializer};
    use std::fmt;

    pub fn serialize<S: Serializer>(data: &[u8], s: S) -> Result<S::Ok, S::Error> {
        if s.is_human_readable() {
            s.serialize_str(&STANDARD.encode(data))
        } else {
            s.serialize_bytes(data)
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Vec<u8>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("base64 string or raw byte buffer")
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
                STANDARD.decode(v).map_err(de::Error::custom)
            }
            fn visit_string<E: de::Error>(self, v: String) -> Result<Self::Value, E> {
                self.visit_str(&v)
            }
            fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<Self::Value, E> {
                Ok(v.to_vec())
            }
            fn visit_byte_buf<E: de::Error>(self, v: Vec<u8>) -> Result<Self::Value, E> {
                Ok(v)
            }
            fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                // Tolerate the `[1, 2, 3]` shape any plain serde_bytes-less
                // producer would emit — lets us read conversations that
                // predated the format-aware serializer, or bytes that
                // another JSON producer rendered as an int array.
                let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(0));
                while let Some(b) = seq.next_element::<u8>()? {
                    out.push(b);
                }
                Ok(out)
            }
        }
        d.deserialize_any(V)
    }
}

/// What a model is willing to ingest / emit on each media axis.
/// Empty `image` / `audio` / `document` on the `input` side means
/// "this model doesn't accept user-supplied media of that kind";
/// empty on the `output` side means "won't emit it." The webui
/// consults `input.image` to gate paste/drop, and scheduler-side
/// pre-dispatch validation rejects attachments whose MIME isn't
/// listed (cleaner 400 from us than a 400 bounced back from
/// upstream after the bytes are on the wire).
///
/// Kept per-MIME (not a boolean) because providers actually
/// diverge by MIME — Gemini accepts HEIC/HEIF, the others don't;
/// Anthropic does GIF but OpenAI only does non-animated GIF. We
/// store what the model accepts, not a lowest-common-denominator.
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq, Eq)]
pub struct ContentCapabilities {
    #[serde(default, skip_serializing_if = "MediaSupport::is_empty")]
    pub input: MediaSupport,
    #[serde(default, skip_serializing_if = "MediaSupport::is_empty")]
    pub output: MediaSupport,
}

/// Per-kind MIME listing for either the input or output side of a
/// model. Audio is kept as a forward-compatible placeholder — it
/// serializes as an empty vec today and becomes real once its
/// `AudioMime` enum ships. Typed as `Vec<String>` for now to avoid
/// churning the protocol when that list grows; image and document use
/// strong enums.
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq, Eq)]
pub struct MediaSupport {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub image: Vec<ImageMime>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub audio: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub document: Vec<DocumentMime>,
}

impl MediaSupport {
    pub fn is_empty(&self) -> bool {
        self.image.is_empty() && self.audio.is_empty() && self.document.is_empty()
    }

    /// Convenience for the common provider case: "this model accepts
    /// JPEG, PNG, WEBP, GIF image input, nothing else." Gemini
    /// overrides with HEIC/HEIF added.
    pub fn standard_image_input() -> Self {
        Self {
            image: vec![
                ImageMime::Jpeg,
                ImageMime::Png,
                ImageMime::Webp,
                ImageMime::Gif,
            ],
            audio: Vec::new(),
            document: Vec::new(),
        }
    }
}

/// Media a client attaches to an outbound user message or a thread-
/// creation request. Restricted subset of [`ContentBlock`] — the
/// scheduler unpacks each variant into the matching content-block kind
/// when assembling the message, so the server never has to trust the
/// client to send arbitrary content-block shapes (no `ToolUse` or
/// `Thinking` on the wire from a client).
///
/// Kept as a `kind`-tagged enum rather than a wrapper struct so
/// extending to audio / document attachments later is a new variant,
/// not a schema migration.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Attachment {
    Image { source: ImageSource },
}

impl Attachment {
    /// Lower an attachment into the content-block the scheduler pushes
    /// onto the conversation. Kept as a method on `Attachment` so new
    /// attachment kinds can't be added without also defining the
    /// content-block mapping — they become compile errors here.
    pub fn into_content_block(self) -> ContentBlock {
        match self {
            Attachment::Image { source } => ContentBlock::Image { source },
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
    /// User-supplied or model-generated image. User-supplied side works
    /// today; model-generated side is a placeholder for the v2 output path
    /// (Gemini native image output, OpenAI Responses `image_generation`
    /// tool) and isn't produced yet. Adapters translate `ImageSource`
    /// into their provider-specific wire shape at send time.
    Image {
        source: ImageSource,
    },
    /// User-supplied or tool-returned document (PDF today). Each
    /// provider does its own server-side decomposition into text +
    /// per-page images; we carry the raw bytes (or URL) through and
    /// adapters translate into the provider-specific document shape:
    /// Anthropic `document` block, OpenAI Chat `file` block / OpenAI
    /// Responses `input_file` block, Gemini `inline_data` part with
    /// `application/pdf` MIME.
    Document {
        source: DocumentSource,
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

impl ToolResultContent {
    /// Borrow all image sources embedded in this tool result, in
    /// order. Returns empty for the `Text` variant and for `Blocks`
    /// variants that happen to carry only text. Provider adapters that
    /// can't ride images inside their native tool-result wire shape
    /// (OpenAI chat/responses, Gemini) use this to splice a follow-up
    /// user message carrying the image attachments.
    pub fn image_sources(&self) -> Vec<&ImageSource> {
        match self {
            ToolResultContent::Text(_) => Vec::new(),
            ToolResultContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Image { source } => Some(source),
                    _ => None,
                })
                .collect(),
        }
    }

    /// Borrow all document sources embedded in this tool result, in
    /// order. Mirror of [`Self::image_sources`]: provider adapters
    /// that can't carry documents inside their native tool-result
    /// wire shape (OpenAI chat / responses, Gemini) splice a follow-
    /// up user message carrying the document attachments.
    pub fn document_sources(&self) -> Vec<&DocumentSource> {
        match self {
            ToolResultContent::Text(_) => Vec::new(),
            ToolResultContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Document { source } => Some(source),
                    _ => None,
                })
                .collect(),
        }
    }
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

    /// Index of the first message that is not lifted into a wire-level
    /// request field by provider adapters. Exactly covers the thread-
    /// prefix `Role::System` at index 0 and the `Role::Tools` manifest
    /// at index 1 — both optional, both at fixed positions — so the
    /// return value is always 0, 1, or 2. Used by the model-request
    /// builder to slice off *just these* before handing the
    /// conversation to the adapter; anything else (including mid-
    /// conversation `Role::System` harness injections like the memory
    /// index) is body content the adapter translates to a wire message.
    pub fn setup_prefix_end(&self) -> usize {
        let mut end = 0;
        if self
            .messages
            .first()
            .is_some_and(|m| m.role == Role::System)
        {
            end = 1;
        }
        if self
            .messages
            .get(end)
            .is_some_and(|m| m.role == Role::Tools)
        {
            end += 1;
        }
        end
    }

    /// Index of the first body message, treating every leading
    /// `Role::System` or `Role::Tools` entry as part of the thread's
    /// initial context. Broader than [`Self::setup_prefix_end`]: a
    /// mid-conversation harness injection (e.g. a memory-index block
    /// pushed as `Role::System` right after the tool manifest) is
    /// counted here but NOT there — it's body content from the
    /// adapter's perspective, but part of the thread's stable opening
    /// frame from a fork's perspective.
    ///
    /// Fork inheritance uses this: copying through this index keeps
    /// the forked prefix byte-identical to the parent's, so the
    /// server's prompt cache still hits on the first send.
    pub fn initial_context_end(&self) -> usize {
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
    fn setup_prefix_end_stops_at_tight_positions() {
        // Only System@0 and Tools@1 count as the setup prefix now.
        // A mid-conversation Role::System (e.g. a memory-index
        // injection placed between Tools and the first user turn) is
        // body content from the adapter's perspective — it must not
        // be included in the setup slice the adapter strips.
        let mut conv = Conversation::new();
        conv.push(Message::system_text("prompt"));
        conv.push(Message::tools_manifest(Vec::new()));
        conv.push(Message::system_text("injected memory index"));
        conv.push(Message::user_text("hi"));
        assert_eq!(conv.setup_prefix_end(), 2);
    }

    #[test]
    fn setup_prefix_end_handles_tools_only_and_system_only() {
        // System present, no Tools yet: 1.
        let mut conv = Conversation::new();
        conv.push(Message::system_text("prompt"));
        conv.push(Message::user_text("hi"));
        assert_eq!(conv.setup_prefix_end(), 1);
        // Tools-only (no System) — unusual but valid when the pod has
        // an empty system prompt.
        let mut conv = Conversation::new();
        conv.push(Message::tools_manifest(Vec::new()));
        conv.push(Message::user_text("hi"));
        assert_eq!(conv.setup_prefix_end(), 1);
    }

    #[test]
    fn initial_context_end_includes_memory_injection() {
        // Fork-inheritance slice covers everything at the head that
        // looks like scaffolding — System + Tools + the memory
        // injection. Truncating at this index in the parent gives a
        // byte-stable prefix for the fork.
        let mut conv = Conversation::new();
        conv.push(Message::system_text("prompt"));
        conv.push(Message::tools_manifest(Vec::new()));
        conv.push(Message::system_text("injected memory index"));
        conv.push(Message::user_text("hi"));
        assert_eq!(conv.initial_context_end(), 3);
    }

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
    fn image_source_bytes_roundtrips_as_base64_in_json() {
        // Human-readable formats (disk JSON) encode the bytes as a
        // compact base64 string, not an int-array. Verifies both
        // directions of the format-aware helper.
        let src = ImageSource::Bytes {
            media_type: ImageMime::Png,
            data: vec![137, 80, 78, 71, 13, 10, 26, 10],
        };
        let json = serde_json::to_string(&src).unwrap();
        assert!(
            json.contains("\"iVBORw0KGgo=\""),
            "expected base64-encoded PNG magic bytes, got: {json}"
        );
        let back: ImageSource = serde_json::from_str(&json).unwrap();
        assert_eq!(back, src);
    }

    #[test]
    fn image_source_bytes_roundtrips_through_cbor_as_raw_bytes() {
        // Binary formats get native byte-string encoding — roundtrip is
        // the behavior we care about; the exact CBOR shape is ciborium's
        // implementation detail.
        let src = ImageSource::Bytes {
            media_type: ImageMime::Jpeg,
            data: vec![0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10],
        };
        let mut buf = Vec::new();
        ciborium::ser::into_writer(&src, &mut buf).unwrap();
        let back: ImageSource = ciborium::de::from_reader(buf.as_slice()).unwrap();
        assert_eq!(back, src);
    }

    #[test]
    fn image_source_bytes_tolerates_legacy_int_array_json() {
        // If a prior producer emitted the raw serde_bytes-less shape
        // (int array) we still deserialize successfully — the
        // human-readable Visitor accepts both forms.
        let legacy = r#"{"type":"bytes","media_type":"png","data":[137,80,78,71]}"#;
        let src: ImageSource = serde_json::from_str(legacy).unwrap();
        match src {
            ImageSource::Bytes { data, media_type } => {
                assert_eq!(data, vec![137, 80, 78, 71]);
                assert_eq!(media_type, ImageMime::Png);
            }
            other => panic!("expected Bytes, got {other:?}"),
        }
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
