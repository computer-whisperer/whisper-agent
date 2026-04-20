package com.cjbal.whisperagent.protocol

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Mirrors `whisper_agent_protocol::Role`. Snake-case wire form.
 */
@Serializable
enum class Role {
    @SerialName("system") System,
    @SerialName("tools") Tools,
    @SerialName("user") User,
    @SerialName("assistant") Assistant,
    @SerialName("tool_result") ToolResult,
}

@Serializable
data class Message(
    val role: Role,
    val content: List<ContentBlock>,
)

typealias Conversation = List<Message>

/**
 * Mirrors `whisper_agent_protocol::ContentBlock`. Serialized via
 * [ContentBlockSerializer] so the internally-tagged CBOR shape matches serde
 * (`#[serde(tag = "type", rename_all = "snake_case")]`) — see the wire-format
 * discussion in [ClientToServerSerializer].
 *
 * `ToolUse.input` and `ToolResult.content` carry an arbitrary JSON value on the
 * wire. kotlinx-serialization-cbor doesn't ship a CBOR tree type the way
 * JsonElement works for JSON, so those fields are stubbed out pending the
 * proof-of-concept decode.
 */
@Serializable(with = ContentBlockSerializer::class)
sealed class ContentBlock {
    data class Text(val text: String) : ContentBlock()

    // TODO: ToolUse.input is `serde_json::Value` — decide how to carry a
    //       CBOR-tree value once the proof-of-concept confirms the shape.
    data class ToolUse(
        val id: String,
        val name: String,
    ) : ContentBlock()

    // TODO: ToolResult.content is `ToolResultContent` (also a sealed enum in
    //       the Rust crate) — model properly once needed for rendering.
    data class ToolResult(
        val toolUseId: String,
        val isError: Boolean = false,
    ) : ContentBlock()

    data class Thinking(val thinking: String) : ContentBlock()

    data class ToolSchema(
        val name: String,
        val description: String,
    ) : ContentBlock()
}
