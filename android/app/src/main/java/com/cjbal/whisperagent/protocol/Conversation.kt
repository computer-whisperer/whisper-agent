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
 * Mirrors `whisper_agent_protocol::ContentBlock`. Internal tagging on `type`
 * plus snake_case variant names — matches the Rust `#[serde(tag = "type", rename_all = "snake_case")]`.
 *
 * ToolUse.input and ToolResult.content carry an arbitrary JSON value on the wire.
 * kotlinx-serialization-cbor doesn't ship a CBOR tree type the way JsonElement works
 * for JSON, so those fields are stubbed out pending the proof-of-concept decode.
 */
@Serializable
sealed class ContentBlock {
    @Serializable
    @SerialName("text")
    data class Text(val text: String) : ContentBlock()

    // TODO: ToolUse.input is `serde_json::Value` — decide how to carry a
    //       CBOR-tree value once the proof-of-concept confirms the shape.
    @Serializable
    @SerialName("tool_use")
    data class ToolUse(
        val id: String,
        val name: String,
    ) : ContentBlock()

    // TODO: ToolResult.content is `ToolResultContent` (also a sealed enum in
    //       the Rust crate) — model properly once needed for rendering.
    @Serializable
    @SerialName("tool_result")
    data class ToolResult(
        @SerialName("tool_use_id") val toolUseId: String,
        @SerialName("is_error") val isError: Boolean = false,
    ) : ContentBlock()

    @Serializable
    @SerialName("thinking")
    data class Thinking(val thinking: String) : ContentBlock()

    @Serializable
    @SerialName("tool_schema")
    data class ToolSchema(
        val name: String,
        val description: String,
    ) : ContentBlock()
}
