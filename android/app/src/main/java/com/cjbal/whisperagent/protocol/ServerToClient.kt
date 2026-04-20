package com.cjbal.whisperagent.protocol

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Messages the server pushes to the client. Mirrors the Rust
 * `whisper_agent_protocol::ServerToClient` enum — v1 subset of variants the
 * Android UI actually handles.
 *
 * Unknown variants land on [Unknown] via the polymorphic default deserializer
 * configured in [Codec], so a server that grows a new event kind doesn't crash
 * the client.
 */
@Serializable
sealed class ServerToClient {

    // --- Thread-list tier -----------------------------------------------------

    @Serializable
    @SerialName("thread_list")
    data class ThreadList(
        @SerialName("correlation_id") val correlationId: String? = null,
        val tasks: List<ThreadSummary>,
    ) : ServerToClient()

    @Serializable
    @SerialName("thread_created")
    data class ThreadCreated(
        @SerialName("thread_id") val threadId: String,
        val summary: ThreadSummary,
        @SerialName("correlation_id") val correlationId: String? = null,
    ) : ServerToClient()

    @Serializable
    @SerialName("thread_state_changed")
    data class ThreadStateChanged(
        @SerialName("thread_id") val threadId: String,
        val state: ThreadStateLabel,
    ) : ServerToClient()

    @Serializable
    @SerialName("thread_title_updated")
    data class ThreadTitleUpdated(
        @SerialName("thread_id") val threadId: String,
        val title: String,
    ) : ServerToClient()

    @Serializable
    @SerialName("thread_archived")
    data class ThreadArchived(
        @SerialName("thread_id") val threadId: String,
    ) : ServerToClient()

    // --- Per-thread tier ------------------------------------------------------

    @Serializable
    @SerialName("thread_snapshot")
    data class Snapshot(
        @SerialName("thread_id") val threadId: String,
        val snapshot: ThreadSnapshot,
    ) : ServerToClient()

    @Serializable
    @SerialName("thread_user_message")
    data class ThreadUserMessage(
        @SerialName("thread_id") val threadId: String,
        val text: String,
    ) : ServerToClient()

    @Serializable
    @SerialName("thread_assistant_begin")
    data class AssistantBegin(
        @SerialName("thread_id") val threadId: String,
        val turn: Int,
    ) : ServerToClient()

    @Serializable
    @SerialName("thread_assistant_text_delta")
    data class AssistantTextDelta(
        @SerialName("thread_id") val threadId: String,
        val delta: String,
    ) : ServerToClient()

    @Serializable
    @SerialName("thread_assistant_reasoning_delta")
    data class AssistantReasoningDelta(
        @SerialName("thread_id") val threadId: String,
        val delta: String,
    ) : ServerToClient()

    @Serializable
    @SerialName("thread_assistant_end")
    data class AssistantEnd(
        @SerialName("thread_id") val threadId: String,
        @SerialName("stop_reason") val stopReason: String? = null,
        val usage: Usage,
    ) : ServerToClient()

    @Serializable
    @SerialName("thread_tool_call_begin")
    data class ToolCallBegin(
        @SerialName("thread_id") val threadId: String,
        @SerialName("tool_use_id") val toolUseId: String,
        val name: String,
        @SerialName("args_preview") val argsPreview: String,
        // TODO: `args: Option<serde_json::Value>` — add once CBOR-tree handling lands.
    ) : ServerToClient()

    @Serializable
    @SerialName("thread_tool_call_end")
    data class ToolCallEnd(
        @SerialName("thread_id") val threadId: String,
        @SerialName("tool_use_id") val toolUseId: String,
        @SerialName("result_preview") val resultPreview: String,
        @SerialName("is_error") val isError: Boolean,
    ) : ServerToClient()

    @Serializable
    @SerialName("thread_loop_complete")
    data class LoopComplete(
        @SerialName("thread_id") val threadId: String,
    ) : ServerToClient()

    @Serializable
    @SerialName("thread_pending_approval")
    data class PendingApproval(
        @SerialName("thread_id") val threadId: String,
        @SerialName("approval_id") val approvalId: String,
        @SerialName("tool_use_id") val toolUseId: String,
        val name: String,
        @SerialName("args_preview") val argsPreview: String,
        val destructive: Boolean = false,
        @SerialName("read_only") val readOnly: Boolean = false,
    ) : ServerToClient()

    @Serializable
    @SerialName("thread_approval_resolved")
    data class ApprovalResolved(
        @SerialName("thread_id") val threadId: String,
        @SerialName("approval_id") val approvalId: String,
        val decision: ApprovalChoice,
    ) : ServerToClient()

    // --- Errors ---------------------------------------------------------------

    @Serializable
    @SerialName("error")
    data class Error(
        @SerialName("correlation_id") val correlationId: String? = null,
        @SerialName("thread_id") val threadId: String? = null,
        val message: String,
    ) : ServerToClient()

    /**
     * Catch-all for variants the Rust enum has but the Kotlin mirror doesn't
     * know about yet (pod / behavior / resource / function tiers, etc.).
     * Carries the wire `type` discriminator so a debug view can at least
     * label it.
     */
    @Serializable
    data class Unknown(val type: String) : ServerToClient()
}
