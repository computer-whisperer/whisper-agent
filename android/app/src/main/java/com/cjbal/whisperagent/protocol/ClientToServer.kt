package com.cjbal.whisperagent.protocol

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Messages the Android client sends upstream. Mirrors the Rust
 * `whisper_agent_protocol::ClientToServer` enum — but only the variants v1 of
 * this client actually constructs. Server still accepts any of its richer
 * variants; add more here as the UI grows.
 */
@Serializable
sealed class ClientToServer {

    @Serializable
    @SerialName("list_threads")
    data class ListThreads(
        @SerialName("correlation_id") val correlationId: String? = null,
    ) : ClientToServer()

    @Serializable
    @SerialName("subscribe_to_thread")
    data class SubscribeToThread(
        @SerialName("thread_id") val threadId: String,
    ) : ClientToServer()

    @Serializable
    @SerialName("unsubscribe_from_thread")
    data class UnsubscribeFromThread(
        @SerialName("thread_id") val threadId: String,
    ) : ClientToServer()

    @Serializable
    @SerialName("send_user_message")
    data class SendUserMessage(
        @SerialName("thread_id") val threadId: String,
        val text: String,
    ) : ClientToServer()

    @Serializable
    @SerialName("approval_decision")
    data class ApprovalDecision(
        @SerialName("thread_id") val threadId: String,
        @SerialName("approval_id") val approvalId: String,
        val decision: ApprovalChoice,
        val remember: Boolean = false,
    ) : ClientToServer()
}
