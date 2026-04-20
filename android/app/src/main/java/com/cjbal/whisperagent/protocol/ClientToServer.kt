package com.cjbal.whisperagent.protocol

import kotlinx.serialization.Serializable

/**
 * Messages the Android client sends upstream. Mirrors the Rust
 * `whisper_agent_protocol::ClientToServer` enum — but only the variants v1 of
 * this client actually constructs. Server still accepts any of its richer
 * variants; add more here as the UI grows.
 *
 * Serialized via [ClientToServerSerializer] (internally-tagged CBOR map)
 * rather than kotlinx-cbor's default polymorphic array encoding, which
 * ciborium on the server cannot decode.
 */
@Serializable(with = ClientToServerSerializer::class)
sealed class ClientToServer {

    data class ListThreads(
        val correlationId: String? = null,
    ) : ClientToServer()

    data class SubscribeToThread(
        val threadId: String,
    ) : ClientToServer()

    data class UnsubscribeFromThread(
        val threadId: String,
    ) : ClientToServer()

    data class SendUserMessage(
        val threadId: String,
        val text: String,
    ) : ClientToServer()

    data class ApprovalDecision(
        val threadId: String,
        val approvalId: String,
        val decision: ApprovalChoice,
        val remember: Boolean = false,
    ) : ClientToServer()
}
