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

    data class CreateThread(
        val initialMessage: String,
        val correlationId: String? = null,
        val podId: String? = null,
        val configOverride: ThreadConfigOverride? = null,
        val bindingsRequest: ThreadBindingsRequest? = null,
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

    data class ListPods(
        val correlationId: String? = null,
    ) : ClientToServer()

    data class ListBackends(
        val correlationId: String? = null,
    ) : ClientToServer()

    data class ListModels(
        val backend: String,
        val correlationId: String? = null,
    ) : ClientToServer()

    data class CancelThread(
        val threadId: String,
    ) : ClientToServer()

    data class ArchiveThread(
        val threadId: String,
    ) : ClientToServer()

    /**
     * Reply to a [ServerToClient.SudoRequested] prompt. `functionId` is
     * the opaque u64 correlator the server assigned to the sudo Function
     * — pulled straight from the matching `SudoRequested` event.
     *
     * Wire-encoded as CBOR unsigned int (major type 0). Kotlin's [Long] is
     * signed, but sudo function ids are a small monotonic counter so the
     * value never overflows the positive range.
     */
    data class ResolveSudo(
        val functionId: Long,
        val decision: SudoDecision,
        val reason: String? = null,
    ) : ClientToServer()

    /**
     * Summarize the conversation into a new thread seeded with that
     * summary. Server rejects mid-turn or if the thread's compaction is
     * disabled; reply comes back as [ServerToClient.ThreadCompacted]
     * carrying the new thread id.
     */
    data class CompactThread(
        val threadId: String,
        val correlationId: String? = null,
    ) : ClientToServer()

    /**
     * Promote a `Failed` thread back to `Idle`. Only valid when the
     * thread is actually failed; anything else comes back as an `Error`.
     */
    data class RecoverThread(
        val threadId: String,
        val correlationId: String? = null,
    ) : ClientToServer()

    /**
     * Fork a thread at the given user-message index. The new thread's
     * conversation is the prefix `[0, fromMessageIndex)`; caller opts in
     * to archiving the source and to resetting capabilities to the pod's
     * current defaults.
     *
     * Server echoes back a `ThreadCreated` with `correlationId` matching
     * what was sent — AppSession uses this to auto-navigate to the
     * fork.
     */
    data class ForkThread(
        val threadId: String,
        val fromMessageIndex: Int,
        val archiveOriginal: Boolean = false,
        val resetCapabilities: Boolean = false,
        val correlationId: String? = null,
    ) : ClientToServer()
}
