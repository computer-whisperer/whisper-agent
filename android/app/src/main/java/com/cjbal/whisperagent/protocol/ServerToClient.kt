package com.cjbal.whisperagent.protocol

import kotlinx.serialization.Serializable

/**
 * Messages the server pushes to the client. Mirrors the Rust
 * `whisper_agent_protocol::ServerToClient` enum — v1 subset of variants the
 * Android UI actually handles.
 *
 * Serialized via [ServerToClientSerializer] (internally-tagged CBOR map) to
 * match serde on the server. Unknown variants land on [Unknown] with the raw
 * `type` string, so a server that grows a new event kind doesn't crash the
 * decoder.
 */
@Serializable(with = ServerToClientSerializer::class)
sealed class ServerToClient {

    // --- Thread-list tier -----------------------------------------------------

    data class ThreadList(
        val correlationId: String? = null,
        val tasks: List<ThreadSummary>,
    ) : ServerToClient()

    data class ThreadCreated(
        val threadId: String,
        val summary: ThreadSummary,
        val correlationId: String? = null,
    ) : ServerToClient()

    data class ThreadStateChanged(
        val threadId: String,
        val state: ThreadStateLabel,
    ) : ServerToClient()

    data class ThreadTitleUpdated(
        val threadId: String,
        val title: String,
    ) : ServerToClient()

    data class ThreadArchived(
        val threadId: String,
    ) : ServerToClient()

    // --- Per-thread tier ------------------------------------------------------

    data class Snapshot(
        val threadId: String,
        val snapshot: ThreadSnapshot,
    ) : ServerToClient()

    data class ThreadUserMessage(
        val threadId: String,
        val text: String,
    ) : ServerToClient()

    data class AssistantBegin(
        val threadId: String,
        val turn: Int,
    ) : ServerToClient()

    data class AssistantTextDelta(
        val threadId: String,
        val delta: String,
    ) : ServerToClient()

    data class AssistantReasoningDelta(
        val threadId: String,
        val delta: String,
    ) : ServerToClient()

    data class AssistantEnd(
        val threadId: String,
        val stopReason: String? = null,
        val usage: Usage,
    ) : ServerToClient()

    data class ToolCallBegin(
        val threadId: String,
        val toolUseId: String,
        val name: String,
        val argsPreview: String,
        // TODO: `args: Option<serde_json::Value>` — add once CBOR-tree handling lands.
    ) : ServerToClient()

    data class ToolCallEnd(
        val threadId: String,
        val toolUseId: String,
        val resultPreview: String,
        val isError: Boolean,
    ) : ServerToClient()

    data class LoopComplete(
        val threadId: String,
    ) : ServerToClient()

    data class PendingApproval(
        val threadId: String,
        val approvalId: String,
        val toolUseId: String,
        val name: String,
        val argsPreview: String,
        val destructive: Boolean = false,
        val readOnly: Boolean = false,
    ) : ServerToClient()

    data class ApprovalResolved(
        val threadId: String,
        val approvalId: String,
        val decision: ApprovalChoice,
    ) : ServerToClient()

    // --- Pod-registry tier ----------------------------------------------------

    data class PodList(
        val correlationId: String? = null,
        val pods: List<PodSummary>,
        /** Pod the server routes `CreateThread { pod_id: None }` to. */
        val defaultPodId: String,
    ) : ServerToClient()

    data class PodCreated(
        val pod: PodSummary,
        val correlationId: String? = null,
    ) : ServerToClient()

    data class PodArchived(
        val podId: String,
    ) : ServerToClient()

    // --- Errors ---------------------------------------------------------------

    data class Error(
        val correlationId: String? = null,
        val threadId: String? = null,
        val message: String,
    ) : ServerToClient()

    /**
     * Catch-all for variants the Rust enum has but the Kotlin mirror doesn't
     * know about yet (pod / behavior / resource / function tiers, etc.).
     * Carries the wire `type` discriminator so a debug view can at least
     * label it.
     */
    data class Unknown(val type: String) : ServerToClient()
}
