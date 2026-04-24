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

    /**
     * `CompactThread` finished: the scheduler spawned [newThreadId] seeded
     * with the extracted summary. AppSession emits a navigation hint so the
     * UI can jump to the continuation thread.
     */
    data class ThreadCompacted(
        val threadId: String,
        val newThreadId: String,
        val summaryText: String,
        val correlationId: String? = null,
    ) : ServerToClient()

    /**
     * Broadcast when the persisted draft for a thread changes. Fanout
     * excludes the client that issued the `SetThreadDraft`. The Android
     * client doesn't send drafts today; we still decode this event so it
     * stops landing as [Unknown].
     */
    data class ThreadDraftUpdated(
        val threadId: String,
        val text: String,
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

    // --- Model-catalog responses ---------------------------------------------

    data class BackendsList(
        val correlationId: String? = null,
        val defaultBackend: String,
        val backends: List<BackendSummary>,
    ) : ServerToClient()

    data class ModelsList(
        val correlationId: String? = null,
        val backend: String,
        val models: List<ModelSummary>,
    ) : ServerToClient()

    // --- Sudo / escalation ----------------------------------------------------

    /**
     * A tool that fell outside the thread's scope wants an escalation decision.
     * Persists until the user resolves it with [ClientToServer.ResolveSudo].
     *
     * `args` is `serde_json::Value` on the server — we don't decode it (no
     * CBOR-tree type available); the decoder's `ignoreUnknownKeys = true`
     * skips it silently. UI shows [toolName] + [reason]; for a richer preview
     * the model's own `ToolCallBegin.argsPreview` is usually already in the
     * conversation.
     */
    data class SudoRequested(
        val functionId: Long,
        val threadId: String,
        val toolName: String,
        val reason: String,
    ) : ServerToClient()

    /**
     * Broadcast after any subscriber resolves a matching [SudoRequested].
     * Emitted to every subscriber, not just the approver, so observers reflect
     * the decision and clear any local banner.
     */
    data class SudoResolved(
        val functionId: Long,
        val threadId: String,
        val decision: SudoDecision,
    ) : ServerToClient()

    // --- Behaviors ------------------------------------------------------------

    /**
     * Reply to [ClientToServer.ListBehaviors] — every behavior the pod
     * currently holds, loaded or load-errored.
     */
    data class BehaviorList(
        val podId: String,
        val behaviors: List<BehaviorSummary>,
        val correlationId: String? = null,
    ) : ServerToClient()

    /**
     * A fresh behavior was authored in the pod (webui or API). Broadcast so
     * every client folds it into the behaviors sidebar without a refetch.
     */
    data class BehaviorCreated(
        val summary: BehaviorSummary,
        val correlationId: String? = null,
    ) : ServerToClient()

    /**
     * An existing behavior was removed. Spawned threads keep their
     * `origin.behavior_id`, so the UI surfaces them under a "Deleted
     * behaviors" bucket instead of under the (now-gone) behavior row.
     */
    data class BehaviorDeleted(
        val podId: String,
        val behaviorId: String,
        val correlationId: String? = null,
    ) : ServerToClient()

    /**
     * A behavior's persisted state changed — a run fired, a pause toggled,
     * last_fired_at advanced. We fold the three fields we actually render
     * (enabled, run_count, last_fired_at) into the cached [BehaviorSummary];
     * other state fields arrive on the wire but are ignored.
     */
    data class BehaviorStateChanged(
        val podId: String,
        val behaviorId: String,
        val state: BehaviorState,
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
