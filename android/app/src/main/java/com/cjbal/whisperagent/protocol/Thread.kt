package com.cjbal.whisperagent.protocol

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
enum class ThreadStateLabel {
    @SerialName("idle") Idle,
    @SerialName("working") Working,
    @SerialName("completed") Completed,
    @SerialName("failed") Failed,
    @SerialName("cancelled") Cancelled,
}

@Serializable
data class Usage(
    @SerialName("input_tokens") val inputTokens: Int = 0,
    @SerialName("output_tokens") val outputTokens: Int = 0,
    @SerialName("cache_read_input_tokens") val cacheReadInputTokens: Int = 0,
    @SerialName("cache_creation_input_tokens") val cacheCreationInputTokens: Int = 0,
)

@Serializable
data class ThreadSummary(
    @SerialName("thread_id") val threadId: String,
    @SerialName("pod_id") val podId: String = "",
    val title: String? = null,
    val state: ThreadStateLabel,
    @SerialName("created_at") val createdAt: String,
    @SerialName("last_active") val lastActive: String,
    @SerialName("continued_from") val continuedFrom: String? = null,
    @SerialName("dispatched_by") val dispatchedBy: String? = null,
    // origin: BehaviorOrigin? is omitted from v1 — decoder keeps working because
    // kotlinx-cbor ignores unknown map entries by default.
)

/**
 * Skeleton. The Rust shape carries model / max_tokens / max_turns / compaction —
 * mirror as needed once the thread detail screen starts reading it.
 */
@Serializable
data class ThreadConfig(
    val model: String = "",
    @SerialName("max_tokens") val maxTokens: Int = 0,
    @SerialName("max_turns") val maxTurns: Int = 0,
)

/** Placeholder — mirror the real `ThreadBindings` when bindings surface on the UI. */
@Serializable
class ThreadBindings

@Serializable
data class TurnEntry(val usage: Usage)

@Serializable
data class TurnLog(val entries: List<TurnEntry> = emptyList())

@Serializable
data class ThreadSnapshot(
    @SerialName("thread_id") val threadId: String,
    @SerialName("pod_id") val podId: String = "",
    val title: String? = null,
    val config: ThreadConfig,
    val bindings: ThreadBindings = ThreadBindings(),
    val state: ThreadStateLabel,
    val conversation: Conversation,
    @SerialName("total_usage") val totalUsage: Usage,
    @SerialName("turn_log") val turnLog: TurnLog = TurnLog(),
    val draft: String = "",
    @SerialName("created_at") val createdAt: String,
    @SerialName("last_active") val lastActive: String,
    val failure: String? = null,
    @SerialName("tool_allowlist") val toolAllowlist: List<String> = emptyList(),
    @SerialName("continued_from") val continuedFrom: String? = null,
    @SerialName("dispatched_by") val dispatchedBy: String? = null,
)
