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
 * Mirrors `whisper_agent_protocol::pod::CompactionConfig`. Every field is
 * always present on the wire (serde default-serializes non-Option fields),
 * so we declare them all with defaults matching the Rust side to tolerate
 * missing fields on legacy snapshots.
 */
@Serializable
data class CompactionConfig(
    val enabled: Boolean = true,
    @SerialName("prompt_file") val promptFile: String = "",
    @SerialName("summary_regex") val summaryRegex: String = "",
    @SerialName("token_threshold") val tokenThreshold: Int? = null,
    @SerialName("continuation_template") val continuationTemplate: String = "",
)

@Serializable
data class ThreadConfig(
    val model: String = "",
    @SerialName("max_tokens") val maxTokens: Int = 0,
    @SerialName("max_turns") val maxTurns: Int = 0,
    val compaction: CompactionConfig = CompactionConfig(),
)

/**
 * Mirrors `whisper_agent_protocol::HostEnvBinding` — internally tagged by
 * `kind` on the Rust side. For v1 we only care about the discriminator and
 * enough context to render a label; the `Inline` variant's nested `spec`
 * carries a `HostEnvSpec` we don't model yet and rely on `ignoreUnknownKeys`
 * to skip.
 */
@Serializable
data class HostEnvBinding(
    /** "named" or "inline" — matches the Rust enum's `#[serde(tag = "kind")]`. */
    val kind: String,
    /** Present on the "named" variant — references a `[[allow.host_env]]` entry. */
    val name: String? = null,
    /** Present on the "inline" variant — host-env provider name. */
    val provider: String? = null,
)

/**
 * Mirrors `whisper_agent_protocol::ThreadBindings`. We don't use these fields
 * in v1 UI, but the server always emits them in the snapshot so we decode
 * them to keep the outer decoder aligned.
 */
@Serializable
data class ThreadBindings(
    val backend: String = "",
    @SerialName("host_env") val hostEnv: List<HostEnvBinding> = emptyList(),
    @SerialName("mcp_hosts") val mcpHosts: List<String> = emptyList(),
    @SerialName("tool_filter") val toolFilter: List<String>? = null,
)

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
    @SerialName("continued_from") val continuedFrom: String? = null,
    @SerialName("dispatched_by") val dispatchedBy: String? = null,
    // origin: BehaviorOrigin? is omitted from v1 — ignoreUnknownKeys skips it.
)
