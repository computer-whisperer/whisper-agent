package com.cjbal.whisperagent.protocol

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Mirrors `whisper_agent_protocol::pod::PodSummary`. Lightweight per-pod entry
 * used by `PodList` and `PodCreated` responses — enough to label a pod in the
 * picker and badge basic state.
 */
@Serializable
data class PodSummary(
    @SerialName("pod_id") val podId: String,
    val name: String,
    val description: String? = null,
    @SerialName("created_at") val createdAt: String,
    @SerialName("thread_count") val threadCount: Int = 0,
    val archived: Boolean = false,
    @SerialName("behaviors_enabled") val behaviorsEnabled: Boolean = true,
)
