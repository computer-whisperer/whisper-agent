package com.cjbal.whisperagent.protocol

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Provenance stamp echoed onto every thread a behavior spawned. Absent on
 * interactive threads. Mirrors `whisper_agent_protocol::BehaviorOrigin` — we
 * only decode it (the Android client never constructs one), so the
 * `trigger_payload` JSON is skipped by `ignoreUnknownKeys` rather than
 * modeled as a type.
 */
@Serializable
data class BehaviorOrigin(
    @SerialName("behavior_id") val behaviorId: String,
    @SerialName("fired_at") val firedAt: String,
)

/**
 * Minimal mirror of `whisper_agent_protocol::BehaviorSummary`. The wire
 * carries a few more fields (`description`, `pod_id`) that we fold in, plus
 * `load_error` which the UI surfaces as an error badge. Fields we don't
 * render are omitted and skipped by `ignoreUnknownKeys` at decode time.
 */
@Serializable
data class BehaviorSummary(
    @SerialName("behavior_id") val behaviorId: String,
    @SerialName("pod_id") val podId: String,
    val name: String,
    val description: String? = null,
    /** "manual" | "cron" | "webhook" | null (when config failed to load). */
    @SerialName("trigger_kind") val triggerKind: String? = null,
    val enabled: Boolean = true,
    @SerialName("run_count") val runCount: Long = 0,
    @SerialName("last_fired_at") val lastFiredAt: String? = null,
    @SerialName("load_error") val loadError: String? = null,
)

/**
 * Persisted trigger state. Only the three fields the Android UI surfaces are
 * modeled; `last_thread_id`, `last_outcome`, `queued_payload` fall through
 * `ignoreUnknownKeys`. Used solely for decoding `BehaviorStateChanged`
 * broadcasts so the cached `BehaviorSummary` can stay fresh.
 */
@Serializable
data class BehaviorState(
    val enabled: Boolean = true,
    @SerialName("run_count") val runCount: Long = 0,
    @SerialName("last_fired_at") val lastFiredAt: String? = null,
)
