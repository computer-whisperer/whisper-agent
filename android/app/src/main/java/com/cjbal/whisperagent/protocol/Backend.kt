package com.cjbal.whisperagent.protocol

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Mirrors `whisper_agent_protocol::BackendSummary`. One entry per catalog
 * backend (`[backends.<name>]` in server config). `defaultModel` is what
 * the server falls back to if a thread's `ThreadConfig.model` is empty; the
 * UI can either surface it as a label or drive model-picker defaults.
 */
@Serializable
data class BackendSummary(
    val name: String,
    val kind: String,
    @SerialName("default_model") val defaultModel: String? = null,
    @SerialName("auth_mode") val authMode: String? = null,
)

/**
 * Mirrors `whisper_agent_protocol::ModelSummary` — an entry in a
 * `ModelsList` response. `displayName` is optional human-readable label;
 * `id` is the wire value the thread's `model` field takes.
 */
@Serializable
data class ModelSummary(
    val id: String,
    @SerialName("display_name") val displayName: String? = null,
)
