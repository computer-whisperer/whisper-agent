package com.cjbal.whisperagent.protocol

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Mirrors `whisper_agent_protocol::ThreadConfigOverride`. Every field is
 * optional: `None` inherits the pod's `thread_defaults`, `Some` replaces
 * just that field. Only `model` is modeled on the Kotlin side for now —
 * the other fields (max_tokens / max_turns / system_prompt / compaction)
 * are rare enough in the mobile UX that adding them on demand is cheaper
 * than carrying unused wire payload.
 *
 * `null` defaults rely on kotlinx-serialization's default
 * `encodeDefaults = false` to keep the emitted CBOR matching serde's
 * `skip_serializing_if = Option::is_none`.
 */
@Serializable
data class ThreadConfigOverride(
    val participants: ThreadParticipants? = null,
    val model: String? = null,
    @SerialName("participant_profiles")
    val participantProfiles: Map<String, ParticipantExecutionProfileRequest>? = null,
)

/** Mobile's currently-supported subset of a participant profile request. */
@Serializable
data class ParticipantExecutionProfileRequest(
    val model: String? = null,
    @SerialName("max_tokens") val maxTokens: Int? = null,
    val bindings: ThreadBindingsRequest = ThreadBindingsRequest(),
)

/**
 * Mirrors `whisper_agent_protocol::ThreadBindingsRequest`. `backend` is
 * the only field the mobile new-thread composer currently surfaces; the
 * others (host_env / mcp_hosts) stay inherited from the pod's
 * `thread_defaults`. See note on [ThreadConfigOverride] for why.
 */
@Serializable
data class ThreadBindingsRequest(
    val backend: String? = null,
)
