package com.cjbal.whisperagent.protocol

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Mirrors `whisper_agent_protocol::permission::SudoDecision` — the three
 * resolutions a user can return for a `SudoRequested` prompt.
 *
 * Wire form is the snake-cased variant name (serde's
 * `#[serde(rename_all = "snake_case")]`), so this enum's [SerialName]
 * values must stay aligned.
 */
@Serializable
enum class SudoDecision {
    @SerialName("approve_once") ApproveOnce,
    @SerialName("approve_remember") ApproveRemember,
    @SerialName("reject") Reject,
}
