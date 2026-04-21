package com.cjbal.whisperagent.protocol

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
enum class ApprovalChoice {
    @SerialName("approve") Approve,
    @SerialName("reject") Reject,
}
