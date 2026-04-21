package com.cjbal.whisperagent.net

sealed class ConnectionState {
    data object Disconnected : ConnectionState()
    data object Connecting : ConnectionState()
    data object Connected : ConnectionState()
    data class Errored(val detail: String) : ConnectionState()
}
