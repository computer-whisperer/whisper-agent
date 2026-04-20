package com.cjbal.whisperagent.net

import android.util.Log
import com.cjbal.whisperagent.protocol.ClientToServer
import com.cjbal.whisperagent.protocol.Codec
import com.cjbal.whisperagent.protocol.ServerToClient
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import java.util.concurrent.TimeUnit

/**
 * Thin wrapper over OkHttp's WebSocket. Owns one socket at a time; the caller
 * (`AppSession`) drives connect / disconnect in response to lifecycle events.
 *
 * Transports CBOR-encoded [ClientToServer] / [ServerToClient] frames and
 * surfaces transport state via [state] plus decoded events via [incoming].
 */
class WireClient {

    private val http: OkHttpClient = OkHttpClient.Builder()
        .pingInterval(20, TimeUnit.SECONDS)
        .build()

    private var socket: WebSocket? = null

    private val _state = MutableStateFlow<ConnectionState>(ConnectionState.Disconnected)
    val state: StateFlow<ConnectionState> = _state.asStateFlow()

    private val _incoming = MutableSharedFlow<ServerToClient>(extraBufferCapacity = 128)
    val incoming: SharedFlow<ServerToClient> = _incoming.asSharedFlow()

    fun connect(serverBaseUrl: String, token: String) {
        disconnect()
        val wsUrl = deriveWsUrl(serverBaseUrl)
        val request = Request.Builder()
            .url(wsUrl)
            .header("Authorization", "Bearer $token")
            .build()
        _state.value = ConnectionState.Connecting
        socket = http.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                _state.value = ConnectionState.Connected
            }

            override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
                val decoded = try {
                    Codec.decodeFromServer(bytes.toByteArray())
                } catch (t: Throwable) {
                    Log.w(TAG, "decode failed", t)
                    return
                }
                _incoming.tryEmit(decoded)
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                // Server only sends binary; text frames are unexpected.
                Log.w(TAG, "ignoring text frame: $text")
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.w(TAG, "ws failure", t)
                _state.value = ConnectionState.Errored(t.message ?: t.javaClass.simpleName)
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                webSocket.close(code, reason)
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                _state.value = ConnectionState.Disconnected
            }
        })
    }

    fun send(msg: ClientToServer): Boolean {
        val ws = socket ?: return false
        val bytes = try {
            Codec.encodeToServer(msg)
        } catch (t: Throwable) {
            Log.w(TAG, "encode failed", t)
            return false
        }
        return ws.send(ByteString.of(*bytes))
    }

    fun disconnect() {
        socket?.close(NORMAL_CLOSURE, null)
        socket = null
        if (_state.value != ConnectionState.Disconnected) {
            _state.value = ConnectionState.Disconnected
        }
    }

    companion object {
        private const val TAG = "WireClient"
        private const val NORMAL_CLOSURE = 1000

        /**
         * `https://host[:port]` → `wss://host[:port]/ws`; `http://` → `ws://`.
         * Mirrors `derive_ws_url` in `whisper-agent-desktop/src/main.rs`.
         */
        internal fun deriveWsUrl(base: String): String {
            val trimmed = base.trim().trimEnd('/')
            val wsScheme = when {
                trimmed.startsWith("https://") -> "wss://" to trimmed.removePrefix("https://")
                trimmed.startsWith("http://") -> "ws://" to trimmed.removePrefix("http://")
                trimmed.startsWith("wss://") -> "wss://" to trimmed.removePrefix("wss://")
                trimmed.startsWith("ws://") -> "ws://" to trimmed.removePrefix("ws://")
                else -> error("unsupported scheme in server URL: $base")
            }
            return "${wsScheme.first}${wsScheme.second}/ws"
        }
    }
}
