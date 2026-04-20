package com.cjbal.whisperagent.net

import android.util.Log
import com.cjbal.whisperagent.auth.ServerConfig
import com.cjbal.whisperagent.auth.SettingsRepository
import com.cjbal.whisperagent.protocol.ClientToServer
import com.cjbal.whisperagent.protocol.ServerToClient
import com.cjbal.whisperagent.protocol.ThreadSnapshot
import com.cjbal.whisperagent.protocol.ThreadSummary
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * Application-wide facade over the wire. Reduces incoming [ServerToClient]
 * events into the state flows the UI subscribes to (threads map + active
 * snapshot). Holds the single [WireClient] for the process and drives its
 * connect/disconnect in response to lifecycle events from [WhisperAgentApp].
 *
 * Deliberately *minimal* for v1: no reconnect backoff, no offline queue. Add
 * those after the proof-of-concept confirms the wire decodes cleanly.
 */
class AppSession(
    private val scope: CoroutineScope,
    private val settings: SettingsRepository,
) {
    private val wire = WireClient()

    val connection: StateFlow<ConnectionState> get() = wire.state

    private val _threads = MutableStateFlow<Map<String, ThreadSummary>>(emptyMap())
    val threads: StateFlow<Map<String, ThreadSummary>> = _threads.asStateFlow()

    private val _activeSnapshot = MutableStateFlow<ThreadSnapshot?>(null)
    val activeSnapshot: StateFlow<ThreadSnapshot?> = _activeSnapshot.asStateFlow()

    private var pumpJob: Job? = null
    private var subscribedThreadId: String? = null
    private var foreground: Boolean = false

    init {
        startPump()
    }

    fun onAppForegrounded() {
        foreground = true
        connectIfConfigured()
    }

    fun onAppBackgrounded() {
        foreground = false
        wire.disconnect()
    }

    /**
     * Call after the user saves new credentials on the settings screen.
     */
    fun restart() {
        wire.disconnect()
        if (foreground) connectIfConfigured()
    }

    fun subscribe(threadId: String) {
        subscribedThreadId = threadId
        _activeSnapshot.value = null
        wire.send(ClientToServer.SubscribeToThread(threadId))
    }

    fun unsubscribe() {
        val id = subscribedThreadId ?: return
        subscribedThreadId = null
        _activeSnapshot.value = null
        wire.send(ClientToServer.UnsubscribeFromThread(id))
    }

    fun submitUserMessage(text: String) {
        val id = subscribedThreadId ?: return
        if (text.isBlank()) return
        wire.send(ClientToServer.SendUserMessage(id, text))
    }

    private fun connectIfConfigured() {
        val cfg: ServerConfig = settings.current() ?: return
        wire.connect(cfg.url, cfg.token)
        // Pull the thread list as soon as the socket is up. Server replies with
        // `ThreadList`, which reducer below folds into [_threads].
        scope.launch {
            // We can't race the open here — the send will no-op until the socket
            // transitions to Connected. The pump listens for that transition
            // and will fire the request instead.
        }
    }

    private fun startPump() {
        pumpJob?.cancel()
        pumpJob = scope.launch {
            // Kick off a ListThreads as soon as the state flips to Connected.
            launch {
                wire.state.collect { s ->
                    if (s is ConnectionState.Connected) {
                        wire.send(ClientToServer.ListThreads())
                        // Re-subscribe the screen the user was last on.
                        subscribedThreadId?.let {
                            wire.send(ClientToServer.SubscribeToThread(it))
                        }
                    }
                    if (s is ConnectionState.Disconnected) {
                        _activeSnapshot.value = null
                    }
                }
            }
            wire.incoming.collect(::reduce)
        }
    }

    private fun reduce(event: ServerToClient) {
        when (event) {
            is ServerToClient.ThreadList -> {
                _threads.value = event.tasks.associateBy { it.threadId }
            }
            is ServerToClient.ThreadCreated -> {
                _threads.update { it + (event.summary.threadId to event.summary) }
            }
            is ServerToClient.ThreadStateChanged -> {
                _threads.update { map ->
                    val existing = map[event.threadId] ?: return@update map
                    map + (event.threadId to existing.copy(state = event.state))
                }
                _activeSnapshot.update { snap ->
                    when {
                        snap == null -> snap
                        snap.threadId == event.threadId -> snap.copy(state = event.state)
                        else -> snap
                    }
                }
            }
            is ServerToClient.ThreadTitleUpdated -> {
                _threads.update { map ->
                    val existing = map[event.threadId] ?: return@update map
                    map + (event.threadId to existing.copy(title = event.title))
                }
            }
            is ServerToClient.ThreadArchived -> {
                _threads.update { it - event.threadId }
            }
            is ServerToClient.Snapshot -> {
                if (event.threadId == subscribedThreadId) {
                    _activeSnapshot.value = event.snapshot
                }
            }
            is ServerToClient.ThreadUserMessage,
            is ServerToClient.AssistantBegin,
            is ServerToClient.AssistantTextDelta,
            is ServerToClient.AssistantReasoningDelta,
            is ServerToClient.AssistantEnd,
            is ServerToClient.ToolCallBegin,
            is ServerToClient.ToolCallEnd,
            is ServerToClient.LoopComplete,
            is ServerToClient.PendingApproval,
            is ServerToClient.ApprovalResolved,
            -> {
                // TODO: streaming reducer — fold deltas into [_activeSnapshot]
                //       so the thread view updates live.
            }
            is ServerToClient.Error -> {
                Log.w(TAG, "server error: ${event.message}")
            }
            is ServerToClient.Unknown -> {
                Log.d(TAG, "unhandled variant: ${event.type}")
            }
        }
    }

    companion object {
        private const val TAG = "AppSession"
    }
}
