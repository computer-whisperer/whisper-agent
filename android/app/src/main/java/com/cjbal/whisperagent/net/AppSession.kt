package com.cjbal.whisperagent.net

import android.util.Log
import com.cjbal.whisperagent.auth.ServerConfig
import com.cjbal.whisperagent.auth.SettingsRepository
import com.cjbal.whisperagent.protocol.ClientToServer
import com.cjbal.whisperagent.protocol.ContentBlock
import com.cjbal.whisperagent.protocol.Message
import com.cjbal.whisperagent.protocol.Role
import com.cjbal.whisperagent.protocol.ServerToClient
import com.cjbal.whisperagent.protocol.ThreadSnapshot
import com.cjbal.whisperagent.protocol.ThreadSummary
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicLong

/**
 * Application-wide facade over the wire. Reduces incoming [ServerToClient]
 * events into the state flows the UI subscribes to (threads map + active
 * snapshot). Holds the single [WireClient] for the process and drives its
 * connect/disconnect in response to lifecycle events from [WhisperAgentApp].
 *
 * Streaming events from the server (assistant text deltas, tool call
 * begin/end, etc.) are folded into [activeSnapshot] in [reduceStreaming]
 * so the thread UI updates live without re-fetching the full snapshot.
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

    /**
     * One-shot signal used by the UI to navigate to a freshly-created
     * thread. Emits the new thread id when the matching `ThreadCreated`
     * arrives for a `CreateThread` we issued.
     */
    private val _pendingNavigation = MutableSharedFlow<String>(extraBufferCapacity = 1)
    val pendingNavigation: SharedFlow<String> = _pendingNavigation.asSharedFlow()

    private val correlationCounter = AtomicLong(0)
    private var pendingCreateCorrelation: String? = null

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

    /**
     * Ask the server to spawn a new thread in the default pod. The matching
     * `ThreadCreated` event will fire [pendingNavigation] with the new
     * thread id so the UI can switch to it.
     */
    fun createThread(initialMessage: String) {
        if (initialMessage.isBlank()) return
        val correlation = "create-${correlationCounter.incrementAndGet()}"
        pendingCreateCorrelation = correlation
        wire.send(
            ClientToServer.CreateThread(
                initialMessage = initialMessage,
                correlationId = correlation,
            ),
        )
    }

    private fun connectIfConfigured() {
        val cfg: ServerConfig = settings.current() ?: return
        wire.connect(cfg.url, cfg.token)
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
                if (event.correlationId != null && event.correlationId == pendingCreateCorrelation) {
                    pendingCreateCorrelation = null
                    _pendingNavigation.tryEmit(event.threadId)
                }
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
            -> reduceStreaming(event)
            is ServerToClient.PendingApproval,
            is ServerToClient.ApprovalResolved,
            -> {
                // TODO: surface approval prompts in the UI.
            }
            is ServerToClient.Error -> {
                Log.w(TAG, "server error: ${event.message}")
            }
            is ServerToClient.Unknown -> {
                Log.d(TAG, "unhandled variant: ${event.type}")
            }
        }
    }

    /**
     * Streaming-event reducer. Folds per-turn events into the active
     * snapshot's conversation so the UI sees text deltas and tool calls
     * as they arrive, without waiting for a full re-snapshot.
     *
     * Only mutates state when the event's `thread_id` matches the
     * currently-subscribed thread (other threads' events are ignored —
     * the server filters this for us, but cheap to double-check).
     */
    private fun reduceStreaming(event: ServerToClient) {
        val eventThreadId = streamingEventThreadId(event) ?: return
        if (eventThreadId != subscribedThreadId) return
        _activeSnapshot.update { snap ->
            if (snap == null) return@update null
            when (event) {
                is ServerToClient.ThreadUserMessage -> snap.appendMessage(
                    Message(role = Role.User, content = listOf(ContentBlock.Text(event.text))),
                )
                is ServerToClient.AssistantBegin -> snap.appendMessage(
                    Message(role = Role.Assistant, content = emptyList()),
                )
                is ServerToClient.AssistantTextDelta -> snap.appendDeltaToLastAssistant(
                    asTextDelta = event.delta,
                )
                is ServerToClient.AssistantReasoningDelta -> snap.appendDeltaToLastAssistant(
                    asThinkingDelta = event.delta,
                )
                is ServerToClient.ToolCallBegin -> snap.appendBlockToLastAssistant(
                    ContentBlock.ToolUse(id = event.toolUseId, name = event.name),
                )
                is ServerToClient.ToolCallEnd -> snap.appendToolResult(
                    ContentBlock.ToolResult(
                        toolUseId = event.toolUseId,
                        isError = event.isError,
                        previewText = event.resultPreview.ifBlank { null },
                    ),
                )
                is ServerToClient.AssistantEnd,
                is ServerToClient.LoopComplete,
                -> snap // No conversation change; usage/state updates are handled elsewhere.
                else -> snap
            }
        }
    }

    private fun streamingEventThreadId(event: ServerToClient): String? = when (event) {
        is ServerToClient.ThreadUserMessage -> event.threadId
        is ServerToClient.AssistantBegin -> event.threadId
        is ServerToClient.AssistantTextDelta -> event.threadId
        is ServerToClient.AssistantReasoningDelta -> event.threadId
        is ServerToClient.AssistantEnd -> event.threadId
        is ServerToClient.ToolCallBegin -> event.threadId
        is ServerToClient.ToolCallEnd -> event.threadId
        is ServerToClient.LoopComplete -> event.threadId
        else -> null
    }

    companion object {
        private const val TAG = "AppSession"
    }
}

// --- Snapshot mutation helpers --------------------------------------------------
//
// All conversation rewrites copy through ThreadSnapshot.copy(...) so existing
// state observers see a fresh value (data-class equality drives Compose
// recomposition). Helpers favor readability over allocation efficiency — typical
// turn produces a few dozen events, not thousands.

private fun ThreadSnapshot.appendMessage(message: Message): ThreadSnapshot =
    copy(conversation = conversation + message)

/** Append `block` to the last Assistant message's content, creating one if absent. */
private fun ThreadSnapshot.appendBlockToLastAssistant(block: ContentBlock): ThreadSnapshot {
    val msgs = conversation.toMutableList()
    val lastIdx = msgs.indexOfLast { it.role == Role.Assistant }
    if (lastIdx < 0) {
        msgs.add(Message(role = Role.Assistant, content = listOf(block)))
    } else {
        val msg = msgs[lastIdx]
        msgs[lastIdx] = msg.copy(content = msg.content + block)
    }
    return copy(conversation = msgs)
}

/**
 * Apply a streaming delta (text or thinking) to the last assistant message.
 * Coalesces consecutive same-kind deltas into the trailing block; otherwise
 * starts a new block of the right kind.
 */
private fun ThreadSnapshot.appendDeltaToLastAssistant(
    asTextDelta: String? = null,
    asThinkingDelta: String? = null,
): ThreadSnapshot {
    val msgs = conversation.toMutableList()
    val lastIdx = msgs.indexOfLast { it.role == Role.Assistant }
    val (msg, msgIdx) = if (lastIdx < 0) {
        Pair(Message(role = Role.Assistant, content = emptyList()), msgs.size)
    } else {
        Pair(msgs[lastIdx], lastIdx)
    }
    val content = msg.content.toMutableList()
    val last = content.lastOrNull()

    when {
        asTextDelta != null -> {
            if (last is ContentBlock.Text) {
                content[content.size - 1] = ContentBlock.Text(last.text + asTextDelta)
            } else {
                content.add(ContentBlock.Text(asTextDelta))
            }
        }
        asThinkingDelta != null -> {
            if (last is ContentBlock.Thinking) {
                content[content.size - 1] = ContentBlock.Thinking(last.thinking + asThinkingDelta)
            } else {
                content.add(ContentBlock.Thinking(asThinkingDelta))
            }
        }
    }

    val newMsg = msg.copy(content = content)
    if (msgIdx == msgs.size) msgs.add(newMsg) else msgs[msgIdx] = newMsg
    return copy(conversation = msgs)
}

/**
 * Drop a ToolResult block into the conversation. If the trailing message is
 * already a Role.ToolResult message (i.e. an earlier tool result this turn
 * already opened it), append the block; otherwise spin up a new ToolResult
 * message.
 */
private fun ThreadSnapshot.appendToolResult(block: ContentBlock.ToolResult): ThreadSnapshot {
    val msgs = conversation.toMutableList()
    val last = msgs.lastOrNull()
    if (last != null && last.role == Role.ToolResult) {
        msgs[msgs.size - 1] = last.copy(content = last.content + block)
    } else {
        msgs.add(Message(role = Role.ToolResult, content = listOf(block)))
    }
    return copy(conversation = msgs)
}
