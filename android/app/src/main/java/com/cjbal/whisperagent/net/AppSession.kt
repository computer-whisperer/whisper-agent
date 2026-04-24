package com.cjbal.whisperagent.net

import android.util.Log
import com.cjbal.whisperagent.auth.ServerConfig
import com.cjbal.whisperagent.auth.SettingsRepository
import com.cjbal.whisperagent.protocol.BackendSummary
import com.cjbal.whisperagent.protocol.ClientToServer
import com.cjbal.whisperagent.protocol.ContentBlock
import com.cjbal.whisperagent.protocol.Message
import com.cjbal.whisperagent.protocol.ModelSummary
import com.cjbal.whisperagent.protocol.PodSummary
import com.cjbal.whisperagent.protocol.Role
import com.cjbal.whisperagent.protocol.ServerToClient
import com.cjbal.whisperagent.protocol.SudoDecision
import com.cjbal.whisperagent.protocol.ThreadBindingsRequest
import com.cjbal.whisperagent.protocol.ThreadConfigOverride
import com.cjbal.whisperagent.protocol.ThreadSnapshot
import com.cjbal.whisperagent.protocol.ThreadSummary
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
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

    private val _pods = MutableStateFlow<Map<String, PodSummary>>(emptyMap())
    val pods: StateFlow<Map<String, PodSummary>> = _pods.asStateFlow()

    /** Server-advertised default pod — destination for `CreateThread { pod_id: None }`. */
    private val _defaultPodId = MutableStateFlow("")
    val defaultPodId: StateFlow<String> = _defaultPodId.asStateFlow()

    /**
     * Pod the UI is currently focused on. Null until the first [ServerToClient.PodList]
     * arrives; afterwards never null as long as the server advertises any pods
     * (we fall back to [defaultPodId] if the remembered selection has been
     * archived).
     */
    private val _selectedPodId = MutableStateFlow<String?>(null)
    val selectedPodId: StateFlow<String?> = _selectedPodId.asStateFlow()

    /**
     * User's explicit preference, persisted across sessions. Kept separate from
     * [_selectedPodId] so a server whose pod catalog temporarily omits the
     * preferred pod doesn't overwrite storage — next time it reappears we
     * return to it.
     */
    private var preferredPodId: String? = settings.lastPodId()

    private val _backends = MutableStateFlow<List<BackendSummary>>(emptyList())
    val backends: StateFlow<List<BackendSummary>> = _backends.asStateFlow()

    /** Server-advertised default backend. Empty until a [BackendsList] arrives. */
    private val _defaultBackend = MutableStateFlow("")
    val defaultBackend: StateFlow<String> = _defaultBackend.asStateFlow()

    /**
     * Model catalog by backend name. Populated lazily — the server round-trips
     * to the backend for each [ClientToServer.ListModels] request, so we only
     * ask for backends the UI is about to show.
     */
    private val _modelsByBackend = MutableStateFlow<Map<String, List<ModelSummary>>>(emptyMap())
    val modelsByBackend: StateFlow<Map<String, List<ModelSummary>>> = _modelsByBackend.asStateFlow()

    private val modelsInFlight = mutableSetOf<String>()

    /**
     * One-shot signal used by the UI to navigate to a freshly-created
     * thread. Emits the new thread id when the matching `ThreadCreated`
     * arrives for a `CreateThread` we issued.
     */
    private val _pendingNavigation = MutableSharedFlow<String>(extraBufferCapacity = 1)
    val pendingNavigation: SharedFlow<String> = _pendingNavigation.asSharedFlow()

    /**
     * Server-sent error strings, fanned out to any UI snackbar host. Buffered
     * so an error arriving while no subscriber is active still gets shown on
     * the next screen composition.
     */
    private val _errors = MutableSharedFlow<String>(extraBufferCapacity = 16)
    val errors: SharedFlow<String> = _errors.asSharedFlow()

    /**
     * Outstanding sudo prompts keyed by thread id. We only model one
     * outstanding prompt per thread — the server serializes sudo Functions
     * within a thread anyway. Cleared on [SudoResolved], thread unsubscribe,
     * and disconnect.
     */
    private val _pendingSudo = MutableStateFlow<Map<String, PendingSudo>>(emptyMap())
    val pendingSudo: StateFlow<Map<String, PendingSudo>> = _pendingSudo.asStateFlow()

    private val correlationCounter = AtomicLong(0)
    private var pendingCreateCorrelation: String? = null

    private var pumpJob: Job? = null
    private var subscribedThreadId: String? = null
    private var foreground: Boolean = false
    private var reconnectJob: Job? = null

    init {
        startPump()
    }

    fun onAppForegrounded() {
        foreground = true
        connectIfConfigured()
    }

    fun onAppBackgrounded() {
        foreground = false
        reconnectJob?.cancel()
        reconnectJob = null
        wire.disconnect()
    }

    /**
     * Call after the user saves new credentials on the settings screen.
     */
    fun restart() {
        reconnectJob?.cancel()
        reconnectJob = null
        wire.disconnect()
        if (foreground) connectIfConfigured()
    }

    fun subscribe(threadId: String) {
        if (subscribedThreadId == threadId) return
        // Tear down the previous subscription inline. We can't rely on the
        // old ThreadScreen's onDispose: NavHost composes the new destination
        // (running this subscribe) before the old one leaves composition, so
        // by the time the old onDispose fires `subscribedThreadId` already
        // points at the new thread. Dropping the sudo banner matches what
        // the server does on UnsubscribeFromThread.
        subscribedThreadId?.let { old ->
            _pendingSudo.update { it - old }
            wire.send(ClientToServer.UnsubscribeFromThread(old))
        }
        subscribedThreadId = threadId
        _activeSnapshot.value = null
        wire.send(ClientToServer.SubscribeToThread(threadId))
    }

    fun unsubscribe(threadId: String) {
        // Guards the cross-dispose race: when a newer ThreadScreen has
        // already claimed `subscribedThreadId`, this call is the old
        // ThreadScreen's late onDispose and must not wipe the new state.
        if (subscribedThreadId != threadId) return
        subscribedThreadId = null
        _activeSnapshot.value = null
        _pendingSudo.update { it - threadId }
        wire.send(ClientToServer.UnsubscribeFromThread(threadId))
    }

    fun submitUserMessage(text: String) {
        val id = subscribedThreadId ?: return
        if (text.isBlank()) return
        wire.send(ClientToServer.SendUserMessage(id, text))
    }

    /**
     * Ask the server to cancel a running thread. Server flips the state to
     * `Cancelled`; we'll pick up the transition via
     * [ServerToClient.ThreadStateChanged].
     */
    fun cancelThread(threadId: String) {
        if (threadId.isBlank()) return
        wire.send(ClientToServer.CancelThread(threadId))
    }

    /**
     * Archive (hide) a thread. It stays on disk but drops off the broadcast
     * list and [ServerToClient.ThreadArchived] prunes it from [threads].
     */
    fun archiveThread(threadId: String) {
        if (threadId.isBlank()) return
        wire.send(ClientToServer.ArchiveThread(threadId))
    }

    /**
     * Resolve an outstanding sudo prompt. Clears the local pending entry so
     * the UI drops the banner immediately — the broadcast [ServerToClient.SudoResolved]
     * is authoritative but races the round-trip.
     */
    /**
     * Ask the server to compact the thread. On success the server will
     * emit `ThreadCompacted` and a `ThreadCreated` for the continuation; we
     * navigate the UI to the new thread via [pendingNavigation].
     */
    fun compactThread(threadId: String) {
        if (threadId.isBlank()) return
        val correlation = "compact-${correlationCounter.incrementAndGet()}"
        pendingCreateCorrelation = correlation
        wire.send(ClientToServer.CompactThread(threadId = threadId, correlationId = correlation))
    }

    /**
     * Promote a failed thread back to `Idle`. Ignored if the thread isn't
     * actually failed — the server rejects with `Error` in that case, which
     * our snackbar surfaces.
     */
    fun recoverThread(threadId: String) {
        if (threadId.isBlank()) return
        wire.send(ClientToServer.RecoverThread(threadId))
    }

    /**
     * Fork the currently-subscribed thread at [fromMessageIndex]. Navigation
     * to the fork rides the standard `ThreadCreated` correlation path.
     */
    fun forkThread(
        threadId: String,
        fromMessageIndex: Int,
        archiveOriginal: Boolean = true,
        resetCapabilities: Boolean = false,
    ) {
        if (threadId.isBlank()) return
        val correlation = "fork-${correlationCounter.incrementAndGet()}"
        pendingCreateCorrelation = correlation
        wire.send(
            ClientToServer.ForkThread(
                threadId = threadId,
                fromMessageIndex = fromMessageIndex,
                archiveOriginal = archiveOriginal,
                resetCapabilities = resetCapabilities,
                correlationId = correlation,
            ),
        )
    }

    fun resolveSudo(threadId: String, decision: SudoDecision, reason: String? = null) {
        val pending = _pendingSudo.value[threadId] ?: return
        wire.send(
            ClientToServer.ResolveSudo(
                functionId = pending.functionId,
                decision = decision,
                reason = reason?.takeIf { it.isNotBlank() },
            ),
        )
        _pendingSudo.update { it - threadId }
    }

    /**
     * Ask the server to spawn a new thread in the currently-selected pod
     * (or the default pod if no selection yet). [backendOverride] and
     * [modelOverride] each default to `null`, meaning "inherit from the pod's
     * `thread_defaults`"; setting either builds the matching
     * [ThreadBindingsRequest] / [ThreadConfigOverride] for the wire. The
     * matching `ThreadCreated` event will fire [pendingNavigation] with the
     * new thread id so the UI can switch to it.
     */
    fun createThread(
        initialMessage: String,
        backendOverride: String? = null,
        modelOverride: String? = null,
    ) {
        if (initialMessage.isBlank()) return
        val correlation = "create-${correlationCounter.incrementAndGet()}"
        pendingCreateCorrelation = correlation
        // Prefer the user's current focus; fall back to the server default
        // implicitly by sending pod_id = null when we haven't received a
        // PodList yet.
        val targetPodId = _selectedPodId.value?.takeIf { it.isNotBlank() }
        val bindings = backendOverride?.takeIf { it.isNotBlank() }
            ?.let { ThreadBindingsRequest(backend = it) }
        val config = modelOverride?.takeIf { it.isNotBlank() }
            ?.let { ThreadConfigOverride(model = it) }
        wire.send(
            ClientToServer.CreateThread(
                initialMessage = initialMessage,
                correlationId = correlation,
                podId = targetPodId,
                configOverride = config,
                bindingsRequest = bindings,
            ),
        )
    }

    /**
     * Ask the server for the model list advertised by [backend]. Responses
     * arrive as [ServerToClient.ModelsList] and populate [modelsByBackend].
     * No-op if a request is already in flight for the same backend or if
     * we already have a cached list.
     */
    fun fetchModels(backend: String) {
        if (backend.isBlank()) return
        if (_modelsByBackend.value.containsKey(backend)) return
        if (!modelsInFlight.add(backend)) return
        wire.send(ClientToServer.ListModels(backend = backend))
    }

    /**
     * Switch the UI's current pod. Persists the preference so the same pod
     * is restored on next launch. Ignored if [podId] isn't in the pod
     * catalog — callers should drive the UI from [pods].
     */
    fun selectPod(podId: String) {
        if (_pods.value[podId] == null) return
        preferredPodId = podId
        _selectedPodId.value = podId
        settings.saveLastPodId(podId)
    }

    private fun connectIfConfigured() {
        val cfg: ServerConfig = settings.current() ?: return
        wire.connect(cfg.url, cfg.token)
    }

    private fun startPump() {
        pumpJob?.cancel()
        pumpJob = scope.launch {
            // Kick off a ListThreads + ListPods as soon as the state flips
            // to Connected. Pod-registry broadcasts (PodCreated / PodArchived)
            // keep the catalog fresh after the initial snapshot.
            launch {
                wire.state.collect { s ->
                    when (s) {
                        is ConnectionState.Connected -> {
                            reconnectJob?.cancel()
                            reconnectJob = null
                            wire.send(ClientToServer.ListThreads())
                            wire.send(ClientToServer.ListPods())
                            wire.send(ClientToServer.ListBackends())
                            // Re-subscribe the screen the user was last on.
                            subscribedThreadId?.let {
                                wire.send(ClientToServer.SubscribeToThread(it))
                            }
                        }
                        ConnectionState.Disconnected,
                        is ConnectionState.Errored,
                        -> {
                            _activeSnapshot.value = null
                            // Drop any pending sudo banners — they're tied to the
                            // session we just lost. The server will re-send
                            // SudoRequested for still-live prompts on re-subscribe.
                            _pendingSudo.value = emptyMap()
                            // Drop cached model lists — they're tied to the server
                            // we just disconnected from; next connection will refetch
                            // on demand.
                            _modelsByBackend.value = emptyMap()
                            modelsInFlight.clear()
                            // Transient drop while the user is actively using the
                            // app? Re-dial with a small backoff rather than sit
                            // silent until the next foreground cycle.
                            if (foreground) scheduleReconnect()
                        }
                        ConnectionState.Connecting -> Unit
                    }
                }
            }
            wire.incoming.collect(::reduce)
        }
    }

    /**
     * Kick off a bounded exponential-backoff reconnect loop. Cheap enough to
     * run on one Job; [startPump]'s Connected branch cancels it the moment we
     * reattach.
     */
    private fun scheduleReconnect() {
        if (reconnectJob?.isActive == true) return
        reconnectJob = scope.launch {
            var delayMs = INITIAL_RECONNECT_DELAY_MS
            while (foreground && wire.state.value !is ConnectionState.Connected) {
                delay(delayMs)
                if (!foreground) return@launch
                connectIfConfigured()
                delayMs = (delayMs * 2).coerceAtMost(MAX_RECONNECT_DELAY_MS)
            }
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
            is ServerToClient.ThreadCompacted -> {
                // The matching ThreadCreated + correlation will drive navigation
                // into the continuation thread. Nothing to do here beyond
                // letting the original thread fall through — the server keeps
                // it in place, flipped to Completed, via ThreadStateChanged.
            }
            is ServerToClient.ThreadDraftUpdated -> {
                // No draft UI yet on Android — event is decoded so it stops
                // landing on the Unknown branch.
            }
            is ServerToClient.PodList -> {
                _pods.value = event.pods
                    .filter { !it.archived }
                    .associateBy { it.podId }
                _defaultPodId.value = event.defaultPodId
                _selectedPodId.value = resolveFocusPodId()
            }
            is ServerToClient.PodCreated -> {
                if (!event.pod.archived) {
                    _pods.update { it + (event.pod.podId to event.pod) }
                }
                if (_selectedPodId.value == null) {
                    _selectedPodId.value = resolveFocusPodId()
                }
            }
            is ServerToClient.PodArchived -> {
                _pods.update { it - event.podId }
                if (_selectedPodId.value == event.podId) {
                    _selectedPodId.value = resolveFocusPodId()
                }
            }
            is ServerToClient.BackendsList -> {
                _backends.value = event.backends
                _defaultBackend.value = event.defaultBackend
            }
            is ServerToClient.ModelsList -> {
                modelsInFlight.remove(event.backend)
                _modelsByBackend.update { it + (event.backend to event.models) }
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
            is ServerToClient.SudoRequested -> {
                _pendingSudo.update {
                    it + (event.threadId to PendingSudo(
                        functionId = event.functionId,
                        threadId = event.threadId,
                        toolName = event.toolName,
                        reason = event.reason,
                    ))
                }
            }
            is ServerToClient.SudoResolved -> {
                _pendingSudo.update { map ->
                    val existing = map[event.threadId] ?: return@update map
                    if (existing.functionId == event.functionId) map - event.threadId else map
                }
            }
            is ServerToClient.Error -> {
                Log.w(TAG, "server error: ${event.message}")
                _errors.tryEmit(event.message)
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
                    ContentBlock.ToolUse(
                        id = event.toolUseId,
                        name = event.name,
                        argsPreview = event.argsPreview.ifBlank { null },
                    ),
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

    /**
     * Pick the pod the UI should currently focus on, honoring the user's
     * stored preference when possible. Order: remembered pod → server default
     * → any pod in the catalog → null (server has no pods yet).
     */
    private fun resolveFocusPodId(): String? {
        val catalog = _pods.value
        if (catalog.isEmpty()) return null
        preferredPodId?.let { if (catalog.containsKey(it)) return it }
        val default = _defaultPodId.value
        if (default.isNotBlank() && catalog.containsKey(default)) return default
        return catalog.keys.firstOrNull()
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
        private const val INITIAL_RECONNECT_DELAY_MS = 1_000L
        private const val MAX_RECONNECT_DELAY_MS = 30_000L
    }
}

/**
 * UI-facing view of an outstanding sudo prompt. `functionId` is the opaque
 * correlator the server uses to match our [ClientToServer.ResolveSudo] to the
 * pending Function — AppSession round-trips it without the UI caring what's
 * inside.
 */
data class PendingSudo(
    val functionId: Long,
    val threadId: String,
    val toolName: String,
    val reason: String,
)

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
