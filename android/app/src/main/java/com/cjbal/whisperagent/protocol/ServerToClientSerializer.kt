package com.cjbal.whisperagent.protocol

import kotlinx.serialization.KSerializer
import kotlinx.serialization.SerializationException
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.builtins.serializer
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.descriptors.buildClassSerialDescriptor
import kotlinx.serialization.encoding.CompositeDecoder
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import kotlinx.serialization.encoding.decodeStructure
import kotlinx.serialization.encoding.encodeStructure

/**
 * Internally-tagged CBOR (de)serializer for [ServerToClient]. Same rationale
 * and pattern as [ClientToServerSerializer] — see that file for the longer
 * explanation.
 *
 * Unknown variants route to [ServerToClient.Unknown] carrying the raw `type`
 * string, matching our forward-compatibility goal: the server can add new
 * events without breaking the socket.
 */
object ServerToClientSerializer : KSerializer<ServerToClient> {

    private const val IDX_TYPE = 0
    private const val IDX_CORRELATION_ID = 1
    private const val IDX_THREAD_ID = 2
    private const val IDX_TASKS = 3
    private const val IDX_SUMMARY = 4
    private const val IDX_STATE = 5
    private const val IDX_TITLE = 6
    private const val IDX_SNAPSHOT = 7
    private const val IDX_TEXT = 8
    private const val IDX_TURN = 9
    private const val IDX_DELTA = 10
    private const val IDX_STOP_REASON = 11
    private const val IDX_USAGE = 12
    private const val IDX_TOOL_USE_ID = 13
    private const val IDX_NAME = 14
    private const val IDX_ARGS_PREVIEW = 15
    private const val IDX_RESULT_PREVIEW = 16
    private const val IDX_IS_ERROR = 17
    private const val IDX_MESSAGE = 18
    private const val IDX_PODS = 19
    private const val IDX_DEFAULT_POD_ID = 20
    private const val IDX_POD = 21
    private const val IDX_POD_ID = 22
    private const val IDX_BACKENDS = 23
    private const val IDX_DEFAULT_BACKEND = 24
    private const val IDX_MODELS = 25
    private const val IDX_BACKEND = 26
    private const val IDX_FUNCTION_ID = 27
    private const val IDX_TOOL_NAME = 28
    private const val IDX_REASON = 29
    private const val IDX_DECISION = 30
    private const val IDX_NEW_THREAD_ID = 31
    private const val IDX_SUMMARY_TEXT = 32
    private const val IDX_BEHAVIORS = 33
    private const val IDX_BEHAVIOR_ID = 34

    private val tasksSerializer = ListSerializer(ThreadSummary.serializer())
    private val podsSerializer = ListSerializer(PodSummary.serializer())
    private val backendsSerializer = ListSerializer(BackendSummary.serializer())
    private val modelsSerializer = ListSerializer(ModelSummary.serializer())
    private val behaviorsSerializer = ListSerializer(BehaviorSummary.serializer())

    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("ServerToClient") {
        element("type", String.serializer().descriptor)
        element("correlation_id", String.serializer().descriptor, isOptional = true)
        element("thread_id", String.serializer().descriptor, isOptional = true)
        element("tasks", tasksSerializer.descriptor, isOptional = true)
        element("summary", ThreadSummary.serializer().descriptor, isOptional = true)
        element("state", ThreadStateLabel.serializer().descriptor, isOptional = true)
        element("title", String.serializer().descriptor, isOptional = true)
        element("snapshot", ThreadSnapshot.serializer().descriptor, isOptional = true)
        element("text", String.serializer().descriptor, isOptional = true)
        element("turn", Int.serializer().descriptor, isOptional = true)
        element("delta", String.serializer().descriptor, isOptional = true)
        element("stop_reason", String.serializer().descriptor, isOptional = true)
        element("usage", Usage.serializer().descriptor, isOptional = true)
        element("tool_use_id", String.serializer().descriptor, isOptional = true)
        element("name", String.serializer().descriptor, isOptional = true)
        element("args_preview", String.serializer().descriptor, isOptional = true)
        element("result_preview", String.serializer().descriptor, isOptional = true)
        element("is_error", Boolean.serializer().descriptor, isOptional = true)
        element("message", String.serializer().descriptor, isOptional = true)
        element("pods", podsSerializer.descriptor, isOptional = true)
        element("default_pod_id", String.serializer().descriptor, isOptional = true)
        element("pod", PodSummary.serializer().descriptor, isOptional = true)
        element("pod_id", String.serializer().descriptor, isOptional = true)
        element("backends", backendsSerializer.descriptor, isOptional = true)
        element("default_backend", String.serializer().descriptor, isOptional = true)
        element("models", modelsSerializer.descriptor, isOptional = true)
        element("backend", String.serializer().descriptor, isOptional = true)
        element("function_id", Long.serializer().descriptor, isOptional = true)
        element("tool_name", String.serializer().descriptor, isOptional = true)
        element("reason", String.serializer().descriptor, isOptional = true)
        element("decision", SudoDecision.serializer().descriptor, isOptional = true)
        element("new_thread_id", String.serializer().descriptor, isOptional = true)
        element("summary_text", String.serializer().descriptor, isOptional = true)
        element("behaviors", behaviorsSerializer.descriptor, isOptional = true)
        element("behavior_id", String.serializer().descriptor, isOptional = true)
        // Note: `summary` (IDX_SUMMARY) is shared between ThreadCreated and
        // BehaviorCreated; `state` (IDX_STATE) is shared between
        // ThreadStateChanged and BehaviorStateChanged. The decode loop
        // picks the right deserializer at read time based on the already-
        // seen `type` field — serde's internally-tagged enum emits `type`
        // before the variant's struct fields, so by the time we hit a
        // shared key the type is known.
    }

    override fun serialize(encoder: Encoder, value: ServerToClient) {
        // Encoding is the server's job — the Android client never sends a
        // ServerToClient. Supported here only for round-trip tests that verify
        // decode consistency. Unknown can't round-trip faithfully (it erases
        // its original payload) so we refuse it.
        encoder.encodeStructure(descriptor) {
            when (value) {
                is ServerToClient.ThreadList -> {
                    encodeStringElement(descriptor, IDX_TYPE, "thread_list")
                    value.correlationId?.let {
                        encodeStringElement(descriptor, IDX_CORRELATION_ID, it)
                    }
                    encodeSerializableElement(descriptor, IDX_TASKS, tasksSerializer, value.tasks)
                }
                is ServerToClient.ThreadCreated -> {
                    encodeStringElement(descriptor, IDX_TYPE, "thread_created")
                    encodeStringElement(descriptor, IDX_THREAD_ID, value.threadId)
                    encodeSerializableElement(
                        descriptor, IDX_SUMMARY, ThreadSummary.serializer(), value.summary,
                    )
                    value.correlationId?.let {
                        encodeStringElement(descriptor, IDX_CORRELATION_ID, it)
                    }
                }
                is ServerToClient.ThreadStateChanged -> {
                    encodeStringElement(descriptor, IDX_TYPE, "thread_state_changed")
                    encodeStringElement(descriptor, IDX_THREAD_ID, value.threadId)
                    encodeSerializableElement(
                        descriptor, IDX_STATE, ThreadStateLabel.serializer(), value.state,
                    )
                }
                is ServerToClient.ThreadTitleUpdated -> {
                    encodeStringElement(descriptor, IDX_TYPE, "thread_title_updated")
                    encodeStringElement(descriptor, IDX_THREAD_ID, value.threadId)
                    encodeStringElement(descriptor, IDX_TITLE, value.title)
                }
                is ServerToClient.ThreadArchived -> {
                    encodeStringElement(descriptor, IDX_TYPE, "thread_archived")
                    encodeStringElement(descriptor, IDX_THREAD_ID, value.threadId)
                }
                is ServerToClient.ThreadCompacted -> {
                    encodeStringElement(descriptor, IDX_TYPE, "thread_compacted")
                    encodeStringElement(descriptor, IDX_THREAD_ID, value.threadId)
                    encodeStringElement(descriptor, IDX_NEW_THREAD_ID, value.newThreadId)
                    encodeStringElement(descriptor, IDX_SUMMARY_TEXT, value.summaryText)
                    value.correlationId?.let {
                        encodeStringElement(descriptor, IDX_CORRELATION_ID, it)
                    }
                }
                is ServerToClient.ThreadDraftUpdated -> {
                    encodeStringElement(descriptor, IDX_TYPE, "thread_draft_updated")
                    encodeStringElement(descriptor, IDX_THREAD_ID, value.threadId)
                    encodeStringElement(descriptor, IDX_TEXT, value.text)
                }
                is ServerToClient.Snapshot -> {
                    encodeStringElement(descriptor, IDX_TYPE, "thread_snapshot")
                    encodeStringElement(descriptor, IDX_THREAD_ID, value.threadId)
                    encodeSerializableElement(
                        descriptor, IDX_SNAPSHOT, ThreadSnapshot.serializer(), value.snapshot,
                    )
                }
                is ServerToClient.ThreadUserMessage -> {
                    encodeStringElement(descriptor, IDX_TYPE, "thread_user_message")
                    encodeStringElement(descriptor, IDX_THREAD_ID, value.threadId)
                    encodeStringElement(descriptor, IDX_TEXT, value.text)
                }
                is ServerToClient.AssistantBegin -> {
                    encodeStringElement(descriptor, IDX_TYPE, "thread_assistant_begin")
                    encodeStringElement(descriptor, IDX_THREAD_ID, value.threadId)
                    encodeIntElement(descriptor, IDX_TURN, value.turn)
                }
                is ServerToClient.AssistantTextDelta -> {
                    encodeStringElement(descriptor, IDX_TYPE, "thread_assistant_text_delta")
                    encodeStringElement(descriptor, IDX_THREAD_ID, value.threadId)
                    encodeStringElement(descriptor, IDX_DELTA, value.delta)
                }
                is ServerToClient.AssistantReasoningDelta -> {
                    encodeStringElement(descriptor, IDX_TYPE, "thread_assistant_reasoning_delta")
                    encodeStringElement(descriptor, IDX_THREAD_ID, value.threadId)
                    encodeStringElement(descriptor, IDX_DELTA, value.delta)
                }
                is ServerToClient.AssistantEnd -> {
                    encodeStringElement(descriptor, IDX_TYPE, "thread_assistant_end")
                    encodeStringElement(descriptor, IDX_THREAD_ID, value.threadId)
                    value.stopReason?.let {
                        encodeStringElement(descriptor, IDX_STOP_REASON, it)
                    }
                    encodeSerializableElement(
                        descriptor, IDX_USAGE, Usage.serializer(), value.usage,
                    )
                }
                is ServerToClient.ToolCallBegin -> {
                    encodeStringElement(descriptor, IDX_TYPE, "thread_tool_call_begin")
                    encodeStringElement(descriptor, IDX_THREAD_ID, value.threadId)
                    encodeStringElement(descriptor, IDX_TOOL_USE_ID, value.toolUseId)
                    encodeStringElement(descriptor, IDX_NAME, value.name)
                    encodeStringElement(descriptor, IDX_ARGS_PREVIEW, value.argsPreview)
                }
                is ServerToClient.ToolCallEnd -> {
                    encodeStringElement(descriptor, IDX_TYPE, "thread_tool_call_end")
                    encodeStringElement(descriptor, IDX_THREAD_ID, value.threadId)
                    encodeStringElement(descriptor, IDX_TOOL_USE_ID, value.toolUseId)
                    encodeStringElement(descriptor, IDX_RESULT_PREVIEW, value.resultPreview)
                    encodeBooleanElement(descriptor, IDX_IS_ERROR, value.isError)
                }
                is ServerToClient.LoopComplete -> {
                    encodeStringElement(descriptor, IDX_TYPE, "thread_loop_complete")
                    encodeStringElement(descriptor, IDX_THREAD_ID, value.threadId)
                }
                is ServerToClient.PodList -> {
                    encodeStringElement(descriptor, IDX_TYPE, "pod_list")
                    value.correlationId?.let {
                        encodeStringElement(descriptor, IDX_CORRELATION_ID, it)
                    }
                    encodeSerializableElement(descriptor, IDX_PODS, podsSerializer, value.pods)
                    encodeStringElement(descriptor, IDX_DEFAULT_POD_ID, value.defaultPodId)
                }
                is ServerToClient.PodCreated -> {
                    encodeStringElement(descriptor, IDX_TYPE, "pod_created")
                    encodeSerializableElement(
                        descriptor, IDX_POD, PodSummary.serializer(), value.pod,
                    )
                    value.correlationId?.let {
                        encodeStringElement(descriptor, IDX_CORRELATION_ID, it)
                    }
                }
                is ServerToClient.PodArchived -> {
                    encodeStringElement(descriptor, IDX_TYPE, "pod_archived")
                    encodeStringElement(descriptor, IDX_POD_ID, value.podId)
                }
                is ServerToClient.BackendsList -> {
                    encodeStringElement(descriptor, IDX_TYPE, "backends_list")
                    value.correlationId?.let {
                        encodeStringElement(descriptor, IDX_CORRELATION_ID, it)
                    }
                    encodeStringElement(descriptor, IDX_DEFAULT_BACKEND, value.defaultBackend)
                    encodeSerializableElement(
                        descriptor, IDX_BACKENDS, backendsSerializer, value.backends,
                    )
                }
                is ServerToClient.ModelsList -> {
                    encodeStringElement(descriptor, IDX_TYPE, "models_list")
                    value.correlationId?.let {
                        encodeStringElement(descriptor, IDX_CORRELATION_ID, it)
                    }
                    encodeStringElement(descriptor, IDX_BACKEND, value.backend)
                    encodeSerializableElement(
                        descriptor, IDX_MODELS, modelsSerializer, value.models,
                    )
                }
                is ServerToClient.SudoRequested -> {
                    encodeStringElement(descriptor, IDX_TYPE, "sudo_requested")
                    encodeLongElement(descriptor, IDX_FUNCTION_ID, value.functionId)
                    encodeStringElement(descriptor, IDX_THREAD_ID, value.threadId)
                    encodeStringElement(descriptor, IDX_TOOL_NAME, value.toolName)
                    encodeStringElement(descriptor, IDX_REASON, value.reason)
                }
                is ServerToClient.SudoResolved -> {
                    encodeStringElement(descriptor, IDX_TYPE, "sudo_resolved")
                    encodeLongElement(descriptor, IDX_FUNCTION_ID, value.functionId)
                    encodeStringElement(descriptor, IDX_THREAD_ID, value.threadId)
                    encodeSerializableElement(
                        descriptor, IDX_DECISION, SudoDecision.serializer(), value.decision,
                    )
                }
                is ServerToClient.BehaviorList -> {
                    encodeStringElement(descriptor, IDX_TYPE, "behavior_list")
                    value.correlationId?.let {
                        encodeStringElement(descriptor, IDX_CORRELATION_ID, it)
                    }
                    encodeStringElement(descriptor, IDX_POD_ID, value.podId)
                    encodeSerializableElement(
                        descriptor, IDX_BEHAVIORS, behaviorsSerializer, value.behaviors,
                    )
                }
                is ServerToClient.BehaviorCreated -> {
                    encodeStringElement(descriptor, IDX_TYPE, "behavior_created")
                    value.correlationId?.let {
                        encodeStringElement(descriptor, IDX_CORRELATION_ID, it)
                    }
                    encodeSerializableElement(
                        descriptor, IDX_SUMMARY, BehaviorSummary.serializer(), value.summary,
                    )
                }
                is ServerToClient.BehaviorDeleted -> {
                    encodeStringElement(descriptor, IDX_TYPE, "behavior_deleted")
                    value.correlationId?.let {
                        encodeStringElement(descriptor, IDX_CORRELATION_ID, it)
                    }
                    encodeStringElement(descriptor, IDX_POD_ID, value.podId)
                    encodeStringElement(descriptor, IDX_BEHAVIOR_ID, value.behaviorId)
                }
                is ServerToClient.BehaviorStateChanged -> {
                    encodeStringElement(descriptor, IDX_TYPE, "behavior_state_changed")
                    encodeStringElement(descriptor, IDX_POD_ID, value.podId)
                    encodeStringElement(descriptor, IDX_BEHAVIOR_ID, value.behaviorId)
                    encodeSerializableElement(
                        descriptor, IDX_STATE, BehaviorState.serializer(), value.state,
                    )
                }
                is ServerToClient.Error -> {
                    encodeStringElement(descriptor, IDX_TYPE, "error")
                    value.correlationId?.let {
                        encodeStringElement(descriptor, IDX_CORRELATION_ID, it)
                    }
                    value.threadId?.let {
                        encodeStringElement(descriptor, IDX_THREAD_ID, it)
                    }
                    encodeStringElement(descriptor, IDX_MESSAGE, value.message)
                }
                is ServerToClient.Unknown ->
                    throw SerializationException("cannot encode ServerToClient.Unknown — decode-only placeholder")
            }
        }
    }

    override fun deserialize(decoder: Decoder): ServerToClient {
        return decoder.decodeStructure(descriptor) {
            var type: String? = null
            var correlationId: String? = null
            var threadId: String? = null
            var tasks: List<ThreadSummary>? = null
            var summary: ThreadSummary? = null
            var state: ThreadStateLabel? = null
            var title: String? = null
            var snapshot: ThreadSnapshot? = null
            var text: String? = null
            var turn: Int? = null
            var delta: String? = null
            var stopReason: String? = null
            var usage: Usage? = null
            var toolUseId: String? = null
            var name: String? = null
            var argsPreview: String? = null
            var resultPreview: String? = null
            var isError = false
            var message: String? = null
            var pods: List<PodSummary>? = null
            var defaultPodId: String? = null
            var pod: PodSummary? = null
            var podId: String? = null
            var backends: List<BackendSummary>? = null
            var defaultBackend: String? = null
            var models: List<ModelSummary>? = null
            var backend: String? = null
            var functionId: Long? = null
            var toolName: String? = null
            var reason: String? = null
            var decision: SudoDecision? = null
            var newThreadId: String? = null
            var summaryText: String? = null
            var behaviors: List<BehaviorSummary>? = null
            var behaviorSummary: BehaviorSummary? = null
            var behaviorId: String? = null
            var behaviorState: BehaviorState? = null

            loop@ while (true) {
                when (val i = decodeElementIndex(descriptor)) {
                    CompositeDecoder.DECODE_DONE -> break@loop
                    IDX_TYPE -> type = decodeStringElement(descriptor, i)
                    IDX_CORRELATION_ID -> correlationId = decodeStringElement(descriptor, i)
                    IDX_THREAD_ID -> threadId = decodeStringElement(descriptor, i)
                    IDX_TASKS -> tasks = decodeSerializableElement(descriptor, i, tasksSerializer)
                    IDX_SUMMARY -> {
                        // `summary` is shared between thread_created and
                        // behavior_created; pick the matching deserializer
                        // using the type read above (serde emits `type`
                        // first). Unknown types decode as ThreadSummary
                        // and will be dropped downstream if they don't fit.
                        if (type == "behavior_created") {
                            behaviorSummary = decodeSerializableElement(
                                descriptor, i, BehaviorSummary.serializer(),
                            )
                        } else {
                            summary = decodeSerializableElement(
                                descriptor, i, ThreadSummary.serializer(),
                            )
                        }
                    }
                    IDX_STATE -> {
                        // Same story as IDX_SUMMARY: `state` is shared with
                        // behavior_state_changed.
                        if (type == "behavior_state_changed") {
                            behaviorState = decodeSerializableElement(
                                descriptor, i, BehaviorState.serializer(),
                            )
                        } else {
                            state = decodeSerializableElement(
                                descriptor, i, ThreadStateLabel.serializer(),
                            )
                        }
                    }
                    IDX_TITLE -> title = decodeStringElement(descriptor, i)
                    IDX_SNAPSHOT -> snapshot = decodeSerializableElement(
                        descriptor, i, ThreadSnapshot.serializer(),
                    )
                    IDX_TEXT -> text = decodeStringElement(descriptor, i)
                    IDX_TURN -> turn = decodeIntElement(descriptor, i)
                    IDX_DELTA -> delta = decodeStringElement(descriptor, i)
                    IDX_STOP_REASON -> stopReason = decodeStringElement(descriptor, i)
                    IDX_USAGE -> usage = decodeSerializableElement(
                        descriptor, i, Usage.serializer(),
                    )
                    IDX_TOOL_USE_ID -> toolUseId = decodeStringElement(descriptor, i)
                    IDX_NAME -> name = decodeStringElement(descriptor, i)
                    IDX_ARGS_PREVIEW -> argsPreview = decodeStringElement(descriptor, i)
                    IDX_RESULT_PREVIEW -> resultPreview = decodeStringElement(descriptor, i)
                    IDX_IS_ERROR -> isError = decodeBooleanElement(descriptor, i)
                    IDX_MESSAGE -> message = decodeStringElement(descriptor, i)
                    IDX_PODS -> pods = decodeSerializableElement(descriptor, i, podsSerializer)
                    IDX_DEFAULT_POD_ID -> defaultPodId = decodeStringElement(descriptor, i)
                    IDX_POD -> pod = decodeSerializableElement(
                        descriptor, i, PodSummary.serializer(),
                    )
                    IDX_POD_ID -> podId = decodeStringElement(descriptor, i)
                    IDX_BACKENDS -> backends = decodeSerializableElement(
                        descriptor, i, backendsSerializer,
                    )
                    IDX_DEFAULT_BACKEND -> defaultBackend = decodeStringElement(descriptor, i)
                    IDX_MODELS -> models = decodeSerializableElement(
                        descriptor, i, modelsSerializer,
                    )
                    IDX_BACKEND -> backend = decodeStringElement(descriptor, i)
                    IDX_FUNCTION_ID -> functionId = decodeLongElement(descriptor, i)
                    IDX_TOOL_NAME -> toolName = decodeStringElement(descriptor, i)
                    IDX_REASON -> reason = decodeStringElement(descriptor, i)
                    IDX_DECISION -> decision = decodeSerializableElement(
                        descriptor, i, SudoDecision.serializer(),
                    )
                    IDX_NEW_THREAD_ID -> newThreadId = decodeStringElement(descriptor, i)
                    IDX_SUMMARY_TEXT -> summaryText = decodeStringElement(descriptor, i)
                    IDX_BEHAVIORS -> behaviors = decodeSerializableElement(
                        descriptor, i, behaviorsSerializer,
                    )
                    IDX_BEHAVIOR_ID -> behaviorId = decodeStringElement(descriptor, i)
                    else -> throw SerializationException("unexpected element index $i")
                }
            }

            when (type) {
                "thread_list" -> ServerToClient.ThreadList(
                    correlationId = correlationId,
                    tasks = tasks ?: emptyList(),
                )
                "thread_created" -> ServerToClient.ThreadCreated(
                    threadId = requireNotNull(threadId) { "missing thread_id" },
                    summary = requireNotNull(summary) { "missing summary" },
                    correlationId = correlationId,
                )
                "thread_state_changed" -> ServerToClient.ThreadStateChanged(
                    threadId = requireNotNull(threadId) { "missing thread_id" },
                    state = requireNotNull(state) { "missing state" },
                )
                "thread_title_updated" -> ServerToClient.ThreadTitleUpdated(
                    threadId = requireNotNull(threadId) { "missing thread_id" },
                    title = requireNotNull(title) { "missing title" },
                )
                "thread_archived" -> ServerToClient.ThreadArchived(
                    threadId = requireNotNull(threadId) { "missing thread_id" },
                )
                "thread_compacted" -> ServerToClient.ThreadCompacted(
                    threadId = requireNotNull(threadId) { "missing thread_id" },
                    newThreadId = requireNotNull(newThreadId) { "missing new_thread_id" },
                    summaryText = summaryText ?: "",
                    correlationId = correlationId,
                )
                "thread_draft_updated" -> ServerToClient.ThreadDraftUpdated(
                    threadId = requireNotNull(threadId) { "missing thread_id" },
                    text = text ?: "",
                )
                "thread_snapshot" -> ServerToClient.Snapshot(
                    threadId = requireNotNull(threadId) { "missing thread_id" },
                    snapshot = requireNotNull(snapshot) { "missing snapshot" },
                )
                "thread_user_message" -> ServerToClient.ThreadUserMessage(
                    threadId = requireNotNull(threadId) { "missing thread_id" },
                    text = requireNotNull(text) { "missing text" },
                )
                "thread_assistant_begin" -> ServerToClient.AssistantBegin(
                    threadId = requireNotNull(threadId) { "missing thread_id" },
                    turn = requireNotNull(turn) { "missing turn" },
                )
                "thread_assistant_text_delta" -> ServerToClient.AssistantTextDelta(
                    threadId = requireNotNull(threadId) { "missing thread_id" },
                    delta = requireNotNull(delta) { "missing delta" },
                )
                "thread_assistant_reasoning_delta" -> ServerToClient.AssistantReasoningDelta(
                    threadId = requireNotNull(threadId) { "missing thread_id" },
                    delta = requireNotNull(delta) { "missing delta" },
                )
                "thread_assistant_end" -> ServerToClient.AssistantEnd(
                    threadId = requireNotNull(threadId) { "missing thread_id" },
                    stopReason = stopReason,
                    usage = requireNotNull(usage) { "missing usage" },
                )
                "thread_tool_call_begin" -> ServerToClient.ToolCallBegin(
                    threadId = requireNotNull(threadId) { "missing thread_id" },
                    toolUseId = requireNotNull(toolUseId) { "missing tool_use_id" },
                    name = requireNotNull(name) { "missing name" },
                    argsPreview = requireNotNull(argsPreview) { "missing args_preview" },
                )
                "thread_tool_call_end" -> ServerToClient.ToolCallEnd(
                    threadId = requireNotNull(threadId) { "missing thread_id" },
                    toolUseId = requireNotNull(toolUseId) { "missing tool_use_id" },
                    resultPreview = requireNotNull(resultPreview) { "missing result_preview" },
                    isError = isError,
                )
                "thread_loop_complete" -> ServerToClient.LoopComplete(
                    threadId = requireNotNull(threadId) { "missing thread_id" },
                )
                "pod_list" -> ServerToClient.PodList(
                    correlationId = correlationId,
                    pods = pods ?: emptyList(),
                    defaultPodId = defaultPodId ?: "",
                )
                "pod_created" -> ServerToClient.PodCreated(
                    pod = requireNotNull(pod) { "missing pod" },
                    correlationId = correlationId,
                )
                "pod_archived" -> ServerToClient.PodArchived(
                    podId = requireNotNull(podId) { "missing pod_id" },
                )
                "backends_list" -> ServerToClient.BackendsList(
                    correlationId = correlationId,
                    defaultBackend = defaultBackend ?: "",
                    backends = backends ?: emptyList(),
                )
                "models_list" -> ServerToClient.ModelsList(
                    correlationId = correlationId,
                    backend = requireNotNull(backend) { "missing backend" },
                    models = models ?: emptyList(),
                )
                "sudo_requested" -> ServerToClient.SudoRequested(
                    functionId = requireNotNull(functionId) { "missing function_id" },
                    threadId = requireNotNull(threadId) { "missing thread_id" },
                    toolName = requireNotNull(toolName) { "missing tool_name" },
                    reason = reason ?: "",
                )
                "sudo_resolved" -> ServerToClient.SudoResolved(
                    functionId = requireNotNull(functionId) { "missing function_id" },
                    threadId = requireNotNull(threadId) { "missing thread_id" },
                    decision = requireNotNull(decision) { "missing decision" },
                )
                "behavior_list" -> ServerToClient.BehaviorList(
                    podId = requireNotNull(podId) { "missing pod_id" },
                    behaviors = behaviors ?: emptyList(),
                    correlationId = correlationId,
                )
                "behavior_created" -> ServerToClient.BehaviorCreated(
                    summary = requireNotNull(behaviorSummary) { "missing summary" },
                    correlationId = correlationId,
                )
                "behavior_deleted" -> ServerToClient.BehaviorDeleted(
                    podId = requireNotNull(podId) { "missing pod_id" },
                    behaviorId = requireNotNull(behaviorId) { "missing behavior_id" },
                    correlationId = correlationId,
                )
                "behavior_state_changed" -> ServerToClient.BehaviorStateChanged(
                    podId = requireNotNull(podId) { "missing pod_id" },
                    behaviorId = requireNotNull(behaviorId) { "missing behavior_id" },
                    state = requireNotNull(behaviorState) { "missing state" },
                )
                "error" -> ServerToClient.Error(
                    correlationId = correlationId,
                    threadId = threadId,
                    message = requireNotNull(message) { "missing message" },
                )
                null -> throw SerializationException(
                    "ServerToClient missing 'type' discriminator — saw: " +
                        "threadId=$threadId, correlationId=$correlationId, " +
                        "tasks=${tasks?.size}, hasSummary=${summary != null}, " +
                        "state=$state, hasSnapshot=${snapshot != null}, " +
                        "text=${text?.take(32)}, delta=${delta?.take(32)}",
                )
                else -> ServerToClient.Unknown(type)
            }
        }
    }
}
