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
    private const val IDX_APPROVAL_ID = 18
    private const val IDX_DESTRUCTIVE = 19
    private const val IDX_READ_ONLY = 20
    private const val IDX_DECISION = 21
    private const val IDX_MESSAGE = 22

    private val tasksSerializer = ListSerializer(ThreadSummary.serializer())

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
        element("approval_id", String.serializer().descriptor, isOptional = true)
        element("destructive", Boolean.serializer().descriptor, isOptional = true)
        element("read_only", Boolean.serializer().descriptor, isOptional = true)
        element("decision", ApprovalChoice.serializer().descriptor, isOptional = true)
        element("message", String.serializer().descriptor, isOptional = true)
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
                is ServerToClient.PendingApproval -> {
                    encodeStringElement(descriptor, IDX_TYPE, "thread_pending_approval")
                    encodeStringElement(descriptor, IDX_THREAD_ID, value.threadId)
                    encodeStringElement(descriptor, IDX_APPROVAL_ID, value.approvalId)
                    encodeStringElement(descriptor, IDX_TOOL_USE_ID, value.toolUseId)
                    encodeStringElement(descriptor, IDX_NAME, value.name)
                    encodeStringElement(descriptor, IDX_ARGS_PREVIEW, value.argsPreview)
                    if (value.destructive) {
                        encodeBooleanElement(descriptor, IDX_DESTRUCTIVE, true)
                    }
                    if (value.readOnly) {
                        encodeBooleanElement(descriptor, IDX_READ_ONLY, true)
                    }
                }
                is ServerToClient.ApprovalResolved -> {
                    encodeStringElement(descriptor, IDX_TYPE, "thread_approval_resolved")
                    encodeStringElement(descriptor, IDX_THREAD_ID, value.threadId)
                    encodeStringElement(descriptor, IDX_APPROVAL_ID, value.approvalId)
                    encodeSerializableElement(
                        descriptor, IDX_DECISION, ApprovalChoice.serializer(), value.decision,
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
            var approvalId: String? = null
            var destructive = false
            var readOnly = false
            var decision: ApprovalChoice? = null
            var message: String? = null

            loop@ while (true) {
                when (val i = decodeElementIndex(descriptor)) {
                    CompositeDecoder.DECODE_DONE -> break@loop
                    IDX_TYPE -> type = decodeStringElement(descriptor, i)
                    IDX_CORRELATION_ID -> correlationId = decodeStringElement(descriptor, i)
                    IDX_THREAD_ID -> threadId = decodeStringElement(descriptor, i)
                    IDX_TASKS -> tasks = decodeSerializableElement(descriptor, i, tasksSerializer)
                    IDX_SUMMARY -> summary = decodeSerializableElement(
                        descriptor, i, ThreadSummary.serializer(),
                    )
                    IDX_STATE -> state = decodeSerializableElement(
                        descriptor, i, ThreadStateLabel.serializer(),
                    )
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
                    IDX_APPROVAL_ID -> approvalId = decodeStringElement(descriptor, i)
                    IDX_DESTRUCTIVE -> destructive = decodeBooleanElement(descriptor, i)
                    IDX_READ_ONLY -> readOnly = decodeBooleanElement(descriptor, i)
                    IDX_DECISION -> decision = decodeSerializableElement(
                        descriptor, i, ApprovalChoice.serializer(),
                    )
                    IDX_MESSAGE -> message = decodeStringElement(descriptor, i)
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
                "thread_pending_approval" -> ServerToClient.PendingApproval(
                    threadId = requireNotNull(threadId) { "missing thread_id" },
                    approvalId = requireNotNull(approvalId) { "missing approval_id" },
                    toolUseId = requireNotNull(toolUseId) { "missing tool_use_id" },
                    name = requireNotNull(name) { "missing name" },
                    argsPreview = requireNotNull(argsPreview) { "missing args_preview" },
                    destructive = destructive,
                    readOnly = readOnly,
                )
                "thread_approval_resolved" -> ServerToClient.ApprovalResolved(
                    threadId = requireNotNull(threadId) { "missing thread_id" },
                    approvalId = requireNotNull(approvalId) { "missing approval_id" },
                    decision = requireNotNull(decision) { "missing decision" },
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
