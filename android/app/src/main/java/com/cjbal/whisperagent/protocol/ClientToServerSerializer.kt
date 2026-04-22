package com.cjbal.whisperagent.protocol

import kotlinx.serialization.KSerializer
import kotlinx.serialization.SerializationException
import kotlinx.serialization.builtins.serializer
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.descriptors.buildClassSerialDescriptor
import kotlinx.serialization.encoding.CompositeDecoder
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import kotlinx.serialization.encoding.decodeStructure
import kotlinx.serialization.encoding.encodeStructure

/**
 * Internally-tagged (de)serializer for [ClientToServer] that produces the
 * same CBOR wire shape as serde's `#[serde(tag = "type", rename_all = "snake_case")]`:
 * a single flat map whose `type` key carries the variant name and whose
 * remaining keys carry the variant's fields.
 *
 * kotlinx-serialization-cbor's default polymorphic encoding is `[tag, content]`
 * as a CBOR array — incompatible with ciborium on the server. This serializer
 * side-steps that by declaring a single `CLASS`-kind descriptor whose elements
 * are the union of every variant's fields. [serialize] writes only the subset
 * applicable to the concrete variant; [deserialize] buffers fields as they
 * arrive and dispatches on `type` at the end.
 *
 * Every new variant added to [ClientToServer] needs:
 *   - an `IDX_*` constant if it introduces a new field,
 *   - the field added to [descriptor],
 *   - a branch in [serialize],
 *   - buffering + dispatch in [deserialize].
 */
object ClientToServerSerializer : KSerializer<ClientToServer> {

    private const val IDX_TYPE = 0
    private const val IDX_CORRELATION_ID = 1
    private const val IDX_THREAD_ID = 2
    private const val IDX_TEXT = 3
    private const val IDX_POD_ID = 4
    private const val IDX_INITIAL_MESSAGE = 5
    private const val IDX_CONFIG_OVERRIDE = 6
    private const val IDX_BINDINGS_REQUEST = 7
    private const val IDX_BACKEND = 8
    private const val IDX_FUNCTION_ID = 9
    private const val IDX_DECISION = 10
    private const val IDX_REASON = 11
    private const val IDX_FROM_MESSAGE_INDEX = 12
    private const val IDX_ARCHIVE_ORIGINAL = 13
    private const val IDX_RESET_CAPABILITIES = 14

    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("ClientToServer") {
        element("type", String.serializer().descriptor)
        element("correlation_id", String.serializer().descriptor, isOptional = true)
        element("thread_id", String.serializer().descriptor, isOptional = true)
        element("text", String.serializer().descriptor, isOptional = true)
        element("pod_id", String.serializer().descriptor, isOptional = true)
        element("initial_message", String.serializer().descriptor, isOptional = true)
        element("config_override", ThreadConfigOverride.serializer().descriptor, isOptional = true)
        element("bindings_request", ThreadBindingsRequest.serializer().descriptor, isOptional = true)
        element("backend", String.serializer().descriptor, isOptional = true)
        element("function_id", Long.serializer().descriptor, isOptional = true)
        element("decision", SudoDecision.serializer().descriptor, isOptional = true)
        element("reason", String.serializer().descriptor, isOptional = true)
        element("from_message_index", Int.serializer().descriptor, isOptional = true)
        element("archive_original", Boolean.serializer().descriptor, isOptional = true)
        element("reset_capabilities", Boolean.serializer().descriptor, isOptional = true)
    }

    override fun serialize(encoder: Encoder, value: ClientToServer) {
        encoder.encodeStructure(descriptor) {
            when (value) {
                is ClientToServer.ListThreads -> {
                    encodeStringElement(descriptor, IDX_TYPE, "list_threads")
                    // Match serde's skip_serializing_if = Option::is_none — emit
                    // the field only when non-null.
                    value.correlationId?.let {
                        encodeStringElement(descriptor, IDX_CORRELATION_ID, it)
                    }
                }
                is ClientToServer.CreateThread -> {
                    encodeStringElement(descriptor, IDX_TYPE, "create_thread")
                    value.correlationId?.let {
                        encodeStringElement(descriptor, IDX_CORRELATION_ID, it)
                    }
                    value.podId?.let {
                        encodeStringElement(descriptor, IDX_POD_ID, it)
                    }
                    encodeStringElement(descriptor, IDX_INITIAL_MESSAGE, value.initialMessage)
                    value.configOverride?.let {
                        encodeSerializableElement(
                            descriptor, IDX_CONFIG_OVERRIDE,
                            ThreadConfigOverride.serializer(), it,
                        )
                    }
                    value.bindingsRequest?.let {
                        encodeSerializableElement(
                            descriptor, IDX_BINDINGS_REQUEST,
                            ThreadBindingsRequest.serializer(), it,
                        )
                    }
                }
                is ClientToServer.SubscribeToThread -> {
                    encodeStringElement(descriptor, IDX_TYPE, "subscribe_to_thread")
                    encodeStringElement(descriptor, IDX_THREAD_ID, value.threadId)
                }
                is ClientToServer.UnsubscribeFromThread -> {
                    encodeStringElement(descriptor, IDX_TYPE, "unsubscribe_from_thread")
                    encodeStringElement(descriptor, IDX_THREAD_ID, value.threadId)
                }
                is ClientToServer.SendUserMessage -> {
                    encodeStringElement(descriptor, IDX_TYPE, "send_user_message")
                    encodeStringElement(descriptor, IDX_THREAD_ID, value.threadId)
                    encodeStringElement(descriptor, IDX_TEXT, value.text)
                }
                is ClientToServer.ListPods -> {
                    encodeStringElement(descriptor, IDX_TYPE, "list_pods")
                    value.correlationId?.let {
                        encodeStringElement(descriptor, IDX_CORRELATION_ID, it)
                    }
                }
                is ClientToServer.ListBackends -> {
                    encodeStringElement(descriptor, IDX_TYPE, "list_backends")
                    value.correlationId?.let {
                        encodeStringElement(descriptor, IDX_CORRELATION_ID, it)
                    }
                }
                is ClientToServer.ListModels -> {
                    encodeStringElement(descriptor, IDX_TYPE, "list_models")
                    value.correlationId?.let {
                        encodeStringElement(descriptor, IDX_CORRELATION_ID, it)
                    }
                    encodeStringElement(descriptor, IDX_BACKEND, value.backend)
                }
                is ClientToServer.CancelThread -> {
                    encodeStringElement(descriptor, IDX_TYPE, "cancel_thread")
                    encodeStringElement(descriptor, IDX_THREAD_ID, value.threadId)
                }
                is ClientToServer.ArchiveThread -> {
                    encodeStringElement(descriptor, IDX_TYPE, "archive_thread")
                    encodeStringElement(descriptor, IDX_THREAD_ID, value.threadId)
                }
                is ClientToServer.ResolveSudo -> {
                    encodeStringElement(descriptor, IDX_TYPE, "resolve_sudo")
                    encodeLongElement(descriptor, IDX_FUNCTION_ID, value.functionId)
                    encodeSerializableElement(
                        descriptor, IDX_DECISION, SudoDecision.serializer(), value.decision,
                    )
                    value.reason?.let {
                        encodeStringElement(descriptor, IDX_REASON, it)
                    }
                }
                is ClientToServer.CompactThread -> {
                    encodeStringElement(descriptor, IDX_TYPE, "compact_thread")
                    encodeStringElement(descriptor, IDX_THREAD_ID, value.threadId)
                    value.correlationId?.let {
                        encodeStringElement(descriptor, IDX_CORRELATION_ID, it)
                    }
                }
                is ClientToServer.RecoverThread -> {
                    encodeStringElement(descriptor, IDX_TYPE, "recover_thread")
                    encodeStringElement(descriptor, IDX_THREAD_ID, value.threadId)
                    value.correlationId?.let {
                        encodeStringElement(descriptor, IDX_CORRELATION_ID, it)
                    }
                }
                is ClientToServer.ForkThread -> {
                    encodeStringElement(descriptor, IDX_TYPE, "fork_thread")
                    encodeStringElement(descriptor, IDX_THREAD_ID, value.threadId)
                    encodeIntElement(
                        descriptor, IDX_FROM_MESSAGE_INDEX, value.fromMessageIndex,
                    )
                    // Matches serde's `#[serde(default)]` — emit only when
                    // non-default to keep the wire spare.
                    if (value.archiveOriginal) {
                        encodeBooleanElement(
                            descriptor, IDX_ARCHIVE_ORIGINAL, value.archiveOriginal,
                        )
                    }
                    if (value.resetCapabilities) {
                        encodeBooleanElement(
                            descriptor, IDX_RESET_CAPABILITIES, value.resetCapabilities,
                        )
                    }
                    value.correlationId?.let {
                        encodeStringElement(descriptor, IDX_CORRELATION_ID, it)
                    }
                }
            }
        }
    }

    override fun deserialize(decoder: Decoder): ClientToServer {
        // The Android client never receives a ClientToServer — this path exists
        // only so round-trip tests can verify our encode is self-consistent.
        return decoder.decodeStructure(descriptor) {
            var type: String? = null
            var correlationId: String? = null
            var threadId: String? = null
            var text: String? = null
            var podId: String? = null
            var initialMessage: String? = null
            var configOverride: ThreadConfigOverride? = null
            var bindingsRequest: ThreadBindingsRequest? = null
            var backend: String? = null
            var functionId: Long? = null
            var decision: SudoDecision? = null
            var reason: String? = null
            var fromMessageIndex: Int? = null
            var archiveOriginal = false
            var resetCapabilities = false

            loop@ while (true) {
                when (val i = decodeElementIndex(descriptor)) {
                    CompositeDecoder.DECODE_DONE -> break@loop
                    IDX_TYPE -> type = decodeStringElement(descriptor, i)
                    IDX_CORRELATION_ID -> correlationId = decodeStringElement(descriptor, i)
                    IDX_THREAD_ID -> threadId = decodeStringElement(descriptor, i)
                    IDX_TEXT -> text = decodeStringElement(descriptor, i)
                    IDX_POD_ID -> podId = decodeStringElement(descriptor, i)
                    IDX_INITIAL_MESSAGE -> initialMessage = decodeStringElement(descriptor, i)
                    IDX_CONFIG_OVERRIDE -> configOverride = decodeSerializableElement(
                        descriptor, i, ThreadConfigOverride.serializer(),
                    )
                    IDX_BINDINGS_REQUEST -> bindingsRequest = decodeSerializableElement(
                        descriptor, i, ThreadBindingsRequest.serializer(),
                    )
                    IDX_BACKEND -> backend = decodeStringElement(descriptor, i)
                    IDX_FUNCTION_ID -> functionId = decodeLongElement(descriptor, i)
                    IDX_DECISION -> decision = decodeSerializableElement(
                        descriptor, i, SudoDecision.serializer(),
                    )
                    IDX_REASON -> reason = decodeStringElement(descriptor, i)
                    IDX_FROM_MESSAGE_INDEX -> fromMessageIndex = decodeIntElement(descriptor, i)
                    IDX_ARCHIVE_ORIGINAL -> archiveOriginal = decodeBooleanElement(descriptor, i)
                    IDX_RESET_CAPABILITIES -> resetCapabilities = decodeBooleanElement(descriptor, i)
                    else -> throw SerializationException("unexpected element index $i")
                }
            }

            when (type) {
                "list_threads" -> ClientToServer.ListThreads(correlationId)
                "create_thread" -> ClientToServer.CreateThread(
                    initialMessage = requireNotNull(initialMessage) { "missing initial_message" },
                    correlationId = correlationId,
                    podId = podId,
                    configOverride = configOverride,
                    bindingsRequest = bindingsRequest,
                )
                "subscribe_to_thread" -> ClientToServer.SubscribeToThread(
                    requireNotNull(threadId) { "missing thread_id" },
                )
                "unsubscribe_from_thread" -> ClientToServer.UnsubscribeFromThread(
                    requireNotNull(threadId) { "missing thread_id" },
                )
                "send_user_message" -> ClientToServer.SendUserMessage(
                    threadId = requireNotNull(threadId) { "missing thread_id" },
                    text = requireNotNull(text) { "missing text" },
                )
                "list_pods" -> ClientToServer.ListPods(correlationId)
                "list_backends" -> ClientToServer.ListBackends(correlationId)
                "list_models" -> ClientToServer.ListModels(
                    backend = requireNotNull(backend) { "missing backend" },
                    correlationId = correlationId,
                )
                "cancel_thread" -> ClientToServer.CancelThread(
                    threadId = requireNotNull(threadId) { "missing thread_id" },
                )
                "archive_thread" -> ClientToServer.ArchiveThread(
                    threadId = requireNotNull(threadId) { "missing thread_id" },
                )
                "resolve_sudo" -> ClientToServer.ResolveSudo(
                    functionId = requireNotNull(functionId) { "missing function_id" },
                    decision = requireNotNull(decision) { "missing decision" },
                    reason = reason,
                )
                "compact_thread" -> ClientToServer.CompactThread(
                    threadId = requireNotNull(threadId) { "missing thread_id" },
                    correlationId = correlationId,
                )
                "recover_thread" -> ClientToServer.RecoverThread(
                    threadId = requireNotNull(threadId) { "missing thread_id" },
                    correlationId = correlationId,
                )
                "fork_thread" -> ClientToServer.ForkThread(
                    threadId = requireNotNull(threadId) { "missing thread_id" },
                    fromMessageIndex = requireNotNull(fromMessageIndex) {
                        "missing from_message_index"
                    },
                    archiveOriginal = archiveOriginal,
                    resetCapabilities = resetCapabilities,
                    correlationId = correlationId,
                )
                null -> throw SerializationException("missing 'type' discriminator")
                else -> throw SerializationException("unknown ClientToServer variant: $type")
            }
        }
    }
}
