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
    private const val IDX_APPROVAL_ID = 4
    private const val IDX_DECISION = 5
    private const val IDX_REMEMBER = 6

    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("ClientToServer") {
        element("type", String.serializer().descriptor)
        element("correlation_id", String.serializer().descriptor, isOptional = true)
        element("thread_id", String.serializer().descriptor, isOptional = true)
        element("text", String.serializer().descriptor, isOptional = true)
        element("approval_id", String.serializer().descriptor, isOptional = true)
        element("decision", ApprovalChoice.serializer().descriptor, isOptional = true)
        element("remember", Boolean.serializer().descriptor, isOptional = true)
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
                is ClientToServer.ApprovalDecision -> {
                    encodeStringElement(descriptor, IDX_TYPE, "approval_decision")
                    encodeStringElement(descriptor, IDX_THREAD_ID, value.threadId)
                    encodeStringElement(descriptor, IDX_APPROVAL_ID, value.approvalId)
                    encodeSerializableElement(
                        descriptor, IDX_DECISION, ApprovalChoice.serializer(), value.decision,
                    )
                    // Serde's `#[serde(default)]` on `remember` means it's
                    // serialized only when true, matching Rust's default false.
                    if (value.remember) {
                        encodeBooleanElement(descriptor, IDX_REMEMBER, true)
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
            var approvalId: String? = null
            var decision: ApprovalChoice? = null
            var remember = false

            loop@ while (true) {
                when (val i = decodeElementIndex(descriptor)) {
                    CompositeDecoder.DECODE_DONE -> break@loop
                    IDX_TYPE -> type = decodeStringElement(descriptor, i)
                    IDX_CORRELATION_ID -> correlationId = decodeStringElement(descriptor, i)
                    IDX_THREAD_ID -> threadId = decodeStringElement(descriptor, i)
                    IDX_TEXT -> text = decodeStringElement(descriptor, i)
                    IDX_APPROVAL_ID -> approvalId = decodeStringElement(descriptor, i)
                    IDX_DECISION -> decision = decodeSerializableElement(
                        descriptor, i, ApprovalChoice.serializer(),
                    )
                    IDX_REMEMBER -> remember = decodeBooleanElement(descriptor, i)
                    else -> throw SerializationException("unexpected element index $i")
                }
            }

            when (type) {
                "list_threads" -> ClientToServer.ListThreads(correlationId)
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
                "approval_decision" -> ClientToServer.ApprovalDecision(
                    threadId = requireNotNull(threadId) { "missing thread_id" },
                    approvalId = requireNotNull(approvalId) { "missing approval_id" },
                    decision = requireNotNull(decision) { "missing decision" },
                    remember = remember,
                )
                null -> throw SerializationException("missing 'type' discriminator")
                else -> throw SerializationException("unknown ClientToServer variant: $type")
            }
        }
    }
}
