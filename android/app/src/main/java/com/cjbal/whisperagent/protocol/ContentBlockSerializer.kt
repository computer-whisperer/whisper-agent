package com.cjbal.whisperagent.protocol

import android.util.Log
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
 * Internally-tagged CBOR serializer for [ContentBlock]. Mirrors serde's
 * `#[serde(tag = "type", rename_all = "snake_case")]` on the Rust side —
 * `ContentBlock::ToolUse.input` and `ContentBlock::ToolResult.content`
 * are skipped pending the CBOR-tree proof-of-concept (same TODOs as
 * noted in [ContentBlock]).
 */
object ContentBlockSerializer : KSerializer<ContentBlock> {

    private const val TAG = "ContentBlockSerde"

    private const val IDX_TYPE = 0
    private const val IDX_TEXT = 1
    private const val IDX_ID = 2
    private const val IDX_NAME = 3
    private const val IDX_TOOL_USE_ID = 4
    private const val IDX_IS_ERROR = 5
    private const val IDX_THINKING = 6
    private const val IDX_DESCRIPTION = 7

    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("ContentBlock") {
        element("type", String.serializer().descriptor)
        element("text", String.serializer().descriptor, isOptional = true)
        element("id", String.serializer().descriptor, isOptional = true)
        element("name", String.serializer().descriptor, isOptional = true)
        element("tool_use_id", String.serializer().descriptor, isOptional = true)
        element("is_error", Boolean.serializer().descriptor, isOptional = true)
        element("thinking", String.serializer().descriptor, isOptional = true)
        element("description", String.serializer().descriptor, isOptional = true)
    }

    override fun serialize(encoder: Encoder, value: ContentBlock) {
        encoder.encodeStructure(descriptor) {
            when (value) {
                is ContentBlock.Text -> {
                    encodeStringElement(descriptor, IDX_TYPE, "text")
                    encodeStringElement(descriptor, IDX_TEXT, value.text)
                }
                is ContentBlock.ToolUse -> {
                    encodeStringElement(descriptor, IDX_TYPE, "tool_use")
                    encodeStringElement(descriptor, IDX_ID, value.id)
                    encodeStringElement(descriptor, IDX_NAME, value.name)
                    // TODO: `input: serde_json::Value` — depends on the
                    //       CBOR-tree proof-of-concept.
                }
                is ContentBlock.ToolResult -> {
                    encodeStringElement(descriptor, IDX_TYPE, "tool_result")
                    encodeStringElement(descriptor, IDX_TOOL_USE_ID, value.toolUseId)
                    // TODO: `content: ToolResultContent` — nested sealed enum.
                    if (value.isError) {
                        encodeBooleanElement(descriptor, IDX_IS_ERROR, true)
                    }
                }
                is ContentBlock.Thinking -> {
                    encodeStringElement(descriptor, IDX_TYPE, "thinking")
                    encodeStringElement(descriptor, IDX_THINKING, value.thinking)
                }
                is ContentBlock.ToolSchema -> {
                    encodeStringElement(descriptor, IDX_TYPE, "tool_schema")
                    encodeStringElement(descriptor, IDX_NAME, value.name)
                    encodeStringElement(descriptor, IDX_DESCRIPTION, value.description)
                }
            }
        }
    }

    override fun deserialize(decoder: Decoder): ContentBlock {
        return decoder.decodeStructure(descriptor) {
            var type: String? = null
            var text: String? = null
            var id: String? = null
            var name: String? = null
            var toolUseId: String? = null
            var isError = false
            var thinking: String? = null
            var description: String? = null

            loop@ while (true) {
                when (val i = decodeElementIndex(descriptor)) {
                    CompositeDecoder.DECODE_DONE -> break@loop
                    IDX_TYPE -> type = decodeStringElement(descriptor, i)
                    IDX_TEXT -> text = decodeStringElement(descriptor, i)
                    IDX_ID -> id = decodeStringElement(descriptor, i)
                    IDX_NAME -> name = decodeStringElement(descriptor, i)
                    IDX_TOOL_USE_ID -> toolUseId = decodeStringElement(descriptor, i)
                    IDX_IS_ERROR -> isError = decodeBooleanElement(descriptor, i)
                    IDX_THINKING -> thinking = decodeStringElement(descriptor, i)
                    IDX_DESCRIPTION -> description = decodeStringElement(descriptor, i)
                    else -> throw SerializationException("unexpected element index $i")
                }
            }

            when (type) {
                "text" -> ContentBlock.Text(requireNotNull(text) { "missing text" })
                    .also { Log.d(TAG, "decoded Text(len=${it.text.length})") }
                "tool_use" -> ContentBlock.ToolUse(
                    id = requireNotNull(id) { "missing id" },
                    name = requireNotNull(name) { "missing name" },
                ).also { Log.d(TAG, "decoded ToolUse(id=${it.id}, name=${it.name})") }
                "tool_result" -> ContentBlock.ToolResult(
                    toolUseId = requireNotNull(toolUseId) { "missing tool_use_id" },
                    isError = isError,
                ).also { Log.d(TAG, "decoded ToolResult(tool_use_id=${it.toolUseId}, is_error=${it.isError})") }
                "thinking" -> ContentBlock.Thinking(requireNotNull(thinking) { "missing thinking" })
                    .also { Log.d(TAG, "decoded Thinking(len=${it.thinking.length})") }
                "tool_schema" -> ContentBlock.ToolSchema(
                    name = requireNotNull(name) { "missing name" },
                    description = requireNotNull(description) { "missing description" },
                ).also { Log.d(TAG, "decoded ToolSchema(name=${it.name})") }
                null -> throw SerializationException(
                    "ContentBlock missing 'type' discriminator — saw: " +
                        "text=${text?.take(32)}, id=$id, name=$name, toolUseId=$toolUseId, " +
                        "isError=$isError, thinking=${thinking?.take(32)}, description=${description?.take(32)}",
                )
                else -> throw SerializationException("unknown ContentBlock variant: $type")
            }
        }
    }
}
