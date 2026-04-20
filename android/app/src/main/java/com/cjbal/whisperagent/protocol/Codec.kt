package com.cjbal.whisperagent.protocol

import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.cbor.Cbor

/**
 * CBOR (de)serialization entry points. Mirrors the four helper functions in
 * `whisper-agent-protocol/src/lib.rs` (`encode_to_server`, `decode_from_server`).
 *
 * The two polymorphic top-level types ([ClientToServer], [ServerToClient]) and
 * the polymorphic [ContentBlock] all use custom `@Serializable(with = ...)`
 * serializers that produce serde-compatible internally-tagged CBOR maps.
 * kotlinx-cbor's default polymorphic encoding is a two-element array, which
 * ciborium on the server rejects — so we don't use it anywhere.
 */
@OptIn(ExperimentalSerializationApi::class)
object Codec {

    private val cbor: Cbor = Cbor {
        ignoreUnknownKeys = true
        // ciborium on the server emits definite-length maps and arrays; match
        // that so test byte-comparisons are stable and the wire format doesn't
        // drift between kotlinx-cbor releases.
        useDefiniteLengthEncoding = true
    }

    fun encodeToServer(msg: ClientToServer): ByteArray =
        cbor.encodeToByteArray(ClientToServer.serializer(), msg)

    fun decodeFromServer(bytes: ByteArray): ServerToClient =
        cbor.decodeFromByteArray(ServerToClient.serializer(), bytes)
}
