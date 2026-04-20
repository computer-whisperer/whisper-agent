package com.cjbal.whisperagent.protocol

import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.cbor.Cbor
import kotlinx.serialization.modules.SerializersModule
import kotlinx.serialization.modules.polymorphic
import kotlinx.serialization.modules.subclass

/**
 * CBOR (de)serialization entry points. Mirrors the four helper functions in
 * `whisper-agent-protocol/src/lib.rs` (`encode_to_server`, `decode_from_server`).
 *
 * Sticking point tracker: kotlinx-serialization-cbor is marked experimental and
 * has rough edges around serde's tagged enum shapes. If a real snapshot fails
 * to round-trip through [decodeFromServer], capture the raw bytes and compare
 * against what `whisper_agent_desktop` prints — that's the source of truth.
 */
@OptIn(ExperimentalSerializationApi::class)
object Codec {

    private val module = SerializersModule {
        polymorphic(ServerToClient::class) {
            subclass(ServerToClient.ThreadList::class)
            subclass(ServerToClient.ThreadCreated::class)
            subclass(ServerToClient.ThreadStateChanged::class)
            subclass(ServerToClient.ThreadTitleUpdated::class)
            subclass(ServerToClient.ThreadArchived::class)
            subclass(ServerToClient.Snapshot::class)
            subclass(ServerToClient.ThreadUserMessage::class)
            subclass(ServerToClient.AssistantBegin::class)
            subclass(ServerToClient.AssistantTextDelta::class)
            subclass(ServerToClient.AssistantReasoningDelta::class)
            subclass(ServerToClient.AssistantEnd::class)
            subclass(ServerToClient.ToolCallBegin::class)
            subclass(ServerToClient.ToolCallEnd::class)
            subclass(ServerToClient.LoopComplete::class)
            subclass(ServerToClient.PendingApproval::class)
            subclass(ServerToClient.ApprovalResolved::class)
            subclass(ServerToClient.Error::class)
            // Server may grow new variants between releases; fall through to
            // Unknown instead of throwing so the socket survives.
            defaultDeserializer { ServerToClient.Unknown.serializer() }
        }

        polymorphic(ClientToServer::class) {
            subclass(ClientToServer.ListThreads::class)
            subclass(ClientToServer.SubscribeToThread::class)
            subclass(ClientToServer.UnsubscribeFromThread::class)
            subclass(ClientToServer.SendUserMessage::class)
            subclass(ClientToServer.ApprovalDecision::class)
        }
    }

    // Class discriminator key defaults to `"type"` — matches the Rust side's
    // `#[serde(tag = "type")]`. It's only configurable via `@JsonClassDiscriminator`
    // on JSON; the CBOR builder has no equivalent knob, but the default value is
    // what we want so no override is necessary.
    private val cbor: Cbor = Cbor {
        serializersModule = module
        ignoreUnknownKeys = true
    }

    fun encodeToServer(msg: ClientToServer): ByteArray =
        cbor.encodeToByteArray(ClientToServer.serializer(), msg)

    fun decodeFromServer(bytes: ByteArray): ServerToClient =
        cbor.decodeFromByteArray(ServerToClient.serializer(), bytes)
}
