package com.cjbal.whisperagent.protocol

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.cbor.Cbor

@OptIn(ExperimentalSerializationApi::class)
class CodecTest {

    private val cbor = Cbor {
        ignoreUnknownKeys = true
        useDefiniteLengthEncoding = true
    }

    // --- ClientToServer -------------------------------------------------------

    @Test
    fun clientToServer_listThreads_bare() {
        // Matches serde `#[serde(tag = "type", rename_all = "snake_case")]` with
        // correlation_id skipped (None + skip_serializing_if = Option::is_none).
        val bytes = Codec.encodeToServer(ClientToServer.ListThreads())
        // Expected CBOR: map with 1 entry { "type": "list_threads" }
        //   0xA1 = map(1)
        //   0x64 = text(4), "type"
        //   0x6C = text(12), "list_threads"
        val expected = byteArrayOf(
            0xA1.toByte(),
            0x64, 't'.code.toByte(), 'y'.code.toByte(), 'p'.code.toByte(), 'e'.code.toByte(),
            0x6C,
            'l'.code.toByte(), 'i'.code.toByte(), 's'.code.toByte(), 't'.code.toByte(),
            '_'.code.toByte(),
            't'.code.toByte(), 'h'.code.toByte(), 'r'.code.toByte(), 'e'.code.toByte(),
            'a'.code.toByte(), 'd'.code.toByte(), 's'.code.toByte(),
        )
        assertContentEquals(expected, bytes)
    }

    @Test
    fun clientToServer_sendUserMessage_roundTrip() {
        val original = ClientToServer.SendUserMessage(threadId = "t-42", text = "hello")
        val bytes = Codec.encodeToServer(original)
        // Decode via the same serializer to check self-consistency.
        val decoded = cbor.decodeFromByteArray(ClientToServer.serializer(), bytes)
        assertEquals(original, decoded)
    }

    // --- ServerToClient -------------------------------------------------------

    @Test
    fun serverToClient_threadList_roundTrip() {
        val original = ServerToClient.ThreadList(
            correlationId = "corr-1",
            tasks = listOf(
                ThreadSummary(
                    threadId = "t-1",
                    podId = "p-1",
                    title = "hello",
                    state = ThreadStateLabel.Idle,
                    createdAt = "2026-04-20T00:00:00Z",
                    lastActive = "2026-04-20T00:00:01Z",
                ),
            ),
        )
        val bytes = cbor.encodeToByteArray(ServerToClient.serializer(), original)
        val decoded = Codec.decodeFromServer(bytes)
        assertEquals(original, decoded)
    }

    @Test
    fun serverToClient_unknown_variantFallback() {
        // Simulate a server frame with a type our Kotlin mirror doesn't know —
        // construct it by hand as a CBOR map { "type": "host_env_provider_added" }.
        // 0x77 = text(23), matching "host_env_provider_added" (CBOR packs
        // lengths 0..23 directly into the major-type byte).
        val rawCbor = byteArrayOf(
            0xA1.toByte(), // map(1)
            0x64, 't'.code.toByte(), 'y'.code.toByte(), 'p'.code.toByte(), 'e'.code.toByte(),
            0x77,
            'h'.code.toByte(), 'o'.code.toByte(), 's'.code.toByte(), 't'.code.toByte(),
            '_'.code.toByte(),
            'e'.code.toByte(), 'n'.code.toByte(), 'v'.code.toByte(),
            '_'.code.toByte(),
            'p'.code.toByte(), 'r'.code.toByte(), 'o'.code.toByte(), 'v'.code.toByte(),
            'i'.code.toByte(), 'd'.code.toByte(), 'e'.code.toByte(), 'r'.code.toByte(),
            '_'.code.toByte(),
            'a'.code.toByte(), 'd'.code.toByte(), 'd'.code.toByte(), 'e'.code.toByte(),
            'd'.code.toByte(),
        )
        val decoded = Codec.decodeFromServer(rawCbor)
        val unknown = assertIs<ServerToClient.Unknown>(decoded)
        assertEquals("host_env_provider_added", unknown.type)
    }

    @Test
    fun serverToClient_assistantTextDelta_roundTrip() {
        val original = ServerToClient.AssistantTextDelta(threadId = "t-1", delta = "Hello ")
        val bytes = cbor.encodeToByteArray(ServerToClient.serializer(), original)
        val decoded = Codec.decodeFromServer(bytes)
        assertEquals(original, decoded)
    }

    @Test
    fun serverToClient_error_roundTrip_nullThreadId() {
        val original = ServerToClient.Error(
            correlationId = null,
            threadId = null,
            message = "something broke",
        )
        val bytes = cbor.encodeToByteArray(ServerToClient.serializer(), original)
        val decoded = Codec.decodeFromServer(bytes)
        assertEquals(original, decoded)
    }

    // --- ContentBlock ---------------------------------------------------------

    @Test
    fun contentBlock_text_roundTrip() {
        val original = ContentBlock.Text("hi")
        val bytes = cbor.encodeToByteArray(ContentBlock.serializer(), original)
        val decoded = cbor.decodeFromByteArray(ContentBlock.serializer(), bytes)
        assertEquals(original, decoded)
    }

    @Test
    fun contentBlock_discriminatorIsFirstKey() {
        // Smoke-check the internal-tagging shape: after the CBOR map header
        // byte, the next payload should be the 4-byte string "type".
        val bytes = cbor.encodeToByteArray(ContentBlock.serializer(), ContentBlock.Text("hi"))
        // header + "type" header (0x64 = text(4))
        assertEquals(0x64.toByte(), bytes[1])
        assertEquals('t'.code.toByte(), bytes[2])
    }

    @Test
    fun contentBlock_thinking_roundTrip() {
        val original = ContentBlock.Thinking("considering")
        val bytes = cbor.encodeToByteArray(ContentBlock.serializer(), original)
        val decoded = cbor.decodeFromByteArray(ContentBlock.serializer(), bytes)
        assertEquals(original, decoded)
    }

    @Test
    fun contentBlock_toolSchema_roundTrip() {
        val original = ContentBlock.ToolSchema(name = "bash", description = "runs shell")
        val bytes = cbor.encodeToByteArray(ContentBlock.serializer(), original)
        val decoded = cbor.decodeFromByteArray(ContentBlock.serializer(), bytes)
        assertEquals(original, decoded)
    }

    // --- Nested: ThreadSnapshot with ContentBlocks ----------------------------

    @Test
    fun serverToClient_snapshot_withNestedContentBlocks_roundTrip() {
        val original = ServerToClient.Snapshot(
            threadId = "t-7",
            snapshot = ThreadSnapshot(
                threadId = "t-7",
                podId = "p-1",
                title = null,
                config = ThreadConfig(model = "claude-opus-4-7", maxTokens = 4096, maxTurns = 32),
                state = ThreadStateLabel.Idle,
                conversation = listOf(
                    Message(role = Role.User, content = listOf(ContentBlock.Text("hello"))),
                    Message(
                        role = Role.Assistant,
                        content = listOf(
                            ContentBlock.Thinking("let me think"),
                            ContentBlock.Text("hi there"),
                        ),
                    ),
                ),
                totalUsage = Usage(inputTokens = 12, outputTokens = 8),
                createdAt = "2026-04-20T00:00:00Z",
                lastActive = "2026-04-20T00:00:01Z",
            ),
        )
        val bytes = cbor.encodeToByteArray(ServerToClient.serializer(), original)
        val decoded = Codec.decodeFromServer(bytes)
        assertEquals(original, decoded)
    }

    // --- Pod-registry variants ------------------------------------------------

    @Test
    fun clientToServer_listPods_bare() {
        val bytes = Codec.encodeToServer(ClientToServer.ListPods())
        // map(1) { "type": "list_pods" }
        val expected = byteArrayOf(
            0xA1.toByte(),
            0x64, 't'.code.toByte(), 'y'.code.toByte(), 'p'.code.toByte(), 'e'.code.toByte(),
            0x69, // text(9)
            'l'.code.toByte(), 'i'.code.toByte(), 's'.code.toByte(), 't'.code.toByte(),
            '_'.code.toByte(),
            'p'.code.toByte(), 'o'.code.toByte(), 'd'.code.toByte(), 's'.code.toByte(),
        )
        assertContentEquals(expected, bytes)
    }

    @Test
    fun serverToClient_podList_roundTrip() {
        val original = ServerToClient.PodList(
            correlationId = "corr-7",
            pods = listOf(
                PodSummary(
                    podId = "default",
                    name = "Default",
                    description = null,
                    createdAt = "2026-04-20T00:00:00Z",
                    threadCount = 3,
                    archived = false,
                    behaviorsEnabled = true,
                ),
                PodSummary(
                    podId = "scratch",
                    name = "Scratch",
                    description = "experimental",
                    createdAt = "2026-04-20T01:00:00Z",
                    threadCount = 0,
                ),
            ),
            defaultPodId = "default",
        )
        val bytes = cbor.encodeToByteArray(ServerToClient.serializer(), original)
        val decoded = Codec.decodeFromServer(bytes)
        assertEquals(original, decoded)
    }

    @Test
    fun serverToClient_podCreated_roundTrip() {
        val original = ServerToClient.PodCreated(
            pod = PodSummary(
                podId = "fresh",
                name = "Fresh",
                createdAt = "2026-04-20T02:00:00Z",
            ),
            correlationId = "corr-make-pod",
        )
        val bytes = cbor.encodeToByteArray(ServerToClient.serializer(), original)
        val decoded = Codec.decodeFromServer(bytes)
        assertEquals(original, decoded)
    }

    @Test
    fun serverToClient_podArchived_roundTrip() {
        val original = ServerToClient.PodArchived(podId = "scratch")
        val bytes = cbor.encodeToByteArray(ServerToClient.serializer(), original)
        val decoded = Codec.decodeFromServer(bytes)
        assertEquals(original, decoded)
    }

    // --- Backend / model catalog ---------------------------------------------

    @Test
    fun clientToServer_listBackends_bare() {
        val bytes = Codec.encodeToServer(ClientToServer.ListBackends())
        // map(1) { "type": "list_backends" }
        assertEquals(0xA1.toByte(), bytes[0])
        val decoded = cbor.decodeFromByteArray(ClientToServer.serializer(), bytes)
        assertEquals(ClientToServer.ListBackends(), decoded)
    }

    @Test
    fun clientToServer_listModels_roundTrip() {
        val original = ClientToServer.ListModels(backend = "anthropic", correlationId = "m-1")
        val bytes = Codec.encodeToServer(original)
        val decoded = cbor.decodeFromByteArray(ClientToServer.serializer(), bytes)
        assertEquals(original, decoded)
    }

    @Test
    fun clientToServer_createThread_withOverrides_roundTrip() {
        val original = ClientToServer.CreateThread(
            initialMessage = "hello",
            correlationId = "create-1",
            podId = "default",
            configOverride = ThreadConfigOverride(model = "claude-opus-4-7"),
            bindingsRequest = ThreadBindingsRequest(backend = "anthropic"),
        )
        val bytes = Codec.encodeToServer(original)
        val decoded = cbor.decodeFromByteArray(ClientToServer.serializer(), bytes)
        assertEquals(original, decoded)
    }

    @Test
    fun clientToServer_createThread_noOverrides_omitsFields() {
        // Serde's skip_serializing_if must round-trip as "fields absent" —
        // our Kotlin nulls should not emit keys on the wire.
        val original = ClientToServer.CreateThread(
            initialMessage = "bare",
            podId = null,
        )
        val bytes = Codec.encodeToServer(original)
        // Expect a map with exactly 2 entries: "type" and "initial_message".
        assertEquals(0xA2.toByte(), bytes[0])
        val decoded = cbor.decodeFromByteArray(ClientToServer.serializer(), bytes)
        assertEquals(original, decoded)
    }

    @Test
    fun serverToClient_backendsList_roundTrip() {
        val original = ServerToClient.BackendsList(
            correlationId = "b-1",
            defaultBackend = "anthropic",
            backends = listOf(
                BackendSummary(
                    name = "anthropic",
                    kind = "anthropic",
                    defaultModel = "claude-opus-4-7",
                    authMode = "api_key",
                ),
                BackendSummary(
                    name = "local-llama",
                    kind = "openai_chat",
                    defaultModel = null,
                    authMode = null,
                ),
            ),
        )
        val bytes = cbor.encodeToByteArray(ServerToClient.serializer(), original)
        val decoded = Codec.decodeFromServer(bytes)
        assertEquals(original, decoded)
    }

    @Test
    fun serverToClient_modelsList_roundTrip() {
        val original = ServerToClient.ModelsList(
            correlationId = "m-1",
            backend = "anthropic",
            models = listOf(
                ModelSummary(id = "claude-opus-4-7", displayName = "Claude Opus 4.7"),
                ModelSummary(id = "claude-sonnet-4-6"),
            ),
        )
        val bytes = cbor.encodeToByteArray(ServerToClient.serializer(), original)
        val decoded = Codec.decodeFromServer(bytes)
        assertEquals(original, decoded)
    }

    // --- Smoke check on the old-broken shape ----------------------------------

    @Test
    fun clientToServer_isNotArrayShape() {
        // Regression guard: kotlinx-cbor's default polymorphic encoding is a
        // 2-element CBOR array (major type 4 = 0x80..0x9F). If someone removes
        // the `@Serializable(with = ...)` annotation, this test catches it.
        val bytes = Codec.encodeToServer(ClientToServer.ListThreads())
        val firstByte = bytes[0].toInt() and 0xFF
        assertTrue(firstByte in 0xA0..0xBF, "expected CBOR map, got first byte 0x${firstByte.toString(16)}")
    }
}
