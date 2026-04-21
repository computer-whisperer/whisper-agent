package com.cjbal.whisperagent.ui.thread

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.cjbal.whisperagent.net.AppSession
import com.cjbal.whisperagent.protocol.ContentBlock
import com.cjbal.whisperagent.protocol.Message
import com.cjbal.whisperagent.protocol.Role
import com.mikepenz.markdown.m3.Markdown

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ThreadScreen(
    threadId: String,
    session: AppSession,
    onBack: () -> Unit,
) {
    DisposableEffect(threadId) {
        session.subscribe(threadId)
        onDispose { session.unsubscribe() }
    }

    val snapshot by session.activeSnapshot.collectAsStateWithLifecycle()
    var draft by remember(threadId) { mutableStateOf("") }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(snapshot?.title?.takeIf { it.isNotBlank() } ?: threadId) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                },
            )
        },
    ) { inner ->
        Column(modifier = Modifier.fillMaxSize().padding(inner)) {
            // Filter out the system + tools prefix; those are plumbing, not chat content.
            val messages = snapshot?.conversation.orEmpty()
                .filter { it.role != Role.System && it.role != Role.Tools }

            val listState = rememberLazyListState()

            // Follow the tail of the conversation as long as the user hasn't
            // scrolled up to read. When they scroll back to the bottom, follow
            // resumes.
            var followBottom by remember(snapshot?.threadId) { mutableStateOf(true) }
            LaunchedEffect(listState) {
                snapshotFlow {
                    val info = listState.layoutInfo
                    val lastVisible = info.visibleItemsInfo.lastOrNull()
                    lastVisible == null || lastVisible.index >= info.totalItemsCount - 1
                }.collect { atBottom -> followBottom = atBottom }
            }

            // Re-scroll whenever either the message count OR the total content
            // length of the last message changes — the latter catches streaming
            // text / reasoning deltas that extend a trailing block in place.
            val contentMarker by remember(snapshot) {
                derivedStateOf {
                    val last = messages.lastOrNull()
                    val lastLen = last?.content?.sumOf { block ->
                        when (block) {
                            is ContentBlock.Text -> block.text.length
                            is ContentBlock.Thinking -> block.thinking.length
                            is ContentBlock.ToolResult -> block.previewText?.length ?: 0
                            else -> 0
                        }
                    } ?: 0
                    messages.size to lastLen
                }
            }
            LaunchedEffect(contentMarker) {
                if (followBottom && messages.isNotEmpty()) {
                    listState.scrollToItem(messages.size - 1)
                }
            }

            LazyColumn(
                state = listState,
                modifier = Modifier
                    .fillMaxSize()
                    .weight(1f, fill = true)
                    .padding(horizontal = 12.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
                contentPadding = PaddingValues(vertical = 12.dp),
            ) {
                items(messages) { msg -> MessageRow(msg) }
            }

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                OutlinedTextField(
                    value = draft,
                    onValueChange = { draft = it },
                    modifier = Modifier.weight(1f),
                    placeholder = { Text("Message") },
                )
                IconButton(
                    onClick = {
                        if (draft.isNotBlank()) {
                            session.submitUserMessage(draft)
                            draft = ""
                        }
                    },
                ) {
                    Icon(Icons.Default.Send, contentDescription = "Send")
                }
            }
        }
    }
}

/**
 * One chat row. User messages go right in a filled bubble; assistant and
 * tool-result messages stay left and render each content block in whatever
 * shape suits it (markdown for text, collapsible chrome for thinking and
 * tool output).
 */
@Composable
private fun MessageRow(msg: Message) {
    when (msg.role) {
        Role.User -> UserMessage(msg)
        Role.Assistant -> AssistantMessage(msg)
        Role.ToolResult -> ToolResultMessage(msg)
        Role.System, Role.Tools -> {
            // Filtered above; defensive no-op if anything slips through.
        }
    }
}

@Composable
private fun UserMessage(msg: Message) {
    val text = msg.content
        .filterIsInstance<ContentBlock.Text>()
        .joinToString("\n\n") { it.text }
    if (text.isBlank()) return
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
        Surface(
            shape = RoundedCornerShape(topStart = 16.dp, topEnd = 16.dp, bottomStart = 16.dp, bottomEnd = 4.dp),
            color = MaterialTheme.colorScheme.primaryContainer,
            modifier = Modifier.widthIn(max = 320.dp),
        ) {
            Text(
                text = text,
                color = MaterialTheme.colorScheme.onPrimaryContainer,
                style = MaterialTheme.typography.bodyLarge,
                modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp),
            )
        }
    }
}

@Composable
private fun AssistantMessage(msg: Message) {
    Column(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        msg.content.forEach { block ->
            when (block) {
                is ContentBlock.Text -> MarkdownText(block.text)
                is ContentBlock.Thinking -> ThinkingBlock(block.thinking)
                is ContentBlock.ToolUse -> ToolUseChip(block)
                is ContentBlock.ToolResult -> ToolResultBlock(block)
                is ContentBlock.ToolSchema -> {
                    // Tool schemas only appear in the filtered-out Role::Tools
                    // prefix; render nothing if one leaks through.
                }
            }
        }
    }
}

@Composable
private fun ToolResultMessage(msg: Message) {
    // A Role::ToolResult message is typically server-injected text (dispatched
    // thread callbacks, etc.) or structured tool_result blocks. Render each
    // block distinctly but keep the whole row styled as tool output.
    Column(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        msg.content.forEach { block ->
            when (block) {
                is ContentBlock.Text -> Surface(
                    shape = RoundedCornerShape(8.dp),
                    color = MaterialTheme.colorScheme.surfaceVariant,
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text(
                        text = block.text,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(12.dp),
                    )
                }
                is ContentBlock.ToolResult -> ToolResultBlock(block)
                else -> { /* shouldn't appear here */ }
            }
        }
    }
}

@Composable
private fun MarkdownText(text: String) {
    Markdown(
        content = text,
        modifier = Modifier.fillMaxWidth(),
    )
}

/**
 * Chain-of-thought block. Collapsed by default — matches the webui's
 * default — since models often produce long reasoning that the user doesn't
 * need to read on every turn.
 */
@Composable
private fun ThinkingBlock(text: String) {
    var expanded by remember(text) { mutableStateOf(false) }
    Surface(
        shape = RoundedCornerShape(8.dp),
        color = MaterialTheme.colorScheme.surfaceContainerHigh,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { expanded = !expanded },
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = "thinking",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    fontStyle = FontStyle.Italic,
                    modifier = Modifier.weight(1f),
                )
                Icon(
                    imageVector = if (expanded) Icons.Default.KeyboardArrowUp else Icons.Default.KeyboardArrowDown,
                    contentDescription = if (expanded) "Collapse" else "Expand",
                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            AnimatedVisibility(visible = expanded) {
                Text(
                    text = text,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    fontStyle = FontStyle.Italic,
                    modifier = Modifier.padding(top = 8.dp),
                )
            }
        }
    }
}

@Composable
private fun ToolUseChip(block: ContentBlock.ToolUse) {
    Surface(
        shape = RoundedCornerShape(8.dp),
        color = MaterialTheme.colorScheme.tertiaryContainer,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Text(
            text = "→ ${block.name}",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onTertiaryContainer,
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
        )
    }
}

@Composable
private fun ToolResultBlock(block: ContentBlock.ToolResult) {
    val color = if (block.isError)
        MaterialTheme.colorScheme.errorContainer
    else
        MaterialTheme.colorScheme.surfaceVariant
    val onColor = if (block.isError)
        MaterialTheme.colorScheme.onErrorContainer
    else
        MaterialTheme.colorScheme.onSurfaceVariant
    Surface(
        shape = RoundedCornerShape(8.dp),
        color = color,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp)) {
            Text(
                text = "← ${if (block.isError) "tool_result (error)" else "tool_result"}",
                style = MaterialTheme.typography.labelSmall,
                color = onColor,
            )
            // `previewText` is populated only from streaming ToolCallEnd events;
            // snapshot-decoded ToolResults leave it null (pending full
            // ToolResultContent modeling).
            block.previewText?.let { preview ->
                Text(
                    text = preview,
                    style = MaterialTheme.typography.bodySmall,
                    color = onColor,
                    modifier = Modifier.padding(top = 4.dp),
                )
            }
        }
    }
}
