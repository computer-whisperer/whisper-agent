package com.cjbal.whisperagent.ui.thread

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.clickable
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.cjbal.whisperagent.net.AppSession
import com.cjbal.whisperagent.net.PendingSudo
import com.cjbal.whisperagent.protocol.ContentBlock
import com.cjbal.whisperagent.protocol.Message
import com.cjbal.whisperagent.protocol.Role
import com.cjbal.whisperagent.protocol.SudoDecision
import com.cjbal.whisperagent.protocol.ThreadSnapshot
import com.cjbal.whisperagent.protocol.ThreadStateLabel
import com.mikepenz.markdown.m3.Markdown
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ThreadScreen(
    threadId: String,
    session: AppSession,
    onBack: () -> Unit,
) {
    DisposableEffect(threadId) {
        session.subscribe(threadId)
        onDispose { session.unsubscribe(threadId) }
    }

    val snapshot by session.activeSnapshot.collectAsStateWithLifecycle()
    val pendingSudoMap by session.pendingSudo.collectAsStateWithLifecycle()
    val pods by session.pods.collectAsStateWithLifecycle()
    var draft by remember(threadId) { mutableStateOf("") }

    val isWorking = snapshot?.state == ThreadStateLabel.Working
    val isFailed = snapshot?.state == ThreadStateLabel.Failed
    val pendingSudo = pendingSudoMap[threadId]
    // Tracks the long-pressed message so we can open the action sheet. Pair
    // carries (absolute conversation index, message) — the absolute index is
    // what ForkThread needs; we can't derive it from the filtered display list.
    var actionTarget by remember(threadId) { mutableStateOf<MessageAction?>(null) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    ThreadHeader(
                        threadId = threadId,
                        snapshot = snapshot,
                        podName = snapshot?.podId?.let { pods[it]?.name }?.ifBlank { null },
                    )
                },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                },
                actions = {
                    if (isWorking) {
                        CircularProgressIndicator(
                            strokeWidth = 2.dp,
                            modifier = Modifier.size(18.dp).padding(end = 8.dp),
                        )
                        IconButton(onClick = { session.cancelThread(threadId) }) {
                            Icon(Icons.Default.Close, contentDescription = "Cancel")
                        }
                    }
                    ThreadOverflowMenu(
                        canCompact = snapshot != null && !isWorking,
                        onArchive = { session.archiveThread(threadId); onBack() },
                        onCompact = { session.compactThread(threadId) },
                    )
                },
            )
        },
    ) { inner ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(inner)
                .imePadding(),
        ) {
            val conversation = snapshot?.conversation.orEmpty()
            // Preserve the absolute conversation index on each message — fork
            // uses it, and without it we couldn't map a filtered row back to
            // the position the server expects.
            val indexedMessages = remember(conversation) {
                conversation.mapIndexedNotNull { idx, msg ->
                    if (msg.role == Role.System || msg.role == Role.Tools) null
                    else IndexedMessage(idx, msg)
                }
            }

            val listState = rememberLazyListState()

            // `followBottom` tracks "is the trailing edge of the list on
            // screen?" — meaning the user is caught up and wants new
            // content to keep auto-scrolling. The check pairs with the
            // `scrollToItem(last, Int.MAX_VALUE)` call below: both sides
            // agree that "the bottom" means the true last item's
            // trailing edge aligned with the viewport's trailing edge.
            // Checking the trailing edge (rather than mere visibility
            // of the last-index item) stops a tall streaming message
            // from fighting the user when they scroll mid-reply.
            var followBottom by remember(snapshot?.threadId) { mutableStateOf(true) }
            LaunchedEffect(listState) {
                snapshotFlow {
                    val info = listState.layoutInfo
                    val last = info.visibleItemsInfo.lastOrNull()
                    when {
                        last == null -> true
                        last.index < info.totalItemsCount - 1 -> false
                        else -> (last.offset + last.size) <= info.viewportEndOffset
                    }
                }.collect { atBottom -> followBottom = atBottom }
            }

            val lastBlockLen = indexedMessages.lastOrNull()?.message?.content?.sumOf { block ->
                when (block) {
                    is ContentBlock.Text -> block.text.length
                    is ContentBlock.Thinking -> block.thinking.length
                    is ContentBlock.ToolResult -> block.previewText?.length ?: 0
                    else -> 0
                }
            } ?: 0
            val contentMarker = indexedMessages.size to lastBlockLen
            LaunchedEffect(contentMarker) {
                if (!followBottom) return@LaunchedEffect
                val total = listState.layoutInfo.totalItemsCount
                if (total <= 0) return@LaunchedEffect
                // Int.MAX_VALUE as scroll offset asks the lazy list to
                // scroll the last item as far toward the start as it
                // can. LazyList clamps that request so the item's
                // trailing edge lands on the viewport's trailing edge —
                // the chat-follows-bottom behaviour, regardless of
                // whether the last message is a two-line stub or a
                // page-long essay.
                listState.scrollToItem(total - 1, Int.MAX_VALUE)
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
                items(indexedMessages, key = { it.absoluteIndex }) { entry ->
                    MessageRow(
                        msg = entry.message,
                        onLongPress = {
                            // Only user + assistant rows get an action sheet —
                            // tool-result rows are noisy and don't have a
                            // useful copyable body.
                            if (entry.message.role == Role.User || entry.message.role == Role.Assistant) {
                                actionTarget = MessageAction(
                                    absoluteIndex = entry.absoluteIndex,
                                    role = entry.message.role,
                                    text = extractPlainText(entry.message),
                                )
                            }
                        },
                    )
                }
                // Quiet total-usage footer once the conversation has started.
                snapshot?.totalUsage?.let { usage ->
                    if (indexedMessages.isNotEmpty()) {
                        item(key = "total-usage") {
                            UsageFooter(
                                inputTokens = usage.inputTokens,
                                outputTokens = usage.outputTokens,
                                cacheReadTokens = usage.cacheReadInputTokens,
                            )
                        }
                    }
                }
            }

            if (isFailed) {
                FailureBanner(
                    failure = snapshot?.failure,
                    onRecover = { session.recoverThread(threadId) },
                )
            }

            if (pendingSudo != null) {
                SudoPromptCard(
                    pending = pendingSudo,
                    onApproveOnce = { session.resolveSudo(threadId, SudoDecision.ApproveOnce) },
                    onApproveRemember = { session.resolveSudo(threadId, SudoDecision.ApproveRemember) },
                    onReject = { session.resolveSudo(threadId, SudoDecision.Reject) },
                )
            }

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .navigationBarsPadding()
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

    actionTarget?.let { target ->
        MessageActionSheet(
            target = target,
            onDismiss = { actionTarget = null },
            onFork = { archiveOriginal ->
                session.forkThread(
                    threadId = threadId,
                    fromMessageIndex = target.absoluteIndex,
                    archiveOriginal = archiveOriginal,
                )
                actionTarget = null
            },
        )
    }
}

private data class IndexedMessage(val absoluteIndex: Int, val message: Message)

/** What the long-press action sheet is operating on. */
private data class MessageAction(
    val absoluteIndex: Int,
    val role: Role,
    val text: String,
)

private fun extractPlainText(msg: Message): String =
    msg.content.mapNotNull {
        when (it) {
            is ContentBlock.Text -> it.text
            is ContentBlock.Thinking -> it.thinking
            else -> null
        }
    }.joinToString("\n\n")

@Composable
private fun ThreadHeader(
    threadId: String,
    snapshot: ThreadSnapshot?,
    podName: String?,
) {
    val title = snapshot?.title?.takeIf { it.isNotBlank() } ?: threadId
    // Combine pod + model + non-happy state into a compact subtitle. Idle and
    // Completed don't contribute (no signal the user cares about); Working is
    // already conveyed by the spinner in the top bar actions.
    val stateLabel = when (snapshot?.state) {
        ThreadStateLabel.Failed -> "failed"
        ThreadStateLabel.Cancelled -> "cancelled"
        else -> null
    }
    val parts = listOfNotNull(
        podName,
        snapshot?.config?.model?.takeIf { it.isNotBlank() },
        stateLabel,
    )
    Column {
        Text(title, style = MaterialTheme.typography.titleMedium)
        if (parts.isNotEmpty()) {
            Text(
                text = parts.joinToString(" · "),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun ThreadOverflowMenu(
    canCompact: Boolean,
    onArchive: () -> Unit,
    onCompact: () -> Unit,
) {
    var expanded by remember { mutableStateOf(false) }
    IconButton(onClick = { expanded = true }) {
        Icon(Icons.Default.MoreVert, contentDescription = "More")
    }
    DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
        DropdownMenuItem(
            text = { Text("Archive") },
            onClick = { expanded = false; onArchive() },
        )
        DropdownMenuItem(
            text = { Text("Compact") },
            enabled = canCompact,
            onClick = { expanded = false; onCompact() },
        )
    }
}

/**
 * Compact summary of thread-wide token usage. Rendered as a dimmed footer
 * at the end of the conversation so it doesn't compete with message content.
 */
@Composable
private fun UsageFooter(inputTokens: Int, outputTokens: Int, cacheReadTokens: Int) {
    if (inputTokens == 0 && outputTokens == 0 && cacheReadTokens == 0) return
    val parts = buildList {
        add("in ${formatTokens(inputTokens)}")
        add("out ${formatTokens(outputTokens)}")
        if (cacheReadTokens > 0) add("cached ${formatTokens(cacheReadTokens)}")
    }
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.Center,
    ) {
        Text(
            text = parts.joinToString(" · "),
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

private fun formatTokens(n: Int): String = when {
    n < 1_000 -> n.toString()
    n < 1_000_000 -> "%.1fk".format(n / 1_000.0)
    else -> "%.1fM".format(n / 1_000_000.0)
}

@Composable
private fun FailureBanner(failure: String?, onRecover: () -> Unit) {
    Surface(
        shape = RoundedCornerShape(12.dp),
        color = MaterialTheme.colorScheme.errorContainer,
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 12.dp, vertical = 4.dp),
    ) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(
                text = "Thread failed",
                style = MaterialTheme.typography.titleSmall,
                color = MaterialTheme.colorScheme.onErrorContainer,
            )
            val body = failure?.takeIf { it.isNotBlank() } ?: "The scheduler reported a failure on this thread."
            Text(
                text = body,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onErrorContainer,
            )
            Row(horizontalArrangement = Arrangement.End, modifier = Modifier.fillMaxWidth()) {
                Button(onClick = onRecover) {
                    Icon(
                        imageVector = Icons.Default.Refresh,
                        contentDescription = null,
                        modifier = Modifier.size(18.dp),
                    )
                    Text("Recover", modifier = Modifier.padding(start = 6.dp))
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun MessageActionSheet(
    target: MessageAction,
    onDismiss: () -> Unit,
    onFork: (archiveOriginal: Boolean) -> Unit,
) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    val scope = rememberCoroutineScopeForSheet()
    val clipboard = LocalClipboardManager.current

    fun hide(action: () -> Unit) {
        scope.launch {
            sheetState.hide()
            action()
        }
    }

    ModalBottomSheet(onDismissRequest = onDismiss, sheetState = sheetState) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .navigationBarsPadding()
                .padding(horizontal = 16.dp, vertical = 8.dp),
        ) {
            val roleLabel = when (target.role) {
                Role.User -> "user message"
                Role.Assistant -> "assistant message"
                else -> "message"
            }
            Text(
                text = roleLabel,
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(bottom = 4.dp),
            )
            if (target.text.isNotBlank()) {
                Text(
                    text = target.text.lineSequence().take(3).joinToString("\n"),
                    maxLines = 3,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(bottom = 8.dp),
                )
            }
            SheetAction("Copy") {
                if (target.text.isNotEmpty()) {
                    clipboard.setText(AnnotatedString(target.text))
                }
                hide(onDismiss)
            }
            if (target.role == Role.User) {
                SheetAction("Fork from here (archive original)") {
                    hide { onFork(true) }
                }
                SheetAction("Fork from here (keep original)") {
                    hide { onFork(false) }
                }
            }
        }
    }
}

@Composable
private fun SheetAction(label: String, onClick: () -> Unit) {
    TextButton(
        onClick = onClick,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Row(modifier = Modifier.fillMaxWidth()) {
            Text(label)
        }
    }
}

@Composable
private fun rememberCoroutineScopeForSheet() = androidx.compose.runtime.rememberCoroutineScope()

/**
 * Banner shown between the conversation and the input bar when the model has
 * requested sudo for an out-of-scope tool call.
 */
@Composable
private fun SudoPromptCard(
    pending: PendingSudo,
    onApproveOnce: () -> Unit,
    onApproveRemember: () -> Unit,
    onReject: () -> Unit,
) {
    Surface(
        shape = RoundedCornerShape(12.dp),
        color = MaterialTheme.colorScheme.tertiaryContainer,
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 12.dp, vertical = 4.dp),
    ) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(
                text = "Sudo requested · ${pending.toolName}",
                style = MaterialTheme.typography.titleSmall,
                color = MaterialTheme.colorScheme.onTertiaryContainer,
            )
            if (pending.reason.isNotBlank()) {
                Text(
                    text = pending.reason,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onTertiaryContainer,
                )
            }
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.End),
            ) {
                TextButton(onClick = onReject) { Text("Reject") }
                OutlinedButton(onClick = onApproveOnce) { Text("Once") }
                Button(onClick = onApproveRemember) { Text("Remember") }
            }
        }
    }
}

/**
 * One chat row. User messages go right in a filled bubble; assistant and
 * tool-result messages stay left and render each content block in whatever
 * shape suits it (markdown for text, collapsible chrome for thinking and
 * tool output).
 *
 * [onLongPress] fires for rows that carry a useful action menu; rows without
 * one (tool results) still get the same modifier for uniform touch behavior,
 * the callback just no-ops via [MessageActionSheet] gating upstream.
 */
@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun MessageRow(msg: Message, onLongPress: () -> Unit) {
    val rowModifier = Modifier
        .fillMaxWidth()
        .combinedClickable(
            onClick = {},
            onLongClick = onLongPress,
        )
    when (msg.role) {
        Role.User -> UserMessage(msg, rowModifier)
        Role.Assistant -> AssistantMessage(msg, rowModifier)
        Role.ToolResult -> ToolResultMessage(msg, rowModifier)
        Role.System, Role.Tools -> {
            // Filtered above; defensive no-op if anything slips through.
        }
    }
}

@Composable
private fun UserMessage(msg: Message, modifier: Modifier) {
    val text = msg.content
        .filterIsInstance<ContentBlock.Text>()
        .joinToString("\n\n") { it.text }
    if (text.isBlank()) return
    Row(modifier = modifier, horizontalArrangement = Arrangement.End) {
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
private fun AssistantMessage(msg: Message, modifier: Modifier) {
    Column(
        modifier = modifier,
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
private fun ToolResultMessage(msg: Message, modifier: Modifier) {
    Column(
        modifier = modifier,
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
        Column(modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp)) {
            Text(
                text = "→ ${block.name}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onTertiaryContainer,
            )
            block.argsPreview?.takeIf { it.isNotBlank() }?.let { preview ->
                Text(
                    text = preview,
                    style = MaterialTheme.typography.bodySmall,
                    fontFamily = FontFamily.Monospace,
                    color = MaterialTheme.colorScheme.onTertiaryContainer,
                    modifier = Modifier.padding(top = 2.dp),
                )
            }
        }
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
