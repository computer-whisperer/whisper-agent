package com.cjbal.whisperagent.ui.thread

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.cjbal.whisperagent.net.AppSession
import com.cjbal.whisperagent.protocol.ContentBlock
import com.cjbal.whisperagent.protocol.Message
import com.cjbal.whisperagent.protocol.Role

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
            val messages = snapshot?.conversation.orEmpty()
                // Drop system + tools prefix messages from the log view — the
                // webui does the same; they're plumbing, not chat content.
                .filter { it.role != Role.System && it.role != Role.Tools }

            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .weight(1f, fill = true)
                    .padding(horizontal = 12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
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

@Composable
private fun MessageRow(msg: Message) {
    val roleLabel = when (msg.role) {
        Role.User -> "user"
        Role.Assistant -> "assistant"
        Role.ToolResult -> "tool"
        Role.System -> "system"
        Role.Tools -> "tools"
    }
    Column(modifier = Modifier.fillMaxWidth()) {
        Text(text = roleLabel, style = MaterialTheme.typography.labelSmall)
        msg.content.forEach { block -> ContentBlockView(block) }
    }
}

@Composable
private fun ContentBlockView(block: ContentBlock) {
    when (block) {
        is ContentBlock.Text -> Text(block.text, style = MaterialTheme.typography.bodyMedium)
        is ContentBlock.Thinking -> Text(
            "(thinking) ${block.thinking}",
            style = MaterialTheme.typography.bodySmall,
        )
        is ContentBlock.ToolUse -> Text(
            "→ ${block.name}",
            style = MaterialTheme.typography.bodySmall,
        )
        is ContentBlock.ToolResult -> Text(
            "← tool_result${if (block.isError) " (error)" else ""}",
            style = MaterialTheme.typography.bodySmall,
        )
        is ContentBlock.ToolSchema -> Text(
            "schema: ${block.name}",
            style = MaterialTheme.typography.bodySmall,
        )
    }
}
