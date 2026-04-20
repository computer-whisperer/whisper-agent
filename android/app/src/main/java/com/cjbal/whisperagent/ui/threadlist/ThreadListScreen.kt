package com.cjbal.whisperagent.ui.threadlist

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.cjbal.whisperagent.net.AppSession
import com.cjbal.whisperagent.net.ConnectionState
import com.cjbal.whisperagent.protocol.ThreadSummary
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ThreadListScreen(
    session: AppSession,
    onOpenThread: (String) -> Unit,
    onOpenSettings: () -> Unit,
) {
    val threads by session.threads.collectAsStateWithLifecycle()
    val connection by session.connection.collectAsStateWithLifecycle()

    var showCompose by remember { mutableStateOf(false) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Threads") },
                actions = {
                    IconButton(onClick = onOpenSettings) {
                        Icon(Icons.Default.Settings, contentDescription = "Settings")
                    }
                },
            )
        },
        floatingActionButton = {
            FloatingActionButton(onClick = { showCompose = true }) {
                Icon(Icons.Default.Add, contentDescription = "New thread")
            }
        },
    ) { inner ->
        Column(modifier = Modifier.fillMaxSize().padding(inner)) {
            ConnectionBanner(connection)
            val ordered = threads.values.sortedByDescending { it.lastActive }
            LazyColumn(modifier = Modifier.fillMaxSize()) {
                items(ordered, key = { it.threadId }) { t ->
                    ThreadRow(t, onClick = { onOpenThread(t.threadId) })
                    HorizontalDivider()
                }
            }
        }
    }

    if (showCompose) {
        NewThreadSheet(
            onDismiss = { showCompose = false },
            onCreate = { text ->
                session.createThread(text)
                showCompose = false
            },
        )
    }
}

@Composable
private fun ConnectionBanner(state: ConnectionState) {
    val text = when (state) {
        ConnectionState.Connecting -> "connecting…"
        ConnectionState.Connected -> null
        ConnectionState.Disconnected -> "disconnected"
        is ConnectionState.Errored -> "error: ${state.detail}"
    } ?: return
    Text(
        text = text,
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 8.dp),
        style = MaterialTheme.typography.bodySmall,
    )
}

@Composable
private fun ThreadRow(summary: ThreadSummary, onClick: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
            .padding(horizontal = 16.dp, vertical = 12.dp),
    ) {
        Text(
            text = summary.title?.takeIf { it.isNotBlank() } ?: summary.threadId,
            style = MaterialTheme.typography.bodyLarge,
        )
        Text(
            text = summary.state.name.lowercase(),
            style = MaterialTheme.typography.bodySmall,
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun NewThreadSheet(
    onDismiss: () -> Unit,
    onCreate: (String) -> Unit,
) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    val scope = rememberCoroutineScope()
    var text by remember { mutableStateOf("") }

    // Collapse the sheet with animation before notifying the parent, so the
    // sheet closes smoothly instead of snapping shut from underneath the FAB.
    fun hide(action: () -> Unit) {
        scope.launch {
            sheetState.hide()
            action()
        }
    }

    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = sheetState,
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .imePadding()
                .navigationBarsPadding()
                .padding(horizontal = 16.dp)
                .padding(bottom = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Text("New thread", style = MaterialTheme.typography.titleMedium)
            Text(
                "Starts a thread in the server's default pod.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            OutlinedTextField(
                value = text,
                onValueChange = { text = it },
                label = { Text("First message") },
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 160.dp),
            )
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.End,
            ) {
                TextButton(onClick = { hide(onDismiss) }) { Text("Cancel") }
                Button(
                    onClick = { hide { onCreate(text.trim()) } },
                    enabled = text.isNotBlank(),
                    modifier = Modifier.padding(start = 8.dp),
                ) { Text("Create") }
            }
        }
    }
}

