package com.cjbal.whisperagent.ui.threadlist

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.cjbal.whisperagent.net.AppSession
import com.cjbal.whisperagent.net.ConnectionState
import com.cjbal.whisperagent.protocol.ThreadSummary

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
        NewThreadDialog(
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

@Composable
private fun NewThreadDialog(
    onDismiss: () -> Unit,
    onCreate: (String) -> Unit,
) {
    var text by remember { mutableStateOf("") }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("New thread") },
        text = {
            Column {
                Text(
                    "Starts a thread in the server's default pod.",
                    style = MaterialTheme.typography.bodySmall,
                )
                Spacer(Modifier.height(8.dp))
                OutlinedTextField(
                    value = text,
                    onValueChange = { text = it },
                    label = { Text("First message") },
                    minLines = 3,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        },
        confirmButton = {
            TextButton(
                onClick = { onCreate(text.trim()) },
                enabled = text.isNotBlank(),
            ) { Text("Create") }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text("Cancel") }
        },
    )
}
