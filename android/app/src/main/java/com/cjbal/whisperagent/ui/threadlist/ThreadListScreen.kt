package com.cjbal.whisperagent.ui.threadlist

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.IconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Settings
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
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
