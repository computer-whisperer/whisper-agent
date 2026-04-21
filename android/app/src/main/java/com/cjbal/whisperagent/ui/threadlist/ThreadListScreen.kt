package com.cjbal.whisperagent.ui.threadlist

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
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
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
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
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.cjbal.whisperagent.net.AppSession
import com.cjbal.whisperagent.net.ConnectionState
import com.cjbal.whisperagent.protocol.PodSummary
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
    val pods by session.pods.collectAsStateWithLifecycle()
    val selectedPodId by session.selectedPodId.collectAsStateWithLifecycle()
    val defaultPodId by session.defaultPodId.collectAsStateWithLifecycle()

    var showCompose by remember { mutableStateOf(false) }

    val selectedPod = selectedPodId?.let { pods[it] }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    PodPickerTitle(
                        pods = pods,
                        selected = selectedPod,
                        defaultPodId = defaultPodId,
                        onPick = session::selectPod,
                    )
                },
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
            val filtered = threads.values
                .asSequence()
                .filter { selectedPodId == null || it.podId == selectedPodId }
                .sortedByDescending { it.lastActive }
                .toList()
            if (filtered.isEmpty()) {
                EmptyThreadList(pods, selectedPod)
            } else {
                LazyColumn(modifier = Modifier.fillMaxSize()) {
                    items(filtered, key = { it.threadId }) { t ->
                        ThreadRow(t, onClick = { onOpenThread(t.threadId) })
                        HorizontalDivider()
                    }
                }
            }
        }
    }

    if (showCompose) {
        NewThreadSheet(
            targetPod = selectedPod,
            onDismiss = { showCompose = false },
            onCreate = { text ->
                session.createThread(text)
                showCompose = false
            },
        )
    }
}

@Composable
private fun PodPickerTitle(
    pods: Map<String, PodSummary>,
    selected: PodSummary?,
    defaultPodId: String,
    onPick: (String) -> Unit,
) {
    var expanded by remember { mutableStateOf(false) }
    val label = selected?.name?.ifBlank { selected.podId }
        ?: if (pods.isEmpty()) "Threads" else "Select pod"

    Box {
        Row(
            modifier = Modifier
                .clickable(enabled = pods.isNotEmpty()) { expanded = true }
                .padding(vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(label, style = MaterialTheme.typography.titleLarge)
            if (pods.isNotEmpty()) {
                Icon(Icons.Default.ArrowDropDown, contentDescription = "Pick pod")
            }
        }
        DropdownMenu(
            expanded = expanded,
            onDismissRequest = { expanded = false },
        ) {
            // Stable order by name so the menu doesn't reshuffle under the
            // user between PodList refreshes.
            val ordered = pods.values.sortedBy { it.name.ifBlank { it.podId } }
            ordered.forEach { pod ->
                val isDefault = pod.podId == defaultPodId
                DropdownMenuItem(
                    text = {
                        Column {
                            Text(pod.name.ifBlank { pod.podId })
                            val subtitleParts = buildList {
                                add("${pod.threadCount} thread${if (pod.threadCount == 1) "" else "s"}")
                                if (isDefault) add("default")
                            }
                            Text(
                                subtitleParts.joinToString(" · "),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    },
                    onClick = {
                        onPick(pod.podId)
                        expanded = false
                    },
                )
            }
        }
    }
}

@Composable
private fun EmptyThreadList(
    pods: Map<String, PodSummary>,
    selected: PodSummary?,
) {
    val message = when {
        pods.isEmpty() -> "Waiting for pods…"
        selected == null -> "Pick a pod to see its threads."
        else -> "No threads in ${selected.name.ifBlank { selected.podId }} yet."
    }
    Box(
        modifier = Modifier.fillMaxSize().padding(32.dp),
        contentAlignment = Alignment.Center,
    ) {
        Text(
            message,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
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
    targetPod: PodSummary?,
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
            val podLabel = targetPod?.name?.ifBlank { targetPod.podId }
            Text(
                if (podLabel != null) "Starts a thread in $podLabel."
                else "Starts a thread in the server's default pod.",
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
