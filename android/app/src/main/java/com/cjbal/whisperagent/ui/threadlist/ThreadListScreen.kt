package com.cjbal.whisperagent.ui.threadlist

import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
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
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowRight
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.cjbal.whisperagent.net.AppSession
import com.cjbal.whisperagent.net.ConnectionState
import com.cjbal.whisperagent.protocol.BackendSummary
import com.cjbal.whisperagent.protocol.BehaviorSummary
import com.cjbal.whisperagent.protocol.ModelSummary
import com.cjbal.whisperagent.protocol.PodSummary
import com.cjbal.whisperagent.protocol.ThreadStateLabel
import com.cjbal.whisperagent.protocol.ThreadSummary
import kotlinx.coroutines.launch
import java.time.Instant
import java.time.temporal.ChronoUnit

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
    val backends by session.backends.collectAsStateWithLifecycle()
    val defaultBackend by session.defaultBackend.collectAsStateWithLifecycle()
    val modelsByBackend by session.modelsByBackend.collectAsStateWithLifecycle()
    val behaviorsByPod by session.behaviorsByPod.collectAsStateWithLifecycle()

    var showCompose by remember { mutableStateOf(false) }
    // `threadId` of the thread the user long-pressed; non-null → confirmation
    // dialog is open. Kept local to the list screen — archive is a list-only
    // operation (you can't archive the thread you're currently viewing).
    var archiveCandidate by remember { mutableStateOf<ThreadSummary?>(null) }
    val snackbarHost = remember { SnackbarHostState() }
    // Per-behavior expand state. Keyed by behavior_id; default false
    // (collapsed). Session-lived — OK for a list screen that rebuilds on
    // pod switch, and the state only affects the on-screen disclosure.
    val behaviorExpanded = remember { mutableStateMapOf<String, Boolean>() }

    LaunchedEffect(session) {
        session.errors.collect { msg ->
            snackbarHost.showSnackbar(msg)
        }
    }

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
        snackbarHost = { SnackbarHost(snackbarHost) },
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
            // Partition into interactive (user-initiated) threads and
            // behavior-spawned ones grouped by behavior_id. Mirrors the
            // webui's sidebar split.
            val interactive = filtered.filter { it.origin == null }
            val threadsByBehavior: Map<String, List<ThreadSummary>> = filtered
                .mapNotNull { t -> t.origin?.behaviorId?.let { it to t } }
                .groupBy({ it.first }, { it.second })
            val behaviors = selectedPodId?.let { behaviorsByPod[it].orEmpty() }.orEmpty()
            val knownBehaviorIds = behaviors.map { it.behaviorId }.toSet()
            val orphanIds = threadsByBehavior.keys.filterNot { it in knownBehaviorIds }

            if (filtered.isEmpty() && behaviors.isEmpty()) {
                EmptyThreadList(pods, selectedPod)
            } else {
                LazyColumn(modifier = Modifier.fillMaxSize()) {
                    if (interactive.isNotEmpty()) {
                        item(key = "header-interactive") {
                            SectionHeader("Interactive (${interactive.size})")
                        }
                        items(interactive, key = { "t-${it.threadId}" }) { t ->
                            ThreadRow(
                                summary = t,
                                indent = false,
                                onClick = { onOpenThread(t.threadId) },
                                onLongClick = { archiveCandidate = t },
                            )
                            HorizontalDivider()
                        }
                    }

                    if (behaviors.isNotEmpty()) {
                        item(key = "header-behaviors") { SectionHeader("Behaviors") }
                        behaviors.forEach { behavior ->
                            val childThreads = threadsByBehavior[behavior.behaviorId].orEmpty()
                            val expanded = behaviorExpanded[behavior.behaviorId] ?: false
                            item(key = "b-${behavior.behaviorId}") {
                                BehaviorRow(
                                    behavior = behavior,
                                    threadCount = childThreads.size,
                                    expanded = expanded,
                                    onToggle = {
                                        behaviorExpanded[behavior.behaviorId] = !expanded
                                    },
                                    onRun = {
                                        session.runBehavior(behavior.podId, behavior.behaviorId)
                                    },
                                    onToggleEnabled = {
                                        session.setBehaviorEnabled(
                                            behavior.podId,
                                            behavior.behaviorId,
                                            !behavior.enabled,
                                        )
                                    },
                                )
                                HorizontalDivider()
                            }
                            if (expanded) {
                                items(childThreads, key = { "bt-${it.threadId}" }) { t ->
                                    ThreadRow(
                                        summary = t,
                                        indent = true,
                                        onClick = { onOpenThread(t.threadId) },
                                        onLongClick = { archiveCandidate = t },
                                    )
                                    HorizontalDivider()
                                }
                            }
                        }
                    }

                    if (orphanIds.isNotEmpty()) {
                        item(key = "header-orphans") { SectionHeader("Deleted behaviors") }
                        orphanIds.forEach { orphanId ->
                            val childThreads = threadsByBehavior[orphanId].orEmpty()
                            val expanded = behaviorExpanded[orphanId] ?: false
                            item(key = "o-$orphanId") {
                                OrphanBehaviorRow(
                                    behaviorId = orphanId,
                                    threadCount = childThreads.size,
                                    expanded = expanded,
                                    onToggle = {
                                        behaviorExpanded[orphanId] = !expanded
                                    },
                                )
                                HorizontalDivider()
                            }
                            if (expanded) {
                                items(childThreads, key = { "ot-${it.threadId}" }) { t ->
                                    ThreadRow(
                                        summary = t,
                                        indent = true,
                                        onClick = { onOpenThread(t.threadId) },
                                        onLongClick = { archiveCandidate = t },
                                    )
                                    HorizontalDivider()
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    archiveCandidate?.let { candidate ->
        AlertDialog(
            onDismissRequest = { archiveCandidate = null },
            title = { Text("Archive thread?") },
            text = {
                Text(
                    "“${candidate.title?.takeIf { it.isNotBlank() } ?: candidate.threadId}” " +
                        "will stop appearing in this list. History stays on disk.",
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    session.archiveThread(candidate.threadId)
                    archiveCandidate = null
                }) { Text("Archive") }
            },
            dismissButton = {
                TextButton(onClick = { archiveCandidate = null }) { Text("Cancel") }
            },
        )
    }

    if (showCompose) {
        NewThreadSheet(
            targetPod = selectedPod,
            backends = backends,
            defaultBackend = defaultBackend,
            modelsByBackend = modelsByBackend,
            onFetchModels = session::fetchModels,
            onDismiss = { showCompose = false },
            onCreate = { text, backendOverride, modelOverride ->
                session.createThread(text, backendOverride, modelOverride)
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

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun ThreadRow(
    summary: ThreadSummary,
    indent: Boolean,
    onClick: () -> Unit,
    onLongClick: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .combinedClickable(onClick = onClick, onLongClick = onLongClick)
            .padding(
                start = if (indent) 32.dp else 16.dp,
                end = 16.dp,
                top = 12.dp,
                bottom = 12.dp,
            ),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        StateDot(summary.state)
        Column(modifier = Modifier.padding(start = 12.dp).weight(1f)) {
            Text(
                text = summary.title?.takeIf { it.isNotBlank() } ?: summary.threadId,
                style = MaterialTheme.typography.bodyLarge,
                maxLines = 1,
            )
            // Only surface non-obvious states — idle threads don't need an
            // extra line saying "idle".
            stateSubtitle(summary.state)?.let { sub ->
                Text(
                    text = sub,
                    style = MaterialTheme.typography.bodySmall,
                    color = stateColor(summary.state),
                )
            }
        }
    }
}

@Composable
private fun SectionHeader(label: String) {
    Text(
        text = label,
        style = MaterialTheme.typography.labelLarge,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        modifier = Modifier
            .fillMaxWidth()
            .background(MaterialTheme.colorScheme.surfaceVariant)
            .padding(horizontal = 16.dp, vertical = 8.dp),
    )
}

@Composable
private fun BehaviorRow(
    behavior: BehaviorSummary,
    threadCount: Int,
    expanded: Boolean,
    onToggle: () -> Unit,
    onRun: () -> Unit,
    onToggleEnabled: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onToggle)
            .padding(start = 8.dp, end = 8.dp, top = 8.dp, bottom = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(
            imageVector = if (expanded) Icons.Default.KeyboardArrowDown else Icons.AutoMirrored.Filled.KeyboardArrowRight,
            contentDescription = if (expanded) "Collapse" else "Expand",
        )
        Column(modifier = Modifier.padding(start = 4.dp).weight(1f)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = behavior.name,
                    style = MaterialTheme.typography.bodyLarge,
                    maxLines = 1,
                    modifier = Modifier.weight(1f, fill = false),
                )
                behavior.triggerKind?.let { kind ->
                    Text(
                        text = " [$kind]",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                if (!behavior.enabled) {
                    Text(
                        text = " paused",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.secondary,
                    )
                }
                if (behavior.loadError != null) {
                    Text(
                        text = " error",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.error,
                    )
                }
            }
            val subtitle = buildList {
                if (threadCount > 0) add("$threadCount run${if (threadCount == 1) "" else "s"}")
                behavior.lastFiredAt?.let { ts ->
                    formatRelativeTime(ts)?.let { add("last fired $it") }
                }
                if (isEmpty() && behavior.loadError == null) add("no runs yet")
            }.joinToString(" · ")
            if (subtitle.isNotEmpty()) {
                Text(
                    text = subtitle,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            behavior.loadError?.let { err ->
                Text(
                    text = err,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                    maxLines = 2,
                )
            }
        }
        // Manual runs always work, even when the trigger is paused — the
        // server gates enable only the automatic schedulers.
        IconButton(onClick = onRun, enabled = behavior.loadError == null) {
            Icon(Icons.Default.PlayArrow, contentDescription = "Run now")
        }
        // Pause/Resume is text-only — the core material icon set doesn't
        // include a Pause glyph, and pulling in the extended icons package
        // just for this one button isn't worth it.
        TextButton(onClick = onToggleEnabled) {
            Text(if (behavior.enabled) "Pause" else "Resume")
        }
    }
}

@Composable
private fun OrphanBehaviorRow(
    behaviorId: String,
    threadCount: Int,
    expanded: Boolean,
    onToggle: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onToggle)
            .padding(horizontal = 8.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(
            imageVector = if (expanded) Icons.Default.KeyboardArrowDown else Icons.AutoMirrored.Filled.KeyboardArrowRight,
            contentDescription = if (expanded) "Collapse" else "Expand",
        )
        Column(modifier = Modifier.padding(start = 4.dp).weight(1f)) {
            Text(
                text = behaviorId,
                style = MaterialTheme.typography.bodyLarge,
                maxLines = 1,
            )
            Text(
                text = "$threadCount thread${if (threadCount == 1) "" else "s"} · behavior deleted",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

/**
 * Render an RFC-3339 timestamp as "5m ago" / "2h ago" / "3d ago". Returns
 * null if [ts] can't be parsed or is in the future — callers should drop
 * the whole "last fired" clause in those cases rather than render a
 * nonsense value.
 */
private fun formatRelativeTime(ts: String): String? {
    val then = runCatching { Instant.parse(ts) }.getOrNull() ?: return null
    val now = Instant.now()
    val secs = ChronoUnit.SECONDS.between(then, now)
    return when {
        secs < 0 -> null
        secs < 60 -> "just now"
        secs < 3600 -> "${secs / 60}m ago"
        secs < 86_400 -> "${secs / 3600}h ago"
        else -> "${secs / 86_400}d ago"
    }
}

/**
 * Small filled circle that badges a thread's state. Idle and Completed get a
 * neutral disk; working / failed / cancelled get their own color so the list
 * communicates activity at a glance.
 */
@Composable
private fun StateDot(state: ThreadStateLabel) {
    Box(
        modifier = Modifier
            .size(10.dp)
            .clip(CircleShape)
            .background(stateColor(state)),
    )
}

@Composable
private fun stateColor(state: ThreadStateLabel) = when (state) {
    ThreadStateLabel.Working -> MaterialTheme.colorScheme.primary
    ThreadStateLabel.Failed -> MaterialTheme.colorScheme.error
    ThreadStateLabel.Cancelled -> MaterialTheme.colorScheme.onSurfaceVariant
    ThreadStateLabel.Completed -> MaterialTheme.colorScheme.tertiary
    ThreadStateLabel.Idle -> MaterialTheme.colorScheme.outlineVariant
}

private fun stateSubtitle(state: ThreadStateLabel): String? = when (state) {
    ThreadStateLabel.Working -> "working"
    ThreadStateLabel.Failed -> "failed"
    ThreadStateLabel.Cancelled -> "cancelled"
    ThreadStateLabel.Idle, ThreadStateLabel.Completed -> null
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun NewThreadSheet(
    targetPod: PodSummary?,
    backends: List<BackendSummary>,
    defaultBackend: String,
    modelsByBackend: Map<String, List<ModelSummary>>,
    onFetchModels: (String) -> Unit,
    onDismiss: () -> Unit,
    onCreate: (text: String, backendOverride: String?, modelOverride: String?) -> Unit,
) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    val scope = rememberCoroutineScope()
    var text by remember { mutableStateOf("") }
    // `null` means "pod default" on both axes. Changing backend resets model
    // because the model catalog is backend-specific.
    var backendChoice by remember { mutableStateOf<String?>(null) }
    var modelChoice by remember { mutableStateOf<String?>(null) }

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
            BackendDropdown(
                backends = backends,
                defaultBackend = defaultBackend,
                selected = backendChoice,
                onPick = {
                    backendChoice = it
                    // Model catalog depends on backend; reset to "Pod default"
                    // so we never send a model that the new backend doesn't
                    // know about.
                    modelChoice = null
                },
            )
            ModelDropdown(
                backend = backendChoice,
                models = backendChoice?.let { modelsByBackend[it] },
                selected = modelChoice,
                onOpen = { backendChoice?.let(onFetchModels) },
                onPick = { modelChoice = it },
            )
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.End,
            ) {
                TextButton(onClick = { hide(onDismiss) }) { Text("Cancel") }
                Button(
                    onClick = {
                        hide { onCreate(text.trim(), backendChoice, modelChoice) }
                    },
                    enabled = text.isNotBlank(),
                    modifier = Modifier.padding(start = 8.dp),
                ) { Text("Create") }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun BackendDropdown(
    backends: List<BackendSummary>,
    defaultBackend: String,
    selected: String?,
    onPick: (String?) -> Unit,
) {
    var expanded by remember { mutableStateOf(false) }
    // "Pod default" is the authoritative phrasing even though we also know
    // the server's global default — the pod's thread_defaults may pick a
    // different backend, and we don't have that resolved here.
    val display = selected ?: "Pod default"
    ExposedDropdownMenuBox(
        expanded = expanded,
        onExpandedChange = { expanded = !expanded },
    ) {
        OutlinedTextField(
            readOnly = true,
            value = display,
            onValueChange = {},
            label = { Text("Backend") },
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
            modifier = Modifier
                .menuAnchor()
                .fillMaxWidth(),
        )
        DropdownMenu(
            expanded = expanded,
            onDismissRequest = { expanded = false },
            modifier = Modifier.fillMaxWidth(0.9f),
        ) {
            DropdownMenuItem(
                text = { Text("Pod default") },
                onClick = {
                    onPick(null)
                    expanded = false
                },
            )
            backends.forEach { b ->
                DropdownMenuItem(
                    text = {
                        Column {
                            val isDefault = b.name == defaultBackend
                            Text(
                                if (isDefault) "${b.name} (server default)" else b.name,
                            )
                            val sub = buildList {
                                add(b.kind)
                                b.defaultModel?.let { add("default: $it") }
                            }.joinToString(" · ")
                            Text(
                                sub,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    },
                    onClick = {
                        onPick(b.name)
                        expanded = false
                    },
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ModelDropdown(
    backend: String?,
    models: List<ModelSummary>?,
    selected: String?,
    onOpen: () -> Unit,
    onPick: (String?) -> Unit,
) {
    var expanded by remember { mutableStateOf(false) }
    // With no pinned backend, the server picks both. The dropdown is
    // visible but disabled so the user understands why they can't choose
    // a specific model.
    val enabled = backend != null
    val display = when {
        selected != null -> selected
        backend == null -> "Inherits from backend"
        else -> "Backend default"
    }
    ExposedDropdownMenuBox(
        expanded = expanded && enabled,
        onExpandedChange = {
            if (!enabled) return@ExposedDropdownMenuBox
            val next = !expanded
            expanded = next
            if (next) onOpen()
        },
    ) {
        OutlinedTextField(
            readOnly = true,
            enabled = enabled,
            value = display,
            onValueChange = {},
            label = { Text("Model") },
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded && enabled) },
            modifier = Modifier
                .menuAnchor()
                .fillMaxWidth(),
        )
        DropdownMenu(
            expanded = expanded && enabled,
            onDismissRequest = { expanded = false },
            modifier = Modifier.fillMaxWidth(0.9f),
        ) {
            DropdownMenuItem(
                text = { Text("Backend default") },
                onClick = {
                    onPick(null)
                    expanded = false
                },
            )
            when {
                models == null -> DropdownMenuItem(
                    text = { Text("Loading…", color = MaterialTheme.colorScheme.onSurfaceVariant) },
                    enabled = false,
                    onClick = {},
                )
                models.isEmpty() -> DropdownMenuItem(
                    text = {
                        Text(
                            "No models reported — backend default only.",
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    },
                    enabled = false,
                    onClick = {},
                )
                else -> models.forEach { m ->
                    DropdownMenuItem(
                        text = {
                            Column {
                                Text(m.displayName?.takeIf { it.isNotBlank() } ?: m.id)
                                if (!m.displayName.isNullOrBlank() && m.displayName != m.id) {
                                    Text(
                                        m.id,
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    )
                                }
                            }
                        },
                        onClick = {
                            onPick(m.id)
                            expanded = false
                        },
                    )
                }
            }
        }
    }
}

