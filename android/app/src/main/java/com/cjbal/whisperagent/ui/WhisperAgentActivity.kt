package com.cjbal.whisperagent.ui

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.cjbal.whisperagent.WhisperAgentApp
import com.cjbal.whisperagent.auth.ServerConfig
import com.cjbal.whisperagent.ui.settings.SettingsScreen
import com.cjbal.whisperagent.ui.theme.WhisperAgentTheme
import com.cjbal.whisperagent.ui.thread.ThreadScreen
import com.cjbal.whisperagent.ui.threadlist.ThreadListScreen
import kotlinx.coroutines.flow.Flow

class WhisperAgentActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val app = application as WhisperAgentApp
        setContent {
            WhisperAgentTheme {
                WhisperAgentApp(app)
            }
        }
    }
}

@Composable
private fun WhisperAgentApp(app: WhisperAgentApp) {
    val nav = rememberNavController()
    val configFlow: Flow<ServerConfig?> = remember { app.settings.observe() }
    val config by configFlow.collectAsStateWithLifecycle(initialValue = app.settings.current())

    // Route the user to settings if we don't have credentials yet.
    var firstRouteDecided by remember { mutableStateOf(false) }
    LaunchedEffect(config, firstRouteDecided) {
        if (firstRouteDecided) return@LaunchedEffect
        val dest = if (config == null) Routes.SETTINGS else Routes.THREADS
        nav.navigate(dest) {
            popUpTo(nav.graph.startDestinationId) { inclusive = true }
        }
        firstRouteDecided = true
    }

    // Auto-open threads that CreateThread just spawned. AppSession emits
    // on pendingNavigation when the server acks a thread-creation request.
    LaunchedEffect(Unit) {
        app.session.pendingNavigation.collect { newThreadId ->
            nav.navigate(Routes.thread(newThreadId))
        }
    }

    NavHost(navController = nav, startDestination = Routes.THREADS) {
        composable(Routes.SETTINGS) {
            SettingsScreen(
                initial = config,
                onSave = { cfg ->
                    app.settings.save(cfg)
                    app.session.restart()
                    nav.navigate(Routes.THREADS) {
                        popUpTo(Routes.SETTINGS) { inclusive = true }
                    }
                },
                onClear = {
                    app.settings.clear()
                    app.session.restart()
                },
            )
        }
        composable(Routes.THREADS) {
            ThreadListScreen(
                session = app.session,
                onOpenThread = { id -> nav.navigate(Routes.thread(id)) },
                onOpenSettings = { nav.navigate(Routes.SETTINGS) },
            )
        }
        composable(Routes.THREAD_PATTERN) { entry ->
            val id = entry.arguments?.getString("id").orEmpty()
            ThreadScreen(
                threadId = id,
                session = app.session,
                onBack = { nav.popBackStack() },
            )
        }
    }
}

object Routes {
    const val SETTINGS = "settings"
    const val THREADS = "threads"
    const val THREAD_PATTERN = "thread/{id}"
    fun thread(id: String) = "thread/$id"
}
