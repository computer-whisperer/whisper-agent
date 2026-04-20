package com.cjbal.whisperagent.ui.settings

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.cjbal.whisperagent.auth.ServerConfig

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    initial: ServerConfig?,
    onSave: (ServerConfig) -> Unit,
) {
    var url by remember { mutableStateOf(initial?.url ?: "https://whisper-agent.cjbal.com") }
    var token by remember { mutableStateOf(initial?.token.orEmpty()) }

    Scaffold(
        topBar = { TopAppBar(title = { Text("Server") }) },
    ) { inner ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(inner)
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            OutlinedTextField(
                value = url,
                onValueChange = { url = it },
                label = { Text("Server URL") },
                singleLine = true,
                modifier = Modifier.fillMaxWidth(),
            )
            OutlinedTextField(
                value = token,
                onValueChange = { token = it },
                label = { Text("Bearer token") },
                singleLine = true,
                modifier = Modifier.fillMaxWidth(),
            )
            Button(
                onClick = { onSave(ServerConfig(url = url.trim(), token = token.trim())) },
                enabled = url.isNotBlank() && token.isNotBlank(),
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text("Save and connect")
            }
        }
    }
}
