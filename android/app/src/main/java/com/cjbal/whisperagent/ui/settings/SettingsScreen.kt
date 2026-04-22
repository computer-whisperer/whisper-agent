package com.cjbal.whisperagent.ui.settings

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.OutlinedButton
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
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.unit.dp
import com.cjbal.whisperagent.auth.ServerConfig

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    initial: ServerConfig?,
    onSave: (ServerConfig) -> Unit,
    onClear: () -> Unit = {},
) {
    var url by remember { mutableStateOf(initial?.url ?: "https://whisper-agent.cjbal.com") }
    var token by remember { mutableStateOf(initial?.token.orEmpty()) }
    var showToken by remember { mutableStateOf(false) }

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
                // Tokens are sensitive enough not to render on a phone you
                // might hand around, but also annoying to paste blind — toggle
                // lets the user verify before saving.
                visualTransformation = if (showToken) VisualTransformation.None
                else PasswordVisualTransformation(),
                trailingIcon = {
                    // Using a text button instead of Visibility / VisibilityOff
                    // icons — those live in material-icons-extended, which
                    // would add ~5 KLOC of icons for one toggle.
                    TextButton(onClick = { showToken = !showToken }) {
                        Text(if (showToken) "Hide" else "Show")
                    }
                },
                modifier = Modifier.fillMaxWidth(),
            )
            Button(
                onClick = { onSave(ServerConfig(url = url.trim(), token = token.trim())) },
                enabled = url.isNotBlank() && token.isNotBlank(),
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text("Save and connect")
            }
            if (initial != null) {
                OutlinedButton(
                    onClick = onClear,
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text("Clear credentials")
                }
            }
        }
    }
}
