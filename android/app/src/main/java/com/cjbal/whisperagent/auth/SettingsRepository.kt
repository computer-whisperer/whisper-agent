package com.cjbal.whisperagent.auth

import android.content.Context
import android.content.SharedPreferences
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow

class SettingsRepository(appContext: Context) {

    private val prefs: SharedPreferences = run {
        val masterKey = MasterKey.Builder(appContext)
            .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
            .build()
        EncryptedSharedPreferences.create(
            appContext,
            PREFS_NAME,
            masterKey,
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM,
        )
    }

    fun current(): ServerConfig? {
        val url = prefs.getString(KEY_URL, null) ?: return null
        val token = prefs.getString(KEY_TOKEN, null) ?: return null
        if (url.isBlank() || token.isBlank()) return null
        return ServerConfig(url = url, token = token)
    }

    fun save(config: ServerConfig) {
        prefs.edit()
            .putString(KEY_URL, config.url)
            .putString(KEY_TOKEN, config.token)
            .apply()
    }

    fun clear() {
        prefs.edit().clear().apply()
    }

    /** Emits the current config immediately, then re-emits whenever it changes. */
    fun observe(): Flow<ServerConfig?> = callbackFlow {
        trySend(current())
        val listener = SharedPreferences.OnSharedPreferenceChangeListener { _, _ ->
            trySend(current())
        }
        prefs.registerOnSharedPreferenceChangeListener(listener)
        awaitClose { prefs.unregisterOnSharedPreferenceChangeListener(listener) }
    }

    companion object {
        private const val PREFS_NAME = "whisper_agent_settings"
        private const val KEY_URL = "server_url"
        private const val KEY_TOKEN = "server_token"
    }
}
