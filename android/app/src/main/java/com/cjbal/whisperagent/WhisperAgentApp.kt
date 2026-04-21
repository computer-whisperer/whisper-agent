package com.cjbal.whisperagent

import android.app.Application
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.ProcessLifecycleOwner
import com.cjbal.whisperagent.auth.SettingsRepository
import com.cjbal.whisperagent.net.AppSession
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob

class WhisperAgentApp : Application() {

    lateinit var settings: SettingsRepository
        private set
    lateinit var session: AppSession
        private set

    private val appScope: CoroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    override fun onCreate() {
        super.onCreate()
        settings = SettingsRepository(applicationContext)
        session = AppSession(appScope, settings)

        ProcessLifecycleOwner.get().lifecycle.addObserver(object : DefaultLifecycleObserver {
            override fun onStart(owner: LifecycleOwner) {
                session.onAppForegrounded()
            }

            override fun onStop(owner: LifecycleOwner) {
                session.onAppBackgrounded()
            }
        })
    }
}
