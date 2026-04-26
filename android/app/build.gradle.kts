plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.serialization)
    alias(libs.plugins.kotlin.compose)
}

// CI exports versionCode/versionName via -P flags (or ORG_GRADLE_PROJECT_*
// env vars), derived from the git tag. Local builds fall back to the
// placeholder values below — fine for `assembleDebug`, but every released
// APK must have a strictly-increasing versionCode or Android refuses the
// over-the-top install.
val appVersionCode: Int = (project.findProperty("versionCode") as String?)?.toInt() ?: 1
val appVersionName: String = (project.findProperty("versionName") as String?) ?: "0.1.0-dev"

// Release signing is env-driven so the workflow can supply credentials
// without committing them. When ANDROID_KEYSTORE_PATH is unset (every
// local build), no `release` signing config is registered and
// `assembleRelease` produces an unsigned APK — use `assembleDebug` for
// anything you actually want to install during development.
val releaseKeystorePath: String? = System.getenv("ANDROID_KEYSTORE_PATH")

android {
    namespace = "com.cjbal.whisperagent"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.cjbal.whisperagent"
        minSdk = 26
        targetSdk = 35
        versionCode = appVersionCode
        versionName = appVersionName
    }

    if (releaseKeystorePath != null) {
        signingConfigs {
            create("release") {
                storeFile = file(releaseKeystorePath)
                storePassword = System.getenv("ANDROID_KEYSTORE_PASSWORD")
                keyAlias = System.getenv("ANDROID_KEY_ALIAS")
                keyPassword = System.getenv("ANDROID_KEY_PASSWORD")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            if (releaseKeystorePath != null) {
                signingConfig = signingConfigs.getByName("release")
            }
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        compose = true
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.lifecycle.process)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.navigation.compose)
    implementation(libs.androidx.security.crypto)

    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.ui.graphics)
    implementation(libs.compose.ui.tooling.preview)
    implementation(libs.compose.material3)
    debugImplementation(libs.compose.ui.tooling)

    implementation(libs.kotlinx.serialization.core)
    implementation(libs.kotlinx.serialization.cbor)
    implementation(libs.kotlinx.serialization.json)

    implementation(libs.okhttp)

    implementation(libs.markdown.renderer.m3)

    testImplementation(kotlin("test"))
}
