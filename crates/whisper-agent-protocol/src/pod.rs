//! Pod config wire types — shared between the server and any client that
//! wants to render or edit a pod's configuration. Owned in this crate (not in
//! `whisper-agent::pod`) because both sides need the same structural shape;
//! TOML parsing and validation stay in the server.
//!
//! `created_at` is an RFC-3339 string rather than a `chrono::DateTime` for
//! the same reason `ThreadSummary.created_at` is — the protocol crate
//! deliberately does not depend on chrono.

use serde::{Deserialize, Serialize};

use crate::permission::AllowMap;
use crate::sandbox::HostEnvSpec;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct PodConfig {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// RFC-3339 timestamp.
    pub created_at: String,

    pub allow: PodAllow,
    pub thread_defaults: ThreadDefaults,
    #[serde(default)]
    pub limits: PodLimits,
    // [[hooks]] reserved for future Lua integration; not parsed yet.
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
pub struct PodAllow {
    #[serde(default)]
    pub backends: Vec<String>,
    #[serde(default)]
    pub mcp_hosts: Vec<String>,
    #[serde(default)]
    pub host_env: Vec<NamedHostEnv>,
    /// Tool gate — default disposition for unlisted tools plus per-tool
    /// overrides. Replaces the old `thread_defaults.approval_policy`
    /// preset enum; threads in the pod inherit this map as their
    /// effective tool scope, which they narrow with their own
    /// remember-approval entries. When omitted, defaults to `allow_all`
    /// (matches the old `AutoApproveAll` preset).
    #[serde(default = "AllowMap::allow_all")]
    pub tools: AllowMap<String>,
}

/// One pod-level "host env" entry — a named (provider, spec) pair the
/// pod's threads are allowed to bind to. The provider name resolves
/// against the server-level catalog in `whisper-agent.toml`. Every
/// `HostEnvSpec` variant refers to a real provisioned environment; a
/// pod that wants "no isolation" for its threads just declares fewer
/// `[[allow.host_env]]` entries (or zero) — threads then run with no
/// host-env MCP connection.
///
/// TOML form:
/// ```toml
/// [[allow.host_env]]
/// name = "rust-dev"
/// provider = "landlock-laptop"
/// type = "landlock"        # HostEnvSpec discriminator
/// allowed_paths = ["/home/me/project:rw", "/:ro"]
/// network = "isolated"
/// ```
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct NamedHostEnv {
    pub name: String,
    pub provider: String,
    #[serde(flatten)]
    pub spec: HostEnvSpec,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ThreadDefaults {
    pub backend: String,
    pub model: String,
    /// Path relative to the pod directory.
    pub system_prompt_file: String,
    pub max_tokens: u32,
    pub max_turns: u32,
    /// Names of `[[allow.host_env]]` entries threads in this pod bind
    /// to by default. Empty allowed (threads fall back to the bare
    /// provider) but if `[allow].host_env` has entries, at least one
    /// must be listed here — the pod validator enforces this.
    ///
    /// On disk accepts both a bare string (legacy pre-multi-env
    /// pods: `host_env = "main"`) and an array (`host_env = ["main",
    /// "edge"]`). Custom deserializer wraps the string form into a
    /// one-entry vec so old pod.tomls load without edits.
    #[serde(default, deserialize_with = "deserialize_string_or_vec")]
    pub host_env: Vec<String>,
    #[serde(default)]
    pub mcp_hosts: Vec<String>,
    /// Compaction defaults threads in this pod inherit. Threads
    /// override via [`crate::ThreadConfigOverride.compaction`].
    #[serde(default)]
    pub compaction: CompactionConfig,
}

/// Accept both a plain string (legacy singular shape) and a
/// `Vec<String>` (new plural shape). Empty string produces an empty
/// vec so the legacy `host_env = ""` idiom still means "no host envs."
pub(crate) fn deserialize_string_or_vec<'de, D>(d: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Shape {
        Many(Vec<String>),
        One(String),
    }
    Ok(match Shape::deserialize(d)? {
        Shape::Many(v) => v,
        Shape::One(s) if s.is_empty() => Vec::new(),
        Shape::One(s) => vec![s],
    })
}

/// Policy for compacting an overlong thread into a fresh continuation.
///
/// Compaction appends the `prompt_file` text (or the built-in default
/// when empty) to the running thread as a final user message, lets the
/// model generate a single summary response bounded by `<summary>…</summary>`,
/// then spawns a new thread seeded with `continuation_template` with
/// `{{summary}}` substituted in. The new thread's
/// [`crate::ThreadSummary.continued_from`] points back at the old one so
/// clients can render the chain.
///
/// Identical shape on both sides of the inheritance chain
/// (`ThreadDefaults` / `ThreadConfig`); [`CompactionConfigOverride`]
/// carries the same fields as `Option`s for per-thread partial overrides.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct CompactionConfig {
    /// Master switch. When false, `/compact` is rejected and
    /// `token_threshold` auto-triggers are skipped.
    #[serde(default = "compaction_enabled_default")]
    pub enabled: bool,
    /// Path to the compaction-instruction file, relative to the pod
    /// directory. Empty ⇒ use the built-in default prompt.
    #[serde(default)]
    pub prompt_file: String,
    /// Regex that extracts the summary body from the model's response.
    /// Group 1 is the summary text. Default matches `<summary>…</summary>`
    /// with leading/trailing whitespace trimmed.
    #[serde(default = "default_summary_regex")]
    pub summary_regex: String,
    /// Auto-compact once the thread's accumulated input tokens exceed
    /// this. `None` ⇒ manual-only. (Auto-trigger lands in a follow-up
    /// commit; the field is already on the wire so pods can declare it.)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_threshold: Option<u32>,
    /// Template for the seed message on the continuation thread.
    /// `{{summary}}` substitutes for the extracted summary body.
    #[serde(default = "default_continuation_template")]
    pub continuation_template: String,
}

fn compaction_enabled_default() -> bool {
    true
}

fn default_summary_regex() -> String {
    r"(?s)<summary>\s*(.*?)\s*</summary>".to_string()
}

fn default_continuation_template() -> String {
    "This thread continues a previous conversation that was compacted for context. \
     The summary below covers the earlier portion; use it as your working context.\n\n\
     {{summary}}"
        .to_string()
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            enabled: compaction_enabled_default(),
            prompt_file: String::new(),
            summary_regex: default_summary_regex(),
            token_threshold: None,
            continuation_template: default_continuation_template(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct PodLimits {
    #[serde(default = "default_max_concurrent_threads")]
    pub max_concurrent_threads: u32,
}

fn default_max_concurrent_threads() -> u32 {
    10
}

impl Default for PodLimits {
    fn default() -> Self {
        Self {
            max_concurrent_threads: default_max_concurrent_threads(),
        }
    }
}

/// Lightweight per-pod entry used by `PodList` responses.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct PodSummary {
    pub pod_id: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub created_at: String,
    pub thread_count: u32,
    pub archived: bool,
    /// Mirrors `PodState.behaviors_enabled`. When false, every
    /// behavior in the pod is gated off regardless of its individual
    /// `BehaviorState.enabled`. Carried here so the pod list can badge
    /// without fetching the full pod state.
    #[serde(default = "pod_behaviors_enabled_default")]
    pub behaviors_enabled: bool,
}

fn pod_behaviors_enabled_default() -> bool {
    true
}

/// Per-pod operational state (not config). Written to
/// `<pod>/pod_state.json`. Kept out of `pod.toml` so operational
/// toggles like pause/resume don't touch version-controlled config.
/// Absent file ⇒ all fields at their defaults.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct PodState {
    /// Master switch for automatic behavior triggers (cron + webhook)
    /// across this pod. Manual `RunBehavior` always works. Default
    /// `true` so a pod without a state file behaves normally.
    #[serde(default = "pod_behaviors_enabled_default")]
    pub behaviors_enabled: bool,
}

impl Default for PodState {
    fn default() -> Self {
        Self {
            behaviors_enabled: true,
        }
    }
}

/// Full pod state delivered in response to `GetPod`. Carries both the parsed
/// config (for structured form rendering) and the raw TOML text (for the raw
/// editor view), so clients pick whichever fits the surface they're showing.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PodSnapshot {
    pub pod_id: String,
    pub config: PodConfig,
    pub toml_text: String,
    pub threads: Vec<crate::ThreadSummary>,
    pub archived: bool,
    /// Behavior catalog — one entry per `<pod>/behaviors/<id>/`. Empty when
    /// the pod has no behaviors subdirectory. Entries whose `behavior.toml`
    /// failed to parse are still present; their `load_error` is populated.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub behaviors: Vec<crate::BehaviorSummary>,
}

/// One entry in a `PodDirListing`. Directories report `size = 0`; the
/// `readonly` flag reflects the server's `is_readonly_path` rule — true
/// for `pod_state.json`, `threads/<id>.json`, and
/// `behaviors/<id>/state.json`, false elsewhere (including for directories).
/// Hidden (dotfile) entries and symlink loops are filtered out server-side.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct FsEntry {
    pub name: String,
    pub is_dir: bool,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub readonly: bool,
}
