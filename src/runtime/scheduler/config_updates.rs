//! Handlers that apply in-memory config mutations produced by the
//! builtin-file tools (edits to `pod.toml`, `system_prompt.md`,
//! `behavior.toml`, or a behavior's `prompt.md`).
//!
//! The builtin tool wrote the change to disk first; these handlers
//! refresh the scheduler's in-memory pod/behavior state and broadcast
//! the resulting summary on the wire. Each tool kind has its own apply
//! helper because the refresh step differs (re-parse config, reload
//! prompt, seed behavior state, etc.).
//!
//! Also hosts `apply_scheduler_command` — the side channel a tool can
//! use to ask the scheduler to run or pause a behavior without routing
//! through the client message dispatch.

use futures::stream::FuturesUnordered;
use tracing::warn;
use whisper_agent_protocol::ServerToClient;

use super::{ConnId, Scheduler};
use crate::runtime::io_dispatch::SchedulerFuture;

impl Scheduler {
    /// Apply a [`PodUpdate`] side effect produced by a builtin tool.
    /// Dispatches to the specific apply helper based on the update
    /// kind; each helper handles the in-memory refresh + broadcast.
    /// The builtin tool already wrote the change to disk — no further
    /// persist work is needed here.
    pub(super) fn apply_pod_update(
        &mut self,
        thread_id: &str,
        update: crate::tools::builtin_tools::PodUpdate,
    ) {
        let Some(task) = self.tasks.get(thread_id) else {
            warn!(%thread_id, "apply_pod_update: task not found");
            return;
        };
        let pod_id = task.pod_id.clone();
        match update {
            crate::tools::builtin_tools::PodUpdate::Config { toml_text, parsed } => {
                self.apply_pod_config_update(&pod_id, toml_text, *parsed, None);
            }
            crate::tools::builtin_tools::PodUpdate::SystemPrompt { text } => {
                self.apply_system_prompt_update(&pod_id, text, None);
            }
            crate::tools::builtin_tools::PodUpdate::BehaviorConfig {
                behavior_id,
                toml_text,
                parsed,
            } => {
                self.apply_behavior_config_update(&pod_id, behavior_id, toml_text, *parsed);
            }
            crate::tools::builtin_tools::PodUpdate::BehaviorPrompt { behavior_id, text } => {
                self.apply_behavior_prompt_update(&pod_id, behavior_id, text);
            }
        }
    }

    /// Refresh (or insert) an in-memory behavior from a successful
    /// `behavior.toml` write, then broadcast `BehaviorCreated` or
    /// `BehaviorUpdated` to every connected client. The builtin tool
    /// has already validated the config and written it to disk (plus
    /// seeded `state.json` on create) before we get here.
    fn apply_behavior_config_update(
        &mut self,
        pod_id: &str,
        behavior_id: String,
        toml_text: String,
        parsed: whisper_agent_protocol::BehaviorConfig,
    ) {
        let Some(pod) = self.pods.get_mut(pod_id) else {
            warn!(%pod_id, %behavior_id, "apply_behavior_config_update: unknown pod");
            return;
        };
        let cron = crate::pod::behaviors::cache_cron(&parsed);
        let is_create = !pod.behaviors.contains_key(&behavior_id);
        if is_create {
            let dir = pod
                .dir
                .join(crate::pod::behaviors::BEHAVIORS_DIR)
                .join(&behavior_id);
            let behavior = crate::pod::behaviors::Behavior {
                id: behavior_id.clone(),
                pod_id: pod_id.to_string(),
                dir,
                cron,
                config: Some(parsed),
                raw_toml: toml_text,
                prompt: String::new(),
                state: whisper_agent_protocol::BehaviorState::default(),
                load_error: None,
            };
            let summary = behavior.summary();
            pod.behaviors.insert(behavior_id.clone(), behavior);
            let ev = ServerToClient::BehaviorCreated {
                correlation_id: None,
                summary,
            };
            for tx in self.router.outbound_snapshot() {
                let _ = tx.send(ev.clone());
            }
        } else {
            let behavior = pod
                .behaviors
                .get_mut(&behavior_id)
                .expect("contains_key above");
            behavior.cron = cron;
            behavior.config = Some(parsed);
            behavior.raw_toml = toml_text;
            behavior.load_error = None;
            let snapshot = behavior.snapshot();
            let ev = ServerToClient::BehaviorUpdated {
                correlation_id: None,
                snapshot,
            };
            for tx in self.router.outbound_snapshot() {
                let _ = tx.send(ev.clone());
            }
        }
    }

    /// Apply a [`crate::tools::builtin_tools::SchedulerCommand`] produced
    /// by a builtin orchestration tool (pod_run_behavior /
    /// pod_set_behavior_enabled). Routes through the same handlers
    /// that serve the equivalent wire messages, so behavior broadcast
    /// shape matches exactly.
    pub(super) fn apply_scheduler_command(
        &mut self,
        thread_id: &str,
        command: crate::tools::builtin_tools::SchedulerCommand,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        let Some(task) = self.tasks.get(thread_id) else {
            warn!(%thread_id, "apply_scheduler_command: task not found");
            return;
        };
        let pod_id = task.pod_id.clone();
        match command {
            crate::tools::builtin_tools::SchedulerCommand::RunBehavior {
                behavior_id,
                payload,
            } => {
                if let Err(e) =
                    self.run_behavior(None, None, &pod_id, &behavior_id, payload, pending_io)
                {
                    warn!(%pod_id, %behavior_id, error = %e,
                        "tool-initiated run_behavior failed to spawn");
                }
            }
            crate::tools::builtin_tools::SchedulerCommand::SetBehaviorEnabled {
                behavior_id,
                enabled,
            } => {
                // conn_id = 0 is fine here — the handler only uses it
                // for error-reply routing, and our validation already
                // rejected unknown-behavior cases at the tool layer.
                // Any lookup failure here logs via warn through
                // send_behavior_error; see handle_set_behavior_enabled.
                self.handle_set_behavior_enabled(0, None, pod_id, behavior_id, enabled);
            }
        }
    }

    /// Refresh an in-memory behavior's `prompt` field from a successful
    /// `prompt.md` write and broadcast `BehaviorUpdated`. The tool
    /// layer rejects prompt writes for unknown ids, so we don't expect
    /// to reach here without an existing behavior — a missing entry
    /// logs a warn and drops the broadcast.
    fn apply_behavior_prompt_update(&mut self, pod_id: &str, behavior_id: String, text: String) {
        let Some(pod) = self.pods.get_mut(pod_id) else {
            warn!(%pod_id, %behavior_id, "apply_behavior_prompt_update: unknown pod");
            return;
        };
        let Some(behavior) = pod.behaviors.get_mut(&behavior_id) else {
            warn!(%pod_id, %behavior_id, "apply_behavior_prompt_update: unknown behavior");
            return;
        };
        behavior.prompt = text;
        let snapshot = behavior.snapshot();
        let ev = ServerToClient::BehaviorUpdated {
            correlation_id: None,
            snapshot,
        };
        for tx in self.router.outbound_snapshot() {
            let _ = tx.send(ev.clone());
        }
    }

    /// Refresh the in-memory pod with a validated new config + raw TOML
    /// and broadcast `PodConfigUpdated` to every connected client. Does
    /// NOT touch disk — callers that originate on-wire updates (the
    /// UpdatePodConfig handler) separately invoke [`persist_pod_config`];
    /// the builtin-tool path has already written the file before
    /// reaching here.
    pub(super) fn apply_pod_config_update(
        &mut self,
        pod_id: &str,
        toml_text: String,
        parsed: whisper_agent_protocol::PodConfig,
        correlation_id: Option<String>,
    ) {
        if let Some(pod) = self.pods.get_mut(pod_id) {
            pod.config = parsed.clone();
            pod.raw_toml = toml_text.clone();
        }
        let ev = ServerToClient::PodConfigUpdated {
            pod_id: pod_id.to_string(),
            toml_text,
            parsed,
            correlation_id,
        };
        for tx in self.router.outbound_snapshot() {
            let _ = tx.send(ev.clone());
        }
    }

    /// Refresh the in-memory pod's cached system prompt and broadcast
    /// `PodSystemPromptUpdated`. Same split as
    /// [`apply_pod_config_update`] — builtin-tool callers have already
    /// written the file; wire-originated callers spawn
    /// [`persist_system_prompt`] alongside.
    fn apply_system_prompt_update(
        &mut self,
        pod_id: &str,
        text: String,
        correlation_id: Option<String>,
    ) {
        if let Some(pod) = self.pods.get_mut(pod_id) {
            pod.system_prompt = text.clone();
        }
        let ev = ServerToClient::PodSystemPromptUpdated {
            pod_id: pod_id.to_string(),
            text,
            correlation_id,
        };
        for tx in self.router.outbound_snapshot() {
            let _ = tx.send(ev.clone());
        }
    }

    /// Spawn a background task that flushes a new pod.toml to disk. On
    /// failure the pod is still live in memory (the broadcast already
    /// went out); we log and optionally surface an error to the
    /// originating connection.
    pub(super) fn persist_pod_config(
        &self,
        pod_id: String,
        toml_text: String,
        err_conn: Option<ConnId>,
    ) {
        let Some(persister) = self.persister.clone() else {
            return;
        };
        let outbound = err_conn.and_then(|c| self.router.outbound(c));
        tokio::spawn(async move {
            if let Err(e) = persister.update_pod_config(&pod_id, &toml_text).await {
                warn!(pod_id = %pod_id, error = %e, "update_pod_config disk write failed");
                if let Some(tx) = outbound {
                    let _ = tx.send(ServerToClient::Error {
                        correlation_id: None,
                        thread_id: None,
                        message: format!("update_pod_config: disk write failed: {e}"),
                    });
                }
            }
        });
    }
}
