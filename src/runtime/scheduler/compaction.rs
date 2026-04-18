//! Compaction — appending a summarize-yourself prompt to an overlong
//! thread, then spawning a fresh continuation thread seeded with the
//! extracted summary.
//!
//! The mechanism follows the Claude Code pattern documented in
//! `docs/research/compaction_claude_code.md`:
//!
//!   1. Client sends `CompactThread { thread_id }`.
//!   2. [`Scheduler::begin_compaction`] validates, flips
//!      `thread.compacting = true`, and appends the thread's configured
//!      compaction prompt as a final user message.
//!   3. The normal step loop runs the model turn.
//!   4. When the turn ends on `end_turn`,
//!      [`Scheduler::finalize_pending_compaction`] fires: it parses the
//!      `<summary>` out of the assistant's response, spawns a new
//!      thread inheriting the same config with
//!      `continued_from = Some(old_thread_id)`, and seeds it by sending
//!      the continuation-template user message.
//!   5. The old thread is left `Completed` in-place (compaction boundary
//!      preserved as history); the new thread becomes the active one.
//!
//! The `compacting` flag lives on `Thread` (not a scheduler side-map)
//! so a compaction in flight survives process restart — on the next
//! startup the model turn still completes against the already-appended
//! user message and the finalize runs as expected.

use futures::stream::FuturesUnordered;
use regex::Regex;
use tracing::{debug, warn};
use whisper_agent_protocol::{ContentBlock, Role, ServerToClient, ThreadConfigOverride};

use super::Scheduler;
use crate::runtime::io_dispatch::SchedulerFuture;

/// Built-in fallback prompt when a thread's `CompactionConfig.prompt_file`
/// is empty. Stripped-down version of the Claude Code compaction prompt
/// (see `docs/research/compaction_claude_code.md`, Appendix A) tuned
/// for whisper-agent's domain — a smaller section schema that works
/// for interactive and long-running behavior threads alike.
pub(super) const BUILTIN_COMPACTION_PROMPT: &str = r"CRITICAL: Respond with TEXT ONLY. Do NOT call any tools. Tool calls will be rejected and waste your only turn.

Produce a compact summary of this conversation that preserves every detail needed to resume the work without loss of context. Wrap the summary in a single <summary>...</summary> block. Inside, include these sections in order:

1. User's primary requests and intent — explicit asks, changes of direction, clarifications. Quote user feedback verbatim when they corrected an approach.
2. Key technical concepts — the domain, frameworks, APIs, and architectural choices shaping the work.
3. Files and code — everything you examined, modified, or created, with enough detail (paths, functions, snippets) that the next turn could resume edits.
4. Errors and fixes — problems hit and how they were resolved.
5. All user messages that are not tool results — in order, so the next session has the user's own words.
6. Current work — what you were doing immediately before this summary request. Include direct quotes from the latest user message.
7. Pending and next step — what's still open. If a next step is clear from the most recent exchange, name it and quote the user verbatim; otherwise note that the conversation concluded.

Do not preface your response. Do not call tools. End after the closing </summary> tag.";

/// Entry point for `ClientToServer::CompactThread`. Validates the
/// request, resolves the thread's compaction prompt, flips
/// `compacting = true`, and appends the prompt as a user message —
/// the same path as `SendUserMessage`, reusing its state transitions
/// and broadcasts. The finalize (summary parse + continuation spawn)
/// fires later via [`Scheduler::finalize_pending_compaction`] when the
/// model turn completes.
impl Scheduler {
    pub(super) fn begin_compaction(
        &mut self,
        thread_id: &str,
        _correlation_id: Option<String>,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) -> Result<(), String> {
        let task = self
            .tasks
            .get(thread_id)
            .ok_or_else(|| "unknown task".to_string())?;
        if !task.config.compaction.enabled {
            return Err("compaction is disabled on this thread".into());
        }
        if task.compacting {
            return Err("compaction already in progress".into());
        }
        if !task.is_idle() {
            return Err("thread is not at a clean turn boundary".into());
        }
        let pod_id = task.pod_id.clone();
        let prompt_file = task.config.compaction.prompt_file.clone();
        let prompt_text = self.resolve_compaction_prompt(&pod_id, &prompt_file)?;

        // Flip the flag first so the finalize hook can see it when the
        // turn completes below.
        if let Some(task) = self.tasks.get_mut(thread_id) {
            task.compacting = true;
        }
        // Reuse send_user_message so title/state broadcasts and dirty
        // tracking run the same as any user follow-up.
        self.send_user_message(thread_id, prompt_text, pending_io);
        self.step_until_blocked(thread_id, pending_io);
        Ok(())
    }

    /// Hook called after `step_until_blocked` returns. When the given
    /// thread is mid-compaction and has reached `Completed`, parse the
    /// `<summary>` from its most recent assistant message, spawn a
    /// continuation thread inheriting the same config, seed it, and
    /// broadcast `ThreadCompacted`.
    ///
    /// No-op when the thread isn't compacting or isn't terminal yet.
    /// Safe to call unconditionally from `step_until_blocked`.
    pub(super) fn finalize_pending_compaction(
        &mut self,
        thread_id: &str,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        let task = match self.tasks.get(thread_id) {
            Some(t) => t,
            None => return,
        };
        if !task.compacting {
            return;
        }
        if !matches!(
            task.public_state(),
            whisper_agent_protocol::ThreadStateLabel::Completed
        ) {
            return;
        }

        // Look for the trailing assistant message; it carries the
        // summary. If the turn failed (Failed state isn't Completed
        // anyway, so we don't reach here for that) or the model
        // declined to emit the block, we clear the compacting flag
        // and bail without spawning a continuation.
        let assistant_text = extract_last_assistant_text(task);
        let regex_src = task.config.compaction.summary_regex.clone();
        let continuation_template = task.config.compaction.continuation_template.clone();
        let pod_id = task.pod_id.clone();
        let old_bindings = task.bindings.clone();
        let old_config = task.config.clone();
        let old_origin = task.origin.clone();

        // Always clear the flag so a failed parse doesn't re-trigger
        // the finalize on every subsequent step.
        if let Some(task) = self.tasks.get_mut(thread_id) {
            task.compacting = false;
            self.mark_dirty(thread_id);
        }

        let Some(summary_text) = extract_summary(&regex_src, &assistant_text) else {
            warn!(
                %thread_id,
                "compaction finalize: failed to extract <summary> from assistant response — \
                 leaving thread Completed without spawning continuation"
            );
            return;
        };

        // Spawn the continuation. We route through `create_task` so
        // bindings are re-resolved (same backend / sandbox / shared
        // MCPs, but the registry sees a fresh user) and the usual
        // `ThreadCreated` broadcast fires.
        let config_override = Some(ThreadConfigOverride {
            model: Some(old_config.model.clone()),
            system_prompt: Some(old_config.system_prompt.clone()),
            max_tokens: Some(old_config.max_tokens),
            max_turns: Some(old_config.max_turns),
            approval_policy: Some(old_config.approval_policy),
            compaction: None, // inherit pod's compaction defaults again
        });
        let bindings_request = Some(whisper_agent_protocol::ThreadBindingsRequest {
            backend: Some(old_bindings.backend.clone()),
            host_env: old_bindings.host_env.clone().and_then(|b| match b {
                whisper_agent_protocol::HostEnvBinding::Named { name } => Some(name),
                // Inline bindings aren't currently addressable from a
                // request; drop to inherit-pod-default.
                whisper_agent_protocol::HostEnvBinding::Inline { .. } => None,
            }),
            mcp_hosts: Some(old_bindings.mcp_hosts.clone()),
        });
        let new_thread_id = match self.create_task(
            None,
            None,
            Some(pod_id.clone()),
            config_override,
            bindings_request,
            old_origin,
            pending_io,
        ) {
            Ok(id) => id,
            Err(e) => {
                warn!(
                    %thread_id, error = %e,
                    "compaction finalize: create_task for continuation failed"
                );
                return;
            }
        };

        // Stamp the continuation linkage. `create_task` has already
        // broadcast `ThreadCreated` for this thread with
        // `continued_from = None` in its summary — the authoritative
        // linkage arrives via the `ThreadCompacted` broadcast below,
        // and any newly-joining client gets the stamped field from
        // `ThreadSnapshot` / `ThreadList`.
        if let Some(new_task) = self.tasks.get_mut(&new_thread_id) {
            new_task.continued_from = Some(thread_id.to_string());
        }
        self.mark_dirty(&new_thread_id);
        self.router
            .broadcast_task_list(ServerToClient::ThreadCompacted {
                thread_id: thread_id.to_string(),
                new_thread_id: new_thread_id.clone(),
                summary_text: summary_text.clone(),
                correlation_id: None,
            });

        // Seed the continuation with the filled-in template. This
        // kicks the thread's first model call.
        let seed_text = render_continuation_template(&continuation_template, &summary_text);
        self.send_user_message(&new_thread_id, seed_text, pending_io);
        self.step_until_blocked(&new_thread_id, pending_io);

        debug!(
            old = %thread_id, new = %new_thread_id, summary_bytes = summary_text.len(),
            "compaction finalize: spawned continuation"
        );
    }

    /// Read the compaction prompt text: either the pod-relative file
    /// named by `prompt_file` when non-empty, or the built-in fallback.
    /// Relative paths are resolved against the pod directory; the file
    /// must exist and be valid UTF-8.
    fn resolve_compaction_prompt(&self, pod_id: &str, prompt_file: &str) -> Result<String, String> {
        if prompt_file.is_empty() {
            return Ok(BUILTIN_COMPACTION_PROMPT.to_string());
        }
        let pod_dir = self
            .pods
            .get(pod_id)
            .map(|p| p.dir.clone())
            .ok_or_else(|| format!("unknown pod `{pod_id}`"))?;
        let path = pod_dir.join(prompt_file);
        std::fs::read_to_string(&path)
            .map_err(|e| format!("read compaction prompt `{}`: {e}", path.display()))
    }
}

/// Walk backward through the conversation to find the most recent
/// assistant message and concatenate all its text blocks.
fn extract_last_assistant_text(task: &crate::runtime::thread::Thread) -> String {
    for msg in task.conversation.messages().iter().rev() {
        if msg.role == Role::Assistant {
            let mut out = String::new();
            for block in &msg.content {
                if let ContentBlock::Text { text } = block {
                    if !out.is_empty() {
                        out.push('\n');
                    }
                    out.push_str(text);
                }
            }
            return out;
        }
    }
    String::new()
}

/// Pull the summary body out of the assistant's response. Compile-time
/// regex would be cheaper, but the regex source lives in config so the
/// user can tweak the parse without a code change; the per-compaction
/// compile cost is negligible against the model turn itself.
fn extract_summary(regex_src: &str, text: &str) -> Option<String> {
    let re = Regex::new(regex_src).ok()?;
    re.captures(text)
        .and_then(|caps| caps.get(1).map(|m| m.as_str().to_string()))
}

/// Substitute `{{summary}}` in the template with the extracted body.
/// Deliberately minimal — matches the `{{payload}}` substitution
/// behaviors use for webhook-triggered behaviors.
fn render_continuation_template(template: &str, summary: &str) -> String {
    if !template.contains("{{summary}}") {
        return template.to_string();
    }
    template.replace("{{summary}}", summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_summary_from_tagged_text() {
        let regex_src = r"(?s)<summary>\s*(.*?)\s*</summary>";
        let text = "prose\n<summary>\nthe body\n</summary>\nafter";
        assert_eq!(
            extract_summary(regex_src, text).as_deref(),
            Some("the body")
        );
    }

    #[test]
    fn returns_none_when_tag_missing() {
        let regex_src = r"(?s)<summary>\s*(.*?)\s*</summary>";
        assert!(extract_summary(regex_src, "nothing to see").is_none());
    }

    #[test]
    fn template_substitutes_summary() {
        let out = render_continuation_template("prefix\n{{summary}}\nsuffix", "THE BODY");
        assert_eq!(out, "prefix\nTHE BODY\nsuffix");
    }

    #[test]
    fn template_without_placeholder_returns_as_is() {
        let out = render_continuation_template("no placeholder", "ignored");
        assert_eq!(out, "no placeholder");
    }
}
