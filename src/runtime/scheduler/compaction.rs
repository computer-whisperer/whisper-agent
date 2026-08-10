//! Compaction — appending a summarize-yourself prompt to an overlong
//! thread, then rolling the weave's head onto a fresh continuation
//! thread seeded with the extracted summary. Prompt mechanics follow
//! the Claude Code pattern in `docs/research/compaction_claude_code.md`.
//!
//! Since step 11 slice 6 the flow is entirely DRIVER-composED: the
//! scheduler's half is admission + resolution (this module — the
//! shared admission for manual and driver-requested compaction, the
//! prompt/regex/template resolution that rides `compaction_ready`,
//! and the inheritance helpers `derive_thread`'s granular `_from`
//! directives copy with), while the driver appends the prompt, runs
//! the summary turn, extracts with `regex_capture`, derives the
//! continuation, and advances the head. The old head is left
//! `Completed` in-place as a dormant auxiliary — drill-down history,
//! no longer ticked; fork is the deliberate revive. A compaction does
//! NOT survive restart: the persister heals the in-flight summary
//! turn to `Failed`, and the driver clears its own marker on the
//! `thread_failed` load notice (titled_chat's contract, pinned by the
//! slice-5 harness tests).

use whisper_agent_protocol::{ContentBlock, Role, ThreadConfigOverride};

use super::Scheduler;

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
5. All user messages from the ORIGINAL conversation, in order, verbatim — so the next session has the user's own words. DO NOT include this compaction instruction you are responding to; it is harness scaffolding, not a user message.
6. Current work — what you were doing immediately before this summary request. Include direct quotes from the most recent original user message (not from this compaction instruction).
7. Pending and next step — what's still open. If a next step is clear from the most recent exchange, name it and quote the user verbatim; otherwise note that the conversation concluded.

Do not preface your response. Do not call tools. End after the closing </summary> tag.";

impl Scheduler {
    /// Overwrite a freshly derived thread's setup prefix (seeded by
    /// `seed_thread_setup` inside `create_task` from current pod
    /// state) with a source thread's snapshot. Copying verbatim keeps
    /// the derived thread's system prompt and tool manifest identical
    /// to what the source was running under — behaviors,
    /// custom-prompted threads, and mid-life pod edits all settle out
    /// the same way (the derive inherits, pod drift doesn't leak
    /// across the boundary) — the `derive_thread { setup_from }`
    /// directive's copy machinery (step 11 slice 5; extracted from the
    /// retired builtin finalize).
    ///
    /// `source_profiles` is the source thread's resolved profile map,
    /// used to realign participants other than the default responder;
    /// the responder's fields re-derive from the copied prefix itself
    /// (legacy parents have no profile map).
    pub(super) fn apply_setup_snapshot(
        &mut self,
        new_thread_id: &str,
        parent_setup: Vec<whisper_agent_protocol::Message>,
        source_profiles: &std::collections::BTreeMap<
            whisper_agent_protocol::ParticipantId,
            whisper_agent_protocol::ParticipantExecutionProfile,
        >,
    ) {
        if let Some(new_task) = self.tasks.get_mut(new_thread_id) {
            let tail: Vec<whisper_agent_protocol::Message> = new_task
                .conversation
                .messages()
                .iter()
                .skip(new_task.conversation.setup_prefix_end())
                .cloned()
                .collect();
            new_task.conversation = whisper_agent_protocol::Conversation::new();
            for msg in parent_setup {
                new_task.conversation.push(msg);
            }
            for msg in tail {
                new_task.conversation.push(msg);
            }
            // Keep the normalized profiles aligned with the verbatim setup
            // snapshot above. Legacy parents have no profile map, so derive
            // the compatibility responder's fields from the copied prefix;
            // additional participants inherit their already-frozen private
            // setup from the parent profile.
            let default_responder = new_task.config.participants.default_responder.clone();
            let default_system_prompt = new_task.conversation.system_prompt_text().to_string();
            let default_tools: Vec<_> = new_task
                .conversation
                .tool_schemas()
                .map(|tool| whisper_agent_protocol::ToolSchema {
                    name: tool.name.to_string(),
                    description: tool.description.to_string(),
                    params: tool.params.to_vec(),
                    kind: tool.kind,
                })
                .collect();
            let context_start = new_task.conversation.setup_prefix_end();
            let context_end = new_task.conversation.initial_context_end();
            let default_context =
                new_task.conversation.messages()[context_start..context_end].to_vec();
            if let Some(profile) = new_task
                .config
                .participant_profiles
                .get_mut(&default_responder)
            {
                profile.system_prompt = default_system_prompt;
                profile.tools = default_tools;
                profile.context = default_context;
            }
            for (participant_id, old_profile) in source_profiles {
                if participant_id == &default_responder {
                    continue;
                }
                if let Some(profile) = new_task.config.participant_profiles.get_mut(participant_id)
                {
                    profile.system_prompt = old_profile.system_prompt.clone();
                    profile.tools = old_profile.tools.clone();
                    profile.context = old_profile.context.clone();
                }
            }
        }
        self.mark_dirty(new_thread_id);
    }

    /// Shared admission + resolution for scripted compaction requests
    /// (step 11 slice 5) — the driver's `request_compaction` effect and
    /// the client's manual CompactThread message on a scripted weave
    /// both come through here. The weave must tick the thread as its
    /// current primary and the thread's config must enable compaction;
    /// idleness is deliberately NOT checked — the driver knows its own
    /// parking discipline and stores the request until quiescence.
    /// Returns the resolved texts (`prompt_file` read at request time,
    /// not creation time, so pod edits keep applying).
    pub(super) fn resolve_scripted_compaction(
        &self,
        weave_id: &str,
        thread_id: &str,
    ) -> Result<ResolvedCompaction, String> {
        let Some(weave) = self.weaves.get(weave_id) else {
            return Err(format!("unknown weave `{weave_id}`"));
        };
        if weave.primary_thread_id() != Some(thread_id) {
            return Err("thread is not the weave's current primary; only the head compacts".into());
        }
        let ticks = weave
            .threads
            .iter()
            .any(|r| r.thread_id == thread_id && r.ticks)
            && self.thread_ticker.get(thread_id).map(String::as_str) == Some(weave_id);
        if !ticks {
            return Err("the weave does not tick this thread".into());
        }
        let Some(task) = self.tasks.get(thread_id) else {
            return Err(format!("thread `{thread_id}` no longer exists"));
        };
        if !task.config.compaction.enabled {
            return Err("compaction is disabled for this thread".into());
        }
        let prompt =
            self.resolve_compaction_prompt(&task.pod_id, &task.config.compaction.prompt_file)?;
        Ok(ResolvedCompaction {
            prompt,
            summary_regex: task.config.compaction.summary_regex.clone(),
            continuation_template: task.config.compaction.continuation_template.clone(),
        })
    }

    /// Read the compaction prompt text: either the pod-relative file
    /// named by `prompt_file` when non-empty, or the built-in fallback.
    /// Relative paths are resolved against the pod directory; the file
    /// must exist and be valid UTF-8.
    pub(super) fn resolve_compaction_prompt(
        &self,
        pod_id: &str,
        prompt_file: &str,
    ) -> Result<String, String> {
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

/// A thread's compaction config resolved for a scripted driver: the
/// prompt text (pod-relative file or built-in default), the Rust-regex
/// summary extractor, and the continuation seed template. Rides into
/// Lua on the `compaction_ready` event.
pub(super) struct ResolvedCompaction {
    pub prompt: String,
    pub summary_regex: String,
    pub continuation_template: String,
}

/// Config-side inheritance snapshot — the continuation carries the
/// source thread's coordination topology and generation knobs
/// verbatim — the `derive_thread { config_from }` directive's copy
/// machinery (step 11 slice 5; extracted from the retired builtin
/// finalize). `compaction`/`autoquery` deliberately stay `None`: the
/// derived thread re-inherits the pod's defaults for those
/// (per-thread overrides do not outlive the head they were set on).
pub(super) fn inherited_config_override(
    old_config: &whisper_agent_protocol::ThreadConfig,
) -> ThreadConfigOverride {
    ThreadConfigOverride {
        // Carry the participant registry verbatim. The compatibility
        // driver will invoke the same default responder, while
        // scripted drivers retain the full topology across derivation.
        participants: Some(old_config.participants.clone()),
        driver: Some(old_config.driver.clone()),
        participant_profiles: (!old_config.participant_profiles.is_empty()).then(|| {
            old_config
                .participant_profiles
                .iter()
                .map(|(participant_id, profile)| {
                    (
                        participant_id.clone(),
                        whisper_agent_protocol::ParticipantExecutionProfileRequest {
                            model: Some(profile.model.clone()),
                            max_tokens: Some(profile.max_tokens),
                            system_prompt: Some(whisper_agent_protocol::SystemPromptChoice::Text {
                                text: profile.system_prompt.clone(),
                            }),
                            bindings: super::bindings_request_from_resolved(&profile.bindings),
                            scope: Some(profile.scope.clone()),
                            tool_surface: Some(profile.tool_surface.clone()),
                            tunables: profile.tunables.clone(),
                        },
                    )
                })
                .collect()
        }),
        model: Some(old_config.model.clone()),
        max_tokens: Some(old_config.max_tokens),
        max_turns: Some(old_config.max_turns),
        // System-prompt override is intentionally `None` here: when the
        // caller also snapshots the setup prefix, `create_task` seeds a
        // fresh System message that `apply_setup_snapshot` then
        // overwrites with the source's verbatim text. Forwarding a
        // file/text override would be redundant and lose that text.
        system_prompt: None,
        compaction: None, // inherit pod's compaction defaults again
        autoquery: None,
        caps: None,
        tools: None,
        knowledge_buckets: None,
        tool_surface: None,
        // Carry the source's tunables verbatim so the derived thread
        // re-issues model calls with the same backend knobs.
        tunables: (!old_config.tunables.is_empty()).then(|| old_config.tunables.clone()),
    }
}

/// Bindings-side inheritance snapshot: the source's host-env list
/// verbatim — same pod allows both source and derived thread, so every
/// entry still resolves, workspace_root pins included. `Inline`
/// variants aren't addressable by name; drop those and inherit the pod
/// default for that slot (rare: `Inline` only exists on the reserved
/// subagent path). The `derive_thread { bindings_from }` directive's
/// copy machinery (extracted from the retired builtin finalize).
pub(super) fn inherited_bindings_request(
    old_bindings: &whisper_agent_protocol::ThreadBindings,
) -> whisper_agent_protocol::ThreadBindingsRequest {
    let inherited_host_env: Vec<whisper_agent_protocol::HostEnvBindingRequest> = old_bindings
        .host_env
        .iter()
        .filter_map(|b| match b {
            whisper_agent_protocol::HostEnvBinding::Named {
                name,
                workspace_root,
                runas,
                options,
            } => Some(whisper_agent_protocol::HostEnvBindingRequest {
                name: name.clone(),
                workspace_root: workspace_root.clone(),
                runas: runas.clone(),
                options: options.clone(),
            }),
            whisper_agent_protocol::HostEnvBinding::Inline { .. } => None,
        })
        .collect();
    whisper_agent_protocol::ThreadBindingsRequest {
        backend: Some(old_bindings.backend.clone()),
        host_env: Some(inherited_host_env),
        mcp_hosts: Some(old_bindings.mcp_hosts.clone()),
    }
}

/// The source thread's setup prefix (system prompt + tool manifest),
/// snapshotted for [`Scheduler::apply_setup_snapshot`].
pub(super) fn setup_prefix_snapshot(
    task: &crate::runtime::thread::Thread,
) -> Vec<whisper_agent_protocol::Message> {
    let setup_end = task.conversation.setup_prefix_end();
    task.conversation.messages()[..setup_end].to_vec()
}

/// Walk backward through the conversation to find the most recent
/// assistant message and concatenate all its text blocks.
pub(super) fn extract_last_assistant_text(task: &crate::runtime::thread::Thread) -> String {
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
