//! Compaction — appending a summarize-yourself prompt to an overlong
//! thread, then rolling the weave's head onto a fresh continuation
//! thread seeded with the extracted summary.
//!
//! Weave-native since migration step 8 (prompt mechanics still follow
//! the Claude Code pattern in `docs/research/compaction_claude_code.md`):
//!
//!   1. Client sends `CompactThread { thread_id }` (or the auto-trigger
//!      fires with a `CallerLink::Weave` caller). Admission requires the
//!      thread to be its ticking weave's current primary.
//!   2. [`Scheduler::launch_compact_thread`] stamps the weave's builtin
//!      driver state (`compacting = Some(thread_id)`) and appends the
//!      thread's configured compaction prompt as a final user message.
//!   3. The normal step loop runs the model turn.
//!   4. When the builtin driver finishes the cycle,
//!      [`Scheduler::finalize_builtin_compaction`] fires from the
//!      boundary path: it parses the `<summary>` out of the assistant's
//!      response, derives a continuation thread into the same weave
//!      (journaled `derive_thread` with a `{kind: "compaction"}`
//!      relationship edge), and advances the head (journaled
//!      `advance_head`) so the continuation becomes the primary.
//!   5. The old thread is left `Completed` in-place as a dormant
//!      auxiliary — drill-down history, no longer ticked. Fork is the
//!      deliberate revive.
//!
//! The `compacting` marker lives on the weave's persisted driver state.
//! A compaction does NOT survive restart: the persister heals every
//! in-flight thread to `Failed`, so `Scheduler::load_state` clears any
//! set marker (with a warning) rather than letting it wedge admission
//! or mis-trigger the finalize against a later ordinary turn. (The
//! pre-step-8 `InFlightOps::COMPACTING` bit claimed restart-resume in
//! its docs; that claim was false for the same heal-to-Failed reason.)

use futures::stream::FuturesUnordered;
use regex::Regex;
use tracing::{debug, warn};
use whisper_agent_protocol::{ContentBlock, Role, ThreadConfigOverride};

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
5. All user messages from the ORIGINAL conversation, in order, verbatim — so the next session has the user's own words. DO NOT include this compaction instruction you are responding to; it is harness scaffolding, not a user message.
6. Current work — what you were doing immediately before this summary request. Include direct quotes from the most recent original user message (not from this compaction instruction).
7. Pending and next step — what's still open. If a next step is clear from the most recent exchange, name it and quote the user verbatim; otherwise note that the conversation concluded.

Do not preface your response. Do not call tools. End after the closing </summary> tag.";

/// Execution body for `Function::CompactThread`. Called by the Function
/// registry's `launch_function` after the synchronous precondition
/// check has already verified `compaction.enabled`, idle state, the
/// thread being its weave's current primary, and no compaction already
/// running on the weave.
///
/// Resolves the thread's compaction prompt, stamps the weave's builtin
/// driver state with the compacting marker, and appends the prompt as a
/// user message — the same path as `SendUserMessage`, reusing its state
/// transitions and broadcasts. The Function stays in `active_functions`
/// until [`Scheduler::finalize_builtin_compaction`] fires
/// `complete_function` when the builtin driver finishes the cycle.
impl Scheduler {
    pub(super) fn launch_compact_thread(
        &mut self,
        thread_id: &str,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        let Some(task) = self.tasks.get(thread_id) else {
            // Precondition already checked existence; defensive.
            return;
        };
        let pod_id = task.pod_id.clone();
        let prompt_file = task.config.compaction.prompt_file.clone();
        let prompt_text = match self.resolve_compaction_prompt(&pod_id, &prompt_file) {
            Ok(s) => s,
            Err(e) => {
                warn!(%thread_id, error = %e, "compact launch: prompt resolution failed");
                // Without the prompt the Function can't proceed. Treat
                // as an execution error and complete immediately.
                if let Some(id) = self.find_compact_function_for(thread_id) {
                    self.complete_function(
                        id,
                        crate::functions::FunctionOutcome::Error(crate::functions::FunctionError {
                            kind: crate::functions::FunctionErrorKind::BadInput,
                            detail: format!("compaction prompt resolution failed: {e}"),
                        }),
                        pending_io,
                    );
                }
                return;
            }
        };

        // Stamp the marker first so the finalize hook can see it when
        // the builtin driver finishes the cycle below.
        let Some(weave_id) = self.thread_ticker.get(thread_id).cloned() else {
            // Admission verified a ticking weave exists; defensive.
            warn!(%thread_id, "compact launch: thread has no ticking weave");
            return;
        };
        if let Some(weave) = self.weaves.get_mut(&weave_id) {
            if let crate::runtime::driver::DriverState::BuiltinSingleAgentChat {
                compacting, ..
            } = &mut weave.driver_state
            {
                *compacting = Some(thread_id.to_string());
            }
            self.mark_weave_dirty(&weave_id);
        }
        // Reuse send_user_message so title/state broadcasts and dirty
        // tracking run the same as any user follow-up.
        self.send_user_message(thread_id, prompt_text, Vec::new(), pending_io);
        self.step_until_blocked(thread_id, pending_io);
    }

    /// Failure cleanup, called from `step_until_blocked` next to the
    /// other terminal hooks. A summary turn that died (provider error,
    /// tool failure) must not leave the weave's compacting marker set:
    /// the weave would refuse future compactions until an unrelated
    /// Completed turn tripped the finalize against an ordinary
    /// response. Clears the marker and completes the CompactThread
    /// Function as an execution error. No-op unless the thread is
    /// Failed and marked. (Cancellation is handled in
    /// `weave_cancelled` / `execute_cancel_thread`.)
    pub(super) fn abort_builtin_compaction_if_failed(
        &mut self,
        thread_id: &str,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        let failed = self.tasks.get(thread_id).is_some_and(|task| {
            matches!(
                task.public_state(),
                whisper_agent_protocol::ThreadStateLabel::Failed
            )
        });
        if !failed {
            return;
        }
        let Some(weave_id) = self.thread_ticker.get(thread_id).cloned() else {
            return;
        };
        let Some(weave) = self.weaves.get_mut(&weave_id) else {
            return;
        };
        let crate::runtime::driver::DriverState::BuiltinSingleAgentChat { compacting, .. } =
            &mut weave.driver_state
        else {
            return;
        };
        if compacting.as_deref() != Some(thread_id) {
            return;
        }
        *compacting = None;
        self.mark_weave_dirty(&weave_id);
        warn!(%thread_id, "compaction summary turn failed — marker cleared");
        if let Some(id) = self.find_compact_function_for(thread_id) {
            self.complete_function(
                id,
                crate::functions::FunctionOutcome::Error(crate::functions::FunctionError {
                    kind: crate::functions::FunctionErrorKind::Execution,
                    detail: "summary turn failed before completing".into(),
                }),
                pending_io,
            );
        }
    }

    /// Auto-trigger hook. Checks whether the given thread has crossed
    /// its compaction `token_threshold` and, if so, registers a
    /// `Function::CompactThread` with a `CallerLink::Weave` caller —
    /// the weave noticing its primary outgrew its context is the true
    /// originator.
    ///
    /// No-ops when:
    ///   - the thread has no threshold configured,
    ///   - the thread is not its ticking weave's current primary (a
    ///     compacted-away head is a dormant auxiliary and never
    ///     re-triggers; the promoted continuation triggers its own
    ///     compaction when it crosses the threshold), or
    ///   - `register_function` rejects (already compacting, not idle —
    ///     the precondition checks inside the Function registry cover
    ///     the same conditions).
    ///
    /// Fires from `step_until_blocked` after the boundary path has
    /// finalized any completed compaction, so a freshly demoted head is
    /// already non-primary when this runs.
    pub(super) fn maybe_auto_compact(
        &mut self,
        thread_id: &str,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        let task = match self.tasks.get(thread_id) {
            Some(t) => t,
            None => return,
        };
        let Some(threshold) = task.config.compaction.token_threshold else {
            return;
        };
        // Scripted-driven threads own their lifecycle; the builtin
        // flow's summary-prompt input would re-enter the driver
        // mid-activation. They get the `advance_head` primitive
        // instead; a compaction driver event waits until a real driver
        // needs one.
        if self.has_scripted_ticker(thread_id) {
            return;
        }
        if task.total_usage.input_tokens <= threshold {
            return;
        }
        // Only the weave's current primary compacts. Replaces the old
        // O(tasks) `continued_from` scan: a head that was compacted
        // away is no longer primary.
        let is_primary = self
            .thread_ticker
            .get(thread_id)
            .and_then(|weave_id| self.weaves.get(weave_id))
            .is_some_and(|weave| weave.primary_thread_id() == Some(thread_id));
        if !is_primary {
            return;
        }
        debug!(
            %thread_id,
            input_tokens = task.total_usage.input_tokens,
            threshold,
            "auto-compaction threshold crossed — triggering"
        );
        let spec = crate::functions::Function::CompactThread {
            thread_id: thread_id.to_string(),
        };

        let caller = crate::functions::CallerLink::Weave {
            weave_id: self
                .thread_ticker
                .get(thread_id)
                .cloned()
                .expect("primary check above requires a ticking weave"),
        };
        match self.register_function(spec, caller) {
            Ok(fn_id) => self.launch_function(fn_id, pending_io),
            Err(e) => {
                // All reject reasons here are benign — the state
                // machine caught up and the compaction is either
                // already running, not admissible, or disabled. Log
                // at debug rather than warn since auto-compact is
                // inherently racy with other turn activity.
                debug!(%thread_id, error = ?e, "auto-compaction skipped");
            }
        }
    }

    /// Hook called from the builtin boundary path when the driver
    /// finishes a cycle on `thread_id`. When the weave's builtin driver
    /// state carries a compacting marker for this thread and the thread
    /// has reached `Completed`, parse the `<summary>` from its most
    /// recent assistant message, derive a continuation thread into the
    /// same weave (journaled `derive_thread` with a
    /// `{kind: "compaction"}` edge), advance the head (journaled
    /// `advance_head` — the old head becomes a dormant auxiliary), and
    /// seed the continuation.
    ///
    /// No-op when no compaction is marked for this thread. Safe to call
    /// after every builtin cycle finish.
    pub(super) fn finalize_builtin_compaction(
        &mut self,
        weave_id: &str,
        thread_id: &str,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        let marked = self.weaves.get(weave_id).is_some_and(|weave| {
            matches!(
                &weave.driver_state,
                crate::runtime::driver::DriverState::BuiltinSingleAgentChat {
                    compacting: Some(t),
                    ..
                } if t == thread_id
            )
        });
        if !marked {
            return;
        }
        let task = match self.tasks.get(thread_id) {
            Some(t) => t,
            None => return,
        };
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
        let old_bindings = task.bindings.clone();
        let old_config = task.config.clone();
        let old_origin = task.origin.clone();
        // Snapshot the parent's setup prefix (Role::System prompt +
        // Role::Tools manifest) so the continuation can inherit it
        // verbatim. Matches fork semantics — a compaction continuation
        // is conceptually "more of the same thread" and should run
        // under the same system prompt the parent was running under,
        // not whatever the pod's current `system_prompt.md` happens to
        // hold. Critical for behavior-origin threads whose parent was
        // created with a behavior-specific setup — re-reading the pod
        // default would silently swap personalities at the compaction
        // boundary.
        let parent_setup = setup_prefix_snapshot(task);

        // Always clear the marker so a failed parse doesn't re-trigger
        // the finalize on every subsequent cycle finish.
        if let Some(weave) = self.weaves.get_mut(weave_id) {
            if let crate::runtime::driver::DriverState::BuiltinSingleAgentChat {
                compacting, ..
            } = &mut weave.driver_state
            {
                *compacting = None;
            }
            self.mark_weave_dirty(weave_id);
        }

        // Locate the in-flight CompactThread Function so we can emit
        // its terminal. There should always be exactly one when the
        // compacting flag is set — the flag is a 1:1 mirror of the
        // registered Function today.
        let compact_fn_id = self.find_compact_function_for(thread_id);

        let Some(summary_text) = extract_summary(&regex_src, &assistant_text) else {
            warn!(
                %thread_id,
                "compaction finalize: failed to extract <summary> from assistant response — \
                 leaving thread Completed without spawning continuation"
            );
            if let Some(id) = compact_fn_id {
                self.complete_function(
                    id,
                    crate::functions::FunctionOutcome::Error(crate::functions::FunctionError {
                        kind: crate::functions::FunctionErrorKind::Execution,
                        detail: "summary extraction failed".into(),
                    }),
                    pending_io,
                );
            }
            return;
        };

        // Derive the continuation into the same weave. Routing through
        // `weave_derive_thread` journals the `derive_thread` record with
        // the `compaction` relationship edge (the durable lineage that
        // replaced `Thread.continued_from`), re-resolves bindings, and
        // fires the usual `ThreadCreated` broadcast. Base scope comes
        // from the current primary — a narrowed parent doesn't widen at
        // the compaction boundary.
        let config_override = Some(inherited_config_override(&old_config));
        let bindings_request = Some(inherited_bindings_request(&old_bindings));
        let relationship = crate::runtime::driver::ThreadRelationship {
            kind: "compaction".to_string(),
            source: Some(crate::runtime::driver::EntryRef {
                thread_id: thread_id.to_string(),
                entry_index: None,
            }),
        };
        let new_thread_id = match self.weave_derive_thread(
            weave_id,
            config_override,
            bindings_request,
            Vec::new(),
            relationship,
            old_origin,
            pending_io,
        ) {
            Ok(id) => id,
            Err(e) => {
                warn!(
                    %thread_id, error = %e,
                    "compaction finalize: deriving continuation failed"
                );
                if let Some(id) = compact_fn_id {
                    self.complete_function(
                        id,
                        crate::functions::FunctionOutcome::Error(crate::functions::FunctionError {
                            kind: crate::functions::FunctionErrorKind::Execution,
                            detail: format!("continuation derive failed: {e}"),
                        }),
                        pending_io,
                    );
                }
                return;
            }
        };

        self.apply_setup_snapshot(
            &new_thread_id,
            parent_setup,
            &old_config.participant_profiles,
        );

        // Advance the head: the continuation becomes the weave's
        // primary; the old head is demoted to a dormant auxiliary
        // (frozen history, drill-down guaranteed). Journals the
        // `advance_head` record, pushes fresh weave snapshots to
        // subscribers, and broadcasts a decorated thread list so
        // client-side weave tags stay coherent.
        if let Err(e) = self.weave_advance_head(weave_id, &new_thread_id) {
            // Admission can't reasonably fail here (the continuation
            // was just derived into this weave, ticked), but if it
            // does, surface it rather than silently leaving two live
            // threads.
            warn!(
                %weave_id, %new_thread_id, error = %e,
                "compaction finalize: advance_head refused"
            );
        }

        // Seed the continuation with the filled-in template. This
        // kicks the thread's first model call.
        let seed_text = render_continuation_template(&continuation_template, &summary_text);
        self.send_user_message(&new_thread_id, seed_text, Vec::new(), pending_io);
        self.step_until_blocked(&new_thread_id, pending_io);

        // Emit the CompactThread Function's success terminal. The
        // client's visible UX already happened via the weave snapshot
        // push and thread-list broadcast inside `weave_advance_head`;
        // this is the registry-bookkeeping side of completion.
        if let Some(id) = compact_fn_id {
            self.complete_function(
                id,
                crate::functions::FunctionOutcome::Success(
                    crate::functions::FunctionTerminal::CompactThread(
                        crate::functions::CompactThreadTerminal {
                            continuation_thread_id: new_thread_id.clone(),
                        },
                    ),
                ),
                pending_io,
            );
        }

        debug!(
            old = %thread_id, new = %new_thread_id, summary_bytes = summary_text.len(),
            "compaction finalize: spawned continuation"
        );
    }

    /// Overwrite a freshly derived thread's setup prefix (seeded by
    /// `seed_thread_setup` inside `create_task` from current pod
    /// state) with a source thread's snapshot. Copying verbatim keeps
    /// the derived thread's system prompt and tool manifest identical
    /// to what the source was running under — behaviors,
    /// custom-prompted threads, and mid-life pod edits all settle out
    /// the same way (the derive inherits, pod drift doesn't leak
    /// across the boundary). Shared by the builtin compaction finalize
    /// and the scripted `derive_thread { setup_from }` directive
    /// (step 11 slice 5).
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
/// verbatim. Shared by the builtin compaction finalize and the
/// scripted `derive_thread { config_from }` directive (step 11 slice
/// 5). `compaction`/`autoquery` deliberately stay `None`: the derived
/// thread re-inherits the pod's defaults for those (the builtin's
/// documented choice — per-thread overrides do not outlive the head
/// they were set on).
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
/// subagent path). Shared by the builtin compaction finalize and the
/// scripted `derive_thread { bindings_from }` directive.
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

    /// Regression: when the model verbatim-quotes a prior tool result
    /// (e.g., a dispatched-thread notification) that itself contains a
    /// `<summary>...</summary>` pair, the default extraction regex must
    /// capture the OUTER block, not stop at the inner closing tag.
    /// Triggered on a real gpt-5.5 thread on 2026-05-03 (k8s mavis pod):
    /// a 45kB summary was truncated to ~19kB at the inner `</summary>`.
    #[test]
    fn extracts_outer_summary_when_body_contains_nested_tag() {
        let regex_src = whisper_agent_protocol::CompactionConfig::default().summary_regex;
        let text = "<summary>\noutside head\n\
                    <dispatched-thread-notification><summary>nested</summary></dispatched-thread-notification>\n\
                    outside tail\n</summary>";
        let got = extract_summary(&regex_src, text).expect("regex should match outer block");
        assert!(got.contains("outside head"), "got: {got}");
        assert!(got.contains("outside tail"), "got: {got}");
        assert!(got.contains("<summary>nested</summary>"), "got: {got}");
    }

    /// The default regex anchors to end-of-input (`\s*\z`), tolerating
    /// trailing whitespace but rejecting trailing prose. The compaction
    /// prompt instructs the model to end after the closing tag.
    #[test]
    fn default_regex_tolerates_trailing_whitespace() {
        let regex_src = whisper_agent_protocol::CompactionConfig::default().summary_regex;
        let text = "<summary>\nbody\n</summary>\n  \n";
        assert_eq!(extract_summary(&regex_src, text).as_deref(), Some("body"));
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
