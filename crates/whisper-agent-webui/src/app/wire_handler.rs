//! Inbound `ServerToClient` event reducer.
//!
//! Owns the giant `match` over wire events plus the small
//! `pending_tool_batch_flush_thread_id` helper that drives the
//! sync-tool-batch flush from the prologue. Lives as an `impl ChatApp`
//! extension block (same convention as `sidebar`) — every arm reaches
//! deep into private fields and modal slots, so the
//! free-function-with-event-vec shape that fits the renderer modules
//! would only inflate the API surface here.

use whisper_agent_protocol::{ClientToServer, ServerToClient, ThreadStateLabel};

use super::conversion::{
    append_streaming_output, build_tool_call_item, conversation_to_items, push_tool_result,
    push_tool_result_from_text, snapshot_summary,
};
use super::editor_render::behavior_summary_from_snapshot;
use super::widgets::open_in_new_tab;
use super::{
    ChatApp, DisplayItem, PendingSudo, ServerConfigSaveSummary, TaskView, ThreadInspector,
};

impl ChatApp {
    pub(super) fn handle_wire(&mut self, msg: ServerToClient) {
        // Flush a pending sync-tool-batch append on the first event that
        // isn't part of the tool-streaming trio. `thread.rs::step()`
        // pushes exactly one `Role::ToolResult` message to the
        // conversation when all tool calls of a turn resolve, without a
        // dedicated event — so arming on `ToolCallBegin` and flushing
        // here keeps `conv_message_count` in step with server state.
        if let Some(tid) = pending_tool_batch_flush_thread_id(&msg)
            && let Some(view) = self.tasks.get_mut(tid)
        {
            view.flush_pending_tool_batch();
        }
        match msg {
            ServerToClient::ThreadCreated {
                thread_id,
                summary,
                correlation_id,
            } => {
                self.upsert_task(summary);
                self.recompute_order();
                // Fork-seed handoff: if this `ThreadCreated` carries
                // the correlation_id we stamped on the outbound
                // `ForkThread`, the server's `fork_task` just minted
                // this id. Issue a `SetThreadDraft` so the new
                // thread's persisted draft holds the forked-from
                // user-message text. Do this *before* `select_task`,
                // which triggers a `SubscribeToThread` whose snapshot
                // will then include the just-written draft.
                let seed_match = self
                    .pending_fork_seed
                    .as_ref()
                    .is_some_and(|(cid, _)| correlation_id.as_deref() == Some(cid.as_str()));
                if seed_match && let Some((_, text)) = self.pending_fork_seed.take() {
                    self.send(ClientToServer::SetThreadDraft {
                        thread_id: thread_id.clone(),
                        text: text.clone(),
                    });
                    // Mirror into the local cache: `select_task` below
                    // loads `self.input` from `view.draft`, and the
                    // snapshot with the server-side draft may not have
                    // landed yet.
                    if let Some(view) = self.tasks.get_mut(&thread_id) {
                        view.draft = text;
                    }
                }
                if self.selected.as_deref() != Some(&thread_id) {
                    // Don't yank focus for background spawns.
                    // Behavior-triggered threads carry an `origin`;
                    // dispatch_thread-spawned children carry
                    // `dispatched_by`. Bare creates, forks, and
                    // compaction continuations all leave both fields
                    // None and keep the current auto-focus UX the
                    // user expects after taking a direct action.
                    let background = self.tasks.get(&thread_id).is_some_and(|v| {
                        v.summary.origin.is_some() || v.summary.dispatched_by.is_some()
                    });
                    if !background {
                        self.select_task(thread_id);
                    }
                }
            }
            ServerToClient::ThreadStateChanged { thread_id, state } => {
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    view.summary.state = state;
                    // Any transition away from Working invalidates
                    // transient stream decorations — a mid-args
                    // tool-call placeholder that never got its
                    // matching ToolCallBegin, or a prefill bar whose
                    // turn just got cancelled. Leaving these on
                    // screen is the "stuck spinner" bug.
                    if state != ThreadStateLabel::Working {
                        view.prefill_progress = None;
                        view.items
                            .retain(|it| !matches!(it, DisplayItem::ToolCallStreaming { .. }));
                    }
                }
            }
            ServerToClient::ThreadTitleUpdated { thread_id, title } => {
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    view.summary.title = Some(title);
                }
            }
            ServerToClient::ThreadArchived { thread_id } => {
                self.tasks.remove(&thread_id);
                self.task_order.retain(|id| id != &thread_id);
                if self.selected.as_deref() == Some(&thread_id) {
                    self.selected = None;
                    self.composing_new = true;
                }
            }
            ServerToClient::ThreadList { tasks, .. } => {
                self.tasks
                    .retain(|id, _| tasks.iter().any(|t| &t.thread_id == id));
                for summary in tasks {
                    self.upsert_task(summary);
                }
                self.recompute_order();
            }
            ServerToClient::ThreadSnapshot {
                thread_id,
                snapshot,
            } => {
                let items = conversation_to_items(&snapshot.conversation, &snapshot.turn_log);
                let backend = snapshot.bindings.backend.clone();
                let model = snapshot.config.model.clone();
                let failure = snapshot.failure.clone();
                let inspector = ThreadInspector {
                    max_tokens: snapshot.config.max_tokens,
                    max_turns: snapshot.config.max_turns,
                    bindings: snapshot.bindings.clone(),
                    origin: snapshot.origin.clone(),
                    created_at: snapshot.created_at.clone(),
                    scope: snapshot.scope.clone(),
                };
                let view = self
                    .tasks
                    .entry(thread_id.clone())
                    .or_insert_with(|| TaskView::new(snapshot_summary(&snapshot)));
                view.summary.state = snapshot.state;
                view.summary.title = snapshot.title;
                view.summary.origin = snapshot.origin.clone();
                view.total_usage = snapshot.total_usage;
                view.items = items;
                view.subscribed = true;
                view.backend = backend;
                view.model = model;
                view.failure = failure;
                view.inspector = inspector;
                view.conv_message_count = snapshot.conversation.len();
                // Any in-flight tool batch carried over by the client is
                // moot — snapshot length is authoritative, so reset the
                // flush flag to avoid double-counting the same append.
                view.pending_tool_batch = false;
                view.draft = snapshot.draft.clone();
                // Sync compose box from the just-arrived snapshot
                // when we're looking at this thread and haven't
                // started typing yet. Existing `input` content wins
                // — a user who switched back before the snapshot
                // landed shouldn't have their typing clobbered.
                if self.selected.as_deref() == Some(&thread_id) && self.input.is_empty() {
                    self.input = snapshot.draft;
                    self.last_input_change_at = None;
                }
            }
            ServerToClient::ThreadDraftUpdated { thread_id, text } => {
                // Skip redundant echoes (reconnect + resubscribe can
                // replay a draft we already have) so a same-text
                // update doesn't stomp the cursor on a selected
                // thread.
                let same = self.tasks.get(&thread_id).is_some_and(|v| v.draft == text);
                if same {
                    return;
                }
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    view.draft = text.clone();
                }
                if self.selected.as_deref() == Some(&thread_id) {
                    self.input = text;
                    self.last_input_change_at = None;
                }
            }
            ServerToClient::ThreadCompacted {
                thread_id,
                new_thread_id,
                ..
            } => {
                // The continuation thread already arrived via its own
                // `ThreadCreated` event with `continued_from = None`
                // in the summary — the linkage stamp happens on the
                // server after `create_task` returns. Patch it in
                // now so the list tier reflects the ancestor.
                if let Some(view) = self.tasks.get_mut(&new_thread_id) {
                    view.summary.continued_from = Some(thread_id);
                }
            }
            ServerToClient::ThreadUserMessage {
                thread_id,
                text,
                attachments,
            } => {
                // User-role message appended to the conversation.
                // Fires for both user-typed follow-ups (the webui used
                // to add these optimistically; that's now removed so
                // the server echo is the single source of truth) and
                // server-injected text (compaction continuation seeds,
                // behavior-trigger prompts). Async dispatch callbacks
                // travel a distinct event (`ThreadToolResultMessage`).
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    let msg_index = view.conv_message_count;
                    view.conv_message_count += 1;
                    view.items.push(DisplayItem::User {
                        text,
                        msg_index,
                        attachments,
                    });
                }
            }
            ServerToClient::ThreadToolResultMessage { thread_id, text } => {
                // Tool-output text appended to the conversation —
                // typically an async `dispatch_thread` callback
                // (XML envelope carrying the child's final result).
                // Pushed as its own `DisplayItem::ToolResult` row at
                // the chronological position where it landed; the
                // default-open check is based on proximity to the
                // matching tool call in the items list so async
                // callbacks (separated from their call by an
                // assistant turn) arrive expanded while immediately-
                // -following results stay collapsed.
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    view.conv_message_count += 1;
                    push_tool_result_from_text(&mut view.items, &text);
                }
            }
            ServerToClient::ThreadAssistantBegin { thread_id, .. } => {
                // A fresh assistant turn starts — drop any progress
                // bar we were showing from an earlier turn so it
                // can't bleed into this one.
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    view.prefill_progress = None;
                }
            }
            ServerToClient::ThreadPrefillProgress {
                thread_id,
                tokens_processed,
                tokens_total,
            } => {
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    // Drop late stream events buffered in the
                    // scheduler's channel when the thread has
                    // already transitioned out of Working — without
                    // this, a just-cancelled turn can have a
                    // progress bar re-appear after the state change
                    // arrived.
                    if view.summary.state == ThreadStateLabel::Working {
                        view.prefill_progress = Some((tokens_processed, tokens_total));
                    }
                }
            }
            ServerToClient::ThreadAssistantTextDelta { thread_id, delta } => {
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    view.prefill_progress = None;
                    if let Some(DisplayItem::AssistantText { text }) = view.items.last_mut() {
                        text.push_str(&delta);
                    } else {
                        view.items.push(DisplayItem::AssistantText { text: delta });
                    }
                }
            }
            ServerToClient::ThreadAssistantImage { thread_id, source } => {
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    view.prefill_progress = None;
                    view.items.push(DisplayItem::AssistantImage { source });
                }
            }
            ServerToClient::ThreadAssistantReasoningDelta { thread_id, delta } => {
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    view.prefill_progress = None;
                    if let Some(DisplayItem::Reasoning { text }) = view.items.last_mut() {
                        text.push_str(&delta);
                    } else {
                        view.items.push(DisplayItem::Reasoning { text: delta });
                    }
                }
            }
            ServerToClient::ThreadToolCallStreaming {
                thread_id,
                tool_use_id,
                name,
                args_chars,
            } => {
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    // Guard against late stream events buffered in
                    // the scheduler's channel arriving after the
                    // thread left Working — otherwise a cancelled
                    // turn can get a stale placeholder added back.
                    if view.summary.state != ThreadStateLabel::Working {
                        return;
                    }
                    // Upsert: if we already have a streaming placeholder
                    // for this call, update its char count in place so
                    // the row doesn't re-order. Otherwise append a new
                    // one at the tail.
                    let existing = view.items.iter_mut().rev().find_map(|it| match it {
                        DisplayItem::ToolCallStreaming {
                            tool_use_id: id, ..
                        } if *id == tool_use_id => Some(it),
                        _ => None,
                    });
                    if let Some(DisplayItem::ToolCallStreaming {
                        name: existing_name,
                        args_chars: existing_chars,
                        ..
                    }) = existing
                    {
                        *existing_name = name;
                        *existing_chars = args_chars;
                    } else {
                        view.items.push(DisplayItem::ToolCallStreaming {
                            tool_use_id,
                            name,
                            args_chars,
                        });
                    }
                }
            }
            ServerToClient::ThreadToolCallBegin {
                thread_id,
                tool_use_id,
                name,
                args_preview,
                args,
            } => {
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    // Arm the batch-append flag; a later non-tool event
                    // flushes it into `conv_message_count += 1` to match
                    // the `Role::ToolResult` message the server pushes
                    // once all tool calls of this turn resolve.
                    view.pending_tool_batch = true;
                    // Remove any in-flight streaming placeholder for
                    // this call; the full tool-call row below replaces
                    // it with name + args + diff etc.
                    view.items.retain(|it| {
                        !matches!(
                            it,
                            DisplayItem::ToolCallStreaming {
                                tool_use_id: id, ..
                            } if *id == tool_use_id
                        )
                    });
                    view.items.push(build_tool_call_item(
                        tool_use_id,
                        name,
                        args.as_ref(),
                        args_preview,
                    ));
                }
            }
            ServerToClient::ThreadToolCallContent {
                thread_id,
                tool_use_id,
                block,
            } => {
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    append_streaming_output(&mut view.items, &tool_use_id, &block);
                }
            }
            ServerToClient::ThreadToolCallEnd {
                thread_id,
                tool_use_id,
                result_preview,
                is_error,
            } => {
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    // Live streaming path carries only a short preview
                    // string — full image attachments arrive on the
                    // conversation replay, so pass empty attachments
                    // here and let the ThreadSnapshot re-populate them.
                    push_tool_result(
                        &mut view.items,
                        tool_use_id,
                        result_preview,
                        is_error,
                        Vec::new(),
                    );
                }
            }
            ServerToClient::ThreadAssistantEnd {
                thread_id, usage, ..
            } => {
                if let Some(view) = self.tasks.get_mut(&thread_id) {
                    view.total_usage.add(&usage);
                    view.conv_message_count += 1;
                    view.items.push(DisplayItem::TurnStats { usage });
                }
            }
            ServerToClient::ThreadLoopComplete { .. } => {}
            ServerToClient::Error {
                thread_id,
                message,
                correlation_id,
            } => {
                // If the pod editor minted this correlation, surface the
                // error inline in the modal instead of as a global banner
                // — failed validation should leave the user's edits
                // intact.
                if let Some(modal) = self.pod_editor_modal.as_mut()
                    && correlation_id.is_some()
                    && modal.pending_correlation == correlation_id
                {
                    modal.error = Some(message);
                    modal.pending_correlation = None;
                    return;
                }
                // Buckets modal in-flight query.
                if let Some(modal) = self.buckets_modal.as_mut()
                    && correlation_id.is_some()
                    && modal.pending_query_correlation == correlation_id
                {
                    modal.pending_query_correlation = None;
                    modal.query_status = super::QueryStatus::Error { message };
                    return;
                }
                // Behavior editor pending save.
                if let Some(modal) = self.behavior_editor_modal.as_mut()
                    && correlation_id.is_some()
                    && modal.pending_correlation == correlation_id
                {
                    modal.error = Some(message);
                    modal.pending_correlation = None;
                    return;
                }
                // "+ New behavior" pending create.
                if let Some(modal) = self.new_behavior_modal.as_mut()
                    && correlation_id.is_some()
                    && modal.pending_correlation == correlation_id
                {
                    modal.error = Some(message);
                    modal.pending_correlation = None;
                    return;
                }
                // File viewer in-flight read or save.
                if let Some(modal) = self.file_viewer_modal.as_mut()
                    && correlation_id.is_some()
                    && modal.pending_correlation == correlation_id
                {
                    modal.error = Some(message);
                    modal.pending_correlation = None;
                    return;
                }
                // JSON viewer in-flight read.
                if let Some(modal) = self.json_viewer_modal.as_mut()
                    && correlation_id.is_some()
                    && modal.pending_correlation == correlation_id
                {
                    modal.error = Some(message);
                    modal.pending_correlation = None;
                    return;
                }
                // Provider editor modal pending add / update.
                if let Some(modal) = self.provider_editor_modal.as_mut()
                    && correlation_id.is_some()
                    && modal.pending_correlation == correlation_id
                {
                    modal.error = Some(message);
                    modal.pending_correlation = None;
                    return;
                }
                // Codex-auth rotation sub-form pending save. Route the
                // detail into the sub-form's error field so the user
                // can fix the paste without retyping; the sub-form
                // stays open until they explicitly cancel.
                if let Some(settings) = self.settings_modal.as_mut()
                    && let Some(sub) = settings.codex_rotate.as_mut()
                    && correlation_id.is_some()
                    && sub.pending_correlation == correlation_id
                {
                    sub.error = Some(message);
                    sub.pending_correlation = None;
                    return;
                }
                // Shared-MCP editor sub-form pending add/update. Keep
                // the form open and surface the connect/validation
                // failure inline so the operator can fix and retry.
                if let Some(settings) = self.settings_modal.as_mut()
                    && let Some(sub) = settings.shared_mcp_editor.as_mut()
                    && correlation_id.is_some()
                    && sub.pending_correlation == correlation_id
                {
                    sub.error = Some(message);
                    sub.pending_correlation = None;
                    return;
                }
                // Shared-MCP remove response — no sub-form, just a
                // banner on the parent tab.
                if let Some(settings) = self.settings_modal.as_mut()
                    && correlation_id.is_some()
                    && message.starts_with("remove_shared_mcp_host:")
                {
                    settings.shared_mcp_banner = Some(Err(message));
                    return;
                }
                // Server-config editor — match either the in-flight
                // fetch or the in-flight save correlation. Either
                // failure stays on the editor as an inline banner.
                if let Some(settings) = self.settings_modal.as_mut()
                    && let Some(editor) = settings.server_config.as_mut()
                    && correlation_id.is_some()
                {
                    if editor.fetch_correlation == correlation_id {
                        editor.banner = Some(Err(format!("fetch: {message}")));
                        editor.fetch_correlation = None;
                        return;
                    }
                    if editor.save_correlation == correlation_id {
                        editor.banner = Some(Err(message));
                        editor.save_correlation = None;
                        return;
                    }
                }
                // Provider remove pending on a specific row. Match by
                // correlation rather than iterating names so a stale
                // remove that targets a now-gone name still resolves.
                if let Some(cid) = correlation_id.as_deref()
                    && let Some((name, _)) = self
                        .provider_remove_pending
                        .iter()
                        .find(|(_, p)| p.correlation == cid)
                        .map(|(n, p)| (n.clone(), p))
                {
                    if let Some(p) = self.provider_remove_pending.get_mut(&name) {
                        p.error = Some(message);
                    }
                    return;
                }
                if let Some(tid) = thread_id.as_ref()
                    && let Some(view) = self.tasks.get_mut(tid)
                {
                    // Persist on the view so it's visible as a banner even after a
                    // resnapshot wipes `items`. The scheduler also records the same
                    // detail on the task's Failed state; this mirrors it locally so
                    // the UI doesn't have to wait on a re-subscribe round-trip.
                    view.failure = Some(message.clone());
                    view.items.push(DisplayItem::SystemNote {
                        text: message,
                        is_error: true,
                    });
                } else {
                    // No task scope — surface via conn detail so the banner reflects it.
                    self.conn_detail = Some(message);
                }
            }
            ServerToClient::BackendsList { backends, .. } => {
                self.backends = backends;
                // Pre-fetch the alphabetically-first backend's models so the
                // picker is ready on first open without a visible delay
                // — that's the entry `effective_picker_backend` falls back
                // to when no pod default is set.
                if let Some(first) = self.backends.iter().map(|b| b.name.clone()).min() {
                    self.request_models_for(&first);
                }
            }
            ServerToClient::EmbeddingProvidersList { providers, .. } => {
                self.embedding_providers = providers;
            }
            ServerToClient::ModelsList {
                backend, models, ..
            } => {
                self.models_by_backend.insert(backend, models);
            }
            ServerToClient::ResourceList { resources, .. } => {
                self.resources.clear();
                for r in resources {
                    self.resources.insert(r.id().to_string(), r);
                }
            }
            ServerToClient::ResourceCreated { resource }
            | ServerToClient::ResourceUpdated { resource } => {
                self.resources.insert(resource.id().to_string(), resource);
            }
            ServerToClient::ResourceDestroyed { id, .. } => {
                self.resources.remove(&id);
            }
            ServerToClient::HostEnvProvidersList { providers, .. } => {
                self.host_env_providers = providers;
            }
            ServerToClient::BucketsList { buckets, .. } => {
                self.buckets = buckets;
            }
            ServerToClient::QueryResults {
                correlation_id,
                query,
                hits,
            } => {
                // Drop the result if the user has fired a follow-up
                // search since this one was sent — comparing against
                // the modal's stored correlation, not the modal's
                // current input string (the user may have typed past
                // the in-flight query).
                if let Some(modal) = self.buckets_modal.as_mut()
                    && correlation_id.as_deref() == modal.pending_query_correlation.as_deref()
                {
                    modal.pending_query_correlation = None;
                    modal.query_status = super::QueryStatus::Results {
                        query,
                        hits,
                        expanded: std::collections::HashSet::new(),
                    };
                }
            }
            ServerToClient::BucketCreated {
                correlation_id,
                summary,
            } => {
                // Insert / replace the bucket entry. Insertion is
                // BTreeMap-ordered server-side; we keep the wire-side
                // Vec sorted by id to match.
                upsert_bucket(&mut self.buckets, summary);
                if let Some(modal) = self.buckets_modal.as_mut()
                    && let Some(form) = modal.creating.as_ref()
                    && form
                        .pending_correlation
                        .as_deref()
                        .is_some_and(|c| Some(c) == correlation_id.as_deref())
                {
                    // Form was pending this correlation — close it.
                    modal.creating = None;
                }
            }
            ServerToClient::BucketDeleted { id, pod_id, .. } => {
                self.buckets.retain(|b| !(b.id == id && b.pod_id == pod_id));
                if let Some(modal) = self.buckets_modal.as_mut() {
                    let key = (pod_id.clone(), id.clone());
                    modal.build_progress.remove(&key);
                    modal.build_errors.remove(&key);
                    if modal.delete_armed.as_deref() == Some(&id) {
                        modal.delete_armed = None;
                    }
                    if modal.selected_bucket.as_deref() == Some(&id) {
                        modal.selected_bucket = None;
                    }
                }
            }
            ServerToClient::BucketBuildStarted {
                bucket_id,
                pod_id,
                started_at,
                ..
            } => {
                if let Some(modal) = self.buckets_modal.as_mut() {
                    modal.build_progress.insert(
                        (pod_id, bucket_id),
                        super::BuildProgressView {
                            phase: whisper_agent_protocol::BucketBuildPhase::Planning,
                            source_records: 0,
                            chunks: 0,
                            started_at,
                        },
                    );
                }
            }
            ServerToClient::BucketBuildProgress {
                bucket_id,
                pod_id,
                phase,
                source_records,
                chunks,
                started_at,
                ..
            } => {
                if let Some(modal) = self.buckets_modal.as_mut() {
                    // Preserve the existing started_at if the server
                    // omitted it on this tick (e.g. a pre-stopwatch
                    // server, or a defensive None fallback). The
                    // anchor only matters once.
                    let key = (pod_id, bucket_id);
                    let prior = modal
                        .build_progress
                        .get(&key)
                        .and_then(|p| p.started_at.clone());
                    modal.build_progress.insert(
                        key,
                        super::BuildProgressView {
                            phase,
                            source_records,
                            chunks,
                            started_at: started_at.or(prior),
                        },
                    );
                }
            }
            ServerToClient::BucketBuildEnded {
                bucket_id,
                pod_id,
                outcome,
                summary,
                ..
            } => {
                if let Some(modal) = self.buckets_modal.as_mut() {
                    let key = (pod_id.clone(), bucket_id.clone());
                    modal.build_progress.remove(&key);
                    match &outcome {
                        whisper_agent_protocol::BucketBuildOutcome::Success => {
                            modal.build_errors.remove(&key);
                        }
                        whisper_agent_protocol::BucketBuildOutcome::Cancelled => {
                            modal.build_errors.insert(key, "build cancelled".into());
                        }
                        whisper_agent_protocol::BucketBuildOutcome::Error { message } => {
                            modal.build_errors.insert(key, message.clone());
                        }
                    }
                }
                if let Some(s) = summary {
                    upsert_bucket(&mut self.buckets, s);
                }
            }
            ServerToClient::PodList {
                pods,
                default_pod_id,
                ..
            } => {
                self.pods = pods.into_iter().map(|p| (p.pod_id.clone(), p)).collect();
                // Fetch the default pod's config so "+ New pod" can clone
                // its sandbox / shared-mcp setup. Cheap round-trip; only
                // sent when the id changes (guarded by string equality).
                if !default_pod_id.is_empty() && default_pod_id != self.server_default_pod_id {
                    self.server_default_pod_id = default_pod_id.clone();
                    self.send(ClientToServer::GetPod {
                        correlation_id: None,
                        pod_id: default_pod_id,
                    });
                }
                // PodList summaries don't carry behavior catalogs —
                // fire one ListBehaviors per pod we haven't seen yet
                // so the pod sections render pre-existing behaviors on
                // first connect. `behaviors_requested` dedups so a
                // PodList refresh doesn't re-request.
                let pod_ids: Vec<String> = self.pods.keys().cloned().collect();
                for pid in pod_ids {
                    self.ensure_behaviors_fetched(&pid);
                }
            }
            ServerToClient::PodCreated { pod, .. } => {
                let pod_id = pod.pod_id.clone();
                self.pods.insert(pod.pod_id.clone(), pod);
                self.ensure_behaviors_fetched(&pod_id);
            }
            ServerToClient::PodConfigUpdated {
                pod_id,
                toml_text,
                parsed,
                correlation_id,
            } => {
                // Mirror the new top-level fields (name/description) onto the
                // summary so the left panel reflects edits without waiting
                // for a full ListPods refresh. thread_count is unchanged by
                // a config edit.
                if let Some(summary) = self.pods.get_mut(&pod_id) {
                    summary.name = parsed.name.clone();
                    summary.description = parsed.description.clone();
                    summary.created_at = parsed.created_at.clone();
                }
                if pod_id == self.server_default_pod_id {
                    self.default_pod_template = Some(parsed.clone());
                }
                // Refresh the compose-form cache so an open compose
                // picker sees the edit immediately.
                self.pod_configs.insert(pod_id.clone(), parsed.clone());
                // The editor stays open across saves — refresh its
                // baseline so subsequent edits are diffed against the
                // newly-persisted state, not the stale one. We keep
                // the user's `working` value if they're matching the
                // correlation we just sent (their own save
                // round-tripped — `working` already matches `parsed`),
                // and we replace `working` if this update came from
                // another client (otherwise their off-screen edits
                // would silently clobber the local view).
                if let Some(modal) = self.pod_editor_modal.as_mut()
                    && modal.pod_id == pod_id
                {
                    let our_save = modal.pending_correlation.is_some()
                        && modal.pending_correlation == correlation_id;
                    modal.server_baseline = Some(parsed.clone());
                    if our_save {
                        // Refresh `working` from the server's parse too —
                        // necessary when we saved from the Raw tab (where
                        // `working` wasn't the source of truth) and a no-op
                        // when we saved from a structured tab.
                        modal.working = Some(parsed);
                        modal.pending_correlation = None;
                        modal.error = None;
                        modal.raw_buffer = toml_text;
                        modal.raw_dirty = false;
                    } else if !modal.is_dirty() {
                        // Foreign update + we have no local edits =>
                        // adopt it cleanly. If we *do* have edits,
                        // leave them alone; the next Save will collide
                        // server-side and we'll show that error.
                        modal.working = Some(parsed);
                        modal.raw_buffer = toml_text;
                        modal.raw_dirty = false;
                        modal.error = None;
                    }
                }
            }
            ServerToClient::PodSystemPromptUpdated { .. } => {
                // No rendered view of the prompt text today, so nothing
                // for the UI to refresh. The event is still delivered so
                // a future "inspect current system prompt" panel can
                // stay in sync without polling.
            }
            ServerToClient::PodDirListing {
                pod_id,
                path,
                entries,
                ..
            } => {
                let key = (pod_id, path);
                self.pod_files_requested.remove(&key);
                self.pod_files.insert(key, entries);
            }
            ServerToClient::PodFileContent {
                pod_id,
                path,
                content,
                readonly,
                correlation_id,
            } => {
                if let Some(modal) = self.file_viewer_modal.as_mut()
                    && modal.pod_id == pod_id
                    && modal.path == path
                    && modal.pending_correlation == correlation_id
                {
                    modal.baseline = Some(content.clone());
                    modal.working = Some(content);
                    modal.readonly = readonly;
                    modal.pending_correlation = None;
                    modal.error = None;
                } else if let Some(modal) = self.json_viewer_modal.as_mut()
                    && modal.pod_id == pod_id
                    && modal.path == path
                    && modal.pending_correlation == correlation_id
                {
                    modal.pending_correlation = None;
                    match serde_json::from_str::<serde_json::Value>(&content) {
                        Ok(value) => {
                            modal.parsed = Some(value);
                            modal.error = None;
                        }
                        Err(e) => {
                            modal.parsed = None;
                            modal.error = Some(format!("parse JSON: {e}"));
                        }
                    }
                }
            }
            ServerToClient::PodFileWritten {
                pod_id,
                path,
                correlation_id,
            } => {
                if let Some(modal) = self.file_viewer_modal.as_mut()
                    && modal.pod_id == pod_id
                    && modal.path == path
                    && modal.pending_correlation == correlation_id
                {
                    // Save succeeded — adopt the working buffer as the
                    // new baseline so the dirty indicator flips to
                    // "saved" and the tree viewer's cached entry (if
                    // any) remains consistent with disk.
                    if let Some(w) = modal.working.clone() {
                        modal.baseline = Some(w);
                    }
                    modal.pending_correlation = None;
                    modal.error = None;
                }
            }
            ServerToClient::PodArchived { pod_id } => {
                self.pods.remove(&pod_id);
                // Drop any threads we were tracking under the archived pod —
                // the server won't send further events for them and they're
                // unreachable from the UI now.
                self.tasks.retain(|_, v| v.summary.pod_id != pod_id);
                self.recompute_order();
                if let Some(sel) = &self.selected
                    && !self.tasks.contains_key(sel)
                {
                    self.selected = None;
                    self.composing_new = true;
                }
                if self.compose_pod_id.as_deref() == Some(pod_id.as_str()) {
                    self.compose_pod_id = None;
                }
                if self.archive_armed_pod.as_deref() == Some(pod_id.as_str()) {
                    self.archive_armed_pod = None;
                }
            }
            ServerToClient::PodSnapshot { snapshot, .. } => {
                // Cache the default pod's config as a template for fresh
                // "+ New pod" creation.
                if snapshot.pod_id == self.server_default_pod_id {
                    self.default_pod_template = Some(snapshot.config.clone());
                }
                // Populate the editor modal if it's open and waiting on
                // this pod's text.
                if let Some(modal) = self.pod_editor_modal.as_mut()
                    && modal.pod_id == snapshot.pod_id
                    && modal.working.is_none()
                {
                    modal.server_baseline = Some(snapshot.config.clone());
                    modal.working = Some(snapshot.config.clone());
                    modal.raw_buffer = snapshot.toml_text.clone();
                    modal.raw_dirty = false;
                }
                // Update the compose-form cache so the host-env picker
                // reflects the current pod config even when the user
                // re-edits the pod without closing the compose form.
                self.pod_configs_requested.remove(&snapshot.pod_id);
                // Pod snapshots inline the behavior catalog so the
                // behaviors panel renders without an extra round trip
                // after opening a pod's detail view.
                self.behaviors_by_pod
                    .insert(snapshot.pod_id.clone(), snapshot.behaviors);
                self.pod_configs
                    .insert(snapshot.pod_id.clone(), snapshot.config);
            }
            ServerToClient::BehaviorList {
                pod_id, behaviors, ..
            } => {
                self.behaviors_by_pod.insert(pod_id, behaviors);
            }
            ServerToClient::BehaviorSnapshot {
                correlation_id,
                snapshot,
            } => {
                self.apply_behavior_snapshot(correlation_id, snapshot);
            }
            ServerToClient::BehaviorStateChanged {
                pod_id,
                behavior_id,
                state,
            } => {
                if let Some(list) = self.behaviors_by_pod.get_mut(&pod_id)
                    && let Some(row) = list.iter_mut().find(|b| b.behavior_id == behavior_id)
                {
                    row.run_count = state.run_count;
                    row.last_fired_at = state.last_fired_at.clone();
                    row.enabled = state.enabled;
                }
            }
            ServerToClient::PodBehaviorsEnabledChanged {
                correlation_id: _,
                pod_id,
                enabled,
            } => {
                if let Some(pod) = self.pods.get_mut(&pod_id) {
                    pod.behaviors_enabled = enabled;
                }
            }
            ServerToClient::BehaviorCreated {
                correlation_id,
                summary,
            } => {
                let list = self
                    .behaviors_by_pod
                    .entry(summary.pod_id.clone())
                    .or_default();
                if let Some(existing) = list
                    .iter_mut()
                    .find(|b| b.behavior_id == summary.behavior_id)
                {
                    *existing = summary.clone();
                } else {
                    list.push(summary.clone());
                    list.sort_by(|a, b| a.behavior_id.cmp(&b.behavior_id));
                }
                // If the creation was initiated from the "+ New behavior"
                // modal, close that and open the editor for the new one.
                if let Some(new_modal) = &self.new_behavior_modal
                    && new_modal.pending_correlation == correlation_id
                    && correlation_id.is_some()
                {
                    let pod_id = summary.pod_id.clone();
                    let behavior_id = summary.behavior_id.clone();
                    self.new_behavior_modal = None;
                    self.open_behavior_editor(pod_id, behavior_id);
                }
            }
            ServerToClient::BehaviorUpdated {
                correlation_id,
                snapshot,
            } => {
                if let Some(list) = self.behaviors_by_pod.get_mut(&snapshot.pod_id) {
                    let summary = behavior_summary_from_snapshot(&snapshot);
                    if let Some(existing) = list
                        .iter_mut()
                        .find(|b| b.behavior_id == snapshot.behavior_id)
                    {
                        *existing = summary;
                    } else {
                        list.push(summary);
                    }
                }
                // If this update matches our in-flight save, reset the
                // editor's baseline so dirty flips back to clean.
                if let Some(modal) = self.behavior_editor_modal.as_mut()
                    && modal.pod_id == snapshot.pod_id
                    && modal.behavior_id == snapshot.behavior_id
                    && (modal.pending_correlation.is_some()
                        && modal.pending_correlation == correlation_id)
                {
                    if let Some(config) = &snapshot.config {
                        modal.baseline_config = Some(config.clone());
                        // Keep the user's working state — they may have
                        // edited further during the round-trip. But if
                        // they haven't, align working with baseline so
                        // raw_buffer regenerates cleanly on next Raw
                        // tab entry.
                        if modal.working_config.as_ref() == modal.baseline_config.as_ref() {
                            modal.raw_buffer = snapshot.toml_text.clone();
                            modal.raw_dirty = false;
                        }
                    }
                    modal.baseline_prompt = snapshot.prompt.clone();
                    modal.pending_correlation = None;
                    modal.error = None;
                }
            }
            ServerToClient::BehaviorDeleted {
                pod_id,
                behavior_id,
                ..
            } => {
                if let Some(list) = self.behaviors_by_pod.get_mut(&pod_id) {
                    list.retain(|b| b.behavior_id != behavior_id);
                }
                if self.delete_armed_behavior.as_ref()
                    == Some(&(pod_id.clone(), behavior_id.clone()))
                {
                    self.delete_armed_behavior = None;
                }
                if let Some(modal) = &self.behavior_editor_modal
                    && modal.pod_id == pod_id
                    && modal.behavior_id == behavior_id
                {
                    self.behavior_editor_modal = None;
                }
            }
            ServerToClient::FunctionList { functions, .. } => {
                // Snapshot replaces the local map wholesale — the
                // server's registry is the source of truth, and
                // any stale ids we tracked before reconnect should
                // be evicted.
                self.active_functions.clear();
                for summary in functions {
                    self.active_functions.insert(summary.function_id, summary);
                }
            }
            ServerToClient::FunctionStarted { summary } => {
                self.active_functions.insert(summary.function_id, summary);
            }
            ServerToClient::FunctionEnded { function_id, .. } => {
                self.active_functions.remove(&function_id);
            }
            ServerToClient::HostEnvProviderAdded {
                provider,
                correlation_id,
            }
            | ServerToClient::HostEnvProviderUpdated {
                provider,
                correlation_id,
            } => {
                if let Some(existing) = self
                    .host_env_providers
                    .iter_mut()
                    .find(|p| p.name == provider.name)
                {
                    *existing = provider;
                } else {
                    self.host_env_providers.push(provider);
                    self.host_env_providers.sort_by(|a, b| a.name.cmp(&b.name));
                }
                // Close the editor modal if it was waiting on this
                // correlation. Edits that bypassed the modal (server-
                // side seed, another client's CRUD) land here with
                // `None` and leave the modal alone.
                if correlation_id.is_some()
                    && let Some(modal) = self.provider_editor_modal.as_ref()
                    && modal.pending_correlation == correlation_id
                {
                    self.provider_editor_modal = None;
                }
            }
            ServerToClient::HostEnvProviderRemoved {
                name,
                correlation_id,
            } => {
                self.host_env_providers.retain(|p| p.name != name);
                // Clear any pending-remove state if this was our
                // request. Another client's remove lands here too
                // (correlation_id: None); nothing to clean up in that
                // case beyond the list itself.
                if let Some(pending) = self.provider_remove_pending.get(&name)
                    && correlation_id.as_deref() == Some(pending.correlation.as_str())
                {
                    self.provider_remove_pending.remove(&name);
                }
                self.provider_remove_armed.remove(&name);
            }
            ServerToClient::CodexAuthUpdated {
                backend,
                correlation_id,
            } => {
                // Only close the sub-form if this ack matches our own
                // pending rotation. (The server currently only unicasts
                // this event to the requester, but treating a stray
                // broadcast as benign costs nothing.)
                if let Some(settings) = self.settings_modal.as_mut() {
                    let is_ours = settings.codex_rotate.as_ref().is_some_and(|s| {
                        s.pending_correlation.is_some()
                            && s.pending_correlation == correlation_id
                            && s.backend == backend
                    });
                    if is_ours {
                        settings.codex_rotate = None;
                        settings.codex_rotate_banner = Some(Ok(backend));
                    }
                }
            }
            ServerToClient::SharedMcpHostsList { hosts, .. } => {
                self.shared_mcp_hosts = hosts;
            }
            ServerToClient::SharedMcpOauthFlowStarted {
                authorization_url,
                correlation_id,
                name: _,
            } => {
                // Open the authorization URL in a new browser tab
                // and flip the editor form into "waiting for
                // authorization" mode so the operator can't send
                // another request. The final SharedMcpHostAdded
                // (or Error) resolves the flow.
                if let Some(settings) = self.settings_modal.as_mut()
                    && let Some(sub) = settings.shared_mcp_editor.as_mut()
                    && sub.pending_correlation.is_some()
                    && sub.pending_correlation == correlation_id
                {
                    sub.oauth_in_flight = true;
                }
                open_in_new_tab(&authorization_url);
            }
            ServerToClient::SharedMcpHostAdded {
                host,
                correlation_id,
            }
            | ServerToClient::SharedMcpHostUpdated {
                host,
                correlation_id,
            } => {
                // Replace-or-insert by name so adds and updates both
                // converge the local list without a round-trip.
                if let Some(existing) = self
                    .shared_mcp_hosts
                    .iter_mut()
                    .find(|h| h.name == host.name)
                {
                    *existing = host.clone();
                } else {
                    self.shared_mcp_hosts.push(host.clone());
                    self.shared_mcp_hosts.sort_by(|a, b| a.name.cmp(&b.name));
                }
                // When the open editor's pending correlation matches,
                // the server accepted the save — close the form and
                // show a success banner.
                if let Some(settings) = self.settings_modal.as_mut() {
                    let close_editor = settings.shared_mcp_editor.as_ref().is_some_and(|s| {
                        s.pending_correlation.is_some() && s.pending_correlation == correlation_id
                    });
                    if close_editor {
                        settings.shared_mcp_editor = None;
                        settings.shared_mcp_banner = Some(Ok(host.name.clone()));
                    }
                }
            }
            ServerToClient::SharedMcpHostRemoved {
                name,
                correlation_id: _,
            } => {
                self.shared_mcp_hosts.retain(|h| h.name != name);
                if let Some(settings) = self.settings_modal.as_mut() {
                    settings.shared_mcp_remove_armed.remove(&name);
                    settings.shared_mcp_banner = Some(Ok(format!("Removed `{name}`.")));
                }
            }
            ServerToClient::SudoRequested {
                function_id,
                thread_id,
                tool_name,
                args,
                reason,
            } => {
                self.pending_sudos.insert(
                    function_id,
                    PendingSudo {
                        thread_id,
                        tool_name,
                        args,
                        reason,
                    },
                );
            }
            ServerToClient::SudoResolved { function_id, .. } => {
                self.pending_sudos.remove(&function_id);
                self.sudo_reject_drafts.remove(&function_id);
            }
            ServerToClient::ServerConfigFetched {
                toml_text,
                correlation_id,
            } => {
                if let Some(settings) = self.settings_modal.as_mut()
                    && let Some(editor) = settings.server_config.as_mut()
                    && correlation_id.is_some()
                    && editor.fetch_correlation == correlation_id
                {
                    editor.original = Some(toml_text.clone());
                    editor.working = toml_text;
                    editor.fetch_correlation = None;
                }
            }
            ServerToClient::FeedPollAccepted { .. } => {
                // Server acknowledged the manual "Poll now" trigger.
                // No UI state to update — the user infers success
                // by watching the bucket's stats refresh on the
                // next ListBuckets / build event. If we add a
                // per-tick wire broadcast in the future, surface
                // it on the row instead of relying on
                // ListBuckets.
            }
            ServerToClient::ServerConfigUpdateResult {
                cancelled_threads,
                restart_required_sections,
                pods_with_missing_backends,
                correlation_id,
            } => {
                if let Some(settings) = self.settings_modal.as_mut()
                    && let Some(editor) = settings.server_config.as_mut()
                    && correlation_id.is_some()
                    && editor.save_correlation == correlation_id
                {
                    // Successful save: the working text is now the
                    // authoritative on-disk content — seed
                    // `original` so the "modified" indicator
                    // correctly shows no diff and the Revert button
                    // greys out.
                    editor.original = Some(editor.working.clone());
                    editor.save_correlation = None;
                    editor.banner = Some(Ok(ServerConfigSaveSummary {
                        cancelled_threads,
                        restart_required_sections,
                        pods_with_missing_backends,
                    }));
                }
            }
        }
    }
}

/// Insert or replace a bucket in the wire-side cache, keeping it sorted
/// by id (matches the server's `BTreeMap` ordering so the WebUI list
/// is stable across `BucketsList` snapshots and incremental updates).
fn upsert_bucket(
    buckets: &mut Vec<whisper_agent_protocol::BucketSummary>,
    incoming: whisper_agent_protocol::BucketSummary,
) {
    if let Some(existing) = buckets.iter_mut().find(|b| b.id == incoming.id) {
        *existing = incoming;
        return;
    }
    let pos = buckets
        .binary_search_by(|b| b.id.as_str().cmp(incoming.id.as_str()))
        .unwrap_or_else(|p| p);
    buckets.insert(pos, incoming);
}

/// Thread id of a `ServerToClient` event for the purpose of flushing a
/// pending sync-tool-batch append. Returns `None` for the three tool-
/// streaming events (Begin / Content / End) — those are the *tool*
/// phase that the flag was armed for, so they must not flush — and for
/// events that aren't associated with a single thread (pod/resource
/// catalog updates, acks, etc.), which are also outside the per-thread
/// append stream.
pub(super) fn pending_tool_batch_flush_thread_id(msg: &ServerToClient) -> Option<&str> {
    match msg {
        // Tool streaming — do not flush.
        ServerToClient::ThreadToolCallBegin { .. }
        | ServerToClient::ThreadToolCallContent { .. }
        | ServerToClient::ThreadToolCallEnd { .. } => None,
        // Snapshot resets the counter itself; let its handler take
        // over rather than flush here.
        ServerToClient::ThreadSnapshot { .. } => None,
        // Per-thread events — flush before processing.
        ServerToClient::ThreadUserMessage { thread_id, .. }
        | ServerToClient::ThreadToolResultMessage { thread_id, .. }
        | ServerToClient::ThreadAssistantBegin { thread_id, .. }
        | ServerToClient::ThreadPrefillProgress { thread_id, .. }
        | ServerToClient::ThreadToolCallStreaming { thread_id, .. }
        | ServerToClient::ThreadAssistantTextDelta { thread_id, .. }
        | ServerToClient::ThreadAssistantReasoningDelta { thread_id, .. }
        | ServerToClient::ThreadAssistantImage { thread_id, .. }
        | ServerToClient::ThreadAssistantEnd { thread_id, .. }
        | ServerToClient::ThreadLoopComplete { thread_id, .. }
        | ServerToClient::ThreadStateChanged { thread_id, .. }
        | ServerToClient::ThreadTitleUpdated { thread_id, .. }
        | ServerToClient::ThreadDraftUpdated { thread_id, .. }
        | ServerToClient::ThreadArchived { thread_id } => Some(thread_id.as_str()),
        _ => None,
    }
}
