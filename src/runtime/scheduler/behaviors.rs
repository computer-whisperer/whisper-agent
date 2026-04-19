//! Behavior scheduling — the cron/webhook/manual-fire code path.
//!
//! Everything from trigger evaluation (which behaviors are due, which
//! are paused) through the actual thread spawn and post-terminal
//! bookkeeping lives here. Split out of the main scheduler module so the
//! behaviors state machine can be read end-to-end without the general
//! client-message dispatch and resource plumbing intercut with it.
//!
//! All methods are still on `Scheduler` — this file just continues the
//! `impl` block from `scheduler.rs`. Visibility of private methods and
//! fields works because a child module inherits access to its parent's
//! privates.

use futures::stream::FuturesUnordered;
use tracing::{info, warn};
use whisper_agent_protocol::{ServerToClient, ThreadStateLabel};

use super::thread_config::{behavior_override_to_requests, render_behavior_prompt};
use super::triggers::{
    count_missed_occurrences, is_cron_due, parse_rfc3339_utc, validate_webhook_target,
};
use super::{ConnId, Scheduler, TriggerFireError};
use crate::runtime::io_dispatch::SchedulerFuture;

impl Scheduler {
    /// Validate a webhook-delivered trigger and dispatch it through
    /// the shared `fire_trigger` path. Returns `Ok` on successful
    /// dispatch (including overlap-queued and overlap-skipped cases
    /// — those aren't sender errors). Returns `Err` only when the
    /// behavior itself can't service the trigger.
    pub(super) fn handle_webhook_trigger(
        &mut self,
        pod_id: &str,
        behavior_id: &str,
        payload: serde_json::Value,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) -> Result<(), TriggerFireError> {
        let pod = self.pods.get(pod_id).ok_or(TriggerFireError::UnknownPod)?;
        let behavior = pod
            .behaviors
            .get(behavior_id)
            .ok_or(TriggerFireError::UnknownBehavior)?;
        if !pod.state.behaviors_enabled || !behavior.state.enabled {
            return Err(TriggerFireError::Paused);
        }
        let overlap = validate_webhook_target(behavior)?;
        self.fire_trigger(
            pod_id,
            behavior_id,
            payload,
            overlap,
            crate::functions::TriggerSource::Webhook,
            pending_io,
        );
        Ok(())
    }

    /// True when automatic triggers for this behavior should be
    /// suppressed — either the behavior itself or its pod is paused.
    /// Manual `RunBehavior` bypasses this check; it never calls here.
    /// Unknown pod/behavior returns `true` (fail-closed) so a race
    /// against a concurrent delete can't re-fire a dying behavior.
    pub(super) fn behavior_auto_paused(&self, pod_id: &str, behavior_id: &str) -> bool {
        let Some(pod) = self.pods.get(pod_id) else {
            return true;
        };
        if !pod.state.behaviors_enabled {
            return true;
        }
        let Some(behavior) = pod.behaviors.get(behavior_id) else {
            return true;
        };
        !behavior.state.enabled
    }
    /// Spawn a thread from a behavior trigger. Resolves the behavior's
    /// effective thread config + bindings against the pod's defaults and
    /// `[allow]` cap, renders the prompt template with the payload, and
    /// drives the resulting thread through `create_task` →
    /// `send_user_message` → `step_until_blocked` like a normal
    /// interactive spawn. The thread carries `BehaviorOrigin` so the
    /// on-terminal hook knows to update behavior state when it finishes.
    ///
    /// `requester` is `Some` for UI-initiated `RunBehavior` (so the
    /// requester's `ThreadCreated` carries their correlation_id) and
    /// `None` for cron/webhook/etc fires that arrive without a client
    /// connection. See `create_task` for the headless-spawn semantics.
    pub(super) fn run_behavior(
        &mut self,
        requester: Option<ConnId>,
        correlation_id: Option<String>,
        pod_id: &str,
        behavior_id: &str,
        payload: Option<serde_json::Value>,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) -> Result<String, String> {
        // Snapshot everything we need out of the behavior entry up front,
        // so we don't hold an immutable borrow of `self.pods` across the
        // `create_task` / `send_user_message` calls (both of which take
        // `&mut self`).
        let (config, prompt) = {
            let pod = self
                .pods
                .get(pod_id)
                .ok_or_else(|| format!("unknown pod `{pod_id}`"))?;
            let behavior = pod
                .behaviors
                .get(behavior_id)
                .ok_or_else(|| format!("unknown behavior `{behavior_id}` under pod `{pod_id}`"))?;
            if let Some(err) = &behavior.load_error {
                return Err(format!(
                    "behavior `{behavior_id}` failed to load and cannot fire: {err}"
                ));
            }
            let config = behavior
                .config
                .clone()
                .ok_or_else(|| "behavior has no parsed config".to_string())?;
            (config, behavior.prompt.clone())
        };

        let payload = payload.unwrap_or(serde_json::Value::Null);
        let rendered_prompt = render_behavior_prompt(&prompt, &payload);

        let (config_override, bindings_request) = behavior_override_to_requests(&config.thread);

        let origin = whisper_agent_protocol::BehaviorOrigin {
            behavior_id: behavior_id.to_string(),
            fired_at: chrono::Utc::now().to_rfc3339(),
            trigger_payload: payload,
        };

        let thread_id = self.create_task(
            requester,
            correlation_id,
            Some(pod_id.to_string()),
            config_override,
            bindings_request,
            Some(origin),
            None,
            pending_io,
        )?;
        self.mark_dirty(&thread_id);
        self.send_user_message(&thread_id, rendered_prompt, pending_io);
        self.step_until_blocked(&thread_id, pending_io);
        // Every fire (manual or trigger-driven) advances `last_fired_at`
        // — for cron, this is the cursor for the next evaluation; for
        // manual, it's the "last run" timestamp the UI shows. Actual
        // run_count / last_outcome still wait for the thread to reach
        // terminal; those fields track completions, not fires.
        if let Some(pod) = self.pods.get_mut(pod_id)
            && let Some(behavior) = pod.behaviors.get_mut(behavior_id)
        {
            behavior.state.last_fired_at = Some(chrono::Utc::now().to_rfc3339());
            let state = behavior.state.clone();
            self.mark_behavior_dirty(pod_id, behavior_id);
            self.router
                .broadcast_task_list(ServerToClient::BehaviorStateChanged {
                    pod_id: pod_id.to_string(),
                    behavior_id: behavior_id.to_string(),
                    state,
                });
        }
        Ok(thread_id)
    }

    /// Scan every loaded pod for cron-triggered behaviors whose next
    /// scheduled fire has arrived, and dispatch them per their overlap
    /// policy. Runs on the cron ticker branch (every `CRON_TICK_SECS`).
    ///
    /// Per-iteration cost is O(behaviors with Cron triggers): each one
    /// is a `Schedule::after` call (cheap — pre-parsed) and one
    /// timestamp comparison. At expected scale (≤few hundred) this is
    /// not a hot path.
    pub(super) fn fire_due_cron_behaviors(
        &mut self,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        let now = chrono::Utc::now();

        // First pass: prime `last_fired_at` for any cron behavior that
        // has never fired. This is the cursor `Schedule::after(cursor)`
        // reads; without a stable cursor, `cron.after(now)` keeps
        // advancing with `now` and we skip the first real fire window.
        // Initializing to `now` means the first fire happens at the
        // next scheduled occurrence after this tick — at most one tick
        // late, bounded by CRON_TICK_SECS.
        let to_prime: Vec<(String, String)> = self
            .pods
            .iter()
            .flat_map(|(pid, pod)| {
                pod.behaviors.iter().filter_map(move |(bid, b)| {
                    if b.cron.is_some() && b.state.last_fired_at.is_none() {
                        Some((pid.clone(), bid.clone()))
                    } else {
                        None
                    }
                })
            })
            .collect();
        // Priming is cheap state-initialization; it runs even for
        // paused behaviors so the moment they resume, we don't prime
        // the cursor to a stale time. The actual fire-decision pass
        // below honors the pause gate.
        for (pod_id, behavior_id) in to_prime {
            if let Some(pod) = self.pods.get_mut(&pod_id)
                && let Some(behavior) = pod.behaviors.get_mut(&behavior_id)
            {
                behavior.state.last_fired_at = Some(now.to_rfc3339());
                self.mark_behavior_dirty(&pod_id, &behavior_id);
            }
        }

        // Second pass: collect behaviors whose next scheduled fire is
        // now past. Done as a gather-then-dispatch because firing
        // takes `&mut self.pods` and we can't hold an iteration borrow.
        let mut due: Vec<(String, String, whisper_agent_protocol::Overlap)> = Vec::new();
        for (pod_id, pod) in &self.pods {
            if pod.archived || !pod.state.behaviors_enabled {
                continue;
            }
            for (behavior_id, behavior) in &pod.behaviors {
                if !behavior.state.enabled {
                    continue;
                }
                let Some(cron) = behavior.cron.as_ref() else {
                    continue;
                };
                let overlap = match behavior.config.as_ref().map(|c| &c.trigger) {
                    Some(whisper_agent_protocol::TriggerSpec::Cron { overlap, .. }) => *overlap,
                    _ => continue,
                };
                let Some(cursor) = parse_rfc3339_utc(behavior.state.last_fired_at.as_deref())
                else {
                    // Priming pass above set this; if it's still None
                    // something is wrong — log and skip.
                    warn!(
                        pod_id = %pod_id,
                        behavior_id = %behavior_id,
                        "cron behavior missing last_fired_at after priming; skipping tick"
                    );
                    continue;
                };
                if is_cron_due(cron, cursor, now) {
                    due.push((pod_id.clone(), behavior_id.clone(), overlap));
                }
            }
        }

        for (pod_id, behavior_id, overlap) in due {
            // Cron fires carry no payload (the trigger itself has no
            // data to transmit); downstream `fire_trigger` still takes
            // a payload for uniformity with webhook delivery.
            self.fire_trigger(
                &pod_id,
                &behavior_id,
                serde_json::Value::Null,
                overlap,
                crate::functions::TriggerSource::Cron,
                pending_io,
            );
        }
    }

    /// Apply overlap policy and dispatch a trigger-driven behavior run.
    /// Shared by the cron ticker and the webhook handler; any future
    /// event-source trigger (slack, imap, file-watch) calls this too.
    ///
    /// `source` is a short tag used only in log lines — the behavior's
    /// trigger variant is the source of truth for overlap; this string
    /// just makes warn-logs readable.
    pub(super) fn fire_trigger(
        &mut self,
        pod_id: &str,
        behavior_id: &str,
        payload: serde_json::Value,
        overlap: whisper_agent_protocol::Overlap,
        source: crate::functions::TriggerSource,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        use whisper_agent_protocol::Overlap;
        let inflight = matches!(overlap, Overlap::Skip | Overlap::QueueOne)
            && self.behavior_has_inflight_run(pod_id, behavior_id);
        if inflight {
            match overlap {
                Overlap::Skip => {
                    tracing::debug!(
                        source = %source.as_str(),
                        pod_id = %pod_id,
                        behavior_id = %behavior_id,
                        "trigger fire skipped: previous run still in flight"
                    );
                    return;
                }
                Overlap::QueueOne => {
                    // Park the payload for consumption by the
                    // on-terminal hook. Overwriting an existing queued
                    // payload is deliberate — QueueOne keeps at most
                    // one pending; later arrivals replace earlier.
                    self.queue_behavior_payload(pod_id, behavior_id, payload);
                    return;
                }
                Overlap::Allow => unreachable!("excluded from `inflight` check"),
            }
        }
        // Allow branch, or Skip/QueueOne with no in-flight run: route
        // through the Function registry with a SchedulerInternal
        // caller-link tagged by trigger source.
        self.register_and_launch_behavior_fire(pod_id, behavior_id, payload, source, pending_io);
    }

    /// Register a `Function::RunBehavior` with a
    /// `SchedulerInternal(BehaviorFire)` caller-link and launch it.
    /// Shared by `fire_trigger` (cron / webhook) and the
    /// on-terminal QueueOne replay path.
    pub(super) fn register_and_launch_behavior_fire(
        &mut self,
        pod_id: &str,
        behavior_id: &str,
        payload: serde_json::Value,
        source: crate::functions::TriggerSource,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        let spec = crate::functions::Function::RunBehavior {
            pod_id: pod_id.to_string(),
            behavior_id: behavior_id.to_string(),
            payload,
        };
        let scope = self.internal_scope();
        let caller = crate::functions::CallerLink::SchedulerInternal(
            crate::functions::InternalOriginator::BehaviorFire {
                pod_id: pod_id.to_string(),
                behavior_id: behavior_id.to_string(),
                source,
            },
        );
        match self.register_function(spec, scope, caller) {
            Ok(fn_id) => self.launch_function(fn_id, pending_io),
            Err(e) => warn!(
                source = %source.as_str(),
                pod_id = %pod_id,
                behavior_id = %behavior_id,
                error = ?e,
                "trigger-driven RunBehavior rejected at registration"
            ),
        }
    }

    /// True when the behavior's `last_thread_id` refers to a thread
    /// that is currently present in memory AND in a non-terminal
    /// state (`Idle`/`Working`). Used by the Skip and QueueOne
    /// overlap paths. Returns false when the behavior has never
    /// fired, when the last thread has been evicted, or when the
    /// last thread reached Completed/Failed/Cancelled.
    pub(super) fn behavior_has_inflight_run(&self, pod_id: &str, behavior_id: &str) -> bool {
        let Some(pod) = self.pods.get(pod_id) else {
            return false;
        };
        let Some(behavior) = pod.behaviors.get(behavior_id) else {
            return false;
        };
        let Some(thread_id) = behavior.state.last_thread_id.as_ref() else {
            return false;
        };
        let Some(task) = self.tasks.get(thread_id) else {
            return false;
        };
        !matches!(
            task.public_state(),
            ThreadStateLabel::Completed | ThreadStateLabel::Failed | ThreadStateLabel::Cancelled
        )
    }

    /// One-time pass at scheduler startup: apply each cron behavior's
    /// `CatchUp` policy to its persisted cursor.
    ///
    /// Without catch-up logic, a behavior whose `last_fired_at` is
    /// stale enough that `cron.after(last_fired_at)` is now in the
    /// past would fire on the very first cron tick after restart —
    /// exactly once, because after firing the cursor advances past
    /// `now`. That default matches `CatchUp::One` perfectly, so this
    /// pass is structured as "tweak the cursor for the other cases":
    ///
    /// - `None`: advance `last_fired_at` to `now`, so the first
    ///   tick sees `cron.after(now)` and doesn't fire.
    /// - `One`: no-op. The first tick's natural behavior fires one.
    /// - `All`: currently behaves like `One` but logs the count of
    ///   missed occurrences. The design explicitly calls out this
    ///   variant as almost never desirable; capping to one avoids
    ///   hammering the scheduler with stale firings after extended
    ///   downtime.
    ///
    /// Behaviors that have never fired (`last_fired_at == None`) are
    /// handled by the tick-time priming pass, not here — catch-up
    /// doesn't apply to something that hasn't happened yet.
    pub(super) fn evaluate_cron_catch_up(&mut self) {
        let now = chrono::Utc::now();

        // One shot gather of all cron behaviors with a persisted
        // cursor that's now stale (next occurrence already past). We
        // collect enough state to log and dispatch in a second pass,
        // keeping the mutable iterations in the later loop.
        struct Missed {
            pod_id: String,
            behavior_id: String,
            catch_up: whisper_agent_protocol::CatchUp,
            missed_count: u64,
        }
        let mut missed: Vec<Missed> = Vec::new();
        for (pod_id, pod) in &self.pods {
            if !pod.state.behaviors_enabled {
                continue;
            }
            for (behavior_id, behavior) in &pod.behaviors {
                if !behavior.state.enabled {
                    continue;
                }
                let Some(cron) = behavior.cron.as_ref() else {
                    continue;
                };
                let Some(cursor) = parse_rfc3339_utc(behavior.state.last_fired_at.as_deref())
                else {
                    continue; // never fired — priming pass handles this
                };
                if !is_cron_due(cron, cursor, now) {
                    continue;
                }
                let catch_up = match behavior.config.as_ref().map(|c| &c.trigger) {
                    Some(whisper_agent_protocol::TriggerSpec::Cron { catch_up, .. }) => *catch_up,
                    _ => continue,
                };
                // Count missed occurrences for logging only. Capped
                // so a grossly-stale cursor (years) doesn't spin.
                let missed_count = count_missed_occurrences(cron, cursor, now, 1000);
                missed.push(Missed {
                    pod_id: pod_id.clone(),
                    behavior_id: behavior_id.clone(),
                    catch_up,
                    missed_count,
                });
            }
        }

        for m in missed {
            match m.catch_up {
                // Advance the cursor past all missed windows; the
                // first tick after startup will see `cron.after(now)`
                // and won't fire.
                whisper_agent_protocol::CatchUp::None => {
                    info!(
                        pod_id = %m.pod_id,
                        behavior_id = %m.behavior_id,
                        missed = m.missed_count,
                        "cron catch_up=none: suppressing missed fires, advancing cursor"
                    );
                    if let Some(pod) = self.pods.get_mut(&m.pod_id)
                        && let Some(behavior) = pod.behaviors.get_mut(&m.behavior_id)
                    {
                        behavior.state.last_fired_at = Some(now.to_rfc3339());
                        self.mark_behavior_dirty(&m.pod_id, &m.behavior_id);
                    }
                }
                // No-op: the first tick's natural flow fires exactly
                // once (because `is_cron_due` returns true) and then
                // advances the cursor via `run_behavior`.
                whisper_agent_protocol::CatchUp::One => {
                    info!(
                        pod_id = %m.pod_id,
                        behavior_id = %m.behavior_id,
                        missed = m.missed_count,
                        "cron catch_up=one: firing once for missed window"
                    );
                }
                // Currently identical to One — capped to prevent
                // flooding. Design reserves this variant for a future
                // case where per-occurrence catch-up is genuinely
                // wanted; log at warn so operators notice.
                whisper_agent_protocol::CatchUp::All => {
                    warn!(
                        pod_id = %m.pod_id,
                        behavior_id = %m.behavior_id,
                        missed = m.missed_count,
                        "cron catch_up=all capped to 1 fire; per-occurrence replay unimplemented"
                    );
                }
            }
        }
    }

    /// Park a payload for a behavior whose overlap policy is
    /// `QueueOne`. Sets `state.queued_payload`, overwriting any
    /// already-parked value (QueueOne = at most one pending; later
    /// arrivals replace earlier). Marks the behavior dirty and
    /// broadcasts `BehaviorStateChanged`. The queued payload is
    /// consumed by `on_behavior_thread_terminal` when the currently
    /// in-flight thread reaches a terminal state.
    pub(super) fn queue_behavior_payload(
        &mut self,
        pod_id: &str,
        behavior_id: &str,
        payload: serde_json::Value,
    ) {
        let Some(pod) = self.pods.get_mut(pod_id) else {
            return;
        };
        let Some(behavior) = pod.behaviors.get_mut(behavior_id) else {
            return;
        };
        behavior.state.queued_payload = Some(payload);
        let state = behavior.state.clone();
        self.mark_behavior_dirty(pod_id, behavior_id);
        self.router
            .broadcast_task_list(ServerToClient::BehaviorStateChanged {
                pod_id: pod_id.to_string(),
                behavior_id: behavior_id.to_string(),
                state,
            });
    }

    /// On-terminal hook for behavior-spawned threads. Idempotent — safe
    /// to call at every teardown site; no-op when the thread either
    /// doesn't exist, has no `origin`, isn't in a terminal state, or has
    /// already been recorded (thread_id already == behavior.state
    /// .last_thread_id with a set outcome). The check against
    /// `last_thread_id` means interactive follow-ups on a Completed
    /// behavior thread don't double-count.
    ///
    /// If `state.queued_payload` is set (QueueOne overlap parked a
    /// payload while this thread was running), consumes it and
    /// re-fires the behavior with that payload — which is why this
    /// takes `pending_io`. The re-fire uses the same `run_behavior`
    /// path, so its terminal will fire this hook again with a fresh
    /// `last_thread_id`; recursion terminates when `queued_payload`
    /// is empty.
    pub(super) fn on_behavior_thread_terminal(
        &mut self,
        thread_id: &str,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        let Some(task) = self.tasks.get(thread_id) else {
            return;
        };
        let Some(origin) = task.origin.clone() else {
            return;
        };
        let outcome = match task.public_state() {
            ThreadStateLabel::Completed => whisper_agent_protocol::BehaviorOutcome::Completed,
            ThreadStateLabel::Failed => whisper_agent_protocol::BehaviorOutcome::Failed {
                message: task.failure_detail().unwrap_or_default(),
            },
            ThreadStateLabel::Cancelled => whisper_agent_protocol::BehaviorOutcome::Cancelled,
            _ => return,
        };
        let pod_id = task.pod_id.clone();
        let Some(pod) = self.pods.get_mut(&pod_id) else {
            return;
        };
        let Some(behavior) = pod.behaviors.get_mut(&origin.behavior_id) else {
            return;
        };
        // Idempotence: if we already recorded this thread's outcome,
        // don't double-count. A Completed thread that receives an
        // interactive follow-up will re-enter Working → Completed and
        // try to fire this hook again; we bail here.
        if behavior.state.last_thread_id.as_deref() == Some(thread_id)
            && behavior.state.last_outcome.is_some()
        {
            return;
        }
        behavior.state.last_thread_id = Some(thread_id.to_string());
        behavior.state.last_outcome = Some(outcome);
        behavior.state.run_count = behavior.state.run_count.saturating_add(1);
        // Take any queued payload before we broadcast — the fire below
        // would see the old state, and we want the wire event to
        // reflect the consumed queue.
        let queued = behavior.state.queued_payload.take();
        let snapshot_state = behavior.state.clone();

        // Persist via the dirty set: `flush_dirty` runs once per
        // scheduler-loop iteration and writes each dirty behavior
        // exactly once, so rapid terminals can't race on disk.
        self.mark_behavior_dirty(&pod_id, &origin.behavior_id);

        // Broadcast so every connected client sees the last-run update.
        self.router
            .broadcast_task_list(ServerToClient::BehaviorStateChanged {
                pod_id: pod_id.clone(),
                behavior_id: origin.behavior_id.clone(),
                state: snapshot_state,
            });

        // Consume the queued payload: re-fire with it. Logged at warn
        // if the fire fails because a queued payload represents
        // user/trigger intent that's getting dropped on the floor —
        // surface it rather than silently swallowing. Drop the
        // payload entirely when the behavior or its pod is now
        // paused: pause means "no automatic fires", and the queued
        // payload was itself an automatic arrival.
        if let Some(payload) = queued {
            if self.behavior_auto_paused(&pod_id, &origin.behavior_id) {
                tracing::debug!(
                    pod_id = %pod_id,
                    behavior_id = %origin.behavior_id,
                    "dropping queued QueueOne payload: behavior/pod paused"
                );
            } else {
                self.register_and_launch_behavior_fire(
                    &pod_id,
                    &origin.behavior_id,
                    payload,
                    crate::functions::TriggerSource::QueuedReplay,
                    pending_io,
                );
            }
        }
    }

    /// `CreateBehavior` handler — validate ids + config, register the
    /// new entry in-memory, broadcast `BehaviorCreated`, kick off the
    /// disk write in the background. Mirrors the `CreatePod` shape: the
    /// in-memory state is immediately queryable; a failing disk write
    /// logs a warn but doesn't roll back the in-memory entry.
    pub(super) fn handle_create_behavior(
        &mut self,
        conn_id: ConnId,
        correlation_id: Option<String>,
        pod_id: String,
        behavior_id: String,
        config: whisper_agent_protocol::BehaviorConfig,
        prompt: String,
    ) {
        if let Err(e) = crate::pod::behaviors::validate_behavior_id(&behavior_id) {
            self.send_behavior_error(conn_id, correlation_id, "create_behavior", e);
            return;
        }
        if let Err(e) = crate::pod::behaviors::validate(&config) {
            self.send_behavior_error(conn_id, correlation_id, "create_behavior", e);
            return;
        }
        let Some(pod) = self.pods.get_mut(&pod_id) else {
            self.send_behavior_error(
                conn_id,
                correlation_id,
                "create_behavior",
                format!("unknown pod `{pod_id}`"),
            );
            return;
        };
        if pod.behaviors.contains_key(&behavior_id) {
            self.send_behavior_error(
                conn_id,
                correlation_id,
                "create_behavior",
                format!("behavior `{behavior_id}` already exists under pod `{pod_id}`"),
            );
            return;
        }
        let raw_toml = match crate::pod::behaviors::to_toml(&config) {
            Ok(t) => t,
            Err(e) => {
                self.send_behavior_error(conn_id, correlation_id, "create_behavior", e);
                return;
            }
        };
        let dir = pod
            .dir
            .join(crate::pod::behaviors::BEHAVIORS_DIR)
            .join(&behavior_id);
        let behavior = crate::pod::behaviors::Behavior {
            id: behavior_id.clone(),
            pod_id: pod_id.clone(),
            dir: dir.clone(),
            cron: crate::pod::behaviors::cache_cron(&config),
            config: Some(config.clone()),
            raw_toml,
            prompt: prompt.clone(),
            state: whisper_agent_protocol::BehaviorState::default(),
            load_error: None,
        };
        let summary = behavior.summary();
        pod.behaviors.insert(behavior_id.clone(), behavior);

        // Broadcast to everyone; correlation_id goes only to the requester.
        let ev = ServerToClient::BehaviorCreated {
            correlation_id: None,
            summary: summary.clone(),
        };
        self.router.broadcast_task_list_except(ev, conn_id);
        self.router.send_to_client(
            conn_id,
            ServerToClient::BehaviorCreated {
                correlation_id,
                summary,
            },
        );

        // Disk write runs in the background. Failure logs a warn — the
        // in-memory entry stays usable but won't survive a restart.
        let pod_dir = self.pods.get(&pod_id).map(|p| p.dir.clone());
        if let Some(pod_dir) = pod_dir {
            let bid = behavior_id;
            tokio::spawn(async move {
                if let Err(e) =
                    crate::pod::behaviors::create_on_disk(&pod_dir, &bid, &config, &prompt).await
                {
                    warn!(behavior_id = %bid, error = %e, "create_behavior disk write failed");
                }
            });
        }
    }

    /// `UpdateBehavior` handler — replace the in-memory entry's config /
    /// prompt / raw_toml (preserving state), broadcast
    /// `BehaviorUpdated`, disk-write in the background.
    pub(super) fn handle_update_behavior(
        &mut self,
        conn_id: ConnId,
        correlation_id: Option<String>,
        pod_id: String,
        behavior_id: String,
        config: whisper_agent_protocol::BehaviorConfig,
        prompt: String,
    ) {
        if let Err(e) = crate::pod::behaviors::validate_behavior_id(&behavior_id) {
            self.send_behavior_error(conn_id, correlation_id, "update_behavior", e);
            return;
        }
        if let Err(e) = crate::pod::behaviors::validate(&config) {
            self.send_behavior_error(conn_id, correlation_id, "update_behavior", e);
            return;
        }
        let Some(pod) = self.pods.get_mut(&pod_id) else {
            self.send_behavior_error(
                conn_id,
                correlation_id,
                "update_behavior",
                format!("unknown pod `{pod_id}`"),
            );
            return;
        };
        let Some(behavior) = pod.behaviors.get_mut(&behavior_id) else {
            self.send_behavior_error(
                conn_id,
                correlation_id,
                "update_behavior",
                format!("unknown behavior `{behavior_id}` under pod `{pod_id}`"),
            );
            return;
        };
        let raw_toml = match crate::pod::behaviors::to_toml(&config) {
            Ok(t) => t,
            Err(e) => {
                self.send_behavior_error(conn_id, correlation_id, "update_behavior", e);
                return;
            }
        };
        behavior.cron = crate::pod::behaviors::cache_cron(&config);
        behavior.config = Some(config.clone());
        behavior.raw_toml = raw_toml;
        behavior.prompt = prompt.clone();
        behavior.load_error = None;
        let snapshot = behavior.snapshot();
        let pod_dir = pod.dir.clone();

        let ev = ServerToClient::BehaviorUpdated {
            correlation_id: None,
            snapshot: snapshot.clone(),
        };
        self.router.broadcast_task_list_except(ev, conn_id);
        self.router.send_to_client(
            conn_id,
            ServerToClient::BehaviorUpdated {
                correlation_id,
                snapshot,
            },
        );

        let bid = behavior_id;
        tokio::spawn(async move {
            if let Err(e) =
                crate::pod::behaviors::update_on_disk(&pod_dir, &bid, &config, &prompt).await
            {
                warn!(behavior_id = %bid, error = %e, "update_behavior disk write failed");
            }
        });
    }

    /// `DeleteBehavior` handler — remove from in-memory pod, broadcast
    /// `BehaviorDeleted`, rmdir in the background. Historical threads
    /// that carry this behavior_id as their origin are untouched; the UI
    /// resolves them as orphaned runs.
    pub(super) fn handle_delete_behavior(
        &mut self,
        conn_id: ConnId,
        correlation_id: Option<String>,
        pod_id: String,
        behavior_id: String,
    ) {
        if let Err(e) = crate::pod::behaviors::validate_behavior_id(&behavior_id) {
            self.send_behavior_error(conn_id, correlation_id, "delete_behavior", e);
            return;
        }
        let Some(pod) = self.pods.get_mut(&pod_id) else {
            self.send_behavior_error(
                conn_id,
                correlation_id,
                "delete_behavior",
                format!("unknown pod `{pod_id}`"),
            );
            return;
        };
        if pod.behaviors.remove(&behavior_id).is_none() {
            self.send_behavior_error(
                conn_id,
                correlation_id,
                "delete_behavior",
                format!("unknown behavior `{behavior_id}` under pod `{pod_id}`"),
            );
            return;
        }
        let pod_dir = pod.dir.clone();

        let ev = ServerToClient::BehaviorDeleted {
            correlation_id: None,
            pod_id: pod_id.clone(),
            behavior_id: behavior_id.clone(),
        };
        self.router.broadcast_task_list_except(ev, conn_id);
        self.router.send_to_client(
            conn_id,
            ServerToClient::BehaviorDeleted {
                correlation_id,
                pod_id,
                behavior_id: behavior_id.clone(),
            },
        );

        let bid = behavior_id;
        tokio::spawn(async move {
            if let Err(e) = crate::pod::behaviors::delete_on_disk(&pod_dir, &bid).await {
                warn!(behavior_id = %bid, error = %e, "delete_behavior disk removal failed");
            }
        });
    }

    /// `SetBehaviorEnabled` handler. Pause / resume a single
    /// behavior, mutating its `state.enabled` flag. On pause we also
    /// drop any `QueueOne`-parked payload (pause means "no automatic
    /// fires", and the queued payload was itself an automatic
    /// arrival). On resume, if the behavior is cron-triggered we
    /// bump `last_fired_at` to `now` so the scheduler doesn't
    /// catch-up-fire for windows missed while paused.
    pub(super) fn handle_set_behavior_enabled(
        &mut self,
        conn_id: ConnId,
        correlation_id: Option<String>,
        pod_id: String,
        behavior_id: String,
        enabled: bool,
    ) {
        let Some(pod) = self.pods.get_mut(&pod_id) else {
            self.send_behavior_error(
                conn_id,
                correlation_id,
                "set_behavior_enabled",
                format!("unknown pod `{pod_id}`"),
            );
            return;
        };
        let Some(behavior) = pod.behaviors.get_mut(&behavior_id) else {
            self.send_behavior_error(
                conn_id,
                correlation_id,
                "set_behavior_enabled",
                format!("unknown behavior `{behavior_id}` under pod `{pod_id}`"),
            );
            return;
        };
        // `BehaviorStateChanged` has no correlation_id field — the
        // broadcast-to-all covers the initiator too. correlation_id
        // is only threaded through the error paths above.
        let _ = correlation_id;
        if behavior.state.enabled == enabled {
            return;
        }
        behavior.state.enabled = enabled;
        if !enabled {
            behavior.state.queued_payload = None;
        } else if behavior.cron.is_some() {
            behavior.state.last_fired_at = Some(chrono::Utc::now().to_rfc3339());
        }
        let snapshot_state = behavior.state.clone();
        self.mark_behavior_dirty(&pod_id, &behavior_id);
        self.router
            .broadcast_task_list(ServerToClient::BehaviorStateChanged {
                pod_id,
                behavior_id,
                state: snapshot_state,
            });
    }

    /// `SetPodBehaviorsEnabled` handler. Flip the pod's master switch
    /// for automatic behavior triggers. Pause clears every per-
    /// behavior queued payload (same pause-semantics as the single-
    /// behavior handler). Resume bumps every cron behavior's cursor
    /// to `now` so catch-up doesn't replay missed windows.
    pub(super) fn handle_set_pod_behaviors_enabled(
        &mut self,
        conn_id: ConnId,
        correlation_id: Option<String>,
        pod_id: String,
        enabled: bool,
    ) {
        let Some(pod) = self.pods.get_mut(&pod_id) else {
            self.send_behavior_error(
                conn_id,
                correlation_id,
                "set_pod_behaviors_enabled",
                format!("unknown pod `{pod_id}`"),
            );
            return;
        };
        if pod.state.behaviors_enabled == enabled {
            self.router.send_to_client(
                conn_id,
                ServerToClient::PodBehaviorsEnabledChanged {
                    correlation_id,
                    pod_id,
                    enabled,
                },
            );
            return;
        }
        pod.state.behaviors_enabled = enabled;
        let pod_state_snapshot = pod.state.clone();
        let pod_dir = pod.dir.clone();
        // Touch every behavior: drop queued payload on pause, bump
        // cron cursor on resume. Collect the ids first so we can
        // broadcast after releasing the &mut pod borrow.
        let mut changed_behaviors: Vec<(String, whisper_agent_protocol::BehaviorState)> =
            Vec::new();
        let now_rfc = chrono::Utc::now().to_rfc3339();
        for (bid, behavior) in pod.behaviors.iter_mut() {
            let mut changed = false;
            if !enabled && behavior.state.queued_payload.is_some() {
                behavior.state.queued_payload = None;
                changed = true;
            }
            if enabled && behavior.cron.is_some() && behavior.state.enabled {
                behavior.state.last_fired_at = Some(now_rfc.clone());
                changed = true;
            }
            if changed {
                changed_behaviors.push((bid.clone(), behavior.state.clone()));
            }
        }
        for (bid, _) in &changed_behaviors {
            self.mark_behavior_dirty(&pod_id, bid);
        }
        let ev = ServerToClient::PodBehaviorsEnabledChanged {
            correlation_id: None,
            pod_id: pod_id.clone(),
            enabled,
        };
        self.router.broadcast_task_list_except(ev, conn_id);
        self.router.send_to_client(
            conn_id,
            ServerToClient::PodBehaviorsEnabledChanged {
                correlation_id,
                pod_id: pod_id.clone(),
                enabled,
            },
        );
        for (bid, state) in changed_behaviors {
            self.router
                .broadcast_task_list(ServerToClient::BehaviorStateChanged {
                    pod_id: pod_id.clone(),
                    behavior_id: bid,
                    state,
                });
        }
        // Persist pod_state.json. Background write; failure warns.
        tokio::spawn(async move {
            if let Err(e) =
                crate::pod::persist::write_pod_state(&pod_dir, &pod_state_snapshot).await
            {
                warn!(
                    pod_dir = %pod_dir.display(),
                    error = %e,
                    "write pod_state.json failed"
                );
            }
        });
    }

    pub(super) fn send_behavior_error(
        &self,
        conn_id: ConnId,
        correlation_id: Option<String>,
        op: &str,
        err: impl std::fmt::Display,
    ) {
        self.router.send_to_client(
            conn_id,
            ServerToClient::Error {
                correlation_id,
                thread_id: None,
                message: format!("{op}: {err}"),
            },
        );
    }
}
