//! `apply_client_message` — the giant match on `ClientToServer` variants.
//!
//! Every wire message a client can send lands here and is dispatched to
//! the matching scheduler action. Split out of the main scheduler
//! module because the dispatch switch alone runs 730+ lines and makes
//! the surrounding lifecycle / resource code harder to read.
//!
//! The method still lives on `Scheduler` — this file just continues the
//! impl block. Private-to-parent methods (`mark_dirty`, `create_task`,
//! `apply_pod_config_update`, `release_thread_resources`, etc.) are
//! visible because child modules inherit access to their parent's
//! privates.

use futures::stream::FuturesUnordered;
use tracing::warn;
use whisper_agent_protocol::{BackendSummary, ClientToServer, ModelSummary, ServerToClient};

use super::{ConnId, Scheduler, pending_approvals_of};
use crate::functions::RejectReason;
use crate::pod::Pod;
use crate::runtime::io_dispatch::SchedulerFuture;

/// Extract a user-facing detail string from a `RejectReason`. Surfaces
/// on WS error frames when a Function fails to register.
pub(super) fn reject_reason_detail(r: &RejectReason) -> &str {
    match r {
        RejectReason::ScopeDenied { detail }
        | RejectReason::PreconditionFailed { detail }
        | RejectReason::InvalidSpec { detail }
        | RejectReason::ResourceBusy { detail } => detail,
    }
}

impl Scheduler {
    pub(super) fn apply_client_message(
        &mut self,
        conn_id: ConnId,
        msg: ClientToServer,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        match msg {
            ClientToServer::CreateThread {
                correlation_id,
                pod_id,
                initial_message,
                config_override,
                bindings_request,
            } => {
                let spec = crate::functions::Function::CreateThread {
                    pod_id,
                    initial_message: Some(initial_message),
                    parent: None,
                    wait_mode: crate::functions::WaitMode::ThreadCreated,
                    config_override,
                    bindings_request,
                };
                let scope = self.ws_client_scope();
                let caller = crate::functions::CallerLink::WsClient {
                    conn_id,
                    correlation_id: correlation_id.clone(),
                };
                match self.register_function(spec, scope, caller) {
                    Ok(fn_id) => self.launch_function(fn_id, pending_io),
                    Err(e) => {
                        warn!(error = ?e, conn_id, "create_thread rejected");
                        self.router.send_to_client(
                            conn_id,
                            ServerToClient::Error {
                                correlation_id,
                                thread_id: None,
                                message: format!("create_thread: {}", reject_reason_detail(&e)),
                            },
                        );
                    }
                }
            }
            ClientToServer::SendUserMessage { thread_id, text } => {
                if self.tasks.contains_key(&thread_id) {
                    self.send_user_message(&thread_id, text, pending_io);
                    self.step_until_blocked(&thread_id, pending_io);
                } else {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id: None,
                            thread_id: Some(thread_id),
                            message: "unknown task".into(),
                        },
                    );
                }
            }
            ClientToServer::ApprovalDecision {
                thread_id,
                approval_id,
                decision,
                remember,
            } => {
                if !self.tasks.contains_key(&thread_id) {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id: None,
                            thread_id: Some(thread_id),
                            message: "unknown task".into(),
                        },
                    );
                    return;
                }
                // Approval now routes to the tool-call Function whose
                // caller-link matches (thread_id, approval_id's
                // tool_use_id). The Function either pushes the
                // deferred IO future (Approve) or synthesizes a denial
                // (Reject); thread-side state is unchanged until the
                // synthetic or real completion lands.
                let resolved = self.resolve_tool_approval(
                    &thread_id,
                    &approval_id,
                    decision,
                    remember,
                    pending_io,
                );
                if resolved {
                    self.mark_dirty(&thread_id);
                    self.step_until_blocked(&thread_id, pending_io);
                } else {
                    // Stale or duplicate decision — no Function was
                    // waiting on this approval id. Ignore silently;
                    // the UI shouldn't surface this as an error
                    // because race conditions can produce benign
                    // duplicates.
                    warn!(
                        %thread_id, %approval_id,
                        "ApprovalDecision for unknown or already-resolved approval"
                    );
                }
            }
            ClientToServer::RemoveToolAllowlistEntry {
                thread_id,
                tool_name,
            } => {
                let removed = self
                    .tasks
                    .get_mut(&thread_id)
                    .map(|t| (t.remove_from_allowlist(&tool_name), t.allowlist_snapshot()));
                match removed {
                    Some((true, snapshot)) => {
                        self.mark_dirty(&thread_id);
                        self.router.broadcast_to_subscribers(
                            &thread_id,
                            ServerToClient::ThreadAllowlistUpdated {
                                thread_id: thread_id.clone(),
                                tool_allowlist: snapshot,
                            },
                        );
                    }
                    Some((false, _)) => { /* nothing to remove — silent no-op */ }
                    None => self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id: None,
                            thread_id: Some(thread_id),
                            message: "unknown task".into(),
                        },
                    ),
                }
            }
            ClientToServer::RebindThread {
                thread_id,
                patch,
                correlation_id,
            } => {
                // Route through the Function registry. Validation
                // errors inside apply_rebind surface as an Error
                // terminal, which complete_function routes back to
                // the originating conn_id + correlation_id.
                let spec = crate::functions::Function::RebindThread {
                    thread_id: thread_id.clone(),
                    patch,
                };
                let scope = self.ws_client_scope();
                let caller = crate::functions::CallerLink::WsClient {
                    conn_id,
                    correlation_id: correlation_id.clone(),
                };
                match self.register_function(spec, scope, caller) {
                    Ok(fn_id) => self.launch_function(fn_id, pending_io),
                    Err(e) => {
                        warn!(error = ?e, conn_id, %thread_id, "rebind_thread rejected");
                        self.router.send_to_client(
                            conn_id,
                            ServerToClient::Error {
                                correlation_id,
                                thread_id: Some(thread_id),
                                message: format!("rebind: {}", reject_reason_detail(&e)),
                            },
                        );
                    }
                }
            }
            ClientToServer::CancelThread { thread_id } => {
                // Route through the Function registry. See
                // `src/runtime/scheduler/functions.rs` and
                // `docs/design_functions.md`. CancelThread is the first
                // operation migrated (Phase 2).
                let spec = crate::functions::Function::CancelThread {
                    thread_id: thread_id.clone(),
                };
                let scope = self.ws_client_scope();
                let caller = crate::functions::CallerLink::WsClient {
                    conn_id,
                    // CancelThread has no correlation_id on the wire —
                    // it produces no direct reply (the visible effect is
                    // the broadcast ThreadStateChanged). `None` is
                    // correct per the CorrelationId contract.
                    correlation_id: None,
                };
                match self.register_function(spec, scope, caller) {
                    Ok(fn_id) => self.launch_function(fn_id, pending_io),
                    Err(e) => {
                        warn!(error = ?e, conn_id, thread_id, "CancelThread rejected");
                        // Pre-migration the handler silently ignored
                        // unknown thread ids; preserve that UX by
                        // dropping PreconditionFailed without an error
                        // event. Scope-denied would be surfaced here
                        // once per-identity scopes land.
                    }
                }
            }
            ClientToServer::ArchiveThread { thread_id } => {
                self.archive_thread(&thread_id);
            }
            ClientToServer::CompactThread {
                thread_id,
                correlation_id,
            } => {
                // Route through the Function registry — same path as
                // auto-compact takes with its SchedulerInternal caller.
                let spec = crate::functions::Function::CompactThread {
                    thread_id: thread_id.clone(),
                };
                let scope = self.ws_client_scope();
                let caller = crate::functions::CallerLink::WsClient {
                    conn_id,
                    correlation_id: correlation_id.clone(),
                };
                match self.register_function(spec, scope, caller) {
                    Ok(fn_id) => self.launch_function(fn_id, pending_io),
                    Err(e) => {
                        warn!(error = ?e, conn_id, %thread_id, "compact_thread rejected");
                        self.router.send_to_client(
                            conn_id,
                            ServerToClient::Error {
                                correlation_id,
                                thread_id: Some(thread_id),
                                message: format!("compact_thread: {}", reject_reason_detail(&e)),
                            },
                        );
                    }
                }
            }
            ClientToServer::SetThreadDraft { thread_id, text } => {
                if let Some(task) = self.tasks.get_mut(&thread_id) {
                    // Guard against no-op writes — keeps disk idle
                    // under keystroke-level traffic and avoids
                    // broadcast echo loops when two clients converge.
                    if task.draft != text {
                        task.draft = text.clone();
                        task.touch();
                        self.mark_dirty(&thread_id);
                        self.router.broadcast_to_subscribers_except(
                            &thread_id,
                            ServerToClient::ThreadDraftUpdated {
                                thread_id: thread_id.clone(),
                                text,
                            },
                            conn_id,
                        );
                    }
                } else {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id: None,
                            thread_id: Some(thread_id),
                            message: "unknown task".into(),
                        },
                    );
                }
            }
            ClientToServer::ForkThread {
                thread_id,
                from_message_index,
                archive_original,
                correlation_id,
            } => match self.fork_task(
                Some(conn_id),
                correlation_id.clone(),
                &thread_id,
                from_message_index,
                pending_io,
            ) {
                Ok(_) => {
                    if archive_original {
                        self.archive_thread(&thread_id);
                    }
                }
                Err(e) => {
                    warn!(error = %e, conn_id, %thread_id, "fork_task rejected");
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: Some(thread_id),
                            message: format!("fork_thread: {e}"),
                        },
                    );
                }
            },
            ClientToServer::SubscribeToThread { thread_id } => {
                if let Some(task) = self.tasks.get(&thread_id) {
                    self.router.subscribe(conn_id, &thread_id);
                    let snapshot = task.snapshot();
                    // Rehydrate any still-pending approvals so the newly-subscribed
                    // client can render the approval UI. The snapshot itself doesn't
                    // carry approval state.
                    let pending = pending_approvals_of(self, task);
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::ThreadSnapshot {
                            thread_id: thread_id.clone(),
                            snapshot,
                        },
                    );
                    for event in pending {
                        self.router.send_to_client(conn_id, event);
                    }
                } else {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id: None,
                            thread_id: Some(thread_id),
                            message: "unknown task".into(),
                        },
                    );
                }
            }
            ClientToServer::UnsubscribeFromThread { thread_id } => {
                self.router.unsubscribe(conn_id, &thread_id);
            }
            ClientToServer::ListThreads { correlation_id } => {
                let tasks = self
                    .tasks
                    .values()
                    .filter(|t| !t.archived)
                    .map(|t| t.summary())
                    .collect();
                self.router.send_to_client(
                    conn_id,
                    ServerToClient::ThreadList {
                        correlation_id,
                        tasks,
                    },
                );
            }
            ClientToServer::ListBackends { correlation_id } => {
                let backends: Vec<BackendSummary> = self
                    .backends
                    .iter()
                    .map(|(name, entry)| BackendSummary {
                        name: name.clone(),
                        kind: entry.kind.clone(),
                        default_model: entry.default_model.clone(),
                    })
                    .collect();
                self.router.send_to_client(
                    conn_id,
                    ServerToClient::BackendsList {
                        correlation_id,
                        default_backend: self.default_backend.clone(),
                        backends,
                    },
                );
            }
            ClientToServer::ListResources { correlation_id } => {
                let resources = self.resources.snapshot_all();
                self.router.send_to_client(
                    conn_id,
                    ServerToClient::ResourceList {
                        correlation_id,
                        resources,
                    },
                );
            }
            ClientToServer::ListHostEnvProviders { correlation_id } => {
                let providers = self.host_env_registry.snapshot();
                self.router.send_to_client(
                    conn_id,
                    ServerToClient::HostEnvProvidersList {
                        correlation_id,
                        providers,
                    },
                );
            }
            ClientToServer::ListPods { correlation_id } => {
                let Some((persister, outbound)) = self.persister_and_outbound(conn_id) else {
                    return;
                };
                let default_pod_id = self.default_pod_id.clone();
                tokio::spawn(async move {
                    match persister.list_pods().await {
                        Ok(pods) => {
                            let _ = outbound.send(ServerToClient::PodList {
                                correlation_id,
                                pods,
                                default_pod_id,
                            });
                        }
                        Err(e) => {
                            let _ = outbound.send(ServerToClient::Error {
                                correlation_id,
                                thread_id: None,
                                message: format!("list_pods: {e}"),
                            });
                        }
                    }
                });
            }
            ClientToServer::ListPodDir {
                correlation_id,
                pod_id,
                path,
            } => {
                let Some((persister, outbound)) = self.persister_and_outbound(conn_id) else {
                    return;
                };
                let rel = path.unwrap_or_default();
                tokio::spawn(async move {
                    match persister.list_pod_dir(&pod_id, &rel).await {
                        Ok(entries) => {
                            let _ = outbound.send(ServerToClient::PodDirListing {
                                correlation_id,
                                pod_id,
                                path: rel,
                                entries,
                            });
                        }
                        Err(e) => {
                            let _ = outbound.send(ServerToClient::Error {
                                correlation_id,
                                thread_id: None,
                                message: format!("list_pod_dir: {e}"),
                            });
                        }
                    }
                });
            }
            ClientToServer::ReadPodFile {
                correlation_id,
                pod_id,
                path,
            } => {
                let Some((persister, outbound)) = self.persister_and_outbound(conn_id) else {
                    return;
                };
                tokio::spawn(async move {
                    match persister.read_pod_file(&pod_id, &path).await {
                        Ok((content, readonly)) => {
                            let _ = outbound.send(ServerToClient::PodFileContent {
                                correlation_id,
                                pod_id,
                                path,
                                content,
                                readonly,
                            });
                        }
                        Err(e) => {
                            let _ = outbound.send(ServerToClient::Error {
                                correlation_id,
                                thread_id: None,
                                message: format!("read_pod_file: {e}"),
                            });
                        }
                    }
                });
            }
            ClientToServer::WritePodFile {
                correlation_id,
                pod_id,
                path,
                content,
            } => {
                let Some((persister, outbound)) = self.persister_and_outbound(conn_id) else {
                    return;
                };
                tokio::spawn(async move {
                    match persister.write_pod_file(&pod_id, &path, &content).await {
                        Ok(()) => {
                            let _ = outbound.send(ServerToClient::PodFileWritten {
                                correlation_id,
                                pod_id,
                                path,
                            });
                        }
                        Err(e) => {
                            let _ = outbound.send(ServerToClient::Error {
                                correlation_id,
                                thread_id: None,
                                message: format!("write_pod_file: {e}"),
                            });
                        }
                    }
                });
            }
            ClientToServer::GetPod {
                correlation_id,
                pod_id,
            } => {
                let Some((persister, outbound)) = self.persister_and_outbound(conn_id) else {
                    return;
                };
                tokio::spawn(async move {
                    match persister.get_pod(&pod_id).await {
                        Ok(Some(snapshot)) => {
                            let _ = outbound.send(ServerToClient::PodSnapshot {
                                snapshot,
                                correlation_id,
                            });
                        }
                        Ok(None) => {
                            let _ = outbound.send(ServerToClient::Error {
                                correlation_id,
                                thread_id: None,
                                message: format!("pod `{pod_id}` not found"),
                            });
                        }
                        Err(e) => {
                            let _ = outbound.send(ServerToClient::Error {
                                correlation_id,
                                thread_id: None,
                                message: format!("get_pod: {e}"),
                            });
                        }
                    }
                });
            }
            ClientToServer::CreatePod {
                correlation_id,
                pod_id,
                mut config,
            } => {
                // Sync work first: validate + register in-memory so a
                // CreateThread that races behind this can find the pod.
                // The disk write happens in the background (logged on
                // failure); the in-memory state is what subsequent
                // CreateThread / GetPod calls consult.
                if let Err(e) = crate::pod::persist::validate_pod_id(&pod_id) {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: None,
                            message: format!("create_pod: {e}"),
                        },
                    );
                    return;
                }
                if config.created_at.is_empty() {
                    config.created_at = chrono::Utc::now().to_rfc3339();
                }
                if let Err(e) = crate::pod::validate(&config) {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: None,
                            message: format!("create_pod: {e}"),
                        },
                    );
                    return;
                }
                if self.pods.contains_key(&pod_id) {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: None,
                            message: format!("create_pod: pod `{pod_id}` already exists"),
                        },
                    );
                    return;
                }
                let raw_toml = match crate::pod::to_toml(&config) {
                    Ok(t) => t,
                    Err(e) => {
                        self.router.send_to_client(
                            conn_id,
                            ServerToClient::Error {
                                correlation_id,
                                thread_id: None,
                                message: format!("create_pod: encode toml: {e}"),
                            },
                        );
                        return;
                    }
                };
                let pod_dir = self
                    .persister
                    .as_ref()
                    .map(|p| p.dir().join(&pod_id))
                    .unwrap_or_else(|| std::path::PathBuf::from(&pod_id));
                // Inherit the default pod's system prompt so the new
                // pod's threads have a working prompt out of the box —
                // otherwise a newly-created pod runs with an empty
                // system prompt until the user hand-writes one, and
                // Anthropic in particular 400s when cache_control
                // lands on an empty system block. Matches the
                // expectation set by the "+ New pod" UX, which clones
                // the default pod's `thread_defaults` template.
                let system_prompt = self
                    .pods
                    .get(&self.default_pod_id)
                    .map(|p| p.system_prompt.clone())
                    .unwrap_or_default();
                let pod = Pod::new(
                    pod_id.clone(),
                    pod_dir,
                    config.clone(),
                    raw_toml,
                    system_prompt.clone(),
                );
                let summary = whisper_agent_protocol::PodSummary {
                    pod_id: pod_id.clone(),
                    name: config.name.clone(),
                    description: config.description.clone(),
                    created_at: config.created_at.clone(),
                    thread_count: 0,
                    archived: false,
                    behaviors_enabled: true,
                };
                self.pods.insert(pod_id.clone(), pod);

                // Broadcast PodCreated to every connected client.
                let ev = ServerToClient::PodCreated {
                    pod: summary,
                    correlation_id,
                };
                for tx in self.router.outbound_snapshot() {
                    let _ = tx.send(ev.clone());
                }

                // Disk write runs in the background. Failure logs warn —
                // the in-memory pod stays usable but won't survive a
                // restart. (No good story for rollback today; the user
                // can retry by deleting and recreating.)
                if let Some(persister) = self.persister.clone() {
                    let pid = pod_id.clone();
                    let prompt = system_prompt.clone();
                    tokio::spawn(async move {
                        if let Err(e) = persister.create_pod(&pid, config, &prompt).await {
                            warn!(pod_id = %pid, error = %e, "create_pod disk write failed");
                        }
                    });
                }
            }
            ClientToServer::UpdatePodConfig {
                correlation_id,
                pod_id,
                toml_text,
            } => {
                if let Err(e) = crate::pod::persist::validate_pod_id(&pod_id) {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: None,
                            message: format!("update_pod_config: {e}"),
                        },
                    );
                    return;
                }
                if !self.pods.contains_key(&pod_id) {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: None,
                            message: format!("update_pod_config: unknown pod `{pod_id}`"),
                        },
                    );
                    return;
                }
                let parsed = match crate::pod::parse_toml(&toml_text) {
                    Ok(c) => c,
                    Err(e) => {
                        self.router.send_to_client(
                            conn_id,
                            ServerToClient::Error {
                                correlation_id,
                                thread_id: None,
                                message: format!("update_pod_config: {e}"),
                            },
                        );
                        return;
                    }
                };
                self.apply_pod_config_update(&pod_id, toml_text.clone(), parsed, correlation_id);
                self.persist_pod_config(pod_id, toml_text, Some(conn_id));
            }
            ClientToServer::ArchivePod { pod_id } => {
                if let Err(e) = crate::pod::persist::validate_pod_id(&pod_id) {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id: None,
                            thread_id: None,
                            message: format!("archive_pod: {e}"),
                        },
                    );
                    return;
                }
                if pod_id == self.default_pod_id {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id: None,
                            thread_id: None,
                            message: "archive_pod: refusing to archive the default pod".into(),
                        },
                    );
                    return;
                }
                let Some(pod) = self.pods.remove(&pod_id) else {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id: None,
                            thread_id: None,
                            message: format!("archive_pod: unknown pod `{pod_id}`"),
                        },
                    );
                    return;
                };
                // Drop in-memory thread state for every thread under this pod.
                // The pod directory is moving to .archived/, so these threads
                // become unreachable — we'd otherwise leak entries in
                // `tasks` / `dirty` / `router` subs, and the
                // resource-registry user sets would carry stale thread_ids
                // that prevent GC from ever marking those resources idle.
                // The per-HostEnvId provisioning guard doesn't need
                // clearing here — it's shared across threads on the same
                // deduped id and the in-flight future will clear itself on
                // completion.
                for thread_id in &pod.threads {
                    let bindings = self.tasks.get(thread_id).map(|t| t.bindings.clone());
                    self.tasks.remove(thread_id);
                    self.dirty.remove(thread_id);
                    self.router.drop_thread(thread_id);
                    if let Some(bindings) = bindings {
                        self.release_thread_resources(thread_id, &pod_id, &bindings);
                    }
                }
                // Broadcast first so every client clears its view before the
                // disk move completes; the disk write is best-effort.
                let ev = ServerToClient::PodArchived {
                    pod_id: pod_id.clone(),
                };
                for tx in self.router.outbound_snapshot() {
                    let _ = tx.send(ev.clone());
                }
                if let Some(persister) = self.persister.clone() {
                    let outbound = self.router.outbound(conn_id);
                    tokio::spawn(async move {
                        if let Err(e) = persister.archive_pod(&pod_id).await {
                            warn!(pod_id = %pod_id, error = %e, "archive_pod disk move failed");
                            if let Some(tx) = outbound {
                                let _ = tx.send(ServerToClient::Error {
                                    correlation_id: None,
                                    thread_id: None,
                                    message: format!("archive_pod: disk move failed: {e}"),
                                });
                            }
                        }
                    });
                }
            }
            ClientToServer::ListBehaviors {
                correlation_id,
                pod_id,
            } => {
                let Some(pod) = self.pods.get(&pod_id) else {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: None,
                            message: format!("list_behaviors: unknown pod `{pod_id}`"),
                        },
                    );
                    return;
                };
                let behaviors: Vec<_> = pod.behaviors.values().map(|b| b.summary()).collect();
                self.router.send_to_client(
                    conn_id,
                    ServerToClient::BehaviorList {
                        correlation_id,
                        pod_id,
                        behaviors,
                    },
                );
            }
            ClientToServer::GetBehavior {
                correlation_id,
                pod_id,
                behavior_id,
            } => {
                let Some(pod) = self.pods.get(&pod_id) else {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: None,
                            message: format!("get_behavior: unknown pod `{pod_id}`"),
                        },
                    );
                    return;
                };
                let Some(behavior) = pod.behaviors.get(&behavior_id) else {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: None,
                            message: format!(
                                "get_behavior: unknown behavior `{behavior_id}` under pod `{pod_id}`"
                            ),
                        },
                    );
                    return;
                };
                self.router.send_to_client(
                    conn_id,
                    ServerToClient::BehaviorSnapshot {
                        correlation_id,
                        snapshot: behavior.snapshot(),
                    },
                );
            }
            ClientToServer::RunBehavior {
                correlation_id,
                pod_id,
                behavior_id,
                payload,
            } => {
                // Manual runs bypass overlap policy entirely — the
                // button in the UI is an explicit user action. Route
                // through the Function registry with a WsClient
                // caller-link so errors flow back to the originator.
                let spec = crate::functions::Function::RunBehavior {
                    pod_id: pod_id.clone(),
                    behavior_id: behavior_id.clone(),
                    payload: payload.unwrap_or(serde_json::Value::Null),
                };
                let scope = self.ws_client_scope();
                let caller = crate::functions::CallerLink::WsClient {
                    conn_id,
                    correlation_id: correlation_id.clone(),
                };
                match self.register_function(spec, scope, caller) {
                    Ok(fn_id) => self.launch_function(fn_id, pending_io),
                    Err(e) => {
                        self.router.send_to_client(
                            conn_id,
                            ServerToClient::Error {
                                correlation_id,
                                thread_id: None,
                                message: format!("run_behavior: {}", reject_reason_detail(&e)),
                            },
                        );
                    }
                }
            }
            ClientToServer::CreateBehavior {
                correlation_id,
                pod_id,
                behavior_id,
                config,
                prompt,
            } => {
                self.handle_create_behavior(
                    conn_id,
                    correlation_id,
                    pod_id,
                    behavior_id,
                    config,
                    prompt,
                );
            }
            ClientToServer::UpdateBehavior {
                correlation_id,
                pod_id,
                behavior_id,
                config,
                prompt,
            } => {
                self.handle_update_behavior(
                    conn_id,
                    correlation_id,
                    pod_id,
                    behavior_id,
                    config,
                    prompt,
                );
            }
            ClientToServer::DeleteBehavior {
                correlation_id,
                pod_id,
                behavior_id,
            } => {
                self.handle_delete_behavior(conn_id, correlation_id, pod_id, behavior_id);
            }
            ClientToServer::SetBehaviorEnabled {
                correlation_id,
                pod_id,
                behavior_id,
                enabled,
            } => {
                self.handle_set_behavior_enabled(
                    conn_id,
                    correlation_id,
                    pod_id,
                    behavior_id,
                    enabled,
                );
            }
            ClientToServer::SetPodBehaviorsEnabled {
                correlation_id,
                pod_id,
                enabled,
            } => {
                self.handle_set_pod_behaviors_enabled(conn_id, correlation_id, pod_id, enabled);
            }
            ClientToServer::ListModels {
                correlation_id,
                backend,
            } => {
                let entry = self.backends.get(&backend);
                let Some(entry) = entry else {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: None,
                            message: format!("unknown backend `{backend}`"),
                        },
                    );
                    return;
                };
                // Spawn a detached task so we don't block the scheduler loop on the
                // backend's network round-trip. The task writes directly to the
                // client's outbound channel — the scheduler holds no intermediate state.
                let provider = entry.provider.clone();
                let Some(outbound) = self.router.outbound(conn_id) else {
                    return;
                };
                tokio::spawn(async move {
                    match provider.list_models().await {
                        Ok(models) => {
                            let models: Vec<ModelSummary> = models
                                .into_iter()
                                .map(|m| ModelSummary {
                                    id: m.id,
                                    display_name: m.display_name,
                                })
                                .collect();
                            let _ = outbound.send(ServerToClient::ModelsList {
                                correlation_id,
                                backend,
                                models,
                            });
                        }
                        Err(e) => {
                            let _ = outbound.send(ServerToClient::Error {
                                correlation_id,
                                thread_id: None,
                                message: format!("list_models failed: {e}"),
                            });
                        }
                    }
                });
            }
            ClientToServer::ListFunctions { correlation_id } => {
                // Snapshot the `active_functions` registry for the
                // caller's own channel. Fresh connects replay this
                // once on join; subsequent lifecycle is delivered via
                // `FunctionStarted` / `FunctionEnded` broadcasts.
                let functions: Vec<_> = self
                    .active_functions
                    .values()
                    .map(|entry| entry.wire_summary())
                    .collect();
                self.router.send_to_client(
                    conn_id,
                    ServerToClient::FunctionList {
                        correlation_id,
                        functions,
                    },
                );
            }
        }
    }
}
