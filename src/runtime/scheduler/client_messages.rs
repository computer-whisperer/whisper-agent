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

use super::{ConnId, Scheduler};
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

/// Translate the direct-auth portion of a wire-shape
/// `SharedMcpAuthInput` (secrets inbound) into the catalog's
/// `SharedMcpAuth`. Returns `None` for `Oauth2Start` — the Add
/// handler inspects the raw input first and routes `Oauth2Start`
/// into the OAuth flow rather than the synchronous add path. So a
/// `None` return here is a caller-error signal, not a domain value.
pub(super) fn shared_mcp_auth_from_direct_input(
    input: whisper_agent_protocol::SharedMcpAuthInput,
) -> Option<crate::tools::shared_mcp_catalog::SharedMcpAuth> {
    use crate::tools::shared_mcp_catalog::SharedMcpAuth;
    use whisper_agent_protocol::SharedMcpAuthInput;
    match input {
        SharedMcpAuthInput::None => Some(SharedMcpAuth::None),
        SharedMcpAuthInput::Bearer { token } => Some(SharedMcpAuth::Bearer { token }),
        SharedMcpAuthInput::Oauth2Start { .. } => None,
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
                initial_attachments,
                config_override,
                bindings_request,
            } => {
                let spec = crate::functions::Function::CreateThread {
                    pod_id,
                    initial_message: Some(initial_message),
                    initial_attachments,
                    parent: None,
                    wait_mode: crate::functions::WaitMode::ThreadCreated,
                    config_override,
                    bindings_request,
                };

                let caller = crate::functions::CallerLink::WsClient {
                    conn_id,
                    correlation_id: correlation_id.clone(),
                };
                match self.register_function(spec, caller) {
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
            ClientToServer::SendUserMessage {
                thread_id,
                text,
                attachments,
            } => {
                if self.tasks.contains_key(&thread_id) {
                    self.rebind_escalation_if_orphaned(&thread_id, conn_id);
                    self.send_user_message(&thread_id, text, attachments, pending_io);
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
            ClientToServer::CancelThread { thread_id } => {
                // Route through the Function registry. See
                // `src/runtime/scheduler/functions.rs` and
                // `docs/design_functions.md`. CancelThread is the first
                // operation migrated (Phase 2).
                let spec = crate::functions::Function::CancelThread {
                    thread_id: thread_id.clone(),
                };

                let caller = crate::functions::CallerLink::WsClient {
                    conn_id,
                    // CancelThread has no correlation_id on the wire —
                    // it produces no direct reply (the visible effect is
                    // the broadcast ThreadStateChanged). `None` is
                    // correct per the CorrelationId contract.
                    correlation_id: None,
                };
                match self.register_function(spec, caller) {
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
                self.archive_thread(&thread_id, pending_io);
            }
            ClientToServer::RecoverThread {
                thread_id,
                correlation_id,
            } => {
                if let Err(msg) = self.recover_thread(&thread_id) {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: Some(thread_id),
                            message: format!("recover_thread: {msg}"),
                        },
                    );
                }
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

                let caller = crate::functions::CallerLink::WsClient {
                    conn_id,
                    correlation_id: correlation_id.clone(),
                };
                match self.register_function(spec, caller) {
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
                reset_capabilities,
                correlation_id,
            } => match self.fork_task(
                Some(conn_id),
                correlation_id.clone(),
                &thread_id,
                from_message_index,
                reset_capabilities,
                pending_io,
            ) {
                Ok(_) => {
                    if archive_original {
                        self.archive_thread(&thread_id, pending_io);
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
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::ThreadSnapshot {
                            thread_id: thread_id.clone(),
                            snapshot,
                        },
                    );
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
                let tasks = self.tasks.values().map(|t| t.summary()).collect();
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
                        auth_mode: entry.auth_mode.clone(),
                    })
                    .collect();
                self.router.send_to_client(
                    conn_id,
                    ServerToClient::BackendsList {
                        correlation_id,
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
                let providers = self.host_env_provider_snapshot();
                self.router.send_to_client(
                    conn_id,
                    ServerToClient::HostEnvProvidersList {
                        correlation_id,
                        providers,
                    },
                );
            }
            ClientToServer::ListBuckets { correlation_id } => {
                let buckets = self.bucket_registry.summaries();
                self.router.send_to_client(
                    conn_id,
                    ServerToClient::BucketsList {
                        correlation_id,
                        buckets,
                    },
                );
            }
            ClientToServer::QueryBuckets {
                correlation_id,
                bucket_ids,
                query,
                top_k,
            } => {
                self.handle_query_buckets(conn_id, correlation_id, bucket_ids, query, top_k);
            }
            ClientToServer::CreateBucket {
                correlation_id,
                id,
                config,
            } => {
                self.handle_create_bucket(conn_id, correlation_id, id, config);
            }
            ClientToServer::DeleteBucket { correlation_id, id } => {
                self.handle_delete_bucket(conn_id, correlation_id, id);
            }
            ClientToServer::StartBucketBuild { correlation_id, id } => {
                self.handle_start_bucket_build(conn_id, correlation_id, id);
            }
            ClientToServer::CancelBucketBuild { correlation_id, id } => {
                self.handle_cancel_bucket_build(conn_id, correlation_id, id);
            }
            ClientToServer::AddHostEnvProvider {
                correlation_id,
                name,
                url,
                token,
            } => match self.add_host_env_provider(name, url, token) {
                Ok(provider) => {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::HostEnvProviderAdded {
                            correlation_id,
                            provider,
                        },
                    );
                }
                Err(e) => {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: None,
                            message: format!("add_host_env_provider: {e}"),
                        },
                    );
                }
            },
            ClientToServer::UpdateHostEnvProvider {
                correlation_id,
                name,
                url,
                token,
            } => match self.update_host_env_provider(name, url, token) {
                Ok(provider) => {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::HostEnvProviderUpdated {
                            correlation_id,
                            provider,
                        },
                    );
                }
                Err(e) => {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: None,
                            message: format!("update_host_env_provider: {e}"),
                        },
                    );
                }
            },
            ClientToServer::RemoveHostEnvProvider {
                correlation_id,
                name,
            } => match self.remove_host_env_provider(&name) {
                Ok(()) => {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::HostEnvProviderRemoved {
                            correlation_id,
                            name,
                        },
                    );
                }
                Err(e) => {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: None,
                            message: format!("remove_host_env_provider: {e}"),
                        },
                    );
                }
            },
            ClientToServer::ListSharedMcpHosts { correlation_id } => {
                let hosts = self.shared_mcp_hosts_snapshot();
                self.router.send_to_client(
                    conn_id,
                    ServerToClient::SharedMcpHostsList {
                        correlation_id,
                        hosts,
                    },
                );
            }
            ClientToServer::AddSharedMcpHost {
                correlation_id,
                name,
                url,
                auth,
            } => {
                if !self.admin_connections.contains(&conn_id) {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: None,
                            message:
                                "add_shared_mcp_host: admin capability required (token is not in [[auth.admins]])"
                                    .into(),
                        },
                    );
                    return;
                }
                if let Err(e) = self.validate_shared_mcp_add(&name, &url) {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: None,
                            message: format!("add_shared_mcp_host: {e}"),
                        },
                    );
                    return;
                }
                // Route on the auth variant: Oauth2Start dispatches
                // the discovery+DCR+authz-URL pipeline; direct
                // None/Bearer go straight to the synchronous connect
                // path.
                match auth {
                    whisper_agent_protocol::SharedMcpAuthInput::Oauth2Start {
                        scope,
                        redirect_base,
                    } => {
                        self.start_shared_mcp_oauth(
                            conn_id,
                            correlation_id,
                            name,
                            url,
                            scope,
                            redirect_base,
                            pending_io,
                        );
                    }
                    other => {
                        let catalog_auth = shared_mcp_auth_from_direct_input(other)
                            .expect("non-Oauth2Start input has a direct mapping");
                        pending_io.push(
                            crate::runtime::io_dispatch::build_shared_mcp_connect_future(
                                conn_id,
                                correlation_id,
                                name,
                                url,
                                catalog_auth,
                                crate::runtime::io_dispatch::SharedMcpOp::Add,
                            ),
                        );
                    }
                }
            }
            ClientToServer::UpdateSharedMcpHost {
                correlation_id,
                name,
                url,
                auth,
            } => {
                if !self.admin_connections.contains(&conn_id) {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: None,
                            message:
                                "update_shared_mcp_host: admin capability required (token is not in [[auth.admins]])"
                                    .into(),
                        },
                    );
                    return;
                }
                // Update refuses Oauth2Start — OAuth hosts must be
                // created fresh (discovery + DCR happen at Add time
                // and aren't re-runnable on an existing catalog entry
                // in a principled way). Direct None/Bearer edits pass
                // through.
                let new_auth = match auth {
                    Some(input) => match shared_mcp_auth_from_direct_input(input) {
                        Some(a) => Some(a),
                        None => {
                            self.router.send_to_client(
                                conn_id,
                                ServerToClient::Error {
                                    correlation_id,
                                    thread_id: None,
                                    message: "update_shared_mcp_host: Oauth2Start is only valid on AddSharedMcpHost"
                                        .into(),
                                },
                            );
                            return;
                        }
                    },
                    None => None,
                };
                let effective_auth =
                    match self.validate_shared_mcp_update(&name, &url, new_auth.as_ref()) {
                        Ok(a) => a,
                        Err(e) => {
                            self.router.send_to_client(
                                conn_id,
                                ServerToClient::Error {
                                    correlation_id,
                                    thread_id: None,
                                    message: format!("update_shared_mcp_host: {e}"),
                                },
                            );
                            return;
                        }
                    };
                pending_io.push(
                    crate::runtime::io_dispatch::build_shared_mcp_connect_future(
                        conn_id,
                        correlation_id,
                        name,
                        url,
                        effective_auth,
                        crate::runtime::io_dispatch::SharedMcpOp::Update,
                    ),
                );
            }
            ClientToServer::RemoveSharedMcpHost {
                correlation_id,
                name,
            } => {
                if !self.admin_connections.contains(&conn_id) {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: None,
                            message:
                                "remove_shared_mcp_host: admin capability required (token is not in [[auth.admins]])"
                                    .into(),
                        },
                    );
                    return;
                }
                match self.remove_shared_mcp_host(&name) {
                    Ok(()) => {
                        self.router.send_to_client(
                            conn_id,
                            ServerToClient::SharedMcpHostRemoved {
                                correlation_id,
                                name,
                            },
                        );
                    }
                    Err(e) => {
                        self.router.send_to_client(
                            conn_id,
                            ServerToClient::Error {
                                correlation_id,
                                thread_id: None,
                                message: format!("remove_shared_mcp_host: {e}"),
                            },
                        );
                    }
                }
            }
            ClientToServer::UpdateCodexAuth {
                correlation_id,
                backend,
                contents,
            } => {
                if !self.admin_connections.contains(&conn_id) {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: None,
                            message:
                                "update_codex_auth: admin capability required (token is not in [[auth.admins]])"
                                    .into(),
                        },
                    );
                    return;
                }
                let Some(entry) = self.backends.get(&backend) else {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: None,
                            message: format!("update_codex_auth: unknown backend `{backend}`"),
                        },
                    );
                    return;
                };
                let provider = entry.provider.clone();
                let Some(outbound) = self.router.client_outbound(conn_id) else {
                    // Client disconnected between send and dispatch —
                    // nothing to ack to. Skip the work; the next admin
                    // session can retry.
                    return;
                };
                tokio::spawn(async move {
                    let result = provider.update_codex_auth(&contents).await;
                    let event = match result {
                        Ok(()) => ServerToClient::CodexAuthUpdated {
                            correlation_id,
                            backend,
                        },
                        Err(e) => ServerToClient::Error {
                            correlation_id,
                            thread_id: None,
                            message: format!("update_codex_auth: {e}"),
                        },
                    };
                    let _ = outbound.send(event);
                });
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
            ClientToServer::FetchServerConfig { correlation_id } => {
                if !self.admin_connections.contains(&conn_id) {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: None,
                            message: "fetch_server_config: admin capability required (token is not in [[auth.admins]])".into(),
                        },
                    );
                    return;
                }
                self.handle_fetch_server_config(conn_id, correlation_id);
            }
            ClientToServer::UpdateServerConfig {
                correlation_id,
                toml_text,
            } => {
                if !self.admin_connections.contains(&conn_id) {
                    self.router.send_to_client(
                        conn_id,
                        ServerToClient::Error {
                            correlation_id,
                            thread_id: None,
                            message: "update_server_config: admin capability required (token is not in [[auth.admins]])".into(),
                        },
                    );
                    return;
                }
                self.handle_update_server_config(conn_id, correlation_id, toml_text, pending_io);
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
                    self.cancel_tokens.remove(thread_id);
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

                let caller = crate::functions::CallerLink::WsClient {
                    conn_id,
                    correlation_id: correlation_id.clone(),
                };
                match self.register_function(spec, caller) {
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
                                    context_window: m.context_window,
                                    max_output_tokens: m.max_output_tokens,
                                    capabilities: m.capabilities,
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
            ClientToServer::ResolveSudo {
                function_id,
                decision,
                reason,
            } => {
                self.resolve_sudo(conn_id, function_id, decision, reason, pending_io);
            }
        }
    }

    /// `QueryBuckets` body — extracted because the handler is large
    /// (resolve providers, spawn the async query, format results) and
    /// the giant match in `apply_client_message` already runs long.
    fn handle_query_buckets(
        &self,
        conn_id: ConnId,
        correlation_id: Option<String>,
        bucket_ids: Vec<String>,
        query: String,
        top_k: u32,
    ) {
        use std::sync::Arc;

        use tokio_util::sync::CancellationToken;
        use whisper_agent_protocol::QueryHit;

        use crate::knowledge::{Bucket, QueryEngine, QueryParams};

        let send_err = |msg: String| {
            self.router.send_to_client(
                conn_id,
                ServerToClient::Error {
                    correlation_id: correlation_id.clone(),
                    thread_id: None,
                    message: msg,
                },
            );
        };

        // Validation: exactly one bucket id, non-empty query, non-zero
        // top_k. Multi-bucket fan-out comes when the dimension-matching
        // UX is figured out.
        if bucket_ids.len() != 1 {
            send_err(format!(
                "QueryBuckets v1 supports exactly one bucket per call, got {}",
                bucket_ids.len(),
            ));
            return;
        }
        if query.trim().is_empty() {
            send_err("QueryBuckets: empty query".into());
            return;
        }
        if top_k == 0 {
            send_err("QueryBuckets: top_k must be > 0".into());
            return;
        }

        let bucket_id = bucket_ids.into_iter().next().unwrap();
        let entry = match self.bucket_registry.buckets.get(&bucket_id) {
            Some(e) => e,
            None => {
                send_err(format!("unknown bucket id: {bucket_id}"));
                return;
            }
        };

        let embedder_name = entry.config.defaults.embedder.clone();
        let embedder = match self.embedding_providers.get(&embedder_name) {
            Some(e) => e.provider.clone(),
            None => {
                send_err(format!(
                    "bucket `{bucket_id}` references embedder `{embedder_name}` which is not configured under [embedding_providers.*]",
                ));
                return;
            }
        };

        // Reranker: pick any configured one. Per-bucket reranker
        // selection lands when bucket.toml grows the field.
        let reranker = match self.rerank_providers.values().next() {
            Some(e) => e.provider.clone(),
            None => {
                send_err(
                    "no reranker configured ([rerank_providers.*] is empty); QueryBuckets needs one"
                        .into(),
                );
                return;
            }
        };

        let Some(outbound) = self.router.outbound(conn_id) else {
            return;
        };
        let registry = self.bucket_registry.clone();

        // Detached task: HNSW rebuild on first load and the subsequent
        // dense+sparse+rerank round trip both want to be off the
        // scheduler thread.
        tokio::spawn(async move {
            let cancel = CancellationToken::new();
            let bucket = match registry.loaded_bucket(&bucket_id).await {
                Ok(b) => b,
                Err(e) => {
                    let _ = outbound.send(ServerToClient::Error {
                        correlation_id,
                        thread_id: None,
                        message: format!("loaded_bucket({bucket_id}): {e}"),
                    });
                    return;
                }
            };
            let engine = QueryEngine::new(embedder, reranker);
            let buckets: Vec<Arc<dyn Bucket>> = vec![bucket];
            let params = QueryParams {
                top_k: top_k as usize,
                ..Default::default()
            };
            match engine.query(&buckets, &query, &params, &cancel).await {
                Ok(results) => {
                    let hits: Vec<QueryHit> = results
                        .into_iter()
                        .map(|r| QueryHit {
                            bucket_id: r.bucket_id.to_string(),
                            chunk_id: r.chunk_id.to_string(),
                            chunk_text: r.chunk_text,
                            source_path: format!("{:?}", r.source_path).to_lowercase(),
                            source_score: r.source_score,
                            rerank_score: r.rerank_score,
                            source_id: r.source_ref.source_id,
                            source_locator: r.source_ref.locator,
                        })
                        .collect();
                    let _ = outbound.send(ServerToClient::QueryResults {
                        correlation_id,
                        query,
                        hits,
                    });
                }
                Err(e) => {
                    let _ = outbound.send(ServerToClient::Error {
                        correlation_id,
                        thread_id: None,
                        message: format!("query failed: {e}"),
                    });
                }
            }
        });
    }
}
