//! Runtime edits to the server-level `whisper-agent.toml`.
//!
//! Two entry points: `handle_fetch_server_config` (admin reads the raw
//! file text) and `handle_update_server_config` (admin replaces the
//! file, and the scheduler hot-swaps the backend catalog in-memory).
//!
//! The update path follows the "validate everything first, then
//! mutate" discipline: parse, validate, build all new `BackendEntry`
//! instances, diff against the current catalog, cancel any thread
//! bound to a removed or modified backend, swap the backend map,
//! write the file, then broadcast a fresh `BackendsList` to all
//! connected clients. If any validation or build step fails, nothing
//! on the scheduler (or on disk) has changed yet — the caller sees a
//! plain `Error` response and the existing threads keep running.
//!
//! Only the `[backends.*]` section hot-swaps. Changes to other
//! sections (shared_mcp_hosts, host_env_providers, secrets, auth) are
//! written to disk but require a server restart to take effect; the
//! response lists which sections are pending so the user knows.

use std::collections::{BTreeMap, HashMap, HashSet};

use futures::stream::FuturesUnordered;
use tracing::warn;
use whisper_agent_protocol::{BackendSummary, ServerToClient};

use super::{ConnId, Scheduler};
use crate::pod::config::Config;
use crate::pod::resources::BackendId;
use crate::runtime::io_dispatch::SchedulerFuture;
use crate::runtime::scheduler::{BackendEntry, EmbeddingProviderEntry, RerankProviderEntry};

/// Diff between an old and new map of named config entries, by name.
/// Generic over the entry value so we can reuse the same logic for
/// backends, embedding providers, and rerank providers — all three are
/// `BTreeMap<String, T>` where `T: PartialEq` and the field-level
/// equality decides "changed".
#[derive(Debug, Default)]
struct CatalogDiff {
    added: Vec<String>,
    removed: Vec<String>,
    changed: Vec<String>,
}

impl CatalogDiff {
    fn is_noop(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.changed.is_empty()
    }
}

fn diff_catalog<'a, I, T>(old: I, new: &BTreeMap<String, T>) -> CatalogDiff
where
    I: IntoIterator<Item = (&'a String, &'a T)>,
    T: PartialEq + 'a,
{
    let old_map: HashMap<&String, &T> = old.into_iter().collect();
    let mut diff = CatalogDiff::default();
    for (name, new_cfg) in new {
        match old_map.get(name) {
            None => diff.added.push(name.clone()),
            Some(old_cfg) if *old_cfg != new_cfg => diff.changed.push(name.clone()),
            _ => {}
        }
    }
    for name in old_map.keys() {
        if !new.contains_key(*name) {
            diff.removed.push((*name).clone());
        }
    }
    diff.added.sort();
    diff.removed.sort();
    diff.changed.sort();
    diff
}

fn restart_required_sections(old: &Config, new: &Config) -> Vec<String> {
    let mut sections = Vec::new();
    if old.shared_mcp_hosts != new.shared_mcp_hosts {
        sections.push("shared_mcp_hosts".into());
    }
    if old.host_env_providers != new.host_env_providers {
        sections.push("host_env_providers".into());
    }
    if old.secrets != new.secrets {
        sections.push("secrets".into());
    }
    if old.auth != new.auth {
        sections.push("auth".into());
    }
    sections
}

impl Scheduler {
    pub(super) fn handle_fetch_server_config(
        &mut self,
        conn_id: ConnId,
        correlation_id: Option<String>,
    ) {
        let Some(path) = self.server_config_path.as_ref() else {
            self.router.send_to_client(
                conn_id,
                ServerToClient::Error {
                    correlation_id,
                    thread_id: None,
                    message: "fetch_server_config: server was started without --config; there is no on-disk file to read".into(),
                },
            );
            return;
        };
        match std::fs::read_to_string(path) {
            Ok(toml_text) => {
                self.router.send_to_client(
                    conn_id,
                    ServerToClient::ServerConfigFetched {
                        correlation_id,
                        toml_text,
                    },
                );
            }
            Err(e) => {
                self.router.send_to_client(
                    conn_id,
                    ServerToClient::Error {
                        correlation_id,
                        thread_id: None,
                        message: format!("fetch_server_config: read {}: {e}", path.display()),
                    },
                );
            }
        }
    }

    pub(super) fn handle_update_server_config(
        &mut self,
        conn_id: ConnId,
        correlation_id: Option<String>,
        toml_text: String,
        pending_io: &mut FuturesUnordered<SchedulerFuture>,
    ) {
        let Some(path) = self.server_config_path.clone() else {
            self.router.send_to_client(
                conn_id,
                ServerToClient::Error {
                    correlation_id,
                    thread_id: None,
                    message: "update_server_config: server was started without --config; cannot persist a runtime edit".into(),
                },
            );
            return;
        };

        let new_cfg: Config = match toml::from_str(&toml_text) {
            Ok(c) => c,
            Err(e) => {
                self.reply_error(conn_id, correlation_id, format!("parse: {e}"));
                return;
            }
        };
        if let Err(e) = new_cfg.validate() {
            self.reply_error(conn_id, correlation_id, format!("validate: {e}"));
            return;
        }

        // Re-read the on-disk file rather than synthesizing a `Config`
        // from in-memory state — runtime catalog edits (Add/RemoveSharedMcpHost,
        // host-env mutations) have drifted from what's on disk, so the
        // file is the only honest baseline for "did this submission
        // actually change a non-backend section".
        let old_cfg: Config = match std::fs::read_to_string(&path) {
            Ok(t) => match toml::from_str(&t) {
                Ok(c) => c,
                Err(e) => {
                    self.reply_error(
                        conn_id,
                        correlation_id,
                        format!("read-back current config for diff: parse {e}"),
                    );
                    return;
                }
            },
            Err(e) => {
                self.reply_error(
                    conn_id,
                    correlation_id,
                    format!("read-back current config for diff: {e}"),
                );
                return;
            }
        };

        // Build every provider before touching scheduler state. A
        // failure here (bad credentials, unreachable auth file)
        // surfaces as a plain Error with nothing yet mutated.
        let mut new_entries: HashMap<String, BackendEntry> = HashMap::new();
        for (name, bcfg) in &new_cfg.backends {
            let provider = match bcfg.build() {
                Ok(p) => p,
                Err(e) => {
                    self.reply_error(
                        conn_id,
                        correlation_id,
                        format!("build backend `{name}`: {e}"),
                    );
                    return;
                }
            };
            new_entries.insert(
                name.clone(),
                BackendEntry {
                    provider,
                    kind: bcfg.kind().into(),
                    default_model: bcfg.default_model().map(str::to_string),
                    auth_mode: bcfg.auth_mode().map(str::to_string),
                    source: bcfg.clone(),
                },
            );
        }

        // Same fail-fast build pass for embedding + rerank providers.
        // Identical pattern: construct everything, surface any error
        // before touching live state, and only then proceed to swap.
        let mut new_embed_entries: HashMap<String, EmbeddingProviderEntry> = HashMap::new();
        for (name, ecfg) in &new_cfg.embedding_providers {
            let provider = match ecfg.build() {
                Ok(p) => p,
                Err(e) => {
                    self.reply_error(
                        conn_id,
                        correlation_id,
                        format!("build embedding provider `{name}`: {e}"),
                    );
                    return;
                }
            };
            new_embed_entries.insert(
                name.clone(),
                EmbeddingProviderEntry {
                    provider,
                    kind: ecfg.kind().into(),
                    auth_mode: ecfg.auth_mode().map(str::to_string),
                    source: ecfg.clone(),
                },
            );
        }
        let mut new_rerank_entries: HashMap<String, RerankProviderEntry> = HashMap::new();
        for (name, rcfg) in &new_cfg.rerank_providers {
            let provider = match rcfg.build() {
                Ok(p) => p,
                Err(e) => {
                    self.reply_error(
                        conn_id,
                        correlation_id,
                        format!("build rerank provider `{name}`: {e}"),
                    );
                    return;
                }
            };
            new_rerank_entries.insert(
                name.clone(),
                RerankProviderEntry {
                    provider,
                    kind: rcfg.kind().into(),
                    auth_mode: rcfg.auth_mode().map(str::to_string),
                    source: rcfg.clone(),
                },
            );
        }

        let diff = diff_catalog(
            self.backends
                .iter()
                .map(|(name, entry)| (name, &entry.source)),
            &new_cfg.backends,
        );
        let embed_diff = diff_catalog(
            self.embedding_providers
                .iter()
                .map(|(name, entry)| (name, &entry.source)),
            &new_cfg.embedding_providers,
        );
        let rerank_diff = diff_catalog(
            self.rerank_providers
                .iter()
                .map(|(name, entry)| (name, &entry.source)),
            &new_cfg.rerank_providers,
        );

        let mut cancel_list: Vec<String> = Vec::new();
        for name in diff.removed.iter().chain(diff.changed.iter()) {
            if let Some(entry) = self.resources.backends.get(&BackendId::for_name(name)) {
                for user in &entry.users {
                    cancel_list.push(user.clone());
                }
            }
        }
        cancel_list.sort();
        cancel_list.dedup();

        // Pods that reference removed backends don't block the update —
        // a thread spawned from such a pod will fail to resolve, and
        // the user is expected to fix the pod.toml (or accept the
        // broken pod).
        let mut pods_with_missing: Vec<String> = Vec::new();
        if !diff.removed.is_empty() {
            let removed: HashSet<&str> = diff.removed.iter().map(String::as_str).collect();
            for (pod_id, pod) in &self.pods {
                if pod
                    .config
                    .allow
                    .backends
                    .iter()
                    .any(|b| removed.contains(b.as_str()))
                {
                    pods_with_missing.push(pod_id.clone());
                }
            }
            pods_with_missing.sort();
        }

        let restart_sections = restart_required_sections(&old_cfg, &new_cfg);
        let write_needed = !diff.is_noop()
            || !embed_diff.is_noop()
            || !rerank_diff.is_noop()
            || !restart_sections.is_empty();

        for thread_id in &cancel_list {
            self.execute_cancel_thread(thread_id, pending_io);
        }

        for name in &diff.removed {
            self.backends.remove(name);
            self.resources.backends.remove(&BackendId::for_name(name));
        }
        for name in diff.added.iter().chain(diff.changed.iter()) {
            if let Some(entry) = new_entries.remove(name) {
                self.resources.insert_backend(
                    name.clone(),
                    entry.kind.clone(),
                    entry.default_model.clone(),
                );
                self.backends.insert(name.clone(), entry);
            }
        }

        // Embedding / rerank providers don't yet have any users
        // (knowledge buckets aren't built), so no thread-cancellation
        // step is needed here. When buckets land they'll bind to a
        // provider; this is where the cancellation list will widen.
        for name in &embed_diff.removed {
            self.embedding_providers.remove(name);
        }
        for name in embed_diff.added.iter().chain(embed_diff.changed.iter()) {
            if let Some(entry) = new_embed_entries.remove(name) {
                self.embedding_providers.insert(name.clone(), entry);
            }
        }
        for name in &rerank_diff.removed {
            self.rerank_providers.remove(name);
        }
        for name in rerank_diff.added.iter().chain(rerank_diff.changed.iter()) {
            if let Some(entry) = new_rerank_entries.remove(name) {
                self.rerank_providers.insert(name.clone(), entry);
            }
        }

        // Disk write follows the in-memory swap: the scheduler is
        // authoritative at runtime, so an order that left disk newer
        // than memory could revert the swap on restart while the
        // operator believed it had landed. A failure here leaves
        // memory ahead of disk; the user sees the error and can retry.
        if write_needed && let Err(e) = std::fs::write(&path, &toml_text) {
            warn!(path = %path.display(), error = %e, "update_server_config: disk write failed after in-memory swap");
            self.reply_error(
                conn_id,
                correlation_id,
                format!(
                    "provider catalogs hot-swapped in memory, but writing {} failed: {e}. Retry the update or edit the file directly.",
                    path.display(),
                ),
            );
            return;
        }

        let backend_summaries: Vec<BackendSummary> = self
            .backends
            .iter()
            .map(|(name, entry)| BackendSummary {
                name: name.clone(),
                kind: entry.kind.clone(),
                default_model: entry.default_model.clone(),
                auth_mode: entry.auth_mode.clone(),
            })
            .collect();
        for tx in self.router.outbound_snapshot() {
            let _ = tx.send(ServerToClient::BackendsList {
                correlation_id: None,
                backends: backend_summaries.clone(),
            });
        }

        self.router.send_to_client(
            conn_id,
            ServerToClient::ServerConfigUpdateResult {
                correlation_id,
                cancelled_threads: cancel_list,
                restart_required_sections: restart_sections,
                pods_with_missing_backends: pods_with_missing,
            },
        );
    }

    fn reply_error(&self, conn_id: ConnId, correlation_id: Option<String>, detail: String) {
        self.router.send_to_client(
            conn_id,
            ServerToClient::Error {
                correlation_id,
                thread_id: None,
                message: format!("update_server_config: {detail}"),
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pod::config::{Auth, BackendConfig};

    fn cfg_anthropic(key: &str) -> BackendConfig {
        BackendConfig::Anthropic {
            auth: Auth::ApiKey {
                value: Some(key.into()),
                env: None,
            },
            default_model: Some("claude-sonnet-4-6".into()),
        }
    }

    fn cfg_openai_chat(url: &str) -> BackendConfig {
        BackendConfig::OpenAiChat {
            base_url: url.into(),
            auth: None,
            default_model: None,
        }
    }

    #[test]
    fn empty_to_empty_is_noop() {
        let old: BTreeMap<String, BackendConfig> = BTreeMap::new();
        let new: BTreeMap<String, BackendConfig> = BTreeMap::new();
        let diff = diff_catalog(old.iter(), &new);
        assert!(diff.is_noop());
    }

    #[test]
    fn add_only() {
        let old: BTreeMap<String, BackendConfig> = BTreeMap::new();
        let mut new = BTreeMap::new();
        new.insert("cloud".into(), cfg_anthropic("k"));
        let diff = diff_catalog(old.iter(), &new);
        assert_eq!(diff.added, vec!["cloud".to_string()]);
        assert!(diff.removed.is_empty());
        assert!(diff.changed.is_empty());
    }

    #[test]
    fn remove_only() {
        let mut old = BTreeMap::new();
        old.insert("cloud".to_string(), cfg_anthropic("k"));
        let new: BTreeMap<String, BackendConfig> = BTreeMap::new();
        let diff = diff_catalog(old.iter(), &new);
        assert_eq!(diff.removed, vec!["cloud".to_string()]);
        assert!(diff.added.is_empty());
        assert!(diff.changed.is_empty());
    }

    #[test]
    fn change_detected_by_value_inequality() {
        let mut old = BTreeMap::new();
        old.insert("cloud".to_string(), cfg_anthropic("old-key"));
        let mut new = BTreeMap::new();
        new.insert("cloud".into(), cfg_anthropic("new-key"));
        let diff = diff_catalog(old.iter(), &new);
        assert_eq!(diff.changed, vec!["cloud".to_string()]);
        assert!(diff.added.is_empty());
        assert!(diff.removed.is_empty());
    }

    #[test]
    fn unchanged_entry_is_noop() {
        let mut old = BTreeMap::new();
        old.insert("cloud".to_string(), cfg_anthropic("same-key"));
        let mut new = BTreeMap::new();
        new.insert("cloud".into(), cfg_anthropic("same-key"));
        let diff = diff_catalog(old.iter(), &new);
        assert!(diff.is_noop());
    }

    #[test]
    fn mixed_add_remove_change() {
        let mut old = BTreeMap::new();
        old.insert("a".to_string(), cfg_anthropic("k1"));
        old.insert(
            "b".to_string(),
            cfg_openai_chat("http://localhost:11434/v1"),
        );
        old.insert("c".to_string(), cfg_anthropic("k3"));

        let mut new = BTreeMap::new();
        new.insert("a".into(), cfg_anthropic("k1")); // unchanged
        new.insert("c".into(), cfg_anthropic("CHANGED")); // changed
        new.insert("d".into(), cfg_anthropic("k4")); // added
        // `b` removed

        let diff = diff_catalog(old.iter(), &new);
        assert_eq!(diff.added, vec!["d".to_string()]);
        assert_eq!(diff.removed, vec!["b".to_string()]);
        assert_eq!(diff.changed, vec!["c".to_string()]);
    }

    #[test]
    fn restart_sections_empty_when_only_backends_change() {
        let old_toml = r#"
[backends.cloud]
kind = "anthropic"
auth = { mode = "api_key", value = "k1" }
"#;
        let new_toml = r#"
[backends.cloud]
kind = "anthropic"
auth = { mode = "api_key", value = "k2" }
"#;
        let old: Config = toml::from_str(old_toml).unwrap();
        let new: Config = toml::from_str(new_toml).unwrap();
        assert!(restart_required_sections(&old, &new).is_empty());
    }

    #[test]
    fn restart_sections_flags_shared_mcp_hosts_change() {
        let old_toml = r#"
[backends.cloud]
kind = "anthropic"
auth = { mode = "api_key", value = "k1" }
"#;
        let new_toml = r#"
[backends.cloud]
kind = "anthropic"
auth = { mode = "api_key", value = "k1" }

[shared_mcp_hosts]
fetch = "http://127.0.0.1:9831/mcp"
"#;
        let old: Config = toml::from_str(old_toml).unwrap();
        let new: Config = toml::from_str(new_toml).unwrap();
        let sections = restart_required_sections(&old, &new);
        assert_eq!(sections, vec!["shared_mcp_hosts".to_string()]);
    }

    #[test]
    fn restart_sections_does_not_flag_embedding_or_rerank_changes() {
        // The whole point of these new sections is hot-swap parity with
        // backends — adding them must not push the operator into a
        // restart-required path.
        let old_toml = r#"
[backends.cloud]
kind = "anthropic"
auth = { mode = "api_key", value = "k1" }
"#;
        let new_toml = r#"
[backends.cloud]
kind = "anthropic"
auth = { mode = "api_key", value = "k1" }

[embedding_providers.local-bge]
kind = "tei"
endpoint = "http://localhost:8080"

[rerank_providers.local-rerank]
kind = "tei"
endpoint = "http://localhost:8081"
"#;
        let old: Config = toml::from_str(old_toml).unwrap();
        let new: Config = toml::from_str(new_toml).unwrap();
        assert!(restart_required_sections(&old, &new).is_empty());
    }

    #[test]
    fn diff_catalog_handles_embedding_provider_configs() {
        // Generic diff function applied to embedding provider configs —
        // exercises the diff plumbing the runtime swap path uses.
        use crate::pod::config::EmbeddingProviderConfig;
        let mut old = BTreeMap::new();
        old.insert(
            "local".to_string(),
            EmbeddingProviderConfig::Tei {
                endpoint: "http://localhost:8080".into(),
                auth: None,
            },
        );
        let mut new = BTreeMap::new();
        new.insert(
            "local".to_string(),
            EmbeddingProviderConfig::Tei {
                endpoint: "http://localhost:9090".into(), // changed
                auth: None,
            },
        );
        new.insert(
            "remote".to_string(),
            EmbeddingProviderConfig::Tei {
                endpoint: "http://embed.internal".into(),
                auth: None,
            },
        );
        let diff = diff_catalog(old.iter(), &new);
        assert_eq!(diff.added, vec!["remote".to_string()]);
        assert_eq!(diff.changed, vec!["local".to_string()]);
        assert!(diff.removed.is_empty());
    }
}
