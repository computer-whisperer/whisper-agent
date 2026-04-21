//! `pod_list_threads` — query this pod's threads with structured filters.

use std::path::Path;

use serde::Deserialize;
use serde_json::{Value, json};

use super::{POD_LIST_THREADS, ToolOutcome, no_update_error, no_update_text};
use crate::pod;
use crate::tools::mcp::{ToolAnnotations, ToolDescriptor as McpTool};

/// Max thread entries returned by a single `pod_list_threads` call. The
/// caller can override via `limit`; this is the ceiling past which we
/// refuse to grow even if `limit` is larger. Keeps the output bounded
/// regardless of how many threads the pod has.
const LIST_THREADS_MAX: usize = 200;

/// Default `limit` when the caller doesn't supply one — small enough to
/// be glance-readable in a single tool response.
const LIST_THREADS_DEFAULT_LIMIT: usize = 30;

/// Cap on how many thread JSONs we parse looking for filter hits before
/// giving up. Threads are scanned in newest-first order (by mtime), so
/// this bounds scan cost for pods with deep history. If we hit the cap
/// without filling `limit`, the output notes it.
const LIST_THREADS_SCAN_CAP: usize = 1000;

pub(super) fn descriptor() -> McpTool {
    McpTool {
        name: POD_LIST_THREADS.into(),
        description: "Query this pod's threads with structured filters. Returns a \
                      short summary per hit (id, state, origin, turns, last_active) \
                      — use `pod_read_file(\"threads/<id>.json\")` afterwards to pull \
                      the full conversation. Threads are scanned newest-first by \
                      modification time, so `since` + `limit` gives you \"recent \
                      runs of behavior X\". Sentinel value `\"interactive\"` in \
                      `behavior_id` filters to threads NOT spawned by any behavior \
                      (direct user interaction). Scan is capped at 1000 files — \
                      narrow with filters if the output notes the cap was hit."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "behavior_id": {
                    "type": "string",
                    "description": "Filter by the behavior that spawned the thread. \
                                    Pass `\"interactive\"` to match threads with no \
                                    behavior origin (direct user chats)."
                },
                "state": {
                    "type": "string",
                    "description": "Filter by state: `idle` | `working` | \
                                    `awaiting_approval` | `completed` | `failed` | \
                                    `cancelled`. Omit for any state."
                },
                "since": {
                    "type": "string",
                    "description": "RFC-3339 timestamp. Only return threads whose \
                                    `last_active` is at or after this time."
                },
                "min_turns": {
                    "type": "integer",
                    "description": "Minimum message count in the conversation."
                },
                "max_turns": {
                    "type": "integer",
                    "description": "Maximum message count in the conversation."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of hits. Default 30, max 200."
                }
            }
        }),
        annotations: ToolAnnotations {
            title: Some("Query pod threads".into()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        },
    }
}

#[derive(Deserialize)]
struct ListThreadsArgs {
    #[serde(default)]
    behavior_id: Option<String>,
    #[serde(default)]
    state: Option<String>,
    #[serde(default)]
    since: Option<String>,
    #[serde(default)]
    min_turns: Option<usize>,
    #[serde(default)]
    max_turns: Option<usize>,
    #[serde(default)]
    limit: Option<usize>,
}

/// Minimal projection of a thread JSON — just the fields the list-
/// threads output needs. Deserialized from the on-disk `threads/<id>.json`
/// so we don't have to round-trip through the full `Thread` type.
#[derive(Debug)]
struct ThreadRow {
    id: String,
    state: &'static str,
    origin_behavior_id: Option<String>,
    turn_count: usize,
    last_active: String,
    title: Option<String>,
}

pub(super) async fn run(pod_dir: &Path, args: Value) -> ToolOutcome {
    let parsed: ListThreadsArgs = match serde_json::from_value(args) {
        Ok(v) => v,
        Err(e) => return no_update_error(format!("invalid arguments: {e}")),
    };
    let limit = parsed
        .limit
        .unwrap_or(LIST_THREADS_DEFAULT_LIMIT)
        .min(LIST_THREADS_MAX);

    let threads_dir = pod_dir.join(pod::THREADS_DIR);
    let Ok(mut rd) = tokio::fs::read_dir(&threads_dir).await else {
        return no_update_text(format!(
            "no `{}/` directory — this pod has no threads yet\n",
            pod::THREADS_DIR
        ));
    };

    // Collect mtime-sorted candidate list first — lets us walk from
    // newest to oldest and stop once we hit the scan cap or fill `limit`.
    let mut candidates: Vec<(std::path::PathBuf, std::time::SystemTime)> = Vec::new();
    while let Ok(Some(entry)) = rd.next_entry().await {
        let name = entry.file_name();
        let Some(name_str) = name.to_str() else {
            continue;
        };
        if name_str.starts_with('.') || !name_str.ends_with(".json") {
            continue;
        }
        let Ok(meta) = entry.metadata().await else {
            continue;
        };
        let mtime = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
        candidates.push((entry.path(), mtime));
    }
    candidates.sort_by_key(|c| std::cmp::Reverse(c.1));

    let total_candidates = candidates.len();
    let mut scanned = 0usize;
    let mut hits: Vec<ThreadRow> = Vec::new();
    for (path, _) in candidates.iter().take(LIST_THREADS_SCAN_CAP) {
        scanned += 1;
        let Ok(bytes) = tokio::fs::read(path).await else {
            continue;
        };
        let Ok(row) = thread_row_from_bytes(&bytes) else {
            continue;
        };
        if !row_matches(&row, &parsed) {
            continue;
        }
        hits.push(row);
        if hits.len() >= limit {
            break;
        }
    }

    no_update_text(render_thread_rows(
        &hits,
        &parsed,
        scanned,
        total_candidates,
        limit,
    ))
}

fn thread_row_from_bytes(bytes: &[u8]) -> Result<ThreadRow, serde_json::Error> {
    let v: Value = serde_json::from_slice(bytes)?;
    let id = v
        .get("id")
        .and_then(|s| s.as_str())
        .unwrap_or("<unknown>")
        .to_string();
    let last_active = v
        .get("last_active")
        .and_then(|s| s.as_str())
        .unwrap_or("")
        .to_string();
    let title = v
        .get("title")
        .and_then(|t| t.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());
    let origin_behavior_id = v
        .get("origin")
        .and_then(|o| if o.is_null() { None } else { Some(o) })
        .and_then(|o| o.get("behavior_id"))
        .and_then(|b| b.as_str())
        .map(|s| s.to_string());
    let turn_count = v
        .get("conversation")
        .and_then(|c| c.get("messages"))
        .and_then(|m| m.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    let internal_kind = v
        .get("internal")
        .and_then(|i| i.get("kind"))
        .and_then(|k| k.as_str())
        .unwrap_or("unknown");
    let state = public_state_for(internal_kind);
    Ok(ThreadRow {
        id,
        state,
        origin_behavior_id,
        turn_count,
        last_active,
        title,
    })
}

/// Collapse the internal `Thread::ThreadInternalState` tag into the
/// public `ThreadStateLabel` names the wire uses.  Mirrors
/// `Thread::public_state` on `ThreadStateLabel` but operates on the
/// JSON string so we don't have to round-trip through the full type.
fn public_state_for(internal_kind: &str) -> &'static str {
    match internal_kind {
        "idle" => "idle",
        "completed" => "completed",
        "waiting_on_resources" | "needs_model_call" | "awaiting_model" | "awaiting_tools" => {
            "working"
        }
        "awaiting_approval" => "awaiting_approval",
        "failed" => "failed",
        "cancelled" => "cancelled",
        _ => "unknown",
    }
}

fn row_matches(row: &ThreadRow, args: &ListThreadsArgs) -> bool {
    if let Some(want_bid) = &args.behavior_id {
        if want_bid == "interactive" {
            if row.origin_behavior_id.is_some() {
                return false;
            }
        } else if row.origin_behavior_id.as_deref() != Some(want_bid.as_str()) {
            return false;
        }
    }
    if let Some(want_state) = &args.state
        && row.state != want_state.as_str()
    {
        return false;
    }
    if let Some(since) = &args.since
        && row.last_active.as_str() < since.as_str()
    {
        // RFC-3339 strings are lexicographically comparable when the
        // offset is normalized to a consistent form. Our writes use
        // `Utc.to_rfc3339()` so it's always `...Z`. Comparing strings
        // avoids pulling chrono into the tool layer.
        return false;
    }
    if let Some(min) = args.min_turns
        && row.turn_count < min
    {
        return false;
    }
    if let Some(max) = args.max_turns
        && row.turn_count > max
    {
        return false;
    }
    true
}

fn render_thread_rows(
    hits: &[ThreadRow],
    args: &ListThreadsArgs,
    scanned: usize,
    total_candidates: usize,
    limit: usize,
) -> String {
    let mut out = String::new();
    if hits.is_empty() {
        out.push_str(&format!(
            "no threads match (scanned {scanned} of {total_candidates})\n"
        ));
        return out;
    }
    // Column widths — align around the longest field actually present.
    let state_w = hits.iter().map(|r| r.state.len()).max().unwrap_or(1).max(5);
    let origin_w = hits
        .iter()
        .map(|r| {
            r.origin_behavior_id
                .as_deref()
                .unwrap_or("interactive")
                .len()
        })
        .max()
        .unwrap_or(1)
        .max(6);
    let id_w = hits.iter().map(|r| r.id.len()).max().unwrap_or(1).max(2);
    for row in hits {
        let origin = row.origin_behavior_id.as_deref().unwrap_or("interactive");
        let title_suffix = row
            .title
            .as_deref()
            .map(|t| format!("  | {t}"))
            .unwrap_or_default();
        out.push_str(&format!(
            "[{state:<state_w$}] turns={turns:>4}  last={last}  origin={origin:<origin_w$}  id={id:<id_w$}{title_suffix}\n",
            state = row.state,
            state_w = state_w,
            turns = row.turn_count,
            last = if row.last_active.is_empty() {
                "-"
            } else {
                &row.last_active
            },
            origin = origin,
            origin_w = origin_w,
            id = row.id,
            id_w = id_w,
            title_suffix = title_suffix,
        ));
    }
    let cap_hit = scanned >= LIST_THREADS_SCAN_CAP && hits.len() < limit;
    out.push_str(&format!(
        "\n{} hit{} (scanned {} of {}{})\n",
        hits.len(),
        if hits.len() == 1 { "" } else { "s" },
        scanned,
        total_candidates,
        if cap_hit { "; scan cap reached" } else { "" }
    ));
    if cap_hit {
        out.push_str(
            "[scan cap reached before `limit` filled — narrow with `since` / \
             `behavior_id` / `state` to see older matches]\n",
        );
    }
    if !any_filter_set(args) {
        out.push_str(
            "[no filters supplied; use `behavior_id` / `state` / `since` / \
             `min_turns` to narrow]\n",
        );
    }
    out
}

fn any_filter_set(args: &ListThreadsArgs) -> bool {
    args.behavior_id.is_some()
        || args.state.is_some()
        || args.since.is_some()
        || args.min_turns.is_some()
        || args.max_turns.is_some()
}

#[cfg(test)]
mod tests {
    use super::super::{dispatch, join_blocks, sample_config, temp_dir};
    use super::*;

    /// Build a minimal thread JSON blob matching the `Thread` on-disk
    /// shape that `pod_list_threads` parses. Only the fields the list
    /// reads are populated; unknown fields are ignored by serde.
    fn write_thread_file(
        dir: &Path,
        id: &str,
        last_active: &str,
        internal_kind: &str,
        origin_behavior_id: Option<&str>,
        message_count: usize,
        title: Option<&str>,
    ) {
        let messages: Vec<Value> = (0..message_count)
            .map(
                |i| json!({ "role": if i % 2 == 0 { "user" } else { "assistant" }, "content": [] }),
            )
            .collect();
        let origin = origin_behavior_id.map(|bid| {
            json!({
                "behavior_id": bid,
                "fired_at": last_active,
                "trigger_payload": null
            })
        });
        let mut body = json!({
            "id": id,
            "pod_id": "p",
            "created_at": last_active,
            "last_active": last_active,
            "title": title,
            "config": {},
            "conversation": { "messages": messages },
            "total_usage": {},
            "internal": { "kind": internal_kind }
        });
        if let Some(o) = origin {
            body["origin"] = o;
        }
        std::fs::write(
            dir.join("threads").join(format!("{id}.json")),
            serde_json::to_vec_pretty(&body).unwrap(),
        )
        .unwrap();
    }

    fn seed_thread_dir(dir: &Path) {
        std::fs::create_dir_all(dir.join("threads")).unwrap();
    }

    #[tokio::test]
    async fn list_threads_filters_by_behavior_id() {
        let dir = temp_dir();
        let cfg = sample_config();
        seed_thread_dir(&dir);
        write_thread_file(
            &dir,
            "a",
            "2026-04-18T10:00:00Z",
            "completed",
            Some("daily"),
            4,
            None,
        );
        write_thread_file(
            &dir,
            "b",
            "2026-04-18T11:00:00Z",
            "completed",
            Some("webhook_x"),
            2,
            None,
        );
        write_thread_file(
            &dir,
            "c",
            "2026-04-18T12:00:00Z",
            "completed",
            None, // interactive
            6,
            None,
        );

        let out = dispatch(
            dir.clone(),
            cfg.clone(),
            vec![],
            crate::permission::PodModifyCap::ModifyAllow,
            POD_LIST_THREADS,
            json!({ "behavior_id": "daily" }),
        )
        .await;
        assert!(!out.result.is_error);
        let text = join_blocks(&out.result.content);
        assert!(text.contains("id=a"), "missing daily thread: {text}");
        assert!(!text.contains("id=b"), "b should be filtered out: {text}");
        assert!(!text.contains("id=c"), "c should be filtered out: {text}");
        assert!(text.contains("1 hit"), "wrong count: {text}");

        // Interactive sentinel filters to origin-less threads.
        let out = dispatch(
            dir.clone(),
            cfg,
            vec![],
            crate::permission::PodModifyCap::ModifyAllow,
            POD_LIST_THREADS,
            json!({ "behavior_id": "interactive" }),
        )
        .await;
        assert!(!out.result.is_error);
        let text = join_blocks(&out.result.content);
        assert!(text.contains("id=c"), "missing interactive thread: {text}");
        assert!(!text.contains("id=a") && !text.contains("id=b"));
    }

    #[tokio::test]
    async fn list_threads_filters_by_state_and_turns() {
        let dir = temp_dir();
        let cfg = sample_config();
        seed_thread_dir(&dir);
        write_thread_file(
            &dir,
            "alpha",
            "2026-04-18T10:00:00Z",
            "failed",
            None,
            3,
            None,
        );
        write_thread_file(
            &dir,
            "beta",
            "2026-04-18T11:00:00Z",
            "completed",
            None,
            10,
            None,
        );
        write_thread_file(
            &dir,
            "gamma",
            "2026-04-18T12:00:00Z",
            "awaiting_tools",
            None,
            2,
            None,
        );

        let out = dispatch(
            dir.clone(),
            cfg.clone(),
            vec![],
            crate::permission::PodModifyCap::ModifyAllow,
            POD_LIST_THREADS,
            json!({ "state": "failed" }),
        )
        .await;
        assert!(!out.result.is_error);
        let text = join_blocks(&out.result.content);
        assert!(text.contains("id=alpha"), "expected alpha: {text}");
        assert!(!text.contains("id=beta") && !text.contains("id=gamma"));

        // awaiting_tools is internal-only; the public label is `working`.
        let out = dispatch(
            dir.clone(),
            cfg.clone(),
            vec![],
            crate::permission::PodModifyCap::ModifyAllow,
            POD_LIST_THREADS,
            json!({ "state": "working" }),
        )
        .await;
        assert!(!out.result.is_error);
        let text = join_blocks(&out.result.content);
        assert!(
            text.contains("id=gamma"),
            "working should match gamma: {text}"
        );

        // Turn-count windowing.
        let out = dispatch(
            dir.clone(),
            cfg,
            vec![],
            crate::permission::PodModifyCap::ModifyAllow,
            POD_LIST_THREADS,
            json!({ "min_turns": 5 }),
        )
        .await;
        assert!(!out.result.is_error);
        let text = join_blocks(&out.result.content);
        assert!(text.contains("id=beta"));
        assert!(!text.contains("id=alpha") && !text.contains("id=gamma"));
    }

    #[tokio::test]
    async fn list_threads_filters_by_since() {
        let dir = temp_dir();
        let cfg = sample_config();
        seed_thread_dir(&dir);
        write_thread_file(
            &dir,
            "old",
            "2026-04-17T12:00:00Z",
            "completed",
            None,
            1,
            None,
        );
        write_thread_file(
            &dir,
            "new",
            "2026-04-18T12:00:00Z",
            "completed",
            None,
            1,
            None,
        );

        let out = dispatch(
            dir.clone(),
            cfg,
            vec![],
            crate::permission::PodModifyCap::ModifyAllow,
            POD_LIST_THREADS,
            json!({ "since": "2026-04-18T00:00:00Z" }),
        )
        .await;
        assert!(!out.result.is_error);
        let text = join_blocks(&out.result.content);
        assert!(text.contains("id=new"));
        assert!(!text.contains("id=old"));
    }

    #[tokio::test]
    async fn list_threads_respects_limit() {
        let dir = temp_dir();
        let cfg = sample_config();
        seed_thread_dir(&dir);
        for i in 0..10 {
            write_thread_file(
                &dir,
                &format!("t{i:02}"),
                &format!("2026-04-18T10:{:02}:00Z", i),
                "completed",
                None,
                1,
                None,
            );
        }
        let out = dispatch(
            dir.clone(),
            cfg,
            vec![],
            crate::permission::PodModifyCap::ModifyAllow,
            POD_LIST_THREADS,
            json!({ "limit": 3 }),
        )
        .await;
        assert!(!out.result.is_error);
        let text = join_blocks(&out.result.content);
        let id_lines = text.lines().filter(|l| l.contains("id=t")).count();
        assert_eq!(id_lines, 3, "expected 3 hit rows, got: {text}");
        assert!(text.contains("3 hits"), "summary wrong: {text}");
    }

    #[tokio::test]
    async fn list_threads_empty_directory_gives_helpful_message() {
        let dir = temp_dir();
        let cfg = sample_config();
        let out = dispatch(
            dir.clone(),
            cfg,
            vec![],
            crate::permission::PodModifyCap::ModifyAllow,
            POD_LIST_THREADS,
            json!({}),
        )
        .await;
        assert!(!out.result.is_error);
        let text = join_blocks(&out.result.content);
        assert!(
            text.contains("no `threads/` directory"),
            "expected empty message: {text}"
        );
    }
}
