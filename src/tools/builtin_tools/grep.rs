//! `pod_grep` — literal substring search across a pod's files.

use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::{Value, json};

use super::{POD_GREP, ToolOutcome, no_update_error, no_update_text};
use crate::tools::mcp::{ToolAnnotations, ToolDescriptor as McpTool};

/// Match cap — beyond this, output is truncated with a "narrow `path`" hint.
/// Keeps the tool response bounded regardless of pattern or pod size.
const GREP_MATCH_CAP: usize = 200;

/// Max caller-supplied context window radius.
const GREP_CONTEXT_MAX: usize = 20;

pub(super) fn descriptor() -> McpTool {
    McpTool {
        name: POD_GREP.into(),
        description: "Search this pod's files for a literal substring — use this to \
                      find which thread logged a particular tool name or error \
                      message, which behaviors mention a keyword, etc. The whole \
                      pod tree is searched by default (`behaviors/`, `threads/`, \
                      `pod.toml`, prompts, state files); pass `path` to narrow to \
                      a file or subdirectory. Dotfiles and `.archived/` are always \
                      skipped. Pattern matching is literal — no regex — so a \
                      period or bracket in `pattern` matches itself. Output is \
                      `path:line: <content>` per match (`-` separator instead of \
                      `:` for `context` lines). Capped at 200 matches — narrow \
                      `path` if your first call hits the cap."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Literal substring to search for. Non-empty."
                },
                "path": {
                    "type": "string",
                    "description": "Relative to the pod directory. May be a file \
                                    (`threads/abc.json`) or a directory \
                                    (`threads/`, `behaviors/daily/`). Omit to \
                                    search the whole pod."
                },
                "context": {
                    "type": "integer",
                    "description": "Number of context lines before and after each \
                                    match (like grep's -C). Default 0; max 20."
                }
            },
            "required": ["pattern"]
        }),
        annotations: ToolAnnotations {
            title: Some("Grep pod files".into()),
            read_only_hint: Some(true),
            destructive_hint: Some(false),
            idempotent_hint: Some(true),
            open_world_hint: Some(false),
        },
    }
}

#[derive(Deserialize)]
struct GrepArgs {
    pattern: String,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    context: Option<usize>,
}

pub(super) async fn run(pod_dir: &Path, args: Value) -> ToolOutcome {
    let parsed: GrepArgs = match serde_json::from_value(args) {
        Ok(v) => v,
        Err(e) => return no_update_error(format!("invalid arguments: {e}")),
    };
    if parsed.pattern.is_empty() {
        return no_update_error("pod_grep: `pattern` must be non-empty".into());
    }
    let context = parsed.context.unwrap_or(0).min(GREP_CONTEXT_MAX);
    let search_root = match resolve_grep_target(pod_dir, parsed.path.as_deref()) {
        Ok(p) => p,
        Err(e) => return no_update_error(e),
    };

    let files = collect_grep_files(&search_root).await;
    let mut out = String::new();
    let mut total_matches = 0usize;
    let mut files_with_hits = 0usize;
    let mut first_file = true;

    'outer: for file_path in &files {
        let rel = file_path
            .strip_prefix(pod_dir)
            .unwrap_or(file_path)
            .to_string_lossy()
            .replace('\\', "/");
        let content = match tokio::fs::read(file_path).await {
            Ok(b) => b,
            Err(_) => continue,
        };
        let Ok(text) = std::str::from_utf8(&content) else {
            // Non-UTF8 files are rare here (configs + JSON + markdown) but
            // defensive — skip rather than crash the tool call.
            continue;
        };
        let lines: Vec<&str> = text.lines().collect();
        let match_lines: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter(|(_, l)| l.contains(&parsed.pattern))
            .map(|(i, _)| i)
            .collect();
        if match_lines.is_empty() {
            continue;
        }
        files_with_hits += 1;
        if !first_file {
            out.push_str("--\n");
        }
        first_file = false;

        // Collapse per-match windows [m-context, m+context] into non-
        // overlapping ranges so neighboring matches don't double-print.
        let windows: Vec<(usize, usize)> = merge_windows(&match_lines, context, lines.len());
        for (range_idx, (start, end)) in windows.iter().enumerate() {
            if range_idx > 0 {
                out.push_str("--\n");
            }
            for (offset, line) in lines[*start..=*end].iter().enumerate() {
                let i = *start + offset;
                let is_match = match_lines.binary_search(&i).is_ok();
                let sep = if is_match { ':' } else { '-' };
                out.push_str(&format!("{rel}:{}{sep} {}\n", i + 1, line));
                if is_match {
                    total_matches += 1;
                    if total_matches >= GREP_MATCH_CAP {
                        out.push_str(&format!(
                            "\n[match cap reached: {GREP_MATCH_CAP} matches shown; narrow with `path` to see more]\n"
                        ));
                        break 'outer;
                    }
                }
            }
        }
    }

    if total_matches == 0 {
        return no_update_text(format!(
            "no matches for `{}` under `{}`\n",
            parsed.pattern,
            parsed.path.as_deref().unwrap_or(".")
        ));
    }
    let summary = format!(
        "\n{total_matches} match{} across {files_with_hits} file{}\n",
        if total_matches == 1 { "" } else { "es" },
        if files_with_hits == 1 { "" } else { "s" }
    );
    out.push_str(&summary);
    no_update_text(out)
}

/// Resolve the caller-supplied `path` against `pod_dir`. `None` → whole
/// pod. Rejects absolute paths and anything climbing out via `..`.
fn resolve_grep_target(pod_dir: &Path, path: Option<&str>) -> Result<PathBuf, String> {
    let Some(rel) = path else {
        return Ok(pod_dir.to_path_buf());
    };
    if rel.is_empty() {
        return Ok(pod_dir.to_path_buf());
    }
    let as_path = Path::new(rel);
    if as_path.is_absolute() {
        return Err(format!(
            "pod_grep: `path` must be relative to the pod dir (got `{rel}`)"
        ));
    }
    for component in as_path.components() {
        match component {
            std::path::Component::Normal(_) => {}
            _ => {
                return Err(format!(
                    "pod_grep: `path` may only contain normal components (got `{rel}`)"
                ));
            }
        }
    }
    Ok(pod_dir.join(as_path))
}

/// Breadth-first walk from `root`, skipping dotfiles / dotdirs and the
/// `.archived/` subtree. Returns files sorted for a stable output
/// across identical calls. If `root` is a file, returns just that file.
async fn collect_grep_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let meta = match tokio::fs::metadata(root).await {
        Ok(m) => m,
        Err(_) => return out,
    };
    if meta.is_file() {
        out.push(root.to_path_buf());
        return out;
    }
    let mut stack: Vec<PathBuf> = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let mut rd = match tokio::fs::read_dir(&dir).await {
            Ok(r) => r,
            Err(_) => continue,
        };
        while let Ok(Some(entry)) = rd.next_entry().await {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with('.') {
                continue;
            }
            let path = entry.path();
            match entry.file_type().await {
                Ok(ft) if ft.is_dir() => stack.push(path),
                Ok(ft) if ft.is_file() => out.push(path),
                _ => {}
            }
        }
    }
    out.sort();
    out
}

/// Merge per-match windows of radius `context` around each line in
/// `match_lines` into a list of non-overlapping `(start, end)` ranges
/// (both inclusive), clipped to `[0, total_lines)`. `match_lines`
/// is assumed sorted.
fn merge_windows(match_lines: &[usize], context: usize, total_lines: usize) -> Vec<(usize, usize)> {
    if total_lines == 0 {
        return Vec::new();
    }
    let mut windows: Vec<(usize, usize)> = Vec::new();
    for &m in match_lines {
        let start = m.saturating_sub(context);
        let end = (m + context).min(total_lines.saturating_sub(1));
        match windows.last_mut() {
            Some((_, last_end)) if start <= last_end.saturating_add(1) => {
                if end > *last_end {
                    *last_end = end;
                }
            }
            _ => windows.push((start, end)),
        }
    }
    windows
}

#[cfg(test)]
mod tests {
    use super::super::{dispatch, join_blocks, sample_config, temp_dir};
    use super::*;
    use crate::pod;

    #[tokio::test]
    async fn grep_matches_across_files() {
        let dir = temp_dir();
        let cfg = sample_config();
        tokio::fs::write(dir.join("pod.toml"), pod::to_toml(&cfg).unwrap())
            .await
            .unwrap();
        tokio::fs::create_dir_all(dir.join("threads"))
            .await
            .unwrap();
        tokio::fs::write(
            dir.join("threads").join("a.json"),
            "alpha line\nsecond error here\n",
        )
        .await
        .unwrap();
        tokio::fs::write(
            dir.join("threads").join("b.json"),
            "no hits here\njust text\n",
        )
        .await
        .unwrap();

        let out = dispatch(
            dir.clone(),
            cfg,
            vec![],
            POD_GREP,
            json!({ "pattern": "error" }),
        )
        .await;
        assert!(!out.result.is_error, "grep errored: {:?}", out.result);
        let text = join_blocks(&out.result.content);
        assert!(
            text.contains("threads/a.json:2:"),
            "missing a.json match: {text}"
        );
        assert!(
            !text.contains("threads/b.json:"),
            "b.json should not appear: {text}"
        );
        assert!(text.contains("1 match"), "missing summary: {text}");
    }

    #[tokio::test]
    async fn grep_emits_context_lines() {
        let dir = temp_dir();
        let cfg = sample_config();
        tokio::fs::write(
            dir.join("system_prompt.md"),
            "line1\nline2\ntarget\nline4\nline5\n",
        )
        .await
        .unwrap();

        let out = dispatch(
            dir.clone(),
            cfg,
            vec![],
            POD_GREP,
            json!({ "pattern": "target", "context": 1 }),
        )
        .await;
        assert!(!out.result.is_error);
        let text = join_blocks(&out.result.content);
        // Context lines use `-`, match line uses `:`.
        assert!(
            text.contains("system_prompt.md:2- line2"),
            "missing context-before: {text}"
        );
        assert!(
            text.contains("system_prompt.md:3: target"),
            "missing match line: {text}"
        );
        assert!(
            text.contains("system_prompt.md:4- line4"),
            "missing context-after: {text}"
        );
    }

    #[tokio::test]
    async fn grep_narrows_to_path() {
        let dir = temp_dir();
        let cfg = sample_config();
        tokio::fs::write(dir.join("pod.toml"), "name = \"target\"")
            .await
            .unwrap();
        tokio::fs::create_dir_all(dir.join("threads"))
            .await
            .unwrap();
        tokio::fs::write(dir.join("threads").join("a.json"), "has target here")
            .await
            .unwrap();

        // Scope to threads/ — pod.toml match must NOT appear.
        let out = dispatch(
            dir.clone(),
            cfg,
            vec![],
            POD_GREP,
            json!({ "pattern": "target", "path": "threads/" }),
        )
        .await;
        assert!(!out.result.is_error);
        let text = join_blocks(&out.result.content);
        assert!(
            text.contains("threads/a.json:"),
            "missing threads hit: {text}"
        );
        assert!(
            !text.contains("pod.toml:"),
            "pod.toml should be out of scope: {text}"
        );
    }

    #[tokio::test]
    async fn grep_rejects_path_traversal() {
        let dir = temp_dir();
        let cfg = sample_config();
        let out = dispatch(
            dir.clone(),
            cfg,
            vec![],
            POD_GREP,
            json!({ "pattern": "x", "path": "../etc" }),
        )
        .await;
        assert!(out.result.is_error);
        let text = join_blocks(&out.result.content);
        assert!(
            text.contains("only contain normal components"),
            "wrong error: {text}"
        );
    }

    #[tokio::test]
    async fn grep_skips_archived_and_dotfiles() {
        let dir = temp_dir();
        let cfg = sample_config();
        tokio::fs::create_dir_all(dir.join(".archived").join("threads"))
            .await
            .unwrap();
        tokio::fs::write(
            dir.join(".archived").join("threads").join("old.json"),
            "has match\n",
        )
        .await
        .unwrap();
        tokio::fs::write(dir.join(".hidden"), "has match\n")
            .await
            .unwrap();
        tokio::fs::write(dir.join("visible.md"), "has match\n")
            .await
            .unwrap();

        let out = dispatch(
            dir.clone(),
            cfg,
            vec![],
            POD_GREP,
            json!({ "pattern": "has match" }),
        )
        .await;
        assert!(!out.result.is_error);
        let text = join_blocks(&out.result.content);
        assert!(text.contains("visible.md:"), "missing visible.md: {text}");
        assert!(
            !text.contains(".archived") && !text.contains(".hidden"),
            "dotfiles leaked: {text}"
        );
    }

    #[tokio::test]
    async fn grep_reports_no_matches() {
        let dir = temp_dir();
        let cfg = sample_config();
        tokio::fs::write(dir.join("pod.toml"), "name = \"ok\"")
            .await
            .unwrap();
        let out = dispatch(
            dir.clone(),
            cfg,
            vec![],
            POD_GREP,
            json!({ "pattern": "nowhere-in-sight" }),
        )
        .await;
        assert!(!out.result.is_error);
        let text = join_blocks(&out.result.content);
        assert!(text.contains("no matches"), "wrong empty output: {text}");
    }

    #[tokio::test]
    async fn grep_caps_total_matches() {
        let dir = temp_dir();
        let cfg = sample_config();
        // Generate a file with GREP_MATCH_CAP + 50 matches on distinct lines.
        let mut body = String::new();
        for _ in 0..GREP_MATCH_CAP + 50 {
            body.push_str("hit\n");
        }
        tokio::fs::write(dir.join("big.txt"), body).await.unwrap();
        let out = dispatch(
            dir.clone(),
            cfg,
            vec![],
            POD_GREP,
            json!({ "pattern": "hit" }),
        )
        .await;
        assert!(!out.result.is_error);
        let text = join_blocks(&out.result.content);
        assert!(
            text.contains("match cap reached"),
            "missing cap notice: {text}"
        );
    }

    #[test]
    fn merge_windows_collapses_overlapping() {
        // Two matches at lines 5 and 7 with context=2 → windows overlap,
        // should merge to a single [3..9] range.
        let merged = merge_windows(&[5, 7], 2, 20);
        assert_eq!(merged, vec![(3, 9)]);
        // Distant matches stay separate.
        let merged = merge_windows(&[5, 50], 2, 100);
        assert_eq!(merged, vec![(3, 7), (48, 52)]);
        // Clamping near start / end.
        let merged = merge_windows(&[0, 99], 2, 100);
        assert_eq!(merged, vec![(0, 2), (97, 99)]);
    }
}
