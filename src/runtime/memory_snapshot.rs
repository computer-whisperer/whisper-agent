//! Build a thread's memory-index + session-context snapshot: the
//! `Role::System` block that sits between the tool manifest and the
//! first user turn.
//!
//! Two halves render together as one message:
//!
//! 1. **Session context** — date/time stamp, the bound model, and
//!    every host-env binding's `name` → `workspace_root` map. The
//!    binding name is also the tool-name prefix (`main_glob`,
//!    `main_bash`, …), so making the mapping explicit lets the
//!    model resolve "where does `main_*` actually run?" without
//!    having to discover it via `pwd`.
//! 2. **Memory** — `<pod_dir>/memory/MEMORY.md` (if present) inlined
//!    under a short preamble explaining the memory system.
//!
//! Persisted into the conversation log like any other turn — same
//! reasoning as the system prompt + tool manifest: the log is the
//! faithful record of what the model actually saw.
//!
//! Provider adapters translate `Role::System` per their wire shape
//! (OpenAI chat / Responses → `role:"system"` inline; Anthropic /
//! Gemini → `role:"user"` with content wrapped in
//! `<system-reminder>` tags). The snapshot itself stays provider-
//! agnostic.
//!
//! The block is *not* regenerated per-send: byte-stability matters
//! for prompt-cache hits across threads (fork) and across turns
//! (normal agent loop). Compaction intentionally re-snapshots (new
//! context, new date, no cache to preserve) — callers opt in via
//! `build_block` at the appropriate moment.

use std::path::Path;

use chrono::{DateTime, Utc};
use whisper_agent_protocol::{ContentBlock, Message, Role};

use crate::pod::fs::{MEMORY_DIR, MEMORY_INDEX};

/// Per-thread bits that go into the session-context section: model
/// identity and the host-env binding map. Borrowed; the caller (the
/// scheduler) owns the strings and builds this on the stack at seed
/// time.
pub struct SessionContext<'a> {
    /// Model id the thread is generating with (e.g. `"gpt-5.5"`).
    /// Empty for single-model endpoints that ignore the field — we
    /// still render the line so the model sees "Model: (default)"
    /// rather than no entry, which would be confusing.
    pub model: &'a str,
    /// Backend catalog name the model resolves through (e.g.
    /// `"openai-subscription"`). Surfaced because the same model id
    /// can route through multiple backends (subscription vs API key)
    /// and the model occasionally needs to know which.
    pub backend: &'a str,
    /// One entry per bound host-env. The binding name is also the
    /// tool prefix (`{name}_bash`, `{name}_glob`), and the
    /// workspace_root is the cwd those tools default to. Empty when
    /// the thread runs bare (no host envs); we omit the section
    /// entirely in that case.
    pub host_envs: &'a [HostEnvInfo<'a>],
}

/// One row in the host-env binding table rendered into the snapshot.
pub struct HostEnvInfo<'a> {
    pub name: &'a str,
    pub workspace_root: Option<&'a Path>,
}

/// Construct the session-context + memory-index snapshot `Message`
/// for a freshly-created thread, reading `<pod_dir>/memory/MEMORY.md`
/// at call time and stamping `now`. The returned message is
/// `Role::System` — adapters translate per wire shape at send time.
///
/// Missing memory dir, missing `MEMORY.md`, and empty `MEMORY.md` all
/// fall through to the "(no memories yet)" form: the block still
/// appears in the conversation so the agent is nudged to start
/// building memories rather than facing a silent empty state.
///
/// Synchronous blocking I/O: the file is small and read exactly once
/// per thread creation / compaction. Not worth pulling async into the
/// scheduler's sync registration path.
pub fn build_block(pod_dir: &Path, now: DateTime<Utc>, session: &SessionContext<'_>) -> Message {
    let index_contents = read_index_or_empty(pod_dir);
    let text = render_block(&index_contents, now, session);
    Message {
        role: Role::System,
        content: vec![ContentBlock::Text { text }],
    }
}

fn read_index_or_empty(pod_dir: &Path) -> Option<String> {
    let index_path = pod_dir.join(MEMORY_DIR).join(MEMORY_INDEX);
    let contents = std::fs::read_to_string(&index_path).ok()?;
    let trimmed = contents.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Render the block body. Two top-level sections — session context
/// (date, model, host envs) and memory — concatenated.
fn render_block(
    index_contents: &Option<String>,
    now: DateTime<Utc>,
    session: &SessionContext<'_>,
) -> String {
    let session_section = render_session_section(now, session);
    let memory_section = render_memory_section(index_contents);
    format!("{session_section}\n{memory_section}")
}

fn render_session_section(now: DateTime<Utc>, session: &SessionContext<'_>) -> String {
    let date = now.format("%Y-%m-%d").to_string();
    let time = now.format("%H:%M UTC").to_string();
    let model = if session.model.is_empty() {
        "(backend default)".to_string()
    } else {
        format!("`{}`", session.model)
    };

    let mut out = String::new();
    out.push_str("# Session\n\n");
    out.push_str(&format!("- Today: {date}, {time}\n"));
    out.push_str(&format!(
        "- Model: {model} (via `{}` backend)\n",
        session.backend
    ));
    if !session.host_envs.is_empty() {
        out.push_str(
            "- Host-env bindings (binding name is also the tool-name prefix; tools default \
             their cwd to the workspace root):\n",
        );
        for env in session.host_envs {
            let root = match env.workspace_root {
                Some(p) => p.display().to_string(),
                None => "(no workspace_root set)".to_string(),
            };
            out.push_str(&format!(
                "  - `{name}` → workspace `{root}` (tools prefixed `{name}_`)\n",
                name = env.name,
                root = root,
            ));
        }
    }
    out
}

fn render_memory_section(index_contents: &Option<String>) -> String {
    let index_section = match index_contents {
        Some(s) => s.as_str(),
        None => "*(no memories yet — start building them as you learn facts worth preserving)*",
    };
    format!(
        "# Memory

You have a persistent, file-based memory dir at `{memory_dir}/` — use it for context that should carry between conversations (who the user is, how they work, ongoing project state, pointers to external systems). The index below was captured at this thread's creation.

## Index

{index_section}

## Saving a memory

One file per memory, named `{{type}}_{{topic}}.md`. Frontmatter + markdown body:

```
---
name: {{short title}}
description: {{one-line purpose — used to judge relevance later}}
type: {{user | feedback | project | reference}}
---
{{body}}
```

Then add a pointer line to `{memory_dir}/{memory_index}`:

```
- [Title](filename.md) — one-line hook (≤150 chars)
```

## Types

- **user** — the user's role, expertise, preferences, how they want to collaborate.
- **feedback** — rules for how to approach work in this project. Lead with the rule, then `**Why:**` and `**How to apply:**` lines. Save both corrections (\"no, don't do X\") AND validated judgment calls (\"yes, that was right\") — confirmations are quieter but just as important.
- **project** — ongoing initiatives, decisions, deadlines, motivations not in code/git. Convert relative dates to absolute. Include `**Why:**` and `**How to apply:**`.
- **reference** — pointers to external systems (issue trackers, dashboards, team docs).

## What NOT to save

- Code patterns, file paths, architecture — re-derivable from the current repo state.
- Git history, who-changed-what — `git log` / `git blame` are authoritative.
- Debugging fix recipes — the fix is in the code, the commit message has the context.
- In-progress conversation state — that's not memory's job.

These exclusions apply even when the user asks. If asked to save a PR list or activity summary, ask what was *surprising* or *non-obvious* — that part is worth keeping.

## Using memory

- Memories go stale. A memory naming a specific function, file, or flag is a past-tense claim — verify before acting on it.
- Check the index before writing; update an existing file before creating a new one.
- If the user says \"ignore memory\" this turn, don't cite or apply remembered facts.",
        memory_dir = MEMORY_DIR,
        memory_index = MEMORY_INDEX,
        index_section = index_section,
    )
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use chrono::TimeZone;

    use super::*;

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Per-test scratch dir under `std::env::temp_dir()`. Matches the
    /// convention in `pod::behaviors::tests` — no `tempfile` dep,
    /// manual cleanup at end of each test.
    fn scratch_dir() -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "wa-memory-snapshot-test-{}-{n}",
            std::process::id()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn fixed_now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 4, 20, 14, 30, 0).unwrap()
    }

    fn empty_session() -> SessionContext<'static> {
        SessionContext {
            model: "test-model",
            backend: "test-backend",
            host_envs: &[],
        }
    }

    fn text_of(msg: &Message) -> &str {
        match &msg.content[0] {
            ContentBlock::Text { text } => text,
            _ => panic!("expected Text block"),
        }
    }

    #[test]
    fn missing_memory_dir_renders_empty_state() {
        let dir = scratch_dir();
        let msg = build_block(&dir, fixed_now(), &empty_session());
        assert_eq!(msg.role, Role::System);
        let text = text_of(&msg);
        assert!(
            text.contains("no memories yet"),
            "expected empty-state marker, got: {text}"
        );
        assert!(text.contains("2026-04-20"), "missing date stamp");
        assert!(text.contains("14:30 UTC"), "missing time stamp");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn existing_memory_index_inlines_its_contents() {
        let dir = scratch_dir();
        let memory_dir = dir.join(MEMORY_DIR);
        std::fs::create_dir_all(&memory_dir).unwrap();
        std::fs::write(
            memory_dir.join(MEMORY_INDEX),
            "- [foo](foo.md) — a memory about foo\n- [bar](bar.md) — a memory about bar\n",
        )
        .unwrap();

        let msg = build_block(&dir, fixed_now(), &empty_session());
        let text = text_of(&msg);
        assert!(text.contains("- [foo](foo.md)"), "missing index entry foo");
        assert!(text.contains("- [bar](bar.md)"), "missing index entry bar");
        assert!(
            !text.contains("no memories yet"),
            "empty-state marker should not appear when index is populated"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn whitespace_only_index_renders_as_empty() {
        // A MEMORY.md that only contains blanks / newlines shouldn't
        // inline a whitespace blob — treat as empty and nudge the
        // agent to start writing.
        let dir = scratch_dir();
        let memory_dir = dir.join(MEMORY_DIR);
        std::fs::create_dir_all(&memory_dir).unwrap();
        std::fs::write(memory_dir.join(MEMORY_INDEX), "   \n\t\n\n").unwrap();

        let msg = build_block(&dir, fixed_now(), &empty_session());
        assert!(text_of(&msg).contains("no memories yet"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn block_describes_the_memory_dir_path() {
        // The agent reads this block to know where memory files live;
        // the `memory/` path must survive into the rendered text.
        let dir = scratch_dir();
        let msg = build_block(&dir, fixed_now(), &empty_session());
        let text = text_of(&msg);
        assert!(
            text.contains("`memory/`"),
            "block must reference the memory/ directory"
        );
        assert!(
            text.contains("`memory/MEMORY.md`"),
            "block must reference the MEMORY.md index path"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn session_section_lists_model_and_backend() {
        let dir = scratch_dir();
        let session = SessionContext {
            model: "gpt-5.5",
            backend: "openai-subscription",
            host_envs: &[],
        };
        let msg = build_block(&dir, fixed_now(), &session);
        let text = text_of(&msg);
        assert!(text.contains("# Session"), "missing session header");
        assert!(text.contains("`gpt-5.5`"), "missing model");
        assert!(
            text.contains("`openai-subscription`"),
            "missing backend name"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn empty_model_renders_default_marker() {
        // Single-model endpoints (local llama.cpp etc.) pass an empty
        // model id. The line should still render so the model sees
        // *some* model identity rather than a blank slot.
        let dir = scratch_dir();
        let session = SessionContext {
            model: "",
            backend: "llamacpp-local",
            host_envs: &[],
        };
        let msg = build_block(&dir, fixed_now(), &session);
        let text = text_of(&msg);
        assert!(text.contains("(backend default)"), "missing default marker");
        assert!(text.contains("`llamacpp-local`"), "missing backend name");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn session_section_lists_host_env_bindings() {
        let dir = scratch_dir();
        let main_root = std::path::PathBuf::from("/home/christian/workspace/aetna-volume");
        let other_root = std::path::PathBuf::from("/work/other");
        let envs = [
            HostEnvInfo {
                name: "main",
                workspace_root: Some(&main_root),
            },
            HostEnvInfo {
                name: "side",
                workspace_root: Some(&other_root),
            },
        ];
        let session = SessionContext {
            model: "x",
            backend: "y",
            host_envs: &envs,
        };
        let msg = build_block(&dir, fixed_now(), &session);
        let text = text_of(&msg);
        assert!(
            text.contains("/home/christian/workspace/aetna-volume"),
            "missing main workspace path"
        );
        assert!(text.contains("/work/other"), "missing side workspace path");
        // The prefix-hint phrasing — model needs to know the binding
        // name doubles as the tool-name prefix.
        assert!(text.contains("`main_`"), "missing main_ tool-prefix hint");
        assert!(text.contains("`side_`"), "missing side_ tool-prefix hint");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn no_host_envs_omits_bindings_subsection() {
        let dir = scratch_dir();
        let msg = build_block(&dir, fixed_now(), &empty_session());
        let text = text_of(&msg);
        assert!(
            !text.contains("Host-env bindings"),
            "bindings header should not appear when no envs are bound"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_workspace_root_renders_marker() {
        // Pre-rebind threads or Inline bindings can land here with
        // `workspace_root: None`. We still want the row, with a marker
        // — silent omission would let the model assume the binding
        // doesn't exist.
        let dir = scratch_dir();
        let envs = [HostEnvInfo {
            name: "main",
            workspace_root: None,
        }];
        let session = SessionContext {
            model: "x",
            backend: "y",
            host_envs: &envs,
        };
        let msg = build_block(&dir, fixed_now(), &session);
        let text = text_of(&msg);
        assert!(text.contains("`main`"), "binding name should still render");
        assert!(
            text.contains("no workspace_root set"),
            "missing-root marker should appear"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
