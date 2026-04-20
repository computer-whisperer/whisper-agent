//! Build a thread's memory-index snapshot: the `Role::System` block
//! that sits between the tool manifest and the first user turn.
//!
//! Reads `<pod_dir>/memory/MEMORY.md` (if present) and inlines its
//! contents under a short preamble explaining the memory system and
//! stamping the thread's creation time. The resulting [`Message`] is
//! persisted into the conversation log like any other turn — same
//! logic as why we persist the system prompt + tool manifest: the
//! log is the faithful record of what the model actually saw.
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

/// Construct the memory-index snapshot `Message` for a freshly-created
/// thread, reading `<pod_dir>/memory/MEMORY.md` at call time and
/// stamping `now`. The returned message is `Role::System` — adapters
/// translate per wire shape at send time.
///
/// Missing memory dir, missing `MEMORY.md`, and empty `MEMORY.md` all
/// fall through to the "(no memories yet)" form: the block still
/// appears in the conversation so the agent is nudged to start
/// building memories rather than facing a silent empty state.
///
/// Synchronous blocking I/O: the file is small and read exactly once
/// per thread creation / compaction. Not worth pulling async into the
/// scheduler's sync registration path.
pub fn build_block(pod_dir: &Path, now: DateTime<Utc>) -> Message {
    let index_contents = read_index_or_empty(pod_dir);
    let text = render_block(&index_contents, now);
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

/// Render the block body. `None` index → the empty-state substitution;
/// `Some(s)` → the file contents (already trimmed). `now` is rendered
/// as UTC date + HH:MM so the agent has a concrete timestamp stamped
/// at thread creation.
fn render_block(index_contents: &Option<String>, now: DateTime<Utc>) -> String {
    let date = now.format("%Y-%m-%d").to_string();
    let time = now.format("%H:%M UTC").to_string();
    let index_section = match index_contents {
        Some(s) => s.as_str(),
        None => "*(no memories yet — start building them as you learn facts worth preserving)*",
    };
    format!(
        "# Memory

You have a persistent, file-based memory dir at `{memory_dir}/` — use it for context that should carry between conversations (who the user is, how they work, ongoing project state, pointers to external systems). The index below was captured at this thread's creation.

Today: {date}, {time}

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
        date = date,
        time = time,
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

    fn text_of(msg: &Message) -> &str {
        match &msg.content[0] {
            ContentBlock::Text { text } => text,
            _ => panic!("expected Text block"),
        }
    }

    #[test]
    fn missing_memory_dir_renders_empty_state() {
        let dir = scratch_dir();
        let msg = build_block(&dir, fixed_now());
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

        let msg = build_block(&dir, fixed_now());
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

        let msg = build_block(&dir, fixed_now());
        assert!(text_of(&msg).contains("no memories yet"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn block_describes_the_memory_dir_path() {
        // The agent reads this block to know where memory files live;
        // the `memory/` path must survive into the rendered text.
        let dir = scratch_dir();
        let msg = build_block(&dir, fixed_now());
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
}
