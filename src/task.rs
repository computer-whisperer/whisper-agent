//! Task — the unit of long-lived agent work.
//!
//! A task owns one conversation, runs against one provider + one MCP host, and is
//! observable (and cancellable) independently of any client connection. Tasks-as-data:
//! the struct is serializable, snapshottable, and mutation goes through a single owner
//! (the [`TaskManager`]'s per-task runner, today; the central scheduler in a future
//! commit).
//!
//! The internal `TaskState` here matches [`TaskStateLabel`] one-to-one for now. The
//! finer internal states from `design_task_scheduler.md` (`AwaitingModel`/
//! `AwaitingTools`, `started_at`, in-flight op ids) come with the explicit-scheduler
//! rework.
//!
//! [`TaskManager`]: crate::task_manager::TaskManager

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use whisper_agent_protocol::{
    Conversation, TaskConfig, TaskSnapshot, TaskStateLabel, TaskSummary, Usage,
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Task {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub last_active: DateTime<Utc>,
    pub title: Option<String>,
    pub config: TaskConfig,
    pub state: TaskStateLabel,
    pub conversation: Conversation,
    pub total_usage: Usage,
    /// True once the user has archived the task. Archived tasks remain loaded so their
    /// conversation is still readable, but they drop off the `ListTasks` broadcast list.
    #[serde(default)]
    pub archived: bool,
}

impl Task {
    pub fn new(id: String, config: TaskConfig) -> Self {
        let now = Utc::now();
        Self {
            id,
            created_at: now,
            last_active: now,
            title: None,
            config,
            state: TaskStateLabel::Idle,
            conversation: Conversation::new(),
            total_usage: Usage::default(),
            archived: false,
        }
    }

    pub fn touch(&mut self) {
        self.last_active = Utc::now();
    }

    pub fn summary(&self) -> TaskSummary {
        TaskSummary {
            task_id: self.id.clone(),
            title: self.title.clone(),
            state: self.state,
            created_at: self.created_at.to_rfc3339(),
            last_active: self.last_active.to_rfc3339(),
        }
    }

    pub fn snapshot(&self) -> TaskSnapshot {
        TaskSnapshot {
            task_id: self.id.clone(),
            title: self.title.clone(),
            config: self.config.clone(),
            state: self.state,
            conversation: self.conversation.clone(),
            total_usage: self.total_usage,
            created_at: self.created_at.to_rfc3339(),
            last_active: self.last_active.to_rfc3339(),
        }
    }
}

/// Derive a title from the user's initial message: trim, collapse internal whitespace,
/// truncate to ~50 chars (rounded to a char boundary) with a trailing ellipsis.
pub fn derive_title(initial_message: &str) -> String {
    let collapsed: String = initial_message.split_whitespace().collect::<Vec<_>>().join(" ");
    const MAX: usize = 50;
    if collapsed.chars().count() <= MAX {
        collapsed
    } else {
        let mut out: String = collapsed.chars().take(MAX).collect();
        out.push('…');
        out
    }
}

pub fn new_task_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("task-{:016x}", nanos as u64)
}
