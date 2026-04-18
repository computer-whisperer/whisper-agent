//! Retention sweep helpers — move terminal threads to `<pod>/.archived/`
//! or delete them outright, per the owning behavior's `on_completion`
//! policy. Pure file-IO functions live here; policy (which action to
//! take for a given thread) is decided in `Scheduler::retention_sweep`.

/// Disposition the retention sweep picks for a single thread.
/// Archive preserves the JSON under `<pod>/.archived/threads/`;
/// Delete removes it outright. Which applies is determined by the
/// owning behavior's `on_completion.retention` policy.
pub(super) enum RetentionAction {
    Archive,
    Delete,
}

/// Move `<pod>/threads/<tid>.json` to `<pod>/.archived/threads/<tid>.json`.
/// Creates the archive subdirectory if missing. Idempotent on already-
/// archived threads — `rename` of a nonexistent source errors, which
/// we surface so the sweep caller can log and move on.
pub(super) async fn archive_thread_json(
    pod_dir: &std::path::Path,
    thread_id: &str,
) -> anyhow::Result<()> {
    let src = pod_dir
        .join(crate::pod::THREADS_DIR)
        .join(format!("{thread_id}.json"));
    let dst_dir = pod_dir.join(".archived").join(crate::pod::THREADS_DIR);
    tokio::fs::create_dir_all(&dst_dir).await?;
    let dst = dst_dir.join(format!("{thread_id}.json"));
    tokio::fs::rename(&src, &dst).await?;
    Ok(())
}

/// Remove `<pod>/threads/<tid>.json` outright. Idempotent for a
/// missing file (returns Ok) so a racey double-delete doesn't error.
pub(super) async fn delete_thread_json(
    pod_dir: &std::path::Path,
    thread_id: &str,
) -> anyhow::Result<()> {
    let path = pod_dir
        .join(crate::pod::THREADS_DIR)
        .join(format!("{thread_id}.json"));
    match tokio::fs::remove_file(&path).await {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn archive_moves_thread_json_into_archived_subdir() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pod_dir =
            std::env::temp_dir().join(format!("wa-archive-test-{}-{n}", std::process::id()));
        let threads_dir = pod_dir.join(crate::pod::THREADS_DIR);
        std::fs::create_dir_all(&threads_dir).unwrap();
        let src = threads_dir.join("t-1.json");
        std::fs::write(&src, b"{}").unwrap();
        archive_thread_json(&pod_dir, "t-1").await.unwrap();
        assert!(!src.exists());
        assert!(
            pod_dir
                .join(".archived")
                .join(crate::pod::THREADS_DIR)
                .join("t-1.json")
                .exists()
        );
        let _ = std::fs::remove_dir_all(&pod_dir);
    }

    #[tokio::test]
    async fn delete_removes_thread_json_and_tolerates_missing() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pod_dir =
            std::env::temp_dir().join(format!("wa-delete-test-{}-{n}", std::process::id()));
        let threads_dir = pod_dir.join(crate::pod::THREADS_DIR);
        std::fs::create_dir_all(&threads_dir).unwrap();
        let path = threads_dir.join("t-2.json");
        std::fs::write(&path, b"{}").unwrap();
        delete_thread_json(&pod_dir, "t-2").await.unwrap();
        assert!(!path.exists());
        // Idempotent: second call succeeds on a missing file.
        delete_thread_json(&pod_dir, "t-2").await.unwrap();
        let _ = std::fs::remove_dir_all(&pod_dir);
    }
}
