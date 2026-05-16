//! Workspace root — acts as the default cwd for relative paths.
//!
//! Prior versions rejected absolute paths and `..` escapes to confine the agent.
//! That enforcement now lives in the sandbox layer (landlock / container bind-mounts),
//! so this module's job is just to turn a user-supplied path into an absolute
//! filesystem path: relative → joined against the root, absolute → used as-is.

use std::path::{Path, PathBuf};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum WorkspaceError {
    #[error("workspace root {path} is unreachable: {source}")]
    Canonicalize {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("workspace root {0} is not a directory")]
    NotADirectory(PathBuf),
}

#[derive(Debug)]
pub struct Workspace {
    root: PathBuf,
    /// Session-level cap on the bash tool's per-invocation timeout.
    /// `None` ⇒ no daemon-supplied default; bash falls back to its
    /// built-in 120 s. The scheduler's `ThreadContext.bash_timeout_secs`
    /// is plumbed down to the worker via `--default-bash-timeout-secs`
    /// at spawn and lands here; the bash tool reads it as both default
    /// AND ceiling, overriding any model-supplied `timeout_seconds`.
    default_bash_timeout_secs: Option<u32>,
}

impl Workspace {
    pub fn new(root: &Path) -> Result<Self, WorkspaceError> {
        let canonical = root
            .canonicalize()
            .map_err(|source| WorkspaceError::Canonicalize {
                path: root.to_path_buf(),
                source,
            })?;
        if !canonical.is_dir() {
            return Err(WorkspaceError::NotADirectory(canonical));
        }
        Ok(Self {
            root: canonical,
            default_bash_timeout_secs: None,
        })
    }

    /// Daemon-supplied bash-timeout default. Replaces the field; pass
    /// `None` to clear.
    pub fn with_default_bash_timeout_secs(mut self, secs: Option<u32>) -> Self {
        self.default_bash_timeout_secs = secs;
        self
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Session-level bash timeout if the daemon set one, else `None`.
    /// `bash_stream` consults this both for the default when the model
    /// omits `timeout_seconds` and as the ceiling when it supplies one.
    pub fn default_bash_timeout_secs(&self) -> Option<u32> {
        self.default_bash_timeout_secs
    }

    /// Absolute path → returned as-is. Relative path → joined against the
    /// workspace root. No containment check; the sandbox enforces boundaries.
    pub fn resolve(&self, path: &Path) -> Result<PathBuf, WorkspaceError> {
        if path.is_absolute() {
            Ok(path.to_path_buf())
        } else {
            Ok(self.root.join(path))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_workspace() -> (tempdir_lite::TempDir, Workspace) {
        let dir = tempdir_lite::TempDir::new();
        let ws = Workspace::new(dir.path()).unwrap();
        (dir, ws)
    }

    #[test]
    fn absolute_path_returned_as_is() {
        let (_dir, ws) = temp_workspace();
        let p = ws.resolve(Path::new("/etc/passwd")).unwrap();
        assert_eq!(p, PathBuf::from("/etc/passwd"));
    }

    #[test]
    fn relative_path_joined_against_root() {
        let (_dir, ws) = temp_workspace();
        let p = ws.resolve(Path::new("a/b/c.txt")).unwrap();
        assert!(p.starts_with(ws.root()));
        assert!(p.ends_with("a/b/c.txt"));
    }

    #[test]
    fn parent_dir_components_now_pass_through() {
        // No containment check — sandbox enforces boundaries.
        let (_dir, ws) = temp_workspace();
        let p = ws.resolve(Path::new("../sibling/file")).unwrap();
        assert!(p.to_string_lossy().contains(".."));
    }
}

// Tiny inline tempdir helper so we don't pull a dev-dep in for these few tests.
// Replace with the `tempfile` crate later if test suite grows.
#[cfg(test)]
mod tempdir_lite {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    pub struct TempDir {
        path: PathBuf,
    }
    impl TempDir {
        pub fn new() -> Self {
            let mut path = std::env::temp_dir();
            // Per-pid + per-call counter — tests in this module run in
            // parallel within the same process and would otherwise collide
            // on the directory name and race each other's Drop cleanup.
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let name = format!("wamh-test-{}-{n}", std::process::id());
            path.push(name);
            std::fs::create_dir_all(&path).unwrap();
            Self { path }
        }
        pub fn path(&self) -> &Path {
            &self.path
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}
