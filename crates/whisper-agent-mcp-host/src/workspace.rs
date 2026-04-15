//! Workspace root and path safety.
//!
//! Every filesystem path the tools accept is resolved through [`Workspace::resolve`], which
//! rejects anything that escapes the configured root. There is intentionally no symlink
//! traversal at this layer — that's a sandboxing concern for the host MCP server's eventual
//! seccomp/namespace deployment, not for path policy.

use std::path::{Component, Path, PathBuf};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum WorkspaceError {
    #[error("workspace root {0} does not exist")]
    NotFound(PathBuf),
    #[error("workspace root {0} is not a directory")]
    NotADirectory(PathBuf),
    #[error("path {0} escapes the workspace root")]
    Escapes(PathBuf),
    #[error("absolute paths are not accepted; got {0}")]
    Absolute(PathBuf),
}

#[derive(Debug)]
pub struct Workspace {
    root: PathBuf,
}

impl Workspace {
    pub fn new(root: &Path) -> Result<Self, WorkspaceError> {
        let canonical = root
            .canonicalize()
            .map_err(|_| WorkspaceError::NotFound(root.to_path_buf()))?;
        if !canonical.is_dir() {
            return Err(WorkspaceError::NotADirectory(canonical));
        }
        Ok(Self { root: canonical })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolve a workspace-relative path to an absolute path under the root.
    /// Rejects absolute paths and any `..` components that would escape the root.
    pub fn resolve(&self, path: &Path) -> Result<PathBuf, WorkspaceError> {
        if path.is_absolute() {
            return Err(WorkspaceError::Absolute(path.to_path_buf()));
        }
        let mut absolute = self.root.clone();
        for component in path.components() {
            match component {
                Component::Normal(part) => absolute.push(part),
                Component::CurDir => {}
                Component::ParentDir => {
                    if !absolute.pop() || !absolute.starts_with(&self.root) {
                        return Err(WorkspaceError::Escapes(path.to_path_buf()));
                    }
                }
                Component::Prefix(_) | Component::RootDir => {
                    return Err(WorkspaceError::Absolute(path.to_path_buf()));
                }
            }
        }
        if !absolute.starts_with(&self.root) {
            return Err(WorkspaceError::Escapes(path.to_path_buf()));
        }
        Ok(absolute)
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
    fn rejects_absolute_paths() {
        let (_dir, ws) = temp_workspace();
        assert!(ws.resolve(Path::new("/etc/passwd")).is_err());
    }

    #[test]
    fn rejects_escaping_parent() {
        let (_dir, ws) = temp_workspace();
        assert!(ws.resolve(Path::new("../escape")).is_err());
        assert!(ws.resolve(Path::new("a/../../escape")).is_err());
    }

    #[test]
    fn accepts_relative_within() {
        let (_dir, ws) = temp_workspace();
        let p = ws.resolve(Path::new("a/b/c.txt")).unwrap();
        assert!(p.starts_with(ws.root()));
    }
}

// Tiny inline tempdir helper so we don't pull a dev-dep in for these few tests.
// Replace with the `tempfile` crate later if test suite grows.
#[cfg(test)]
mod tempdir_lite {
    use std::path::{Path, PathBuf};

    pub struct TempDir {
        path: PathBuf,
    }
    impl TempDir {
        pub fn new() -> Self {
            let mut path = std::env::temp_dir();
            let name = format!("wamh-test-{}", std::process::id());
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
