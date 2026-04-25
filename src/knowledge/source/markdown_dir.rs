//! [`MarkdownDir`] — source adapter that walks a directory tree and emits
//! one [`SourceRecord`] per `.md` file.
//!
//! The simplest adapter, intended for workspace-style buckets (project
//! notes, design docs, agent memory) and as the first end-to-end test
//! target while the bucket pipeline is being brought up.
//!
//! Globs (`include_globs` / `exclude_globs` from `bucket.toml`) are
//! deliberately ignored in this adapter — `MarkdownDir` is `.md`-extension
//! by definition. A future `FileGlob` adapter handles the general case.

use std::fs;
use std::path::{Path, PathBuf};

use super::{SourceAdapter, SourceError, SourceRecord};

pub struct MarkdownDir {
    root: PathBuf,
}

impl MarkdownDir {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

impl SourceAdapter for MarkdownDir {
    fn enumerate(&self) -> Box<dyn Iterator<Item = Result<SourceRecord, SourceError>> + Send + '_> {
        Box::new(MarkdownDirIter::new(self.root.clone()))
    }
}

/// Stack-based recursive directory walker. We avoid pulling in `walkdir`
/// because the traversal is simple enough to express directly and the dep
/// surface stays smaller. `BTreeSet`-ordered children would give
/// platform-stable enumeration order; for now we collect into a `Vec` and
/// sort, which is enough for tests without an extra import.
struct MarkdownDirIter {
    /// Files queued for read. Pushed when a directory is expanded; popped
    /// to drive `next()`. We pre-sort each directory's entries so the
    /// iteration order is stable across runs (important for tests, and
    /// makes build-state checkpoints deterministic).
    pending_files: Vec<PathBuf>,

    /// Directories queued for expansion. Drained when `pending_files` is
    /// empty so we always finish a directory before descending into the
    /// next one — keeps order intuitive for users following a build log.
    pending_dirs: Vec<PathBuf>,

    /// First IO error encountered. Surfaces through the iterator and then
    /// the iterator returns `None`.
    initial_error: Option<SourceError>,
}

impl MarkdownDirIter {
    fn new(root: PathBuf) -> Self {
        let mut iter = Self {
            pending_files: Vec::new(),
            pending_dirs: vec![root.clone()],
            initial_error: None,
        };

        if !root.exists() {
            iter.initial_error = Some(SourceError::NotFound(root.display().to_string()));
            iter.pending_dirs.clear();
        }
        iter
    }

    /// Expand one queued directory: push its `.md` files into
    /// `pending_files` and its sub-directories into `pending_dirs`.
    /// Returns `Some(Err)` if reading the directory fails — we surface
    /// the error through `next()` rather than swallowing it.
    fn expand_one(&mut self) -> Option<Result<(), SourceError>> {
        let dir = self.pending_dirs.pop()?;
        let read = match fs::read_dir(&dir) {
            Ok(read) => read,
            Err(error) => {
                return Some(Err(SourceError::Io {
                    path: dir.display().to_string(),
                    error,
                }));
            }
        };

        let mut files = Vec::new();
        let mut subdirs = Vec::new();
        for entry in read {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    return Some(Err(SourceError::Io {
                        path: dir.display().to_string(),
                        error,
                    }));
                }
            };
            let path = entry.path();
            let file_type = match entry.file_type() {
                Ok(t) => t,
                Err(error) => {
                    return Some(Err(SourceError::Io {
                        path: path.display().to_string(),
                        error,
                    }));
                }
            };
            if file_type.is_dir() {
                subdirs.push(path);
            } else if file_type.is_file() && is_markdown(&path) {
                files.push(path);
            }
            // Symlinks and other types are skipped silently for v1.
        }

        // Stable order: alphabetical within each directory.
        files.sort();
        subdirs.sort();

        // Push subdirs in reverse so the first sub-directory pops first
        // (depth-first, alphabetical).
        for subdir in subdirs.into_iter().rev() {
            self.pending_dirs.push(subdir);
        }
        // Push files in reverse so the first file pops first.
        for file in files.into_iter().rev() {
            self.pending_files.push(file);
        }
        Some(Ok(()))
    }
}

fn is_markdown(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("md"))
}

impl Iterator for MarkdownDirIter {
    type Item = Result<SourceRecord, SourceError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(err) = self.initial_error.take() {
            return Some(Err(err));
        }

        loop {
            if let Some(file) = self.pending_files.pop() {
                return Some(read_markdown_file(&file));
            }
            match self.expand_one()? {
                Ok(()) => continue, // newly expanded files/dirs queued; loop
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

fn read_markdown_file(path: &Path) -> Result<SourceRecord, SourceError> {
    let bytes = fs::read(path).map_err(|error| SourceError::Io {
        path: path.display().to_string(),
        error,
    })?;
    let text = String::from_utf8(bytes).map_err(|_| SourceError::InvalidUtf8 {
        path: path.display().to_string(),
    })?;
    Ok(SourceRecord::new(path.display().to_string(), text))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    fn write_md(dir: &Path, name: &str, body: &str) {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, body).unwrap();
    }

    fn collect(adapter: &MarkdownDir) -> Vec<SourceRecord> {
        adapter
            .enumerate()
            .collect::<Result<Vec<_>, _>>()
            .expect("no errors in test fixture")
    }

    #[test]
    fn empty_dir_yields_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let records = collect(&MarkdownDir::new(tmp.path()));
        assert!(records.is_empty());
    }

    #[test]
    fn missing_dir_surfaces_not_found() {
        let bogus = std::env::temp_dir().join("whisper-agent-knowledge-test-nonexistent-xyz");
        let _ = fs::remove_dir_all(&bogus);
        let adapter = MarkdownDir::new(&bogus);
        let mut iter = adapter.enumerate();
        let first = iter.next().expect("one error item expected").unwrap_err();
        assert!(matches!(first, SourceError::NotFound(_)), "{first:?}");
        assert!(iter.next().is_none(), "iterator ends after error");
    }

    #[test]
    fn yields_one_record_per_md_file() {
        let tmp = tempfile::tempdir().unwrap();
        write_md(tmp.path(), "a.md", "alpha content");
        write_md(tmp.path(), "b.md", "beta content");

        let records = collect(&MarkdownDir::new(tmp.path()));
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].text, "alpha content");
        assert_eq!(records[1].text, "beta content");
        // source_ids are file paths
        assert!(records[0].source_id.ends_with("a.md"));
        assert!(records[1].source_id.ends_with("b.md"));
    }

    #[test]
    fn ignores_non_md_files() {
        let tmp = tempfile::tempdir().unwrap();
        write_md(tmp.path(), "keep.md", "yes");
        fs::write(tmp.path().join("ignore.txt"), "no").unwrap();
        fs::write(tmp.path().join("README"), "no").unwrap();

        let records = collect(&MarkdownDir::new(tmp.path()));
        assert_eq!(records.len(), 1);
        assert!(records[0].source_id.ends_with("keep.md"));
    }

    #[test]
    fn recurses_subdirectories() {
        let tmp = tempfile::tempdir().unwrap();
        write_md(tmp.path(), "top.md", "T");
        write_md(tmp.path(), "subdir/inner.md", "I");
        write_md(tmp.path(), "subdir/deep/deeper.md", "D");

        let records = collect(&MarkdownDir::new(tmp.path()));
        let texts: Vec<&str> = records.iter().map(|r| r.text.as_str()).collect();
        assert_eq!(texts.len(), 3);
        assert!(texts.contains(&"T"));
        assert!(texts.contains(&"I"));
        assert!(texts.contains(&"D"));
    }

    #[test]
    fn enumeration_order_is_stable_across_runs() {
        let tmp = tempfile::tempdir().unwrap();
        // Files in random-ish order; expect alphabetical
        write_md(tmp.path(), "z.md", "z");
        write_md(tmp.path(), "a.md", "a");
        write_md(tmp.path(), "m.md", "m");

        let first: Vec<String> = collect(&MarkdownDir::new(tmp.path()))
            .into_iter()
            .map(|r| r.source_id)
            .collect();
        let second: Vec<String> = collect(&MarkdownDir::new(tmp.path()))
            .into_iter()
            .map(|r| r.source_id)
            .collect();
        assert_eq!(first, second);
        assert!(first[0].ends_with("a.md"));
        assert!(first[1].ends_with("m.md"));
        assert!(first[2].ends_with("z.md"));
    }

    #[test]
    fn case_insensitive_extension_match() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("upper.MD"), "u").unwrap();
        fs::write(tmp.path().join("mixed.Md"), "m").unwrap();
        fs::write(tmp.path().join("lower.md"), "l").unwrap();

        let records = collect(&MarkdownDir::new(tmp.path()));
        assert_eq!(records.len(), 3);
    }

    #[test]
    fn invalid_utf8_surfaces_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("bad.md"), [0xff, 0xfe, 0xfd]).unwrap();

        let adapter = MarkdownDir::new(tmp.path());
        let mut iter = adapter.enumerate();
        let item = iter.next().unwrap();
        assert!(
            matches!(item, Err(SourceError::InvalidUtf8 { .. })),
            "{item:?}"
        );
    }

    #[test]
    fn content_hash_is_stable_per_text() {
        let tmp = tempfile::tempdir().unwrap();
        write_md(tmp.path(), "first.md", "same content");
        write_md(tmp.path(), "second.md", "same content");

        let records = collect(&MarkdownDir::new(tmp.path()));
        assert_eq!(records.len(), 2);
        assert_eq!(
            records[0].content_hash, records[1].content_hash,
            "identical text → identical hash regardless of file path",
        );
    }
}
