//! Append-only structured log for resumable slot builds.
//!
//! Each `build.state` file is a sequence of jsonl records — one
//! [`BuildStateRecord`] per line, internally tagged on `type`. The
//! build pipeline appends to it as it makes progress; on resume, the
//! file is replayed top-to-bottom to recover where the previous
//! attempt left off.
//!
//! Format choice (jsonl over a binary log) is for grep-ability and
//! schema evolution — the high-volume planning records live in the
//! sibling [`planned.bin`](super::planned_log) sidecar, so this file
//! stays small (kilobytes for the phase markers + a few tens of MB
//! for the per-batch checkpoints at full-wiki scale) and remains
//! human-greppable for "where did we get to?" debugging.
//!
//! See `docs/design_knowledge_db.md` § "Build pipeline" for how this
//! file fits the overall pipeline.
//!
//! ## Record sequence (success path)
//!
//! ```text
//! PhaseChanged { phase: planning }
//! (planned.bin is written in parallel with this phase, ×N records)
//! PhaseChanged { phase: indexing }
//! BatchEmbedded { batch_index, source_records_completed, chunks_completed }   ×M
//! PhaseChanged { phase: building_dense }
//! PhaseChanged { phase: finalizing }
//! BuildCompleted
//! ```
//!
//! Cancel / crash leaves the file ending without `BuildCompleted`;
//! that's the resume signal. Drift detection on resume is via the
//! `content_hash` recorded in `planned.bin` — re-walking the source
//! produces fresh hashes; matching ones short-circuit, mismatched
//! ones force re-emission of the affected record.

use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::disk_bucket::BuildPhase;

/// One entry in `build.state`. Internally tagged so the file is
/// grep-friendly (`grep '"type":"batch_embedded"'` works).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BuildStateRecord {
    /// Build pipeline transitioned into a new phase. Emitted at the
    /// boundary; the writer that produced records before this entry
    /// was running in the *previous* phase.
    PhaseChanged { phase: BuildPhase },

    /// Durability barrier. Every chunk emitted by source records
    /// `0..=source_records_completed - 1` has been flushed + fsynced
    /// to chunks.bin / vectors.bin / tantivy.
    ///
    /// `chunks_completed` is the cumulative chunk count up to this
    /// point — a cross-check value for resume to validate
    /// chunks.bin against the log without scanning the whole file.
    BatchEmbedded {
        batch_index: u64,
        source_records_completed: u64,
        chunks_completed: u64,
    },

    /// Terminal success record. The slot's `manifest.toml` has been
    /// written with `state = "ready"` and the `active` pointer flipped.
    /// Presence of this record means "do not resume; the slot is done."
    BuildCompleted,
}

/// Append-only writer for `build.state`. Opens the file in append
/// mode (creating if absent), so existing entries from a prior build
/// attempt are preserved. `append` is buffered and cheap — call
/// [`Self::sync`] at durability boundaries (end of Planning, after
/// each `BatchEmbedded`, after `BuildCompleted`) so a crash leaves a
/// resume-correct file. Now that the high-volume planning records
/// live in `planned.bin`, this file is small enough that a periodic
/// fsync at every BatchEmbedded checkpoint is essentially free.
pub struct BuildStateWriter {
    file: BufWriter<File>,
}

impl BuildStateWriter {
    /// Open `path` for append, creating the file if it doesn't exist.
    /// Existing entries are kept (they're the prior attempt's
    /// progress and the resume code consumes them).
    pub fn create_or_open(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            file: BufWriter::new(file),
        })
    }

    /// Append one record as a jsonl line. Buffered; no fsync. Caller
    /// must invoke [`Self::sync`] at durability boundaries (see the
    /// type-level docs for which records require sync).
    pub fn append(&mut self, record: &BuildStateRecord) -> io::Result<()> {
        // serde_json::to_writer doesn't append a newline; we add one
        // ourselves so each record lands on its own line.
        serde_json::to_writer(&mut self.file, record)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        self.file.write_all(b"\n")?;
        Ok(())
    }

    /// Flush the BufWriter to the OS file, then `sync_data()` to push
    /// it to durable storage. After this returns, every record passed
    /// to [`Self::append`] before this call survives a crash.
    pub fn sync(&mut self) -> io::Result<()> {
        self.file.flush()?;
        self.file.get_ref().sync_data()?;
        Ok(())
    }
}

/// Read every record from `path` in order. Returns an empty vec if
/// the file doesn't exist (fresh build); errors on a malformed line
/// (the file is system-managed — a corrupt entry is a real bug worth
/// surfacing rather than silently skipping).
pub fn read_all(path: &Path) -> io::Result<Vec<BuildStateRecord>> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    let mut out = Vec::new();
    for (lineno, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let rec: BuildStateRecord = serde_json::from_str(&line).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("build.state line {} parse error: {e}", lineno + 1),
            )
        })?;
        out.push(rec);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_full_sequence() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("build.state");

        let mut w = BuildStateWriter::create_or_open(&path).unwrap();
        w.append(&BuildStateRecord::PhaseChanged {
            phase: BuildPhase::Planning,
        })
        .unwrap();
        w.append(&BuildStateRecord::PhaseChanged {
            phase: BuildPhase::Indexing,
        })
        .unwrap();
        w.append(&BuildStateRecord::BatchEmbedded {
            batch_index: 0,
            source_records_completed: 2,
            chunks_completed: 5,
        })
        .unwrap();
        w.append(&BuildStateRecord::BuildCompleted).unwrap();
        drop(w);

        let records = read_all(&path).unwrap();
        assert_eq!(records.len(), 4);
        assert_eq!(
            records[0],
            BuildStateRecord::PhaseChanged {
                phase: BuildPhase::Planning
            }
        );
        assert_eq!(
            records[2],
            BuildStateRecord::BatchEmbedded {
                batch_index: 0,
                source_records_completed: 2,
                chunks_completed: 5,
            }
        );
        assert_eq!(records[3], BuildStateRecord::BuildCompleted);
    }

    #[test]
    fn append_preserves_existing_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("build.state");

        {
            let mut w = BuildStateWriter::create_or_open(&path).unwrap();
            w.append(&BuildStateRecord::BatchEmbedded {
                batch_index: 0,
                source_records_completed: 1,
                chunks_completed: 1,
            })
            .unwrap();
        }
        // Re-open and append more.
        {
            let mut w = BuildStateWriter::create_or_open(&path).unwrap();
            w.append(&BuildStateRecord::BatchEmbedded {
                batch_index: 1,
                source_records_completed: 2,
                chunks_completed: 3,
            })
            .unwrap();
        }

        let records = read_all(&path).unwrap();
        assert_eq!(records.len(), 2);
        match (&records[0], &records[1]) {
            (
                BuildStateRecord::BatchEmbedded { batch_index: a, .. },
                BuildStateRecord::BatchEmbedded { batch_index: b, .. },
            ) => {
                assert_eq!(*a, 0);
                assert_eq!(*b, 1);
            }
            other => panic!("unexpected records: {other:?}"),
        }
    }

    #[test]
    fn read_all_returns_empty_when_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("nonexistent.state");
        let records = read_all(&path).unwrap();
        assert!(records.is_empty());
    }

    #[test]
    fn read_all_skips_blank_lines() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("build.state");
        std::fs::write(
            &path,
            "{\"type\":\"build_completed\"}\n\n{\"type\":\"build_completed\"}\n",
        )
        .unwrap();
        let records = read_all(&path).unwrap();
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn read_all_errors_on_malformed_line() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("build.state");
        std::fs::write(&path, "{ this is not json }\n").unwrap();
        let err = read_all(&path).unwrap_err();
        assert!(err.to_string().contains("line 1"));
    }

    #[test]
    fn jsonl_format_matches_expected_shape() {
        // The on-disk format is part of the contract for grep-ability;
        // pin it explicitly so a serde rename or tag change is caught
        // by the test.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("build.state");
        let mut w = BuildStateWriter::create_or_open(&path).unwrap();
        w.append(&BuildStateRecord::BatchEmbedded {
            batch_index: 7,
            source_records_completed: 1024,
            chunks_completed: 4096,
        })
        .unwrap();
        drop(w);
        let bytes = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            bytes,
            "{\"type\":\"batch_embedded\",\"batch_index\":7,\
             \"source_records_completed\":1024,\"chunks_completed\":4096}\n"
        );
    }

    #[test]
    fn phase_serializes_snake_case() {
        // Pin the phase serialization too — this is what the resume
        // code matches against and what shows up if a user greps the
        // file looking for "where did we get to."
        let rec = BuildStateRecord::PhaseChanged {
            phase: BuildPhase::BuildingDense,
        };
        let line = serde_json::to_string(&rec).unwrap();
        assert_eq!(
            line,
            "{\"type\":\"phase_changed\",\"phase\":\"building_dense\"}"
        );
    }
}
