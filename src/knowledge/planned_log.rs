//! Binary append-only log of planning-phase records: `(source_id,
//! content_hash)` pairs in the order the source adapter emitted them.
//!
//! This file is the bulk of a slot's resume journal at wiki scale —
//! one entry per source record, ~6.8 M on enwiki. Storing it as
//! length-prefixed binary instead of jsonl + hex-encoded hashes both
//! shrinks the on-disk footprint (~50 % smaller) and avoids the
//! json/hex parse cost on every resume.
//!
//! ## On-disk format
//!
//! 16-byte file header followed by a sequence of records:
//!
//! ```text
//! header  := magic:8 | version:u32_le | reserved:u32_le
//! magic   := b"WAPLAN\0\0"
//! version := 1
//! record  := source_id_len:u32_le | source_id:utf8 | content_hash:[u8; 32]
//! ```
//!
//! Records carry no end-of-stream marker — a clean append at any point
//! is the resume-correct truncation. A partial trailing record (the
//! writer crashed mid-record) surfaces as an `UnexpectedEof` from the
//! iterator; resume code treats that as "the prior planning phase
//! didn't finish, replan from scratch", same as a missing file.
//!
//! Source-side dedup happens in `chunker_fut`, not here — every record
//! the source iterator produces lands in this log even if its
//! `content_hash` matches an earlier one. That's load-bearing: the
//! resume code uses the log positionally to drift-check the source
//! against the prior plan, and the dedup-skipped records carry the
//! hash that the chunker_fut's `seen_hashes` needs to be primed with.

use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: [u8; 8] = *b"WAPLAN\0\0";
const VERSION: u32 = 1;
const HEADER_LEN: usize = MAGIC.len() + 4 + 4;

/// Append-only writer for `planned.bin`. Buffered; call [`Self::sync`]
/// at the end of the planning phase to make the log durable. The
/// header is written when the file is created — re-opening an existing
/// log skips that step and seeks to the end for further appends.
pub struct PlannedWriter {
    file: BufWriter<File>,
}

impl PlannedWriter {
    /// Open `path` for append. Creates the file (with header) if
    /// absent; otherwise verifies the existing header and positions
    /// for further appends.
    pub fn create_or_open(path: &Path) -> io::Result<Self> {
        let mut existed = false;
        let file = match OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)
        {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                existed = true;
                OpenOptions::new().read(true).append(true).open(path)?
            }
            Err(e) => return Err(e),
        };
        let mut writer = BufWriter::new(file);
        if !existed {
            writer.write_all(&MAGIC)?;
            writer.write_all(&VERSION.to_le_bytes())?;
            writer.write_all(&0u32.to_le_bytes())?; // reserved
        } else {
            // Verify header on re-open — wrong file at this path is
            // a real bug; surface it before any append corrupts it.
            let mut hdr = [0u8; HEADER_LEN];
            let mut r = BufReader::new(writer.get_mut().try_clone()?);
            r.read_exact(&mut hdr).map_err(|e| {
                io::Error::new(
                    e.kind(),
                    format!("planned.bin: header read failed at {}: {e}", path.display()),
                )
            })?;
            check_header(&hdr).map_err(|msg| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("{} at {}", msg, path.display()),
                )
            })?;
        }
        Ok(Self { file: writer })
    }

    /// Append one `(source_id, content_hash)` record. Buffered; the
    /// caller must invoke [`Self::sync`] at the planning-phase
    /// durability boundary.
    pub fn append(&mut self, source_id: &str, content_hash: &[u8; 32]) -> io::Result<()> {
        let bytes = source_id.as_bytes();
        let len: u32 = bytes
            .len()
            .try_into()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "source_id ≥ 4 GiB"))?;
        self.file.write_all(&len.to_le_bytes())?;
        self.file.write_all(bytes)?;
        self.file.write_all(content_hash)?;
        Ok(())
    }

    /// Flush + fsync. After this returns, every appended record
    /// survives a crash.
    pub fn sync(&mut self) -> io::Result<()> {
        self.file.flush()?;
        self.file.get_ref().sync_data()?;
        Ok(())
    }
}

/// Streaming iterator over `(source_id, content_hash)` pairs in
/// `planned.bin`. Reads the header up front and yields records one at
/// a time. A partial trailing record surfaces as `UnexpectedEof` — the
/// iterator returns it once and then returns `None`.
pub struct PlannedReader {
    inner: BufReader<File>,
    failed: bool,
}

impl PlannedReader {
    /// Open `path` for reading and verify the header. Returns the
    /// underlying io error on a missing file (callers that want to
    /// treat "no log yet" specially should check for `NotFound`).
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut inner = BufReader::new(file);
        let mut hdr = [0u8; HEADER_LEN];
        inner.read_exact(&mut hdr).map_err(|e| {
            io::Error::new(
                e.kind(),
                format!("planned.bin: header read failed at {}: {e}", path.display()),
            )
        })?;
        check_header(&hdr).map_err(|msg| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{} at {}", msg, path.display()),
            )
        })?;
        Ok(Self {
            inner,
            failed: false,
        })
    }
}

impl Iterator for PlannedReader {
    type Item = io::Result<(String, [u8; 32])>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }
        let mut len_buf = [0u8; 4];
        match self.inner.read_exact(&mut len_buf) {
            Ok(()) => {}
            // Clean EOF at a record boundary — end of stream.
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return None,
            Err(e) => {
                self.failed = true;
                return Some(Err(e));
            }
        }
        let len = u32::from_le_bytes(len_buf) as usize;
        // Cap at 4 MiB to surface a corrupted length field early
        // rather than allocate a multi-gigabyte string.
        if len > 4 * 1024 * 1024 {
            self.failed = true;
            return Some(Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("planned.bin: implausible source_id length {len}"),
            )));
        }
        let mut id_buf = vec![0u8; len];
        if let Err(e) = self.inner.read_exact(&mut id_buf) {
            self.failed = true;
            return Some(Err(e));
        }
        let mut hash = [0u8; 32];
        if let Err(e) = self.inner.read_exact(&mut hash) {
            self.failed = true;
            return Some(Err(e));
        }
        let source_id = match String::from_utf8(id_buf) {
            Ok(s) => s,
            Err(e) => {
                self.failed = true;
                return Some(Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("planned.bin: source_id is not utf-8: {e}"),
                )));
            }
        };
        Some(Ok((source_id, hash)))
    }
}

/// Read the entire log into memory. Convenient for tests; production
/// code on the resume path uses [`PlannedReader`] directly to avoid
/// materializing the full ~6.8 M-record list at wiki scale.
pub fn read_all(path: &Path) -> io::Result<Vec<(String, [u8; 32])>> {
    let reader = match PlannedReader::open(path) {
        Ok(r) => r,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    reader.collect()
}

fn check_header(hdr: &[u8; HEADER_LEN]) -> Result<(), String> {
    if hdr[..MAGIC.len()] != MAGIC {
        return Err(format!(
            "planned.bin: bad magic — got {:02x?}, expected {:02x?}",
            &hdr[..MAGIC.len()],
            MAGIC,
        ));
    }
    let version = u32::from_le_bytes(hdr[8..12].try_into().unwrap());
    if version != VERSION {
        return Err(format!(
            "planned.bin: unsupported version {version} (this build expects {VERSION})"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    #[test]
    fn roundtrip_happy_path() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("planned.bin");

        let mut w = PlannedWriter::create_or_open(&path).unwrap();
        w.append("Apollo program", &h(0xab)).unwrap();
        w.append("Saturn V", &h(0xcd)).unwrap();
        w.append("Über méssy unicode", &h(0xef)).unwrap();
        w.sync().unwrap();
        drop(w);

        let got = read_all(&path).unwrap();
        assert_eq!(got.len(), 3);
        assert_eq!(got[0], ("Apollo program".to_string(), h(0xab)));
        assert_eq!(got[1], ("Saturn V".to_string(), h(0xcd)));
        assert_eq!(got[2], ("Über méssy unicode".to_string(), h(0xef)));
    }

    #[test]
    fn append_preserves_existing_records() {
        // Re-opening must not rewrite the header or truncate prior
        // entries — the planning-phase fsync invariant relies on
        // this property if a crash/relaunch lands mid-planning.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("planned.bin");

        {
            let mut w = PlannedWriter::create_or_open(&path).unwrap();
            w.append("first", &h(1)).unwrap();
            w.sync().unwrap();
        }
        {
            let mut w = PlannedWriter::create_or_open(&path).unwrap();
            w.append("second", &h(2)).unwrap();
            w.sync().unwrap();
        }

        let got = read_all(&path).unwrap();
        assert_eq!(
            got,
            vec![("first".to_string(), h(1)), ("second".to_string(), h(2)),]
        );
    }

    #[test]
    fn missing_file_reads_as_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("nope.bin");
        assert!(read_all(&path).unwrap().is_empty());
    }

    #[test]
    fn streaming_reader_yields_in_order() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("planned.bin");
        let mut w = PlannedWriter::create_or_open(&path).unwrap();
        for i in 0..1000 {
            w.append(&format!("article-{i:04}"), &h((i % 251) as u8))
                .unwrap();
        }
        w.sync().unwrap();
        drop(w);

        let r = PlannedReader::open(&path).unwrap();
        for (i, item) in r.enumerate() {
            let (id, hash) = item.unwrap();
            assert_eq!(id, format!("article-{i:04}"));
            assert_eq!(hash, h((i % 251) as u8));
        }
    }

    #[test]
    fn header_required_on_open() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("planned.bin");
        std::fs::write(&path, b"GARBAGE_BYTES_NOT_A_HEADER_AT_ALL").unwrap();
        match PlannedReader::open(&path) {
            Err(e) => assert_eq!(e.kind(), io::ErrorKind::InvalidData),
            Ok(_) => panic!("expected InvalidData"),
        }
    }

    #[test]
    fn version_mismatch_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("planned.bin");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&999u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        match PlannedReader::open(&path) {
            Err(e) => {
                assert_eq!(e.kind(), io::ErrorKind::InvalidData);
                assert!(e.to_string().contains("unsupported version 999"));
            }
            Ok(_) => panic!("expected InvalidData"),
        }
    }

    #[test]
    fn truncated_record_surfaces_as_eof_error() {
        // A planning crash mid-record leaves a length prefix without
        // the body. The reader yields one Err and then None.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("planned.bin");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        // First record: complete
        bytes.extend_from_slice(&5u32.to_le_bytes());
        bytes.extend_from_slice(b"hello");
        bytes.extend_from_slice(&[0xaa; 32]);
        // Second record: length says 100 but only 3 bytes follow.
        bytes.extend_from_slice(&100u32.to_le_bytes());
        bytes.extend_from_slice(b"abc");
        std::fs::write(&path, &bytes).unwrap();

        let mut r = PlannedReader::open(&path).unwrap();
        let first = r.next().unwrap().unwrap();
        assert_eq!(first.0, "hello");
        let second = r.next().unwrap();
        assert!(second.is_err(), "expected error on truncated body");
        assert!(r.next().is_none());
    }

    #[test]
    fn implausible_length_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("planned.bin");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();

        let mut r = PlannedReader::open(&path).unwrap();
        let next = r.next().unwrap();
        match next {
            Err(e) => assert_eq!(e.kind(), io::ErrorKind::InvalidData),
            Ok(rec) => panic!("expected error, got {rec:?}"),
        }
    }

    #[test]
    fn wire_format_pinned() {
        // The on-disk layout is part of the contract — keep this
        // test as the canary so a future refactor can't silently
        // change it without a deliberate version bump.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("planned.bin");
        let mut w = PlannedWriter::create_or_open(&path).unwrap();
        w.append("ab", &[0x11; 32]).unwrap();
        w.sync().unwrap();
        drop(w);

        let raw = std::fs::read(&path).unwrap();
        assert_eq!(&raw[..8], &MAGIC);
        assert_eq!(&raw[8..12], &1u32.to_le_bytes());
        assert_eq!(&raw[12..16], &0u32.to_le_bytes());
        // record: 4-byte length (= 2), 2 bytes "ab", 32-byte hash
        assert_eq!(&raw[16..20], &2u32.to_le_bytes());
        assert_eq!(&raw[20..22], b"ab");
        assert_eq!(&raw[22..54], &[0x11; 32]);
        assert_eq!(raw.len(), 54);
    }
}
