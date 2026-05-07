//! Parallel decoder for bzip2 multistream archives.
//!
//! Wikipedia's `pages-articles-multistream.xml.bz2` dumps are a sequence
//! of independent bz2 streams concatenated end-to-end. Each stream is
//! independently decodable, which means decompression is embarrassingly
//! parallel — but `bzip2::read::MultiBzDecoder` walks the streams
//! sequentially on a single thread, leaving N-1 cores idle. Build runs
//! against the full enwiki dump have the chunker_fut tokio task doing
//! decompression + XML parsing + chunking on one worker; moving
//! decompression off-thread *and* parallelizing it across cores frees
//! that worker to overlap parsing with the next stream's decode.
//!
//! [`ParallelMultiBzDecoder`] mmaps the file, scans once for byte-
//! aligned bz2 stream-start signatures (10-byte magic — false-positive
//! rate is ~2^-38 over a 24 GB scan), and dispatches each stream's
//! compressed byte range to a worker pool. A coordinator reorders the
//! workers' decompressed output and feeds it into a `Read + Send`
//! consumer that drops in for `MultiBzDecoder` at call sites.
//!
//! Files with no detectable stream-start magic (truncated header,
//! non-bz2 file with a `.bz2` extension) fall back to a single-threaded
//! `BzDecoder` over the whole file, matching the prior behavior on the
//! same edge cases.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use memmap2::Mmap;

/// Byte-aligned signature at the start of every bz2 stream:
/// `BZh<digit>` (4 bytes) + the bzip2 first-block magic
/// `0x314159265359` (6 bytes). Each stream's first block magic sits at
/// byte offset 4 (the bzip2 header is exactly 32 bits, then the next
/// 48 bits are this magic — both byte-aligned). Beyond byte 10 the
/// bitstream is huffman-packed, so byte-aligned matches mid-stream are
/// astronomical (`9 * 256^-9` per offset).
const STREAM_MAGIC_LEN: usize = 10;

/// Bound on the consumer-facing channel. Each entry is a full
/// decompressed stream (~5–10 MB on enwiki), so 2 ≈ 10–20 MB headroom.
const OUT_CHANNEL_CAPACITY: usize = 2;

/// Bound on the worker → coordinator channel. With N workers each
/// holding one in-flight decode, peak in-flight is `N + REORDER` where
/// reorder slots cap how far ahead workers can run before the
/// out-of-order tail of the BTreeMap kicks back at them.
const REORDER_CAPACITY: usize = 4;

/// Drop-in replacement for `bzip2::read::MultiBzDecoder` with parallel
/// per-stream decompression. Implements `Read + Send` so call sites can
/// keep wrapping it in `BufReader` and feeding it to existing parsers
/// (the MediaWiki XML iterator) unchanged.
pub struct ParallelMultiBzDecoder {
    rx: Receiver<DecompressedChunk>,
    current: Vec<u8>,
    pos: usize,
    eof: bool,
    /// Workers + coordinator. Joined on drop so the decoder doesn't
    /// leak threads (each thread exits naturally once its upstream
    /// channel closes — see `Drop`).
    threads: Vec<JoinHandle<()>>,
}

enum DecompressedChunk {
    Bytes(Vec<u8>),
    Err(io::Error),
    Eof,
}

impl ParallelMultiBzDecoder {
    /// Open `path`, mmap it, and spawn the decompression pipeline.
    /// Returns an error only on file open / mmap failure; bz2 decode
    /// errors surface from `Read::read` once they reach the consumer.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        if metadata.len() == 0 {
            return Ok(Self::eof_only());
        }
        // SAFETY: the mmap is treated as immutable bytes for the
        // decoder's lifetime. Concurrent external mutation of the
        // file would race, but Wikipedia dumps are
        // download-then-read.
        let mmap = unsafe { Mmap::map(&file)? };
        let mmap = Arc::new(mmap);
        let offsets = scan_stream_offsets(&mmap);

        if offsets.is_empty() {
            // No bz2 magic anywhere — fall back to a single-stream
            // decode of the whole file. Real bz2 always starts with
            // the magic at offset 0; this branch covers truncated
            // headers and `.bz2` files that aren't actually bzip2.
            return Ok(Self::single_stream(mmap, 0, metadata.len() as usize));
        }

        let total_len = mmap.len();
        let ranges: Vec<(usize, usize)> = offsets
            .iter()
            .enumerate()
            .map(|(i, &start)| {
                let end = offsets.get(i + 1).copied().unwrap_or(total_len);
                (start, end)
            })
            .collect();

        let n_streams = ranges.len();
        let n_workers = worker_count(n_streams);

        // Job channel: feeder pushes all jobs upfront (bounded only
        // by `n_streams`, ~tens of thousands × 24 bytes — trivial),
        // then drops the sender so workers see EOF on drain.
        let (job_tx, job_rx) = mpsc::channel::<(usize, usize, usize)>();
        let job_rx = Arc::new(Mutex::new(job_rx));

        let (res_tx, res_rx) = mpsc::sync_channel::<(usize, io::Result<Vec<u8>>)>(REORDER_CAPACITY);
        let (out_tx, out_rx) = mpsc::sync_channel::<DecompressedChunk>(OUT_CHANNEL_CAPACITY);

        let mut threads = Vec::with_capacity(n_workers + 1);
        for _ in 0..n_workers {
            let mmap = Arc::clone(&mmap);
            let job_rx = Arc::clone(&job_rx);
            let res_tx = res_tx.clone();
            threads.push(thread::spawn(move || worker_loop(mmap, job_rx, res_tx)));
        }
        // Original `res_tx` must be dropped so `res_rx` sees closure
        // once every worker clone has exited.
        drop(res_tx);

        for (idx, (start, end)) in ranges.into_iter().enumerate() {
            // Send is infallible — we hold the only sender and the
            // workers' Arc<Mutex<Receiver>> is alive.
            let _ = job_tx.send((idx, start, end));
        }
        drop(job_tx);

        threads.push(thread::spawn(move || {
            coordinator_loop(n_streams, res_rx, out_tx);
        }));

        Ok(Self {
            rx: out_rx,
            current: Vec::new(),
            pos: 0,
            eof: false,
            threads,
        })
    }

    fn eof_only() -> Self {
        let (tx, rx) = mpsc::sync_channel::<DecompressedChunk>(1);
        let _ = tx.send(DecompressedChunk::Eof);
        Self {
            rx,
            current: Vec::new(),
            pos: 0,
            eof: false,
            threads: Vec::new(),
        }
    }

    fn single_stream(mmap: Arc<Mmap>, start: usize, end: usize) -> Self {
        let (out_tx, out_rx) = mpsc::sync_channel::<DecompressedChunk>(OUT_CHANNEL_CAPACITY);
        let handle = thread::spawn(move || {
            let slice = &mmap[start..end];
            let mut decoded = Vec::new();
            match bzip2::read::BzDecoder::new(slice).read_to_end(&mut decoded) {
                Ok(_) => {
                    let _ = out_tx.send(DecompressedChunk::Bytes(decoded));
                    let _ = out_tx.send(DecompressedChunk::Eof);
                }
                Err(e) => {
                    let _ = out_tx.send(DecompressedChunk::Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("bz2 decode (single-stream fallback): {e}"),
                    )));
                }
            }
        });
        Self {
            rx: out_rx,
            current: Vec::new(),
            pos: 0,
            eof: false,
            threads: vec![handle],
        }
    }
}

fn worker_count(n_streams: usize) -> usize {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    // Cap at 8: beyond that, mmap-backed reads start contending with
    // the process's other cores (chunker, embedder) for the same L3.
    // Floor at 1; n_streams may legitimately be 1 (single-stream dump).
    cores.clamp(1, 8).min(n_streams)
}

fn worker_loop(
    mmap: Arc<Mmap>,
    job_rx: Arc<Mutex<Receiver<(usize, usize, usize)>>>,
    res_tx: SyncSender<(usize, io::Result<Vec<u8>>)>,
) {
    loop {
        let job = {
            let rx = match job_rx.lock() {
                Ok(g) => g,
                Err(_) => return,
            };
            rx.recv()
        };
        let (idx, start, end) = match job {
            Ok(j) => j,
            Err(_) => return,
        };
        let slice = &mmap[start..end];
        let mut decoded = Vec::new();
        let result = bzip2::read::BzDecoder::new(slice)
            .read_to_end(&mut decoded)
            .map(|_| decoded)
            .map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "bz2 decode failed for stream {idx} ({} bytes at offset {}): {e}",
                        end - start,
                        start,
                    ),
                )
            });
        if res_tx.send((idx, result)).is_err() {
            return;
        }
    }
}

fn coordinator_loop(
    n_streams: usize,
    res_rx: Receiver<(usize, io::Result<Vec<u8>>)>,
    out_tx: SyncSender<DecompressedChunk>,
) {
    let mut next: usize = 0;
    let mut pending: BTreeMap<usize, io::Result<Vec<u8>>> = BTreeMap::new();
    while next < n_streams {
        while let Some(result) = pending.remove(&next) {
            match result {
                Ok(bytes) => {
                    if !bytes.is_empty() && out_tx.send(DecompressedChunk::Bytes(bytes)).is_err() {
                        return;
                    }
                }
                Err(e) => {
                    let _ = out_tx.send(DecompressedChunk::Err(e));
                    return;
                }
            }
            next += 1;
        }
        if next >= n_streams {
            break;
        }
        match res_rx.recv() {
            Ok((idx, result)) => {
                pending.insert(idx, result);
            }
            Err(_) => {
                let _ = out_tx.send(DecompressedChunk::Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!(
                        "decompression workers exited with {} of {} streams pending",
                        n_streams - next,
                        n_streams,
                    ),
                )));
                return;
            }
        }
    }
    let _ = out_tx.send(DecompressedChunk::Eof);
}

impl Read for ParallelMultiBzDecoder {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        loop {
            if self.pos < self.current.len() {
                let to_copy = (self.current.len() - self.pos).min(buf.len());
                buf[..to_copy].copy_from_slice(&self.current[self.pos..self.pos + to_copy]);
                self.pos += to_copy;
                return Ok(to_copy);
            }
            if self.eof {
                return Ok(0);
            }
            match self.rx.recv() {
                Ok(DecompressedChunk::Bytes(bytes)) => {
                    self.current = bytes;
                    self.pos = 0;
                }
                Ok(DecompressedChunk::Err(e)) => {
                    self.eof = true;
                    return Err(e);
                }
                Ok(DecompressedChunk::Eof) | Err(_) => {
                    self.eof = true;
                    return Ok(0);
                }
            }
        }
    }
}

impl Drop for ParallelMultiBzDecoder {
    fn drop(&mut self) {
        // Drain anything sitting in the consumer channel so a blocked
        // sender wakes up; closure of `out_rx` then propagates back
        // through coord → workers → feeder by way of channel-closed
        // errors on each `send`/`recv`.
        self.eof = true;
        while self.rx.try_recv().is_ok() {}
        for h in self.threads.drain(..) {
            let _ = h.join();
        }
    }
}

/// Find every byte-aligned bz2 stream-start in `data`. See
/// [`STREAM_MAGIC_LEN`] for why a 10-byte byte-aligned signature is
/// reliable on real-world compressed streams.
fn scan_stream_offsets(data: &[u8]) -> Vec<usize> {
    let mut offsets = Vec::new();
    if data.len() < STREAM_MAGIC_LEN {
        return offsets;
    }
    let limit = data.len() - (STREAM_MAGIC_LEN - 1);
    let mut i = 0;
    while i < limit {
        if data[i] == b'B'
            && data[i + 1] == b'Z'
            && data[i + 2] == b'h'
            && data[i + 3].is_ascii_digit()
            && data[i + 3] != b'0'
            && data[i + 4] == 0x31
            && data[i + 5] == 0x41
            && data[i + 6] == 0x59
            && data[i + 7] == 0x26
            && data[i + 8] == 0x53
            && data[i + 9] == 0x59
        {
            offsets.push(i);
            i += STREAM_MAGIC_LEN;
        } else {
            i += 1;
        }
    }
    offsets
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    fn encode_stream(body: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        let mut enc = bzip2::write::BzEncoder::new(&mut buf, bzip2::Compression::default());
        enc.write_all(body).unwrap();
        enc.finish().unwrap();
        buf
    }

    fn write_to_temp(bytes: &[u8]) -> tempfile::NamedTempFile {
        let f = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(f.path(), bytes).unwrap();
        f
    }

    fn read_all(mut r: ParallelMultiBzDecoder) -> Vec<u8> {
        let mut out = Vec::new();
        r.read_to_end(&mut out).unwrap();
        out
    }

    #[test]
    fn scanner_finds_each_stream_start_once() {
        let s1 = encode_stream(b"first stream payload");
        let s2 = encode_stream(b"second stream payload, slightly longer");
        let mut combined = Vec::new();
        combined.extend_from_slice(&s1);
        combined.extend_from_slice(&s2);

        let offsets = scan_stream_offsets(&combined);
        assert_eq!(offsets, vec![0, s1.len()]);
    }

    #[test]
    fn scanner_returns_empty_for_non_bz2() {
        let data = b"this is some text without bzip2 magic anywhere in it.";
        assert!(scan_stream_offsets(data).is_empty());
    }

    #[test]
    fn scanner_skips_partial_magic() {
        // 'BZh' followed by a non-digit then a digit — the level-byte
        // check rejects, scanner advances by one and keeps looking.
        let mut data = Vec::new();
        data.extend_from_slice(b"BZh\x00\x31\x41\x59\x26\x53\x59"); // 0 isn't 1..=9
        data.extend_from_slice(&encode_stream(b"hello"));
        let offsets = scan_stream_offsets(&data);
        assert_eq!(offsets, vec![10]);
    }

    #[test]
    fn single_stream_decodes_identically_to_bzdecoder() {
        let body: Vec<u8> = (0..10_000).map(|i| (i % 251) as u8).collect();
        let stream = encode_stream(&body);
        let f = write_to_temp(&stream);

        let dec = ParallelMultiBzDecoder::open(f.path()).unwrap();
        let got = read_all(dec);

        let mut expected = Vec::new();
        bzip2::read::BzDecoder::new(stream.as_slice())
            .read_to_end(&mut expected)
            .unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn multistream_decodes_identically_to_multibzdecoder() {
        let bodies: Vec<Vec<u8>> = (0..6)
            .map(|i| {
                let pat = format!("stream-{i}-payload-");
                pat.repeat(2_500).into_bytes()
            })
            .collect();
        let mut combined = Vec::new();
        for body in &bodies {
            combined.extend_from_slice(&encode_stream(body));
        }
        let f = write_to_temp(&combined);

        let dec = ParallelMultiBzDecoder::open(f.path()).unwrap();
        let got = read_all(dec);

        let mut expected = Vec::new();
        bzip2::read::MultiBzDecoder::new(combined.as_slice())
            .read_to_end(&mut expected)
            .unwrap();
        assert_eq!(got, expected);

        // And the concatenation of bodies should match exactly.
        let body_concat: Vec<u8> = bodies.into_iter().flatten().collect();
        assert_eq!(got, body_concat);
    }

    #[test]
    fn small_buf_reads_yield_full_payload() {
        // Exercise the Read::read inner loop — 1-byte-at-a-time
        // reads must traverse multiple decompressed-stream boundaries
        // without losing or duplicating bytes.
        let bodies: Vec<Vec<u8>> = (0..4)
            .map(|i| format!("part-{i}-").repeat(500).into_bytes())
            .collect();
        let mut combined = Vec::new();
        for body in &bodies {
            combined.extend_from_slice(&encode_stream(body));
        }
        let f = write_to_temp(&combined);
        let body_concat: Vec<u8> = bodies.iter().flatten().copied().collect();

        let mut dec = ParallelMultiBzDecoder::open(f.path()).unwrap();
        let mut got = Vec::with_capacity(body_concat.len());
        let mut byte = [0u8; 1];
        loop {
            match dec.read(&mut byte).unwrap() {
                0 => break,
                n => got.extend_from_slice(&byte[..n]),
            }
        }
        assert_eq!(got, body_concat);
    }

    #[test]
    fn empty_file_yields_zero_bytes() {
        let f = write_to_temp(&[]);
        let dec = ParallelMultiBzDecoder::open(f.path()).unwrap();
        assert!(read_all(dec).is_empty());
    }

    #[test]
    fn corrupt_payload_surfaces_error_on_read() {
        // Valid 10-byte stream-start signature, then garbage. The
        // worker's `BzDecoder::read_to_end` errors; the coordinator
        // forwards it to the consumer.
        let mut bad = Vec::new();
        bad.extend_from_slice(b"BZh9");
        bad.extend_from_slice(&[0x31, 0x41, 0x59, 0x26, 0x53, 0x59]);
        bad.extend_from_slice(&[0xFFu8; 64]);
        let f = write_to_temp(&bad);

        let mut dec = ParallelMultiBzDecoder::open(f.path()).unwrap();
        let mut sink = Vec::new();
        let err = dec.read_to_end(&mut sink).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn many_streams_finish_in_order() {
        // Stress the reorder buffer: enough streams that the
        // out-of-order decode order can't accidentally happen to be
        // sequential. Each body is uniquely identifiable so we
        // catch any bytes-out-of-order regression.
        const N: usize = 64;
        let bodies: Vec<Vec<u8>> = (0..N)
            .map(|i| format!("stream-{i:04}-").repeat(200).into_bytes())
            .collect();
        let mut combined = Vec::new();
        for body in &bodies {
            combined.extend_from_slice(&encode_stream(body));
        }
        let f = write_to_temp(&combined);

        let dec = ParallelMultiBzDecoder::open(f.path()).unwrap();
        let got = read_all(dec);
        let body_concat: Vec<u8> = bodies.iter().flatten().copied().collect();
        assert_eq!(got, body_concat);
    }

    #[test]
    fn open_missing_file_returns_io_error() {
        let bogus = std::env::temp_dir().join("whisper-agent-parallel-bz2-missing.bz2");
        let _ = std::fs::remove_file(&bogus);
        match ParallelMultiBzDecoder::open(&bogus) {
            Err(e) => assert_eq!(e.kind(), io::ErrorKind::NotFound),
            Ok(_) => panic!("expected NotFound"),
        }
    }
}
