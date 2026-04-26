//! Foundational types for the knowledge-bucket layer.
//!
//! These types appear in the [`Bucket`](super::Bucket) trait surface and in
//! the persisted on-disk schemas. They're deliberately thin — most carry
//! strings, hashes, or enum variants and have no dependency on the indexing
//! libraries (`hnsw_rs`, `tantivy`) or the providers layer.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Stable, human-typeable handle for a bucket.
///
/// Carries a [`scope`](BucketScope) (server-level or pod-local) and a `name`
/// that matches the bucket's directory on disk. The directory name is the
/// authoritative id — once a bucket exists, renaming would break historical
/// references in conversation history and citations. A bucket's display
/// `name` lives in `bucket.toml` and is freely editable.
///
/// Stringified form is `"<scope>:<name>"` — e.g. `"server:wikipedia_en"`,
/// `"pod:my_workspace"`. The scope prefix exists at the LLM tool boundary;
/// internally we carry the typed pair.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BucketId {
    pub scope: BucketScope,
    pub name: String,
}

impl BucketId {
    pub fn new(scope: BucketScope, name: impl Into<String>) -> Self {
        Self {
            scope,
            name: name.into(),
        }
    }

    pub fn server(name: impl Into<String>) -> Self {
        Self::new(BucketScope::Server, name)
    }

    pub fn pod(name: impl Into<String>) -> Self {
        Self::new(BucketScope::Pod, name)
    }
}

impl fmt::Display for BucketId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.scope.as_str(), self.name)
    }
}

impl FromStr for BucketId {
    type Err = ParseBucketIdError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (scope_str, name) = s.split_once(':').ok_or(ParseBucketIdError::MissingScope)?;
        let scope = scope_str.parse::<BucketScope>()?;
        validate_bucket_name(name)?;
        Ok(Self {
            scope,
            name: name.to_string(),
        })
    }
}

fn validate_bucket_name(name: &str) -> Result<(), ParseBucketIdError> {
    if name.is_empty() {
        return Err(ParseBucketIdError::EmptyName);
    }
    if name == "." || name == ".." {
        return Err(ParseBucketIdError::InvalidName(name.to_string()));
    }
    for ch in name.chars() {
        let ok = ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' || ch == '.';
        if !ok {
            return Err(ParseBucketIdError::InvalidName(name.to_string()));
        }
    }
    Ok(())
}

/// Whether a bucket lives under the server's knowledge directory (shared
/// across pods, gated by `[allow.knowledge_buckets]`) or under a pod
/// directory (implicit access for that pod only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BucketScope {
    Server,
    Pod,
}

impl BucketScope {
    pub fn as_str(self) -> &'static str {
        match self {
            BucketScope::Server => "server",
            BucketScope::Pod => "pod",
        }
    }
}

impl FromStr for BucketScope {
    type Err = ParseBucketIdError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "server" => Ok(BucketScope::Server),
            "pod" => Ok(BucketScope::Pod),
            other => Err(ParseBucketIdError::UnknownScope(other.to_string())),
        }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ParseBucketIdError {
    #[error("missing `<scope>:` prefix")]
    MissingScope,
    #[error("unknown scope `{0}` (expected `server` or `pod`)")]
    UnknownScope(String),
    #[error("empty bucket name")]
    EmptyName,
    #[error("invalid bucket name `{0}` (allowed: ASCII alphanumeric, `_`, `-`, `.`)")]
    InvalidName(String),
}

/// Stable identifier for a chunk within a bucket.
///
/// Derived as `blake3(source_record_hash || chunk_offset_in_record)`. Stable
/// across slots within a bucket as long as the chunker config is unchanged
/// — embedder rotation reuses chunk ids; chunker changes invalidate them.
/// See `docs/design_knowledge_db.md` § "Bucket and chunk identity" for
/// the full reasoning.
///
/// Stored as the raw 32-byte blake3 digest. Stringified form is the
/// lowercase 64-char hex of the digest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChunkId(pub [u8; 32]);

impl ChunkId {
    /// Compute the chunk id from a source record hash and the chunk's
    /// offset within that record. The offset is the chunker's per-record
    /// counter (0 for the first chunk emitted from a record, 1 for the
    /// second, etc.) — not a byte offset.
    pub fn from_source(source_record_hash: &[u8; 32], chunk_offset: u64) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(source_record_hash);
        hasher.update(&chunk_offset.to_le_bytes());
        Self(*hasher.finalize().as_bytes())
    }

    /// Short hex form for human-friendly display (first 12 hex chars).
    /// Not a substitute for the full id in any persistent context.
    pub fn short(self) -> String {
        let mut s = String::with_capacity(12);
        for byte in &self.0[..6] {
            use std::fmt::Write;
            let _ = write!(&mut s, "{byte:02x}");
        }
        s
    }
}

impl fmt::Display for ChunkId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in &self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl FromStr for ChunkId {
    type Err = ParseChunkIdError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.len() != 64 {
            return Err(ParseChunkIdError::WrongLength(s.len()));
        }
        let mut bytes = [0u8; 32];
        let raw = s.as_bytes();
        for (i, byte) in bytes.iter_mut().enumerate() {
            let hi = hex_nybble(raw[i * 2])?;
            let lo = hex_nybble(raw[i * 2 + 1])?;
            *byte = (hi << 4) | lo;
        }
        Ok(Self(bytes))
    }
}

fn hex_nybble(byte: u8) -> Result<u8, ParseChunkIdError> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(ParseChunkIdError::NotHex(byte as char)),
    }
}

impl Serialize for ChunkId {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        ser.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for ChunkId {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        let s = String::deserialize(de)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ParseChunkIdError {
    #[error("expected 64 hex chars, got {0}")]
    WrongLength(usize),
    #[error("not a hex character: `{0}`")]
    NotHex(char),
}

/// ULID-shaped identifier for a slot directory under a bucket. Carried as a
/// string for now; a typed wrapper can come later if the type-system payoff
/// outweighs the friction.
pub type SlotId = String;

/// Adapter-specific reference to where a chunk came from.
///
/// `source_id` is the adapter's identifier for the source record (file
/// path, article title, ...). `locator` is an optional human-readable
/// pointer into the record (byte range, section anchor, ...). Both are
/// adapter-defined and treated opaquely outside the adapter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceRef {
    pub source_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub locator: Option<String>,
}

/// A chunk with its persisted text and provenance, as returned by
/// [`Bucket::fetch_chunk`](super::Bucket::fetch_chunk) and embedded in
/// [`Candidate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub id: ChunkId,
    pub text: String,
    pub source_ref: SourceRef,
}

/// Input to [`Bucket::insert`](super::Bucket::insert) — chunk text + its
/// source reference, before the chunk id is computed.
///
/// Callers don't supply chunk ids; the bucket computes them from the
/// source-record hash and offset (see [`ChunkId::from_source`]). The
/// `source_record_hash` field carries the hash of the source record this
/// chunk came from — the bucket combines it with the per-record offset to
/// produce a stable id.
#[derive(Debug, Clone)]
pub struct NewChunk {
    pub text: String,
    pub source_ref: SourceRef,
    pub source_record_hash: [u8; 32],
    pub chunk_offset: u64,
}

/// Which retrieval path produced a candidate. Useful for dedup, observability,
/// and per-path analytics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SearchPath {
    Dense,
    Sparse,
}

/// A single candidate from a retrieval path, before reranker fusion.
///
/// `source_score` is the raw score from the underlying path (dense cosine,
/// BM25 raw) — exposed for observability only; it's not comparable across
/// paths or buckets, which is exactly why the reranker is the unification
/// layer.
///
/// `source_ref` is the chunk's adapter-level provenance — file path for
/// markdown_dir, article title for mediawiki_xml. Carried through so the
/// LLM-facing surface (`knowledge_query` tool) and the WebUI can cite
/// where each hit came from rather than just naming a chunk hash.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub bucket_id: BucketId,
    pub chunk_id: ChunkId,
    pub chunk_text: String,
    pub source_ref: SourceRef,
    pub source_score: f32,
    pub path: SearchPath,
}

/// Lifecycle status of a bucket's active slot, exposed via
/// [`Bucket::status`](super::Bucket::status).
#[derive(Debug, Clone)]
pub enum BucketStatus {
    /// Active slot is fully built and serving queries.
    Ready,
    /// First-time build of a new slot (no prior `active` slot to fall back to).
    Building {
        slot_id: SlotId,
        progress: BuildProgress,
    },
    /// Compaction in progress — there's an active slot serving queries
    /// while a new slot is being built from current state.
    Compacting {
        slot_id: SlotId,
        progress: BuildProgress,
    },
    /// Bucket directory exists but no slot has finished building yet
    /// (just-created bucket, or its only slot failed before reaching
    /// `ready`).
    NoActiveSlot,
    /// Bucket is unservable. `reason` is a short human-readable summary.
    Failed { reason: String },
}

/// Coarse build-progress signal exposed via [`BucketStatus`]. The internal
/// build-state file (`build.state`) carries the fine-grained sub-state for
/// resumability — this is just enough for UI surfacing.
#[derive(Debug, Clone)]
pub struct BuildProgress {
    /// Current named stage — `"chunking"`, `"embedding"`,
    /// `"sparse_indexing"`, `"final_index_pass"`, etc. Values are
    /// stage-machine internal; UI shouldn't case-match.
    pub stage: String,
    /// Items completed in the current stage.
    pub completed: u64,
    /// Total items in the current stage if known. `None` means "in progress
    /// with unknown total" — e.g. the planning stage hasn't produced a
    /// chunk count yet.
    pub total: Option<u64>,
}

/// Operation errors surfaced from the [`Bucket`](super::Bucket) trait.
///
/// Mirror of the transient/terminal classification on
/// [`crate::providers::model::ModelError`] in spirit, though the variants
/// here are bucket-specific.
#[derive(Debug, Error)]
pub enum BucketError {
    #[error("bucket not found: {0}")]
    NotFound(BucketId),

    #[error("no active slot for bucket {0}")]
    NoActiveSlot(BucketId),

    #[error("dimension mismatch: query vector is {query}-dim, slot expects {slot}-dim")]
    DimensionMismatch { query: u32, slot: u32 },

    #[error("retrieval path disabled for this bucket: {0:?}")]
    PathDisabled(SearchPath),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("config error: {0}")]
    Config(String),

    #[error("provider error: {0}")]
    Provider(String),

    #[error("cancelled")]
    Cancelled,

    #[error("other: {0}")]
    Other(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_id_roundtrips_through_string() {
        let id = BucketId::server("wikipedia_en");
        assert_eq!(id.to_string(), "server:wikipedia_en");
        assert_eq!("server:wikipedia_en".parse::<BucketId>().unwrap(), id);

        let pod = BucketId::pod("my-workspace");
        assert_eq!(pod.to_string(), "pod:my-workspace");
        assert_eq!("pod:my-workspace".parse::<BucketId>().unwrap(), pod);
    }

    #[test]
    fn bucket_id_rejects_missing_scope() {
        assert_eq!(
            "wikipedia".parse::<BucketId>(),
            Err(ParseBucketIdError::MissingScope),
        );
    }

    #[test]
    fn bucket_id_rejects_unknown_scope() {
        assert!(matches!(
            "global:wikipedia".parse::<BucketId>(),
            Err(ParseBucketIdError::UnknownScope(s)) if s == "global",
        ));
    }

    #[test]
    fn bucket_id_rejects_path_traversal() {
        for bad in [
            "server:",
            "server:.",
            "server:..",
            "server:foo/bar",
            "server:foo\\bar",
            "server:foo bar",
        ] {
            let res = bad.parse::<BucketId>();
            assert!(res.is_err(), "expected error for `{bad}`, got {res:?}");
        }
    }

    #[test]
    fn bucket_id_accepts_dots_underscores_dashes() {
        for ok in [
            "server:foo",
            "server:foo_bar",
            "server:foo-bar",
            "server:foo.v2",
        ] {
            ok.parse::<BucketId>()
                .unwrap_or_else(|e| panic!("{ok} should parse: {e}"));
        }
    }

    #[test]
    fn chunk_id_is_deterministic_in_inputs() {
        let hash = [0xab; 32];
        let a = ChunkId::from_source(&hash, 7);
        let b = ChunkId::from_source(&hash, 7);
        assert_eq!(a, b);

        let c = ChunkId::from_source(&hash, 8);
        assert_ne!(a, c);

        let other_hash = [0xcd; 32];
        let d = ChunkId::from_source(&other_hash, 7);
        assert_ne!(a, d);
    }

    #[test]
    fn chunk_id_hex_roundtrips() {
        let id = ChunkId::from_source(&[0xab; 32], 42);
        let hex = id.to_string();
        assert_eq!(hex.len(), 64);
        let parsed: ChunkId = hex.parse().unwrap();
        assert_eq!(parsed, id);
    }

    #[test]
    fn chunk_id_short_is_12_chars() {
        let id = ChunkId::from_source(&[0; 32], 0);
        assert_eq!(id.short().len(), 12);
    }

    #[test]
    fn chunk_id_parse_rejects_bad_length() {
        assert_eq!(
            "deadbeef".parse::<ChunkId>(),
            Err(ParseChunkIdError::WrongLength(8)),
        );
    }

    #[test]
    fn chunk_id_parse_rejects_non_hex() {
        let mut bad = "0".repeat(63);
        bad.push('z');
        assert_eq!(bad.parse::<ChunkId>(), Err(ParseChunkIdError::NotHex('z')));
    }

    #[test]
    fn chunk_id_serializes_as_hex_string() {
        let id = ChunkId::from_source(&[0xab; 32], 0);
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, format!("\"{id}\""));

        let back: ChunkId = serde_json::from_str(&json).unwrap();
        assert_eq!(back, id);
    }
}
