//! Feed drivers for `tracked` source-kind buckets — the acquisition
//! layer that complements [`SourceAdapter`](super::SourceAdapter).
//!
//! Where a `SourceAdapter` *parses* an archive (MediaWiki XML, markdown
//! tree, ...), a [`FeedDriver`] *fetches* the archive on the system's
//! schedule. The two are orthogonal: a tracked Wikipedia bucket uses
//! [`WikipediaDriver`] to download the latest base + daily incrementals
//! and feeds those into the existing
//! [`MediaWikiXml`](super::MediaWikiXml) parser. Acquisition is what
//! changed; parsing is what stayed.
//!
//! Closed-enum dispatch by design — v1 ships [`WikipediaDriver`] only.
//! Adding `Wikidata` later is a new module + a new enum variant; no
//! config-driven plugin loading. See `docs/design_knowledge_db.md` §
//! "Tracked sources: self-maintained feeds" for the full design.
//!
//! Methods are async via the same `BoxFuture` shape used by the
//! [`Bucket`](super::super::bucket::Bucket) trait — object-safe, holdable
//! behind `Arc<dyn FeedDriver>` in the per-bucket worker.

pub mod wikipedia;

pub use wikipedia::WikipediaDriver;

use std::future::Future;
use std::path::Path;
use std::pin::Pin;

use thiserror::Error;
use tokio_util::sync::CancellationToken;

/// `Pin<Box<dyn Future>>` alias matching the
/// [`Bucket`](super::super::bucket::BoxFuture) and providers layers.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Driver-specific id for a base snapshot. Wikipedia uses `YYYYMMDD`
/// (e.g. `"20260401"`); other drivers may use other formats. Treat as
/// opaque outside the driver — the only operations the rest of the
/// system performs are equality, ordering (lexicographic, which matches
/// chronological for `YYYYMMDD` and any reasonable format), and string
/// display for `feed-state.toml` and UI rendering.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SnapshotId(pub String);

impl SnapshotId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for SnapshotId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Driver-specific id for one delta posted between base snapshots.
/// Wikipedia's daily incrementals also use `YYYYMMDD`; same opacity
/// contract as [`SnapshotId`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeltaId(pub String);

impl DeltaId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for DeltaId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Errors a [`FeedDriver`] can surface.
///
/// `Network` covers any HTTP / TCP failure short of a clean 404 (the
/// per-bucket worker treats it as transient and retries with backoff);
/// `NotFound` is a clean upstream-says-no (skip and try later);
/// `Checksum` is a content-integrity failure (delete the bad file and
/// retry); `Cancelled` propagates cooperatively from the supplied
/// cancellation token; `Io` covers local-filesystem failures while
/// writing the destination.
#[derive(Debug, Error)]
pub enum FeedError {
    #[error("network: {0}")]
    Network(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("checksum mismatch for {what}: expected {expected}, got {actual}")]
    Checksum {
        what: String,
        expected: String,
        actual: String,
    },

    #[error("parse error: {0}")]
    Parse(String),

    #[error("io: {0}")]
    Io(String),

    #[error("cancelled")]
    Cancelled,
}

/// Acquisition layer for a `tracked` bucket. The per-bucket feed worker
/// drives the schedule; this trait abstracts the "how do I talk to a
/// specific feed family" question.
///
/// All methods are cancellation-aware: long-running I/O (especially
/// [`fetch_base`](FeedDriver::fetch_base) for an enwiki monthly base —
/// ~24 GB) must abort cooperatively when the supplied token fires and
/// return [`FeedError::Cancelled`]. The token is the same scheduler-
/// owned cancellation primitive used by [`Bucket`](super::super::Bucket)
/// methods, so cancelling a bucket cancels its feed traffic too.
pub trait FeedDriver: Send + Sync {
    /// The most recent base snapshot id this feed publishes.
    /// Implementations typically list a directory and pick the latest
    /// entry. The result may not yet be fully published — callers
    /// should be prepared for [`fetch_base`](Self::fetch_base) to fail
    /// with [`FeedError::NotFound`] when the listing was published
    /// before the file landed.
    fn latest_base<'a>(
        &'a self,
        cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Result<SnapshotId, FeedError>>;

    /// Download the named base snapshot to `dest`. Idempotent:
    /// implementations should detect an already-complete file at `dest`
    /// (size + checksum match) and short-circuit. Partial files are
    /// resumed via HTTP `Range` when possible, otherwise re-downloaded.
    fn fetch_base<'a>(
        &'a self,
        id: &'a SnapshotId,
        dest: &'a Path,
        cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Result<(), FeedError>>;

    /// List delta ids available *strictly after* `since`, in
    /// chronological order. `None` means "list everything currently
    /// available." Empty result is normal when the bucket is up to
    /// date.
    fn list_deltas_since<'a>(
        &'a self,
        since: Option<&'a DeltaId>,
        cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Result<Vec<DeltaId>, FeedError>>;

    /// Download the named delta to `dest`. Same idempotency contract as
    /// [`fetch_base`](Self::fetch_base) — but deltas are small enough
    /// (~810 MB/day for enwiki) that resume isn't critical in practice.
    fn fetch_delta<'a>(
        &'a self,
        id: &'a DeltaId,
        dest: &'a Path,
        cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Result<(), FeedError>>;

    /// Adapter-id used to *parse* this driver's downloads — the bucket
    /// runtime looks this up against the same registry that the
    /// `kind = "stored"` path uses (`"mediawiki_xml"`, `"markdown_dir"`,
    /// ...). Acquisition and parsing are orthogonal: the same
    /// MediaWiki XML parser handles both stored-bucket archives and
    /// tracked-bucket downloads.
    fn parse_adapter(&self) -> &'static str;
}
