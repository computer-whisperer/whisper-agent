//! Parser for `<bucket>/bucket.toml` — the bucket-level config the user
//! edits to define a bucket's source, chunker, search-path enablement, and
//! defaults for new slot builds.
//!
//! Changes to this file apply to the *next* slot build (or the next
//! source-rescan / compaction operation that consults it). Existing slots
//! reflect the config that was active when they were built — captured in
//! their own [`SlotManifest`](super::manifest::SlotManifest).
//!
//! See `docs/design_knowledge_db.md` § "Configuration schemas" for the
//! canonical reference and the user-editing rules.

use std::path::PathBuf;

use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BucketConfig {
    /// Display name. Free-form, user-editable. The bucket's *id* is its
    /// directory name; this is purely for UI and logs.
    pub name: String,

    #[serde(default)]
    pub description: Option<String>,

    /// Set on creation; informational. Not `Option` because every
    /// well-formed bucket records its creation time.
    pub created_at: chrono::DateTime<chrono::Utc>,

    pub source: SourceConfig,

    #[serde(default)]
    pub chunker: ChunkerConfig,

    #[serde(default)]
    pub search_paths: SearchPathsConfig,

    pub defaults: DefaultsConfig,

    #[serde(default)]
    pub compaction: CompactionConfig,
}

impl BucketConfig {
    /// Parse from a TOML string. Convenience wrapper over `toml::from_str`
    /// that returns a [`crate::knowledge::BucketError::Config`].
    pub fn from_toml_str(s: &str) -> Result<Self, super::types::BucketError> {
        toml::from_str(s).map_err(|e| super::types::BucketError::Config(e.to_string()))
    }
}

/// Source-of-truth for a bucket's content. Tagged on `kind`.
///
/// - `stored` — a reproducible archive on disk we own (wikipedia XML dump,
///   arxiv PDF cache). Re-ingestable any time.
/// - `linked` — a path or URL pointing at content we don't own. Tolerates
///   the target moving or being edited; chunks are snapshotted into the
///   bucket on ingest so queries remain serviceable when the source
///   drifts.
/// - `managed` — content authored exclusively through the API
///   (pod memory, agent-authored notes). No external source; mutations
///   come via [`Bucket::insert`](super::Bucket::insert) /
///   [`Bucket::tombstone`](super::Bucket::tombstone).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SourceConfig {
    Stored {
        adapter: String,
        archive_path: PathBuf,
    },
    Linked {
        adapter: String,
        path: PathBuf,
        #[serde(default)]
        include_globs: Vec<String>,
        #[serde(default)]
        exclude_globs: Vec<String>,
        #[serde(default)]
        rescan_on_pod_start: bool,
        #[serde(default)]
        rescan_strategy: RescanStrategy,
    },
    Managed {},
}

impl SourceConfig {
    pub fn kind(&self) -> &'static str {
        match self {
            SourceConfig::Stored { .. } => "stored",
            SourceConfig::Linked { .. } => "linked",
            SourceConfig::Managed { .. } => "managed",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RescanStrategy {
    /// Walk the source, recompute per-record hashes, diff against current
    /// state, apply targeted inserts/tombstones. Right for medium-to-large
    /// linked buckets.
    #[default]
    DiffByHash,
    /// Throw out the existing slot's data and rebuild from scratch.
    /// Cheaper for small workspace buckets where diff-walk overhead
    /// outweighs the build cost.
    Rebuild,
}

/// Chunking strategy + parameters. Tagged on `strategy`.
///
/// Resolved into a runnable chunker plus a frozen
/// [`ChunkerSnapshot`](super::manifest::ChunkerSnapshot) at slot-build
/// time via [`resolve_chunker`](super::chunker::resolve_chunker).
/// Changes to this field in `bucket.toml` only affect *new* slot builds —
/// existing slots remain on the chunker config they were built with, which
/// is what makes chunk ids stable across embedder rotations.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, serde::Serialize)]
#[serde(tag = "strategy", rename_all = "snake_case")]
pub enum ChunkerConfig {
    TokenBased {
        chunk_tokens: u32,
        #[serde(default)]
        overlap_tokens: u32,
        #[serde(default)]
        tokenizer: TokenizerSource,
    },
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        ChunkerConfig::TokenBased {
            chunk_tokens: 500,
            overlap_tokens: 50,
            tokenizer: TokenizerSource::default(),
        }
    }
}

impl ChunkerConfig {
    pub fn strategy(&self) -> &'static str {
        match self {
            ChunkerConfig::TokenBased { .. } => "token_based",
        }
    }
}

/// Where the chunker's tokenizer comes from. The default — `Auto` — asks
/// the bucket's configured embedder for its model id at slot-build time
/// and fetches `tokenizer.json` from the HuggingFace Hub. Explicit values
/// (`HfModel`, `Path`) override that. `Heuristic` skips real BPE entirely
/// and falls back to a `chars_per_token` ratio — useful for tests, dev
/// loops without a tokenizer file on disk, and tiny buckets where chunk
/// quality doesn't matter.
///
/// Per the design doc, the chunker's tokenizer is **independent of the
/// embedder** in steady state — chunk-ids stay stable across embedder
/// rotations as long as the chunker config doesn't change. The `Auto`
/// variant resolves once at slot-build and freezes the result into the
/// slot manifest, so subsequent embedder rotations don't invalidate
/// chunk-ids unless the user also edits the chunker config.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TokenizerSource {
    /// Resolve from the bucket's configured embedder via
    /// `EmbeddingProvider::list_models()`. On any resolution failure
    /// (no network, model has no `tokenizer.json`, embedder doesn't
    /// expose a model id), falls back to [`TokenizerSource::Heuristic`]
    /// with `chars_per_token = 4` and a `tracing::warn`.
    #[default]
    Auto,
    /// Explicit HuggingFace model id (e.g. `Qwen/Qwen3-Embedding-0.6B`).
    /// Fetched and cached via `hf-hub` at slot-build start. Hard error
    /// on resolution failure — explicit means explicit.
    HfModel { model_id: String },
    /// Local `tokenizer.json` path. Hard error if the file can't be read
    /// or parsed.
    Path { path: PathBuf },
    /// Char-window heuristic — no tokenizer, just `chars_per_token`
    /// applied to a sliding window. Always succeeds; used as the
    /// auto-resolution fallback.
    Heuristic { chars_per_token: u32 },
}

/// Per-bucket flags for whether dense and sparse retrieval are exposed.
///
/// Defaults: both enabled. Some content (highly structured tables, code
/// with rare identifiers) is better served sparse-only; some prose
/// content might be configured dense-only when terseness matters more
/// than safety net. Disabling a path causes the corresponding `*_search`
/// method on the bucket to return empty candidate lists.
#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize)]
pub struct SearchPathsConfig {
    #[serde(default)]
    pub dense: DensePathConfig,
    #[serde(default)]
    pub sparse: SparsePathConfig,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct DensePathConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
}

impl Default for DensePathConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SparsePathConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Tokenizer preset for tantivy. Adapter-specific values (`"code"`,
    /// `"en_stem"`, language-specific) come later; for now the values are
    /// validated against tantivy's set at slot-build time, not here.
    #[serde(default = "default_tokenizer")]
    pub tokenizer: String,
}

impl Default for SparsePathConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            tokenizer: default_tokenizer(),
        }
    }
}

/// Defaults applied when a new slot is built. Slot-creation operations
/// may override these; the chosen values get *frozen* into the new slot's
/// manifest at build time.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct DefaultsConfig {
    /// Provider name from server config (`[embedding_providers.{name}]`).
    pub embedder: String,

    #[serde(default)]
    pub serving_mode: ServingMode,

    #[serde(default)]
    pub quantization: Quantization,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ServingMode {
    /// Vectors + HNSW graph loaded into a `Vec<f32>` / in-memory adjacency
    /// arrays on slot activation. Sub-ms search; fits all-wiki + arxiv
    /// comfortably on c-srv3's 3 TB DDR3.
    #[default]
    Ram,
    /// Vectors + graph mmap'd; OS page cache handles hot/cold. 5–50 ms
    /// search on NVMe; right for smaller boxes where the index doesn't
    /// fit in RAM.
    Disk,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Quantization {
    #[default]
    F32,
    F16,
    Int8,
}

/// Heuristics that drive automatic compaction. Manual triggers are always
/// available regardless of these settings.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CompactionConfig {
    /// Compact when delta-layer chunk count exceeds this fraction of base.
    #[serde(default = "default_delta_ratio")]
    pub delta_ratio_threshold: f32,

    /// Compact when tombstone count exceeds this fraction of total chunks.
    #[serde(default = "default_tombstone_ratio")]
    pub tombstone_ratio_threshold: f32,

    /// Never compact more often than this. Humantime-formatted (e.g.
    /// `"6h"`, `"30m"`); parsed by consumers, not here.
    #[serde(default = "default_min_interval")]
    pub min_interval: String,

    /// Always compact at least this often, even if the heuristic
    /// thresholds haven't been crossed.
    #[serde(default = "default_max_interval")]
    pub max_interval: String,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            delta_ratio_threshold: default_delta_ratio(),
            tombstone_ratio_threshold: default_tombstone_ratio(),
            min_interval: default_min_interval(),
            max_interval: default_max_interval(),
        }
    }
}

fn default_true() -> bool {
    true
}

fn default_tokenizer() -> String {
    "default".to_string()
}

fn default_delta_ratio() -> f32 {
    0.20
}

fn default_tombstone_ratio() -> f32 {
    0.10
}

fn default_min_interval() -> String {
    "6h".to_string()
}

fn default_max_interval() -> String {
    "30d".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_stored_bucket_with_full_config() {
        let toml = r#"
name = "Wikipedia (English)"
description = "Full English wikipedia, main namespace"
created_at = "2026-04-25T10:23:11Z"

[source]
kind = "stored"
adapter = "mediawiki_xml"
archive_path = "source/enwiki-2026-04-15.xml.bz2"

[chunker]
strategy = "token_based"
chunk_tokens = 500
overlap_tokens = 50

[search_paths]

[search_paths.dense]
enabled = true

[search_paths.sparse]
enabled = true
tokenizer = "default"

[defaults]
embedder = "tei_qwen3_embed_0_6b"
serving_mode = "ram"
quantization = "f32"

[compaction]
delta_ratio_threshold = 0.20
tombstone_ratio_threshold = 0.10
min_interval = "6h"
max_interval = "30d"
"#;
        let cfg = BucketConfig::from_toml_str(toml).unwrap();
        assert_eq!(cfg.name, "Wikipedia (English)");
        assert_eq!(
            cfg.description.as_deref(),
            Some("Full English wikipedia, main namespace")
        );
        assert_eq!(cfg.source.kind(), "stored");
        assert!(matches!(
            &cfg.source,
            SourceConfig::Stored { adapter, .. } if adapter == "mediawiki_xml"
        ));
        assert_eq!(cfg.chunker.strategy(), "token_based");
        assert!(cfg.search_paths.dense.enabled);
        assert!(cfg.search_paths.sparse.enabled);
        assert_eq!(cfg.search_paths.sparse.tokenizer, "default");
        assert_eq!(cfg.defaults.embedder, "tei_qwen3_embed_0_6b");
        assert_eq!(cfg.defaults.serving_mode, ServingMode::Ram);
        assert_eq!(cfg.defaults.quantization, Quantization::F32);
    }

    #[test]
    fn parses_linked_bucket_minimal() {
        let toml = r#"
name = "Notes"
created_at = "2026-04-25T10:23:11Z"

[source]
kind = "linked"
adapter = "markdown_dir"
path = "/home/me/notes"

[defaults]
embedder = "tei_qwen3_embed_0_6b"
"#;
        let cfg = BucketConfig::from_toml_str(toml).unwrap();
        assert_eq!(cfg.source.kind(), "linked");
        let SourceConfig::Linked {
            adapter,
            path,
            include_globs,
            exclude_globs,
            rescan_on_pod_start,
            rescan_strategy,
        } = &cfg.source
        else {
            panic!("expected Linked, got {:?}", cfg.source);
        };
        assert_eq!(adapter, "markdown_dir");
        assert_eq!(path.to_str(), Some("/home/me/notes"));
        assert!(include_globs.is_empty());
        assert!(exclude_globs.is_empty());
        assert!(!rescan_on_pod_start);
        assert_eq!(*rescan_strategy, RescanStrategy::DiffByHash);

        // Defaults filled in for omitted sections
        assert_eq!(cfg.chunker, ChunkerConfig::default());
        assert!(cfg.search_paths.dense.enabled);
        assert!(cfg.search_paths.sparse.enabled);
        assert_eq!(cfg.defaults.serving_mode, ServingMode::Ram);
    }

    #[test]
    fn parses_managed_bucket() {
        let toml = r#"
name = "Pod Memory"
created_at = "2026-04-25T10:23:11Z"

[source]
kind = "managed"

[defaults]
embedder = "tei_qwen3_embed_0_6b"
"#;
        let cfg = BucketConfig::from_toml_str(toml).unwrap();
        assert_eq!(cfg.source.kind(), "managed");
        assert!(matches!(cfg.source, SourceConfig::Managed { .. }));
    }

    #[test]
    fn rejects_unknown_source_kind() {
        let toml = r#"
name = "Bad"
created_at = "2026-04-25T10:23:11Z"
[source]
kind = "imaginary"
[defaults]
embedder = "tei"
"#;
        BucketConfig::from_toml_str(toml).unwrap_err();
    }

    #[test]
    fn rejects_unknown_top_level_field() {
        let toml = r#"
name = "Bad"
created_at = "2026-04-25T10:23:11Z"
mystery_field = "should be rejected"
[source]
kind = "managed"
[defaults]
embedder = "tei"
"#;
        BucketConfig::from_toml_str(toml).unwrap_err();
    }

    #[test]
    fn defaults_apply_to_optional_subtables() {
        let toml = r#"
name = "Defaults"
created_at = "2026-04-25T10:23:11Z"
[source]
kind = "managed"
[defaults]
embedder = "tei"
"#;
        let cfg = BucketConfig::from_toml_str(toml).unwrap();
        assert_eq!(cfg.compaction, CompactionConfig::default());
        assert_eq!(cfg.compaction.delta_ratio_threshold, 0.20);
        assert_eq!(cfg.compaction.tombstone_ratio_threshold, 0.10);
        assert_eq!(cfg.compaction.min_interval, "6h");
        assert_eq!(cfg.compaction.max_interval, "30d");
    }

    #[test]
    fn rescan_strategy_rebuild_parses() {
        let toml = r#"
name = "Notes"
created_at = "2026-04-25T10:23:11Z"
[source]
kind = "linked"
adapter = "markdown_dir"
path = "/notes"
rescan_strategy = "rebuild"
[defaults]
embedder = "tei"
"#;
        let cfg = BucketConfig::from_toml_str(toml).unwrap();
        let SourceConfig::Linked {
            rescan_strategy, ..
        } = cfg.source
        else {
            panic!("expected Linked");
        };
        assert_eq!(rescan_strategy, RescanStrategy::Rebuild);
    }
}
