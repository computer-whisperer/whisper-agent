//! Parser for `<bucket>/slots/<slot_id>/manifest.toml` — the slot-level
//! manifest written by the build pipeline. System-managed; never edited
//! by the user.
//!
//! Most fields are *frozen* at slot creation. The state, build timestamps,
//! and statistics block update during the slot's life. The serving block
//! can be flipped without rebuild — re-activating in a different mode just
//! reloads the slot.
//!
//! See `docs/design_knowledge_db.md` § "Configuration schemas" for the
//! canonical reference.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::config::{Quantization, ServingMode};
use super::types::SlotId;

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SlotManifest {
    /// ULID-shaped slot id, immutable, also the directory name under
    /// `<bucket>/slots/`.
    pub slot_id: SlotId,

    /// Slot lifecycle state. Updates during build; final state once the
    /// slot has reached `Ready`, `Failed`, or `Archived`.
    pub state: SlotState,

    pub created_at: chrono::DateTime<chrono::Utc>,

    /// `None` while the slot is still in the `Planning` stage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build_started_at: Option<chrono::DateTime<chrono::Utc>>,

    /// `None` until the slot reaches `Ready`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build_completed_at: Option<chrono::DateTime<chrono::Utc>>,

    /// Frozen snapshot of the chunker config that produced this slot's
    /// chunks, including the *resolved* tokenizer (kind + content hash).
    /// The chunker is what determines chunk-id stability — if a new
    /// chunker is needed, build a new slot rather than mutating an
    /// existing one.
    pub chunker_snapshot: ChunkerSnapshot,

    pub embedder: EmbedderSnapshot,

    /// Sparse-path snapshot. Absent if sparse was disabled at slot-build
    /// time (the slot has no tantivy index in that case).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sparse: Option<SparseSnapshot>,

    pub serving: ServingSnapshot,

    #[serde(default)]
    pub stats: SlotStats,

    /// Set on slots produced by compaction; absent for slots built from
    /// source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lineage: Option<SlotLineage>,
}

impl SlotManifest {
    pub fn from_toml_str(s: &str) -> Result<Self, super::types::BucketError> {
        toml::from_str(s).map_err(|e| super::types::BucketError::Config(e.to_string()))
    }

    pub fn to_toml_string(&self) -> Result<String, super::types::BucketError> {
        toml::to_string_pretty(self).map_err(|e| super::types::BucketError::Config(e.to_string()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SlotState {
    /// Source enumeration in progress; chunking hasn't started.
    Planning,
    /// Chunking, embedding, sparse indexing, and/or final passes in flight.
    /// Resumable sub-state lives in `build.state`, not in the manifest.
    Building,
    /// All artifacts present and durable; queryable.
    Ready,
    /// Build aborted. Slot may have partial artifacts on disk; not
    /// queryable. A failed slot is left in place for diagnosis until
    /// explicitly cleaned up.
    Failed,
    /// Demoted from `active` and slated for GC after the retention
    /// window. Still queryable until the directory is removed.
    Archived,
}

/// Embedder this slot was built against. Provider name binds to a current
/// server-config entry; `model_id` is what TEI actually reported via
/// `/info` at build time. Both are recorded so a query against the slot
/// can validate the embedder hasn't been silently rebound to a different
/// model under the same provider name.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct EmbedderSnapshot {
    pub provider: String,
    pub model_id: String,
    pub dimension: u32,
}

/// Frozen chunker snapshot for the slot manifest. Carries the chunking
/// parameters plus the *resolved* tokenizer — the kind + content hash of
/// whatever `tokenizer.json` (or heuristic ratio) the slot was built
/// against. Distinct from
/// [`ChunkerConfig`](super::config::ChunkerConfig) because the user-config
/// allows the unresolved `Auto` variant; once a slot is built, the
/// snapshot records the concrete result that was used.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ChunkerSnapshot {
    /// Strategy id — currently always `"token_based"`. Future strategies
    /// (e.g. `"markdown_aware"`) would extend this; the field exists for
    /// forward-compat.
    pub strategy: String,
    pub chunk_tokens: u32,
    #[serde(default)]
    pub overlap_tokens: u32,
    pub tokenizer: TokenizerSnapshot,
}

/// Resolved tokenizer identity for a built slot. The content_hash is
/// `blake3` over the raw `tokenizer.json` bytes — drift detection on
/// resume can compare against this to catch a tokenizer file being
/// edited under an in-progress build.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TokenizerSnapshot {
    HfModel {
        model_id: String,
        content_hash: String,
    },
    Path {
        path: PathBuf,
        content_hash: String,
    },
    Heuristic {
        chars_per_token: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct SparseSnapshot {
    pub tokenizer: String,
}

/// Current serving configuration for the slot. May be flipped without
/// rebuild — re-activating the slot picks up the new mode and reloads.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ServingSnapshot {
    pub mode: ServingMode,
    pub quantization: Quantization,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct SlotStats {
    /// Source records walked during ingest (articles, files, ...). May be
    /// `0` while still planning.
    #[serde(default)]
    pub source_records: u64,

    /// Total chunks in the slot's *base* layer. Stable after build.
    #[serde(default)]
    pub chunk_count: u64,

    /// Total vectors in the base layer. Equals `chunk_count` once dense
    /// indexing finishes; lags during the embedding stage.
    #[serde(default)]
    pub vector_count: u64,

    /// Chunks added since the last compaction (the delta layer). Reset
    /// to 0 when a compacted slot replaces this one.
    #[serde(default)]
    pub delta_chunk_count: u64,

    /// Tombstoned chunk count. Reset to 0 on compaction (tombstoned
    /// chunks are physically removed).
    #[serde(default)]
    pub tombstone_count: u64,

    /// Total disk footprint of the slot (all files combined).
    #[serde(default)]
    pub disk_size_bytes: u64,

    /// In-memory footprint when loaded in `ram` serving mode. Meaningless
    /// for `disk` mode (recorded as 0 or omitted).
    #[serde(default)]
    pub ram_size_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct SlotLineage {
    /// Slot id of the slot this one was compacted from.
    pub prior_slot: SlotId,

    pub compacted_at: chrono::DateTime<chrono::Utc>,

    /// Chunks dropped during compaction (tombstones reclaimed).
    pub compaction_dropped_chunks: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_full_manifest() {
        let toml = r#"
slot_id = "01HVAB1234567890ABCDEFGHJK"
state = "ready"
created_at = "2026-04-25T10:00:00Z"
build_started_at = "2026-04-25T10:01:32Z"
build_completed_at = "2026-04-29T03:14:00Z"

[chunker_snapshot]
strategy = "token_based"
chunk_tokens = 500
overlap_tokens = 50
[chunker_snapshot.tokenizer]
kind = "heuristic"
chars_per_token = 4

[embedder]
provider = "tei_qwen3_embed_0_6b"
model_id = "Qwen/Qwen3-Embedding-0.6B"
dimension = 1024

[sparse]
tokenizer = "default"

[serving]
mode = "ram"
quantization = "f32"

[stats]
source_records = 6843217
chunk_count = 13487213
vector_count = 13487213
delta_chunk_count = 0
tombstone_count = 0
disk_size_bytes = 84213456789
ram_size_bytes = 56789123456
"#;
        let m = SlotManifest::from_toml_str(toml).unwrap();
        assert_eq!(m.slot_id, "01HVAB1234567890ABCDEFGHJK");
        assert_eq!(m.state, SlotState::Ready);
        assert!(m.build_started_at.is_some());
        assert!(m.build_completed_at.is_some());
        assert_eq!(m.chunker_snapshot.strategy, "token_based");
        assert_eq!(m.chunker_snapshot.chunk_tokens, 500);
        assert_eq!(m.chunker_snapshot.overlap_tokens, 50);
        assert!(matches!(
            m.chunker_snapshot.tokenizer,
            TokenizerSnapshot::Heuristic { chars_per_token: 4 },
        ));
        assert_eq!(m.embedder.provider, "tei_qwen3_embed_0_6b");
        assert_eq!(m.embedder.model_id, "Qwen/Qwen3-Embedding-0.6B");
        assert_eq!(m.embedder.dimension, 1024);
        assert_eq!(
            m.sparse.as_ref().map(|s| s.tokenizer.as_str()),
            Some("default")
        );
        assert_eq!(m.serving.mode, ServingMode::Ram);
        assert_eq!(m.serving.quantization, Quantization::F32);
        assert_eq!(m.stats.chunk_count, 13_487_213);
        assert!(m.lineage.is_none());
    }

    #[test]
    fn parses_manifest_without_optional_blocks() {
        let toml = r#"
slot_id = "01HVAB1234567890ABCDEFGHJK"
state = "building"
created_at = "2026-04-25T10:00:00Z"

[chunker_snapshot]
strategy = "token_based"
chunk_tokens = 500
[chunker_snapshot.tokenizer]
kind = "heuristic"
chars_per_token = 4

[embedder]
provider = "tei"
model_id = "model-x"
dimension = 768

[serving]
mode = "disk"
quantization = "f32"
"#;
        let m = SlotManifest::from_toml_str(toml).unwrap();
        assert_eq!(m.state, SlotState::Building);
        assert!(m.build_started_at.is_none());
        assert!(m.build_completed_at.is_none());
        assert!(m.sparse.is_none());
        assert!(m.lineage.is_none());
        assert_eq!(m.stats, SlotStats::default());
    }

    #[test]
    fn parses_manifest_with_lineage() {
        let toml = r#"
slot_id = "01HVCD7777"
state = "ready"
created_at = "2026-05-15T02:30:00Z"
build_started_at = "2026-05-15T02:30:00Z"
build_completed_at = "2026-05-15T03:45:00Z"

[chunker_snapshot]
strategy = "token_based"
chunk_tokens = 500
[chunker_snapshot.tokenizer]
kind = "heuristic"
chars_per_token = 4

[embedder]
provider = "tei"
model_id = "m"
dimension = 768

[serving]
mode = "ram"
quantization = "f32"

[lineage]
prior_slot = "01HV9876ABCDEFGHJKLMNPQRST"
compacted_at = "2026-05-15T02:30:00Z"
compaction_dropped_chunks = 1234567
"#;
        let m = SlotManifest::from_toml_str(toml).unwrap();
        let lineage = m.lineage.expect("expected lineage");
        assert_eq!(lineage.prior_slot, "01HV9876ABCDEFGHJKLMNPQRST");
        assert_eq!(lineage.compaction_dropped_chunks, 1_234_567);
    }

    #[test]
    fn manifest_roundtrips_through_toml() {
        let toml = r#"
slot_id = "01HVAB"
state = "ready"
created_at = "2026-04-25T10:00:00Z"
build_started_at = "2026-04-25T10:01:00Z"
build_completed_at = "2026-04-29T03:14:00Z"

[chunker_snapshot]
strategy = "token_based"
chunk_tokens = 500
overlap_tokens = 50
[chunker_snapshot.tokenizer]
kind = "heuristic"
chars_per_token = 4

[embedder]
provider = "tei"
model_id = "m"
dimension = 1024

[sparse]
tokenizer = "default"

[serving]
mode = "ram"
quantization = "f32"

[stats]
chunk_count = 100
vector_count = 100
"#;
        let m = SlotManifest::from_toml_str(toml).unwrap();
        let serialized = m.to_toml_string().unwrap();
        let m2 = SlotManifest::from_toml_str(&serialized).unwrap();
        assert_eq!(m, m2);
    }

    #[test]
    fn rejects_unknown_top_level_field() {
        let toml = r#"
slot_id = "x"
state = "ready"
created_at = "2026-04-25T10:00:00Z"
unexpected_field = "nope"

[chunker_snapshot]
strategy = "token_based"
chunk_tokens = 500
[chunker_snapshot.tokenizer]
kind = "heuristic"
chars_per_token = 4

[embedder]
provider = "tei"
model_id = "m"
dimension = 1024

[serving]
mode = "ram"
quantization = "f32"
"#;
        SlotManifest::from_toml_str(toml).unwrap_err();
    }

    #[test]
    fn rejects_unknown_state() {
        let toml = r#"
slot_id = "x"
state = "imaginary"
created_at = "2026-04-25T10:00:00Z"

[chunker_snapshot]
strategy = "token_based"
chunk_tokens = 500
[chunker_snapshot.tokenizer]
kind = "heuristic"
chars_per_token = 4

[embedder]
provider = "tei"
model_id = "m"
dimension = 1024

[serving]
mode = "ram"
quantization = "f32"
"#;
        SlotManifest::from_toml_str(toml).unwrap_err();
    }
}
