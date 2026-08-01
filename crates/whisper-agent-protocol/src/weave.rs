//! Client-facing weave tier (migration step 7b).
//!
//! The weave tier lands *additively* beside the thread tier: the
//! per-thread list/subscription/streaming machinery is untouched and
//! remains the single ground-truth stream. A weave snapshot describes
//! the coordination layer — driver identity, referenced threads with
//! roles and relationships, and the presentation structure the driver
//! composed. Composition is client-side: presentation blocks
//! *reference* thread ids, and a client subscribes to those threads
//! through the existing per-thread tier, mounting their streams into
//! presentation slots. Drill-down to any referenced thread is therefore
//! literally the same machinery as normal rendering.

use serde::{Deserialize, Serialize};

use crate::ThreadDriverConfig;

/// One block of a weave's presentation structure. The driver composes
/// blocks; fixed client machinery paints them. The vocabulary is
/// deliberately minimal and grows only as real drivers demand
/// (docs/design_configurable_threads.md, Presentation).
///
/// Blocks can only reference threads the weave references — the
/// scheduler validates and drops any block naming an unreferenced
/// thread. Presentation curates; it cannot conceal (the snapshot's
/// `threads` list is always complete) or fabricate.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PresentationBlock {
    /// The conversation head. Typed input targets this thread; a
    /// compaction roll is the driver advancing this pointer.
    PrimaryTranscript { thread_id: String },
    /// Auxiliary referenced threads (checkers, subagents) the driver
    /// wants surfaced alongside the head.
    ThreadList { thread_ids: Vec<String> },
    /// One-line driver status for the chrome.
    Status { text: String },
}

/// Role of one thread reference within its weave.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WeaveThreadRole {
    Primary,
    Auxiliary,
}

/// One referenced thread in a [`WeaveSnapshot`] — the drill-down
/// ground truth. Always complete regardless of what the presentation
/// blocks choose to surface.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct WeaveThreadRefInfo {
    pub thread_id: String,
    pub role: WeaveThreadRole,
    /// Whether this weave is the thread's ticker (single-ticker rule:
    /// at most one weave ticks an active thread; `false` means the
    /// thread is referenced here but driven elsewhere or dormant).
    pub ticks: bool,
    /// Relationship kind stamped when the thread was derived or
    /// adopted ("check", "fork", ...). `None` on legacy singleton refs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relationship: Option<String>,
}

/// Client-facing snapshot of one weave: driver identity plus thread
/// refs plus presentation. Sent in reply to `SubscribeToWeave` and
/// re-broadcast to weave subscribers whenever refs or presentation
/// change. Snapshots are small (a handful of refs and blocks), so
/// updates resend the whole snapshot rather than patching.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct WeaveSnapshot {
    pub weave_id: String,
    pub pod_id: String,
    pub driver: ThreadDriverConfig,
    /// Content hash of the scripted driver program this weave last
    /// ran. `None` for builtin drivers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub driver_program_hash: Option<String>,
    /// Every thread this weave references — the guaranteed drill-down
    /// list, independent of what `presentation` surfaces.
    pub threads: Vec<WeaveThreadRefInfo>,
    /// The driver's composed display. Never empty on the wire: weaves
    /// whose driver composes nothing (builtin weaves, scripted drivers
    /// without a `present` function, or a `present` that errored) get
    /// the degenerate presentation — one `primary_transcript` block
    /// plus a `thread_list` of the auxiliaries, synthesized
    /// server-side.
    pub presentation: Vec<PresentationBlock>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_round_trips_with_tagged_blocks() {
        let snapshot = WeaveSnapshot {
            weave_id: "w1".into(),
            pod_id: "pod".into(),
            driver: ThreadDriverConfig::Scripted {
                name: "auto_mode_checker".into(),
            },
            driver_program_hash: Some("abc123".into()),
            threads: vec![
                WeaveThreadRefInfo {
                    thread_id: "t1".into(),
                    role: WeaveThreadRole::Primary,
                    ticks: true,
                    relationship: None,
                },
                WeaveThreadRefInfo {
                    thread_id: "t2".into(),
                    role: WeaveThreadRole::Auxiliary,
                    ticks: true,
                    relationship: Some("check".into()),
                },
            ],
            presentation: vec![
                PresentationBlock::PrimaryTranscript {
                    thread_id: "t1".into(),
                },
                PresentationBlock::Status {
                    text: "checking".into(),
                },
                PresentationBlock::ThreadList {
                    thread_ids: vec!["t2".into()],
                },
            ],
        };
        let json = serde_json::to_value(&snapshot).unwrap();
        assert_eq!(json["threads"][1]["relationship"], "check");
        assert_eq!(json["presentation"][0]["kind"], "primary_transcript");
        assert_eq!(json["threads"][0]["role"], "primary");
        let decoded: WeaveSnapshot = serde_json::from_value(json).unwrap();
        assert_eq!(decoded, snapshot);
    }

    #[test]
    fn legacy_singleton_ref_decodes_without_relationship() {
        let decoded: WeaveThreadRefInfo = serde_json::from_value(serde_json::json!({
            "thread_id": "t1", "role": "primary", "ticks": true,
        }))
        .unwrap();
        assert_eq!(decoded.relationship, None);
    }
}
