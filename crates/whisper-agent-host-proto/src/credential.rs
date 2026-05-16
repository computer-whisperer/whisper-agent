//! Credential-publication types for the host-env link.
//!
//! A daemon may be configured (in its own TOML) to manage one or more
//! server-side credential files — typically the user's ChatGPT
//! subscription OAuth tokens at `~/.codex/auth.json`, which only the
//! daemon's machine can read because that's where `codex login` was
//! run. The daemon owns the local copy: it watches for external
//! rewrites and runs its own refresh-token loop, then pushes the
//! current on-disk contents to the scheduler over this protocol so
//! the scheduler's matching backend stays current.
//!
//! Pairing is bilateral: the scheduler's backend declares which
//! daemon name is allowed to publish for it (`auth.manager = "..."`),
//! and the daemon declares which backend each local file targets
//! (`backend = "..."` in `[[publish_credential]]`). The scheduler
//! rejects a [`crate::Frame::PublishCredential`] whose `backend` does
//! not name a backend whose declared manager matches the connected
//! daemon.

use serde::{Deserialize, Serialize};

/// Payload of a [`crate::Frame::PublishCredential`]. Tagged by `kind`
/// so future credential families (Google OAuth, an opaque API-key
/// rotation, etc.) drop in as new variants without forcing a wire
/// reshuffle.
///
/// The payload deliberately ships the full on-disk file contents,
/// not a parsed structure. The scheduler validates by re-parsing
/// (e.g. `CodexAuth::from_contents` for the `Codex` kind), which is
/// the same path the existing admin-paste rotation uses — one
/// validation, one persistence, no schema duplication between the
/// daemon and scheduler.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CredentialPayload {
    /// Raw `auth.json` contents (the file the Codex CLI maintains at
    /// `~/.codex/auth.json`). The scheduler writes this verbatim to
    /// the backend's configured `auth.path`, then swaps the in-memory
    /// `CodexAuth` derived from it.
    Codex { contents: String },
}

/// Outcome of a publish. Sent in [`crate::Frame::CredentialAck`].
///
/// Daemons treat `Rejected` as fatal-for-this-backend — repeating the
/// same publish will just produce the same rejection — and back off
/// until either the local file changes or operator intervention. A
/// missing ack within a sensible window is logged but not retried
/// from this layer; the next periodic refresh will publish again
/// naturally.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "result", rename_all = "snake_case")]
pub enum CredentialAckResult {
    /// Scheduler accepted the payload, persisted it to the backend's
    /// credential file, and swapped the in-memory state.
    Accepted,
    /// Scheduler refused. `reason` is a human-readable diagnostic
    /// (unknown backend, manager mismatch, payload parse failure).
    Rejected { reason: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(payload: &CredentialPayload) {
        let bytes = serde_json::to_vec(payload).expect("encode");
        let back: CredentialPayload = serde_json::from_slice(&bytes).expect("decode");
        assert_eq!(payload, &back);
    }

    #[test]
    fn codex_payload_round_trips() {
        round_trip(&CredentialPayload::Codex {
            contents: r#"{"tokens":{"id_token":"x"}}"#.to_string(),
        });
    }

    #[test]
    fn ack_result_round_trips_each_variant() {
        for ack in [
            CredentialAckResult::Accepted,
            CredentialAckResult::Rejected {
                reason: "no such backend".into(),
            },
        ] {
            let bytes = serde_json::to_vec(&ack).expect("encode");
            let back: CredentialAckResult = serde_json::from_slice(&bytes).expect("decode");
            assert_eq!(ack, back);
        }
    }
}
