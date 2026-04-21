//! Permission model.
//!
//! All primitives live in `whisper-agent-protocol::permission` so server
//! and webui share the canonical wire shapes. This module re-exports
//! them and fixes server-internal type aliases.
//!
//! See `docs/design_permissions_rework.md` for the full design.

pub use whisper_agent_protocol::permission::{
    AllowMap, BehaviorOpsCap, DispatchCap, Disposition, Escalation, PodModifyCap, Scope, SetOrAll,
    SudoDecision,
};

pub type BackendName = String;
pub type ProviderName = String;
pub type HostName = String;
pub type ToolName = String;
pub type PodId = String;
