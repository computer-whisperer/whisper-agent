//! Task runtime: the single-tokio-task scheduler and the thread state
//! machine it drives.
//!
//! [`scheduler`] owns the authoritative state and routes messages in and
//! out. [`thread`] is the per-task state machine — tasks-as-data that
//! `step` into I/O requests and `apply_io_result` to consume completions.
//! [`io_dispatch`] builds the futures that carry those I/O requests to
//! MCP/model/provision backends. [`audit`] is the append-only tool-call
//! log that records every step's decision.

pub mod audit;
pub mod io_dispatch;
pub mod memory_snapshot;
pub mod scheduler;
pub mod thread;
