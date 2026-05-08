//! Daemon ↔ worker IPC over Unix `socketpair`.
//!
//! The daemon spawns this binary with FD 3 dup'd to one end of an
//! `AF_UNIX/SOCK_STREAM` socketpair (see
//! `whisper-agent-host-daemon/src/worker.rs`). This module owns FD 3
//! for the worker process: sends a [`WorkerFrame::Hello`] handshake
//! advertising the tool catalog, then loops on inbound
//! [`WorkerFrame::InvokeTool`] / [`WorkerFrame::CancelCall`], spawning
//! one task per call to drive [`crate::tools::call_stream`] and
//! emitting [`WorkerFrame::ToolChunk`] / [`WorkerFrame::ToolFinal`] back.
//!
//! # Wire format
//!
//! Length-prefixed CBOR over the stream socket: 4-byte big-endian
//! payload length, then the encoded frame. Reuses
//! `whisper-agent-worker-proto` for the frame types — the proto crate
//! is sync/tokio-free, so the length-prefix helpers live here.
//!
//! # Cancellation
//!
//! `CancelCall` arrives but is currently logged-and-dropped to match
//! the daemon-side behavior in `whisper-agent-host-daemon/src/connection.rs`
//! (where `CancelCall` from the scheduler is also a no-op pending a
//! later phase). When we wire real cancellation, this is the place to
//! grow a per-call abort registry.

use std::collections::HashMap;
use std::os::fd::FromRawFd;
use std::sync::Arc;

use futures::StreamExt;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};
use whisper_agent_worker_proto::{
    CallId, CallToolResult, ContentBlock, PROTOCOL_VERSION, ToolDescriptor, WorkerFrame,
};

use crate::tools::{self, ToolStreamItem};
use crate::workspace::Workspace;

/// FD number the daemon dup'd the socketpair end into. Hard-coded as
/// 3 because there's no general way to discover "which inherited FD
/// is the IPC socket"; the launcher contract is what makes it 3.
const IPC_FD: i32 = 3;

/// Cap on a single decoded frame's payload length. 16 MiB is well
/// past the 30 KB final tool-result cap and the 256 KB streaming cap;
/// it exists to bound a misbehaving daemon's ability to OOM the
/// worker by claiming a `u32::MAX`-sized frame.
const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;

/// Outbound channel buffer. One slot per concluded chunk / final +
/// modest headroom for the writer task draining slower than tools
/// produce. Tool calls themselves coalesce ~30 KB worth of body into
/// a single `ToolFinal`, so 64 buffered frames is a generous ceiling.
const OUT_CHANNEL_BOUND: usize = 64;

/// Take FD 3 (handed in by the daemon at spawn) and run the IPC
/// frame loop until the daemon disconnects or the writer task fails.
pub async fn run(workspace: Arc<Workspace>) -> anyhow::Result<()> {
    let socket = take_ipc_socket()?;
    let (reader, writer) = socket.into_split();

    let (out_tx, out_rx) = mpsc::channel::<WorkerFrame>(OUT_CHANNEL_BOUND);
    let writer_task = tokio::spawn(writer_loop(writer, out_rx));

    // Send Hello first — daemon waits on this with a startup-timeout
    // before accepting any tool calls. If the channel send fails, the
    // writer task already exited, which means we're hosed before we
    // started; bail out early.
    let hello = WorkerFrame::Hello {
        worker_version: env!("CARGO_PKG_VERSION").into(),
        protocol_version: PROTOCOL_VERSION,
        tools: tools::descriptors(),
    };
    if out_tx.send(hello).await.is_err() {
        anyhow::bail!("writer task exited before Hello could be sent");
    }
    info!(version = env!("CARGO_PKG_VERSION"), "worker handshake sent");

    reader_loop(reader, workspace, out_tx).await;

    // Reader returned: daemon hung up or sent a malformed frame.
    // Drop the inbound tx so the writer task drains and exits.
    drop(writer_task.await);
    Ok(())
}

/// Convert the inherited FD 3 into a tokio [`UnixStream`]. Sets
/// non-blocking on the underlying std socket — tokio refuses to
/// register a blocking FD with the reactor.
fn take_ipc_socket() -> anyhow::Result<UnixStream> {
    // SAFETY: the daemon's spawn contract guarantees FD 3 is a live
    // AF_UNIX/SOCK_STREAM socket owned by this process. We take
    // ownership exactly once, here.
    let std_socket = unsafe { std::os::unix::net::UnixStream::from_raw_fd(IPC_FD) };
    std_socket
        .set_nonblocking(true)
        .map_err(|e| anyhow::anyhow!("set_nonblocking on FD {IPC_FD}: {e}"))?;
    UnixStream::from_std(std_socket)
        .map_err(|e| anyhow::anyhow!("tokio::UnixStream::from_std: {e}"))
}

/// Drain the outbound channel and serialize each frame to the
/// socket. Exits on channel close (reader_loop dropped its sender)
/// or socket write error.
async fn writer_loop(mut writer: OwnedWriteHalf, mut out_rx: mpsc::Receiver<WorkerFrame>) {
    while let Some(frame) = out_rx.recv().await {
        if let Err(e) = write_frame(&mut writer, &frame).await {
            warn!(error = %e, "frame write failed; closing IPC link");
            break;
        }
    }
    debug!("writer loop exiting");
}

/// Read frames off the IPC socket and dispatch each to a per-call
/// task. Returns when the socket EOFs, errors, or a malformed frame
/// arrives. Spawned per-call tasks outlive this call but die with
/// the runtime when the worker process exits.
async fn reader_loop(
    mut reader: OwnedReadHalf,
    workspace: Arc<Workspace>,
    out_tx: mpsc::Sender<WorkerFrame>,
) {
    let mut call_tasks: HashMap<CallId, JoinHandle<()>> = HashMap::new();
    loop {
        match read_frame(&mut reader).await {
            Ok(Some(WorkerFrame::InvokeTool {
                call_id,
                tool_name,
                arguments,
                attachments,
            })) => {
                let handle = spawn_invoke(
                    call_id,
                    tool_name,
                    arguments,
                    attachments,
                    workspace.clone(),
                    out_tx.clone(),
                );
                if let Some(prev) = call_tasks.insert(call_id, handle) {
                    // Daemon should not reuse a call_id within a
                    // session, but if it does we abort the prior task
                    // rather than letting two finals race for the
                    // same id.
                    warn!(%call_id, "call_id collided — aborting prior task");
                    prev.abort();
                }
            }
            Ok(Some(WorkerFrame::CancelCall { call_id })) => {
                // Phase 6.0b: log-and-drop, mirroring the daemon's
                // current `CancelCall` handling on the scheduler-bound
                // wire. Real cancellation lands in a later phase.
                warn!(%call_id, "CancelCall received but not yet implemented");
            }
            Ok(Some(other)) => {
                warn!(
                    kind = frame_kind(&other),
                    "unexpected worker-bound frame; ignoring"
                );
            }
            Ok(None) => {
                info!("daemon closed IPC socket");
                break;
            }
            Err(e) => {
                warn!(error = %e, "frame read failed; closing IPC link");
                break;
            }
        }
        // Reap finished call tasks so the map doesn't grow without
        // bound under a long-running session. Cheap: HashMap::retain
        // walks once per inbound frame, which is bounded by daemon
        // throughput.
        call_tasks.retain(|_, h| !h.is_finished());
    }
}

/// Spawn the per-call task that drives [`tools::call_stream`] and
/// pushes [`WorkerFrame::ToolChunk`] / [`WorkerFrame::ToolFinal`]
/// onto the outbound channel.
fn spawn_invoke(
    call_id: CallId,
    tool_name: String,
    arguments: serde_json::Value,
    attachments: Vec<ContentBlock>,
    workspace: Arc<Workspace>,
    out_tx: mpsc::Sender<WorkerFrame>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut stream = match tools::call_stream(&workspace, &tool_name, arguments, attachments) {
            Ok(s) => s,
            Err(tools::ToolDispatchError::UnknownTool(name)) => {
                let _ = out_tx
                    .send(WorkerFrame::ToolFinal {
                        call_id,
                        result: CallToolResult::error_text(format!("unknown tool: {name}")),
                    })
                    .await;
                return;
            }
        };
        loop {
            match stream.next().await {
                Some(ToolStreamItem::Chunk(block)) => {
                    if out_tx
                        .send(WorkerFrame::ToolChunk { call_id, block })
                        .await
                        .is_err()
                    {
                        // Writer side gone; no point continuing the
                        // stream (and its underlying child process
                        // gets reaped via kill_on_drop when we drop).
                        return;
                    }
                }
                Some(ToolStreamItem::Final(result)) => {
                    let _ = out_tx
                        .send(WorkerFrame::ToolFinal { call_id, result })
                        .await;
                    return;
                }
                None => {
                    // Stream ended without a Final — synthesize one so
                    // the daemon doesn't hang waiting for a terminal.
                    let _ = out_tx
                        .send(WorkerFrame::ToolFinal {
                            call_id,
                            result: CallToolResult::error_text(
                                "tool stream ended without a Final frame",
                            ),
                        })
                        .await;
                    return;
                }
            }
        }
    })
}

/// Length-prefixed write: 4-byte big-endian payload length, then the
/// CBOR-encoded frame. The proto crate is sync/tokio-free; framing
/// for the stream socket lives here so both sides format identically.
async fn write_frame(writer: &mut OwnedWriteHalf, frame: &WorkerFrame) -> anyhow::Result<()> {
    let bytes = frame
        .encode_cbor()
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
    let len = u32::try_from(bytes.len())
        .map_err(|_| anyhow::anyhow!("frame too large to fit u32 length prefix"))?;
    writer.write_all(&len.to_be_bytes()).await?;
    writer.write_all(&bytes).await?;
    Ok(())
}

/// Length-prefixed read. Returns `Ok(None)` on clean EOF (daemon hung
/// up between frames), `Err` on a partial frame or codec failure.
async fn read_frame(reader: &mut OwnedReadHalf) -> anyhow::Result<Option<WorkerFrame>> {
    let mut len_buf = [0u8; 4];
    match reader.read_exact(&mut len_buf).await {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > MAX_FRAME_BYTES {
        anyhow::bail!("frame length {len} exceeds {MAX_FRAME_BYTES}-byte cap");
    }
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).await?;
    WorkerFrame::decode_cbor(&buf)
        .map(Some)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))
}

fn frame_kind(f: &WorkerFrame) -> &'static str {
    match f {
        WorkerFrame::Hello { .. } => "Hello",
        WorkerFrame::InvokeTool { .. } => "InvokeTool",
        WorkerFrame::CancelCall { .. } => "CancelCall",
        WorkerFrame::ToolChunk { .. } => "ToolChunk",
        WorkerFrame::ToolFinal { .. } => "ToolFinal",
    }
}

// `descriptors()` returns ToolDescriptor under the alias used in
// tools.rs; reference the canonical name here so a reader of just
// this module can find the type without chasing the alias.
#[allow(dead_code)]
type _ToolDescriptor = ToolDescriptor;
