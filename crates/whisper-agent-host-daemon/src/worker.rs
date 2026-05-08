//! Per-session worker spawn + IPC handle.
//!
//! Daemon-side code that takes a [`HostEnvSpec::Landlock`], creates an
//! `AF_UNIX/SOCK_STREAM` socketpair, spawns the
//! `whisper-agent-mcp-host` child with one end dup'd to FD 3 and a
//! landlock ruleset applied in `pre_exec`, then awaits the child's
//! [`WorkerFrame::Hello`] handshake on the daemon-side end of the
//! socket. The returned [`Worker`] owns the child + a
//! frame-multiplexing channel; per-call dispatch goes through
//! [`Worker::invoke`].
//!
//! Container provisioning still returns [`WorkerError::Unsupported`].
//! Adding it does not require IPC changes — the same socketpair +
//! Hello handshake works inside a container, only the spec
//! evaluation differs.
//!
//! # Why FD 3
//!
//! Stdin/stdout/stderr keep their normal roles (stdin nulled, stdout
//! nulled, stderr piped so the daemon captures + tail-buffers
//! the child's logs for failure reporting). FD 3 is the canonical
//! "first inheritable extra FD" — the launcher contract is what
//! makes it 3, both sides hard-code that.

use std::collections::HashMap;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, mpsc, oneshot};
use tracing::{info, warn};
use whisper_agent_protocol::sandbox::{AccessMode, HostEnvSpec, NetworkPolicy, PathAccess};
use whisper_agent_worker_proto::{
    CallId, CallToolResult, ContentBlock, PROTOCOL_VERSION as WORKER_PROTOCOL_VERSION,
    ToolDescriptor, WorkerFrame,
};

/// Last N stderr bytes we retain for failure reporting. 4 KiB catches
/// a typical `anyhow!` chain without letting a chatty child bloat
/// memory.
const STDERR_TAIL_BYTES: usize = 4096;

/// How long we wait for the child's [`WorkerFrame::Hello`] handshake.
/// Bind-free (no port allocation, just process spawn + one socket
/// write) so the work is dominated by exec + ld.so + landlock setup —
/// a few hundred milliseconds at the high end. 10 s is comfortable
/// while still catching a wedged child.
const READY_TIMEOUT_SECS: u64 = 10;

/// Cap on a single decoded frame's payload length. 16 MiB mirrors the
/// worker side; the wire is daemon-private so an oversized frame is
/// either a bug or a malicious worker — either way, refuse rather
/// than try to allocate.
const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;

/// Inbound command channel buffer. Tool calls are control-plane and
/// rare relative to per-call streaming; a modest buffer is plenty.
const CMD_CHANNEL_BOUND: usize = 64;

/// FD number the daemon dup2's the child end of the socketpair onto.
/// Hard-coded by both ends of the IPC contract.
const IPC_FD: i32 = 3;

/// A running worker child + the IPC handle to it.
///
/// The daemon owns this for the session's lifetime. Drop semantics:
/// the inner [`Child`] kills the process via `kill_on_drop(true)`,
/// the [`Worker::cmd_tx`] sender drop also closes the IPC frame loop
/// (which then ignores the now-orphan socket). All in-flight call
/// oneshots resolve to [`WorkerError::Disconnected`] when the loop
/// exits.
pub struct Worker {
    pub child: Child,
    /// Tool catalog the worker advertised at handshake. The daemon
    /// caches this per-worker so the scheduler-bound capabilities
    /// surface stays consistent without re-probing.
    pub tools: Vec<ToolDescriptor>,
    cmd_tx: mpsc::Sender<WorkerCommand>,
}

#[derive(Debug, thiserror::Error)]
pub enum WorkerError {
    #[error(
        "ThreadContext.workspace_root is required — the scheduler must set it explicitly \
         (the daemon no longer guesses from the spec's RW paths)"
    )]
    NoWorkspaceRoot,
    #[error(
        "workspace_root override `{override_path}` is not under any read-write path in the spec"
    )]
    WorkspaceRootNotInSpec { override_path: String },
    #[error("spawn failed: {0}")]
    Spawn(String),
    #[error(
        "worker exited before handshake (exit_code={code:?}){}",
        format_stderr_suffix(stderr_tail)
    )]
    ChildExited {
        code: Option<i32>,
        stderr_tail: String,
    },
    #[error(
        "worker handshake didn't arrive within {seconds}s{}",
        format_stderr_suffix(stderr_tail)
    )]
    StartupTimeout { seconds: u64, stderr_tail: String },
    #[error(
        "worker protocol version mismatch (worker={worker}, daemon={daemon}){}",
        format_stderr_suffix(stderr_tail)
    )]
    ProtocolMismatch {
        worker: u32,
        daemon: u32,
        stderr_tail: String,
    },
    #[error("worker IPC closed before responding")]
    Disconnected,
    #[error("unsupported: {0}")]
    Unsupported(String),
}

fn format_stderr_suffix(tail: &str) -> String {
    if tail.is_empty() {
        String::new()
    } else {
        format!(": {tail}")
    }
}

/// One command the daemon hands the IPC frame loop. Call-shaped
/// rather than enum'd-per-frame because the loop doesn't expose the
/// raw frames — it owns the call_id ↔ oneshot routing on the daemon
/// side.
enum WorkerCommand {
    Invoke {
        call_id: CallId,
        tool_name: String,
        arguments: Value,
        attachments: Vec<ContentBlock>,
        result_tx: oneshot::Sender<Result<CallToolResult, WorkerError>>,
    },
    Cancel {
        call_id: CallId,
    },
}

impl Worker {
    /// Run one tool call against this worker. Returns the worker's
    /// terminal [`CallToolResult`] (which itself may carry
    /// `is_error = Some(true)` for tool-level failures), or a
    /// [`WorkerError`] for IPC-level failures (worker died, socket
    /// dropped, etc.).
    pub async fn invoke(
        &self,
        call_id: CallId,
        tool_name: String,
        arguments: Value,
        attachments: Vec<ContentBlock>,
    ) -> Result<CallToolResult, WorkerError> {
        let (result_tx, result_rx) = oneshot::channel();
        if self
            .cmd_tx
            .send(WorkerCommand::Invoke {
                call_id,
                tool_name,
                arguments,
                attachments,
                result_tx,
            })
            .await
            .is_err()
        {
            return Err(WorkerError::Disconnected);
        }
        result_rx.await.unwrap_or(Err(WorkerError::Disconnected))
    }

    /// Forward a cancel to the worker. Today the worker logs and
    /// drops it (matches daemon-side `CancelCall` handling); the
    /// terminal [`CallToolResult`] from the in-flight call remains
    /// authoritative.
    pub async fn cancel(&self, call_id: CallId) {
        let _ = self.cmd_tx.send(WorkerCommand::Cancel { call_id }).await;
    }
}

/// Dispatch on spec kind. Today only [`HostEnvSpec::Landlock`] is
/// implemented; container support is a future item.
///
/// `workspace_root_override` lets the per-session
/// [`whisper_agent_host_proto::ThreadContext::workspace_root`] pick a
/// specific writable path from the spec when the spec advertises more
/// than one. `None` ⇒ fall back to the first read-write path in the
/// spec.
pub async fn spawn(
    spec: &HostEnvSpec,
    mcp_host_bin: &str,
    workspace_root_override: Option<&Path>,
) -> Result<Worker, WorkerError> {
    match spec {
        HostEnvSpec::Landlock {
            allowed_paths,
            network,
        } => {
            spawn_landlock(
                allowed_paths,
                network,
                mcp_host_bin,
                workspace_root_override,
            )
            .await
        }
        HostEnvSpec::Container { .. } => Err(WorkerError::Unsupported(
            "container provisioning not yet implemented".into(),
        )),
    }
}

/// Pick the workspace root for a session. The caller (scheduler via
/// `ThreadContext.workspace_root`) **must** supply an override; the
/// daemon no longer guesses from the spec's RW paths. The override
/// must be at or below one of the spec's RW paths (landlock allows
/// access below a granted node, so a sub-path of an RW root is itself
/// RW). `None` is rejected with [`WorkerError::NoWorkspaceRoot`].
pub(crate) fn resolve_workspace_root(
    allowed_paths: &[PathAccess],
    workspace_root_override: Option<&Path>,
) -> Result<PathBuf, WorkerError> {
    let Some(override_path) = workspace_root_override else {
        return Err(WorkerError::NoWorkspaceRoot);
    };
    let rw_roots: Vec<&str> = allowed_paths
        .iter()
        .filter(|p| p.mode == AccessMode::ReadWrite)
        .map(|p| p.path.as_str())
        .collect();
    let inside_rw = rw_roots
        .iter()
        .any(|root| override_path.starts_with(Path::new(root)));
    if inside_rw {
        Ok(override_path.to_path_buf())
    } else {
        Err(WorkerError::WorkspaceRootNotInSpec {
            override_path: override_path.display().to_string(),
        })
    }
}

/// Resolve `mcp_host_bin` to an absolute path and return its parent.
/// The parent directory needs read+execute access in the landlock
/// ruleset so the binary can be loaded after exec.
fn resolve_bin_dir(mcp_host_bin: &str) -> Result<String, WorkerError> {
    let bin_path = std::path::Path::new(mcp_host_bin);
    let abs = if bin_path.is_absolute() {
        bin_path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| WorkerError::Spawn(format!("cwd: {e}")))?
            .join(bin_path)
    };
    let canonical = abs
        .canonicalize()
        .map_err(|e| WorkerError::Spawn(format!("cannot resolve {mcp_host_bin}: {e}")))?;
    let parent = canonical
        .parent()
        .ok_or_else(|| WorkerError::Spawn("binary has no parent directory".into()))?;
    Ok(parent.to_string_lossy().into_owned())
}

async fn spawn_landlock(
    allowed_paths: &[PathAccess],
    network: &NetworkPolicy,
    mcp_host_bin: &str,
    workspace_root_override: Option<&Path>,
) -> Result<Worker, WorkerError> {
    let workspace_root_path = resolve_workspace_root(allowed_paths, workspace_root_override)?;
    let workspace_root = workspace_root_path.to_string_lossy().into_owned();

    let bin_dir = resolve_bin_dir(mcp_host_bin)?;

    if matches!(network, NetworkPolicy::AllowList { .. }) {
        warn!(
            "landlock cannot do per-host network filtering; \
             AllowList will be treated as Isolated"
        );
    }

    info!(
        %workspace_root,
        %bin_dir,
        paths = allowed_paths.len(),
        "spawning landlock-sandboxed worker"
    );

    let allowed_paths_owned: Vec<PathAccess> = allowed_paths.to_vec();
    let network_owned = network.clone();

    // socketpair: (parent_fd, child_fd). Both ends without CLOEXEC
    // initially; we set CLOEXEC on the parent end immediately so
    // subsequent worker spawns don't accidentally inherit it. The
    // child end gets dup2'd onto FD 3 in pre_exec, which produces a
    // fresh FD without CLOEXEC — that's the one the child reads.
    let (parent_fd, child_fd) = make_socketpair()?;
    let child_fd_raw = child_fd.as_raw_fd();

    let mut cmd = Command::new(mcp_host_bin);
    cmd.args(["--workspace-root", &workspace_root]);
    cmd.stdin(std::process::Stdio::null());
    // Worker emits no stdout in normal operation (logs go to stderr);
    // null'ing keeps a runaway println from blocking on a backed-up
    // pipe.
    cmd.stdout(std::process::Stdio::null());
    cmd.stderr(std::process::Stdio::piped());
    cmd.kill_on_drop(true);

    // SAFETY: `pre_exec` runs in the forked child between fork and
    // exec. The closure calls only async-signal-safe libc functions
    // (dup2) plus the landlock ruleset application. `child_fd_raw`
    // is the pre-fork value; valid in the child until we exec.
    unsafe {
        cmd.pre_exec(move || {
            // Place the socket at FD 3. dup2 produces a new FD
            // without CLOEXEC, so it survives the upcoming exec.
            if libc::dup2(child_fd_raw, IPC_FD) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            apply_landlock(&allowed_paths_owned, &network_owned, &bin_dir).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::PermissionDenied, e.to_string())
            })?;
            Ok(())
        });
    }

    let mut child = cmd.spawn().map_err(|e| WorkerError::Spawn(e.to_string()))?;

    // Parent's copy of the child end is no longer needed — the
    // child has its own FD 3 via dup2. Drop closes our handle so
    // the kernel's refcount drops to one (the child); the socket
    // dies cleanly when the child exits.
    drop(child_fd);

    // Stderr tail capture: same shape as the v1 daemon. The drain
    // task lives until the pipe closes, which happens on child
    // exit.
    let stderr_tail: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
    if let Some(pipe) = child.stderr.take() {
        let tail = stderr_tail.clone();
        tokio::spawn(drain_stderr(pipe, tail));
    }

    // Convert the parent end into a tokio UnixStream. set_nonblocking
    // is mandatory before from_std — tokio refuses to register a
    // blocking FD with the reactor.
    let std_socket = std::os::unix::net::UnixStream::from(parent_fd);
    std_socket
        .set_nonblocking(true)
        .map_err(|e| WorkerError::Spawn(format!("set_nonblocking on parent socket: {e}")))?;
    let socket = UnixStream::from_std(std_socket)
        .map_err(|e| WorkerError::Spawn(format!("tokio::UnixStream::from_std: {e}")))?;
    let (mut reader, writer) = socket.into_split();

    // Wait for Hello, racing the child's exit so a worker that
    // crashes during startup surfaces as ChildExited (with the
    // captured stderr tail) rather than StartupTimeout.
    let hello_outcome = tokio::select! {
        biased;
        frame = read_frame(&mut reader) => HelloOutcome::Frame(frame),
        status = child.wait() => HelloOutcome::Exited(status.ok().and_then(|s| s.code())),
        _ = tokio::time::sleep(Duration::from_secs(READY_TIMEOUT_SECS)) => HelloOutcome::Timeout,
    };

    let tools = match hello_outcome {
        HelloOutcome::Frame(Ok(Some(WorkerFrame::Hello {
            protocol_version,
            tools,
            ..
        }))) => {
            if protocol_version != WORKER_PROTOCOL_VERSION {
                let _ = child.kill().await;
                return Err(WorkerError::ProtocolMismatch {
                    worker: protocol_version,
                    daemon: WORKER_PROTOCOL_VERSION,
                    stderr_tail: read_tail(&stderr_tail).await,
                });
            }
            tools
        }
        HelloOutcome::Frame(Ok(Some(other))) => {
            let _ = child.kill().await;
            return Err(WorkerError::Spawn(format!(
                "expected Hello, got {}",
                frame_kind(&other)
            )));
        }
        HelloOutcome::Frame(Ok(None)) | HelloOutcome::Exited(_) => {
            // Reader EOF or child exit. Capture exit code if we have
            // it; otherwise try to wait briefly for one.
            let code = match hello_outcome {
                HelloOutcome::Exited(c) => c,
                _ => child.try_wait().ok().flatten().and_then(|s| s.code()),
            };
            return Err(WorkerError::ChildExited {
                code,
                stderr_tail: read_tail(&stderr_tail).await,
            });
        }
        HelloOutcome::Frame(Err(e)) => {
            let _ = child.kill().await;
            return Err(WorkerError::Spawn(format!("read Hello: {e}")));
        }
        HelloOutcome::Timeout => {
            let _ = child.kill().await;
            return Err(WorkerError::StartupTimeout {
                seconds: READY_TIMEOUT_SECS,
                stderr_tail: read_tail(&stderr_tail).await,
            });
        }
    };

    info!(
        worker_protocol_version = WORKER_PROTOCOL_VERSION,
        tool_count = tools.len(),
        "worker handshake complete"
    );

    let (cmd_tx, cmd_rx) = mpsc::channel::<WorkerCommand>(CMD_CHANNEL_BOUND);
    tokio::spawn(frame_loop(reader, writer, cmd_rx));

    Ok(Worker {
        child,
        tools,
        cmd_tx,
    })
}

enum HelloOutcome {
    Frame(anyhow::Result<Option<WorkerFrame>>),
    Exited(Option<i32>),
    Timeout,
}

/// Long-lived task that owns both halves of the IPC socket plus the
/// inbound command channel. Routes `ToolFinal`s back to per-call
/// oneshots; drops `ToolChunk`s (matches the daemon's existing
/// scheduler-bound behavior — phase 2b doesn't surface streaming
/// chunks). Resolves all pending oneshots to `Disconnected` and
/// exits when either the read side EOFs or the command channel
/// closes.
async fn frame_loop(
    mut reader: OwnedReadHalf,
    mut writer: OwnedWriteHalf,
    mut cmd_rx: mpsc::Receiver<WorkerCommand>,
) {
    let mut pending: HashMap<CallId, oneshot::Sender<Result<CallToolResult, WorkerError>>> =
        HashMap::new();
    loop {
        tokio::select! {
            cmd = cmd_rx.recv() => match cmd {
                Some(WorkerCommand::Invoke { call_id, tool_name, arguments, attachments, result_tx }) => {
                    let frame = WorkerFrame::InvokeTool { call_id, tool_name, arguments, attachments };
                    if write_frame(&mut writer, &frame).await.is_err() {
                        let _ = result_tx.send(Err(WorkerError::Disconnected));
                        break;
                    }
                    if let Some(prev) = pending.insert(call_id, result_tx) {
                        // Daemon side picked a duplicate call_id.
                        // Unblock the prior caller so they don't hang.
                        warn!(%call_id, "duplicate call_id — prior caller will see Disconnected");
                        let _ = prev.send(Err(WorkerError::Disconnected));
                    }
                }
                Some(WorkerCommand::Cancel { call_id }) => {
                    let frame = WorkerFrame::CancelCall { call_id };
                    if write_frame(&mut writer, &frame).await.is_err() {
                        break;
                    }
                }
                None => {
                    // Worker handle dropped; we're done.
                    break;
                }
            },
            frame = read_frame(&mut reader) => match frame {
                Ok(Some(WorkerFrame::ToolFinal { call_id, result })) => {
                    if let Some(tx) = pending.remove(&call_id) {
                        let _ = tx.send(Ok(result));
                    } else {
                        warn!(%call_id, "ToolFinal for unknown call_id — dropping");
                    }
                }
                Ok(Some(WorkerFrame::ToolChunk { .. })) => {
                    // Streaming chunks aren't surfaced by the daemon
                    // today; the model still gets the full result via
                    // the eventual ToolFinal.
                }
                Ok(Some(other)) => {
                    warn!(kind = frame_kind(&other), "unexpected daemon-bound frame from worker — ignoring");
                }
                Ok(None) => {
                    // Worker hung up cleanly. Resolve all pending as
                    // Disconnected so callers don't hang waiting for a
                    // ToolFinal that won't arrive.
                    break;
                }
                Err(e) => {
                    warn!(error = %e, "worker frame read failed");
                    break;
                }
            },
        }
    }
    for (_, tx) in pending.drain() {
        let _ = tx.send(Err(WorkerError::Disconnected));
    }
}

fn make_socketpair() -> Result<(OwnedFd, OwnedFd), WorkerError> {
    let mut fds = [0i32; 2];
    // SAFETY: passing a 2-element array as required; libc writes the
    // two FDs into it on success. No unsafety beyond the FFI call
    // shape.
    let rc = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) };
    if rc < 0 {
        return Err(WorkerError::Spawn(format!(
            "socketpair: {}",
            std::io::Error::last_os_error()
        )));
    }
    // SAFETY: socketpair returned success, fds[0] and fds[1] are
    // freshly-allocated FDs we now own.
    let parent = unsafe { OwnedFd::from_raw_fd(fds[0]) };
    let child = unsafe { OwnedFd::from_raw_fd(fds[1]) };
    // Set CLOEXEC on the parent end so subsequent worker spawns in
    // this process don't accidentally inherit it. The child end
    // doesn't matter — it dies with this OwnedFd as soon as we drop
    // it after pre_exec runs.
    set_cloexec(parent.as_raw_fd())?;
    Ok((parent, child))
}

fn set_cloexec(fd: i32) -> Result<(), WorkerError> {
    // SAFETY: F_GETFD/F_SETFD are read/write of the FD's flag word;
    // safe to call against any open FD we own.
    unsafe {
        let flags = libc::fcntl(fd, libc::F_GETFD);
        if flags < 0 {
            return Err(WorkerError::Spawn(format!(
                "F_GETFD: {}",
                std::io::Error::last_os_error()
            )));
        }
        if libc::fcntl(fd, libc::F_SETFD, flags | libc::FD_CLOEXEC) < 0 {
            return Err(WorkerError::Spawn(format!(
                "F_SETFD: {}",
                std::io::Error::last_os_error()
            )));
        }
    }
    Ok(())
}

async fn drain_stderr(mut pipe: tokio::process::ChildStderr, tail: Arc<Mutex<Vec<u8>>>) {
    let mut real = tokio::io::stderr();
    let mut chunk = [0u8; 4096];
    loop {
        match pipe.read(&mut chunk).await {
            Ok(0) => break,
            Ok(n) => {
                let bytes = &chunk[..n];
                let _ = real.write_all(bytes).await;
                let _ = real.flush().await;
                let mut buf = tail.lock().await;
                buf.extend_from_slice(bytes);
                if buf.len() > STDERR_TAIL_BYTES {
                    let drop_n = buf.len() - STDERR_TAIL_BYTES;
                    buf.drain(..drop_n);
                }
            }
            Err(_) => break,
        }
    }
}

async fn read_tail(buf: &Arc<Mutex<Vec<u8>>>) -> String {
    let b = buf.lock().await;
    String::from_utf8_lossy(&b).trim().to_string()
}

/// Length-prefixed write: 4-byte big-endian payload length, then the
/// CBOR-encoded frame. Mirrors the worker side in
/// `whisper-agent-mcp-host/src/ipc.rs`.
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

/// Length-prefixed read. `Ok(None)` on clean EOF, `Err` on partial
/// frame or codec failure.
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

/// Apply the Landlock ruleset to the current process. Adds the spec's
/// `allowed_paths` *on top of* a fixed set of implicit grants the
/// daemon needs every worker to have:
///
/// - `/usr`, `/lib`, `/lib64`, `/bin`, `/sbin` read-only — shared libs
///   and standard binaries the worker (and any subprocesses, e.g. the
///   `bash` tool's `/bin/sh`) link against and exec.
/// - `/etc` read-only — libc nss/resolver config (`ld.so.cache`,
///   `resolv.conf`, `nsswitch.conf`, `passwd` for `getpwuid`).
/// - `/proc` read-only — many tools (cargo, rustc, ps) require it.
/// - `/dev/null`, `/dev/zero` **read+write** — character devices with
///   silent writes; granting write so shell idioms (`cmd 2>/dev/null`,
///   `cmd >/dev/null`) and `open(O_RDWR)` for mmap-based zero-init
///   work. Bytes go nowhere.
/// - `/dev/urandom` read-only — read for entropy; writes (which would
///   add entropy and require CAP_SYS_ADMIN anyway) stay denied.
/// - `/tmp` **read+write** — convenience for build temp files (cargo,
///   rustc, gcc). NB: this means a thread can read/write *any* file
///   under `/tmp`, not just its own workspace if the workspace happens
///   to live under `/tmp`.
/// - `bin_dir` read-only — directory containing the `mcp-host` binary
///   so exec works.
///
/// These grants are not surfaced in `pod.toml` and not visible to the
/// model. Operators relying on `allowed_paths` as the *complete* set
/// of accessible filesystem locations should know about them — see
/// `docs/design_permissions.md` (Sandboxing) and the webui's
/// "Allowed paths" hint, both of which mirror this list. If you change
/// the implicit set, update those surfaces too.
fn apply_landlock(
    allowed_paths: &[PathAccess],
    network: &NetworkPolicy,
    bin_dir: &str,
) -> Result<(), landlock::RulesetError> {
    use landlock::{
        ABI, Access, AccessFs, AccessNet, Ruleset, RulesetAttr, RulesetCreatedAttr,
        path_beneath_rules,
    };

    let abi = ABI::V6;

    let mut ruleset = Ruleset::default().handle_access(AccessFs::from_all(abi))?;

    if !matches!(network, NetworkPolicy::Unrestricted) {
        ruleset = ruleset.handle_access(AccessNet::BindTcp | AccessNet::ConnectTcp)?;
    }

    let mut created = ruleset.create()?;

    // Implicit grants — keep in sync with the docstring above and the
    // mirrors in design_permissions.md and editor_render.rs.
    created = created.add_rules(path_beneath_rules(
        ["/usr", "/lib", "/lib64", "/etc", "/bin", "/sbin"],
        AccessFs::from_read(abi),
    ))?;

    // /dev/null + /dev/zero need write access — shell `2>/dev/null`
    // redirects open the file with O_WRONLY, and mmap-based zero-init
    // opens /dev/zero O_RDWR. Read-only blocks both. /dev/urandom stays
    // read-only (writes add entropy via a privileged ioctl anyway).
    created = created.add_rules(path_beneath_rules(
        ["/dev/null", "/dev/zero"],
        AccessFs::from_all(abi),
    ))?;
    created = created.add_rules(path_beneath_rules(
        ["/dev/urandom"],
        AccessFs::from_read(abi),
    ))?;

    created = created.add_rules(path_beneath_rules(["/proc"], AccessFs::from_read(abi)))?;
    created = created.add_rules(path_beneath_rules(["/tmp"], AccessFs::from_all(abi)))?;
    created = created.add_rules(path_beneath_rules([bin_dir], AccessFs::from_read(abi)))?;

    for pa in allowed_paths {
        let access = match pa.mode {
            AccessMode::ReadOnly => AccessFs::from_read(abi),
            AccessMode::ReadWrite => AccessFs::from_all(abi),
        };
        created = created.add_rules(path_beneath_rules([pa.path.as_str()], access))?;
    }

    let _status = created.restrict_self()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_workspace_root_errors_when_override_missing() {
        // The daemon no longer guesses; the scheduler is responsible
        // for setting workspace_root explicitly via ThreadContext.
        let paths = vec![
            PathAccess::read_only("/lib"),
            PathAccess::read_write("/work/a"),
        ];
        assert!(matches!(
            resolve_workspace_root(&paths, None),
            Err(WorkerError::NoWorkspaceRoot)
        ));
    }

    #[test]
    fn resolve_workspace_root_errors_when_override_missing_and_no_rw_path() {
        let paths = vec![PathAccess::read_only("/lib")];
        assert!(matches!(
            resolve_workspace_root(&paths, None),
            Err(WorkerError::NoWorkspaceRoot)
        ));
    }

    #[test]
    fn resolve_workspace_root_accepts_override_matching_rw_path() {
        let paths = vec![PathAccess::read_write("/work/a")];
        let override_path = Path::new("/work/a");
        assert_eq!(
            resolve_workspace_root(&paths, Some(override_path)).unwrap(),
            PathBuf::from("/work/a")
        );
    }

    #[test]
    fn resolve_workspace_root_accepts_override_under_rw_path() {
        // Landlock grants access below the granted node, so a sub-path
        // of an RW root is itself writable — accept it as a workspace.
        let paths = vec![PathAccess::read_write("/work")];
        let override_path = Path::new("/work/projects/foo");
        assert_eq!(
            resolve_workspace_root(&paths, Some(override_path)).unwrap(),
            PathBuf::from("/work/projects/foo")
        );
    }

    #[test]
    fn resolve_workspace_root_rejects_override_outside_rw_paths() {
        let paths = vec![
            PathAccess::read_write("/work"),
            PathAccess::read_only("/lib"),
        ];
        let override_path = Path::new("/lib/somewhere");
        match resolve_workspace_root(&paths, Some(override_path)) {
            Err(WorkerError::WorkspaceRootNotInSpec { override_path: p }) => {
                assert_eq!(p, "/lib/somewhere");
            }
            other => panic!("expected WorkspaceRootNotInSpec, got {other:?}"),
        }
    }

    /// Smoke-test the IPC round-trip without spawning a real worker:
    /// stand up an in-process `(reader, writer)` pair, hand it to
    /// `frame_loop`, drive an Invoke through the public `cmd_tx`,
    /// and assert the worker (us, in the test) sees `InvokeTool` and
    /// the daemon side sees the synthesized `ToolFinal`.
    #[tokio::test]
    async fn frame_loop_routes_invoke_to_final() {
        let (daemon_end, worker_end) = UnixStream::pair().expect("socketpair");
        let (d_reader, d_writer) = daemon_end.into_split();
        let (mut w_reader, mut w_writer) = worker_end.into_split();

        let (cmd_tx, cmd_rx) = mpsc::channel::<WorkerCommand>(8);
        let loop_handle = tokio::spawn(frame_loop(d_reader, d_writer, cmd_rx));

        // Daemon-side: send an invoke
        let (result_tx, result_rx) = oneshot::channel();
        cmd_tx
            .send(WorkerCommand::Invoke {
                call_id: CallId(7),
                tool_name: "read_file".into(),
                arguments: serde_json::json!({"path": "/tmp/x"}),
                attachments: vec![],
                result_tx,
            })
            .await
            .expect("send invoke");

        // Worker-side: read the InvokeTool, send back a ToolFinal.
        let frame = read_frame(&mut w_reader)
            .await
            .expect("read")
            .expect("some");
        match frame {
            WorkerFrame::InvokeTool {
                call_id, tool_name, ..
            } => {
                assert_eq!(call_id, CallId(7));
                assert_eq!(tool_name, "read_file");
            }
            other => panic!("expected InvokeTool, got {other:?}"),
        }
        write_frame(
            &mut w_writer,
            &WorkerFrame::ToolFinal {
                call_id: CallId(7),
                result: CallToolResult::text("hello"),
            },
        )
        .await
        .expect("write final");

        // Daemon-side: invoke result delivers
        let r = result_rx.await.expect("oneshot").expect("ok");
        assert_eq!(r.content.len(), 1);

        drop(cmd_tx);
        drop(w_writer);
        drop(w_reader);
        let _ = loop_handle.await;
    }

    /// When the worker side of the socket drops with calls in-flight,
    /// each pending caller resolves to `Disconnected` rather than
    /// hanging forever.
    #[tokio::test]
    async fn frame_loop_resolves_pending_as_disconnected_on_eof() {
        let (daemon_end, worker_end) = UnixStream::pair().expect("socketpair");
        let (d_reader, d_writer) = daemon_end.into_split();
        let (mut w_reader, _w_writer) = worker_end.into_split();

        let (cmd_tx, cmd_rx) = mpsc::channel::<WorkerCommand>(8);
        let loop_handle = tokio::spawn(frame_loop(d_reader, d_writer, cmd_rx));

        let (result_tx, result_rx) = oneshot::channel();
        cmd_tx
            .send(WorkerCommand::Invoke {
                call_id: CallId(1),
                tool_name: "x".into(),
                arguments: serde_json::Value::Null,
                attachments: vec![],
                result_tx,
            })
            .await
            .expect("send invoke");

        // Worker-side: read the frame so the daemon's write completes,
        // then drop the read half — this closes the daemon's read
        // side and triggers the EOF path.
        let _ = read_frame(&mut w_reader).await;
        drop(w_reader);
        // Drop the worker write half too, fully closing the socket.
        drop(_w_writer);

        let r = result_rx.await.expect("oneshot");
        assert!(matches!(r, Err(WorkerError::Disconnected)));

        drop(cmd_tx);
        let _ = loop_handle.await;
    }
}
