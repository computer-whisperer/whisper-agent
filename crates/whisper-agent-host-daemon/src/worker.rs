//! Per-session worker spawn.
//!
//! Lifted from `whisper-agent-sandbox/src/provision.rs` largely intact.
//! Daemon-side code that takes a [`HostEnvSpec::Landlock`] and produces
//! a running `whisper-agent-mcp-host` child bound to a loopback port,
//! returning the URL and the per-session bearer the daemon will present
//! when proxying tool calls.
//!
//! Container provisioning still returns [`WorkerError::Unsupported`] —
//! same status as in the v1 sandbox crate. Adding it does not require
//! protocol changes; `DaemonCapabilities::spec_kinds` already lets the
//! daemon advertise `Container` only when this module knows how to
//! fulfill it.

use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use rand::Rng;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tracing::{info, warn};
use whisper_agent_protocol::sandbox::{AccessMode, HostEnvSpec, NetworkPolicy, PathAccess};

/// Last N stderr bytes we retain for failure reporting. 4 KiB catches
/// a typical `anyhow!` chain without letting a chatty child bloat
/// memory.
const STDERR_TAIL_BYTES: usize = 4096;

/// How long we wait for the child's `listening <addr>` handshake.
/// Bind + flush is normally milliseconds; 10 s is comfortable while
/// still catching a wedged child.
const READY_TIMEOUT_SECS: u64 = 10;

/// A running worker child.
///
/// The daemon owns this for the session's lifetime; `Drop` on the
/// inner [`Child`] kills the process (`kill_on_drop(true)` set at
/// spawn).
pub struct Worker {
    pub child: Child,
    pub mcp_url: String,
    /// Per-session bearer presented on every `tools/call` HTTP request.
    /// Generated fresh per spawn, never logged, never reused.
    pub mcp_token: String,
}

#[derive(Debug, thiserror::Error)]
pub enum WorkerError {
    #[error("no read-write path in spec — cannot determine workspace root")]
    NoWorkspaceRoot,
    #[error(
        "workspace_root override `{override_path}` is not under any read-write path in the spec"
    )]
    WorkspaceRootNotInSpec { override_path: String },
    #[error("spawn failed: {0}")]
    Spawn(String),
    #[error(
        "MCP host exited before becoming ready (exit_code={code:?}){}",
        format_stderr_suffix(stderr_tail)
    )]
    ChildExited {
        code: Option<i32>,
        stderr_tail: String,
    },
    #[error(
        "MCP host failed to become ready within {seconds}s{}",
        format_stderr_suffix(stderr_tail)
    )]
    StartupTimeout { seconds: u64, stderr_tail: String },
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

/// Dispatch on spec kind. Today only [`HostEnvSpec::Landlock`] is
/// implemented; container support is a future item.
///
/// `workspace_root_override` lets the per-session
/// [`whisper_agent_host_proto::ThreadContext::workspace_root`] pick a
/// specific writable path from the spec when the spec advertises more
/// than one. `None` ⇒ fall back to the first read-write path in the
/// spec, preserving pre-5a behavior.
pub async fn spawn(
    spec: &HostEnvSpec,
    mcp_host_bin: &str,
    bind_ip: IpAddr,
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
                bind_ip,
                workspace_root_override,
            )
            .await
        }
        HostEnvSpec::Container { .. } => Err(WorkerError::Unsupported(
            "container provisioning not yet implemented".into(),
        )),
    }
}

/// Pick the workspace root for a session. If the caller supplies an
/// override, it must be one of the spec's read-write paths (or a
/// descendant of one — landlock allows access below the granted node,
/// so a sub-path of an RW root is itself RW). Otherwise fall back to
/// the first RW path in the spec.
pub(crate) fn resolve_workspace_root(
    allowed_paths: &[PathAccess],
    workspace_root_override: Option<&Path>,
) -> Result<PathBuf, WorkerError> {
    let rw_roots: Vec<&str> = allowed_paths
        .iter()
        .filter(|p| p.mode == AccessMode::ReadWrite)
        .map(|p| p.path.as_str())
        .collect();
    if rw_roots.is_empty() {
        return Err(WorkerError::NoWorkspaceRoot);
    }
    match workspace_root_override {
        None => Ok(PathBuf::from(rw_roots[0])),
        Some(override_path) => {
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
    }
}

fn generate_mcp_token() -> String {
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    let mut s = String::with_capacity(64);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(&mut s, "{b:02x}");
    }
    s
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
    bind_ip: IpAddr,
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

    // Port 0 lets the kernel pick. Child binds, then prints
    // `listening <addr>` on stdout; we read the line to learn the
    // actual port. Eliminates "another worker grabbed the port first"
    // races between concurrent sessions on the same daemon.
    let listen_addr = SocketAddr::new(bind_ip, 0).to_string();

    info!(
        %workspace_root,
        %listen_addr,
        %bin_dir,
        paths = allowed_paths.len(),
        "spawning landlock-sandboxed MCP host"
    );

    let allowed_paths_owned: Vec<PathAccess> = allowed_paths.to_vec();
    let network_owned = network.clone();

    let mcp_token = generate_mcp_token();

    let mut cmd = Command::new(mcp_host_bin);
    cmd.args([
        "--listen",
        &listen_addr,
        "--workspace-root",
        &workspace_root,
        "--token-stdin",
    ]);
    cmd.stdin(std::process::Stdio::piped());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());
    cmd.kill_on_drop(true);

    unsafe {
        cmd.pre_exec(move || {
            apply_landlock(&allowed_paths_owned, &network_owned, &bin_dir).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::PermissionDenied, e.to_string())
            })
        });
    }

    let mut child = cmd.spawn().map_err(|e| WorkerError::Spawn(e.to_string()))?;

    // Hand the per-session bearer over stdin and close — keeps it out
    // of argv and the env (both readable via /proc/<pid>/... under the
    // same uid).
    {
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| WorkerError::Spawn("child stdin not captured".into()))?;
        stdin
            .write_all(mcp_token.as_bytes())
            .await
            .map_err(|e| WorkerError::Spawn(format!("writing token to child stdin: {e}")))?;
        stdin
            .write_all(b"\n")
            .await
            .map_err(|e| WorkerError::Spawn(format!("writing token to child stdin: {e}")))?;
        stdin
            .shutdown()
            .await
            .map_err(|e| WorkerError::Spawn(format!("closing child stdin: {e}")))?;
    }

    let stderr_tail: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
    if let Some(pipe) = child.stderr.take() {
        let tail = stderr_tail.clone();
        tokio::spawn(drain_stderr(pipe, tail));
    }

    let stdout_pipe = child
        .stdout
        .take()
        .ok_or_else(|| WorkerError::Spawn("child stdout not captured".into()))?;
    match wait_for_listening_line(stdout_pipe, READY_TIMEOUT_SECS, &mut child).await {
        ListeningOutcome::Bound { addr } => {
            let mcp_url = format!("http://{addr}/mcp");
            info!(%mcp_url, "MCP host ready");
            Ok(Worker {
                child,
                mcp_url,
                mcp_token,
            })
        }
        ListeningOutcome::Exited { code } => Err(WorkerError::ChildExited {
            code,
            stderr_tail: read_tail(&stderr_tail).await,
        }),
        ListeningOutcome::Timeout { seconds } => {
            let _ = child.kill().await;
            Err(WorkerError::StartupTimeout {
                seconds,
                stderr_tail: read_tail(&stderr_tail).await,
            })
        }
        ListeningOutcome::BadHandshake { detail } => {
            let _ = child.kill().await;
            let tail = read_tail(&stderr_tail).await;
            let joined = if tail.is_empty() {
                detail
            } else {
                format!("{detail}: {tail}")
            };
            Err(WorkerError::Spawn(joined))
        }
    }
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

    created = created.add_rules(path_beneath_rules(
        ["/usr", "/lib", "/lib64", "/etc", "/bin", "/sbin"],
        AccessFs::from_read(abi),
    ))?;

    created = created.add_rules(path_beneath_rules(
        ["/dev/null", "/dev/urandom", "/dev/zero"],
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

enum ListeningOutcome {
    Bound { addr: SocketAddr },
    Exited { code: Option<i32> },
    Timeout { seconds: u64 },
    BadHandshake { detail: String },
}

async fn wait_for_listening_line(
    stdout: tokio::process::ChildStdout,
    timeout_secs: u64,
    child: &mut Child,
) -> ListeningOutcome {
    let read_line = async {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        match reader.read_line(&mut line).await {
            Ok(0) => ListeningOutcome::BadHandshake {
                detail: "child stdout closed before emitting `listening <addr>`".into(),
            },
            Ok(_) => {
                let trimmed = line.trim();
                match parse_listening_line(trimmed) {
                    Some(addr) => ListeningOutcome::Bound { addr },
                    None => ListeningOutcome::BadHandshake {
                        detail: format!("unexpected child handshake: {trimmed:?}"),
                    },
                }
            }
            Err(e) => ListeningOutcome::BadHandshake {
                detail: format!("reading child handshake: {e}"),
            },
        }
    };
    let exit_watch = async {
        let code = child.wait().await.ok().and_then(|s| s.code());
        ListeningOutcome::Exited { code }
    };
    let deadline = tokio::time::sleep(Duration::from_secs(timeout_secs));

    tokio::select! {
        outcome = read_line => outcome,
        outcome = exit_watch => outcome,
        _ = deadline => ListeningOutcome::Timeout { seconds: timeout_secs },
    }
}

fn parse_listening_line(line: &str) -> Option<SocketAddr> {
    let rest = line.strip_prefix("listening ")?;
    rest.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_listening_line_accepts_ipv4() {
        let a = parse_listening_line("listening 127.0.0.1:12345").unwrap();
        assert_eq!(a.to_string(), "127.0.0.1:12345");
    }

    #[test]
    fn parse_listening_line_accepts_ipv6() {
        let a = parse_listening_line("listening [::1]:42").unwrap();
        assert!(a.is_ipv6());
        assert_eq!(a.port(), 42);
    }

    #[test]
    fn parse_listening_line_rejects_missing_prefix() {
        assert!(parse_listening_line("127.0.0.1:8000").is_none());
    }

    #[test]
    fn parse_listening_line_rejects_garbage() {
        assert!(parse_listening_line("listening not-a-socket").is_none());
        assert!(parse_listening_line("listening 127.0.0.1").is_none());
    }

    #[test]
    fn resolve_workspace_root_uses_first_rw_when_no_override() {
        let paths = vec![
            PathAccess::read_only("/lib"),
            PathAccess::read_write("/work/a"),
            PathAccess::read_write("/work/b"),
        ];
        assert_eq!(
            resolve_workspace_root(&paths, None).unwrap(),
            PathBuf::from("/work/a")
        );
    }

    #[test]
    fn resolve_workspace_root_errors_when_no_rw_path() {
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
}
