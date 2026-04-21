//! Sandbox provisioning backends.
//!
//! Each backend takes a [`HostEnvSpec`] variant and produces a running MCP host
//! process inside the appropriate isolation boundary.

use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use rand::RngCore;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tracing::info;
use whisper_agent_protocol::sandbox::{AccessMode, HostEnvSpec, NetworkPolicy, PathAccess};

/// How much of the child's stderr we retain and surface to callers
/// when it dies or times out. 4 KiB is enough to catch a typical
/// `anyhow!` chain without letting a chatty child bloat memory.
const STDERR_TAIL_BYTES: usize = 4096;

/// How long we wait for the spawned mcp-host to print its
/// `listening <addr>` handshake line. The child has to bind a TCP
/// listener and flush one line — normally ~milliseconds. A second
/// daemon running on the same host can slow things down; 10 s is
/// comfortable while still catching a wedged child.
const READY_TIMEOUT_SECS: u64 = 10;

pub struct ProvisionedSession {
    pub child: Child,
    pub mcp_url: String,
    /// Per-sandbox bearer the scheduler must present on every `/mcp`
    /// request. Generated per provision (never reused), passed to the
    /// MCP host once via stdin, then returned in `ProvisionResponse`.
    pub mcp_token: String,
}

/// Generate a fresh 256-bit random token, hex-encoded. One token per
/// provision — never stored, never logged, only handed to the one
/// scheduler that requested this sandbox.
fn generate_mcp_token() -> String {
    // `rand::rng()` is a CSPRNG seeded from OS entropy — fine for
    // ephemeral session secrets and doesn't drag the app into the
    // fallible `TryRngCore` API that `OsRng` now requires in rand 0.9.
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    let mut s = String::with_capacity(64);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(&mut s, "{b:02x}");
    }
    s
}

#[derive(Debug, thiserror::Error)]
pub enum ProvisionError {
    #[error("no read-write path in spec — cannot determine workspace root")]
    NoWorkspaceRoot,
    #[error("spawn failed: {0}")]
    Spawn(String),
    /// MCP host exited before it bound its listen socket. `code` is the
    /// process exit code (`None` for signal-killed). `stderr_tail` is
    /// everything the child printed to stderr up to the point of
    /// failure (truncated to the last `STDERR_TAIL_BYTES`) so the
    /// daemon's 500 response surfaces the real cause instead of a
    /// generic timeout.
    #[error(
        "MCP host exited before becoming ready (exit_code={code:?}){}",
        format_stderr_suffix(stderr_tail)
    )]
    ChildExited {
        code: Option<i32>,
        stderr_tail: String,
    },
    /// Deadline hit before child bound its listen socket. Treated as a
    /// distinct failure mode from `ChildExited` because the child is
    /// still alive (we kill it) and the cause may be external
    /// (blocking syscall, wedged filesystem) rather than a clean error
    /// path. `stderr_tail` carries whatever the child wrote in the
    /// interim in case it's still informative.
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

pub async fn provision(
    spec: &HostEnvSpec,
    mcp_host_bin: &str,
    bind_ip: IpAddr,
) -> Result<ProvisionedSession, ProvisionError> {
    match spec {
        HostEnvSpec::Landlock {
            allowed_paths,
            network,
        } => provision_landlock(allowed_paths, network, mcp_host_bin, bind_ip).await,
        HostEnvSpec::Container { .. } => Err(ProvisionError::Unsupported(
            "container provisioning not yet implemented".into(),
        )),
    }
}

/// Resolve `mcp_host_bin` to an absolute path and return its parent directory.
/// The parent needs read+execute access in the landlock ruleset so the binary
/// can be loaded after exec.
fn resolve_bin_dir(mcp_host_bin: &str) -> Result<String, ProvisionError> {
    let bin_path = std::path::Path::new(mcp_host_bin);
    let abs = if bin_path.is_absolute() {
        bin_path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| ProvisionError::Spawn(format!("cwd: {e}")))?
            .join(bin_path)
    };
    let canonical = abs
        .canonicalize()
        .map_err(|e| ProvisionError::Spawn(format!("cannot resolve {mcp_host_bin}: {e}")))?;
    let parent = canonical
        .parent()
        .ok_or_else(|| ProvisionError::Spawn("binary has no parent directory".into()))?;
    Ok(parent.to_string_lossy().into_owned())
}

async fn provision_landlock(
    allowed_paths: &[PathAccess],
    network: &NetworkPolicy,
    mcp_host_bin: &str,
    bind_ip: IpAddr,
) -> Result<ProvisionedSession, ProvisionError> {
    let workspace_root = allowed_paths
        .iter()
        .find(|p| p.mode == AccessMode::ReadWrite)
        .map(|p| p.path.clone())
        .ok_or(ProvisionError::NoWorkspaceRoot)?;

    let bin_dir = resolve_bin_dir(mcp_host_bin)?;

    // Ask the kernel to pick a free port (port 0). The child binds,
    // then prints `listening <addr>` on stdout; we read that line to
    // learn the actual port. This eliminates the whole class of
    // "another daemon got there first" races (which were real:
    // different sandbox daemons on the same host could otherwise hand
    // the scheduler a URL pointing at each other's mcp-host).
    //
    // `bind_ip` still controls the interface family: loopback-only
    // daemons produce loopback-only children, `[::]`/`0.0.0.0`
    // daemons produce wildcard-bound children. The port is the only
    // thing the kernel picks.
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
    // stdin is piped so we can hand the MCP host its per-sandbox bearer
    // without exposing it in argv or the environment (both readable via
    // /proc/<pid>/... under the same uid).
    cmd.stdin(std::process::Stdio::piped());
    // stdout is piped so we can read the child's `listening <addr>`
    // handshake line after it binds. The child's own tracing logs go
    // to stderr (drained below), so stdout carries only this one
    // structured handshake line.
    cmd.stdout(std::process::Stdio::piped());
    // stderr is piped so we can grab a tail on startup failure. The
    // spawned drainer below tees to the daemon's own stderr so
    // journalctl still shows the child's output at runtime.
    cmd.stderr(std::process::Stdio::piped());
    cmd.kill_on_drop(true);

    unsafe {
        cmd.pre_exec(move || {
            apply_landlock(&allowed_paths_owned, &network_owned, &bin_dir).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::PermissionDenied, e.to_string())
            })
        });
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| ProvisionError::Spawn(e.to_string()))?;

    // Write the token + newline and close stdin so the child's
    // `read_line` returns. Dropping the handle is what closes the pipe.
    {
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| ProvisionError::Spawn("child stdin not captured".into()))?;
        stdin
            .write_all(mcp_token.as_bytes())
            .await
            .map_err(|e| ProvisionError::Spawn(format!("writing token to child stdin: {e}")))?;
        stdin
            .write_all(b"\n")
            .await
            .map_err(|e| ProvisionError::Spawn(format!("writing token to child stdin: {e}")))?;
        stdin
            .shutdown()
            .await
            .map_err(|e| ProvisionError::Spawn(format!("closing child stdin: {e}")))?;
    }

    // Start draining stderr in the background: tee to the daemon's own
    // stderr (so journalctl keeps showing live child output) and also
    // retain the last STDERR_TAIL_BYTES in a shared buffer for error
    // reporting. Drops naturally when the child exits and closes the
    // pipe.
    let stderr_tail: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
    if let Some(pipe) = child.stderr.take() {
        let tail = stderr_tail.clone();
        tokio::spawn(drain_stderr(pipe, tail));
    }

    let stdout_pipe = child
        .stdout
        .take()
        .ok_or_else(|| ProvisionError::Spawn("child stdout not captured".into()))?;
    match wait_for_listening_line(stdout_pipe, READY_TIMEOUT_SECS, &mut child).await {
        ListeningOutcome::Bound { addr } => {
            let mcp_url = format!("http://{addr}/mcp");
            info!(%mcp_url, "MCP host ready");
            Ok(ProvisionedSession {
                child,
                mcp_url,
                mcp_token,
            })
        }
        ListeningOutcome::Exited { code } => Err(ProvisionError::ChildExited {
            code,
            stderr_tail: read_tail(&stderr_tail).await,
        }),
        ListeningOutcome::Timeout { seconds } => {
            let _ = child.kill().await;
            Err(ProvisionError::StartupTimeout {
                seconds,
                stderr_tail: read_tail(&stderr_tail).await,
            })
        }
        ListeningOutcome::BadHandshake { detail } => {
            // Child exited or closed stdout without emitting a
            // well-formed "listening <addr>" line. Surface the stderr
            // tail — the real reason (bind failure, panic) lives there.
            let _ = child.kill().await;
            let tail = read_tail(&stderr_tail).await;
            let joined = if tail.is_empty() {
                detail
            } else {
                format!("{detail}: {tail}")
            };
            Err(ProvisionError::Spawn(joined))
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
                // Tee to the daemon's own stderr: journalctl keeps the
                // user-visible stream, even after startup succeeds.
                let _ = real.write_all(bytes).await;
                let _ = real.flush().await;
                // Retain a bounded tail for error reporting.
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

    // Base system paths — read + execute so the MCP host, cargo, etc. can run.
    created = created.add_rules(path_beneath_rules(
        ["/usr", "/lib", "/lib64", "/etc", "/bin", "/sbin"],
        AccessFs::from_read(abi),
    ))?;

    // Device nodes the child needs.
    created = created.add_rules(path_beneath_rules(
        ["/dev/null", "/dev/urandom", "/dev/zero"],
        AccessFs::from_read(abi),
    ))?;

    // /proc read-only (cargo, rustc, many tools need it).
    created = created.add_rules(path_beneath_rules(["/proc"], AccessFs::from_read(abi)))?;

    // /tmp read-write (compilation tempfiles, etc.).
    created = created.add_rules(path_beneath_rules(["/tmp"], AccessFs::from_all(abi)))?;

    // The MCP host binary's directory — read+execute so exec() works.
    created = created.add_rules(path_beneath_rules([bin_dir], AccessFs::from_read(abi)))?;

    // User-specified paths from the spec.
    for pa in allowed_paths {
        let access = match pa.mode {
            AccessMode::ReadOnly => AccessFs::from_read(abi),
            AccessMode::ReadWrite => AccessFs::from_all(abi),
        };
        created = created.add_rules(path_beneath_rules([pa.path.as_str()], access))?;
    }

    // For Isolated: we handled net rights above but added no NetPort rules,
    // so all TCP is denied. AllowList can't do per-host filtering with
    // landlock (only port-based), so treat it as Isolated with a warning.
    if matches!(network, NetworkPolicy::AllowList { .. }) {
        // Log in parent context would be better, but we're in pre_exec.
        // The daemon handler logs a warning before spawning.
    }

    let status = created.restrict_self()?;
    // RulesetStatus tells us if landlock was actually enforced.
    let _ = status;

    Ok(())
}

enum ListeningOutcome {
    /// Child emitted a well-formed `listening <addr>` handshake line.
    /// `addr` is the parsed socket address — use this to build the
    /// mcp_url. The child is still running; caller owns it.
    Bound { addr: SocketAddr },
    /// Child exited before emitting a handshake. Surface with the
    /// stderr tail so the operator sees the real cause (bind failure,
    /// panic) rather than a generic timeout.
    Exited { code: Option<i32> },
    /// Deadline hit; no handshake line. Child is still running;
    /// caller is expected to kill it.
    Timeout { seconds: u64 },
    /// Child's stdout closed or emitted a malformed line without the
    /// child itself exiting with an observable code. Caller kills it
    /// and formats a Spawn error with this detail + stderr tail.
    BadHandshake { detail: String },
}

/// Read the child's stdout until one of:
/// - a well-formed `listening <addr>` line arrives (happy path)
/// - child exits (surface exit code)
/// - `timeout_secs` elapses (surface timeout)
/// - stdout closes without a handshake line (malformed: some non-
///   emitting panic or a version-skewed child)
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

/// Parse the one-line handshake the mcp-host prints after binding:
/// `listening 127.0.0.1:12345` (or `[::1]:12345` for IPv6). Returns
/// `None` if the shape doesn't match — caller treats that as a
/// malformed handshake.
fn parse_listening_line(line: &str) -> Option<SocketAddr> {
    let rest = line.strip_prefix("listening ")?;
    rest.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn child_exited_error_includes_stderr_tail() {
        let err = ProvisionError::ChildExited {
            code: Some(1),
            stderr_tail: "Error: invalid workspace root \"/tmp/xyz\"".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("exit_code=Some(1)"), "got: {msg}");
        assert!(msg.contains("invalid workspace root"), "got: {msg}");
    }

    #[test]
    fn child_exited_without_stderr_has_no_dangling_colon() {
        let err = ProvisionError::ChildExited {
            code: Some(137),
            stderr_tail: String::new(),
        };
        let msg = err.to_string();
        assert!(!msg.ends_with(": "), "got: {msg}");
        assert!(!msg.ends_with(':'), "got: {msg}");
    }

    #[test]
    fn startup_timeout_includes_partial_stderr() {
        let err = ProvisionError::StartupTimeout {
            seconds: 10,
            stderr_tail: "warning: still initializing".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("within 10s"), "got: {msg}");
        assert!(msg.contains("still initializing"), "got: {msg}");
    }

    #[test]
    fn startup_timeout_without_stderr_is_clean() {
        let err = ProvisionError::StartupTimeout {
            seconds: 10,
            stderr_tail: String::new(),
        };
        assert_eq!(
            err.to_string(),
            "MCP host failed to become ready within 10s"
        );
    }

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
}
