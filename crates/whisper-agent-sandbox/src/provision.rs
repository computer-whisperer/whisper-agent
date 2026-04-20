//! Sandbox provisioning backends.
//!
//! Each backend takes a [`HostEnvSpec`] variant and produces a running MCP host
//! process inside the appropriate isolation boundary.

use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::sync::atomic::{AtomicU16, Ordering};
use std::time::Duration;

use rand::RngCore;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tracing::info;
use whisper_agent_protocol::sandbox::{AccessMode, HostEnvSpec, NetworkPolicy, PathAccess};

/// How much of the child's stderr we retain and surface to callers
/// when it dies or times out. 4 KiB is enough to catch a typical
/// `anyhow!` chain without letting a chatty child bloat memory.
const STDERR_TAIL_BYTES: usize = 4096;

/// Port range for spawned MCP host instances. Each provision bumps the
/// counter. Starts well above the sibling daemon ports (sandbox 9810/9820,
/// fetch 9830, search 9831) so the in-tree dev harness and the packaged
/// systemd service can coexist on one host without colliding.
static NEXT_PORT: AtomicU16 = AtomicU16::new(9900);

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

    let port = NEXT_PORT.fetch_add(1, Ordering::Relaxed);
    // Bind the MCP host child on the same interface the daemon itself
    // listens on. If the daemon is on `[::]` or `0.0.0.0` (dual-stack
    // wildcard), the child is reachable on every interface the daemon
    // is; if the daemon is loopback-only, the child stays loopback-only
    // as well. This keeps the child's network exposure in lockstep with
    // an operator's conscious choice for the daemon.
    let listen_addr = SocketAddr::new(bind_ip, port).to_string();

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
    cmd.stdout(std::process::Stdio::inherit());
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

    match wait_for_ready_or_exit(&listen_addr, 10, &mut child).await {
        WaitOutcome::Listening => {
            let mcp_url = format!("http://{listen_addr}/mcp");
            info!(%mcp_url, "MCP host ready");
            Ok(ProvisionedSession {
                child,
                mcp_url,
                mcp_token,
            })
        }
        WaitOutcome::Exited { code } => Err(ProvisionError::ChildExited {
            code,
            stderr_tail: read_tail(&stderr_tail).await,
        }),
        WaitOutcome::Timeout { seconds } => {
            // Child is still running and not responding — kill it so
            // we don't leak the process, then surface whatever stderr
            // did accumulate (may be empty for a hung child).
            let _ = child.kill().await;
            Err(ProvisionError::StartupTimeout {
                seconds,
                stderr_tail: read_tail(&stderr_tail).await,
            })
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

enum WaitOutcome {
    /// Child bound the listen port — TCP connect succeeded.
    Listening,
    /// Child exited before binding. Surface with stderr tail so the
    /// operator sees the real reason (bad config, missing path, etc.)
    /// rather than a generic timeout.
    Exited { code: Option<i32> },
    /// Neither happened within the deadline. Child is still running;
    /// caller is expected to kill it.
    Timeout { seconds: u64 },
}

/// Poll TCP connect on `addr` until it succeeds, the child exits, or
/// the deadline is hit — whichever comes first. Distinguishing the
/// three outcomes is what lets the caller surface "your pod spec
/// pointed at a nonexistent path" instead of "timeout waiting for
/// readiness".
async fn wait_for_ready_or_exit(addr: &str, timeout_secs: u64, child: &mut Child) -> WaitOutcome {
    let connect_loop = async {
        loop {
            if TcpStream::connect(addr).await.is_ok() {
                return WaitOutcome::Listening;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    };
    let exit_watch = async {
        let code = child.wait().await.ok().and_then(|s| s.code());
        WaitOutcome::Exited { code }
    };
    let deadline = tokio::time::sleep(Duration::from_secs(timeout_secs));

    tokio::select! {
        outcome = connect_loop => outcome,
        outcome = exit_watch => outcome,
        _ = deadline => WaitOutcome::Timeout { seconds: timeout_secs },
    }
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
}
