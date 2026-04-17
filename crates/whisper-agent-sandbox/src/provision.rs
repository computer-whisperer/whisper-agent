//! Sandbox provisioning backends.
//!
//! Each backend takes a [`HostEnvSpec`] variant and produces a running MCP host
//! process inside the appropriate isolation boundary.

use std::sync::atomic::{AtomicU16, Ordering};
use std::time::Duration;

use tokio::net::TcpStream;
use tokio::process::{Child, Command};
use tracing::info;
use whisper_agent_protocol::sandbox::{AccessMode, NetworkPolicy, PathAccess, HostEnvSpec};

/// Port range for spawned MCP host instances. Each provision bumps the counter.
static NEXT_PORT: AtomicU16 = AtomicU16::new(9820);

pub struct ProvisionedSession {
    pub child: Child,
    pub mcp_url: String,
}

#[derive(Debug, thiserror::Error)]
pub enum ProvisionError {
    #[error("no read-write path in spec — cannot determine workspace root")]
    NoWorkspaceRoot,
    #[error("spawn failed: {0}")]
    Spawn(String),
    #[error("MCP host failed to become ready within {0}s")]
    StartupTimeout(u64),
    #[error("unsupported: {0}")]
    Unsupported(String),
}

pub async fn provision(
    spec: &HostEnvSpec,
    mcp_host_bin: &str,
) -> Result<ProvisionedSession, ProvisionError> {
    match spec {
        HostEnvSpec::Landlock {
            allowed_paths,
            network,
        } => provision_landlock(allowed_paths, network, mcp_host_bin).await,
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
) -> Result<ProvisionedSession, ProvisionError> {
    let workspace_root = allowed_paths
        .iter()
        .find(|p| p.mode == AccessMode::ReadWrite)
        .map(|p| p.path.clone())
        .ok_or(ProvisionError::NoWorkspaceRoot)?;

    let bin_dir = resolve_bin_dir(mcp_host_bin)?;

    let port = NEXT_PORT.fetch_add(1, Ordering::Relaxed);
    let listen_addr = format!("127.0.0.1:{port}");

    info!(
        %workspace_root,
        %listen_addr,
        %bin_dir,
        paths = allowed_paths.len(),
        "spawning landlock-sandboxed MCP host"
    );

    let allowed_paths_owned: Vec<PathAccess> = allowed_paths.to_vec();
    let network_owned = network.clone();

    let mut cmd = Command::new(mcp_host_bin);
    cmd.args([
        "--listen",
        &listen_addr,
        "--workspace-root",
        &workspace_root,
    ]);
    cmd.stdin(std::process::Stdio::null());
    cmd.stdout(std::process::Stdio::inherit());
    cmd.stderr(std::process::Stdio::inherit());
    cmd.kill_on_drop(true);

    unsafe {
        cmd.pre_exec(move || {
            apply_landlock(&allowed_paths_owned, &network_owned, &bin_dir).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::PermissionDenied, e.to_string())
            })
        });
    }

    let child = cmd
        .spawn()
        .map_err(|e| ProvisionError::Spawn(e.to_string()))?;

    wait_for_ready(&listen_addr, 10).await?;

    let mcp_url = format!("http://{listen_addr}/mcp");
    info!(%mcp_url, "MCP host ready");

    Ok(ProvisionedSession { child, mcp_url })
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

/// Poll until a TCP connection to `addr` succeeds, or give up after
/// `timeout_secs`.
async fn wait_for_ready(addr: &str, timeout_secs: u64) -> Result<(), ProvisionError> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(timeout_secs);
    loop {
        if tokio::time::Instant::now() >= deadline {
            return Err(ProvisionError::StartupTimeout(timeout_secs));
        }
        match TcpStream::connect(addr).await {
            Ok(_) => return Ok(()),
            Err(_) => tokio::time::sleep(Duration::from_millis(50)).await,
        }
    }
}
