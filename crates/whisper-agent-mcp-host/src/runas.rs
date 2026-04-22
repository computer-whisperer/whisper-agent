//! Resolve Unix users by name and apply privilege drop in a forked
//! child via `Command::pre_exec`.
//!
//! Lookup runs in the parent: it does NSS / passwd-file IO, which is
//! not async-signal-safe and so cannot live inside `pre_exec`. The
//! apply step runs in the child between fork and exec, using only
//! async-signal-safe libc calls (`setgroups`, `setgid`, `setuid`).
//!
//! Whether `setuid` actually succeeds is left to the kernel: if the
//! host process lacks the privilege to assume the requested user, the
//! child returns the libc error and `Command::spawn` surfaces it as a
//! clean tool-result error. There is no allowlist in this layer.

use std::ffi::{CStr, CString};
use std::io;

#[derive(Debug, Clone)]
pub struct UserIdentity {
    pub name: String,
    pub uid: libc::uid_t,
    pub gid: libc::gid_t,
    pub groups: Vec<libc::gid_t>,
    pub home: String,
}

#[derive(Debug, thiserror::Error)]
pub enum LookupError {
    #[error("user `{0}` not found")]
    NotFound(String),
    #[error("user name contains a NUL byte")]
    InvalidName,
    #[error("getpwnam_r failed: {0}")]
    Io(io::Error),
}

pub fn lookup(name: &str) -> Result<UserIdentity, LookupError> {
    let c_name = CString::new(name).map_err(|_| LookupError::InvalidName)?;
    let mut pwd: libc::passwd = unsafe { std::mem::zeroed() };
    let mut buf = vec![0u8; 4096];
    let mut result: *mut libc::passwd = std::ptr::null_mut();

    loop {
        let rc = unsafe {
            libc::getpwnam_r(
                c_name.as_ptr(),
                &mut pwd,
                buf.as_mut_ptr() as *mut libc::c_char,
                buf.len(),
                &mut result,
            )
        };
        if rc == 0 {
            break;
        }
        // ERANGE means our scratch buffer is too small for this entry's
        // strings; double and retry, but cap so a malformed NSS source
        // can't make us OOM.
        if rc == libc::ERANGE && buf.len() < 1 << 20 {
            buf.resize(buf.len() * 2, 0);
            continue;
        }
        return Err(LookupError::Io(io::Error::from_raw_os_error(rc)));
    }
    if result.is_null() {
        return Err(LookupError::NotFound(name.to_string()));
    }

    let home = unsafe { CStr::from_ptr(pwd.pw_dir) }
        .to_string_lossy()
        .into_owned();
    let uid = pwd.pw_uid;
    let gid = pwd.pw_gid;

    let groups = supplementary_groups(&c_name, gid)?;
    Ok(UserIdentity {
        name: name.to_string(),
        uid,
        gid,
        groups,
        home,
    })
}

fn supplementary_groups(
    c_name: &CStr,
    primary_gid: libc::gid_t,
) -> Result<Vec<libc::gid_t>, LookupError> {
    let mut ngroups: libc::c_int = 32;
    let mut groups: Vec<libc::gid_t> = vec![0; ngroups as usize];
    loop {
        let rc = unsafe {
            libc::getgrouplist(
                c_name.as_ptr(),
                primary_gid,
                groups.as_mut_ptr(),
                &mut ngroups,
            )
        };
        if rc >= 0 {
            groups.truncate(ngroups as usize);
            return Ok(groups);
        }
        // -1 means the buffer was too small; ngroups was updated with
        // the required size. Defend against a buggy libc that fails to
        // grow ngroups so we never spin forever.
        let needed = ngroups as usize;
        if needed <= groups.len() {
            return Err(LookupError::Io(io::Error::other(
                "getgrouplist returned -1 without growing ngroups",
            )));
        }
        groups.resize(needed, 0);
    }
}

/// Apply the resolved identity to the *current* process. Intended for
/// use inside `Command::pre_exec`, where it runs in the forked child
/// and so only changes the child's identity.
///
/// # Safety
///
/// This permanently changes the process's effective and real
/// uid/gid/groups. Calling it outside of a `pre_exec` closure (or any
/// other forked-child context) will alter the calling process's
/// identity for the remainder of its lifetime — including for any
/// other concurrent work running in that process.
pub unsafe fn apply_in_child(id: &UserIdentity) -> io::Result<()> {
    if unsafe { libc::setgroups(id.groups.len(), id.groups.as_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    if unsafe { libc::setgid(id.gid) } != 0 {
        return Err(io::Error::last_os_error());
    }
    if unsafe { libc::setuid(id.uid) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_root() {
        // root is the one user we can rely on existing on every Unix host
        // these tests will run on.
        let id = lookup("root").expect("root must resolve");
        assert_eq!(id.name, "root");
        assert_eq!(id.uid, 0);
        assert_eq!(id.gid, 0);
        assert!(!id.home.is_empty());
        // root's primary gid (0) should appear in the supplementary list
        // returned by getgrouplist.
        assert!(id.groups.contains(&0));
    }

    #[test]
    fn lookup_unknown_returns_not_found() {
        let err = lookup("definitely_not_a_real_user_zzz_9217").unwrap_err();
        assert!(matches!(err, LookupError::NotFound(_)));
    }

    #[test]
    fn lookup_rejects_nul_byte() {
        let err = lookup("bad\0name").unwrap_err();
        assert!(matches!(err, LookupError::InvalidName));
    }
}
