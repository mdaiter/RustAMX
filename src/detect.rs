//! AMX hardware detection.

use std::ffi::CStr;
use std::sync::OnceLock;

/// Detected AMX version based on Apple Silicon generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AmxVersion {
    M1,
    M2,
    M3,
    M4,
    /// Unknown Apple Silicon (still has AMX support)
    Unknown,
}

static AMX_AVAILABLE: OnceLock<Option<AmxVersion>> = OnceLock::new();

fn sysctl_string(name: &CStr) -> Option<String> {
    use std::os::raw::{c_char, c_int, c_void};

    extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }

    let mut size: usize = 0;
    let name_ptr = name.as_ptr();

    // SAFETY: sysctlbyname is a standard macOS syscall
    unsafe {
        if sysctlbyname(name_ptr, std::ptr::null_mut(), &mut size, std::ptr::null_mut(), 0) != 0 {
            return None;
        }

        let mut buf = vec![0u8; size];
        if sysctlbyname(name_ptr, buf.as_mut_ptr().cast(), &mut size, std::ptr::null_mut(), 0) != 0 {
            return None;
        }

        buf.truncate(size);
        if buf.last() == Some(&0) {
            buf.pop();
        }

        String::from_utf8(buf).ok()
    }
}

fn detect_internal() -> Option<AmxVersion> {
    let brand = sysctl_string(c"machdep.cpu.brand_string")?;

    if !brand.contains("Apple") {
        return None;
    }

    let version = match () {
        _ if brand.contains("M1") => AmxVersion::M1,
        _ if brand.contains("M2") => AmxVersion::M2,
        _ if brand.contains("M3") => AmxVersion::M3,
        _ if brand.contains("M4") => AmxVersion::M4,
        _ => AmxVersion::Unknown,
    };

    Some(version)
}

/// Detect AMX availability and version.
///
/// Returns `Some(version)` if running on Apple Silicon, `None` otherwise.
/// Result is cached after first call.
///
/// # Example
///
/// ```no_run
/// use mac_amx::detect;
///
/// match detect() {
///     Some(version) => println!("AMX available: {version:?}"),
///     None => println!("AMX not available"),
/// }
/// ```
#[must_use]
pub fn detect() -> Option<AmxVersion> {
    *AMX_AVAILABLE.get_or_init(detect_internal)
}

/// Check if AMX is available.
///
/// Equivalent to `detect().is_some()`.
#[must_use]
#[inline]
pub fn is_available() -> bool {
    detect().is_some()
}
