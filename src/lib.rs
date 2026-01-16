//! Apple AMX (Apple Matrix Coprocessor) bindings for Rust.
//!
//! Provides low-level access to AMX instructions on Apple Silicon.

#![cfg(all(target_arch = "aarch64", target_os = "macos"))]
#![allow(clippy::missing_safety_doc)]

use std::arch::asm;
use std::ffi::CStr;
use std::sync::OnceLock;

// ============================================================================
// Detection
// ============================================================================

/// Detected AMX version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmxVersion {
    M1,
    M2,
    M3,
    M4,
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
        // Remove null terminator if present
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

    // All Apple Silicon Macs have AMX
    let version = match () {
        _ if brand.contains("M1") => AmxVersion::M1,
        _ if brand.contains("M2") => AmxVersion::M2,
        _ if brand.contains("M3") => AmxVersion::M3,
        _ if brand.contains("M4") => AmxVersion::M4,
        _ => AmxVersion::Unknown,
    };

    Some(version)
}

/// Detect if AMX is available on this system.
#[must_use]
pub fn detect() -> Option<AmxVersion> {
    *AMX_AVAILABLE.get_or_init(detect_internal)
}

/// Returns `true` if AMX is available on this system.
#[must_use]
pub fn is_available() -> bool {
    detect().is_some()
}

// ============================================================================
// Raw AMX Instructions
// ============================================================================

// Opcode encoding: base 0x00201000, instruction in bits [9:5], GPR in bits [4:0]
const AMX_OP_BASE: u32 = 0x00201000;

macro_rules! define_amx_op {
    ($name:ident, $opcode:expr) => {
        #[inline(always)]
        pub unsafe fn $name(operand: u64) {
            const OP: u32 = AMX_OP_BASE + ($opcode << 5);
            asm!(
                ".word {op}",
                op = const OP,
                in("x0") operand,
                options(nostack)
            );
        }
    };
}

define_amx_op!(amx_ldx, 0);
define_amx_op!(amx_ldy, 1);
define_amx_op!(amx_stx, 2);
define_amx_op!(amx_sty, 3);
define_amx_op!(amx_ldz, 4);
define_amx_op!(amx_stz, 5);
define_amx_op!(amx_ldzi, 6);
define_amx_op!(amx_stzi, 7);
define_amx_op!(amx_extrx, 8);
define_amx_op!(amx_extry, 9);
define_amx_op!(amx_fma64, 10);
define_amx_op!(amx_fms64, 11);
define_amx_op!(amx_fma32, 12);
define_amx_op!(amx_fms32, 13);
define_amx_op!(amx_mac16, 14);
define_amx_op!(amx_fma16, 15);
define_amx_op!(amx_fms16, 16);
define_amx_op!(amx_vecint, 18);
define_amx_op!(amx_vecfp, 19);
define_amx_op!(amx_matint, 20);
define_amx_op!(amx_matfp, 21);
define_amx_op!(amx_genlut, 22);

/// Enable AMX coprocessor. Must be called before any AMX operations.
#[inline(always)]
pub unsafe fn amx_set() {
    asm!(
        "nop", "nop", "nop",
        ".word {op}",
        op = const AMX_OP_BASE + (17 << 5),
        options(nostack)
    );
}

/// Disable AMX coprocessor.
#[inline(always)]
pub unsafe fn amx_clr() {
    asm!(
        "nop", "nop", "nop",
        ".word {op}",
        op = const AMX_OP_BASE + (17 << 5) + 1,
        options(nostack)
    );
}

// ============================================================================
// Mid-level Operations
// ============================================================================

/// Mid-level AMX operations with ergonomic encoding.
pub mod ops {
    use super::*;

    const ADDR_MASK: u64 = (1 << 56) - 1;

    /// Encode a load/store operand for X/Y registers.
    #[inline(always)]
    const fn encode_xy_ldst(addr: u64, reg: u64, pair: bool) -> u64 {
        ((pair as u64) << 62) | ((reg & 0x7) << 56) | (addr & ADDR_MASK)
    }

    /// Encode a load/store operand for Z registers.
    #[inline(always)]
    const fn encode_z_ldst(addr: u64, row: u64, pair: bool) -> u64 {
        ((pair as u64) << 62) | ((row & 0x3F) << 56) | (addr & ADDR_MASK)
    }

    /// Encode an FMA/MAC operand.
    #[inline(always)]
    const fn encode_fma(x_off: u64, y_off: u64, z_row: u64, vector_mode: bool) -> u64 {
        ((vector_mode as u64) << 63)
            | ((z_row & 0x3F) << 20)
            | ((x_off & 0x1FF) << 10)
            | (y_off & 0x1FF)
    }

    /// Load 64 (or 128 if pair) bytes into X register.
    #[inline(always)]
    pub unsafe fn ldx(addr: *const u8, reg: u64, pair: bool) {
        amx_ldx(encode_xy_ldst(addr as u64, reg, pair));
    }

    /// Load 64 (or 128 if pair) bytes into Y register.
    #[inline(always)]
    pub unsafe fn ldy(addr: *const u8, reg: u64, pair: bool) {
        amx_ldy(encode_xy_ldst(addr as u64, reg, pair));
    }

    /// Load 64 (or 128 if pair) bytes into Z register row.
    #[inline(always)]
    pub unsafe fn ldz(addr: *const u8, row: u64, pair: bool) {
        amx_ldz(encode_z_ldst(addr as u64, row, pair));
    }

    /// Store 64 (or 128 if pair) bytes from X register.
    #[inline(always)]
    pub unsafe fn stx(addr: *mut u8, reg: u64, pair: bool) {
        amx_stx(encode_xy_ldst(addr as u64, reg, pair));
    }

    /// Store 64 (or 128 if pair) bytes from Y register.
    #[inline(always)]
    pub unsafe fn sty(addr: *mut u8, reg: u64, pair: bool) {
        amx_sty(encode_xy_ldst(addr as u64, reg, pair));
    }

    /// Store 64 (or 128 if pair) bytes from Z register row.
    #[inline(always)]
    pub unsafe fn stz(addr: *mut u8, row: u64, pair: bool) {
        amx_stz(encode_z_ldst(addr as u64, row, pair));
    }

    /// FMA for f32: matrix mode computes outer product, vector mode computes pointwise.
    #[inline(always)]
    pub unsafe fn fma32(x_offset: u64, y_offset: u64, z_row: u64, vector_mode: bool) {
        amx_fma32(encode_fma(x_offset, y_offset, z_row, vector_mode));
    }

    /// FMA for f64.
    #[inline(always)]
    pub unsafe fn fma64(x_offset: u64, y_offset: u64, z_row: u64, vector_mode: bool) {
        amx_fma64(encode_fma(x_offset, y_offset, z_row, vector_mode));
    }

    /// FMA for f16.
    #[inline(always)]
    pub unsafe fn fma16(x_offset: u64, y_offset: u64, z_row: u64, vector_mode: bool) {
        amx_fma16(encode_fma(x_offset, y_offset, z_row, vector_mode));
    }

    /// Integer multiply-accumulate for i16.
    #[inline(always)]
    pub unsafe fn mac16(x_offset: u64, y_offset: u64, z_row: u64, vector_mode: bool) {
        amx_mac16(encode_fma(x_offset, y_offset, z_row, vector_mode));
    }
}

// ============================================================================
// SGEMM Implementation
// ============================================================================

/// Single-precision matrix multiplication using AMX.
pub mod sgemm {
    use super::ops::{fma32, ldx, ldy, ldz, stz};
    use super::*;

    /// Compute C = A × B for square matrices (simple implementation).
    ///
    /// Uses 16x16 tiles with AMX FMA instructions.
    ///
    /// # Panics
    /// Panics if AMX is not available or slices are incorrectly sized.
    pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
        assert!(is_available(), "AMX not available");
        assert_eq!(a.len(), n * n, "A must be n×n");
        assert_eq!(b.len(), n * n, "B must be n×n");
        assert_eq!(c.len(), n * n, "C must be n×n");

        if n == 0 {
            return;
        }

        // Zero out C
        c.fill(0.0);

        // SAFETY: We've verified AMX is available and slice sizes are correct
        unsafe {
            amx_set();

            // Process in 16x16 tiles (AMX f32 natural tile size)
            // For fma32 matrix mode with z_row=0, Z is accessed at rows j*4
            // So we use Z rows 0, 4, 8, 12, ..., 60 for the 16 output rows
            for i in (0..n).step_by(16) {
                for j in (0..n).step_by(16) {
                    // Load C[i:i+16, j:j+16] into Z registers
                    // fma32 uses Z[j*4 + (z_row & 3)] for row j, so with z_row=0 we need Z[0,4,8,...]
                    for row in 0..16.min(n - i) {
                        let c_row = &c[(i + row) * n + j..];
                        let load_len = 16.min(n - j);

                        let mut tmp = [0.0f32; 16];
                        tmp[..load_len].copy_from_slice(&c_row[..load_len]);
                        // Load into Z row (row * 4) to match fma32 access pattern
                        ldz(tmp.as_ptr().cast(), (row * 4) as u64, false);
                    }

                    // Accumulate A[i:, :] @ B[:, j:]
                    for k in (0..n).step_by(16) {
                        for kk in 0..16.min(n - k) {
                            // Load column kk of A tile into Y
                            let mut a_col = [0.0f32; 16];
                            for row in 0..16.min(n - i) {
                                a_col[row] = a[(i + row) * n + k + kk];
                            }
                            ldy(a_col.as_ptr().cast(), 0, false);

                            // Load row kk of B tile into X
                            let mut b_row = [0.0f32; 16];
                            let b_start = (k + kk) * n + j;
                            let b_len = 16.min(n - j);
                            b_row[..b_len].copy_from_slice(&b[b_start..b_start + b_len]);
                            ldx(b_row.as_ptr().cast(), 0, false);

                            // FMA: Z[j*4] += Y[j] * X[i] for outer product
                            fma32(0, 0, 0, false);
                        }
                    }

                    // Store Z registers back to C
                    for row in 0..16.min(n - i) {
                        let mut tmp = [0.0f32; 16];
                        stz(tmp.as_mut_ptr().cast(), (row * 4) as u64, false);

                        let store_len = 16.min(n - j);
                        let c_row = &mut c[(i + row) * n + j..];
                        c_row[..store_len].copy_from_slice(&tmp[..store_len]);
                    }
                }
            }

            amx_clr();
        }
    }

    /// Raw pointer interface for C FFI compatibility.
    ///
    /// # Safety
    /// Pointers must be valid for `size * size` f32 elements.
    pub unsafe fn sgemm_raw(a: *const f32, b: *const f32, c: *mut f32, size: usize) {
        if size == 0 || !is_available() {
            return;
        }

        let a_slice = std::slice::from_raw_parts(a, size * size);
        let b_slice = std::slice::from_raw_parts(b, size * size);
        let c_slice = std::slice::from_raw_parts_mut(c, size * size);

        matmul(a_slice, b_slice, c_slice, size);
    }
}

// ============================================================================
// RAII Guard
// ============================================================================

/// RAII guard for AMX state. Enables AMX on creation, disables on drop.
///
/// # Panics
/// Panics if AMX is not available on this hardware.
pub struct AmxGuard(());

impl AmxGuard {
    /// Enable AMX and return a guard that disables it on drop.
    ///
    /// # Panics
    /// Panics if AMX is not available.
    #[must_use]
    pub fn new() -> Self {
        assert!(is_available(), "AMX not available on this hardware");
        // SAFETY: We just verified AMX is available
        unsafe { amx_set() };
        Self(())
    }

    /// Try to enable AMX, returning `None` if unavailable.
    #[must_use]
    pub fn try_new() -> Option<Self> {
        if is_available() {
            // SAFETY: We just verified AMX is available
            unsafe { amx_set() };
            Some(Self(()))
        } else {
            None
        }
    }
}

impl Default for AmxGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AmxGuard {
    fn drop(&mut self) {
        // SAFETY: If we have a guard, AMX was successfully enabled
        unsafe { amx_clr() };
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect() {
        let result = detect();
        println!("AMX detection: {result:?}");
        assert!(result.is_some(), "AMX should be available on Apple Silicon");
    }

    #[test]
    fn test_amx_guard() {
        let guard = AmxGuard::try_new();
        assert!(guard.is_some(), "Should be able to create guard on Apple Silicon");
    }

    #[test]
    fn test_ldx_stx_roundtrip() {
        let Some(_guard) = AmxGuard::try_new() else { return };

        let input: [f32; 16] = std::array::from_fn(|i| i as f32);
        let mut output = [0.0f32; 16];

        // SAFETY: Guard ensures AMX is enabled, arrays are valid
        unsafe {
            ops::ldx(input.as_ptr().cast(), 0, false);
            ops::stx(output.as_mut_ptr().cast(), 0, false);
        }

        assert_eq!(input, output);
    }

    #[test]
    fn test_matmul_small() {
        if !is_available() {
            return;
        }

        // 64x64 identity-ish test
        const N: usize = 64;
        let a: Vec<f32> = (0..N * N).map(|i| if i % (N + 1) == 0 { 1.0 } else { 0.0 }).collect();
        let b: Vec<f32> = (0..N * N).map(|i| (i % N) as f32).collect();
        let mut c = vec![0.0f32; N * N];

        sgemm::matmul(&a, &b, &mut c, N);

        // With identity A, C should equal B
        for (i, (&ci, &bi)) in c.iter().zip(b.iter()).enumerate() {
            assert!(
                (ci - bi).abs() < 1e-5,
                "Mismatch at {i}: got {ci}, expected {bi}"
            );
        }
    }
}
