//! Mid-level AMX operations with ergonomic operand encoding.
//!
//! This module provides a more convenient interface than [`raw`](crate::raw)
//! by handling operand encoding automatically.
//!
//! # Example
//!
//! ```no_run
//! use mac_amx::{AmxGuard, ops};
//!
//! let _guard = AmxGuard::new();
//!
//! let mut data = [0.0f32; 16];
//! unsafe {
//!     ops::ldx(data.as_ptr().cast(), 0, false);
//!     // ... perform operations ...
//!     ops::stx(data.as_mut_ptr().cast(), 0, false);
//! }
//! ```
//!
//! # Safety
//!
//! All functions require AMX to be enabled (via [`AmxGuard`](crate::AmxGuard) or
//! [`raw::amx_set`](crate::raw::amx_set)).

#![allow(clippy::missing_safety_doc)]

use crate::raw;

const ADDR_MASK: u64 = (1 << 56) - 1;

/// Encode load/store operand for X/Y registers.
#[inline(always)]
const fn encode_xy(addr: u64, reg: u64, pair: bool) -> u64 {
    ((pair as u64) << 62) | ((reg & 0x7) << 56) | (addr & ADDR_MASK)
}

/// Encode load/store operand for Z registers.
#[inline(always)]
const fn encode_z(addr: u64, row: u64, pair: bool) -> u64 {
    ((pair as u64) << 62) | ((row & 0x3F) << 56) | (addr & ADDR_MASK)
}

/// Encode FMA/MAC operand.
#[inline(always)]
const fn encode_fma(x_off: u64, y_off: u64, z_row: u64, vector_mode: bool) -> u64 {
    ((vector_mode as u64) << 63)
        | ((z_row & 0x3F) << 20)
        | ((x_off & 0x1FF) << 10)
        | (y_off & 0x1FF)
}

// ============================================================================
// Load Operations
// ============================================================================

/// Load into X register.
///
/// # Arguments
/// - `addr`: Source memory address
/// - `reg`: X register index (0-7)
/// - `pair`: If true, load 128 bytes into consecutive registers
#[inline(always)]
pub unsafe fn ldx(addr: *const u8, reg: u64, pair: bool) {
    raw::amx_ldx(encode_xy(addr as u64, reg, pair));
}

/// Load into Y register.
///
/// # Arguments
/// - `addr`: Source memory address
/// - `reg`: Y register index (0-7)
/// - `pair`: If true, load 128 bytes into consecutive registers
#[inline(always)]
pub unsafe fn ldy(addr: *const u8, reg: u64, pair: bool) {
    raw::amx_ldy(encode_xy(addr as u64, reg, pair));
}

/// Load into Z register row.
///
/// # Arguments
/// - `addr`: Source memory address
/// - `row`: Z register row (0-63)
/// - `pair`: If true, load 128 bytes into consecutive rows
#[inline(always)]
pub unsafe fn ldz(addr: *const u8, row: u64, pair: bool) {
    raw::amx_ldz(encode_z(addr as u64, row, pair));
}

// ============================================================================
// Store Operations
// ============================================================================

/// Store from X register.
///
/// # Arguments
/// - `addr`: Destination memory address
/// - `reg`: X register index (0-7)
/// - `pair`: If true, store 128 bytes from consecutive registers
#[inline(always)]
pub unsafe fn stx(addr: *mut u8, reg: u64, pair: bool) {
    raw::amx_stx(encode_xy(addr as u64, reg, pair));
}

/// Store from Y register.
///
/// # Arguments
/// - `addr`: Destination memory address
/// - `reg`: Y register index (0-7)
/// - `pair`: If true, store 128 bytes from consecutive registers
#[inline(always)]
pub unsafe fn sty(addr: *mut u8, reg: u64, pair: bool) {
    raw::amx_sty(encode_xy(addr as u64, reg, pair));
}

/// Store from Z register row.
///
/// # Arguments
/// - `addr`: Destination memory address
/// - `row`: Z register row (0-63)
/// - `pair`: If true, store 128 bytes from consecutive rows
#[inline(always)]
pub unsafe fn stz(addr: *mut u8, row: u64, pair: bool) {
    raw::amx_stz(encode_z(addr as u64, row, pair));
}

// ============================================================================
// FMA Operations
// ============================================================================

/// Fused multiply-add for f32.
///
/// - **Matrix mode** (`vector_mode=false`): `Z[j][i] += X[i] × Y[j]` (outer product)
/// - **Vector mode** (`vector_mode=true`): `Z[row][i] += X[i] × Y[i]` (pointwise)
///
/// # Arguments
/// - `x_offset`: Byte offset into X register file (0-511)
/// - `y_offset`: Byte offset into Y register file (0-511)
/// - `z_row`: Z register row (0-63, high bits ignored in matrix mode)
/// - `vector_mode`: false for outer product, true for pointwise
#[inline(always)]
pub unsafe fn fma32(x_offset: u64, y_offset: u64, z_row: u64, vector_mode: bool) {
    raw::amx_fma32(encode_fma(x_offset, y_offset, z_row, vector_mode));
}

/// Fused multiply-add for f64.
///
/// Same semantics as [`fma32`] but for double precision.
#[inline(always)]
pub unsafe fn fma64(x_offset: u64, y_offset: u64, z_row: u64, vector_mode: bool) {
    raw::amx_fma64(encode_fma(x_offset, y_offset, z_row, vector_mode));
}

/// Fused multiply-add for f16.
///
/// Same semantics as [`fma32`] but for half precision.
#[inline(always)]
pub unsafe fn fma16(x_offset: u64, y_offset: u64, z_row: u64, vector_mode: bool) {
    raw::amx_fma16(encode_fma(x_offset, y_offset, z_row, vector_mode));
}

/// Fused multiply-subtract for f32.
///
/// Same as [`fma32`] but subtracts: `Z -= X × Y`.
#[inline(always)]
pub unsafe fn fms32(x_offset: u64, y_offset: u64, z_row: u64, vector_mode: bool) {
    raw::amx_fms32(encode_fma(x_offset, y_offset, z_row, vector_mode));
}

/// Fused multiply-subtract for f64.
#[inline(always)]
pub unsafe fn fms64(x_offset: u64, y_offset: u64, z_row: u64, vector_mode: bool) {
    raw::amx_fms64(encode_fma(x_offset, y_offset, z_row, vector_mode));
}

/// Fused multiply-subtract for f16.
#[inline(always)]
pub unsafe fn fms16(x_offset: u64, y_offset: u64, z_row: u64, vector_mode: bool) {
    raw::amx_fms16(encode_fma(x_offset, y_offset, z_row, vector_mode));
}

// ============================================================================
// Integer Operations
// ============================================================================

/// Integer multiply-accumulate for i16.
///
/// Same semantics as [`fma32`] but for 16-bit integers.
#[inline(always)]
pub unsafe fn mac16(x_offset: u64, y_offset: u64, z_row: u64, vector_mode: bool) {
    raw::amx_mac16(encode_fma(x_offset, y_offset, z_row, vector_mode));
}
