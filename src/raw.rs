//! Raw AMX instruction wrappers.
//!
//! These are the lowest-level AMX primitives, mapping directly to hardware instructions.
//! For most use cases, prefer the higher-level [`ops`](crate::ops) module.
//!
//! # Safety
//!
//! All functions in this module are unsafe and require:
//! - AMX must be enabled via [`amx_set`] before use (except `amx_set` itself)
//! - AMX must be available on the hardware (check with [`crate::is_available`])
//! - For load/store operations, memory addresses must be valid

#![allow(clippy::missing_safety_doc)]

use std::arch::asm;

const AMX_OP_BASE: u32 = 0x00201000;

macro_rules! define_amx_op {
    ($(#[$meta:meta])* $name:ident, $opcode:expr) => {
        $(#[$meta])*
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

// Load instructions
define_amx_op!(
    /// Load 64 bytes into X register.
    amx_ldx, 0
);
define_amx_op!(
    /// Load 64 bytes into Y register.
    amx_ldy, 1
);
define_amx_op!(
    /// Load 64 bytes into Z register row.
    amx_ldz, 4
);
define_amx_op!(
    /// Load 64 bytes into Z register (interleaved mode).
    amx_ldzi, 6
);

// Store instructions
define_amx_op!(
    /// Store 64 bytes from X register.
    amx_stx, 2
);
define_amx_op!(
    /// Store 64 bytes from Y register.
    amx_sty, 3
);
define_amx_op!(
    /// Store 64 bytes from Z register row.
    amx_stz, 5
);
define_amx_op!(
    /// Store 64 bytes from Z register (interleaved mode).
    amx_stzi, 7
);

// Extract instructions
define_amx_op!(
    /// Extract from Y to X.
    amx_extrx, 8
);
define_amx_op!(
    /// Extract from X to Y.
    amx_extry, 9
);

// Floating-point FMA instructions
define_amx_op!(
    /// Fused multiply-add for f64.
    amx_fma64, 10
);
define_amx_op!(
    /// Fused multiply-subtract for f64.
    amx_fms64, 11
);
define_amx_op!(
    /// Fused multiply-add for f32.
    amx_fma32, 12
);
define_amx_op!(
    /// Fused multiply-subtract for f32.
    amx_fms32, 13
);
define_amx_op!(
    /// Fused multiply-add for f16.
    amx_fma16, 15
);
define_amx_op!(
    /// Fused multiply-subtract for f16.
    amx_fms16, 16
);

// Integer instructions
define_amx_op!(
    /// Integer multiply-accumulate for i16.
    amx_mac16, 14
);
define_amx_op!(
    /// Vector integer operation.
    amx_vecint, 18
);
define_amx_op!(
    /// Matrix integer operation.
    amx_matint, 20
);

// Vector/Matrix floating-point
define_amx_op!(
    /// Vector floating-point operation.
    amx_vecfp, 19
);
define_amx_op!(
    /// Matrix floating-point operation.
    amx_matfp, 21
);

// Lookup table
define_amx_op!(
    /// Generate lookup table indices.
    amx_genlut, 22
);

/// Enable AMX coprocessor.
///
/// Must be called before any other AMX operations.
///
/// # Safety
/// - AMX must be available on this hardware
/// - Should be balanced with a call to [`amx_clr`]
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
///
/// Should be called when done with AMX operations.
///
/// # Safety
/// - AMX must have been enabled with [`amx_set`]
#[inline(always)]
pub unsafe fn amx_clr() {
    asm!(
        "nop", "nop", "nop",
        ".word {op}",
        op = const AMX_OP_BASE + (17 << 5) + 1,
        options(nostack)
    );
}
