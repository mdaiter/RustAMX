# mac_amx

Rust bindings for Apple AMX (Apple Matrix Coprocessor) on Apple Silicon.

## Build & Test

```bash
cargo build
cargo test
cargo clippy
cargo doc --open
```

## Public API

```rust
// High-level (safe)
use mac_amx::{Matrix, is_available, detect, AmxVersion};

let a = Matrix::identity(64);
let b = Matrix::fill(64, 64, 2.0);
let c = a.matmul(&b);

// Mid-level (unsafe, requires AmxGuard)
use mac_amx::{AmxGuard, ops};

let _guard = AmxGuard::new();
unsafe {
    ops::ldx(ptr, 0, false);
    ops::fma32(0, 0, 0, false);
}

// Low-level (raw instructions)
use mac_amx::raw;
```

## Module Structure

```
src/
  lib.rs      # Matrix type, AmxGuard, matmul implementation
  detect.rs   # AMX detection via sysctl
  ops.rs      # Mid-level ops with operand encoding
  raw.rs      # Raw AMX instructions (inline asm)
```

## Key Types

- `Matrix` - Row-major f32 matrix with AMX-accelerated matmul
- `AmxGuard` - RAII guard for AMX enable/disable
- `AmxVersion` - M1/M2/M3/M4/Unknown

## AMX Register Layout

- X: 8 × 64 bytes (512 bytes total)
- Y: 8 × 64 bytes (512 bytes total)
- Z: 64 × 64 bytes (4096 bytes total, accumulator)

For f32 FMA matrix mode, Z rows are accessed at `j*4 + (z_row & 3)`.

## Reference

- `amx/` - C reference implementation (read-only)
- `amx/Instructions.md` - Full instruction reference
- `amx/fma.md` - FMA encoding details
