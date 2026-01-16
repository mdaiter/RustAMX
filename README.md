# mac_amx

Rust bindings for Apple's AMX (Apple Matrix Coprocessor) on Apple Silicon.

AMX is an undocumented SIMD coprocessor in M1/M2/M3/M4 chips that accelerates matrix operations. This crate provides safe high-level APIs and unsafe low-level access to AMX instructions.

## Features

- **Zero dependencies** - just `std`
- **Safe `Matrix` type** with AMX-accelerated multiplication
- **Automatic fallback** to scalar code on non-AMX hardware
- **Three API levels**: safe, mid-level, and raw instructions

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS
- Rust nightly (for inline assembly)

## Installation

```toml
[dependencies]
mac_amx = { git = "https://github.com/mdaiter/RustAMX" }
```

## Usage

### High-Level (Safe)

```rust
use mac_amx::{Matrix, is_available};

// Check availability
if is_available() {
    println!("AMX available!");
}

// Matrix operations - automatically uses AMX when beneficial
let a = Matrix::identity(64);
let b = Matrix::fill(64, 64, 2.0);
let c = a.matmul(&b);

println!("Result: {:?}", &c.data()[..4]);
```

### Mid-Level (Custom Kernels)

```rust
use mac_amx::{AmxGuard, ops};

// RAII guard enables AMX, disables on drop
let _guard = AmxGuard::new();

let mut x_data = [1.0f32; 16];
let mut y_data = [2.0f32; 16];
let mut z_data = [0.0f32; 16];

unsafe {
    // Load data into AMX registers
    ops::ldx(x_data.as_ptr().cast(), 0, false);
    ops::ldy(y_data.as_ptr().cast(), 0, false);
    ops::ldz(z_data.as_ptr().cast(), 0, false);

    // Fused multiply-add: Z += X * Y (outer product)
    ops::fma32(0, 0, 0, false);

    // Store result
    ops::stz(z_data.as_mut_ptr().cast(), 0, false);
}
```

### Low-Level (Raw Instructions)

```rust
use mac_amx::raw;

unsafe {
    raw::amx_set();  // Enable AMX

    // Encode operand manually: addr in bits 0-55, reg in bits 56-58
    let operand = (ptr as u64) | (reg << 56);
    raw::amx_ldx(operand);

    raw::amx_clr();  // Disable AMX
}
```

## API Overview

| Level | Module | Safety | Use Case |
|-------|--------|--------|----------|
| High | `Matrix` | Safe | General matrix math |
| Mid | `ops` | Unsafe | Custom kernels |
| Low | `raw` | Unsafe | Full hardware control |

### Matrix Type

```rust
Matrix::zeros(rows, cols)      // Zero-filled matrix
Matrix::fill(rows, cols, val)  // Constant-filled matrix
Matrix::identity(n)            // Identity matrix
Matrix::from_slice(r, c, data) // From existing data
Matrix::from_vec(r, c, vec)    // From Vec (takes ownership)

matrix.matmul(&other)          // Matrix multiplication
matrix.transpose()             // Transpose
matrix.add(&other)             // Element-wise addition
matrix.sub(&other)             // Element-wise subtraction
matrix.scale(scalar)           // Scalar multiplication

matrix[(row, col)]             // Index access
matrix.data()                  // Raw slice access
```

### Detection

```rust
use mac_amx::{detect, is_available, AmxVersion};

match detect() {
    Some(AmxVersion::M1) => println!("M1"),
    Some(AmxVersion::M2) => println!("M2"),
    Some(AmxVersion::M3) => println!("M3"),
    Some(AmxVersion::M4) => println!("M4"),
    Some(AmxVersion::Unknown) => println!("Unknown Apple Silicon"),
    None => println!("Not Apple Silicon"),
}

if is_available() {
    // Use AMX
}
```

## AMX Architecture

AMX has three register files:

| Register | Count | Size | Total | Purpose |
|----------|-------|------|-------|---------|
| X | 8 | 64 B | 512 B | Input operand |
| Y | 8 | 64 B | 512 B | Input operand |
| Z | 64 | 64 B | 4 KB | Accumulator |

Key operations:
- **Load/Store**: Move data between memory and registers
- **FMA**: Fused multiply-add (outer product or pointwise)
- **MAC**: Integer multiply-accumulate

## Performance

AMX excels at dense matrix operations. For a 1024×1024 f32 matmul on M1 Max:

| Method | Performance |
|--------|-------------|
| Naive scalar | ~2 GFLOPS |
| AMX | ~300+ GFLOPS |

## References

- [corsix/amx](https://github.com/corsix/amx) - AMX reverse engineering documentation
- [Apple Silicon AMX](https://gist.github.com/dougallj/7a75a3be1ec69ca550e7c36dc75e0d6f) - Instruction details

## License

MIT

## Contributing

Contributions welcome. Please ensure `cargo test` and `cargo clippy` pass.
