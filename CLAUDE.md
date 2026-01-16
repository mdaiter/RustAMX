# mac_amx

Rust bindings for Apple AMX (Apple Matrix Coprocessor) on Apple Silicon.

## Build & Test

```bash
cargo build
cargo test
cargo clippy
```

## Architecture

- `src/lib.rs` - Single-file library with all AMX bindings

### Code Structure

1. **Detection** - `detect()`, `is_available()` via sysctl
2. **Raw instructions** - `amx_ldx`, `amx_fma32`, etc. (inline asm)
3. **Mid-level ops** - `ops::ldx()`, `ops::fma32()` with operand encoding
4. **SGEMM** - `sgemm::matmul()` for matrix multiplication
5. **AmxGuard** - RAII for AMX enable/disable

### AMX Register Layout

- X registers: 8 x 64 bytes (load sources)
- Y registers: 8 x 64 bytes (load sources)
- Z registers: 64 x 64 bytes (accumulators)

For f32 FMA matrix mode with z_row=0, Z is accessed at rows `j*4` (interleaved).

## Key Files

- `amx/` - Reference C implementation and documentation (read-only)
- `amx/Instructions.md` - Instruction reference
- `amx/fma.md` - FMA instruction details including Z indexing

## Style

- Prefer iterators over index loops where memory is contiguous
- Use `copy_from_slice` for bulk copies
- Strided/gathered access requires indexed loops
- All public unsafe fns need AMX enabled first (except `amx_set`)
