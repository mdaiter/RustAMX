// amx.h - C bindings for Apple AMX coprocessor
// Pure C with inline assembly, matching Rust semantics
//
// Memory safety guarantees:
// - All functions with pointer args require valid, aligned memory
// - Load/store operate on exactly 64 bytes (or 128 with pair=true)
// - No hidden allocations or copies

#ifndef AMX_H
#define AMX_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// AMX Version Detection
// ============================================================================

typedef enum {
    AMX_VERSION_NONE = -1,      // Not Apple Silicon
    AMX_VERSION_UNKNOWN = 0,    // Unknown Apple Silicon (has AMX)
    AMX_VERSION_M1 = 1,
    AMX_VERSION_M2 = 2,
    AMX_VERSION_M3 = 3,
    AMX_VERSION_M4 = 4,
} AmxVersion;

/// Detect AMX availability and version.
/// Result is cached after first call (thread-safe).
/// Returns AMX_VERSION_NONE if not on Apple Silicon.
AmxVersion amx_detect(void);

/// Check if AMX is available. Equivalent to amx_detect() != AMX_VERSION_NONE.
static inline bool amx_is_available(void) {
    return amx_detect() != AMX_VERSION_NONE;
}

// ============================================================================
// AMX Control (Enable/Disable)
// ============================================================================

/// Enable AMX coprocessor. Must be called before any AMX operations.
/// SAFETY: AMX must be available (check with amx_is_available first)
void amx_set(void);

/// Disable AMX coprocessor. Call when done with AMX operations.
/// SAFETY: AMX must have been enabled with amx_set
void amx_clr(void);

// ============================================================================
// Raw AMX Instructions
// ============================================================================

// All raw functions take a pre-encoded 64-bit operand.
// SAFETY: AMX must be enabled, operand must be correctly encoded.

void amx_ldx(uint64_t operand);
void amx_ldy(uint64_t operand);
void amx_ldz(uint64_t operand);
void amx_ldzi(uint64_t operand);

void amx_stx(uint64_t operand);
void amx_sty(uint64_t operand);
void amx_stz(uint64_t operand);
void amx_stzi(uint64_t operand);

void amx_extrx(uint64_t operand);
void amx_extry(uint64_t operand);

void amx_fma64(uint64_t operand);
void amx_fms64(uint64_t operand);
void amx_fma32(uint64_t operand);
void amx_fms32(uint64_t operand);
void amx_fma16(uint64_t operand);
void amx_fms16(uint64_t operand);

void amx_mac16(uint64_t operand);
void amx_vecint(uint64_t operand);
void amx_vecfp(uint64_t operand);
void amx_matint(uint64_t operand);
void amx_matfp(uint64_t operand);
void amx_genlut(uint64_t operand);

// ============================================================================
// Operand Encoding Helpers
// ============================================================================

#define AMX_ADDR_MASK ((1ULL << 56) - 1)

/// Encode X/Y register load/store operand.
/// - addr: Memory address (56 bits)
/// - reg: Register index 0-7
/// - pair: Load/store 128 bytes into consecutive registers
static inline uint64_t amx_encode_xy(const void *addr, uint64_t reg, bool pair) {
    return ((uint64_t)pair << 62) | ((reg & 0x7) << 56) | ((uint64_t)addr & AMX_ADDR_MASK);
}

/// Encode Z register load/store operand.
/// - addr: Memory address (56 bits)
/// - row: Row index 0-63
/// - pair: Load/store 128 bytes into consecutive rows
static inline uint64_t amx_encode_z(const void *addr, uint64_t row, bool pair) {
    return ((uint64_t)pair << 62) | ((row & 0x3F) << 56) | ((uint64_t)addr & AMX_ADDR_MASK);
}

/// Encode FMA/MAC operand.
/// - x_offset: Byte offset into X register file (0-511)
/// - y_offset: Byte offset into Y register file (0-511)
/// - z_row: Z register row (0-63)
/// - vector_mode: false=outer product, true=pointwise
static inline uint64_t amx_encode_fma(uint64_t x_offset, uint64_t y_offset, 
                                       uint64_t z_row, bool vector_mode) {
    return ((uint64_t)vector_mode << 63)
         | ((z_row & 0x3F) << 20)
         | ((x_offset & 0x1FF) << 10)
         | (y_offset & 0x1FF);
}

// ============================================================================
// Mid-Level Operations (Ergonomic wrappers)
// ============================================================================

/// Load 64 bytes into X register.
/// SAFETY: addr must point to at least 64 valid bytes (128 if pair=true)
static inline void amx_load_x(const void *addr, uint64_t reg, bool pair) {
    amx_ldx(amx_encode_xy(addr, reg, pair));
}

/// Load 64 bytes into Y register.
static inline void amx_load_y(const void *addr, uint64_t reg, bool pair) {
    amx_ldy(amx_encode_xy(addr, reg, pair));
}

/// Load 64 bytes into Z register row.
static inline void amx_load_z(const void *addr, uint64_t row, bool pair) {
    amx_ldz(amx_encode_z(addr, row, pair));
}

/// Store 64 bytes from X register.
/// SAFETY: addr must point to at least 64 writable bytes (128 if pair=true)
static inline void amx_store_x(void *addr, uint64_t reg, bool pair) {
    amx_stx(amx_encode_xy(addr, reg, pair));
}

/// Store 64 bytes from Y register.
static inline void amx_store_y(void *addr, uint64_t reg, bool pair) {
    amx_sty(amx_encode_xy(addr, reg, pair));
}

/// Store 64 bytes from Z register row.
static inline void amx_store_z(void *addr, uint64_t row, bool pair) {
    amx_stz(amx_encode_z(addr, row, pair));
}

/// FMA for f32: Z += X * Y
/// vector_mode=false: outer product, vector_mode=true: pointwise
static inline void amx_fma32_op(uint64_t x_off, uint64_t y_off, uint64_t z_row, bool vector_mode) {
    amx_fma32(amx_encode_fma(x_off, y_off, z_row, vector_mode));
}

/// FMA for f64
static inline void amx_fma64_op(uint64_t x_off, uint64_t y_off, uint64_t z_row, bool vector_mode) {
    amx_fma64(amx_encode_fma(x_off, y_off, z_row, vector_mode));
}

/// FMA for f16
static inline void amx_fma16_op(uint64_t x_off, uint64_t y_off, uint64_t z_row, bool vector_mode) {
    amx_fma16(amx_encode_fma(x_off, y_off, z_row, vector_mode));
}

/// FMS (subtract) for f32: Z -= X * Y
static inline void amx_fms32_op(uint64_t x_off, uint64_t y_off, uint64_t z_row, bool vector_mode) {
    amx_fms32(amx_encode_fma(x_off, y_off, z_row, vector_mode));
}

/// FMS for f64
static inline void amx_fms64_op(uint64_t x_off, uint64_t y_off, uint64_t z_row, bool vector_mode) {
    amx_fms64(amx_encode_fma(x_off, y_off, z_row, vector_mode));
}

/// FMS for f16
static inline void amx_fms16_op(uint64_t x_off, uint64_t y_off, uint64_t z_row, bool vector_mode) {
    amx_fms16(amx_encode_fma(x_off, y_off, z_row, vector_mode));
}

/// Integer MAC for i16
static inline void amx_mac16_op(uint64_t x_off, uint64_t y_off, uint64_t z_row, bool vector_mode) {
    amx_mac16(amx_encode_fma(x_off, y_off, z_row, vector_mode));
}

// ============================================================================
// High-Level Matrix Operations
// ============================================================================

/// Opaque matrix handle. Use amx_matrix_* functions to manipulate.
/// 
/// Matrix storage is optimized for AMX:
/// - Data is 64-byte aligned (required for AMX load/store)
/// - Row stride is padded to 16-float (64-byte) boundary
/// - This enables direct AMX loads without intermediate copies
///
/// Memory layout: row-major with padding
/// For a 17x17 matrix, stride = 32 (rounded up to 16-float boundary)
/// Row 0: [0..16] [17..31 padding]
/// Row 1: [32..48] [49..63 padding]
/// ...
typedef struct AmxMatrix AmxMatrix;

/// Create a zero-filled matrix.
/// Returns NULL on allocation failure.
AmxMatrix *amx_matrix_zeros(size_t rows, size_t cols);

/// Create a matrix filled with a constant value.
AmxMatrix *amx_matrix_fill(size_t rows, size_t cols, float value);

/// Create an identity matrix (must be square).
AmxMatrix *amx_matrix_identity(size_t n);

/// Create a matrix from existing data (copies the data).
/// data must have at least rows*cols elements.
AmxMatrix *amx_matrix_from_data(size_t rows, size_t cols, const float *data);

/// Create a matrix that takes ownership of data (no copy).
/// data must be allocated with malloc and have exactly rows*cols elements.
/// After this call, the matrix owns the data - do not free it yourself.
AmxMatrix *amx_matrix_from_owned(size_t rows, size_t cols, float *data);

/// Clone a matrix (deep copy).
AmxMatrix *amx_matrix_clone(const AmxMatrix *m);

/// Free a matrix. Safe to call with NULL.
void amx_matrix_free(AmxMatrix *m);

/// Get number of rows.
size_t amx_matrix_rows(const AmxMatrix *m);

/// Get number of columns.
size_t amx_matrix_cols(const AmxMatrix *m);

/// Get row stride in floats (>= cols, multiple of 16).
/// Use this when iterating over raw data: row i starts at data[i * stride].
size_t amx_matrix_stride(const AmxMatrix *m);

/// Get pointer to underlying data (row-major order with stride).
/// Valid until matrix is modified or freed.
const float *amx_matrix_data(const AmxMatrix *m);

/// Get mutable pointer to underlying data.
/// Invalidates any previous data pointers if copy-on-write triggers.
float *amx_matrix_data_mut(AmxMatrix *m);

/// Get element at (row, col). No bounds checking.
float amx_matrix_get(const AmxMatrix *m, size_t row, size_t col);

/// Set element at (row, col). No bounds checking.
void amx_matrix_set(AmxMatrix *m, size_t row, size_t col, float value);

/// Matrix multiplication: result = a * b
/// Returns NULL if dimensions don't match or allocation fails.
/// Uses AMX acceleration for square matrices when available.
AmxMatrix *amx_matrix_matmul(const AmxMatrix *a, const AmxMatrix *b);

/// Transpose a matrix.
AmxMatrix *amx_matrix_transpose(const AmxMatrix *m);

/// Element-wise addition: result = a + b
/// Returns NULL if shapes don't match.
AmxMatrix *amx_matrix_add(const AmxMatrix *a, const AmxMatrix *b);

/// Element-wise subtraction: result = a - b
AmxMatrix *amx_matrix_sub(const AmxMatrix *a, const AmxMatrix *b);

/// Scalar multiplication: result = m * scalar
AmxMatrix *amx_matrix_scale(const AmxMatrix *m, float scalar);

#ifdef __cplusplus
}
#endif

#endif // AMX_H
