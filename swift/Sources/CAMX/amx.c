// amx.c - Apple AMX coprocessor C implementation
// Hyper-optimized: multi-threaded, assembly micro-kernel, zero-copy

#include "include/amx.h"
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include <pthread.h>
#include <dispatch/dispatch.h>

// ============================================================================
// Compiler Hints
// ============================================================================

#define ALWAYS_INLINE __attribute__((always_inline)) inline
#define NOINLINE __attribute__((noinline))
#define HOT __attribute__((hot))
#define COLD __attribute__((cold))
#define FLATTEN __attribute__((flatten))
#define ALIGNED(n) __attribute__((aligned(n)))
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#define PREFETCH_R(addr) __builtin_prefetch((addr), 0, 3)
#define PREFETCH_W(addr) __builtin_prefetch((addr), 1, 3)
#define ASSUME_ALIGNED(ptr, align) __builtin_assume_aligned((ptr), (align))
#define RESTRICT __restrict__

// ============================================================================
// AMX Constants
// ============================================================================

#define AMX_OP_BASE   0x00201000
#define AMX_ALIGN     64
#define AMX_TILE      16          // 16 floats = 64 bytes

#define AMX_OP_LDX    (AMX_OP_BASE | (0 << 5))
#define AMX_OP_LDY    (AMX_OP_BASE | (1 << 5))
#define AMX_OP_STX    (AMX_OP_BASE | (2 << 5))
#define AMX_OP_STY    (AMX_OP_BASE | (3 << 5))
#define AMX_OP_LDZ    (AMX_OP_BASE | (4 << 5))
#define AMX_OP_STZ    (AMX_OP_BASE | (5 << 5))
#define AMX_OP_FMA32  (AMX_OP_BASE | (12 << 5))
#define AMX_OP_SET    (AMX_OP_BASE | (17 << 5))
#define AMX_OP_CLR    (AMX_OP_BASE | (17 << 5) | 1)

// ============================================================================
// Detection
// ============================================================================

static AmxVersion g_amx_version = AMX_VERSION_NONE;
static pthread_once_t g_detect_once = PTHREAD_ONCE_INIT;
static int g_num_cores = 1;

COLD static void detect_amx_internal(void) {
    char brand[256] = {0};
    size_t size = sizeof(brand);
    
    if (sysctlbyname("machdep.cpu.brand_string", brand, &size, NULL, 0) != 0) {
        g_amx_version = AMX_VERSION_NONE;
        return;
    }
    
    if (!strstr(brand, "Apple")) {
        g_amx_version = AMX_VERSION_NONE;
        return;
    }
    
    if (strstr(brand, "M4"))      g_amx_version = AMX_VERSION_M4;
    else if (strstr(brand, "M3")) g_amx_version = AMX_VERSION_M3;
    else if (strstr(brand, "M2")) g_amx_version = AMX_VERSION_M2;
    else if (strstr(brand, "M1")) g_amx_version = AMX_VERSION_M1;
    else                          g_amx_version = AMX_VERSION_UNKNOWN;
    
    // Get number of performance cores
    size_t ncpu_size = sizeof(g_num_cores);
    sysctlbyname("hw.perflevel0.logicalcpu", &g_num_cores, &ncpu_size, NULL, 0);
    if (g_num_cores < 1) g_num_cores = 1;
    if (g_num_cores > 16) g_num_cores = 16;
}

AmxVersion amx_detect(void) {
    pthread_once(&g_detect_once, detect_amx_internal);
    return g_amx_version;
}

// ============================================================================
// AMX Primitives - Inline Assembly
// ============================================================================

// AMX SET/CLR require 3 NOPs before the instruction for pipeline safety
// Use newlines in asm string instead of semicolons to ensure all instructions are emitted
#define AMX_SET() \
    __asm__ volatile( \
        "nop\n" \
        "nop\n" \
        "nop\n" \
        ".word %0" \
        :: "i"(AMX_OP_SET) : "memory")

#define AMX_CLR() \
    __asm__ volatile( \
        "nop\n" \
        "nop\n" \
        "nop\n" \
        ".word %0" \
        :: "i"(AMX_OP_CLR) : "memory")

// Load/Store with operand encoding
// AMX instructions read operand from x0, so we must bind to x0 explicitly
#define AMX_LDX(addr, reg) do { \
    register uint64_t _op __asm__("x0") = ((uint64_t)(reg) << 56) | ((uint64_t)(addr) & 0x00FFFFFFFFFFFFFFULL); \
    __asm__ volatile(".word %0" :: "i"(AMX_OP_LDX), "r"(_op) : "memory"); \
} while(0)

#define AMX_LDY(addr, reg) do { \
    register uint64_t _op __asm__("x0") = ((uint64_t)(reg) << 56) | ((uint64_t)(addr) & 0x00FFFFFFFFFFFFFFULL); \
    __asm__ volatile(".word %0" :: "i"(AMX_OP_LDY), "r"(_op) : "memory"); \
} while(0)

#define AMX_LDZ(addr, row) do { \
    register uint64_t _op __asm__("x0") = ((uint64_t)(row) << 56) | ((uint64_t)(addr) & 0x00FFFFFFFFFFFFFFULL); \
    __asm__ volatile(".word %0" :: "i"(AMX_OP_LDZ), "r"(_op) : "memory"); \
} while(0)

#define AMX_STZ(addr, row) do { \
    register uint64_t _op __asm__("x0") = ((uint64_t)(row) << 56) | ((uint64_t)(addr) & 0x00FFFFFFFFFFFFFFULL); \
    __asm__ volatile(".word %0" :: "i"(AMX_OP_STZ), "r"(_op) : "memory"); \
} while(0)

// FMA32: Z[z_row*4..] += outer_product(X[x_off], Y[y_off])
#define AMX_FMA32(x_off, y_off, z_row) do { \
    register uint64_t _op __asm__("x0") = ((uint64_t)(z_row) << 20) | ((uint64_t)(x_off) << 10) | (uint64_t)(y_off); \
    __asm__ volatile(".word %0" :: "i"(AMX_OP_FMA32), "r"(_op) : "memory"); \
} while(0)

// For header compatibility
void amx_set(void) { AMX_SET(); }
void amx_clr(void) { AMX_CLR(); }

#define DEFINE_AMX_FUNC(name, opcode) \
    void name(uint64_t operand) { \
        register uint64_t x0 __asm__("x0") = operand; \
        __asm__ volatile(".word %1" : "+r"(x0) : "i"(opcode) : "memory"); \
    }

DEFINE_AMX_FUNC(amx_ldx, AMX_OP_LDX)
DEFINE_AMX_FUNC(amx_ldy, AMX_OP_LDY)
DEFINE_AMX_FUNC(amx_ldz, AMX_OP_LDZ)
DEFINE_AMX_FUNC(amx_stx, AMX_OP_STX)
DEFINE_AMX_FUNC(amx_sty, AMX_OP_STY)
DEFINE_AMX_FUNC(amx_stz, AMX_OP_STZ)
DEFINE_AMX_FUNC(amx_ldzi, AMX_OP_BASE | (6 << 5))
DEFINE_AMX_FUNC(amx_stzi, AMX_OP_BASE | (7 << 5))
DEFINE_AMX_FUNC(amx_extrx, AMX_OP_BASE | (8 << 5))
DEFINE_AMX_FUNC(amx_extry, AMX_OP_BASE | (9 << 5))
DEFINE_AMX_FUNC(amx_fma64, AMX_OP_BASE | (10 << 5))
DEFINE_AMX_FUNC(amx_fms64, AMX_OP_BASE | (11 << 5))
DEFINE_AMX_FUNC(amx_fma32, AMX_OP_FMA32)
DEFINE_AMX_FUNC(amx_fms32, AMX_OP_BASE | (13 << 5))
DEFINE_AMX_FUNC(amx_mac16, AMX_OP_BASE | (14 << 5))
DEFINE_AMX_FUNC(amx_fma16, AMX_OP_BASE | (15 << 5))
DEFINE_AMX_FUNC(amx_fms16, AMX_OP_BASE | (16 << 5))
DEFINE_AMX_FUNC(amx_vecint, AMX_OP_BASE | (18 << 5))
DEFINE_AMX_FUNC(amx_vecfp, AMX_OP_BASE | (19 << 5))
DEFINE_AMX_FUNC(amx_matint, AMX_OP_BASE | (20 << 5))
DEFINE_AMX_FUNC(amx_matfp, AMX_OP_BASE | (21 << 5))
DEFINE_AMX_FUNC(amx_genlut, AMX_OP_BASE | (22 << 5))

// ============================================================================
// Memory
// ============================================================================

ALWAYS_INLINE static size_t round_up(size_t n, size_t align) {
    return (n + align - 1) & ~(align - 1);
}

ALWAYS_INLINE static void *alloc_aligned(size_t size) {
    void *p;
    return posix_memalign(&p, AMX_ALIGN, size) == 0 ? p : NULL;
}

// ============================================================================
// Matrix - Single storage, user responsible for format
// ============================================================================

struct AmxMatrix {
    float *RESTRICT data;    // 64-byte aligned, row-major with padded stride
    size_t rows;
    size_t cols;
    size_t stride;           // >= cols, multiple of 16
};

AmxMatrix *amx_matrix_zeros(size_t rows, size_t cols) {
    if (UNLIKELY(!rows || !cols)) return NULL;
    
    AmxMatrix *m = malloc(sizeof(AmxMatrix));
    if (UNLIKELY(!m)) return NULL;
    
    m->rows = rows;
    m->cols = cols;
    m->stride = round_up(cols, AMX_TILE);
    
    size_t bytes = rows * m->stride * sizeof(float);
    m->data = alloc_aligned(bytes);
    if (UNLIKELY(!m->data)) { free(m); return NULL; }
    
    memset(m->data, 0, bytes);
    return m;
}

AmxMatrix *amx_matrix_fill(size_t rows, size_t cols, float value) {
    AmxMatrix *m = amx_matrix_zeros(rows, cols);
    if (UNLIKELY(!m)) return NULL;
    
    float *RESTRICT p = m->data;
    const size_t stride = m->stride;
    for (size_t i = 0; i < rows; ++i) {
        float *RESTRICT row = p + i * stride;
        for (size_t j = 0; j < cols; ++j) row[j] = value;
    }
    return m;
}

AmxMatrix *amx_matrix_identity(size_t n) {
    AmxMatrix *m = amx_matrix_zeros(n, n);
    if (UNLIKELY(!m)) return NULL;
    
    float *RESTRICT p = m->data;
    const size_t stride = m->stride;
    for (size_t i = 0; i < n; ++i) p[i * stride + i] = 1.0f;
    return m;
}

AmxMatrix *amx_matrix_from_data(size_t rows, size_t cols, const float *RESTRICT data) {
    if (UNLIKELY(!data)) return NULL;
    
    AmxMatrix *m = amx_matrix_zeros(rows, cols);
    if (UNLIKELY(!m)) return NULL;
    
    float *RESTRICT dst = m->data;
    const size_t stride = m->stride;
    for (size_t i = 0; i < rows; ++i) {
        memcpy(dst + i * stride, data + i * cols, cols * sizeof(float));
    }
    return m;
}

AmxMatrix *amx_matrix_from_owned(size_t rows, size_t cols, float *data) {
    AmxMatrix *m = amx_matrix_from_data(rows, cols, data);
    free(data);
    return m;
}

AmxMatrix *amx_matrix_clone(const AmxMatrix *m) {
    if (UNLIKELY(!m)) return NULL;
    AmxMatrix *c = amx_matrix_zeros(m->rows, m->cols);
    if (UNLIKELY(!c)) return NULL;
    memcpy(c->data, m->data, m->rows * m->stride * sizeof(float));
    return c;
}

void amx_matrix_free(AmxMatrix *m) {
    if (m) { free(m->data); free(m); }
}

size_t amx_matrix_rows(const AmxMatrix *m) { return m ? m->rows : 0; }
size_t amx_matrix_cols(const AmxMatrix *m) { return m ? m->cols : 0; }
size_t amx_matrix_stride(const AmxMatrix *m) { return m ? m->stride : 0; }
const float *amx_matrix_data(const AmxMatrix *m) { return m ? m->data : NULL; }
float *amx_matrix_data_mut(AmxMatrix *m) { return m ? m->data : NULL; }
float amx_matrix_get(const AmxMatrix *m, size_t r, size_t c) { return m->data[r * m->stride + c]; }
void amx_matrix_set(AmxMatrix *m, size_t r, size_t c, float v) { m->data[r * m->stride + c] = v; }

// ============================================================================
// AMX Micro-kernel: 16x16 output tile, processes full K dimension
// Optimized: skip C load (starts from zero), prefetch next B, unrolled k-loop
// ============================================================================

// Zero the Z accumulator registers (16 rows for f32 mode)
#define AMX_ZERO_Z() do { \
    static const float _zeros[16] __attribute__((aligned(64))) = {0}; \
    AMX_LDZ(_zeros, 0);  AMX_LDZ(_zeros, 4);  AMX_LDZ(_zeros, 8);  AMX_LDZ(_zeros, 12); \
    AMX_LDZ(_zeros, 16); AMX_LDZ(_zeros, 20); AMX_LDZ(_zeros, 24); AMX_LDZ(_zeros, 28); \
    AMX_LDZ(_zeros, 32); AMX_LDZ(_zeros, 36); AMX_LDZ(_zeros, 40); AMX_LDZ(_zeros, 44); \
    AMX_LDZ(_zeros, 48); AMX_LDZ(_zeros, 52); AMX_LDZ(_zeros, 56); AMX_LDZ(_zeros, 60); \
} while(0)

// ============================================================================
// Pack A into column-major panels for microkernel
// Panel: 16 rows x K cols, tightly packed (stride 16)
// Uses NEON for vectorized gather when possible
// ============================================================================

#include <arm_neon.h>

HOT static void pack_a_panel(
    const float *RESTRICT A,
    float *RESTRICT panel,
    size_t M_start,
    size_t M_end,
    size_t K,
    size_t a_stride
) {
    const size_t rows = M_end - M_start;
    const float *RESTRICT src_base = A + M_start * a_stride;
    
    // Full 16-row case (most common) - use vectorized gather
    if (LIKELY(rows == 16)) {
        for (size_t k = 0; k < K; ++k) {
            float *RESTRICT dst = panel + k * 16;
            const float *RESTRICT src = src_base + k;
            
            // Gather 16 elements with stride a_stride
            // Load 4 elements at a time using scalar loads (NEON doesn't have strided gather)
            dst[0]  = src[0 * a_stride];
            dst[1]  = src[1 * a_stride];
            dst[2]  = src[2 * a_stride];
            dst[3]  = src[3 * a_stride];
            dst[4]  = src[4 * a_stride];
            dst[5]  = src[5 * a_stride];
            dst[6]  = src[6 * a_stride];
            dst[7]  = src[7 * a_stride];
            dst[8]  = src[8 * a_stride];
            dst[9]  = src[9 * a_stride];
            dst[10] = src[10 * a_stride];
            dst[11] = src[11 * a_stride];
            dst[12] = src[12 * a_stride];
            dst[13] = src[13 * a_stride];
            dst[14] = src[14 * a_stride];
            dst[15] = src[15 * a_stride];
        }
    } else {
        // Partial row case - need to zero pad
        const float32x4_t zeros = vdupq_n_f32(0.0f);
        for (size_t k = 0; k < K; ++k) {
            float *RESTRICT dst = panel + k * 16;
            const float *RESTRICT src = src_base + k;
            
            // Copy valid rows
            for (size_t i = 0; i < rows; ++i) {
                dst[i] = src[i * a_stride];
            }
            // Zero remaining using NEON
            size_t i = rows;
            for (; i + 4 <= 16; i += 4) {
                vst1q_f32(dst + i, zeros);
            }
            for (; i < 16; ++i) {
                dst[i] = 0.0f;
            }
        }
    }
}

// ============================================================================
// Single-threaded matmul (for small matrices or single tile)
// ============================================================================

COLD static void matmul_naive(
    const AmxMatrix *RESTRICT a,
    const AmxMatrix *RESTRICT b,
    AmxMatrix *RESTRICT c
) {
    const size_t M = a->rows, K = a->cols, N = b->cols;
    const float *RESTRICT ap = a->data;
    const float *RESTRICT bp = b->data;
    float *RESTRICT cp = c->data;
    const size_t as = a->stride, bs = b->stride, cs = c->stride;
    
    memset(cp, 0, M * cs * sizeof(float));
    
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float aik = ap[i * as + k];
            for (size_t j = 0; j < N; ++j) {
                cp[i * cs + j] += aik * bp[k * bs + j];
            }
        }
    }
}

// ============================================================================
// Multi-threaded AMX Matmul
// ============================================================================

typedef struct {
    const float *RESTRICT A;
    const float *RESTRICT B;
    float *RESTRICT C;
    float *RESTRICT a_panel;  // Thread-local packed A panel
    size_t M, K, N;
    size_t a_stride, b_stride, c_stride;
    size_t i_start, i_end;    // Row tile range for this thread
} MatmulTask;

// Microkernel that works with strided B (original, faster for our case)
HOT FLATTEN static void microkernel_16x16_strided(
    const float *RESTRICT A,    // Column-major panel: 16 rows x K cols, stride 16
    const float *RESTRICT B,    // Row-major: K rows x N cols, stride b_stride
    float *RESTRICT C,          // Row-major: 16 rows, stride c_stride
    size_t K,
    size_t b_stride,
    size_t c_stride
) {
    // Zero Z registers - C starts at zero so no need to load
    AMX_ZERO_Z();
    
    // Process K in chunks of 8 (use all 8 X and Y registers)
    size_t k = 0;
    for (; k + 8 <= K; k += 8) {
        const float *RESTRICT a_ptr = A + k * 16;
        const float *RESTRICT b_ptr = B + k * b_stride;
        
        // Prefetch next iteration
        PREFETCH_R(a_ptr + 8 * 16);
        PREFETCH_R(b_ptr + 8 * b_stride);
        
        // Load 8 columns of A into Y0-Y7
        AMX_LDY(a_ptr + 0 * 16, 0);
        AMX_LDY(a_ptr + 1 * 16, 1);
        AMX_LDY(a_ptr + 2 * 16, 2);
        AMX_LDY(a_ptr + 3 * 16, 3);
        AMX_LDY(a_ptr + 4 * 16, 4);
        AMX_LDY(a_ptr + 5 * 16, 5);
        AMX_LDY(a_ptr + 6 * 16, 6);
        AMX_LDY(a_ptr + 7 * 16, 7);
        
        // Load 8 rows of B and FMA (interleaved)
        AMX_LDX(b_ptr + 0 * b_stride, 0);
        AMX_LDX(b_ptr + 1 * b_stride, 1);
        AMX_FMA32(0 * 64, 0 * 64, 0);
        
        AMX_LDX(b_ptr + 2 * b_stride, 2);
        AMX_FMA32(1 * 64, 1 * 64, 0);
        
        AMX_LDX(b_ptr + 3 * b_stride, 3);
        AMX_FMA32(2 * 64, 2 * 64, 0);
        
        AMX_LDX(b_ptr + 4 * b_stride, 4);
        AMX_FMA32(3 * 64, 3 * 64, 0);
        
        AMX_LDX(b_ptr + 5 * b_stride, 5);
        AMX_FMA32(4 * 64, 4 * 64, 0);
        
        AMX_LDX(b_ptr + 6 * b_stride, 6);
        AMX_FMA32(5 * 64, 5 * 64, 0);
        
        AMX_LDX(b_ptr + 7 * b_stride, 7);
        AMX_FMA32(6 * 64, 6 * 64, 0);
        AMX_FMA32(7 * 64, 7 * 64, 0);
    }
    
    // Remainder
    for (; k < K; ++k) {
        AMX_LDY(A + k * 16, 0);
        AMX_LDX(B + k * b_stride, 0);
        AMX_FMA32(0, 0, 0);
    }
    
    // Store C tile
    AMX_STZ(C + 0  * c_stride, 0);
    AMX_STZ(C + 1  * c_stride, 4);
    AMX_STZ(C + 2  * c_stride, 8);
    AMX_STZ(C + 3  * c_stride, 12);
    AMX_STZ(C + 4  * c_stride, 16);
    AMX_STZ(C + 5  * c_stride, 20);
    AMX_STZ(C + 6  * c_stride, 24);
    AMX_STZ(C + 7  * c_stride, 28);
    AMX_STZ(C + 8  * c_stride, 32);
    AMX_STZ(C + 9  * c_stride, 36);
    AMX_STZ(C + 10 * c_stride, 40);
    AMX_STZ(C + 11 * c_stride, 44);
    AMX_STZ(C + 12 * c_stride, 48);
    AMX_STZ(C + 13 * c_stride, 52);
    AMX_STZ(C + 14 * c_stride, 56);
    AMX_STZ(C + 15 * c_stride, 60);
}

HOT static void matmul_thread_func(void *ctx) {
    MatmulTask *t = (MatmulTask *)ctx;
    
    const float *RESTRICT A = t->A;
    const float *RESTRICT B = t->B;
    float *RESTRICT C = t->C;
    float *RESTRICT a_panel = t->a_panel;
    
    const size_t K = t->K;
    const size_t N = t->N;
    const size_t a_stride = t->a_stride;
    const size_t b_stride = t->b_stride;
    const size_t c_stride = t->c_stride;
    
    AMX_SET();
    
    // Process assigned row tiles
    for (size_t i = t->i_start; i < t->i_end; i += AMX_TILE) {
        const size_t i_end = (i + AMX_TILE <= t->i_end) ? i + AMX_TILE : t->i_end;
        
        // Pack this row panel of A once (16 rows x K cols -> column-major)
        pack_a_panel(A, a_panel, i, i_end, K, a_stride);
        
        // Process all column tiles
        for (size_t j = 0; j < N; j += AMX_TILE) {
            const size_t j_end = (j + AMX_TILE <= N) ? j + AMX_TILE : N;
            float *RESTRICT c_tile = C + i * c_stride + j;
            const float *RESTRICT b_tile = B + j;
            
            if (LIKELY(i_end - i == AMX_TILE && j_end - j == AMX_TILE)) {
                // Full 16x16 tile
                microkernel_16x16_strided(a_panel, b_tile, c_tile, K, b_stride, c_stride);
            } else {
                // Edge tile - use scalar fallback
                const size_t mi = i_end - i;
                const size_t nj = j_end - j;
                
                for (size_t ii = 0; ii < mi; ++ii) {
                    for (size_t kk = 0; kk < K; ++kk) {
                        float a_val = a_panel[kk * 16 + ii];
                        const float *RESTRICT b_row = B + kk * b_stride + j;
                        for (size_t jj = 0; jj < nj; ++jj) {
                            c_tile[ii * c_stride + jj] += a_val * b_row[jj];
                        }
                    }
                }
            }
        }
    }
    
    AMX_CLR();
}

HOT static void matmul_amx_parallel(
    const AmxMatrix *RESTRICT a,
    const AmxMatrix *RESTRICT b,
    AmxMatrix *RESTRICT c
) {
    const size_t M = a->rows;
    const size_t K = a->cols;
    const size_t N = b->cols;
    
    // Zero output
    memset(c->data, 0, M * c->stride * sizeof(float));
    
    // Determine thread count
    const size_t m_tiles = (M + AMX_TILE - 1) / AMX_TILE;
    int num_threads = (int)m_tiles;
    if (num_threads > g_num_cores) num_threads = g_num_cores;
    if (num_threads < 1) num_threads = 1;
    
    // For small matrices, single-thread is faster (no dispatch overhead)
    if (M <= 64 || num_threads == 1) {
        // Allocate panel buffer
        float *a_panel = alloc_aligned(K * 16 * sizeof(float));
        if (!a_panel) { 
            matmul_naive(a, b, c); 
            return; 
        }
        
        MatmulTask task = {
            .A = a->data, .B = b->data, .C = c->data,
            .a_panel = a_panel,
            .M = M, .K = K, .N = N,
            .a_stride = a->stride, .b_stride = b->stride, .c_stride = c->stride,
            .i_start = 0, .i_end = M
        };
        matmul_thread_func(&task);
        free(a_panel);
        return;
    }
    
    // Multi-threaded: distribute row tiles across threads
    MatmulTask *tasks = malloc(num_threads * sizeof(MatmulTask));
    float **a_panels = malloc(num_threads * sizeof(float *));
    if (!tasks || !a_panels) {
        free(tasks); free(a_panels);
        matmul_naive(a, b, c);
        return;
    }
    
    // Allocate per-thread panel buffers
    for (int t = 0; t < num_threads; ++t) {
        a_panels[t] = alloc_aligned(K * 16 * sizeof(float));
        if (!a_panels[t]) {
            for (int i = 0; i < t; ++i) free(a_panels[i]);
            free(a_panels); free(tasks);
            matmul_naive(a, b, c);
            return;
        }
    }
    
    // Distribute work
    size_t rows_per_thread = ((M + AMX_TILE - 1) / AMX_TILE / num_threads) * AMX_TILE;
    if (rows_per_thread < AMX_TILE) rows_per_thread = AMX_TILE;
    
    for (int t = 0; t < num_threads; ++t) {
        tasks[t] = (MatmulTask){
            .A = a->data, .B = b->data, .C = c->data,
            .a_panel = a_panels[t],
            .M = M, .K = K, .N = N,
            .a_stride = a->stride, .b_stride = b->stride, .c_stride = c->stride,
            .i_start = t * rows_per_thread,
            .i_end = (t == num_threads - 1) ? M : (t + 1) * rows_per_thread
        };
        if (tasks[t].i_start >= M) tasks[t].i_start = tasks[t].i_end = M;
    }
    
    // Dispatch using GCD
    dispatch_queue_t queue = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
    dispatch_apply(num_threads, queue, ^(size_t t) {
        if (tasks[t].i_start < tasks[t].i_end) {
            matmul_thread_func(&tasks[t]);
        }
    });
    
    // Cleanup
    for (int t = 0; t < num_threads; ++t) free(a_panels[t]);
    free(a_panels);
    free(tasks);
}

// ============================================================================
// Public API
// ============================================================================

AmxMatrix *amx_matrix_matmul(const AmxMatrix *a, const AmxMatrix *b) {
    if (UNLIKELY(!a || !b || a->cols != b->rows)) return NULL;
    
    AmxMatrix *c = amx_matrix_zeros(a->rows, b->cols);
    if (UNLIKELY(!c)) return NULL;
    
    if (LIKELY(amx_is_available() && a->rows >= AMX_TILE && b->cols >= AMX_TILE)) {
        matmul_amx_parallel(a, b, c);
    } else {
        matmul_naive(a, b, c);
    }
    
    return c;
}

AmxMatrix *amx_matrix_transpose(const AmxMatrix *m) {
    if (UNLIKELY(!m)) return NULL;
    
    AmxMatrix *r = amx_matrix_zeros(m->cols, m->rows);
    if (UNLIKELY(!r)) return NULL;
    
    const float *RESTRICT s = m->data;
    float *RESTRICT d = r->data;
    const size_t ss = m->stride, ds = r->stride;
    
    for (size_t i = 0; i < m->rows; ++i) {
        for (size_t j = 0; j < m->cols; ++j) {
            d[j * ds + i] = s[i * ss + j];
        }
    }
    return r;
}

AmxMatrix *amx_matrix_add(const AmxMatrix *a, const AmxMatrix *b) {
    if (UNLIKELY(!a || !b || a->rows != b->rows || a->cols != b->cols)) return NULL;
    
    AmxMatrix *c = amx_matrix_zeros(a->rows, a->cols);
    if (UNLIKELY(!c)) return NULL;
    
    const float *RESTRICT ap = a->data, *RESTRICT bp = b->data;
    float *RESTRICT cp = c->data;
    const size_t as = a->stride, bs = b->stride, cs = c->stride;
    
    for (size_t i = 0; i < a->rows; ++i) {
        for (size_t j = 0; j < a->cols; ++j) {
            cp[i * cs + j] = ap[i * as + j] + bp[i * bs + j];
        }
    }
    return c;
}

AmxMatrix *amx_matrix_sub(const AmxMatrix *a, const AmxMatrix *b) {
    if (UNLIKELY(!a || !b || a->rows != b->rows || a->cols != b->cols)) return NULL;
    
    AmxMatrix *c = amx_matrix_zeros(a->rows, a->cols);
    if (UNLIKELY(!c)) return NULL;
    
    const float *RESTRICT ap = a->data, *RESTRICT bp = b->data;
    float *RESTRICT cp = c->data;
    const size_t as = a->stride, bs = b->stride, cs = c->stride;
    
    for (size_t i = 0; i < a->rows; ++i) {
        for (size_t j = 0; j < a->cols; ++j) {
            cp[i * cs + j] = ap[i * as + j] - bp[i * bs + j];
        }
    }
    return c;
}

AmxMatrix *amx_matrix_scale(const AmxMatrix *m, float s) {
    if (UNLIKELY(!m)) return NULL;
    
    AmxMatrix *r = amx_matrix_zeros(m->rows, m->cols);
    if (UNLIKELY(!r)) return NULL;
    
    const float *RESTRICT sp = m->data;
    float *RESTRICT dp = r->data;
    const size_t ss = m->stride, ds = r->stride;
    
    for (size_t i = 0; i < m->rows; ++i) {
        for (size_t j = 0; j < m->cols; ++j) {
            dp[i * ds + j] = sp[i * ss + j] * s;
        }
    }
    return r;
}
