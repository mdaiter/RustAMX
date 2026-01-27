// Quick benchmark for AMX matmul
#include "Sources/CAMX/include/amx.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define ITERATIONS 100

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

int main(int argc, char **argv) {
    int n = 256;
    if (argc > 1) n = atoi(argv[1]);
    
    printf("AMX version: %d\n", amx_detect());
    printf("Matrix size: %dx%d\n", n, n);
    printf("Iterations: %d\n\n", ITERATIONS);
    
    AmxMatrix *a = amx_matrix_fill(n, n, 1.0f);
    AmxMatrix *b = amx_matrix_fill(n, n, 2.0f);
    
    // Warmup
    AmxMatrix *c = amx_matrix_matmul(a, b);
    amx_matrix_free(c);
    
    double start = get_time_ms();
    
    for (int i = 0; i < ITERATIONS; ++i) {
        c = amx_matrix_matmul(a, b);
        amx_matrix_free(c);
    }
    
    double elapsed = get_time_ms() - start;
    double per_iter = elapsed / ITERATIONS;
    
    // 2 * n^3 FLOPs for matmul
    double flops = 2.0 * n * n * n;
    double gflops = (flops / (per_iter / 1000.0)) / 1e9;
    
    printf("Results:\n");
    printf("  Total time: %.2f ms\n", elapsed);
    printf("  Per iteration: %.3f ms\n", per_iter);
    printf("  Throughput: %.2f GFLOPS\n", gflops);
    
    // Verify result
    c = amx_matrix_matmul(a, b);
    float expected = n * 2.0f;
    float actual = amx_matrix_get(c, 0, 0);
    printf("\nVerification: c[0,0] = %.1f (expected %.1f) %s\n", 
           actual, expected, (actual == expected) ? "OK" : "FAIL");
    
    amx_matrix_free(a);
    amx_matrix_free(b);
    amx_matrix_free(c);
    
    return 0;
}
