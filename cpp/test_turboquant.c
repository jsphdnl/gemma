/*
 * Test TurboQuant C implementation.
 * Compiles with: cc -O2 -o test_turboquant test_turboquant.c turboquant.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "turboquant.h"

/* Simple PRNG for test data */
static unsigned int test_seed = 12345;
static float test_randf(void) {
    test_seed = test_seed * 1103515245 + 12345;
    return ((float)(test_seed >> 16) / 32768.0f) - 1.0f;
}

static float vec_mse(const float *a, const float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

static float vec_norm(const float *a, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * a[i];
    return sqrtf(sum);
}

int main(void) {
    printf("TurboQuant C Implementation Test\n");
    printf("=================================\n\n");

    int configs[][2] = {{128, 4}, {128, 3}, {128, 2}, {256, 4}, {256, 3}, {256, 2}};
    int n_configs = sizeof(configs) / sizeof(configs[0]);

    for (int c = 0; c < n_configs; c++) {
        int dim = configs[c][0];
        int nbits = configs[c][1];

        tq_ctx ctx;
        if (tq_init(&ctx, dim, nbits, 42) != 0) {
            printf("  dim=%d nbits=%d: INIT FAILED\n", dim, nbits);
            continue;
        }

        /* Test roundtrip on 100 random vectors */
        int n_vecs = 100;
        float total_mse = 0.0f;
        float total_norm_sq = 0.0f;

        uint8_t *packed = (uint8_t *)malloc(ctx.packed_bytes);
        float *vec = (float *)malloc(dim * sizeof(float));
        float *decoded = (float *)malloc(dim * sizeof(float));

        for (int v = 0; v < n_vecs; v++) {
            /* Random vector */
            for (int i = 0; i < dim; i++)
                vec[i] = test_randf();

            float norm;
            tq_encode(&ctx, vec, packed, &norm);
            tq_decode(&ctx, packed, norm, decoded);

            /* MSE on unit vectors */
            float vn = vec_norm(vec, dim);
            if (vn > 1e-10f) {
                float inv = 1.0f / vn;
                float mse = 0.0f;
                for (int i = 0; i < dim; i++) {
                    float d = vec[i] * inv - decoded[i] / norm;
                    mse += d * d;
                }
                total_mse += mse;
                total_norm_sq += 1.0f;
            }
        }

        float avg_mse = total_mse / n_vecs;
        float ratio = tq_compression_ratio(&ctx);
        int bpv = tq_bytes_per_vector(&ctx);

        printf("  dim=%3d  %d-bit: MSE=%.5f  compress=%.1fx  (%d vs %d bytes/vec)\n",
               dim, nbits, avg_mse, ratio, bpv, dim * 4);

        free(packed);
        free(vec);
        free(decoded);
        tq_free(&ctx);
    }

    /* Benchmark: encode/decode throughput */
    printf("\nBenchmark (dim=256, 4-bit, 10000 vectors):\n");
    {
        int dim = 256;
        tq_ctx ctx;
        tq_init(&ctx, dim, 4, 42);

        float *vec = (float *)malloc(dim * sizeof(float));
        uint8_t *packed = (uint8_t *)malloc(ctx.packed_bytes);
        float *decoded = (float *)malloc(dim * sizeof(float));
        float norm;

        for (int i = 0; i < dim; i++) vec[i] = test_randf();

        int n_iters = 10000;

        /* Encode benchmark */
        clock_t t0 = clock();
        for (int i = 0; i < n_iters; i++)
            tq_encode(&ctx, vec, packed, &norm);
        clock_t t1 = clock();
        double encode_ms = 1000.0 * (double)(t1 - t0) / CLOCKS_PER_SEC;

        /* Decode benchmark */
        t0 = clock();
        for (int i = 0; i < n_iters; i++)
            tq_decode(&ctx, packed, norm, decoded);
        t1 = clock();
        double decode_ms = 1000.0 * (double)(t1 - t0) / CLOCKS_PER_SEC;

        printf("  Encode: %.1f ms / %d vecs = %.3f us/vec\n",
               encode_ms, n_iters, encode_ms * 1000.0 / n_iters);
        printf("  Decode: %.1f ms / %d vecs = %.3f us/vec\n",
               decode_ms, n_iters, decode_ms * 1000.0 / n_iters);
        printf("  Throughput: %.0f encodes/s, %.0f decodes/s\n",
               n_iters / (encode_ms / 1000.0), n_iters / (decode_ms / 1000.0));

        free(vec);
        free(packed);
        free(decoded);
        tq_free(&ctx);
    }

    printf("\nDone.\n");
    return 0;
}
