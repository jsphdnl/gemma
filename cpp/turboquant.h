/*
 * TurboQuant — KV Cache Compression in C
 *
 * Implements the TurboQuant algorithm:
 *   1. Normalize vector, store norm
 *   2. Rotate by random orthogonal matrix R
 *   3. Quantize each coordinate with Lloyd-Max codebook
 *   4. Pack indices into uint8
 *
 * Usage:
 *   tq_ctx ctx;
 *   tq_init(&ctx, 256, 4, 42);           // dim=256, 4-bit, seed=42
 *
 *   float vec[256] = {...};
 *   uint8_t packed[128];                  // 256 * 4bits / 8 = 128 bytes
 *   float norm;
 *   tq_encode(&ctx, vec, packed, &norm);  // compress
 *
 *   float decoded[256];
 *   tq_decode(&ctx, packed, norm, decoded); // decompress
 *
 *   tq_free(&ctx);
 */

#ifndef TURBOQUANT_H
#define TURBOQUANT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Max supported dimension and levels */
#define TQ_MAX_DIM    512
#define TQ_MAX_LEVELS 32   /* up to 5-bit */

typedef struct {
    int dim;
    int nbits;
    int n_levels;

    /* Rotation matrix R[dim][dim] (row-major) */
    float *R;

    /* Lloyd-Max codebook */
    float boundaries[TQ_MAX_LEVELS + 1];  /* n_levels + 1 decision boundaries */
    float levels[TQ_MAX_LEVELS];          /* n_levels reconstruction levels */

    /* Bytes per packed vector (excluding norm) */
    int packed_bytes;
} tq_ctx;

/* Initialize context: precompute rotation matrix, set codebook.
 * Returns 0 on success, -1 on failure. */
int  tq_init(tq_ctx *ctx, int dim, int nbits, uint64_t seed);

/* Free allocated memory */
void tq_free(tq_ctx *ctx);

/* Encode: vec[dim] -> packed[packed_bytes] + norm
 * packed must have at least ctx->packed_bytes bytes */
void tq_encode(const tq_ctx *ctx, const float *vec,
               uint8_t *packed, float *norm);

/* Decode: packed[packed_bytes] + norm -> out[dim] */
void tq_decode(const tq_ctx *ctx, const uint8_t *packed,
               float norm, float *out);

/* Compression ratio vs float32 baseline */
float tq_compression_ratio(const tq_ctx *ctx);

/* Bytes per vector (packed indices + 2 bytes norm) */
int tq_bytes_per_vector(const tq_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif /* TURBOQUANT_H */
