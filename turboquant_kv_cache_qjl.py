"""
TurboQuant-style KV Cache Compression Prototype

Implements coarse quantization + QJL sketch correction for KV cache compression
in a minimal transformer attention pipeline.

Key idea:
  - Store K_hat = quantize(K), V_hat = quantize(V)  (coarse, low-bit)
  - Store QJL sketch of residual: sign(A @ (K - K_hat))
  - At attention time: scores ≈ Q @ K_hat.T + QJL_correction
  - QJL_correction ≈ (A @ Q).T @ sign(A @ r_K) / m
"""

import torch
import torch.nn.functional as F


# =============================================================================
# Phase 1, Step 1: Toy data
# =============================================================================

def make_toy_data(batch=1, tokens=8, dim=16, seed=42):
    """Generate random Q, K, V tensors for testing."""
    torch.manual_seed(seed)
    Q = torch.randn(batch, tokens, dim)
    K = torch.randn(batch, tokens, dim)
    V = torch.randn(batch, tokens, dim)
    return Q, K, V


# =============================================================================
# Phase 1, Step 2: Baseline attention
# =============================================================================

def baseline_attention(Q, K, V):
    """Standard scaled dot-product attention.

    Args:
        Q: (batch, tokens_q, dim)
        K: (batch, tokens_k, dim)
        V: (batch, tokens_k, dim)

    Returns:
        output: (batch, tokens_q, dim)
        scores: (batch, tokens_q, tokens_k) — raw logits before softmax
    """
    # scores: (batch, tokens_q, tokens_k)
    scores = Q @ K.transpose(-1, -2)
    weights = F.softmax(scores, dim=-1)  # (batch, tokens_q, tokens_k)
    output = weights @ V                 # (batch, tokens_q, dim)
    return output, scores


# =============================================================================
# Phase 1, Step 3: Coarse symmetric quantization
# =============================================================================

def quantize(x, bits=3):
    """Symmetric per-tensor uniform quantization.

    Maps x into 2^bits levels symmetrically around zero.

    Args:
        x: arbitrary tensor
        bits: quantization bit-width (2, 3, or 4)

    Returns:
        x_hat: dequantized approximation (same shape as x, full precision)
    """
    n_levels = 2 ** bits
    half_levels = n_levels // 2  # e.g. bits=3 → 4 positive levels

    # Scale factor: map [-max, +max] to [-half_levels, +half_levels]
    x_max = x.abs().max()
    if x_max == 0:
        return x.clone()

    scale = x_max / half_levels

    # Quantize: round to nearest integer level, clamp
    x_int = torch.clamp(torch.round(x / scale), -half_levels, half_levels)

    # Dequantize back to float
    x_hat = x_int * scale
    return x_hat


# =============================================================================
# Phase 1, Step 5: QJL sketch
# =============================================================================

class QJLSketch:
    """Quantized Johnson-Lindenstrauss sketch for residual correction.

    Stores sign(A @ r.T) where A is a random ±1/√d projection matrix
    and r is the quantization residual.

    Args:
        m: number of random projections (sketch dimension)
        d: feature dimension of the vectors being sketched
        seed: random seed for reproducibility
    """

    def __init__(self, m, d, seed=123):
        self.m = m
        self.d = d

        # Random projection matrix A: (m, d)
        # Entries are ±1/√d (Rademacher scaled)
        gen = torch.Generator()
        gen.manual_seed(seed)
        signs = torch.randint(0, 2, (m, d), generator=gen).float() * 2 - 1  # ±1
        self.A = signs / (d ** 0.5)  # (m, d)

    def sketch(self, r):
        """Compute sign sketch of residual vectors.

        Args:
            r: (batch, tokens, d) — residual = original - quantized

        Returns:
            sketch: (batch, m, tokens) — sign(A @ r.T)
        """
        # r.transpose(-1, -2): (batch, d, tokens)
        # A @ r.T: (m, d) @ (batch, d, tokens) → need broadcast
        # Use einsum for clarity: A is (m, d), r is (batch, tokens, d)
        # Result should be (batch, m, tokens)
        Ar = torch.einsum("md, btd -> bmt", self.A, r)  # (batch, m, tokens)
        return torch.sign(Ar)  # (batch, m, tokens)

    def correct_scores(self, Q, sketch_K):
        """Compute QJL correction term for attention scores.

        correction = (A @ Q.T).T @ sketch_K / m
                   = Q @ A.T @ sketch_K / m

        Conceptually: each column of sketch_K is sign(A @ r_k) for one key token.
        We project Q through A, then dot with the sketch to approximate Q @ r_K.T.

        Args:
            Q: (batch, tokens_q, d)
            sketch_K: (batch, m, tokens_k)

        Returns:
            correction: (batch, tokens_q, tokens_k)
        """
        # AQ = A @ Q.T: (m, d) @ (batch, d, tokens_q) → (batch, m, tokens_q)
        AQ = torch.einsum("md, bqd -> bmq", self.A, Q)  # (batch, m, tokens_q)

        # correction = AQ.T @ sketch_K / m
        # AQ: (batch, m, tokens_q), sketch_K: (batch, m, tokens_k)
        # AQ.transpose → (batch, tokens_q, m) @ (batch, m, tokens_k) → (batch, tokens_q, tokens_k)
        correction = torch.einsum("bmq, bmt -> bqt", AQ, sketch_K) / self.m

        return correction  # (batch, tokens_q, tokens_k)

    def correct_values(self, weights, sketch_V):
        """Compute QJL correction term for value aggregation.

        For V correction we need: weights @ r_V ≈ weights @ A.T @ sketch_V.T ...
        Actually: correction_V = (weights @ sketch_V.T) @ A / m

        But sketch_V is sign(A @ r_V.T) with shape (batch, m, tokens_k).
        sketch_V.T in the tokens/m sense: (batch, tokens_k, m)

        correction = weights @ sketch_V.transpose(-1,-2) @ A / m
                   = (batch, tq, tk) @ (batch, tk, m) @ (m, d) / m
                   = (batch, tq, d)

        Args:
            weights: (batch, tokens_q, tokens_k) — attention weights (after softmax)
            sketch_V: (batch, m, tokens_k) — sign sketch of V residual

        Returns:
            correction: (batch, tokens_q, d)
        """
        # weights @ sketch_V.T: (batch, tq, tk) @ (batch, tk, m) → (batch, tq, m)
        w_sketch = weights @ sketch_V.transpose(-1, -2)  # (batch, tq, m)

        # w_sketch @ A: (batch, tq, m) @ (m, d) → (batch, tq, d)
        correction = torch.einsum("bqm, md -> bqd", w_sketch, self.A) / self.m

        return correction  # (batch, tokens_q, d)


# =============================================================================
# Phase 1, Steps 6-7: Approximate attention with QJL-corrected scores
# =============================================================================

def qjl_attention(Q, K, V, bits=3, m=32, seed=123, correct_v=False):
    """Attention with quantized KV cache + QJL residual correction.

    Pipeline:
      1. Quantize K → K_hat, V → V_hat
      2. Compute residuals r_K = K - K_hat, r_V = V - V_hat
      3. Sketch residuals: sign(A @ r.T)
      4. Approximate scores = Q @ K_hat.T + QJL_correction_K
      5. Approximate output = softmax(scores) @ V_hat [+ QJL_correction_V]

    Args:
        Q: (batch, tokens_q, dim)
        K: (batch, tokens_k, dim)
        V: (batch, tokens_k, dim)
        bits: quantization bit-width
        m: QJL sketch dimension (number of random projections)
        seed: random seed for projection matrix
        correct_v: if True, also apply QJL correction for V

    Returns:
        output_approx: (batch, tokens_q, dim)
        scores_approx: (batch, tokens_q, tokens_k)
    """
    d = Q.shape[-1]

    # Step 3: Quantize K and V
    K_hat = quantize(K, bits=bits)  # (batch, tokens_k, dim)
    V_hat = quantize(V, bits=bits)  # (batch, tokens_k, dim)

    # Step 4: Compute residuals
    r_K = K - K_hat  # (batch, tokens_k, dim)
    r_V = V - V_hat  # (batch, tokens_k, dim)

    # Step 5: Create QJL sketch
    qjl = QJLSketch(m=m, d=d, seed=seed)
    sketch_K = qjl.sketch(r_K)  # (batch, m, tokens_k)
    sketch_V = qjl.sketch(r_V)  # (batch, m, tokens_k)

    # Step 6: Coarse scores + QJL correction
    coarse_scores = Q @ K_hat.transpose(-1, -2)     # (batch, tq, tk)
    correction_K = qjl.correct_scores(Q, sketch_K)  # (batch, tq, tk)
    scores_approx = coarse_scores + correction_K     # (batch, tq, tk)

    # Step 7: Attention weights and output
    weights_approx = F.softmax(scores_approx, dim=-1)  # (batch, tq, tk)
    output_approx = weights_approx @ V_hat              # (batch, tq, dim)

    if correct_v:
        correction_V = qjl.correct_values(weights_approx, sketch_V)
        output_approx = output_approx + correction_V

    return output_approx, scores_approx


# =============================================================================
# Phase 1, Step 8: Comparison metrics
# =============================================================================

def compare(baseline_scores, approx_scores, baseline_output, approx_output, label=""):
    """Print error metrics between baseline and approximate results."""
    score_mae = (baseline_scores - approx_scores).abs().mean().item()
    score_max = (baseline_scores - approx_scores).abs().max().item()
    output_mae = (baseline_output - approx_output).abs().mean().item()
    output_max = (baseline_output - approx_output).abs().max().item()

    # Relative error (avoid div-by-zero)
    score_rel = score_mae / (baseline_scores.abs().mean().item() + 1e-8)
    output_rel = output_mae / (baseline_output.abs().mean().item() + 1e-8)

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Scores  — MAE: {score_mae:.6f}  Max: {score_max:.6f}  Rel: {score_rel:.4f}")
    print(f"  Output  — MAE: {output_mae:.6f}  Max: {output_max:.6f}  Rel: {output_rel:.4f}")


# =============================================================================
# Phase 2: Sweep over parameters
# =============================================================================

def sweep_parameters(Q, K, V, baseline_output, baseline_scores):
    """Test different bit-widths and sketch dimensions."""
    print("\n" + "#" * 60)
    print("  PHASE 2: Parameter sweep")
    print("#" * 60)

    for bits in [2, 3, 4]:
        for m in [8, 16, 32, 64, 128]:
            out, sc = qjl_attention(Q, K, V, bits=bits, m=m, correct_v=False)
            compare(baseline_scores, sc, baseline_output, out,
                    f"bits={bits}, m={m:3d}, V-correction=OFF")

    # Best config with V correction
    print("\n" + "-" * 60)
    print("  With V correction enabled (bits=3)")
    print("-" * 60)
    for m in [32, 64, 128]:
        out, sc = qjl_attention(Q, K, V, bits=3, m=m, correct_v=True)
        compare(baseline_scores, sc, baseline_output, out,
                f"bits=3, m={m:3d}, V-correction=ON")


# =============================================================================
# Phase 3: Clean class-based interface
# =============================================================================

class QuantizedKVCache:
    """KV cache that stores coarse-quantized K/V plus QJL sketches of residuals.

    Memory stored per token:
      - K_hat, V_hat: quantized (could be packed to `bits` per value)
      - sketch_K, sketch_V: 1 bit per entry, m entries per token

    Usage:
        cache = QuantizedKVCache(dim=16, bits=3, m=32)
        cache.encode(K, V)
        scores = cache.decode_scores(Q)
        output = cache.decode_output(Q, correct_v=True)
    """

    def __init__(self, dim, bits=3, m=32, seed=123):
        self.dim = dim
        self.bits = bits
        self.m = m
        self.qjl = QJLSketch(m=m, d=dim, seed=seed)

        # These are populated by encode()
        self.K_hat = None
        self.V_hat = None
        self.sketch_K = None
        self.sketch_V = None

    def encode(self, K, V):
        """Quantize K/V and store QJL sketches of residuals.

        Args:
            K: (batch, tokens, dim)
            V: (batch, tokens, dim)
        """
        self.K_hat = quantize(K, bits=self.bits)
        self.V_hat = quantize(V, bits=self.bits)

        r_K = K - self.K_hat
        r_V = V - self.V_hat

        self.sketch_K = self.qjl.sketch(r_K)  # (batch, m, tokens)
        self.sketch_V = self.qjl.sketch(r_V)  # (batch, m, tokens)

    def decode_scores(self, Q):
        """Compute approximate attention scores: Q @ K.T ≈ Q @ K_hat.T + correction.

        Args:
            Q: (batch, tokens_q, dim)

        Returns:
            scores_approx: (batch, tokens_q, tokens_k)
        """
        coarse = Q @ self.K_hat.transpose(-1, -2)
        correction = self.qjl.correct_scores(Q, self.sketch_K)
        return coarse + correction

    def decode_output(self, Q, correct_v=False):
        """Full approximate attention: softmax(approx_scores) @ V_hat [+ V correction].

        Args:
            Q: (batch, tokens_q, dim)
            correct_v: whether to apply QJL correction for V

        Returns:
            output: (batch, tokens_q, dim)
        """
        scores = self.decode_scores(Q)
        weights = F.softmax(scores, dim=-1)
        output = weights @ self.V_hat

        if correct_v:
            output = output + self.qjl.correct_values(weights, self.sketch_V)

        return output, scores

    def memory_summary(self, tokens):
        """Estimate memory usage vs full-precision baseline.

        Returns dict with byte counts (assuming float32 baseline).
        """
        baseline_bytes = 2 * tokens * self.dim * 4  # K + V in float32

        # Quantized K_hat, V_hat: still stored as float32 here (could be packed)
        # In a real implementation these would be `bits` per value
        quant_bytes = 2 * tokens * self.dim * (self.bits / 8)

        # Sketches: 1 bit per entry, m entries per token, for K and V
        sketch_bytes = 2 * tokens * self.m * (1 / 8)

        # Projection matrix A: shared across all tokens (one-time cost)
        proj_bytes = self.m * self.dim * 4  # float32

        total_compressed = quant_bytes + sketch_bytes + proj_bytes

        return {
            "baseline_bytes": baseline_bytes,
            "compressed_bytes": total_compressed,
            "ratio": baseline_bytes / total_compressed if total_compressed > 0 else float('inf'),
            "quant_bytes": quant_bytes,
            "sketch_bytes": sketch_bytes,
            "proj_bytes": proj_bytes,
        }


# =============================================================================
# Main: run all phases
# =============================================================================

def main():
    print("TurboQuant-style KV Cache Compression Prototype")
    print("=" * 60)

    # --- Phase 1 ---
    print("\n### Phase 1: Minimal working prototype ###\n")

    batch, tokens, dim = 1, 8, 16
    Q, K, V = make_toy_data(batch=batch, tokens=tokens, dim=dim)
    print(f"Toy data: batch={batch}, tokens={tokens}, dim={dim}")
    print(f"  Q shape: {Q.shape}")
    print(f"  K shape: {K.shape}")
    print(f"  V shape: {V.shape}")

    # Baseline
    baseline_output, baseline_scores = baseline_attention(Q, K, V)
    print(f"\nBaseline attention:")
    print(f"  scores shape: {baseline_scores.shape}")
    print(f"  output shape: {baseline_output.shape}")

    # QJL-corrected (default: bits=3, m=32)
    approx_output, approx_scores = qjl_attention(Q, K, V, bits=3, m=32)

    compare(baseline_scores, approx_scores, baseline_output, approx_output,
            "Phase 1 result: bits=3, m=32, no V-correction")

    # --- Phase 2 ---
    sweep_parameters(Q, K, V, baseline_output, baseline_scores)

    # --- Phase 3 ---
    print("\n" + "#" * 60)
    print("  PHASE 3: Clean class-based interface")
    print("#" * 60)

    cache = QuantizedKVCache(dim=dim, bits=3, m=64)
    cache.encode(K, V)
    class_output, class_scores = cache.decode_output(Q, correct_v=True)

    compare(baseline_scores, class_scores, baseline_output, class_output,
            "QuantizedKVCache: bits=3, m=64, V-correction=ON")

    # Memory estimate
    mem = cache.memory_summary(tokens)
    print(f"\n  Memory estimate ({tokens} tokens, dim={dim}):")
    print(f"    Baseline (float32 K+V): {mem['baseline_bytes']:.0f} bytes")
    print(f"    Compressed total:       {mem['compressed_bytes']:.0f} bytes")
    print(f"      Quantized K_hat+V_hat:  {mem['quant_bytes']:.0f} bytes")
    print(f"      Sketches (K+V):         {mem['sketch_bytes']:.0f} bytes")
    print(f"      Projection matrix A:    {mem['proj_bytes']:.0f} bytes")
    print(f"    Compression ratio:      {mem['ratio']:.2f}x")

    # --- Larger test ---
    print("\n" + "#" * 60)
    print("  Larger test: tokens=64, dim=64")
    print("#" * 60)

    Q2, K2, V2 = make_toy_data(batch=1, tokens=64, dim=64, seed=99)
    base_out2, base_sc2 = baseline_attention(Q2, K2, V2)

    cache2 = QuantizedKVCache(dim=64, bits=3, m=128)
    cache2.encode(K2, V2)
    approx_out2, approx_sc2 = cache2.decode_output(Q2, correct_v=True)

    compare(base_sc2, approx_sc2, base_out2, approx_out2,
            "QuantizedKVCache: tokens=64, dim=64, bits=3, m=128, V-corr=ON")

    mem2 = cache2.memory_summary(64)
    print(f"\n  Memory: {mem2['baseline_bytes']:.0f} → {mem2['compressed_bytes']:.0f} bytes "
          f"({mem2['ratio']:.2f}x compression)")

    print("\n" + "=" * 60)
    print("  Done. All phases complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
