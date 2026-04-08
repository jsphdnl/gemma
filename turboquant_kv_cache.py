"""
TurboQuant KV Cache Compression — Real Algorithm

Implements the TurboQuant algorithm from:
  "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
  Zandieh, Daliri, Hadian, Mirrokni (arXiv:2504.19874, 2025)

Algorithm (TurboQuant_mse):
  1. Random orthogonal rotation R (QR of Gaussian matrix)
     → each coordinate of R @ (v/||v||) follows a known sphere-marginal distribution
  2. Lloyd-Max optimal scalar quantization per coordinate
     → precomputed codebook for the sphere marginal, no calibration needed
  3. Store packed quantization indices + scalar norm per vector

This is fully data-oblivious: the codebook depends only on (dimension, nbits).
The random rotation is the key trick — it makes the per-coordinate distribution
predictable regardless of the input vector's direction.

Theoretical guarantee: MSE distortion within ~2.7x of the Shannon lower bound.
"""

import torch
import torch.nn.functional as F
import math
from scipy import integrate


# =============================================================================
# 1. Sphere marginal distribution
# =============================================================================

def sphere_marginal_pdf(x, d):
    """PDF of a single coordinate of a unit vector uniform on S^{d-1}.

    If u ~ Uniform(S^{d-1}), then each u_i has density:
      p(x) = C_d * (1 - x^2)^{(d-3)/2}   for x in [-1, 1]

    where C_d = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)).

    For random orthogonal R and fixed unit vector v:
      (R @ v)_i has this same distribution (since R @ v ~ Uniform(S^{d-1})).
    """
    if abs(x) >= 1.0:
        return 0.0
    # Use log-gamma for numerical stability
    log_C = math.lgamma(d / 2) - 0.5 * math.log(math.pi) - math.lgamma((d - 1) / 2)
    return math.exp(log_C) * (1.0 - x * x) ** ((d - 3) / 2)


# =============================================================================
# 2. Lloyd-Max optimal scalar quantizer
# =============================================================================

def lloyd_max_codebook(d, nbits, n_iter=200):
    """Compute Lloyd-Max optimal quantization codebook for the sphere marginal.

    The Lloyd-Max algorithm finds quantization levels that minimize MSE
    for a known probability distribution. It alternates:
      1. Given levels → boundaries = midpoints between adjacent levels
      2. Given boundaries → levels = conditional expectation E[X | bin_i]

    Args:
        d: vector dimension (determines the sphere marginal shape)
        nbits: quantization bit-width (2, 3, or 4)
        n_iter: Lloyd-Max iterations

    Returns:
        boundaries: (n_levels + 1,) tensor of decision boundaries
        levels: (n_levels,) tensor of reconstruction levels (centroids)
    """
    n_levels = 2 ** nbits
    pdf = lambda x: sphere_marginal_pdf(x, d)

    # Initialize levels uniformly across [-1, 1]
    levels = [(2 * i + 1) / n_levels - 1.0 for i in range(n_levels)]

    for _ in range(n_iter):
        # Boundaries: midpoints between adjacent levels, clamped to [-1, 1]
        boundaries = [-1.0]
        for i in range(n_levels - 1):
            boundaries.append((levels[i] + levels[i + 1]) / 2)
        boundaries.append(1.0)

        # Levels: conditional expectation within each bin
        new_levels = []
        for i in range(n_levels):
            a, b = boundaries[i], boundaries[i + 1]
            num, _ = integrate.quad(lambda x: x * pdf(x), a, b)
            den, _ = integrate.quad(pdf, a, b)
            if den > 1e-15:
                new_levels.append(num / den)
            else:
                new_levels.append((a + b) / 2)

        levels = new_levels

    return (
        torch.tensor(boundaries, dtype=torch.float32),
        torch.tensor(levels, dtype=torch.float32),
    )


def codebook_mse(d, boundaries, levels):
    """Compute expected per-coordinate MSE for a given codebook.

    MSE_coord = sum_i integral_{b_i}^{b_{i+1}} (x - level_i)^2 * pdf(x) dx
    Full-vector MSE = d * MSE_coord  (for unit vectors)
    """
    pdf = lambda x: sphere_marginal_pdf(x, d)
    n_levels = len(levels)
    total_mse = 0.0
    for i in range(n_levels):
        a = boundaries[i].item()
        b = boundaries[i + 1].item()
        lev = levels[i].item()
        val, _ = integrate.quad(lambda x: (x - lev) ** 2 * pdf(x), a, b)
        total_mse += val
    return total_mse


# =============================================================================
# 3. Random orthogonal matrix (Haar-distributed)
# =============================================================================

def random_orthogonal(d, seed=42):
    """Generate a uniformly random orthogonal matrix via QR decomposition.

    Uses the standard method: QR-decompose a random Gaussian matrix,
    then fix the sign convention to ensure Haar measure.

    Returns:
        R: (d, d) orthogonal matrix (R @ R.T = I)
    """
    gen = torch.Generator().manual_seed(seed)
    G = torch.randn(d, d, generator=gen)
    Q, R_factor = torch.linalg.qr(G)
    # Fix sign to get uniform Haar distribution
    Q = Q @ torch.diag(torch.sign(torch.diag(R_factor)))
    return Q  # (d, d), orthogonal


# =============================================================================
# 4. Bit packing (uint4 / uint2)
# =============================================================================

def pack_uint4(indices):
    """Pack 4-bit indices into uint8 (2 indices per byte).

    Args:
        indices: (..., dim) tensor of ints in [0, 15]. dim must be even.

    Returns:
        packed: (..., dim // 2) uint8 tensor
    """
    assert indices.shape[-1] % 2 == 0, "dim must be even for uint4 packing"
    low = indices[..., 0::2].to(torch.uint8)
    high = indices[..., 1::2].to(torch.uint8)
    return (high << 4) | low


def unpack_uint4(packed, dim):
    """Unpack uint8 back to 4-bit indices.

    Returns:
        indices: (..., dim) int tensor
    """
    low = (packed & 0x0F).to(torch.int32)
    high = ((packed >> 4) & 0x0F).to(torch.int32)
    result = torch.empty(*packed.shape[:-1], dim, dtype=torch.int32)
    result[..., 0::2] = low
    result[..., 1::2] = high
    return result


def pack_uint2(indices):
    """Pack 2-bit indices into uint8 (4 indices per byte).

    Args:
        indices: (..., dim) tensor of ints in [0, 3]. dim must be divisible by 4.
    """
    assert indices.shape[-1] % 4 == 0, "dim must be divisible by 4 for uint2 packing"
    i = indices.to(torch.uint8)
    return i[..., 0::4] | (i[..., 1::4] << 2) | (i[..., 2::4] << 4) | (i[..., 3::4] << 6)


def unpack_uint2(packed, dim):
    """Unpack uint8 back to 2-bit indices."""
    result = torch.empty(*packed.shape[:-1], dim, dtype=torch.int32)
    result[..., 0::4] = (packed & 0x03).to(torch.int32)
    result[..., 1::4] = ((packed >> 2) & 0x03).to(torch.int32)
    result[..., 2::4] = ((packed >> 4) & 0x03).to(torch.int32)
    result[..., 3::4] = ((packed >> 6) & 0x03).to(torch.int32)
    return result


# =============================================================================
# 5. TurboQuantizer — core encode/decode
# =============================================================================

class TurboQuantizer:
    """Implements the TurboQuant vector quantizer.

    For a vector v ∈ R^d:
      Encode: norm = ||v||, u = v/norm, r = R @ u, indices = quantize(r)
      Decode: r_hat = codebook[indices], u_hat = R^T @ r_hat, v_hat = norm * u_hat

    The codebook is precomputed from the sphere marginal distribution
    (depends only on d and nbits, not on data).
    """

    def __init__(self, dim, nbits=4, seed=42):
        self.dim = dim
        self.nbits = nbits
        self.n_levels = 2 ** nbits

        # Precompute codebook (one-time cost)
        self.boundaries, self.levels = lloyd_max_codebook(dim, nbits)

        # Per-coordinate MSE and full-vector MSE (theoretical, for unit vectors)
        self.coord_mse = codebook_mse(dim, self.boundaries, self.levels)
        self.vector_mse = dim * self.coord_mse

        # Random orthogonal rotation matrix: (dim, dim)
        self.R = random_orthogonal(dim, seed=seed)

        # Internal boundaries for torch.bucketize (exclude -1 and +1 endpoints)
        self._bucket_bounds = self.boundaries[1:-1]

        # Track which device tensors have been moved to
        self._device = None

    def _to_device(self, device):
        """Move quantizer tensors to the given device (lazy, once)."""
        if self._device != device:
            self.R = self.R.to(device)
            self.levels = self.levels.to(device)
            self._bucket_bounds = self._bucket_bounds.to(device)
            self._device = device

    def encode(self, x):
        """Encode vectors: rotate → quantize → pack.

        Args:
            x: (*, dim) tensor of input vectors

        Returns:
            packed: packed quantization indices
            norms: (*,) tensor of vector norms (float32)
        """
        self._to_device(x.device)

        # Compute norms and normalize
        norms = x.norm(dim=-1, keepdim=True)  # (*, 1)
        # Handle zero vectors
        safe_norms = norms.clamp(min=1e-10)
        u = x / safe_norms  # (*, dim), unit vectors

        # Random rotation: u_rot = u @ R.T  (batch matmul with shared R)
        u_rot = u @ self.R.T  # (*, dim)

        # Quantize each coordinate: find which bin it falls into
        indices = torch.bucketize(u_rot, self._bucket_bounds)  # (*, dim), values in [0, n_levels-1]
        indices = indices.clamp(0, self.n_levels - 1)

        # Pack indices
        if self.nbits == 4:
            packed = pack_uint4(indices)
        elif self.nbits == 2:
            packed = pack_uint2(indices)
        else:
            # For 3-bit or other widths, store raw indices (no packing)
            packed = indices.to(torch.uint8)

        return packed, norms.squeeze(-1)

    def decode(self, packed, norms):
        """Decode: unpack → codebook lookup → inverse rotate → scale.

        Args:
            packed: packed quantization indices (from encode)
            norms: (*,) tensor of vector norms

        Returns:
            x_hat: (*, dim) reconstructed vectors
        """
        # Unpack indices
        if self.nbits == 4:
            indices = unpack_uint4(packed, self.dim)
        elif self.nbits == 2:
            indices = unpack_uint2(packed, self.dim)
        else:
            indices = packed.to(torch.int64)

        # Codebook lookup: indices → reconstruction levels
        u_rot_hat = self.levels[indices]  # (*, dim)

        # Inverse rotation: u_hat = u_rot_hat @ R
        u_hat = u_rot_hat @ self.R  # (*, dim)

        # Scale by original norm
        x_hat = u_hat * norms.unsqueeze(-1)  # (*, dim)

        return x_hat

    def roundtrip(self, x):
        """Encode then decode — convenience for testing.

        Returns:
            x_hat: (*, dim) reconstructed vectors
        """
        packed, norms = self.encode(x)
        return self.decode(packed, norms)

    def bytes_per_vector(self):
        """Storage cost per vector in bytes."""
        index_bits = self.dim * self.nbits
        norm_bytes = 2  # float16
        return index_bits / 8 + norm_bytes

    def baseline_bytes_per_vector(self, dtype_bytes=4):
        """Baseline storage (uncompressed) per vector in bytes."""
        return self.dim * dtype_bytes

    def compression_ratio(self, dtype_bytes=4):
        """Compression ratio vs uncompressed storage."""
        return self.baseline_bytes_per_vector(dtype_bytes) / self.bytes_per_vector()


# =============================================================================
# 6. TurboQuant KV Cache
# =============================================================================

class TurboQuantCache:
    """KV cache that stores TurboQuant-compressed K and V.

    Usage:
        cache = TurboQuantCache(dim=128, nbits=4)
        cache.encode(K, V)
        output, scores = cache.attention(Q)
    """

    def __init__(self, dim, nbits=4, seed=42):
        self.quantizer = TurboQuantizer(dim, nbits=nbits, seed=seed)

        self.K_packed = None
        self.K_norms = None
        self.V_packed = None
        self.V_norms = None

    def encode(self, K, V):
        """Compress K and V into the cache.

        Args:
            K: (batch, tokens, dim)
            V: (batch, tokens, dim)
        """
        self.K_packed, self.K_norms = self.quantizer.encode(K)
        self.V_packed, self.V_norms = self.quantizer.encode(V)

    def decode_K(self):
        """Decompress K."""
        return self.quantizer.decode(self.K_packed, self.K_norms)

    def decode_V(self):
        """Decompress V."""
        return self.quantizer.decode(self.V_packed, self.V_norms)

    def attention(self, Q):
        """Compute attention using decompressed KV cache.

        Args:
            Q: (batch, tokens_q, dim)

        Returns:
            output: (batch, tokens_q, dim)
            scores: (batch, tokens_q, tokens_k) — raw logits
        """
        K_hat = self.decode_K()
        V_hat = self.decode_V()
        scores = Q @ K_hat.transpose(-1, -2)
        weights = F.softmax(scores, dim=-1)
        output = weights @ V_hat
        return output, scores

    def append(self, new_K, new_V):
        """Append new token(s) to the cache (autoregressive decoding).

        Args:
            new_K: (batch, new_tokens, dim)
            new_V: (batch, new_tokens, dim)
        """
        new_K_packed, new_K_norms = self.quantizer.encode(new_K)
        new_V_packed, new_V_norms = self.quantizer.encode(new_V)

        if self.K_packed is None:
            self.K_packed = new_K_packed
            self.K_norms = new_K_norms
            self.V_packed = new_V_packed
            self.V_norms = new_V_norms
        else:
            # Concatenate along the token dimension (dim=-2 for packed, -1 for norms)
            self.K_packed = torch.cat([self.K_packed, new_K_packed], dim=-2)
            self.K_norms = torch.cat([self.K_norms, new_K_norms], dim=-1)
            self.V_packed = torch.cat([self.V_packed, new_V_packed], dim=-2)
            self.V_norms = torch.cat([self.V_norms, new_V_norms], dim=-1)

    def memory_summary(self, tokens):
        """Memory estimate for the cache."""
        q = self.quantizer
        compressed = 2 * tokens * q.bytes_per_vector()  # K + V
        baseline = 2 * tokens * q.baseline_bytes_per_vector()
        rotation_overhead = q.dim * q.dim * 4  # R matrix, float32, shared

        return {
            "baseline_bytes": baseline,
            "compressed_bytes": compressed + rotation_overhead,
            "kv_bytes_only": compressed,
            "rotation_bytes": rotation_overhead,
            "ratio": baseline / (compressed + rotation_overhead),
            "ratio_amortized": baseline / compressed,  # ignoring fixed R overhead
            "bytes_per_vector": q.bytes_per_vector(),
        }


# =============================================================================
# 7. Baseline attention
# =============================================================================

def baseline_attention(Q, K, V):
    """Standard dot-product attention (no scaling for simplicity)."""
    scores = Q @ K.transpose(-1, -2)
    weights = F.softmax(scores, dim=-1)
    output = weights @ V
    return output, scores


# =============================================================================
# 8. Comparison utilities
# =============================================================================

def compare(base_scores, approx_scores, base_output, approx_output, label=""):
    """Print error metrics."""
    s_mae = (base_scores - approx_scores).abs().mean().item()
    s_max = (base_scores - approx_scores).abs().max().item()
    o_mae = (base_output - approx_output).abs().mean().item()
    o_max = (base_output - approx_output).abs().max().item()
    s_rel = s_mae / (base_scores.abs().mean().item() + 1e-8)
    o_rel = o_mae / (base_output.abs().mean().item() + 1e-8)

    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"{'=' * 65}")
    print(f"  Scores — MAE: {s_mae:.6f}  Max: {s_max:.6f}  Rel: {s_rel:.4f}")
    print(f"  Output — MAE: {o_mae:.6f}  Max: {o_max:.6f}  Rel: {o_rel:.4f}")


# =============================================================================
# 9. Main
# =============================================================================

def main():
    torch.manual_seed(42)
    print("TurboQuant KV Cache Compression — Real Algorithm")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Phase 1: Verify quantizer on toy data
    # ------------------------------------------------------------------
    print("\n### Phase 1: Quantizer correctness ###\n")

    for dim in [16, 64, 128]:
        for nbits in [2, 3, 4]:
            q = TurboQuantizer(dim, nbits=nbits)

            # Generate random vectors and measure roundtrip MSE
            x = torch.randn(100, dim)
            x_hat = q.roundtrip(x)
            norms = x.norm(dim=-1, keepdim=True)
            u = x / norms.clamp(min=1e-10)
            u_hat = x_hat / norms.clamp(min=1e-10)
            measured_mse = (u - u_hat).pow(2).sum(dim=-1).mean().item()
            theoretical_mse = q.vector_mse
            ratio = q.compression_ratio()

            print(f"  dim={dim:3d}  bits={nbits}  "
                  f"MSE(theory)={theoretical_mse:.5f}  "
                  f"MSE(measured)={measured_mse:.5f}  "
                  f"compress={ratio:.1f}x  "
                  f"({q.bytes_per_vector():.0f} vs {q.baseline_bytes_per_vector():.0f} bytes/vec)")

    # Shannon lower bound comparison
    print("\n  Shannon lower bound comparison (dim=128):")
    for nbits in [2, 3, 4]:
        shannon = 2 ** (-2 * nbits)  # approximate for unit vectors
        q = TurboQuantizer(128, nbits=nbits)
        print(f"    bits={nbits}: Shannon≥{shannon:.5f}  "
              f"TurboQuant={q.vector_mse:.5f}  "
              f"ratio={q.vector_mse / shannon:.2f}x Shannon")

    # ------------------------------------------------------------------
    # Phase 2: Attention quality
    # ------------------------------------------------------------------
    print("\n\n### Phase 2: Attention quality ###\n")

    batch, tokens, dim = 1, 8, 16
    Q = torch.randn(batch, tokens, dim)
    K = torch.randn(batch, tokens, dim)
    V = torch.randn(batch, tokens, dim)
    base_out, base_sc = baseline_attention(Q, K, V)

    print(f"Toy data: batch={batch}, tokens={tokens}, dim={dim}")

    for nbits in [2, 3, 4]:
        cache = TurboQuantCache(dim=dim, nbits=nbits)
        cache.encode(K, V)
        out, sc = cache.attention(Q)
        compare(base_sc, sc, base_out, out, f"bits={nbits}, dim={dim}")

    # Larger test
    print("\n--- Larger test ---")
    for dim in [64, 128]:
        Q2 = torch.randn(1, 32, dim)
        K2 = torch.randn(1, 32, dim)
        V2 = torch.randn(1, 32, dim)
        base_out2, base_sc2 = baseline_attention(Q2, K2, V2)

        for nbits in [3, 4]:
            cache = TurboQuantCache(dim=dim, nbits=nbits)
            cache.encode(K2, V2)
            out2, sc2 = cache.attention(Q2)
            compare(base_sc2, sc2, base_out2, out2,
                    f"bits={nbits}, dim={dim}, tokens=32")

    # ------------------------------------------------------------------
    # Phase 3: Bit packing roundtrip verification
    # ------------------------------------------------------------------
    print("\n\n### Phase 3: Bit packing verification ###\n")

    for nbits in [2, 4]:
        q = TurboQuantizer(128, nbits=nbits)
        x = torch.randn(4, 128)
        packed, norms = q.encode(x)
        x_hat = q.decode(packed, norms)
        # Verify double-quantize gives same result (idempotency)
        packed2, norms2 = q.encode(x_hat)
        x_hat2 = q.decode(packed2, norms2)
        double_quant_err = (x_hat - x_hat2).abs().max().item()
        pack_label = f"uint{nbits}" if nbits in [2, 4] else "raw"
        print(f"  {nbits}-bit ({pack_label}): packed shape {packed.shape}, "
              f"double-quantize max err = {double_quant_err:.2e}, "
              f"storage = {packed.numel() * packed.element_size()} bytes for {x.shape[0]} vectors")

    # ------------------------------------------------------------------
    # Phase 4: Autoregressive cache append
    # ------------------------------------------------------------------
    print("\n\n### Phase 4: Autoregressive cache simulation ###\n")

    dim = 64
    cache = TurboQuantCache(dim=dim, nbits=4)
    all_K = torch.randn(1, 16, dim)
    all_V = torch.randn(1, 16, dim)

    # Append tokens one at a time
    for t in range(16):
        cache.append(all_K[:, t:t+1, :], all_V[:, t:t+1, :])

    # Compare with bulk encode
    cache_bulk = TurboQuantCache(dim=dim, nbits=4)
    cache_bulk.encode(all_K, all_V)

    K_inc = cache.decode_K()
    K_bulk = cache_bulk.decode_K()
    diff = (K_inc - K_bulk).abs().max().item()
    print(f"  Incremental vs bulk encode max diff: {diff:.2e} (should be 0.0)")

    Q_test = torch.randn(1, 1, dim)
    out_inc, _ = cache.attention(Q_test)
    out_bulk, _ = cache_bulk.attention(Q_test)
    diff_out = (out_inc - out_bulk).abs().max().item()
    print(f"  Attention output diff: {diff_out:.2e}")

    # ------------------------------------------------------------------
    # Phase 5: Memory summary
    # ------------------------------------------------------------------
    print("\n\n### Phase 5: Memory estimates ###\n")

    configs = [
        (128, 4, 1024, "Typical LLM (d=128, 4-bit, 1K tokens)"),
        (128, 4, 8192, "Typical LLM (d=128, 4-bit, 8K tokens)"),
        (128, 4, 131072, "Typical LLM (d=128, 4-bit, 128K tokens)"),
        (256, 4, 8192, "Gemma-style (d=256, 4-bit, 8K tokens)"),
        (128, 2, 8192, "Aggressive (d=128, 2-bit, 8K tokens)"),
    ]

    for dim, nbits, tokens, desc in configs:
        cache = TurboQuantCache(dim=dim, nbits=nbits)
        mem = cache.memory_summary(tokens)
        print(f"  {desc}")
        print(f"    Baseline: {mem['baseline_bytes'] / 1024:.0f} KB  "
              f"Compressed: {mem['kv_bytes_only'] / 1024:.0f} KB  "
              f"R overhead: {mem['rotation_bytes'] / 1024:.0f} KB  "
              f"Ratio: {mem['ratio_amortized']:.2f}x")

    print("\n" + "=" * 65)
    print("  Done.")
    print("=" * 65)


if __name__ == "__main__":
    main()
