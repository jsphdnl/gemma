"""
Test TurboQuant KV cache compression on a real model.

Loads a model, runs a forward pass, captures KV cache tensors,
applies TurboQuant compression/decompression, and measures:
  1. Per-layer KV reconstruction MSE
  2. Attention score distortion
  3. Output logit divergence
  4. Memory savings
"""

import torch
import torch.nn.functional as F
import time
import sys
from turboquant_kv_cache import TurboQuantizer


# =============================================================================
# Config
# =============================================================================

MODEL_NAME = "google/gemma-4-E2B-it"
PROMPT = "The quick brown fox jumps over the lazy dog. In the field of machine learning,"
NBITS = 4
DEVICE = "cpu"  # MPS has issues with some ops; CPU is safest for correctness testing


def load_model(model_name):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=DEVICE,
    )
    model.eval()

    cfg = model.config
    # Gemma 4 uses text_config for nested attributes
    text_cfg = getattr(cfg, 'text_config', cfg)

    print(f"  Model type: {cfg.model_type}")
    hidden = getattr(text_cfg, 'hidden_size', None)
    print(f"  Hidden size: {hidden}")
    print(f"  Num layers: {text_cfg.num_hidden_layers}")
    print(f"  Num attention heads: {text_cfg.num_attention_heads}")
    print(f"  Num KV heads: {getattr(text_cfg, 'num_key_value_heads', 'N/A')}")
    head_dim = getattr(text_cfg, 'head_dim', hidden // text_cfg.num_attention_heads if hidden else 64)
    print(f"  Head dim: {head_dim}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    return model, tokenizer, head_dim


def extract_kv_cache(model, tokenizer, prompt):
    """Run forward pass and extract KV cache tensors."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    seq_len = inputs["input_ids"].shape[1]
    print(f"\nPrompt: \"{prompt}\"")
    print(f"Token count: {seq_len}")

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    logits = outputs.logits  # (batch, seq, vocab)
    past_kv = outputs.past_key_values

    # Extract K, V tensors from the cache object
    kv_pairs = []
    if hasattr(past_kv, 'key_cache'):
        # DynamicCache — use key_cache / value_cache lists directly
        for layer_idx in range(len(past_kv.key_cache)):
            K = past_kv.key_cache[layer_idx]   # (batch, kv_heads, seq, head_dim)
            V = past_kv.value_cache[layer_idx]  # (batch, kv_heads, seq, head_dim)
            if K is not None and K.numel() > 0:
                kv_pairs.append((K, V))
    else:
        # Legacy tuple format
        for item in past_kv:
            kv_pairs.append((item[0], item[1]))

    return kv_pairs, logits, inputs


def analyze_kv_stats(kv_pairs):
    """Print statistics about the raw KV cache."""
    print(f"\n{'=' * 70}")
    print(f"  KV Cache Statistics ({len(kv_pairs)} layers)")
    print(f"{'=' * 70}")

    total_bytes = 0
    for i, (K, V) in enumerate(kv_pairs):
        k_norm = K.norm(dim=-1).mean().item()
        v_norm = V.norm(dim=-1).mean().item()
        k_std = K.std().item()
        v_std = V.std().item()
        layer_bytes = (K.numel() + V.numel()) * K.element_size()
        total_bytes += layer_bytes

        if i < 5 or i == len(kv_pairs) - 1:
            print(f"  Layer {i:2d}: K{list(K.shape)} "
                  f"norm={k_norm:.2f} std={k_std:.3f} | "
                  f"V norm={v_norm:.2f} std={v_std:.3f} | "
                  f"{layer_bytes / 1024:.1f} KB")
        elif i == 5:
            print(f"  ...")

    print(f"\n  Total KV cache: {total_bytes / 1024:.1f} KB ({total_bytes / 1024 / 1024:.2f} MB)")


def test_turboquant_per_layer(kv_pairs, head_dim, nbits=4):
    """Apply TurboQuant to each layer's KV cache and measure reconstruction error."""
    print(f"\n{'=' * 70}")
    print(f"  TurboQuant Compression (nbits={nbits})")
    print(f"{'=' * 70}")

    # Detect unique head dims across layers (Gemma 4 has mixed 256/512)
    unique_dims = sorted(set(K.shape[-1] for K, V in kv_pairs))
    print(f"  Head dims found: {unique_dims}")

    # Precompute a quantizer per unique dim
    quantizers = {}
    t0 = time.time()
    for d in unique_dims:
        print(f"  Precomputing Lloyd-Max codebook for dim={d}, bits={nbits}...")
        q = TurboQuantizer(dim=d, nbits=nbits)
        quantizers[d] = q
        print(f"    MSE(theory)={q.vector_mse:.6f}  "
              f"compress={q.compression_ratio():.1f}x  "
              f"({q.bytes_per_vector():.0f} vs {q.baseline_bytes_per_vector():.0f} bytes/vec)")
    print(f"  Codebooks ready in {time.time() - t0:.2f}s\n")

    total_k_mse = 0
    total_v_mse = 0
    total_orig_bytes = 0
    total_comp_bytes = 0
    layer_results = []

    for i, (K, V) in enumerate(kv_pairs):
        # K shape: (batch, kv_heads, seq, head_dim)
        dim = K.shape[-1]
        quantizer = quantizers[dim]

        # Reshape to (batch * kv_heads * seq, head_dim) for quantization
        K_flat = K.reshape(-1, dim)
        V_flat = V.reshape(-1, dim)

        # Compress and decompress
        K_hat = quantizer.roundtrip(K_flat).reshape(K.shape)
        V_hat = quantizer.roundtrip(V_flat).reshape(V.shape)

        # MSE on the actual vectors (not unit-normalized)
        k_mse = (K - K_hat).pow(2).sum(dim=-1).mean().item()
        v_mse = (V - V_hat).pow(2).sum(dim=-1).mean().item()

        # Relative error
        k_rel = k_mse / (K.pow(2).sum(dim=-1).mean().item() + 1e-10)
        v_rel = v_mse / (V.pow(2).sum(dim=-1).mean().item() + 1e-10)

        total_k_mse += k_mse
        total_v_mse += v_mse

        n_vectors = K_flat.shape[0] + V_flat.shape[0]
        orig = n_vectors * quantizer.baseline_bytes_per_vector()
        comp = n_vectors * quantizer.bytes_per_vector()
        total_orig_bytes += orig
        total_comp_bytes += comp

        layer_results.append({
            "K_hat": K_hat, "V_hat": V_hat,
            "k_mse": k_mse, "v_mse": v_mse,
        })

        if i < 5 or i == len(kv_pairs) - 1:
            print(f"  Layer {i:2d} (d={dim:3d}): K_MSE={k_mse:.6f} (rel={k_rel:.5f})  "
                  f"V_MSE={v_mse:.6f} (rel={v_rel:.5f})")
        elif i == 5:
            print(f"  ...")

    n_layers = len(kv_pairs)
    print(f"\n  Average K MSE: {total_k_mse / n_layers:.6f}")
    print(f"  Average V MSE: {total_v_mse / n_layers:.6f}")
    print(f"  Memory: {total_orig_bytes / 1024:.0f} KB → {total_comp_bytes / 1024:.0f} KB")

    return quantizers, layer_results


def test_attention_distortion(kv_pairs, layer_results, model, tokenizer, prompt):
    """Compare attention scores using original vs compressed KV cache."""
    print(f"\n{'=' * 70}")
    print(f"  Attention Score Distortion")
    print(f"{'=' * 70}")

    # We'll compute Q @ K.T for original and compressed K, per layer
    # To get Q, we need to hook into the model's attention layers
    # Simpler approach: directly compute score differences from K and K_hat

    for i in range(min(5, len(kv_pairs))):
        K = kv_pairs[i][0]        # (batch, kv_heads, seq, head_dim)
        K_hat = layer_results[i]["K_hat"]
        V = kv_pairs[i][1]
        V_hat = layer_results[i]["V_hat"]

        # Simulate: use K as both Q and K (self-attention proxy)
        # Real Q would differ due to different projection, but this tests the score path
        # For GQA: Q has more heads than KV. We'll just test with the KV head count.
        scores_orig = K @ K.transpose(-1, -2)  # (batch, heads, seq, seq)
        scores_comp = K @ K_hat.transpose(-1, -2)

        score_mae = (scores_orig - scores_comp).abs().mean().item()
        score_max = (scores_orig - scores_comp).abs().max().item()

        # Attention output difference
        w_orig = F.softmax(scores_orig, dim=-1)
        w_comp = F.softmax(scores_comp, dim=-1)  # using orig K as Q, compressed K for scores
        out_orig = w_orig @ V
        out_comp = w_comp @ V_hat

        out_mae = (out_orig - out_comp).abs().mean().item()

        print(f"  Layer {i:2d}: score MAE={score_mae:.4f}  max={score_max:.4f}  "
              f"output MAE={out_mae:.6f}")


def test_generation(model, tokenizer, kv_pairs, layer_results, inputs):
    """Compare next-token predictions with original vs compressed cache."""
    print(f"\n{'=' * 70}")
    print(f"  Generation Quality (next-token logit comparison)")
    print(f"{'=' * 70}")

    # Get baseline logits (already computed, but let's also check greedy token)
    with torch.no_grad():
        baseline_out = model(**inputs, use_cache=False)
    baseline_logits = baseline_out.logits[:, -1, :]  # last token logits
    baseline_token = baseline_logits.argmax(dim=-1)
    baseline_prob = F.softmax(baseline_logits, dim=-1)

    # Build compressed cache and run forward pass with it
    # We need to construct a DynamicCache with compressed KV
    from transformers import DynamicCache

    compressed_cache = DynamicCache()
    for i, result in enumerate(layer_results):
        compressed_cache.update(
            result["K_hat"],
            result["V_hat"],
            layer_idx=i,
        )

    # Forward pass using compressed cache (just the last position)
    with torch.no_grad():
        # The model expects input_ids for the positions NOT in the cache
        # Since we have cache for all positions, we pass just the last token
        last_token = inputs["input_ids"][:, -1:]
        # Position IDs for the last token
        cache_len = kv_pairs[0][0].shape[2]  # seq_len in cache

        try:
            comp_out = model(
                input_ids=last_token,
                past_key_values=compressed_cache,
                use_cache=False,
            )
            comp_logits = comp_out.logits[:, -1, :]
            comp_token = comp_logits.argmax(dim=-1)
            comp_prob = F.softmax(comp_logits, dim=-1)

            # Compare
            logit_mae = (baseline_logits - comp_logits).abs().mean().item()
            logit_max = (baseline_logits - comp_logits).abs().max().item()
            kl_div = F.kl_div(comp_prob.log(), baseline_prob, reduction='batchmean').item()

            baseline_word = tokenizer.decode(baseline_token[0])
            comp_word = tokenizer.decode(comp_token[0])

            print(f"  Logit MAE: {logit_mae:.6f}")
            print(f"  Logit max diff: {logit_max:.6f}")
            print(f"  KL divergence: {kl_div:.6f}")
            print(f"  Baseline next token: \"{baseline_word}\" (id={baseline_token[0].item()})")
            print(f"  Compressed next token: \"{comp_word}\" (id={comp_token[0].item()})")
            print(f"  Same prediction: {'YES' if baseline_token[0] == comp_token[0] else 'NO'}")
        except Exception as e:
            print(f"  Cache-based generation failed: {e}")
            print(f"  (This may require model-specific cache handling)")


def main():
    print("TurboQuant — Real Model Test")
    print("=" * 70)

    # Load model
    try:
        model, tokenizer, head_dim = load_model(MODEL_NAME)
    except Exception as e:
        print(f"\nFailed to load {MODEL_NAME}: {e}")
        print("Trying fallback model...")
        # Try smaller fallback models in order
        fallbacks = [
            "google/gemma-3-1b-it",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "HuggingFaceTB/SmolLM2-135M",
        ]
        model = None
        for fb in fallbacks:
            try:
                print(f"\nTrying {fb}...")
                model, tokenizer, head_dim = load_model(fb)
                break
            except Exception as e2:
                print(f"  Failed: {e2}")
                continue
        if model is None:
            print("No model could be loaded. Exiting.")
            sys.exit(1)

    # Run forward pass and extract KV cache
    kv_pairs, logits, inputs = extract_kv_cache(model, tokenizer, PROMPT)

    # KV cache statistics
    analyze_kv_stats(kv_pairs)

    # TurboQuant compression per layer
    _, layer_results = test_turboquant_per_layer(kv_pairs, head_dim, nbits=NBITS)

    # Attention distortion
    test_attention_distortion(kv_pairs, layer_results, model, tokenizer, PROMPT)

    # Generation quality
    test_generation(model, tokenizer, kv_pairs, layer_results, inputs)

    # Also test at 2-bit and 3-bit
    print(f"\n{'=' * 70}")
    print(f"  Bit-width comparison (average K MSE across layers)")
    print(f"{'=' * 70}")

    for nbits in [2, 3, 4]:
        # Build quantizer per unique dim
        quantizers = {}
        k_mses = []
        for K, V in kv_pairs:
            d = K.shape[-1]
            if d not in quantizers:
                quantizers[d] = TurboQuantizer(dim=d, nbits=nbits)
            q = quantizers[d]
            K_flat = K.reshape(-1, d)
            K_hat = q.roundtrip(K_flat)
            mse = (K_flat - K_hat).pow(2).sum(dim=-1).mean().item()
            k_mses.append(mse)
        avg = sum(k_mses) / len(k_mses)
        ratios = [q.compression_ratio() for q in quantizers.values()]
        print(f"  {nbits}-bit: avg K MSE = {avg:.6f}, compression = {min(ratios):.1f}-{max(ratios):.1f}x")

    print(f"\n{'=' * 70}")
    print(f"  Done.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
