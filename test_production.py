"""
TurboQuant Production Test — Gemma 4 E2B-it and Qwen2.5-7B

Tests TurboQuant KV cache compression on production-scale models:
  1. Prefill fidelity (should be 0.0 diff)
  2. Token-by-token generation quality
  3. Memory savings
  4. Outlier layer detection

Usage:
  uv run python test_production.py                    # Gemma 4 E2B (default)
  uv run python test_production.py --model qwen7b     # Qwen2.5-7B (needs ~14GB)
  uv run python test_production.py --model qwen3b     # Qwen2.5-3B (lighter)
"""

import torch
import sys
import time

MODELS = {
    "gemma4": "google/gemma-4-E2B-it",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
}

PROMPT = "The quick brown fox jumps over the lazy dog. In the field of machine learning,"
MAX_NEW_TOKENS = 50


def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading {model_name}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Use bfloat16 to save memory; fall back to float32 if needed
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map="cpu"
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32, device_map="cpu"
        )

    model.eval()
    load_time = time.time() - t0

    cfg = getattr(model.config, 'text_config', model.config)
    hidden = getattr(cfg, 'hidden_size', 0)
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, 'num_key_value_heads', n_heads)
    head_dim = getattr(cfg, 'head_dim', hidden // n_heads if hidden else 0)
    n_layers = cfg.num_hidden_layers
    n_params = sum(p.numel() for p in model.parameters()) / 1e9

    print(f"  Loaded in {load_time:.1f}s")
    print(f"  {n_params:.1f}B params | {n_layers} layers | {n_kv} KV heads | head_dim={head_dim}")
    print(f"  dtype: {next(model.parameters()).dtype}")

    return model, tokenizer


def run_tests(model, tokenizer):
    from turboquant_cache import TurboQuantDynamicCache

    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    seq_len = inputs["input_ids"].shape[1]
    print(f"\n  Prompt: \"{PROMPT}\"")
    print(f"  Tokens: {seq_len}")

    # ── 1. Outlier calibration ──
    print(f"\n{'─' * 60}")
    print("  Outlier Layer Detection")
    print(f"{'─' * 60}")

    skip = TurboQuantDynamicCache.calibrate_skip_layers(model, tokenizer)
    print(f"  Skip layers: {skip if skip else 'none'}")

    # K norm profile
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
    kv = out.past_key_values
    if hasattr(kv, 'layers'):
        norms = []
        for layer in kv.layers:
            if hasattr(layer, 'keys') and layer.keys is not None:
                norms.append(layer.keys.float().norm(dim=-1).mean().item())
        if norms:
            median = sorted(norms)[len(norms) // 2]
            worst = max(norms)
            worst_idx = norms.index(worst)
            print(f"  K norms: median={median:.1f}, max={worst:.1f} (layer {worst_idx}, {worst/median:.1f}x)")

    # ── 2. Prefill fidelity ──
    print(f"\n{'─' * 60}")
    print("  Prefill Fidelity")
    print(f"{'─' * 60}")

    with torch.no_grad():
        baseline_out = model(**inputs, use_cache=False)
    baseline_logits = baseline_out.logits

    for nbits in [4, 3]:
        cache = TurboQuantDynamicCache(nbits=nbits, skip_layers=skip)
        with torch.no_grad():
            comp_out = model(**inputs, past_key_values=cache, use_cache=True)

        diff = (baseline_logits.float() - comp_out.logits.float()).abs()
        cache.compress()
        mem = cache.memory_summary()

        print(f"  {nbits}-bit: logit diff = {diff.max().item():.1e} | "
              f"KV: {mem['baseline_bytes']/1024:.0f} KB → {mem['compressed_bytes']/1024:.0f} KB "
              f"({mem['ratio']:.1f}x)")

    # ── 3. Baseline generation ──
    print(f"\n{'─' * 60}")
    print("  Generation Comparison")
    print(f"{'─' * 60}")

    t0 = time.time()
    with torch.no_grad():
        baseline_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    baseline_time = time.time() - t0
    baseline_text = tokenizer.decode(baseline_ids[0], skip_special_tokens=True)

    print(f"\n  Baseline ({baseline_time:.1f}s):")
    print(f"    {baseline_text}")

    # ── 4. TurboQuant generation ──
    for nbits in [4, 3]:
        cache = TurboQuantDynamicCache(nbits=nbits, skip_layers=skip)

        t0 = time.time()
        with torch.no_grad():
            pout = model(inputs["input_ids"], past_key_values=cache, use_cache=True)
        cache.compress()

        generated = inputs["input_ids"].clone()
        next_token = pout.logits[:, -1:, :].argmax(dim=-1)
        generated = torch.cat([generated, next_token], dim=-1)

        for _ in range(MAX_NEW_TOKENS - 1):
            with torch.no_grad():
                pout = model(next_token, past_key_values=cache, use_cache=True)
            next_token = pout.logits[:, -1:, :].argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=-1)

        gen_time = time.time() - t0
        comp_text = tokenizer.decode(generated[0], skip_special_tokens=True)

        min_len = min(len(baseline_ids[0]), len(generated[0]))
        n_match = (baseline_ids[0][:min_len] == generated[0][:min_len]).sum().item()
        mem = cache.memory_summary()

        print(f"\n  TurboQuant {nbits}-bit ({gen_time:.1f}s, skip {skip}):")
        print(f"    {comp_text}")
        print(f"    Token match: {n_match}/{min_len} ({n_match/min_len*100:.1f}%)")
        print(f"    KV: {mem['baseline_bytes']/1024:.0f} → {mem['compressed_bytes']/1024:.0f} KB "
              f"({mem['ratio']:.1f}x compression)")

    # ── 5. Memory projection at scale ──
    print(f"\n{'─' * 60}")
    print("  Memory Projection at Scale")
    print(f"{'─' * 60}")

    # Extrapolate from current cache to longer sequences
    if hasattr(kv, 'layers') and kv.layers:
        layer0 = kv.layers[0]
        if hasattr(layer0, 'keys') and layer0.keys is not None:
            n_kv_layers = len([l for l in kv.layers if hasattr(l, 'keys') and l.keys is not None])
            _, heads, _, dim = layer0.keys.shape
            elem = layer0.keys.element_size()
            n_skip = len(skip)

            from turboquant_kv_cache import TurboQuantizer
            q = TurboQuantizer(dim=dim, nbits=4)

            for tokens in [1024, 4096, 8192, 32768, 131072]:
                baseline_kb = 2 * tokens * heads * dim * elem * n_kv_layers / 1024
                compressed_layers = n_kv_layers - n_skip
                comp_kb = (2 * tokens * heads * dim * elem * n_skip +
                           2 * tokens * heads * q.bytes_per_vector() * compressed_layers) / 1024
                ratio = baseline_kb / comp_kb if comp_kb > 0 else 0
                print(f"    {tokens:>7} tokens: {baseline_kb/1024:>7.1f} MB → {comp_kb/1024:>7.1f} MB ({ratio:.1f}x)")


def main():
    model_key = "gemma4"
    if len(sys.argv) > 1 and sys.argv[1] == "--model":
        model_key = sys.argv[2]

    model_name = MODELS.get(model_key, model_key)

    print("=" * 60)
    print("  TurboQuant Production Test")
    print("=" * 60)

    try:
        model, tokenizer = load_model(model_name)
    except Exception as e:
        print(f"\n  Failed to load {model_name}: {e}")
        print("  Try: --model qwen3b (smaller)")
        sys.exit(1)

    run_tests(model, tokenizer)

    print(f"\n{'=' * 60}")
    print("  Done.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
