"""
Test TurboQuant cache integration with a real model.

Strategy:
  - Prefill: no compression → bit-identical logits
  - After prefill: batch-compress KV cache
  - Decode: compressed cache used for generation
"""

import torch
import torch.nn.functional as F
from turboquant_cache import TurboQuantDynamicCache

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cpu"
PROMPT = "The quick brown fox jumps over the lazy dog. In the field of machine learning,"
MAX_NEW_TOKENS = 30


def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, device_map=DEVICE
    )
    model.eval()

    cfg = getattr(model.config, 'text_config', model.config)
    print(f"  Type: {model.config.model_type}, "
          f"Layers: {cfg.num_hidden_layers}, "
          f"KV heads: {getattr(cfg, 'num_key_value_heads', '?')}, "
          f"Head dim: {getattr(cfg, 'head_dim', getattr(cfg, 'hidden_size', 0) // cfg.num_attention_heads)}")

    return model, tokenizer


def test_prefill_fidelity(model, tokenizer):
    """Prefill should be bit-identical since we don't compress during it."""
    print(f"\n{'=' * 70}")
    print("  Test 1: Prefill Fidelity (should be 0.0 diff)")
    print(f"{'=' * 70}")

    inputs = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)
    seq_len = inputs["input_ids"].shape[1]
    print(f"  Prompt: \"{PROMPT}\"")
    print(f"  Tokens: {seq_len}")

    with torch.no_grad():
        baseline = model(**inputs, use_cache=False)

    for nbits in [4, 3, 2]:
        cache = TurboQuantDynamicCache(nbits=nbits)
        with torch.no_grad():
            result = model(**inputs, past_key_values=cache, use_cache=True)

        diff = (baseline.logits - result.logits).abs()
        print(f"\n  {nbits}-bit: Logit MAE={diff.mean().item():.8f}  Max={diff.max().item():.8f}")

        # Now compress and check memory
        cache.compress()
        mem = cache.memory_summary()
        print(f"    After compression: {mem['baseline_bytes'] / 1024:.0f} KB → "
              f"{mem['compressed_bytes'] / 1024:.0f} KB ({mem['ratio']:.1f}x)")


def test_decode_with_manual_loop(model, tokenizer):
    """Generate token-by-token with manual loop to test decode quality."""
    print(f"\n{'=' * 70}")
    print("  Test 2: Token-by-token Generation")
    print(f"{'=' * 70}")

    inputs = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]

    # --- Baseline: standard generation ---
    with torch.no_grad():
        baseline_ids = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )
    baseline_text = tokenizer.decode(baseline_ids[0], skip_special_tokens=True)
    print(f"\n  Baseline:\n    {baseline_text}")

    # --- TurboQuant: manual decode loop ---
    for nbits in [4, 3]:
        cache = TurboQuantDynamicCache(nbits=nbits)

        # Step 1: Prefill (no compression)
        with torch.no_grad():
            out = model(input_ids, past_key_values=cache, use_cache=True)

        # Step 2: Compress prefill cache
        cache.compress()

        # Step 3: Decode loop
        generated = input_ids.clone()
        next_token = out.logits[:, -1:, :].argmax(dim=-1)  # (1, 1)
        generated = torch.cat([generated, next_token], dim=-1)

        for _ in range(MAX_NEW_TOKENS - 1):
            with torch.no_grad():
                out = model(next_token, past_key_values=cache, use_cache=True)
            next_token = out.logits[:, -1:, :].argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=-1)

        comp_text = tokenizer.decode(generated[0], skip_special_tokens=True)

        # Compare tokens
        min_len = min(len(baseline_ids[0]), len(generated[0]))
        n_match = (baseline_ids[0][:min_len] == generated[0][:min_len]).sum().item()

        mem = cache.memory_summary()

        print(f"\n  {nbits}-bit TurboQuant:")
        print(f"    {comp_text}")
        print(f"    Token match: {n_match}/{min_len} ({n_match/min_len*100:.1f}%)")
        print(f"    KV memory: {mem['compressed_bytes'] / 1024:.0f} KB ({mem['ratio']:.1f}x)")


def test_outlier_calibration(model, tokenizer):
    """Test automatic outlier layer detection."""
    print(f"\n{'=' * 70}")
    print("  Test 3: Outlier Layer Calibration")
    print(f"{'=' * 70}")

    skip = TurboQuantDynamicCache.calibrate_skip_layers(model, tokenizer)
    print(f"  Outlier layers: {skip if skip else 'none detected'}")

    if skip:
        cache = TurboQuantDynamicCache(nbits=4, skip_layers=skip)
        inputs = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            model(inputs["input_ids"], past_key_values=cache, use_cache=True)

        cache.compress()
        mem = cache.memory_summary()
        print(f"  With skip: {mem['baseline_bytes'] / 1024:.0f} KB → "
              f"{mem['compressed_bytes'] / 1024:.0f} KB ({mem['ratio']:.1f}x)")


def main():
    print("TurboQuant Cache Integration Test")
    print("=" * 70)

    model, tokenizer = load_model(MODEL_NAME)
    test_prefill_fidelity(model, tokenizer)
    test_decode_with_manual_loop(model, tokenizer)
    test_outlier_calibration(model, tokenizer)

    print(f"\n{'=' * 70}")
    print("  All tests complete.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
