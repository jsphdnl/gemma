"""
Gemma 4 E4B-it — TurboQuant tuning experiments:
  1. Diverse prompts (not just repetitive patterns)
  2. Progressive layer skipping
  3. 5-bit quantization (lower MSE)
"""

import torch
import time
from turboquant_cache import TurboQuantDynamicCache


def load():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Loading Gemma 4 E4B-it...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E4B-it", dtype=torch.bfloat16, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E4B-it")
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return model, tokenizer


def generate_with_cache(model, tokenizer, prompt, nbits=4, skip_layers=None, max_tokens=50):
    """Run prefill + compressed decode, return (text, token_ids, cache)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    cache = TurboQuantDynamicCache(nbits=nbits, skip_layers=skip_layers or set())

    with torch.no_grad():
        out = model(inputs["input_ids"], past_key_values=cache, use_cache=True)
    cache.compress()

    generated = inputs["input_ids"].clone()
    next_token = out.logits[:, -1:, :].argmax(dim=-1)
    generated = torch.cat([generated, next_token], dim=-1)

    for _ in range(max_tokens - 1):
        with torch.no_grad():
            out = model(next_token, past_key_values=cache, use_cache=True)
        next_token = out.logits[:, -1:, :].argmax(dim=-1)
        generated = torch.cat([generated, next_token], dim=-1)

    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return text, generated[0], cache


def generate_baseline(model, tokenizer, prompt, max_tokens=50):
    """Standard generation without compression."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return tokenizer.decode(ids[0], skip_special_tokens=True), ids[0]


def compare(baseline_ids, comp_ids):
    """Return (n_match, total, pct)."""
    min_len = min(len(baseline_ids), len(comp_ids))
    n_match = (baseline_ids[:min_len] == comp_ids[:min_len]).sum().item()
    return n_match, min_len, n_match / min_len * 100


def main():
    model, tokenizer = load()

    # ================================================================
    # Experiment 1: Diverse prompts
    # ================================================================
    print(f"\n{'=' * 70}")
    print("  Experiment 1: Diverse Prompts (4-bit, no skip)")
    print(f"{'=' * 70}")

    prompts = [
        "Explain the difference between supervised and unsupervised learning in simple terms.",
        "Write a Python function that checks if a string is a palindrome.",
        "The capital of France is Paris. The capital of Germany is Berlin. The capital of Japan is",
        "Once upon a time, in a small village nestled between two mountains, there lived a",
        "List three benefits of regular exercise: 1.",
    ]

    for prompt in prompts:
        base_text, base_ids = generate_baseline(model, tokenizer, prompt, max_tokens=40)
        comp_text, comp_ids, _ = generate_with_cache(model, tokenizer, prompt, nbits=4, max_tokens=40)
        n, total, pct = compare(base_ids, comp_ids)

        print(f"\n  Prompt: \"{prompt[:60]}...\"")
        print(f"  Baseline: {base_text[len(prompt):len(prompt)+120]}")
        print(f"  TQ 4-bit: {comp_text[len(prompt):len(prompt)+120]}")
        print(f"  Match: {n}/{total} ({pct:.0f}%)")

    # ================================================================
    # Experiment 2: Progressive layer skipping
    # ================================================================
    print(f"\n{'=' * 70}")
    print("  Experiment 2: Progressive Layer Skipping (4-bit)")
    print(f"{'=' * 70}")

    test_prompt = "Explain the difference between supervised and unsupervised learning in simple terms."
    base_text, base_ids = generate_baseline(model, tokenizer, test_prompt, max_tokens=40)
    print(f"\n  Baseline: {base_text[len(test_prompt):]}")

    # Get number of KV cache layers
    inputs = tokenizer("Hi", return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
    n_kv_layers = len([l for l in out.past_key_values.layers
                       if hasattr(l, 'keys') and l.keys is not None and l.keys.numel() > 0])
    print(f"  KV layers: {n_kv_layers}")

    for n_skip in [0, 2, 4, 6, 8, 12]:
        skip = set(range(n_skip))  # Skip first N layers
        comp_text, comp_ids, cache = generate_with_cache(
            model, tokenizer, test_prompt, nbits=4, skip_layers=skip, max_tokens=40
        )
        n, total, pct = compare(base_ids, comp_ids)
        mem = cache.memory_summary()
        print(f"  Skip first {n_skip:2d}: match={n:2d}/{total} ({pct:4.0f}%) | "
              f"compress={mem['ratio']:.1f}x | {comp_text[len(test_prompt):len(test_prompt)+80]}")

    # ================================================================
    # Experiment 3: Bit-width comparison (2, 3, 4, 5 bit)
    # ================================================================
    print(f"\n{'=' * 70}")
    print("  Experiment 3: Bit-width Comparison")
    print(f"{'=' * 70}")

    print(f"\n  Baseline: {base_text[len(test_prompt):]}")

    for nbits in [5, 4, 3, 2]:
        comp_text, comp_ids, cache = generate_with_cache(
            model, tokenizer, test_prompt, nbits=nbits, max_tokens=40
        )
        n, total, pct = compare(base_ids, comp_ids)
        mem = cache.memory_summary()
        print(f"  {nbits}-bit: match={n:2d}/{total} ({pct:4.0f}%) | "
              f"compress={mem['ratio']:.1f}x | {comp_text[len(test_prompt):len(test_prompt)+80]}")

    # ================================================================
    # Experiment 4: Best combo — 5-bit + skip first 4 layers
    # ================================================================
    print(f"\n{'=' * 70}")
    print("  Experiment 4: Best Combinations")
    print(f"{'=' * 70}")

    combos = [
        (4, set()),
        (4, {0, 1, 2, 3}),
        (5, set()),
        (5, {0, 1, 2, 3}),
        (5, {0, 1, 2, 3, 4, 5}),
    ]

    for prompt in prompts[:3]:
        base_text, base_ids = generate_baseline(model, tokenizer, prompt, max_tokens=40)
        print(f"\n  Prompt: \"{prompt[:55]}...\"")
        print(f"  Baseline: {base_text[len(prompt):len(prompt)+100]}")

        for nbits, skip in combos:
            comp_text, comp_ids, cache = generate_with_cache(
                model, tokenizer, prompt, nbits=nbits, skip_layers=skip, max_tokens=40
            )
            n, total, pct = compare(base_ids, comp_ids)
            mem = cache.memory_summary()
            skip_str = f"skip {len(skip)}" if skip else "no skip"
            print(f"    {nbits}b {skip_str:>8}: {pct:4.0f}% match, {mem['ratio']:.1f}x | "
                  f"{comp_text[len(prompt):len(prompt)+80]}")

    print(f"\n{'=' * 70}")
    print("  Done.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
