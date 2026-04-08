"""
TurboQuant — Interactive demo with compressed KV cache.

Usage:
  uv run python run.py                                    # E2B, 4-bit
  uv run python run.py --model google/gemma-4-E4B-it      # bigger model
  uv run python run.py --bits 3                           # more compression
  uv run python run.py --compile                          # 2.4x faster (34s warmup)
  uv run python run.py --no-turbo                         # baseline (no compression)
"""

import torch
import argparse
import time
from turboquant_cache import TurboQuantDynamicCache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 5])
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--no-turbo", action="store_true", help="disable TurboQuant (baseline)")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map=device
    )
    model.eval()

    # Build stop token set once
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    for attr in ["eot_token"]:
        tok_str = getattr(tokenizer, attr, None)
        if tok_str:
            tid = tokenizer.convert_tokens_to_ids(tok_str)
            if tid is not None:
                stop_ids.add(tid)
    for tok_name in ["<eos>", "</s>", "<turn|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok_name)
        if tid is not None and tid != tokenizer.unk_token_id:
            stop_ids.add(tid)

    mode = "baseline" if args.no_turbo else f"TurboQuant {args.bits}-bit"
    print(f"Ready. [{mode}]\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not prompt:
            continue

        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        # Prefill
        if args.no_turbo:
            cache = None
            t0 = time.time()
            with torch.no_grad():
                out = model(inputs["input_ids"], use_cache=True)
            cache_obj = out.past_key_values
        else:
            cache = TurboQuantDynamicCache(nbits=args.bits)
            t0 = time.time()
            with torch.no_grad():
                out = model(inputs["input_ids"], past_key_values=cache, use_cache=True)
            cache.compress()
            cache_obj = cache

        # Decode
        tokens = []
        next_token = out.logits[:, -1:, :].argmax(dim=-1)

        for _ in range(args.max_tokens):
            tok_id = next_token.item()
            if tok_id in stop_ids:
                break
            tokens.append(tok_id)
            with torch.no_grad():
                out = model(next_token, past_key_values=cache_obj, use_cache=True)
            next_token = out.logits[:, -1:, :].argmax(dim=-1)

        elapsed = time.time() - t0
        response = tokenizer.decode(tokens, skip_special_tokens=True)
        tok_per_sec = len(tokens) / elapsed if elapsed > 0 else 0

        print(f"\nGemma: {response}")
        info = f"{len(tokens)} tokens, {elapsed:.1f}s, {tok_per_sec:.1f} tok/s"
        if cache and not args.no_turbo:
            mem = cache.memory_summary()
            info += f", KV: {mem['baseline_bytes']/1024:.0f}→{mem['compressed_bytes']/1024:.0f} KB ({mem['ratio']:.1f}x)"
        print(f"  [{info}]\n")


if __name__ == "__main__":
    main()
