"""
TurboQuant HuggingFace Cache Integration

Drop-in replacement for DynamicCache that compresses KV after prefill.

Strategy (matching vivekvar/turboquant):
  - PREFILL: store original K, V (no compression) → bit-identical logits
  - After prefill: batch-compress all cached K, V
  - DECODE: new tokens are compressed on arrival; cached tokens stay compressed
  - Attention uses decompressed values (on-the-fly from compressed storage)

Usage:
    cache = TurboQuantDynamicCache(nbits=4)
    output = model.generate(**inputs, past_key_values=cache)
"""

import torch
from transformers.cache_utils import DynamicCache, CacheLayerMixin
from turboquant_kv_cache import TurboQuantizer


class TurboQuantLayer(CacheLayerMixin):
    """A single layer's KV store with post-prefill compression.

    Lifecycle:
      1. Prefill: stores original K, V (like DynamicLayer)
      2. compress(): batch-compresses stored K, V into packed indices + norms
      3. Decode: new tokens compressed on arrival, appended to compressed store
      4. keys/values properties return decompressed tensors for attention
    """

    def __init__(self, quantizer: TurboQuantizer, skip: bool = False):
        self.quantizer = quantizer
        self.skip = skip

        # Original storage (used during prefill, cleared after compression)
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None

        # Compressed storage (populated after compress() call)
        self.k_packed = None
        self.k_norms = None
        self.v_packed = None
        self.v_norms = None
        self._compressed = False  # Whether cache has been compressed

        # Cached decompressed tensors (lazily rebuilt when compressed data changes)
        self._keys_decompressed: torch.Tensor | None = None
        self._values_decompressed: torch.Tensor | None = None
        self._decomp_valid = False

        self.is_initialized = False
        self.is_sliding = False
        self._shape_meta = None  # (batch, heads, ?, head_dim) — store for reshaping

    def lazy_initialization(self, key_states, value_states):
        self._dtype = key_states.dtype
        self._device = key_states.device
        self.keys = torch.tensor([], dtype=self._dtype, device=self._device)
        self.values = torch.tensor([], dtype=self._dtype, device=self._device)
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        """Store new K, V. During prefill: store original. During decode: compress."""
        batch, heads, new_tokens, head_dim = key_states.shape
        self._shape_meta = (batch, heads, head_dim)

        if self.skip or not self._compressed:
            # PREFILL path (or skip layer): store original values
            if not self.is_initialized or self.keys is None or self.keys.numel() == 0:
                self.keys = key_states
                self.values = value_states
                self.is_initialized = True
            else:
                self.keys = torch.cat([self.keys, key_states], dim=-2)
                self.values = torch.cat([self.values, value_states], dim=-2)
            return self.keys, self.values

        # DECODE path: compress new tokens and append to compressed storage
        # Cast to float32 for quantizer (rotation matrix is float32)
        orig_dtype = key_states.dtype
        flat_k = key_states.reshape(-1, head_dim).float()
        flat_v = value_states.reshape(-1, head_dim).float()

        k_packed, k_norms = self.quantizer.encode(flat_k)
        v_packed, v_norms = self.quantizer.encode(flat_v)

        self.k_packed = torch.cat([self.k_packed, k_packed], dim=-2)
        self.k_norms = torch.cat([self.k_norms, k_norms], dim=-1)
        self.v_packed = torch.cat([self.v_packed, v_packed], dim=-2)
        self.v_norms = torch.cat([self.v_norms, v_norms], dim=-1)
        self._decomp_valid = False

        # Rebuild decompressed tensors for attention (cast back to model dtype)
        all_keys, all_values = self._decompress_all()
        self.keys = all_keys.to(orig_dtype)
        self.values = all_values.to(orig_dtype)
        return self.keys, self.values

    def compress(self):
        """Batch-compress all stored K, V. Call after prefill."""
        if self._compressed or self.skip:
            return
        if self.keys is None or self.keys.numel() == 0:
            return

        self._orig_dtype = self.keys.dtype
        batch, heads, seq_len, head_dim = self.keys.shape
        self._shape_meta = (batch, heads, head_dim)

        # Cast to float32 for quantizer
        flat_k = self.keys.reshape(-1, head_dim).float()
        flat_v = self.values.reshape(-1, head_dim).float()

        self.k_packed, self.k_norms = self.quantizer.encode(flat_k)
        self.v_packed, self.v_norms = self.quantizer.encode(flat_v)
        self._compressed = True
        self._decomp_valid = False

        # Replace keys/values with decompressed versions (cast back)
        keys, values = self._decompress_all()
        self.keys = keys.to(self._orig_dtype)
        self.values = values.to(self._orig_dtype)

    def _decompress_all(self):
        """Decompress all packed K, V back to full tensors."""
        batch, heads, head_dim = self._shape_meta
        n_flat = self.k_norms.shape[-1]
        seq_len = n_flat // (batch * heads)

        k_flat = self.quantizer.decode(self.k_packed, self.k_norms)
        v_flat = self.quantizer.decode(self.v_packed, self.v_norms)

        keys = k_flat.reshape(batch, heads, seq_len, head_dim)
        values = v_flat.reshape(batch, heads, seq_len, head_dim)
        return keys, values

    def get_seq_length(self):
        if self.keys is None:
            return 0
        return self.keys.shape[-2]

    def get_max_cache_shape(self):
        return None

    def reset(self):
        self.k_packed = self.k_norms = None
        self.v_packed = self.v_norms = None
        self.keys = self.values = None
        self._compressed = False
        self._decomp_valid = False
        self.is_initialized = False

    def reorder_cache(self, beam_idx):
        if self.keys is not None:
            device = self.keys.device
            self.keys = self.keys.index_select(0, beam_idx.to(device))
            self.values = self.values.index_select(0, beam_idx.to(device))
        if self.k_packed is not None:
            device = self.k_packed.device
            self.k_packed = self.k_packed.index_select(0, beam_idx.to(device))
            self.k_norms = self.k_norms.index_select(0, beam_idx.to(device))
            self.v_packed = self.v_packed.index_select(0, beam_idx.to(device))
            self.v_norms = self.v_norms.index_select(0, beam_idx.to(device))

    def get_mask_sizes(self, query_length):
        seq_len = self.get_seq_length()
        return seq_len, seq_len

    @property
    def device(self):
        if self.keys is not None:
            return self.keys.device
        return torch.device("cpu")

    @property
    def dtype(self):
        if self.keys is not None:
            return self.keys.dtype
        return torch.float32


class TurboQuantDynamicCache(DynamicCache):
    """Drop-in DynamicCache with post-prefill TurboQuant compression.

    Prefill logits are bit-identical (no compression during prefill).
    After prefill, K/V are batch-compressed for memory savings.
    Decode tokens are compressed on arrival.

    Args:
        nbits: quantization bit-width (2, 3, or 4)
        skip_layers: set of layer indices to keep uncompressed (outlier layers)
        seed: random seed for the rotation matrix
    """

    def __init__(self, nbits=4, skip_layers=None, seed=42):
        super().__init__()
        self.nbits = nbits
        self.skip_layers = skip_layers or set()
        self.seed = seed
        self._quantizers: dict[int, TurboQuantizer] = {}
        self._prefill_compressed = False

    def _get_quantizer(self, head_dim):
        if head_dim not in self._quantizers:
            self._quantizers[head_dim] = TurboQuantizer(
                dim=head_dim, nbits=self.nbits, seed=self.seed
            )
        return self._quantizers[head_dim]

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        # Create layer if needed
        while len(self.layers) <= layer_idx:
            self.layers.append(None)

        if self.layers[layer_idx] is None:
            head_dim = key_states.shape[-1]
            skip = layer_idx in self.skip_layers
            self.layers[layer_idx] = TurboQuantLayer(
                quantizer=self._get_quantizer(head_dim),
                skip=skip,
            )

        # Auto-compress after prefill: if we have stored tokens and this is
        # a single-token update, compress the prefill cache first
        layer = self.layers[layer_idx]
        if (not self._prefill_compressed
                and layer.is_initialized
                and layer.keys is not None
                and layer.keys.numel() > 0
                and key_states.shape[-2] == 1):
            # First decode token — compress all layers now
            self._compress_prefill()

        return layer.update(key_states, value_states, *args, **kwargs)

    def _compress_prefill(self):
        """Batch-compress all layers' KV cache (called once after prefill)."""
        if self._prefill_compressed:
            return
        for layer in self.layers:
            if layer is not None:
                layer.compress()
        self._prefill_compressed = True

    def compress(self):
        """Manually trigger compression (alternative to auto-detection)."""
        self._compress_prefill()

    def memory_summary(self):
        total_compressed = 0
        total_baseline = 0

        for layer in self.layers:
            if layer is None or not layer.is_initialized:
                continue
            seq_len = layer.get_seq_length()
            if layer.skip or not layer._compressed:
                elem_size = layer.keys.element_size()
                heads = layer.keys.shape[1]
                dim = layer.keys.shape[-1]
                layer_bytes = 2 * seq_len * heads * dim * elem_size
                total_compressed += layer_bytes
                total_baseline += layer_bytes
            else:
                q = layer.quantizer
                heads = layer._shape_meta[1]
                n_vecs = seq_len * heads
                total_compressed += 2 * n_vecs * q.bytes_per_vector()
                total_baseline += 2 * n_vecs * q.baseline_bytes_per_vector()

        rotation_overhead = sum(q.dim * q.dim * 4 for q in self._quantizers.values())

        return {
            "compressed_bytes": total_compressed + rotation_overhead,
            "baseline_bytes": total_baseline,
            "rotation_overhead_bytes": rotation_overhead,
            "ratio": total_baseline / (total_compressed + rotation_overhead) if total_compressed > 0 else 0,
        }

    @classmethod
    def calibrate_skip_layers(cls, model, tokenizer, threshold=3.0, prompt="Hello world"):
        """Auto-detect outlier layers by K norm distribution."""
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)

        past_kv = outputs.past_key_values
        norms = []
        if hasattr(past_kv, 'layers'):
            for layer in past_kv.layers:
                if hasattr(layer, 'keys') and layer.keys is not None:
                    norms.append(layer.keys.norm(dim=-1).mean().item())
                else:
                    norms.append(0.0)

        if not norms:
            return set()

        median_norm = sorted(norms)[len(norms) // 2]
        skip = set()
        for i, n in enumerate(norms):
            if median_norm > 0 and n / median_norm > threshold:
                skip.add(i)
        return skip
