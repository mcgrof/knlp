"""Physical FP8 cache with per-token, per-head scales and BF16 reads.

This is a numerical and storage reference, not a fused attention backend.
Each attention call temporarily reconstructs its own K/V inputs. Recurrent
state stays at the model's native precision. No reconstructed cache is kept.
"""

import torch
from transformers.cache_utils import (
    Cache,
    DynamicLayer,
    DynamicSlidingWindowLayer,
    LAYER_TYPE_CACHE_MAPPING,
)

MODES = ("native", "k16v8", "k8v8")


def encode(x):
    scale = x.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 448.0
    return (x.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn), scale


def decode(data, scale, dtype):
    return (data.float() * scale).to(dtype) if scale is not None else data


class PackedLayer(DynamicLayer):
    def __init__(self, mode, window=None):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.sliding_window = window
        self.is_sliding = window is not None
        self.cumulative_length = 0
        self.k_scale = self.v_scale = None

    def update(self, key_states, value_states, *args, **kwargs):
        if key_states.dtype != value_states.dtype:
            raise ValueError("Reference requires matching compute dtypes")
        if not self.is_initialized:
            self.dtype = key_states.dtype
            self.device = key_states.device
            self.keys = self.values = None
            self.is_initialized = True
        self.cumulative_length += key_states.shape[-2]
        outputs = []
        for attr, scale_attr, x, quantized in (
            ("keys", "k_scale", key_states, self.mode == "k8v8"),
            ("values", "v_scale", value_states, self.mode != "native"),
        ):
            data, scale = encode(x) if quantized else (x.clone(), None)
            old = getattr(self, attr)
            if old is not None:
                data = torch.cat((old, data), dim=-2)
                if quantized:
                    scale = torch.cat((getattr(self, scale_attr), scale), dim=-2)
            outputs.append(decode(data, scale, self.dtype))
            if self.is_sliding:
                # clone discards the old backing, including during long prefill.
                data = data[..., -self.sliding_window + 1 :, :].clone()
                if scale is not None:
                    scale = scale[..., -self.sliding_window + 1 :, :].clone()
            setattr(self, attr, data)
            setattr(self, scale_attr, scale)
        return tuple(outputs)

    def get_seq_length(self):
        return self.cumulative_length

    def get_mask_sizes(self, query_length):
        if self.is_sliding:
            offset = max(self.cumulative_length - self.sliding_window + 1, 0)
            return (
                min(self.cumulative_length, self.sliding_window - 1) + query_length,
                offset,
            )
        return self.cumulative_length + query_length, 0

    def get_max_cache_shape(self):
        return self.sliding_window if self.is_sliding else -1

    def crop(self, max_length):
        raise NotImplementedError("Speculative decoding is outside this reference")

    def batch_repeat_interleave(self, repeats):
        raise NotImplementedError("Only ordinary greedy decoding is qualified")

    def batch_select_indices(self, indices):
        raise NotImplementedError("Only ordinary greedy decoding is qualified")


def make_cache(config, mode):
    config = config.get_text_config(decoder=True)
    if mode not in MODES:
        raise ValueError(mode)
    kinds = list(config.layer_types)
    shared = getattr(config, "num_kv_shared_layers", 0)
    if shared:
        kinds = kinds[:-shared]
    layers = []
    for kind in kinds:
        if kind in ("full_attention", "sliding_attention"):
            window = config.sliding_window if kind == "sliding_attention" else None
            layers.append(PackedLayer(mode, window))
        elif kind in LAYER_TYPE_CACHE_MAPPING:
            layers.append(LAYER_TYPE_CACHE_MAPPING[kind](config))
        else:
            raise ValueError(f"Unqualified layer type: {kind}")
    return Cache(layers=layers)


def native_library_cache(config):
    """Explicit native layer list, including zero shared-layer Gemma configs."""
    config = config.get_text_config(decoder=True)
    layers = []
    kinds = list(config.layer_types)
    shared = getattr(config, "num_kv_shared_layers", 0)
    if shared:
        kinds = kinds[:-shared]
    for kind in kinds:
        if kind == "full_attention":
            layers.append(DynamicLayer(config))
        elif kind == "sliding_attention":
            layers.append(DynamicSlidingWindowLayer(config))
        else:
            layers.append(LAYER_TYPE_CACHE_MAPPING[kind](config))
    return Cache(layers=layers)


def inventory(cache):
    rows = []
    storage = {}
    for index, layer in enumerate(cache.layers):
        for attr in (
            "keys",
            "values",
            "k_scale",
            "v_scale",
            "conv_states",
            "recurrent_states",
        ):
            tensor = getattr(layer, attr, None)
            if not isinstance(tensor, torch.Tensor):
                continue
            backing = tensor.untyped_storage()
            key = (str(tensor.device), backing.data_ptr())
            storage[key] = backing.nbytes()
            rows.append(
                dict(
                    layer=index,
                    field=attr,
                    shape=list(tensor.shape),
                    dtype=str(tensor.dtype),
                    logical_bytes=tensor.numel() * tensor.element_size(),
                    backing_bytes=backing.nbytes(),
                    storage_id=len(storage) - 1,
                )
            )
    return {
        "tensors": rows,
        "logical_bytes": sum(r["logical_bytes"] for r in rows),
        "unique_backing_bytes": sum(storage.values()),
    }
