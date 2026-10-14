"""Write-once FP8 roundtrips, preserving fresh-token attention semantics."""

import math

import torch

from tools.kv.prequantized.recipes import FRESH, WRITE_FIRST, recipe


def reference_e4m3(value):
    """Scalar E4M3FN reference with ties to even and finite saturation."""
    sign = -1 if value < 0 else 1
    x = min(abs(value), 448.0)
    if x < 2**-6:
        return sign * round(x / 2**-9) * 2**-9
    exponent = math.floor(math.log2(x))
    step = 2 ** (exponent - 3)
    return sign * min(round(x / step) * step, 448.0)


def qdq(tensor, scale):
    scale = torch.as_tensor(scale, device=tensor.device, dtype=torch.float32)
    x = tensor.float()
    return ((x / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale).to(
        tensor.dtype
    )


class Observer:
    def __init__(self, arm="k16_v16", scales=None, calibrate=False, cache_recipe=FRESH):
        recipe(cache_recipe)
        self.cache_recipe = cache_recipe
        self.arm = arm
        self.scales = scales
        self.calibrate = calibrate
        self.maxima = {}
        self.counts = {}
        self.diagnostics = {}
        self.attention_reads = {}
        self.returned = {}

    def transform(self, layer, kind, tensor):
        key = f"{layer}.{kind}"
        self.counts.setdefault(
            layer, {"written_tokens": 0, "past_tokens_read": 0, "calls": 0}
        )
        if self.calibrate:
            maximum = tensor.float().abs().max()
            if key in self.maxima:
                maximum = torch.maximum(maximum, self.maxima[key])
            self.maxima[key] = maximum
        enabled = (kind == "k" and self.arm in ("k8_v16", "k8_v8")) or (
            kind == "v" and self.arm in ("k16_v8", "k8_v8")
        )
        if not enabled:
            return tensor
        if self.scales is None or key not in self.scales:
            raise ValueError(f"missing frozen scale: {key}")
        scale = self.scales[key]
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be positive and finite")
        rounded = qdq(tensor, scale)
        original = tensor.float()
        err = rounded.float() - original
        stats = torch.stack(
            [
                (original.abs() > scale * 448).sum().float(),
                (rounded == 0).sum().float(),
                (~torch.isfinite(rounded)).sum().float(),
                err.abs().sum(),
                err.square().sum(),
            ]
        ).double()
        item = self.diagnostics.setdefault(
            key, {"stats": torch.zeros_like(stats), "elements": 0}
        )
        item["stats"] += stats
        item["elements"] += tensor.numel()
        if "head_sample" not in item:
            item["head_sample"] = (
                err[0, : min(4, err.shape[1]), : min(4, err.shape[2]), :8]
                .detach()
                .cpu()
                .tolist()
            )
        return rounded

    def frozen_scales(self):
        return {
            key: float((x.float() / 448).item()) if x.item() else 1.0
            for key, x in self.maxima.items()
        }

    def receipt(self):
        diagnostics = {}
        for key, item in self.diagnostics.items():
            clip, zero, nonfinite, absolute, square = item["stats"].cpu().tolist()
            diagnostics[key] = {
                "elements": item["elements"],
                "clipped": int(clip),
                "zeros": int(zero),
                "nonfinite": int(nonfinite),
                "mean_absolute_error": absolute / item["elements"],
                "mean_squared_error": square / item["elements"],
                "head_sample": item["head_sample"],
            }
        return {
            "cache": self.counts,
            "attention_reads": self.attention_reads,
            "quantization": diagnostics,
        }


def make_cache(config, observer):
    from transformers.cache_utils import DynamicCache

    class WriteCache(DynamicCache):
        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            if layer_idx < len(self.layers):
                old_k = self.layers[layer_idx].keys
                old_v = self.layers[layer_idx].values
            else:
                old_k = old_v = None
            old_len = 0 if old_k is None or not old_k.numel() else old_k.shape[-2]
            # The unmodified implementation controls cumulative positions and eviction.
            returned_k, returned_v = super().update(
                key_states, value_states, layer_idx, cache_kwargs
            )
            stored_k = observer.transform(layer_idx, "k", key_states)
            stored_v = observer.transform(layer_idx, "v", value_states)
            if old_len:
                stored_k = torch.cat([old_k, stored_k], dim=-2)
                stored_v = torch.cat([old_v, stored_v], dim=-2)
            if observer.cache_recipe == WRITE_FIRST:
                returned_k, returned_v = stored_k, stored_v
            layer = self.layers[layer_idx]
            retained = layer.keys.shape[-2]
            layer.keys = stored_k[..., -retained:, :]
            layer.values = stored_v[..., -retained:, :]
            counts = observer.counts[layer_idx]
            counts["written_tokens"] += key_states.shape[-2]
            counts["past_tokens_read"] += old_len
            counts["calls"] += 1
            observer.returned[layer_idx] = (
                returned_k.data_ptr(),
                returned_v.data_ptr(),
                old_len,
            )
            # Fresh K/V are unrounded for prefill and the current decode token.
            # Only previously stored entries presented to attention are rounded.
            return returned_k, returned_v

    return WriteCache(config=config)


class AttentionReadProof:
    """Check actual attention inputs against the tensors returned by Cache.update."""

    def __init__(self, model, observer):
        self.model = model
        self.observer = observer
        self.original = None

    def __enter__(self):
        import importlib
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        backend = self.model.config._attn_implementation
        if backend == "eager":
            self.target = importlib.import_module(
                type(self.model.model.layers[0].self_attn).__module__
            )
            self.original = self.target.eager_attention_forward
        else:
            self.target = ALL_ATTENTION_FUNCTIONS
            self.original = self.target[backend]

        def checked(module, query, key, value, *args, **kwargs):
            idx = module.layer_idx
            expected = self.observer.returned.get(idx)
            if expected is None or (key.data_ptr(), value.data_ptr()) != expected[:2]:
                raise RuntimeError(
                    "attention did not consume the observed cache update"
                )
            self.observer.attention_reads[idx] = (
                self.observer.attention_reads.get(idx, 0) + expected[2]
            )
            return self.original(module, query, key, value, *args, **kwargs)

        if backend == "eager":
            self.target.eager_attention_forward = checked
        else:
            self.target[backend] = checked
        return self

    def __exit__(self, *_):
        if self.model.config._attn_implementation == "eager":
            self.target.eager_attention_forward = self.original
        else:
            self.target[self.model.config._attn_implementation] = self.original
