# SPDX-License-Identifier: MIT
"""Physical precision-placement reference; Q and attention arithmetic stay native.

Fresh operands arrive at the model's normal cache-write site after RoPE.
Only stored K/V change. Reconstruction is temporary for each attention layer;
this implementation provides no fused-kernel timing claim.
"""
import torch
from transformers.cache_utils import Cache, DynamicLayer, DynamicSlidingWindowLayer

ARMS = ("N", "AF", "RF", "SF", "AD", "RD", "SD")


def encode(x, fixed=None):
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Only native 16-bit operands are qualified")
    torch._assert_async(torch.isfinite(x).all(), "Nonfinite cache operand")
    xf = x.float()
    scale = (
        xf.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448
        if fixed is None
        else fixed
    )
    if scale.dtype != torch.float32:
        raise ValueError("Scales must be FP32")
    torch._assert_async(
        (torch.isfinite(scale) & (scale > 0)).all(),
        "Scales must be finite and positive",
    )
    return (xf / scale).clamp(-448, 448).to(torch.float8_e4m3fn), scale


def decode(codes, scale, dtype):
    return (codes.float() * scale).to(dtype) if scale is not None else codes


class Layer(DynamicLayer):
    def __init__(self, arm, index, fixed=None, observer=None, window=None):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        if arm.endswith("F") and fixed is None:
            raise ValueError("Missing frozen scales")
        self.arm = arm
        self.index = index
        self.fixed = fixed
        self.observer = observer
        self.sliding_window = window
        self.is_sliding = window is not None
        self.cumulative_length = 0
        self.k_scale = self.v_scale = None
        self.temporary_native_bytes_peak = 0

    def update(self, key_states, value_states, *args, **kwargs):
        if (
            key_states.shape != value_states.shape
            or key_states.dtype != value_states.dtype
        ):
            raise ValueError("K/V geometry and arithmetic must match")
        if key_states.ndim != 4 or key_states.dtype not in (
            torch.float16,
            torch.bfloat16,
        ):
            raise ValueError("Expected [batch,head,token,dimension] native operands")
        if not self.is_initialized:
            self.dtype = key_states.dtype
            self.device = key_states.device
            self.keys = self.values = None
            self.is_initialized = True
        elif self.dtype != key_states.dtype or self.device != key_states.device:
            raise ValueError("Cache dtype/device changed")
        if self.observer is not None:
            self.observer(self.index, key_states, value_states)
        self.cumulative_length += key_states.shape[-2]
        outputs = []
        temporary = 0
        for name, sname, x, component in [
            ("keys", "k_scale", key_states, "K"),
            ("values", "v_scale", value_states, "V"),
        ]:
            quantized = self.arm[0] == "S" or self.arm[0] == (
                "R" if component == "K" else "A"
            )
            fixed = None
            if quantized and self.arm.endswith("F"):
                fixed = getattr(self, sname)
                if fixed is None:
                    fixed = torch.tensor(
                        self.fixed[component], dtype=torch.float32, device=x.device
                    )
            data, scale = encode(x, fixed) if quantized else (x.clone(), None)
            old = getattr(self, name)
            if old is not None:
                data = torch.cat([old, data], dim=-2)
                if quantized and self.arm.endswith("D"):
                    scale = torch.cat([getattr(self, sname), scale], dim=-2)
            result = decode(data, scale, self.dtype)
            if quantized:
                temporary += result.numel() * result.element_size()
            outputs.append(result)
            if self.is_sliding:
                data = data[..., -self.sliding_window + 1 :, :].clone()
                if quantized and self.arm.endswith("D"):
                    scale = scale[..., -self.sliding_window + 1 :, :].clone()
            setattr(self, name, data)
            setattr(self, sname, scale)
        self.temporary_native_bytes_peak = max(
            self.temporary_native_bytes_peak, temporary
        )
        return tuple(outputs)

    def get_seq_length(self):
        return self.cumulative_length

    def get_mask_sizes(self, query_length):
        if self.is_sliding:
            return min(
                self.cumulative_length, self.sliding_window - 1
            ) + query_length, max(self.cumulative_length - self.sliding_window + 1, 0)
        return self.cumulative_length + query_length, 0

    def get_max_cache_shape(self):
        return self.sliding_window if self.is_sliding else -1

    def crop(self, *args):
        raise NotImplementedError("Speculative decoding unqualified")

    def batch_repeat_interleave(self, *args):
        raise NotImplementedError("Beam search unqualified")

    def batch_select_indices(self, *args):
        raise NotImplementedError("Beam search unqualified")


def make_cache(config, arm, fixed=None, observer=None, library=False):
    cfg = config.get_text_config(decoder=True)
    if cfg.model_type not in ("qwen3", "qwen2"):
        raise ValueError("Only specified dense Qwen model paths are admitted")
    kinds = list(cfg.layer_types)
    if len(kinds) != cfg.num_hidden_layers:
        raise ValueError("Shared or unrecognized layer mapping")
    if fixed is not None and len(fixed) != len(kinds):
        raise ValueError("Scale layer count")
    layers = []
    for index, kind in enumerate(kinds):
        if kind not in ("full_attention", "sliding_attention"):
            raise ValueError("Unsupported recurrent or other state layout")
        window = cfg.sliding_window if kind == "sliding_attention" else None
        if library:
            layers.append(
                DynamicSlidingWindowLayer(cfg) if window else DynamicLayer(cfg)
            )
        else:
            layers.append(
                Layer(
                    arm,
                    index,
                    None if fixed is None else fixed[index],
                    observer,
                    window,
                )
            )
    return Cache(layers=layers)


def inventory(cache):
    stores = {}
    rows = []
    for index, layer in enumerate(cache.layers):
        for attr in ("keys", "values", "k_scale", "v_scale"):
            x = getattr(layer, attr, None)
            if not isinstance(x, torch.Tensor):
                continue
            st = x.untyped_storage()
            key = (str(x.device), st.data_ptr())
            if key not in stores:
                stores[key] = (len(stores), st.nbytes())
            ident, nbytes = stores[key]
            rows.append(
                dict(
                    layer=index,
                    field=attr,
                    shape=list(x.shape),
                    stride=list(x.stride()),
                    dtype=str(x.dtype),
                    logical_bytes=x.numel() * x.element_size(),
                    backing_bytes=nbytes,
                    storage_id=ident,
                )
            )
    return dict(
        tensors=rows,
        logical_bytes=sum(r["logical_bytes"] for r in rows),
        unique_backing_bytes=sum(v[1] for v in stores.values()),
        scale_logical_bytes=sum(
            r["logical_bytes"] for r in rows if r["field"].endswith("scale")
        ),
        maximum_layer_reconstructed_native_bytes=max(
            (getattr(l, "temporary_native_bytes_peak", 0) for l in cache.layers),
            default=0,
        ),
        temporary_note="Native reconstruction is transient; float32 conversion/multiply workspace and allocator high-water are measured separately by Torch peaks. No persistent native shadow.",
    )
