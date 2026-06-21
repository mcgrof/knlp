"""GPT-J K/V cache fake-quant -- the eager-only special case the ALL_ATTENTION_FUNCTIONS harness
cannot reach. In transformers 5.x GPT-J has no SDPA path and its eager attention does NOT route
through the attention interface, so the registry monkeypatch is a SILENT NO-OP and GPT-J would
falsely report "FP8-K tolerant". Instead we patch each GPTJAttention._attn instance -- the boundary
that receives post-RoPE, cache-updated K/V immediately before QK^T (the same read-point as the
modern harness). Design from a Codex (gpt-5.5) consult.

GPT-J matters as the partial-RoPE BIASLESS control: Phi/Pythia/NeoX all carry a K-bias that confounds
bias vs partial-RoPE, so a biasless partial-RoPE model is the clean separator.

FAIL-CLOSED: install() asserts every layer is a GPTJAttention with the expected _attn signature and
that the patched count equals n_layer; verify() runs a K-nuke + audit and raises unless the quant
provably changed K -- so a false "tolerant" can never be published.
"""

import inspect
import os
import sys
import types

import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # tools/kv
import k_bias_common as kbc  # noqa: E402


def _q(x, spec):
    if spec["fmt"] is None:
        return x
    return kbc._quant_lastdims(
        x, spec["fmt"], spec["bits"], spec["layout"], spec["group"], False
    )


class GPTJKVAttnPatch:
    """Patch GPTJAttention._attn to fake-quantize post-RoPE K (and V). k_spec/v_spec are parse_spec
    dicts. mode: 'full' | 'rotary_only' | 'passthrough_only' | 'passthrough_prebias' over K's
    channels (rotary_dim splits the head). knuke/audit are for the causality self-check.
    """

    def __init__(
        self,
        model,
        infos,
        k_spec,
        v_spec,
        rotary_dim=None,
        mode="full",
        pass_bias_by_layer=None,
        knuke=False,
        audit=False,
    ):
        self.model = model
        self.infos = infos
        self.k = k_spec
        self.v = v_spec
        self.rd = rotary_dim
        self.mode = mode
        self.pass_bias = pass_bias_by_layer or {}
        self.knuke = knuke
        self.audit = audit
        self.saved = []
        self.calls = 0
        self.max_k_delta = 0.0

    def _qk(self, k):
        if self.knuke:
            return torch.zeros_like(k)
        if self.mode == "full" or self.rd is None:
            return _q(k, self.k)
        rot, pas = k[..., : self.rd], k[..., self.rd :]
        if self.mode == "rotary_only":
            return torch.cat([_q(rot, self.k), pas], dim=-1)
        if self.mode == "passthrough_only":
            return torch.cat([rot, _q(pas, self.k) if pas.numel() else pas], dim=-1)
        if self.mode == "passthrough_prebias":
            return torch.cat(
                [rot, _q(pas, self.k)], dim=-1
            )  # bias subtracted by caller per layer
        return _q(k, self.k)

    def install(self):
        impl = getattr(self.model.config, "_attn_implementation", None)
        if impl != "eager":
            raise RuntimeError(
                f"GPT-J patch needs attn_implementation='eager', got {impl!r}"
            )
        expected = getattr(
            self.model.config,
            "num_hidden_layers",
            getattr(self.model.config, "n_layer", len(self.infos)),
        )
        patch = self
        for info in self.infos:
            mod = info["attn_module"]
            if mod.__class__.__name__ != "GPTJAttention":
                raise RuntimeError(
                    f"expected GPTJAttention, got {mod.__class__.__name__}"
                )
            params = list(inspect.signature(mod._attn).parameters)
            if params[:3] != ["query", "key", "value"]:
                raise RuntimeError(f"unexpected GPT-J _attn signature: {params[:4]}")
            orig = mod._attn  # bound method
            self.saved.append((mod, orig))
            li = info["layer_idx"]

            def wrapped(attn_mod, query, key, value, *a, _orig=orig, _li=li, **kw):
                patch.calls += 1
                pb = patch.pass_bias.get(_li)
                if (
                    patch.mode == "passthrough_prebias"
                    and pb is not None
                    and patch.rd is not None
                ):
                    b = pb.to(key.dtype).to(key.device)[None, :, None, :]
                    rot, pas = key[..., : patch.rd], key[..., patch.rd :]
                    pas = _q(pas - b, patch.k) + b
                    k2 = torch.cat([rot, pas], dim=-1)
                else:
                    k2 = patch._qk(key)
                if patch.audit:
                    patch.max_k_delta = max(
                        patch.max_k_delta, (k2.float() - key.float()).abs().max().item()
                    )
                v2 = _q(value, patch.v)
                return _orig(query, k2, v2, *a, **kw)

            mod._attn = types.MethodType(wrapped, mod)
        if len(self.saved) != expected:
            raise RuntimeError(
                f"patched {len(self.saved)} attn modules, expected {expected}"
            )
        return self

    def remove(self):
        for mod, orig in self.saved:
            mod._attn = orig
        self.saved.clear()

    def __enter__(self):
        return self.install()

    def __exit__(self, *a):
        self.remove()


@torch.no_grad()
def verify_intercepts(model, infos, device="cpu"):
    """Prove the patch is causal -- a single K-nuke+audit run shows it intercepts every layer
    (calls == n_layer), it touches K (audit delta = max|K| > 0), and zeroing K changes the logits
    (no-op would be EXACTLY 0). Raises on a silent no-op so a false 'tolerant' can never be
    published. (fp8 of small K can be near-lossless on a tiny model, so the audit nukes K rather
    than relying on the fp8 delta; the fp8 path is the same install, exercised by the run itself.)
    """
    ids = torch.randint(0, 64, (1, 6)).to(device)
    base = model(ids).logits.detach().clone()
    k8, bf16 = kbc.parse_spec("fp8:per_tensor"), kbc.parse_spec("bf16")
    with GPTJKVAttnPatch(model, infos, k8, bf16, knuke=True, audit=True) as p:
        nuked = model(ids).logits.detach().clone()
    delta = (base.float() - nuked.float()).abs().max().item()
    if p.calls < len(infos) or p.max_k_delta <= 0.0:
        raise RuntimeError(
            f"GPT-J patch no-op: calls={p.calls} max_k_delta={p.max_k_delta}"
        )
    if (
        delta <= 1e-4
    ):  # a true no-op is exactly 0; 1e-4 clears fp noise, passes tiny models
        raise RuntimeError(
            f"GPT-J K-nuke did not change logits (delta={delta}) -- not causal"
        )
    return dict(
        calls=p.calls, max_k_delta=round(p.max_k_delta, 6), knuke_delta=round(delta, 6)
    )
