#!/usr/bin/env python3
"""Rotary-position handling for cartridge keys, with a self-test.

A cartridge stores its keys after rotary position embedding (RoPE):
slot i of the cache was produced at absolute position i, so the stored
key is the pre-RoPE key rotated by an angle that grows with i. Anything
that compares or averages keys across slots, or across cartridges whose
slot i holds a different token, has to remove that rotation first; the
per-slot angle is a property of the position, not of the content, and
it is put back when the result is written into a cartridge again.

Qwen3 rotates the two halves of each head dimension as pairs (d, d +
D/2) with frequency theta^(-2d/D), so pair d rotates by position *
theta^(-2d/D); the low d pairs turn fast, the high ones barely move
within a cartridge. ``derot`` and ``rerot`` apply the inverse and the
forward rotation for a whole cache tensor of shape [..., P, D], slot
axis second to last, position = slot index (the library initializes a
cartridge with position_ids 0..P-1 over the frozen sink and the
trainable slots alike).

The self-test needs the model: it captures the pre-RoPE keys straight
from ``k_norm`` with a forward hook while building a first-p cartridge,
derotates the cartridge's stored keys, and checks per layer that they
match (cosine over the fast pairs above 0.9999), that shifting every
position by one breaks the match (cosine well below that), and that
derot followed by rerot returns the bf16 keys within bf16 rounding.

Env (selftest only):
    MODEL    Qwen/Qwen3-8B (default)
    RECORD   document text to build the test cartridge from
    P        cartridge length (default 64; small keeps the test fast)
    DEVICE   cuda (default)

Usage:
    python3 cas_kv_rope.py selftest
"""

import os
import sys

import torch

FAST_PAIRS = 32  # the pairs whose rotation is visible within a cartridge


def rope_cos_sin(positions, head_dim, theta, device=None, dtype=torch.float32):
    """cos and sin of shape [P, D] in the model's (freqs, freqs) layout."""
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    pos = torch.as_tensor(positions, device=device, dtype=torch.float32)
    freqs = pos[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def rotate(x, positions, theta=1e6, inverse=False):
    """Rotate the slot axis of ``x`` [..., P, D] by its positions; computed
    in float32, returned in the input dtype."""
    dt = x.dtype
    xf = x.float()
    cos, sin = rope_cos_sin(positions, x.shape[-1], theta, device=x.device)
    if inverse:
        sin = -sin
    return ((xf * cos) + (_rotate_half(xf) * sin)).to(dt)


def derot(keys, positions=None, theta=1e6):
    """Remove the rotary embedding: stored keys -> pre-RoPE keys."""
    if positions is None:
        positions = torch.arange(keys.shape[-2], device=keys.device)
    return rotate(keys, positions, theta, inverse=True)


def rerot(keys, positions=None, theta=1e6):
    """Apply the rotary embedding: pre-RoPE keys -> stored keys."""
    if positions is None:
        positions = torch.arange(keys.shape[-2], device=keys.device)
    return rotate(keys, positions, theta, inverse=False)


def fast_pair_cosine(a, b, n_pairs=FAST_PAIRS):
    """Cosine over the ``n_pairs`` fastest-rotating pairs, per head, with
    all slots flattened together: a, b [H, P, D]."""
    d = a.shape[-1] // 2
    idx = list(range(n_pairs)) + list(range(d, d + n_pairs))
    af = a[..., idx].float().flatten(-2)
    bf = b[..., idx].float().flatten(-2)
    return torch.nn.functional.cosine_similarity(af, bf, dim=-1)


def selftest():
    os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/tmp/cas_kv_rope")
    from transformers import AutoTokenizer
    from cartridges.cache import AttnConfig
    from cartridges.initialization.tokenization_utils import (
        MODEL_TO_SYSTEM_PROMPT_TOKENIZER,
    )
    from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cas_make_init import kv_from_ids

    model_name = os.environ.get("MODEL", "Qwen/Qwen3-8B")
    device = os.environ.get("DEVICE", "cuda")
    p = int(os.environ.get("P", "64"))
    tok = AutoTokenizer.from_pretrained(model_name)
    text = open(os.environ["RECORD"]).read()
    fn = MODEL_TO_SYSTEM_PROMPT_TOKENIZER[tok.name_or_path.lower()]
    ids = fn(tokenizer=tok, content=text, max_tokens=p).squeeze(0).tolist()
    model = FlexQwen3ForCausalLM.from_pretrained(model_name).to(device)
    model = model.to(torch.bfloat16).eval()
    theta = float(model.config.rope_theta)
    attn = AttnConfig(
        n_layers=model.config.num_hidden_layers,
        n_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
    )
    captured = {}
    hooks = [
        layer.self_attn.k_norm.register_forward_hook(
            lambda m, i, o, l=l: captured.__setitem__(l, o.detach().float().cpu())
        )
        for l, layer in enumerate(model.model.layers)
    ]
    keys, _ = kv_from_ids(model, attn, ids)  # stored (rotated) keys, bf16
    for h in hooks:
        h.remove()
    print(f"[rope] theta={theta:.0f} p={p} layers={len(keys)}", flush=True)

    pos = torch.arange(p)
    worst_ok, best_shift, worst_rt = 1.0, 0.0, 0.0
    for l, k in enumerate(keys):
        pre = captured[l][0].transpose(0, 1)  # [B,T,H,D] -> [H,T,D]
        stored = k[0].float()  # [H,P,D]
        ok = fast_pair_cosine(derot(stored, pos, theta), pre).min().item()
        # a one-slot error must be visible: judge it on the 8 fastest pairs
        # (>= 0.18 rad per position), where slow pairs cannot mask it
        shifted = fast_pair_cosine(derot(stored, pos + 1, theta), pre, 8).max().item()
        rt = rerot(derot(k[0], pos, theta), pos, theta)
        rel = ((rt.float() - stored).norm() / stored.norm()).item()
        worst_ok, best_shift = min(worst_ok, ok), max(best_shift, shifted)
        worst_rt = max(worst_rt, rel)
        if l in (0, 1, len(keys) // 2, len(keys) - 1):
            print(
                f"[rope] L{l}: derot cos min {ok:.6f}  off-by-one cos max "
                f"{shifted:.4f}  bf16 round trip rel {rel:.2e}",
                flush=True,
            )
    # stored keys are bf16, so ~3e-4 relative noise is the floor
    passed = worst_ok > 0.999 and best_shift < 0.98 and worst_rt < 2e-2
    print(
        f"ROPE_SELFTEST {'PASS' if passed else 'FAIL'} derot_cos_min={worst_ok:.6f} "
        f"offbyone_cos_max={best_shift:.4f} roundtrip_rel_max={worst_rt:.2e}",
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        sys.exit(selftest())
    print(__doc__)
