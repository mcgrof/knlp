"""Independent KV quantizer and cached-attention qualification."""

import math

import torch

from research.mo_variants.model import Model
from .kv_cache import CachePolicy, bytes_per_token, fakequant, install, parity
from .model import Model as Backbone
from .model import author


def scalar_reference(values, bits, method):
    values = list(map(float, values))
    if method == "symmetric":
        m = 2 ** (bits - 1) - 1
        s = max(map(abs, values)) / m
        return [0.0 if s == 0 else max(-m, min(m, round(v / s))) * s for v in values]
    lower, upper = min(values), max(values)
    m = 2**bits - 1
    s = (upper - lower) / m
    return [
        lower if s == 0 else max(0, min(m, round((v - lower) / s))) * s + lower
        for v in values
    ]


@torch.no_grad()
def qualify(device):
    torch.manual_seed(89104)
    tests = []
    x = torch.randn(2, 2, 7, 64, device=device)
    x[0, 0, 0].zero_()
    x[0, 1, 0].fill_(1.7)
    for method in ("symmetric", "affine"):
        for bits in (2, 3, 4, 8):
            q = fakequant(x, bits, method)
            ref = torch.tensor(
                [
                    scalar_reference(v, bits, method)
                    for v in x.flatten(0, 2).cpu().tolist()
                ],
                device=device,
            ).reshape_as(x)
            torch.testing.assert_close(q, ref, atol=2e-6, rtol=2e-6)
            # Token independence is the condition allowing full-window Q/DQ
            # to represent a cache that quantizes each token once at write.
            joined = torch.cat(
                [fakequant(x[:, :, t : t + 1], bits, method) for t in range(7)], dim=2
            )
            assert torch.equal(joined, q)
            tests.append(f"scalar-reference-and-token-independence-{method}-{bits}")
    assert fakequant(x, 16) is x
    model = Model("V01", 2, 89104, vocab=97).to(device).eval()
    ids = torch.randint(0, 97, (2, 7), device=device)
    cast = lambda: torch.autocast(device_type=device, dtype=torch.bfloat16)
    with cast():
        original = model(ids)
        with install(model, CachePolicy()):
            passthrough = model(ids)
    assert torch.equal(original, passthrough)
    tests.append("whole-model-bit-identical-noquant-adapter")
    attn = model.lm.model.layers[0].self_attn
    hidden = torch.randn(2, 7, 128, device=device).bfloat16()
    positions = torch.arange(7, device=device).unsqueeze(0)
    errors = []
    cache_rounding = []
    for method, kb, vb in (
        ("symmetric", 16, 16),
        ("symmetric", 4, 4),
        ("symmetric", 16, 2),
        ("symmetric", 2, 16),
        ("affine", 2, 2),
    ):
        policy = CachePolicy(kb, vb, method)
        with install(model, policy), cast():
            full = attn(hidden, position_ids=positions, use_cache=True)
            past, pieces = None, []
            for t in range(7):
                previous = past
                out, _, past = attn(
                    hidden[:, t : t + 1],
                    position_ids=positions[:, t : t + 1],
                    past_key_value=past,
                    use_cache=True,
                )
                pieces.append(out)
                if previous is not None:
                    assert torch.equal(past[0][:, :, :t], previous[0])
                    assert torch.equal(past[1][:, :, :t], previous[1])
            decoded = torch.cat(pieces, dim=1)
            # Independent explicit-attention oracle: original Q/K/V and RoPE,
            # mathematical mask and softmax, without SDPA or cache_forward.
            q = attn._shape(attn.q_proj(hidden), 7, 2)
            k = attn._shape(attn.k_proj(hidden), 7, 2)
            v = attn._shape(attn.v_proj(hidden), 7, 2)
            cos, sin = attn.rotary_emb(v, seq_len=7)
            q, k = author.apply_rotary_pos_emb(q, k, cos, sin, positions)
            k, v = fakequant(k, kb, method), fakequant(v, vb, method)
            with torch.autocast(device_type=device, enabled=False):
                scores = q.float() @ k.float().transpose(-2, -1) / math.sqrt(64)
                mask = torch.ones(7, 7, device=device, dtype=torch.bool).triu(1)
                scores.masked_fill_(mask, float("-inf"))
                oracle = (scores.softmax(-1) @ v.float()).to(v.dtype)
            oracle = attn.o_proj(oracle.transpose(1, 2).reshape(2, 7, 128))
        # BF16 SDPA versus explicit FP32 softmax and full versus single GEMM
        # round differently; use a bound in BF16 output units and retain error.
        for other in (decoded, oracle):
            error = (full[0].float() - other.float()).abs().max().item()
            assert error <= 0.004, error
            errors.append(error)
        # Full versus one-token projection GEMMs can round BF16 differently
        # on CUDA, including a quantization-bin crossing. Exact cache-value
        # equality is not a valid cross-GEMM condition. Prior cache entries
        # must instead remain bit identical as each new token is appended.
        cache_rounding.append(
            {
                "method": method,
                "k_bits": kb,
                "v_bits": vb,
                "k_max_abs": (past[0].float() - full[2][0].float()).abs().max().item(),
                "v_max_abs": (past[1].float() - full[2][1].float()).abs().max().item(),
            }
        )
        tests.append(f"incremental-cache-and-explicit-attention-{method}-{kb}-{vb}")
    # Q/DQ is active in the actual complete residual architecture.
    with install(model, CachePolicy(2, 2)), cast():
        quantized = model(ids)
    assert not torch.equal(original, quantized)
    tests.append("quantization-changes-whole-model-output")
    for arm in ("R", "M"):
        backbone = Backbone(arm, 2, 89104, vocab=97).to(device).eval()
        parity(backbone, ids, device)
        tests.append("whole-model-caller-parity-" + arm)
    b = bytes_per_token({"k_bits": 4, "v_bits": 4, "method": "symmetric"})
    assert b["ideal_total_bytes"] == 4608 and b["bf16_total_bytes"] == 16384
    tests.append("packed-byte-accounting-includes-fp32-scales")
    return {
        "status": "pass",
        "device": device,
        "tests": tests,
        "max_attention_error": max(errors),
        "cross_gemm_cache_rounding": cache_rounding,
    }
