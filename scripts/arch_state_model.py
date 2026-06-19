"""Phase 0 / Table 1: derive the decode STATE SLOPE for each architecture from config alone.

The architecture-pivot premise is that full-attention decode keeps a context-LINEAR key scan no
scheduler removes, and the fix is to change the state representation. This quantifies that from
each model's config.json (no GPU): the resident KV/latent/recurrent state and the per-decode-token
bytes read, as

    resident_state_bytes(ctx) = fixed_state + slope * ctx
    decode_bytes_per_token(ctx) = active_weight_bytes + state_read(ctx)   (a roofline floor)

The context SLOPE is the headline. Qwen2.5 GQA scans all layers; MLA shrinks the per-layer cache;
Gated-DeltaNet / KDA make 75% of layers a FIXED recurrent state so only 25% of the stack carries a
slope. Slopes come from clear config dims; the recurrent FIXED state is an estimate (head x d_k x
d_v) flagged to confirm against the real vLLM allocation (per the plan: do not trust copied dims).
This is an idealized roofline -- it ignores MoE routing, kernel maturity, conv state -- so it sizes
the OPPORTUNITY; the pods then measure whether MoE/kernel overhead eats it.
"""

import json

HBM_TBPS = 3.35e12  # H100 HBM3 bytes/s
CTXS = [4096, 32768, 65536, 131072, 262144, 1048576]


def fmt_bytes(b):
    for u, d in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if b >= d:
            return f"{b/d:.2f}{u}"
    return f"{b:.0f}B"


# --- architecture state descriptors (dims read from the fetched config.json) ---
# elem_per_tok_per_layer = the cache elements a context-bearing layer stores per token.
# A GQA layer caches K+V = 2 * n_kv_heads * head_dim. An MLA layer caches kv_lora_rank +
# qk_rope_head_dim (one shared latent + decoupled RoPE key, NOT per-head). Recurrent layers
# cache nothing per token; they keep a FIXED state matrix ~ n_v_heads * d_k * d_v.
ARCH = {
    "Qwen2.5-7B (GQA, control)": dict(
        layers=28,
        active_w=7.0e9,  # dense 7B, INT8 = ~7 GB
        ctx_layers=28,
        elem=2 * 4 * 128,  # K+V, 4 kv heads, head_dim 128
        fixed_state=0,
        note="all 28 layers GQA; pre-bias FP8 K8/V8",
    ),
    "DeepSeek-V2-Lite (native MLA)": dict(
        layers=27,
        active_w=2.4e9,  # 16B total, 2.4B active (MoE)
        ctx_layers=27,
        elem=512 + 64,  # kv_lora_rank + qk_rope_head_dim
        fixed_state=0,
        note="all 27 layers MLA latent(512)+rope(64); active 2.4B",
    ),
    "Qwen3-Next-80B-A3B (GDN+GQA)": dict(
        layers=48,
        active_w=3.0e9,  # 80B total, 3B active
        ctx_layers=12,  # full_attention_interval=4 -> 48/4 = 12 attn layers
        elem=2 * 2 * 256,  # K+V, 2 kv heads, head_dim 256
        # 36 GDN layers, recurrent state ~ n_v_heads(32) x d_k(128) x d_v(128); BF16 state
        fixed_state=36 * 32 * 128 * 128 * 2,
        note="12 GQA + 36 GatedDeltaNet (fixed state, ESTIMATE); active 3B",
    ),
    "Kimi-Linear-48B-A3B (KDA+MLA)": dict(
        layers=27,
        active_w=3.0e9,  # 48B total, 3B active
        ctx_layers=7,  # 7 full_attn (MLA) layers per linear_attn_config
        elem=512 + 64,  # MLA latent + rope
        # 20 KDA layers, recurrent state ~ n_heads(32) x d_k(128) x d_v(128); BF16 state
        fixed_state=20 * 32 * 128 * 128 * 2,
        note="7 MLA + 20 KDA (fixed state, ESTIMATE); active 3B",
    ),
}


def slope_bytes(a, bytes_per_elem):
    return a["ctx_layers"] * a["elem"] * bytes_per_elem


def main():
    print("=" * 92)
    print(
        "TABLE 1: architecture state model  (FP8 cache = 1 byte/elem; slope is the headline)"
    )
    print("=" * 92)
    print(
        f"{'model':<32}{'ctx-layers':>11}{'slope B/tok':>13}{'fixed state':>13}"
        f"{'vs Qwen slope':>14}"
    )
    print("-" * 92)
    base = slope_bytes(ARCH["Qwen2.5-7B (GQA, control)"], 1)
    for name, a in ARCH.items():
        s = slope_bytes(a, 1)  # FP8
        print(
            f"{name:<32}{a['ctx_layers']:>5}/{a['layers']:<5}{s:>13,}"
            f"{fmt_bytes(a['fixed_state']):>13}{s/base:>13.2f}x"
        )

    print("\n" + "=" * 92)
    print("Resident cache state vs context (FP8 cache):  fixed_state + slope*ctx")
    print("=" * 92)
    hdr = "".join(f"{(str(c//1024)+'K'):>11}" for c in CTXS)
    print(f"{'model':<32}{hdr}")
    print("-" * 92)
    for name, a in ARCH.items():
        s = slope_bytes(a, 1)
        row = "".join(f"{fmt_bytes(a['fixed_state'] + s*c):>11}" for c in CTXS)
        print(f"{name:<32}{row}")

    print("\n" + "=" * 92)
    print(
        "Decode roofline FLOOR per token = active_weights(int8/fp8) + state_read(FP8) / 3.35TB/s"
    )
    print(
        "(idealized: ignores MoE routing, kernel overhead, conv state -- sizes the opportunity)"
    )
    print("=" * 92)
    print(f"{'model':<32}{hdr}")
    print("-" * 92)
    for name, a in ARCH.items():
        s = slope_bytes(a, 1)
        row = ""
        for c in CTXS:
            state_read = a["fixed_state"] + s * c
            tpot_ms = (a["active_w"] + state_read) / HBM_TBPS * 1e3
            row += f"{tpot_ms:>10.2f} "
        print(f"{name:<32}{row}")
    print(
        "\nnote: weights read = ACTIVE params only (MoE reads ~3B not the full 48-80B); the "
        "\nQwen control reads its full 7B dense. State-read = whole resident cache once/token."
    )

    print("\n" + "=" * 92)
    print("Layer composition + provenance")
    print("=" * 92)
    for name, a in ARCH.items():
        print(f"  {name}: {a['note']}")
    print(
        "\nCAVEATS: recurrent FIXED-state is an ESTIMATE (n_heads x d_k x d_v, BF16) -- confirm "
        "against\nthe real vLLM allocation (F2/G2/H2). Slopes use clear config dims and are firm. "
        "MoE active-\nweight bytes assume the published active-param counts. TPOT here is a roofline "
        "floor, NOT a\nmeasured runtime -- the pods test whether MoE/kernel overhead eats the gap."
    )


if __name__ == "__main__":
    main()
