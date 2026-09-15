#!/usr/bin/env python3
"""Where the decode-side bytes go, per token, for the matched micro cells
and for a production Gated DeltaNet geometry.

For each hybrid layout this prints the bytes a decode step must move per
token: the recurrent state (read and written every token, so counted
twice) at a chosen state precision, and the attention KV cache read at a
chosen context length and KV precision. Pure arithmetic from the tensor
shapes; no model is loaded. The point is to show which quantization lever
matters at which scale.

    python3 scripts/decode_bytes_per_token.py
"""

BYTES = dict(fp32=4, bf16=2, fp8=1)


def gdn_state_values(heads, head_dim, expand_v=2):
    return heads * head_dim * (head_dim * expand_v)


def mom_state_values(memories, heads, head_dim, active, expand_v=2):
    per = heads * head_dim * (head_dim * expand_v)
    return dict(total=memories * per, active=active * per)


def attn_kv_values_per_token(kv_heads, head_dim):
    return 2 * kv_heads * head_dim


def row(
    name, layers_rec, layers_attn, rec_values_active, kv_heads, head_dim, ctx, sp, kp
):
    rec = 2 * layers_rec * rec_values_active * BYTES[sp]  # read + write per token
    kv = layers_attn * ctx * attn_kv_values_per_token(kv_heads, head_dim) * BYTES[kp]
    return name, rec / 2**20, kv / 2**20, (rec + kv) / 2**20


def table(title, rows):
    print(f"\n{title}")
    print(
        "| layout | recurrent state MB/token (read+write) | attention KV MB/token | total |"
    )
    print("|---|---|---|---|")
    for name, r, k, t in rows:
        print(f"| {name} | {r:.2f} | {k:.2f} | {t:.2f} |")


# micro contract: width 512, eight layers, attention 8 heads of 64
for ctx in (512, 4096):
    for sp, kp in (("fp32", "bf16"), ("fp8", "bf16"), ("fp8", "fp8")):
        rows = [
            row(
                "gdn3 (6 GDN 5x64 + 2 attn)",
                6,
                2,
                gdn_state_values(5, 64),
                8,
                64,
                ctx,
                sp,
                kp,
            ),
            row(
                "gdn3w (6 GDN 10x64 + 2 attn)",
                6,
                2,
                gdn_state_values(10, 64),
                8,
                64,
                ctx,
                sp,
                kp,
            ),
            row(
                "gdn3h128x3 (6 GDN 3x128 + 2 attn)",
                6,
                2,
                gdn_state_values(3, 128),
                8,
                64,
                ctx,
                sp,
                kp,
            ),
            row(
                "mom3 (6 MoM 5mem x 2x64, 3 active + 2 attn)",
                6,
                2,
                mom_state_values(5, 2, 64, 3)["active"],
                8,
                64,
                ctx,
                sp,
                kp,
            ),
            row(
                "gdn7 (7 GDN 5x64 + 1 attn)",
                7,
                1,
                gdn_state_values(5, 64),
                8,
                64,
                ctx,
                sp,
                kp,
            ),
            row("attn (8 attn)", 0, 8, 0, 8, 64, ctx, sp, kp),
        ]
        table(f"micro (width 512, 8 layers), context {ctx}, state {sp}, KV {kp}", rows)

# production-like Gated DeltaNet geometry: 24 layers, 4 heads of 256 (value 512),
# hybrid with one attention layer in four, attention 8 KV heads of 128
for ctx in (8192, 32768):
    for sp, kp in (("fp32", "bf16"), ("fp8", "bf16"), ("fp8", "fp8")):
        rows = [
            row(
                "pure GDN 24 layers 4x256",
                24,
                0,
                gdn_state_values(4, 256),
                8,
                128,
                ctx,
                sp,
                kp,
            ),
            row(
                "3:1 hybrid 18 GDN + 6 attn",
                18,
                6,
                gdn_state_values(4, 256),
                8,
                128,
                ctx,
                sp,
                kp,
            ),
            row(
                "7:1 hybrid 21 GDN + 3 attn",
                21,
                3,
                gdn_state_values(4, 256),
                8,
                128,
                ctx,
                sp,
                kp,
            ),
            row("full attention 24 layers", 0, 24, 0, 8, 128, ctx, sp, kp),
        ]
        table(
            f"production-like (24 layers, GDN 4 heads x 256), context {ctx}, state {sp}, KV {kp}",
            rows,
        )
