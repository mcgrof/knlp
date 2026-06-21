"""Phase 1: metadata audit. Discovers structure at runtime (never assumes Phi): QKV separate/fused,
per-proj bias, partial-RoPE rotary_dim / pass_through_dim, qk_layernorm, rope_theta, etc.
"""

import argparse
import csv
import os

import torch

import _phicommon as pc
from _phicommon import kbc, t2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--models-file")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    specs = t2.load_models_spec(args.models_file, args.models, args.only)
    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for m in specs:
        mid, sn = m["model_id"], m["short_name"]
        try:
            model, tok = kbc.load_model(mid, args.dtype, "cpu", t2.trc_for(mid))
            cfg = model.config
            infos = kbc.discover_attention(model)
            rotary_dim, head_dim = pc.rotary_split(model)
            i0 = infos[0]
            rows.append(
                dict(
                    model=sn,
                    model_id=mid,
                    arch=cfg.architectures[0] if cfg.architectures else "",
                    hidden_size=getattr(cfg, "hidden_size", None),
                    num_layers=len(infos),
                    n_q_heads=i0["n_q_heads"],
                    n_kv_heads=i0["n_kv_heads"],
                    head_dim=head_dim,
                    fused_qkv=i0["fused"],
                    has_q_bias=i0["has_q_bias"],
                    has_k_bias=i0["has_k_bias"],
                    has_v_bias=i0["has_v_bias"],
                    qk_layernorm=pc.has_qk_norm(model),
                    partial_rotary_factor=getattr(cfg, "partial_rotary_factor", None),
                    rotary_dim=rotary_dim,
                    pass_through_dim=head_dim - rotary_dim,
                    rope_theta=i0["rope_theta"],
                    max_position_embeddings=getattr(
                        cfg, "max_position_embeddings", None
                    ),
                    attention_dropout=getattr(cfg, "attention_dropout", None),
                    attn_impl=cfg._attn_implementation,
                    partial_rope=(rotary_dim < head_dim),
                )
            )
            print(
                f"[meta] {sn}: heads={i0['n_q_heads']}/{i0['n_kv_heads']} head_dim={head_dim} "
                f"rotary={rotary_dim} passthru={head_dim-rotary_dim} "
                f"k_bias={i0['has_k_bias']} qknorm={pc.has_qk_norm(model)} fused={i0['fused']}"
            )
            del model
        except Exception as e:
            print(f"[FAIL] {sn}: {type(e).__name__}: {str(e)[:120]}")
    if rows:
        with open(
            os.path.join(args.output_dir, "metadata_audit.csv"), "w", newline=""
        ) as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            [w.writerow(r) for r in rows]
        md = [
            "# Metadata audit",
            "",
            "| model | arch | heads(q/kv) | head_dim | rotary | passthru "
            "| k_bias | qk_norm | fused |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            md.append(
                f"| {r['model']} | {r['arch']} | {r['n_q_heads']}/{r['n_kv_heads']} | "
                f"{r['head_dim']} | {r['rotary_dim']} | {r['pass_through_dim']} | "
                f"{r['has_k_bias']} | {r['qk_layernorm']} | {r['fused_qkv']} |"
            )
        open(os.path.join(args.output_dir, "metadata_audit.md"), "w").write(
            "\n".join(md)
        )
    print(f"[meta] -> {args.output_dir}")


if __name__ == "__main__":
    main()
