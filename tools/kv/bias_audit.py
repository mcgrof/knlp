"""Deliverable 1: K-projection-bias discovery + magnitude audit (cheap; weights only, no forward).

Loads each model, discovers the attention layout (separate vs fused QKV) and whether Q/K/V carry
a bias FROM THE ACTUAL MODULES, extracts the K bias (slicing fused QKV when needed), and computes
per-layer and per-KV-head magnitude statistics. Answers Q1/Q2: which models actually have a K
bias, and which have a LARGE one. abs>448 is a raw-e4m3 diagnostic only -- the load-bearing stats
are max_over_p99 / max_over_rms (outlier dominance) which is what crushes a per-tensor FP8 scale.
"""

import argparse
import os

import torch

import k_bias_common as kbc


def audit_model(model_id, dtype, device, trust_remote_code, out_dir, short_name):
    model, tok = kbc.load_model(model_id, dtype, device, trust_remote_code)
    infos = kbc.discover_attention(model)
    cfg = model.config
    layer_rows, head_rows = [], []
    has_any_k_bias = False
    for info in infos:
        L = info["layer_idx"]
        kb = kbc.k_bias_vector(
            info, dtype=torch.float32
        )  # [n_kv*head_dim], zeros if none
        has_any_k_bias = has_any_k_bias or info["has_k_bias"]
        st = kbc.bias_stats(kb)
        layer_rows.append(
            dict(
                model=short_name,
                layer=L,
                fused=info["fused"],
                has_q_bias=info["has_q_bias"],
                has_k_bias=info["has_k_bias"],
                has_v_bias=info["has_v_bias"],
                k_bias_numel=kb.numel(),
                n_kv_heads=info["n_kv_heads"],
                head_dim=info["head_dim"],
                **st,
            )
        )
        # per KV head
        kb_h = kb.view(info["n_kv_heads"], info["head_dim"])
        for h in range(info["n_kv_heads"]):
            sh = kbc.bias_stats(kb_h[h])
            head_rows.append(dict(model=short_name, layer=L, kv_head=h, **sh))

    # model-level summary = max/mean across layers of the key stats
    import statistics as S

    def agg(field, fn):
        vals = [r[field] for r in layer_rows]
        return fn(vals) if vals else 0.0

    summary = dict(
        model=short_name,
        model_id=model_id,
        num_layers=len(infos),
        fused=infos[0]["fused"] if infos else None,
        has_k_bias=has_any_k_bias,
        has_q_bias=infos[0]["has_q_bias"] if infos else None,
        has_v_bias=infos[0]["has_v_bias"] if infos else None,
        n_q_heads=getattr(cfg, "num_attention_heads", None),
        n_kv_heads=getattr(cfg, "num_key_value_heads", None),
        head_dim=infos[0]["head_dim"] if infos else None,
        rope_theta=infos[0]["rope_theta"] if infos else None,
        max_abs_k_bias=agg("max_abs", max),
        mean_max_abs_k_bias=agg("max_abs", lambda v: S.mean(v)),
        max_max_over_p99=agg("max_over_p99", max),
        mean_max_over_p99=agg("max_over_p99", lambda v: S.mean(v)),
        max_max_over_rms=agg("max_over_rms", max),
        max_rms_k_bias=agg("rms", max),
        max_frac_gt_448=agg("frac_abs_gt_448", max),
    )

    base = os.path.join(out_dir, short_name)
    lf = [
        "model",
        "layer",
        "fused",
        "has_q_bias",
        "has_k_bias",
        "has_v_bias",
        "k_bias_numel",
        "n_kv_heads",
        "head_dim",
        "max_abs",
        "p999_abs",
        "p99_abs",
        "p95_abs",
        "mean_abs",
        "rms",
        "std",
        "max_over_p99",
        "max_over_rms",
        "frac_abs_gt_448",
    ]
    hf = [
        "model",
        "layer",
        "kv_head",
        "max_abs",
        "p999_abs",
        "p99_abs",
        "p95_abs",
        "mean_abs",
        "rms",
        "std",
        "max_over_p99",
        "max_over_rms",
        "frac_abs_gt_448",
    ]
    kbc.write_csv(os.path.join(base, "bias_stats_layer.csv"), layer_rows, lf)
    kbc.write_csv(os.path.join(base, "bias_stats_head.csv"), head_rows, hf)
    kbc.write_json(os.path.join(base, "metadata.json"), dict(summary=summary))
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--models-file", default=None, help="yaml with models: list")
    ap.add_argument("--seq-len", type=int, default=2048)  # unused here; uniform CLI
    ap.add_argument("--num-prompts", type=int, default=32)  # unused here
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-dir", default="artifacts/k_bias_audit/bias")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--skip-large-models", action="store_true")
    args = ap.parse_args()

    targets = []
    if args.model:
        targets.append(
            dict(
                model_id=args.model, short_name=args.model.split("/")[-1], tier="w7900"
            )
        )
    if args.models_file:
        import yaml

        spec = yaml.safe_load(open(args.models_file))
        for m in spec["models"]:
            if args.skip_large_models and m.get("tier") == "large":
                continue
            targets.append(m)

    summaries = []
    sf = [
        "model",
        "model_id",
        "num_layers",
        "fused",
        "has_k_bias",
        "has_q_bias",
        "has_v_bias",
        "n_q_heads",
        "n_kv_heads",
        "head_dim",
        "rope_theta",
        "max_abs_k_bias",
        "mean_max_abs_k_bias",
        "max_max_over_p99",
        "mean_max_over_p99",
        "max_max_over_rms",
        "max_rms_k_bias",
        "max_frac_gt_448",
    ]
    for t in targets:
        sn = t.get("short_name", t["model_id"].split("/")[-1])
        try:
            s = audit_model(
                t["model_id"],
                args.dtype,
                args.device,
                args.trust_remote_code
                or "deepseek" in t["model_id"].lower()
                or "phi" in t["model_id"].lower(),
                args.output_dir,
                sn,
            )
            summaries.append(s)
            print(
                f"[ok] {sn}: has_k_bias={s['has_k_bias']} max_abs={s['max_abs_k_bias']:.2f} "
                f"max_over_p99={s['max_max_over_p99']:.1f} fused={s['fused']}"
            )
        except Exception as e:
            print(f"[FAIL] {sn}: {type(e).__name__}: {str(e)[:120]}")
    kbc.write_csv(
        os.path.join(args.output_dir, "model_bias_summary.csv"), summaries, sf
    )
    print(f"\n[bias_audit] wrote {len(summaries)} model summaries -> {args.output_dir}")


if __name__ == "__main__":
    main()
