"""Phase 6: attention-score diagnostics. Tests whether Phi is attention-score-scale / numerically
fragile. In the attention hook we recompute pre-softmax scores from (q,k,scaling,mask) and capture
per-layer: score amax/p99/rms, inf/nan counts, softmax entropy, max prob, first-token (sink) mass,
recent-token mass -- for BF16 and under normal-FP8, reporting deltas. Large BF16 score amax or big
FP8 routing/entropy deltas at small K error => score-scale fragility, not the K-bias mechanism.
measurement_level=fake_quant.
"""

import argparse
import csv
import os
import statistics as St

import torch

import _phicommon as pc
from _phicommon import F, kbc, t2


def make_diag(cap, by_mod, orig, quant=None):
    def hook(module, q, k, v, attention_mask, scaling=None, dropout=0.0, **kw):
        info = by_mod.get(id(module))
        if info is not None:
            kk = k
            if quant is not None:
                kk = kbc._quant_lastdims(k, "fp8", 8, "per_tensor", 128, False)
                v = kbc._quant_lastdims(v, "fp8", 8, "per_tensor", 128, False)
            g = module.num_key_value_groups
            ks = kk.repeat_interleave(g, dim=1)
            sc = scaling if scaling is not None else (q.shape[-1] ** -0.5)
            s = torch.matmul(q.float(), ks.float().transpose(2, 3)) * sc  # [B,Hq,T,T]
            Tk = ks.shape[2]
            if attention_mask is not None:
                s = s + attention_mask[:, :, :, :Tk].float()
            probs = torch.softmax(s, dim=-1)
            ent = (
                -(probs.clamp(min=1e-12) * probs.clamp(min=1e-12).log())
                .sum(-1)
                .mean()
                .item()
            )
            cap[info["layer_idx"]] = dict(
                score_amax=s.abs().amax().item(),
                score_p99=s.float()
                .abs()
                .flatten()
                .sort()
                .values[min(s.numel() - 1, int(0.99 * s.numel()))]
                .item(),
                score_rms=s.float().pow(2).mean().sqrt().item(),
                n_inf=int(torch.isinf(s).sum().item()),
                n_nan=int(torch.isnan(s).sum().item()),
                entropy=ent,
                max_prob=probs.amax().item(),
                first_token_mass=probs[..., 0].mean().item(),
                recent_mass=(
                    probs.diagonal(dim1=-2, dim2=-1).mean().item()
                    if probs.shape[-1] == probs.shape[-2]
                    else 0.0
                ),
            )
        return orig(
            module, q, k, v, attention_mask, dropout=dropout, scaling=scaling, **kw
        )

    return hook


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--models-file")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--num-prompts", type=int, default=16)
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
            model, tok = kbc.load_model(mid, args.dtype, args.device, t2.trc_for(mid))
            infos = kbc.discover_attention(model)
            by_mod = {id(i["attn_module"]): i for i in infos}
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

            impl = model.config._attn_implementation
            orig = ALL_ATTENTION_FUNCTIONS[impl]
            ids_list = kbc.calib_prompts(tok, n=args.num_prompts, seq_len=args.seq_len)

            def run(quant):
                acc = {}
                cap = {}
                ALL_ATTENTION_FUNCTIONS[impl] = make_diag(
                    cap, by_mod, orig, quant=quant
                )
                for ids in ids_list:
                    cap.clear()
                    model(torch.tensor(ids).unsqueeze(0).to(args.device))
                    for L, d in cap.items():
                        acc.setdefault(L, []).append(d)
                ALL_ATTENTION_FUNCTIONS[impl] = orig
                return {
                    L: {k: St.mean([r[k] for r in v]) for k in v[0]}
                    for L, v in acc.items()
                }

            bf = run(None)
            fp = run("fp8")
            for L in sorted(bf):
                b, f = bf[L], fp[L]
                rows.append(
                    dict(
                        model=sn,
                        layer=L,
                        bf16_score_amax=b["score_amax"],
                        bf16_score_p99=b["score_p99"],
                        bf16_entropy=b["entropy"],
                        bf16_first_token_mass=b["first_token_mass"],
                        bf16_max_prob=b["max_prob"],
                        n_inf=b["n_inf"],
                        n_nan=b["n_nan"],
                        fp8_entropy_delta=f["entropy"] - b["entropy"],
                        fp8_first_token_mass_delta=f["first_token_mass"]
                        - b["first_token_mass"],
                        fp8_max_prob_delta=f["max_prob"] - b["max_prob"],
                    )
                )
            amax = max(r["bf16_score_amax"] for r in rows if r["model"] == sn)
            ed = St.mean(
                [abs(r["fp8_entropy_delta"]) for r in rows if r["model"] == sn]
            )
            print(
                f"[score] {sn}: max BF16 score_amax={amax:.1f} mean|FP8 entropy delta|={ed:.3f}"
            )
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[FAIL] {sn}: {type(e).__name__}: {str(e)[:140]}")
    if rows:
        with open(
            os.path.join(args.output_dir, "attention_score_diagnostics.csv"),
            "w",
            newline="",
        ) as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            [w.writerow(r) for r in rows]
    print(f"[score] -> {args.output_dir}")


if __name__ == "__main__":
    main()
