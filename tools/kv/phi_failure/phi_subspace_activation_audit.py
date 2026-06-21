"""Phase 2: subspace activation audit. Splits post-RoPE K into the rotary subspace (first
rotary_dim head channels) and the pass-through subspace (the rest), per layer, and compares their
distributions. If Phi's per-tensor FP8 scale is dominated by one subspace (big pass/rotary ratio),
that is the partial-RoPE mixed-distribution failure mode. Controls (Qwen/Mistral) are full-RoPE
(pass-through dim 0) and serve as the no-mixture reference. measurement_level=activation_audit.
"""

import argparse
import csv
import os
import statistics as St

import torch

import _phicommon as pc
from _phicommon import kbc, t2

FP8_MAX = 448.0


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--models-file")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--num-prompts", type=int, default=64)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    specs = t2.load_models_spec(args.models_file, args.models, args.only)
    os.makedirs(args.output_dir, exist_ok=True)
    layer_rows = []
    for m in specs:
        mid, sn = m["model_id"], m["short_name"]
        try:
            model, tok = kbc.load_model(mid, args.dtype, args.device, t2.trc_for(mid))
            infos = kbc.discover_attention(model)
            rotary_dim, head_dim = pc.rotary_split(model)
            info_by_mod = {id(i["attn_module"]): i for i in infos}
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

            impl = model.config._attn_implementation
            orig = ALL_ATTENTION_FUNCTIONS[impl]
            cap = {}

            def hook(module, q, k, v, attention_mask, scaling=None, dropout=0.0, **kw):
                info = info_by_mod.get(id(module))
                if info is not None:
                    cap[info["layer_idx"]] = (
                        k.detach().float()
                    )  # post-RoPE K [B,nkv,T,hd]
                return orig(
                    module,
                    q,
                    k,
                    v,
                    attention_mask,
                    dropout=dropout,
                    scaling=scaling,
                    **kw,
                )

            ALL_ATTENTION_FUNCTIONS[impl] = hook
            acc = {}
            ids_list = kbc.calib_prompts(tok, n=args.num_prompts, seq_len=args.seq_len)
            for ids in ids_list:
                cap.clear()
                model(torch.tensor(ids).unsqueeze(0).to(args.device))
                for L, k in cap.items():
                    k = k[0]  # [nkv, T, hd]
                    rot = k[..., :rotary_dim]
                    pas = k[..., rotary_dim:]
                    sr, sp = pc.stats(rot), (pc.stats(pas) if pas.numel() else None)
                    full = pc.stats(k)
                    d = acc.setdefault(L, [])
                    rec = dict(
                        rot_amax=sr["amax"],
                        rot_p99=sr["p99"],
                        rot_rms=sr["rms"],
                        rot_fp8_scale_loss=sr["amax"] / max(sr["p99"], 1e-9),
                        full_amax=full["amax"],
                        full_p99=full["p99"],
                    )
                    if sp:
                        rec.update(
                            pas_amax=sp["amax"],
                            pas_p99=sp["p99"],
                            pas_rms=sp["rms"],
                            pas_over_rot_amax=sp["amax"] / max(sr["amax"], 1e-9),
                            pas_over_rot_p99=sp["p99"] / max(sr["p99"], 1e-9),
                            pas_over_rot_rms=sp["rms"] / max(sr["rms"], 1e-9),
                            pas_fp8_scale_loss=sp["amax"] / max(sp["p99"], 1e-9),
                        )
                    d.append(rec)
            ALL_ATTENTION_FUNCTIONS[impl] = orig
            for L in sorted(acc):
                keys = acc[L][0].keys()
                row = dict(
                    model=sn,
                    layer=L,
                    rotary_dim=rotary_dim,
                    pass_through_dim=head_dim - rotary_dim,
                )
                for kk in keys:
                    row[kk] = St.mean([r[kk] for r in acc[L] if kk in r])
                layer_rows.append(row)
            por = St.mean(
                [
                    r.get("pas_over_rot_amax", 1.0)
                    for r in layer_rows
                    if r["model"] == sn
                ]
            )
            print(
                f"[subspace] {sn}: rotary={rotary_dim}/{head_dim} "
                f"mean pass/rot amax={por:.2f} (>>1 => mixed-subspace FP8 hazard)"
            )
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[FAIL] {sn}: {type(e).__name__}: {str(e)[:140]}")
    if layer_rows:
        allk = sorted({k for r in layer_rows for k in r})
        with open(
            os.path.join(args.output_dir, "subspace_activation_layer.csv"),
            "w",
            newline="",
        ) as f:
            w = csv.DictWriter(f, fieldnames=allk)
            w.writeheader()
            for r in layer_rows:
                w.writerow({k: r.get(k, "") for k in allk})
    print(f"[subspace] -> {args.output_dir}")


if __name__ == "__main__":
    main()
