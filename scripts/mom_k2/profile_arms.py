"""Bounded profile of the baseline and the candidate. Separate from timing.

Every event is categorised, so a category total is a total and not the sum of
whatever rows fit on a screen. Operator attribution, summed kernel work and the
union of kernel intervals are reported separately and never added together; the
remainder inside the wall interval is named unattributed rather than called
launch overhead.
"""
import argparse, json, os, sys, time
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("MATCHED_MICRO_SOURCE", os.path.dirname(HERE)))
LR = 0.0024

CATS = [
    ("router, dispatch, sort and packing",
     ("sort", "argsort", "topk", "scatter", "gather", "index", "cumsum", "nonzero",
      "pad", "unpad", "softmax_warp", "Softmax")),
    ("projections and matrix multiply", ("Cijk", "gemm", "GEMM", "bmm", "addmm")),
    ("convolution", ("conv", "causal_conv")),
    ("recurrent kernels", ("chunk_gated_delta", "delta_rule", "ChunkGatedDelta")),
    ("normalisation and reduction", ("norm", "Norm", "reduce", "Reduce", "rms", "layer_norm")),
    ("elementwise and copies", ("elementwise", "copy", "Copy", "cat", "fill", "zero")),
    ("optimizer", ("adam", "Adam", "foreach", "lerp", "addcmul", "addcdiv")),
]


def cat_of(n):
    for c, keys in CATS:
        if any(k in n for k in keys):
            return c
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--data-dir",
        default=os.environ.get("MATCHED_MICRO_DATA"),
        required=not bool(os.environ.get("MATCHED_MICRO_DATA")),
        help="packed token directory from matched_micro_data.prepare",
    )
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--arms", default="G0,K2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numpy as np, torch
    import torch.nn.functional as F
    import arms as A, matched_micro_data as mmd
    from torch.profiler import profile, ProfilerActivity
    from torch.autograd import DeviceType
    ctx, mb = 2048, 8
    rep = dict(kind="k2_bounded_profile", steps=a.steps,
               categories=[c for c, _ in CATS] + ["other"], arms={})

    for aid in a.arms.split(","):
        model, _ = A.build(aid, 41011, a.device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                                weight_decay=0.1)
        st = mmd.Stream(a.data_dir, "train", ctx, mb)
        aux = 0.01 if A.ARMS[aid]["family"] == "mom" else 0.0

        def one():
            opt.zero_grad(set_to_none=True)
            x, _ = st.next_batch(a.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(x[:, :-1]).float()
                L = F.cross_entropy(lg.reshape(-1, lg.size(-1)), x[:, 1:].reshape(-1))
                if aux:
                    L = L + aux * model.load_balance_loss()
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        for _ in range(6):
            one()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
            torch.cuda.synchronize(); w0 = time.perf_counter()
            for _ in range(a.steps):
                one()
            torch.cuda.synchronize()
            wall = (time.perf_counter() - w0) * 1e3

        kern, ops = [], 0.0
        for e in pr.events():
            if e.self_device_time_total <= 0:
                continue
            if e.device_type == DeviceType.CUDA:
                kern.append((e.key, e.self_device_time_total / 1e3,
                             e.time_range.start, e.time_range.end))
            elif e.device_type == DeviceType.CPU:
                ops += e.self_device_time_total / 1e3
        tot = sum(v for _, v, _, _ in kern)
        cat = defaultdict(float); members = defaultdict(int)
        for k, ms, _, _ in kern:
            cat[cat_of(k)] += ms; members[cat_of(k)] += 1
        iv = sorted((s, e) for _, _, s, e in kern)
        union, cs, ce = 0.0, None, None
        for s, e in iv:
            if cs is None:
                cs, ce = s, e
            elif s <= ce:
                ce = max(ce, e)
            else:
                union += ce - cs; cs, ce = s, e
        if cs is not None:
            union += ce - cs
        union /= 1e3
        byname = defaultdict(float)
        for k, ms, _, _ in kern:
            byname[k] += ms
        rep["arms"][aid] = dict(
            wall_ms=wall, kernel_sum_ms=tot, operator_attributed_ms=ops,
            union_of_kernel_intervals_ms=union,
            unattributed_ms=wall - union,
            unattributed_fraction=(wall - union) / wall,
            per_step_ms=wall / a.steps,
            distinct_kernels=len(byname),
            coverage=sum(cat.values()) / tot,
            categories={c: dict(ms=cat[c], per_step_ms=cat[c] / a.steps,
                                share=cat[c] / tot, kernels=members[c])
                        for c in sorted(cat)},
            top_kernels=[dict(name=k[:66], ms=v, category=cat_of(k))
                         for k, v in sorted(byname.items(), key=lambda x: -x[1])[:12]],
            note=("kernel sum and operator attribution are two views of the same "
                  "device work and are never added; the union is device occupancy; "
                  "the remainder inside the wall is unattributed, not named"))
        print("%-3s wall %.0f ms (%.1f/step) | kernels %.0f | union %.0f | "
              "unattributed %.0f (%.1f%%) | coverage %.4f over %d kernels"
              % (aid, wall, wall / a.steps, tot, union, wall - union,
                 100 * (wall - union) / wall, sum(cat.values()) / tot, len(byname)))
        for c in sorted(cat, key=lambda c: -cat[c]):
            print("     %-38s %8.2f ms  %5.1f%%" % (c, cat[c], 100 * cat[c] / tot))
        del model, opt, st
        torch.cuda.empty_cache()

    ids = a.arms.split(",")
    if len(ids) == 2:
        g, k = rep["arms"][ids[0]], rep["arms"][ids[1]]
        rep["difference"] = {c: dict(baseline_ms=g["categories"].get(c, {}).get("ms", 0.0),
                                     candidate_ms=k["categories"].get(c, {}).get("ms", 0.0),
                                     excess_ms=(k["categories"].get(c, {}).get("ms", 0.0)
                                                - g["categories"].get(c, {}).get("ms", 0.0)))
                             for c in set(g["categories"]) | set(k["categories"])}
        print("\nper category, candidate minus baseline:")
        for c, v in sorted(rep["difference"].items(), key=lambda kv: -kv[1]["excess_ms"]):
            print("  %-38s %+8.2f ms" % (c, v["excess_ms"]))
    json.dump(rep, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
