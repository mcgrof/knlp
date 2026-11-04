#!/usr/bin/env python3
"""Freeze one fixed EMO expert pool from a few calibration texts (GPU).

Reimplements EMO's greedy layer-by-layer selection
(src/hf_training/greedy_prune_layerwise.py in allenai/EMO): for each layer
in order, with all earlier layers already restricted, average the router's
softmax probabilities over every calibration token, keep the highest
``routed`` routed experts plus the shared expert, and restrict the layer.
Routed and shared experts use separate softmaxes, as in the model's forward
pass. Note that EMO's ``keep_k`` counts the shared expert; this script
takes the routed count explicitly, so ``--routed 32`` keeps 33 experts.

Usage:
    calibrate_pool.py MODEL_DIR CALIBRATION.json OUT_POOL.json --routed 32
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_model, prune_layer_inplace, runtime_identity, write_json  # noqa


class EarlyExit(Exception):
    pass


@torch.no_grad()
def layer_avg_probs(model, batches, li):
    layer = model.model.layers[li]
    moe = layer.mlp
    n, ns = moe.num_experts, moe.num_shared_experts
    tot = torch.zeros(n, dtype=torch.float64)
    ntok = 0
    cap = []

    def hook(mod, inp, out):
        cap.append(out[1].detach().float())

    def stop(mod, inp):
        raise EarlyExit()

    h1 = moe.register_forward_hook(hook)
    h2 = None
    if li + 1 < len(model.model.layers):
        h2 = model.model.layers[li + 1].register_forward_pre_hook(stop)
    try:
        for ids in batches:
            cap.clear()
            try:
                model(input_ids=ids)
            except EarlyExit:
                pass
            assert len(cap) == 1
            lg = cap[0]
            tot[: n - ns] += F.softmax(lg[:, : n - ns], dim=-1).sum(0).double().cpu()
            tot[n - ns :] += F.softmax(lg[:, n - ns :], dim=-1).sum(0).double().cpu()
            ntok += lg.shape[0]
    finally:
        h1.remove()
        if h2 is not None:
            h2.remove()
    return tot / ntok, ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("calibration")
    ap.add_argument("out")
    ap.add_argument("--routed", type=int, default=32)
    args = ap.parse_args()

    model, tok = load_model(args.model_dir)
    with open(args.calibration) as f:
        cal = json.load(f)
    batches = [
        torch.tensor([tok(t, add_special_tokens=False)["input_ids"]], device="cuda")
        for t in cal["texts"]
    ]
    t0 = time.time()
    layers = {}
    ns = model.model.layers[0].mlp.num_shared_experts
    for li, layer in enumerate(model.model.layers):
        moe = layer.mlp
        n = moe.num_experts
        probs, ntok = layer_avg_probs(model, batches, li)
        std = probs[: n - ns]
        routed = sorted(torch.topk(std, args.routed).indices.tolist())
        shared = list(range(n - ns, n))
        keep = routed + shared
        # Probability mass of the kept routed experts under the full router.
        mass = float(std[routed].sum())
        layers[str(li)] = {
            "routed": routed,
            "shared": shared,
            "routed_prob_mass": mass,
            "avg_probs": [round(float(x), 8) for x in probs.tolist()],
            "calibration_tokens": ntok,
        }
        prune_layer_inplace(moe, keep, ns)
        print(f"layer {li}: kept routed mass {mass:.3f}", flush=True)

    pool = {
        "method": "EMO greedy layer-wise average router probability",
        "routed_per_layer": args.routed,
        "num_shared_experts": ns,
        "total_per_layer": args.routed + ns,
        "note": f"{args.routed + ns}-total pool ({args.routed} routed + {ns} shared); EMO keep_k counts the shared expert",
        "calibration_file": os.path.abspath(args.calibration),
        "calibration_count": len(cal["texts"]),
        "model_dir": os.path.abspath(args.model_dir),
        "runtime": runtime_identity(),
        "seconds": round(time.time() - t0, 2),
        "layers": layers,
    }
    write_json(args.out, pool)
    print("saved", args.out)


if __name__ == "__main__":
    main()
