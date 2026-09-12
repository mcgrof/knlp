#!/usr/bin/env python3
"""Evaluation-time path ablation for trained Mixture-of-Memories stacks.

Re-scores a checkpoint from the matched micro harness on the held-out
validation slice three ways: intact, with the routed memories zeroed
(the shared memory answers alone), and with the shared memory switched
off (the routed memories answer alone). The recall sweep showed the
paper-configuration cell keeps its recall in the shared memory; this
asks whether the language-model gain lives there too. No training.

    python3 scripts/matched_micro_path_ablation.py --data-dir ... ckpt.pt [ckpt.pt ...]
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matched_micro_train import CONTRACT, StackLM, TokenStream, ce_loss  # noqa: E402


@torch.no_grad()
def val_loss(model, batches):
    losses = [ce_loss(model(idx), idx).item() for idx in batches]
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--data-dir", default="matched-micro-data")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="path_ablation.json")
    args = ap.parse_args()
    device = torch.device(args.device)
    stream = TokenStream(args.data_dir, CONTRACT["seq_len"], args.batch)
    batches = stream.val_batches(device, max_batches=args.val_batches)

    import fla.layers.mom as mom_mod

    orig_reconstruct = mom_mod.reconstruct
    rows = []
    for path in args.checkpoints:
        ck = torch.load(path, map_location="cpu")
        cfg = ck["config"]
        if cfg.get("kind") != "stack":
            print(f"skip {path}: not a stack arm")
            continue
        model = StackLM(cfg, CONTRACT["vocab_size"])
        model.load_state_dict(ck["model"])
        model.to(device).eval()
        blocks = [b for b in model.blocks if b.mixer_kind == "M"]
        row = dict(checkpoint=path, arm=ck["arm"], intact=val_loss(model, batches))
        if blocks and blocks[0].mixer.shared_mem:
            mom_mod.reconstruct = lambda *a, **k: torch.zeros_like(
                orig_reconstruct(*a, **k)
            )
            try:
                row["routed_off"] = val_loss(model, batches)
            finally:
                mom_mod.reconstruct = orig_reconstruct
            for b in blocks:
                b.mixer.shared_mem = False
            row["shared_off"] = val_loss(model, batches)
            for b in blocks:
                b.mixer.shared_mem = True
        rows.append(row)
        print(
            f"{ck['arm']:8s} {os.path.basename(os.path.dirname(path)) or path}: "
            + "  ".join(f"{k} {v:.4f}" for k, v in row.items() if isinstance(v, float)),
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    with open(args.out, "w") as f:
        json.dump(
            dict(contract=dict(CONTRACT), val_batches=args.val_batches, rows=rows),
            f,
            indent=1,
        )
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
