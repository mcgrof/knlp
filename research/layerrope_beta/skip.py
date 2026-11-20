# SPDX-License-Identifier: MIT
"""Measure how much each layer of a trained story model can be skipped.

A layer that can be skipped need not be read from storage. This script
asks, for trained 32-layer models of each residual method, how much the
loss rises when layers are skipped, and whether a layer's measured size
predicts that rise. Skipping a site (the attention or MLP branch of a
layer) means its branch output is replaced by zeros: the residual
stream, and for mHC the stream mixing, still run, but the site's
weights are never used.

Per model it records:

    base        the validation loss with nothing skipped, through this
                script's forward, and the difference from the
                trainer's own evaluation (a check of the forward);
    sizes       per layer and site, the LayerRoPE output gain (mean
                absolute value; LayerRoPE arms only) and the branch's
                root-mean-square output over the residual stream's, on
                the first batches of the curve split;
    single      the loss rise when one layer (both sites) or one site
                is skipped;
    late        the loss rise when the last k layers are skipped;
    ranked      the loss rise when the k layers with the smallest
                single-layer rise are skipped together, with the order
                taken from the curve split and the rise measured on the
                final split, so the order is not fitted to the split it
                is scored on.

Losses are the mean cross-entropy, in nats, over adjacent 256-token
windows, as in training. "Bytes" are the parameter bytes of the skipped
layers over all layers' parameter bytes (embeddings and head excluded).

Usage:
    python -m research.layerrope_beta.skip --data DATA_DIR \\
        --checkpoints CKPT_ROOT --out OUT.json JOB_ID [JOB_ID ...]
        [--limit-windows N]
"""

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from research.layerrope_beta.model import Model
from research.tinystories_depth.model import mhc
from research.tinystories_depth.kv_cache import evaluate

SITES = ("attn", "mlp")
WINDOW = 256


def parse_job(job):
    """Arm, seed, alpha_out and beta from a final job id."""
    m = re.fullmatch(
        r"final-(R|M|L|ML)(?:-a-(minus)?(\d+)p(\d+)-beta-(minus)?(\d+)p(\d+))?"
        r"-lr-\d+p\d+-seed(\d+)",
        job,
    )
    if m is None:
        raise ValueError(job)
    arm, seed = m.group(1), int(m.group(8))
    a = beta = 0.0
    if m.group(3) is not None:
        a = float(f"{m.group(3)}.{m.group(4)}") * (-1 if m.group(2) else 1)
        beta = float(f"{m.group(6)}.{m.group(7)}") * (-1 if m.group(5) else 1)
    return arm, seed, a, beta


def forward(model, ids, skip=frozenset(), stats=None):
    """The model's forward with skipped sites; stats collects branch sizes."""
    h = model.lm.model.embed_tokens(ids).bfloat16()
    positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
    if model.routes is not None:
        h = mhc.expand_to_mhc_ref(h, 4)
    for layer, block in enumerate(model.lm.model.layers):
        for index, site in enumerate(SITES):
            if model.routes is not None:
                pre, post, carry, _ = model.routes[2 * layer + index](h)
            if (layer, site) in skip:
                branch = None
            else:
                if model.routes is None:
                    inp = h
                else:
                    with torch.autocast(device_type=h.device.type, enabled=False):
                        inp = mhc.mhc_pre_apply_mix_ref(h, pre)
                if site == "attn":
                    inp = block.input_layernorm(inp, block._gain("in_attn"))
                    branch = block.self_attn(
                        inp, position_ids=positions, use_cache=False
                    )[0]
                else:
                    inp = block.post_attention_layernorm(inp, block._gain("in_mlp"))
                    branch = block.mlp(inp)
                branch = block._block_output(branch, site)
                if stats is not None:
                    rms = lambda x: x.float().pow(2).mean().sqrt().item()  # noqa
                    stats.setdefault((layer, site), []).append(rms(branch) / rms(h))
            if model.routes is None:
                if branch is not None:
                    h = h + branch
            else:
                if branch is None:
                    branch = torch.zeros(
                        h.shape[:-2] + h.shape[-1:], dtype=h.dtype, device=h.device
                    )
                with torch.autocast(device_type=h.device.type, enabled=False):
                    h = mhc.mhc_post_ref(branch, h, post, carry)
    if model.routes is not None:
        h = h.float().mean(-2).bfloat16()
    return model.lm.lm_head(model.lm.model.norm(h))


@torch.no_grad()
def loss(model, windows, skip=frozenset(), batch_size=64, stats_batches=0):
    """Mean cross-entropy over the windows; optional branch-size stats."""
    total, n, stats = 0.0, 0, ({} if stats_batches else None)
    for b, start in enumerate(range(0, len(windows), batch_size)):
        x = windows[start : start + batch_size]
        st = stats if b < stats_batches else None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = forward(model, x[:, :-1], skip, st)
        ce = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            x[:, 1:].reshape(-1),
            reduction="none",
        )
        total += ce.double().reshape(len(x), -1).mean(-1).sum().item()
        n += len(x)
    value = total / n
    if not math.isfinite(value):
        raise FloatingPointError("Nonfinite loss")
    if stats_batches:
        return value, {k: float(np.mean(v)) for k, v in stats.items()}
    return value


def windows_of(tokens, limit=None):
    count = (len(tokens) - 1) // WINDOW
    if limit:
        count = min(count, limit)
    idx = np.arange(count)[:, None] * WINDOW + np.arange(WINDOW + 1)[None, :]
    return torch.as_tensor(np.array(tokens[idx], dtype=np.int64), device="cuda")


def site_params(model):
    out = {}
    for layer, block in enumerate(model.lm.model.layers):
        out[(layer, "attn")] = sum(p.numel() for p in block.self_attn.parameters())
        out[(layer, "mlp")] = sum(p.numel() for p in block.mlp.parameters())
    return out


def run_model(job, data, checkpoints, limit):
    arm, seed, a, beta = parse_job(job)
    model = Model(arm, 32, seed, beta=beta, alpha_out=a).cuda()
    state = torch.load(
        Path(checkpoints) / "language" / job / "step2000.pt",
        map_location="cuda",
        weights_only=False,
    )
    model.load_state_dict(state["model"])
    model.eval()
    depth = len(model.lm.model.layers)
    curve = windows_of(np.load(data / "curve.npy", mmap_mode="r"), limit)
    final = windows_of(np.load(data / "final.npy", mmap_mode="r"), limit)
    began = time.monotonic()

    base_c, sizes = loss(model, curve, stats_batches=2)
    base_f = loss(model, final)
    ref = evaluate(
        model,
        np.load(data / "final.npy", mmap_mode="r")[: len(final) * WINDOW + 1],
        dict(name="clean", k_bits=16, v_bits=16, method="none"),
        "cuda",
    )["ce"]
    gains = {}
    if model.lm.model.layerrope is not None:
        for layer, block in enumerate(model.lm.model.layers):
            for site in SITES:
                g = block._gain("out_" + site)
                gains[f"{layer}:{site}"] = g.float().abs().mean().item()

    params = site_params(model)
    all_bytes = sum(params.values())

    def layer_skip(layers):
        return frozenset((l, s) for l in layers for s in SITES)

    def frac(skip):
        return sum(params[k] for k in skip) / all_bytes

    single = []
    for l in range(depth):
        row = {"layer": l}
        for name, skip in (
            ("both", layer_skip([l])),
            ("attn", frozenset([(l, "attn")])),
            ("mlp", frozenset([(l, "mlp")])),
        ):
            row[name + "_curve"] = loss(model, curve, skip) - base_c
            row[name + "_final"] = loss(model, final, skip) - base_f
        single.append(row)
    late = []
    for k in range(1, depth // 2 + 1):
        skip = layer_skip(range(depth - k, depth))
        late.append(
            {"k": k, "rise": loss(model, final, skip) - base_f, "bytes": frac(skip)}
        )
    order = sorted(range(depth), key=lambda l: single[l]["both_curve"])
    ranked = []
    for k in range(1, depth):
        skip = layer_skip(order[:k])
        r = loss(model, final, skip) - base_f
        ranked.append({"k": k, "rise": r, "bytes": frac(skip)})
        if r > 1.0:
            break

    def within(rows, bound):
        ok = [r for r in rows if r["rise"] <= bound]
        return max((r["bytes"] for r in ok), default=0.0)

    out = {
        "job": job,
        "arm": arm,
        "seed": seed,
        "alpha_out": a,
        "beta_init": beta,
        "windows": {"curve": len(curve), "final": len(final)},
        "base_curve": base_c,
        "base_final": base_f,
        "trainer_eval_final": ref,
        "forward_check": base_f - ref,
        "gain": gains,
        "branch_rms_ratio": {f"{k[0]}:{k[1]}": v for k, v in sizes.items()},
        "single": single,
        "late": late,
        "ranked_order": order,
        "ranked": ranked,
        "bytes_within": {
            str(b): {"late": within(late, b), "ranked": within(ranked, b)}
            for b in (0.02, 0.05, 0.1)
        },
        "seconds": time.monotonic() - began,
    }
    del model, state
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--checkpoints", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit-windows", type=int)
    ap.add_argument("jobs", nargs="+")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    res = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "args": {k: str(v) for k, v in vars(args).items()},
        "models": [],
    }
    for job in args.jobs:
        r = run_model(job, args.data, args.checkpoints, args.limit_windows)
        res["models"].append(r)
        print(
            f"{job}: base {r['base_final']:.4f} (check {r['forward_check']:+.2e}), "
            f"bytes within 0.1 nat: late {r['bytes_within']['0.1']['late']:.3f} "
            f"ranked {r['bytes_within']['0.1']['ranked']:.3f}, "
            f"{r['seconds']:.0f} s",
            flush=True,
        )
        args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
