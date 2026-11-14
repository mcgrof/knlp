#!/usr/bin/env python3
"""Measure the gradient noise scale of cartridge training at saved
checkpoints, to estimate the critical batch size.

The native recipe takes one optimizer step per 128 packed 2,048-token
sequences, and the trainer runs those sequences one at a time with
gradient accumulation, so the global batch buys no throughput: it only
sets how many steps a run gets from its tokens. Whether 128 is near the
batch the problem can use is a property of the gradients. The simple
noise scale of McCandlish et al., "An Empirical Model of Large-Batch
Training" (arXiv 1812.06162), is

    B_simple = tr(Sigma) / |G|^2

where G is the true gradient and Sigma the covariance of the gradient
of one sequence. Batches well below B_simple are dominated by noise
and gain almost linearly from more data per step; batches well above
it waste data. B_simple is in the same unit as the batch, here packed
sequences.

For each checkpoint this script takes K packed training sequences, the
same K for every checkpoint, computes the gradient of the trainer's own
per-sequence loss (``ce_by_token.mean()`` over the stored top-k
entries, the library's formula without the accumulation divisor) with
respect to the trainable keys and values, and forms the paper's
unbiased estimators from the per-sequence squared norms and the mean
gradient (batch sizes 1 and K):

    |G|^2   ~ (K |g_K|^2 - mean |g_i|^2) / (K - 1)
    tr(S)   ~ (mean |g_i|^2 - |g_K|^2) * K / (K - 1)

The sequences are split into equal groups and a delete-one-group
jackknife gives a standard error. Keys and values are reported apart
and per layer, since values move about four times more than keys. The
frozen sink is not a parameter and has no gradient.

Env:
    MODEL      Qwen/Qwen3-8B (default)
    PARQUET    training conversations (the run's DATA_PARQUET)
    LIMIT      rows to read (default all)
    SEED       packing seed (default 42, the trainer's)
    CARTS      comma list of ``name=path`` checkpoints
    K          sequences per checkpoint (default 256)
    GNS_GROUPS jackknife groups, must divide K (default 16)
    PICK_SEED  seed that picks the K sequences (default 1234)
    OUT_JSON   results (default grad_noise.json)
    DEVICE     cuda (default)

Set CARTRIDGES_GRAD_CKPT=1 and LOAD_BF16=1 on a 24 GiB card, as for
training. Prints one ``GNS,<name>,<B_simple>,<se>,<B_keys>,<B_values>,
<loss>,adam=<B_adam>,adam_k=,adam_v=`` line per checkpoint and
``GNS_DONE`` at the end. B_adam is the Adam-preconditioned scale (see
``adam_scale``): it weights each coordinate by 1/sqrt(E[g^2]), so a few
layers with a large persistent gradient no longer collapse the estimate
the way they do in the raw B_simple.
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/tmp/cas_grad_noise")
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.datasets import TrainDataset, DataSource
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cas_cart_init import load_cart  # noqa: E402

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
PARQUET = os.environ["PARQUET"]
LIMIT = os.environ.get("LIMIT")
SEED = int(os.environ.get("SEED", "42"))
CARTS = [c for c in os.environ.get("CARTS", "").split(",") if c]
K = int(os.environ.get("K", "256"))
GROUPS = int(os.environ.get("GNS_GROUPS", "16"))
PICK_SEED = int(os.environ.get("PICK_SEED", "1234"))
OUT_JSON = os.environ.get("OUT_JSON", "grad_noise.json")
DEVICE = os.environ.get("DEVICE", "cuda")
SEQ_LEN = 2048
assert K % GROUPS == 0, "GNS_GROUPS must divide K"


def parse_carts(items):
    out = []
    for it in items:
        if "=" in it:
            name, path = it.split("=", 1)
        else:
            name, path = Path(it).stem, it
        out.append((name, path))
    return out


def estimate(sum_g_sq, sum_sq, n):
    """Paper estimators from |sum of n gradients|^2 and the sum of the n
    per-sequence squared norms. Returns (|G|^2, tr(Sigma), B_simple)."""
    small = sum_sq / n
    big = sum_g_sq / (n * n)
    g2 = (n * big - small) / (n - 1)
    s = (small - big) * n / (n - 1)
    return g2, s, (s / g2 if g2 > 0 else float("inf"))


def noise_scale(gsum, sq, groups):
    """gsum: [GROUPS, P] per-group gradient sums (fp32, CPU); sq: [K]
    per-sequence squared norms over the same P coordinates. Returns the
    full-sample estimate and a delete-one-group jackknife SE of B."""
    total = gsum.sum(0)
    g2, s, b = estimate(float(total.double().square().sum()), float(sq.sum()), K)
    per = K // groups
    sq_g = sq.view(groups, per).sum(1)
    reps = []
    for j in range(groups):
        t = total - gsum[j]
        reps.append(
            estimate(
                float(t.double().square().sum()),
                float(sq.sum() - sq_g[j]),
                K - per,
            )[2]
        )
    reps = np.array(reps)
    finite = np.isfinite(reps)
    se = (
        float(np.sqrt((groups - 1) / groups * ((reps - reps.mean()) ** 2).sum()))
        if finite.all()
        else float("inf")
    )
    return {"G2": g2, "trSigma": s, "B_simple": b, "B_se": se}


def adam_scale(total, ss, n, eps=1e-12):
    """Adam-preconditioned noise scale over a set of coordinates.

    B_simple = tr(Sigma)/|G|^2 is derived for plain gradient descent, so a
    layer with a very large mean gradient dominates it even though Adam caps
    that layer's step near the learning rate. The preconditioned noise scale
    of McCandlish et al. (their per-coordinate generalization) is
    tr(P Sigma)/(G^T P G); with Adam the preconditioner is
    P = diag(1/sqrt(v)), v the second moment of the per-sequence gradient.
    We estimate v_i = E[g_i^2] = ss_i/n and form the paper's unbiased G_i^2
    and Sigma_ii per coordinate, then weight both by 1/sqrt(v_i). This is the
    batch the problem wants as Adam actually sees it. No jackknife SE: the
    per-group coordinate sums are not kept.

    total: [P] sum of the per-sequence gradients over n sequences.
    ss:    [P] sum of the per-sequence squared gradients over n sequences.
    """
    total = total.double()
    ss = ss.double()
    mean = total / n
    second = ss / n  # E[g_i^2] ~ Adam's v_i
    msq = mean.square()
    g2 = (n * msq - second) / (n - 1)  # unbiased G_i^2
    var = (second - msq) * n / (n - 1)  # unbiased Sigma_ii
    w = 1.0 / (second.sqrt() + eps)  # Adam preconditioner 1/sqrt(v_i)
    den = float((w * g2).sum())
    num = float((w * var).sum())
    return num / den if den > 0 else float("inf")


def measure(model, dataset, picks, cache, n_layers):
    params_k = list(cache.trainable_keys)
    params_v = list(cache.trainable_values)
    per = K // GROUPS
    # per-group gradient sums, kept per kind and layer: [GROUPS, L, P_layer]
    shape_k = [p.numel() for p in params_k]
    shape_v = [p.numel() for p in params_v]
    gk = torch.zeros(GROUPS, sum(shape_k), dtype=torch.float32)
    gv = torch.zeros(GROUPS, sum(shape_v), dtype=torch.float32)
    sq_k = torch.zeros(K, n_layers, dtype=torch.float64)
    sq_v = torch.zeros(K, n_layers, dtype=torch.float64)
    acc_k = torch.zeros(sum(shape_k), dtype=torch.float32, device=DEVICE)
    acc_v = torch.zeros(sum(shape_v), dtype=torch.float32, device=DEVICE)
    # Per-coordinate sum of squared gradients over all K sequences, for the
    # Adam-preconditioned scale. These never reset across groups.
    ssk = torch.zeros(sum(shape_k), dtype=torch.float32, device=DEVICE)
    ssv = torch.zeros(sum(shape_v), dtype=torch.float32, device=DEVICE)
    losses, secs = [], []
    for i, b in enumerate(picks):
        t0 = time.time()
        batch = dataset[int(b)]
        cache.clear()
        for p in params_k + params_v:
            p.grad = None
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(
                input_ids=batch.input_ids.to(DEVICE),
                seq_ids=batch.element_ids.to(DEVICE),
                position_ids=batch.position_ids.to(DEVICE),
                use_cache=True,
                past_key_values=cache,
            )
            pred = F.log_softmax(out.logits, dim=-1)[
                0,
                batch.topk_token_idxs.to(DEVICE) - 1,
                batch.topk_token_ids.to(DEVICE),
            ]
            # the trainer's per-micro-batch loss, without the 1/accum factor
            loss = (-batch.topk_logprobs.to(DEVICE).exp() * pred).mean()
        loss.backward()
        del out, pred
        fk = torch.cat([p.grad.float().reshape(-1) for p in params_k])
        fv = torch.cat([p.grad.float().reshape(-1) for p in params_v])
        sq_k[i] = (
            torch.stack([p.grad.float().square().sum() for p in params_k])
            .double()
            .cpu()
        )
        sq_v[i] = (
            torch.stack([p.grad.float().square().sum() for p in params_v])
            .double()
            .cpu()
        )
        acc_k += fk
        acc_v += fv
        ssk += fk.square()
        ssv += fv.square()
        if (i + 1) % per == 0:
            g = (i + 1) // per - 1
            gk[g] = acc_k.cpu()
            gv[g] = acc_v.cpu()
            acc_k.zero_()
            acc_v.zero_()
        losses.append(float(loss))
        torch.cuda.synchronize()
        secs.append(time.time() - t0)
        if i == 0 or (i + 1) % 32 == 0:
            print(
                f"[gns] {i + 1}/{K} loss={losses[-1]:.5f} "
                f"{secs[-1]:.2f}s/seq peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GiB",
                flush=True,
            )
    ssk = ssk.cpu()
    ssv = ssv.cpu()
    tot_k = gk.sum(0)
    tot_v = gv.sum(0)
    res = {
        "loss_mean": float(np.mean(losses)),
        "sec_per_seq_median": float(np.median(secs[1:] or secs)),
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "all": noise_scale(torch.cat([gk, gv], 1), (sq_k.sum(1) + sq_v.sum(1)), GROUPS),
        "keys": noise_scale(gk, sq_k.sum(1), GROUPS),
        "values": noise_scale(gv, sq_v.sum(1), GROUPS),
        "per_layer": {"keys": [], "values": []},
    }
    res["all"]["B_adam"] = adam_scale(torch.cat([tot_k, tot_v]), torch.cat([ssk, ssv]), K)
    res["keys"]["B_adam"] = adam_scale(tot_k, ssk, K)
    res["values"]["B_adam"] = adam_scale(tot_v, ssv, K)
    for kind, gs, sq, sizes, ss in (
        ("keys", gk, sq_k, shape_k, ssk),
        ("values", gv, sq_v, shape_v, ssv),
    ):
        off = 0
        for layer, n in enumerate(sizes):
            r = noise_scale(gs[:, off : off + n], sq[:, layer], GROUPS)
            ba = adam_scale(gs[:, off : off + n].sum(0), ss[off : off + n], K)
            res["per_layer"][kind].append(
                {"B_simple": r["B_simple"], "G2": r["G2"], "B_adam": ba}
            )
            off += n
    return res


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    src = DataSource(path=PARQUET, type="local", limit=int(LIMIT) if LIMIT else None)
    dataset = TrainDataset.Config(
        data_sources=[src],
        top_k_logits=20,
        packed_seq_length=SEQ_LEN,
        packing_mode="truncate",
    ).instantiate(tokenizer=tok, seed=SEED)
    n = len(dataset)
    assert K <= n, f"K={K} > {n} packed sequences"
    picks = np.random.default_rng(PICK_SEED).choice(n, size=K, replace=False)
    print(
        f"[gns] {PARQUET}: {len(dataset.elements)} rows -> {n} packed "
        f"sequences; K={K} in {GROUPS} groups",
        flush=True,
    )
    kw = {"torch_dtype": torch.bfloat16} if os.environ.get("LOAD_BF16") == "1" else {}
    model = (
        FlexQwen3ForCausalLM.from_pretrained(MODEL, **kw).to(DEVICE).to(torch.bfloat16)
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    attn = AttnConfig(
        n_layers=model.config.num_hidden_layers,
        n_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
    )
    results = {
        "model": MODEL,
        "parquet": os.path.abspath(PARQUET),
        "seed": SEED,
        "n_packed": n,
        "K": K,
        "groups": GROUPS,
        "pick_seed": PICK_SEED,
        "picks": [int(x) for x in picks],
        "unit": "packed 2048-token sequences (the trainer's micro-batch)",
        "carts": {},
    }
    for name, path in parse_carts(CARTS):
        t1 = time.time()
        keys, values, nfrozen = load_cart(path, device=DEVICE)
        cache = TrainableCache(
            config=attn, init_keys=keys, init_values=values, num_frozen_tokens=nfrozen
        ).to(DEVICE)
        torch.cuda.reset_peak_memory_stats()
        r = measure(model, dataset, picks, cache, attn.n_layers)
        r.update(path=os.path.abspath(path), frozen=nfrozen, seconds=time.time() - t1)
        results["carts"][name] = r
        a, k_, v_ = r["all"], r["keys"], r["values"]
        print(
            f"GNS,{name},{a['B_simple']:.1f},{a['B_se']:.1f},"
            f"{k_['B_simple']:.1f},{v_['B_simple']:.1f},{r['loss_mean']:.6f},"
            f"adam={a['B_adam']:.1f},adam_k={k_['B_adam']:.1f},adam_v={v_['B_adam']:.1f}",
            flush=True,
        )
        Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_JSON, "w") as f:
            json.dump(results, f, indent=1)
        del cache, keys, values
        torch.cuda.empty_cache()
    print(f"GNS_DONE {OUT_JSON} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
