#!/usr/bin/env python3
"""Synthetic multi-query associative recall for the matched micro arms.

The matched campaign and the Mixture-of-Memories ablation measure
language-model loss, which says little about how many associations a
fixed-size recurrent state can hold. This harness trains small stacks
from scratch on the multi-query associative recall task (Zoology, Arora
et al. 2023; the recall-throughput tradeoff paper uses the same task):
a sequence of N key-value pairs followed by the same N keys in random
order, each of which must be answered with its value. Accuracy as a
function of N, at and beyond the training length, is the capacity
curve; the pair count where an arm drops below 90 percent is its
effective capacity.

Arms reuse the matched harness's blocks (scripts/matched_micro_train.py)
as pure stacks with no attention layer, since in a hybrid the attention
layers can carry the retrieval:

    attn    two-layer RoPE attention (the ceiling)
    gdn     Gated DeltaNet, five heads (the campaign's cell)
    gdn6    six heads: the state one MoM token activates (top-2 + shared)
    gdnw    ten heads: MoM's total state
    mom     Mixture-of-Memories, two heads per memory (mom3's cell)
    moms    five heads per memory, one shared key/value projection
    titans  Titans memory-as-context transformer (library default,
            anchored algorithm; recorded as in the matched harness)

Two diagnostics run on the MoM arms at evaluation: routing
consistency (the fraction of queries whose top-2 memories share at
least one memory with the position that wrote the matching value) and
accuracy with the shared memory switched off, which says how much of
the recall runs through the always-on path instead of the routed ones.

    python3 scripts/mqar_recall.py --arm gdn --out-dir mqar-runs
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matched_micro_train import (  # noqa: E402
    MOM_ROUTING,
    Attention,
    StackLM,
    environment_manifest,
)

PAD = 0
# the token that follows every query in the input.  The answer itself
# must NOT be fed back: with answers in the input a two-layer attention
# stack learns to output any value not yet used as an answer, which
# scores exactly the mean of 1/(N-j) over query order j (0.52 at four
# pairs, 0.20 at sixteen) without ever binding a key to its value
FILL = 1
# set by main() from --vocab: keys occupy the lower half of the
# vocabulary above the two special tokens, values the upper half
VOCAB = 8192
KEY_LO, KEY_HI = 2, VOCAB // 2
VAL_LO, VAL_HI = VOCAB // 2, VOCAB


def set_vocab(vocab):
    global VOCAB, KEY_LO, KEY_HI, VAL_LO, VAL_HI
    VOCAB = vocab
    KEY_LO, KEY_HI = 2, vocab // 2
    VAL_LO, VAL_HI = vocab // 2, vocab


ARMS = dict(
    attn=dict(kind="stack", dim=512, layers=2, heads=8, layout="A"),
    # RoPE alone leaves a two-layer stack stuck near half accuracy on
    # four pairs at this scale: the previous-token head forms slowly.
    # The Zoology attention baselines carry learned absolute position
    # embeddings, which make that head trivial; this variant adds them.
    attn_pos=dict(kind="stack", dim=512, layers=2, heads=8, layout="A", abs_pos=True),
    # Neither position scheme forms an induction circuit at this scale:
    # both variants sit at 1/N, a value picked from the context without
    # the key.  Every linear cell here carries a size-4 short
    # convolution on q, k and v that hands it the previous token; this
    # variant gives attention the same convolution, so the ceiling is
    # attention with the linear cells' inductive bias, not a two-layer
    # stack that has to discover it
    attn_conv=dict(
        kind="stack", dim=512, layers=2, heads=8, layout="A", short_conv=True
    ),
    gdn=dict(kind="stack", dim=512, layers=2, heads=8, gdn_heads=5, layout="G"),
    gdn6=dict(kind="stack", dim=512, layers=2, heads=8, gdn_heads=6, layout="G"),
    gdnw=dict(kind="stack", dim=512, layers=2, heads=8, gdn_heads=10, layout="G"),
    mom=dict(
        kind="stack", dim=512, layers=2, heads=8, mom_heads=2, layout="M", **MOM_ROUTING
    ),
    moms=dict(
        kind="stack",
        dim=512,
        layers=2,
        heads=8,
        mom_heads=5,
        single_kv_proj=True,
        layout="M",
        **MOM_ROUTING,
    ),
    titans=dict(
        kind="titans",
        dim=512,
        depth=2,
        segment_len=128,
        heads=8,
        dim_head=64,
        num_persist_mem_tokens=4,
        num_longterm_mem_tokens=4,
        neural_memory_batch_size=None,
        memory_expansion=2.0,
    ),
)


# ---------------------------------------------------------------------------
# task
# ---------------------------------------------------------------------------


def make_examples(rng, num_pairs, count, seq_len):
    """count sequences of num_pairs key-value pairs followed by the keys
    in random order, each followed by a filler token, with the value as
    the target at the key; returns (inputs, targets, write_pos, read_pos)
    where targets are -100 except at query positions, and the two index
    arrays map pair i to the position that wrote its value and the
    position that asks for it."""
    n = num_pairs
    assert 4 * n <= seq_len
    x = np.full((count, seq_len), PAD, dtype=np.int64)
    y = np.full((count, seq_len), -100, dtype=np.int64)
    write_pos = np.zeros((count, n), dtype=np.int64)
    read_pos = np.zeros((count, n), dtype=np.int64)
    for c in range(count):
        keys = rng.choice(np.arange(KEY_LO, KEY_HI), size=n, replace=False)
        vals = rng.integers(VAL_LO, VAL_HI, size=n)
        x[c, 0 : 2 * n : 2] = keys
        x[c, 1 : 2 * n : 2] = vals
        order = rng.permutation(n)
        q = 2 * n + 2 * np.arange(n)
        x[c, q] = keys[order]
        x[c, q + 1] = FILL
        y[c, q] = vals[order]
        write_pos[c, order] = 2 * np.arange(n) + 1
        read_pos[c, order] = q
    return x, y, write_pos, read_pos


def train_batch(rng, levels, batch, seq_len):
    xs, ys = [], []
    per = rng.choice(levels, size=batch)
    for n in per:
        x, y, _, _ = make_examples(rng, int(n), 1, seq_len)
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


class ConvAttention(Attention):
    """The harness's RoPE attention with fla's short convolution (kernel
    4, SiLU) on the q, k and v projections, as in GatedDeltaNet."""

    def __init__(self, dim, heads):
        super().__init__(dim, heads)
        from fla.modules import ShortConvolution

        self.convs = nn.ModuleList(
            ShortConvolution(hidden_size=dim, kernel_size=4, activation="silu")
            for _ in range(3)
        )

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (conv(z)[0] for conv, z in zip(self.convs, (q, k, v)))
        q, k, v = (
            z.view(b, t, self.heads, self.dim_head).transpose(1, 2) for z in (q, k, v)
        )
        q, k = self.rope(q, k)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(o.transpose(1, 2).reshape(b, t, d))


class PosEmbed(nn.Module):
    """Token embedding plus a learned absolute position embedding; wraps
    the stack's embedding so the tied output head keeps its weight."""

    def __init__(self, embed, max_len, dim):
        super().__init__()
        self.embed = embed
        self.pos = nn.Embedding(max_len, dim)
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, idx):
        return self.embed(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))


def build(arm, device, seed, dim=None):
    cfg = dict(ARMS[arm])
    if dim:
        cfg["dim"] = dim
    torch.manual_seed(seed)
    if cfg["kind"] == "stack":
        model = StackLM(cfg, VOCAB)
        if cfg.get("abs_pos"):
            model.embed = PosEmbed(model.embed, 4096, cfg["dim"])
        if cfg.get("short_conv"):
            for blk in model.blocks:
                if blk.mixer_kind == "A":
                    blk.mixer = ConvAttention(cfg["dim"], cfg["heads"])
    else:
        from titans_pytorch import MemoryAsContextTransformer

        model = MemoryAsContextTransformer(
            num_tokens=VOCAB,
            dim=cfg["dim"],
            depth=cfg["depth"],
            segment_len=cfg["segment_len"],
            heads=cfg["heads"],
            dim_head=cfg["dim_head"],
            num_persist_mem_tokens=cfg["num_persist_mem_tokens"],
            num_longterm_mem_tokens=cfg["num_longterm_mem_tokens"],
            neural_memory_batch_size=cfg["neural_memory_batch_size"],
            neural_memory_kwargs=dict(
                default_model_kwargs=dict(
                    depth=2, expansion_factor=cfg["memory_expansion"]
                )
            ),
        )
    return model.to(device)


def logits_of(model, x):
    out = model(x)
    return out[0] if isinstance(out, tuple) else out


def masked_ce(logits, y):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


def mom_blocks(model):
    return [b for b in getattr(model, "blocks", []) if b.mixer_kind == "M"]


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model, arm, device, levels, examples, seed, eval_batch):
    """Per-N accuracy at the query positions, plus the MoM diagnostics."""
    model.eval()
    report = {}
    for n in levels:
        rng = np.random.default_rng(seed + n)
        seq_len = 4 * n
        x, y, wpos, rpos = make_examples(rng, n, examples, seq_len)
        correct = total = 0
        overlap_hits = overlap_total = 0
        # correctness by query order (recency of the question) and by
        # pair index (age of the association) for the first and last
        # quarter of each, a forgetting/recency read on every arm
        by_order = np.zeros(n)
        by_pair = np.zeros(n)
        for s in range(0, examples, eval_batch):
            xb = torch.from_numpy(x[s : s + eval_batch]).to(device)
            yb = torch.from_numpy(y[s : s + eval_batch]).to(device)
            pred = logits_of(model, xb).argmax(-1)
            mask = yb != -100
            correct += (pred[mask] == yb[mask]).sum().item()
            total += mask.sum().item()
            hit = (pred == yb).cpu().numpy()  # (b, seq)
            r = rpos[s : s + eval_batch]  # (b, n): pair i -> query position
            rows = np.arange(r.shape[0])[:, None]
            hit_pair = hit[rows, r]  # (b, n) indexed by pair
            by_pair += hit_pair.sum(0)
            order = np.argsort(r, axis=1)  # pair index in query order
            by_order += np.take_along_axis(hit_pair, order, axis=1).sum(0)
            for blk in mom_blocks(model):
                # (batch*seq, memories) -> top-2 memory sets per position
                logits = blk.router_logits.view(xb.shape[0], xb.shape[1], -1)
                top = logits.topk(2, dim=-1).indices  # (b, s, 2)
                b_idx = torch.arange(xb.shape[0], device=device)[:, None]
                w = torch.from_numpy(wpos[s : s + eval_batch]).to(device)
                r = torch.from_numpy(rpos[s : s + eval_batch]).to(device)
                wset = top[b_idx, w]  # (b, n, 2)
                rset = top[b_idx, r]
                hit = (wset[..., :, None] == rset[..., None, :]).any(-1).any(-1)
                overlap_hits += hit.sum().item()
                overlap_total += hit.numel()
        entry = dict(pairs=n, seq_len=seq_len, accuracy=correct / max(1, total))
        q = max(1, n // 4)
        entry["acc_first_queries"] = float(by_order[:q].sum() / (q * examples))
        entry["acc_last_queries"] = float(by_order[-q:].sum() / (q * examples))
        entry["acc_oldest_pairs"] = float(by_pair[:q].sum() / (q * examples))
        entry["acc_newest_pairs"] = float(by_pair[-q:].sum() / (q * examples))
        if overlap_total:
            entry["routing_overlap"] = overlap_hits / overlap_total
            # two independent top-k draws from E memories share a member
            # with probability 1 - C(E-k, k) / C(E, k): 0.833 for top-2
            # of four, which is what an untrained router reads
            blk = mom_blocks(model)[0].mixer
            e, k = blk.num_memories, blk.topk
            entry["routing_overlap_chance"] = 1 - math.comb(e - k, k) / math.comb(e, k)
        blocks = mom_blocks(model)
        if blocks:
            for blk in blocks:
                blk.mixer.shared_mem = False
            correct_off = 0
            for s in range(0, examples, eval_batch):
                xb = torch.from_numpy(x[s : s + eval_batch]).to(device)
                yb = torch.from_numpy(y[s : s + eval_batch]).to(device)
                pred = logits_of(model, xb).argmax(-1)
                mask = yb != -100
                correct_off += (pred[mask] == yb[mask]).sum().item()
            for blk in blocks:
                blk.mixer.shared_mem = True
            entry["accuracy_shared_off"] = correct_off / max(1, total)
        report[n] = entry
    model.train()
    return report


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def lr_at(step, steps, lr, warmup):
    if step < warmup:
        return lr * (step + 1) / warmup
    p = (step - warmup) / max(1, steps - warmup)
    return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--train-pairs", default="4,8,16,32,64")
    ap.add_argument("--eval-pairs", default="4,8,16,32,64,128,256")
    ap.add_argument("--eval-examples", type=int, default=512)
    ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out-dir", default="mqar-runs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument(
        "--dim",
        type=int,
        default=None,
        help="override every arm's model width (the cells keep their head "
        "configuration, so state size is unchanged)",
    )
    ap.add_argument(
        "--vocab",
        type=int,
        default=8192,
        help="task vocabulary; half keys, half values (smaller trains faster)",
    )
    args = ap.parse_args()
    set_vocab(args.vocab)

    device = torch.device(args.device)
    levels = [int(n) for n in args.train_pairs.split(",")]
    eval_levels = [int(n) for n in args.eval_pairs.split(",")]
    seq_len = 4 * max(levels)
    cfg = dict(ARMS[args.arm])
    if args.dim:
        cfg["dim"] = args.dim
    model = build(args.arm, device, args.seed, args.dim)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    rng = np.random.default_rng(args.seed + 7)
    os.makedirs(args.out_dir, exist_ok=True)
    run = dict(
        arm=args.arm,
        config=dict(cfg),
        args=vars(args),
        task=dict(
            vocab=VOCAB,
            train_pairs=levels,
            train_seq_len=seq_len,
            eval_pairs=eval_levels,
            answers_in_input=False,
        ),
        params=params,
        environment=environment_manifest(),
        history=[],
        evals=[],
    )
    print(
        f"arm={args.arm} params={params:,} train seq {seq_len} device={device}",
        flush=True,
    )
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    tokens = 0
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        x, y = train_batch(rng, levels, args.batch, seq_len)
        xb = torch.from_numpy(x).to(device)
        yb = torch.from_numpy(y).to(device)
        opt.zero_grad(set_to_none=True)
        logits = logits_of(model, xb)
        loss = masked_ce(logits, yb)
        total = loss
        aux = None
        scale = cfg.get("aux_loss_scale", 0.0)
        if scale and hasattr(model, "load_balance_loss"):
            aux = model.load_balance_loss()
            total = loss + scale * aux
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tokens += xb.numel()
        if step % args.log_every == 0 or step == args.steps - 1:
            el = time.time() - t0
            entry = dict(step=step, loss=loss.item(), tok_s=tokens / el, elapsed_s=el)
            if aux is not None:
                entry["aux_loss"] = aux.item()
            if device.type == "cuda":
                entry["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
            run["history"].append(entry)
            print(
                f"  step {step:5d}  loss {loss.item():.4f}  tok/s {entry['tok_s']:.0f}",
                flush=True,
            )
        if args.eval_every and step and step % args.eval_every == 0:
            rep = evaluate(
                model,
                args.arm,
                device,
                eval_levels,
                args.eval_examples,
                args.seed,
                args.eval_batch,
            )
            run["evals"].append(dict(step=step, report=rep))
            print(
                f"  step {step:5d}  acc "
                + " ".join(f"N{n}={r['accuracy']:.3f}" for n, r in rep.items()),
                flush=True,
            )
    rep = evaluate(
        model,
        args.arm,
        device,
        eval_levels,
        args.eval_examples,
        args.seed,
        args.eval_batch,
    )
    run["final"] = dict(report=rep, wall_s=time.time() - t0)
    if device.type == "cuda":
        run["final"]["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    print(
        "  final acc " + " ".join(f"N{n}={r['accuracy']:.3f}" for n, r in rep.items())
    )
    for n, r in rep.items():
        if "routing_overlap" in r:
            print(
                f"  N{n}: routing overlap {r['routing_overlap']:.3f} "
                f"(chance {r['routing_overlap_chance']:.3f})  "
                f"shared-off acc {r['accuracy_shared_off']:.3f}"
            )
    path = os.path.join(args.out_dir, f"{args.arm}.json")
    with open(path, "w") as f:
        json.dump(run, f, indent=1)
    print(f"  -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
