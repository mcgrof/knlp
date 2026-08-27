#!/usr/bin/env python3
"""Matched micro-scale training harness for the Titans/Hope evaluation.

Open-weight comparisons cannot isolate architecture, so the evaluation
plan trains small models under one controlled contract: one tokenizer,
one materialized token stream consumed in the same order, one outer
optimizer and schedule, and parameter counts matched within tolerance.
This harness supplies that contract for five arms:

    attn      full-attention transformer (control)
    gdn3      3:1 Gated DeltaNet / full-attention hybrid
    gdn7      7:1 Gated DeltaNet / full-attention hybrid
    titans    Titans memory-as-context transformer (lucidrains
              candidate, audited in the fidelity lane)
    hope      Hope paper block (self-modifying Titans + continuum
              memory; kmccleary3301 candidate, audited likewise)

The attention and hybrid arms are implemented here (RoPE attention,
pre-norm blocks, fla's GatedDeltaNet layer for the linear positions).
The Titans and Hope arms use the audited community candidates
unchanged, because the point is to evaluate those candidates; their
audited configuration obligations are honored and recorded — the
Titans arm records that the surprise-gradient anchor never advances
under the library default, and the Hope arm runs the two-pass
stop-gradient training its repository defines, with the self-mod
convolution disabled because it keeps no cross-call state.

Hope trains and evaluates prequentially: each chunk is scored before
its own targets feed the memory through the teach signal, so no
position's loss ever sees its own or a future target.

Data is materialized once (`prepare-data`) to a token file all arms
read identically; the final slice is held out for validation. Every
run records its environment, configuration, losses, throughput, and
peak memory as JSON.

    python3 scripts/matched_micro_train.py prepare-data --tokens 40000000
    python3 scripts/matched_micro_train.py params
    python3 scripts/matched_micro_train.py train --arm gdn3 --steps 300
"""

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# the matched contract
# ---------------------------------------------------------------------------

CONTRACT = dict(
    tokenizer="openai-community/gpt2",
    vocab_size=50257,
    dataset="roneneldan/TinyStories",
    split="train",
    seq_len=512,
    batch_size=8,
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    warmup_steps=100,
    min_lr_ratio=0.1,
    grad_clip=1.0,
    seed=1234,
    val_fraction=0.02,
)

# per-arm architecture configs, sized by the `params` subcommand to sit
# within 5% of the attention control's total parameter count
ARMS = dict(
    attn=dict(kind="stack", dim=512, layers=8, heads=8, layout="A"),
    gdn3=dict(kind="stack", dim=512, layers=8, heads=8, gdn_heads=5, layout="GGGA"),
    gdn7=dict(kind="stack", dim=512, layers=8, heads=8, gdn_heads=5, layout="GGGGGGGA"),
    titans=dict(
        kind="titans",
        dim=384,
        depth=4,
        segment_len=128,
        heads=8,
        dim_head=48,
        num_persist_mem_tokens=4,
        num_longterm_mem_tokens=4,
        # the library default: the surprise-gradient anchor never
        # advances.  Recorded as part of the algorithm identity per the
        # Titans fidelity verdict.
        neural_memory_batch_size=None,
        # memory-MLP expansion halved to land inside the parameter
        # tolerance; a capacity lever, recorded here
        memory_expansion=2.0,
    ),
    hope=dict(
        kind="hope",
        dim=384,
        layers=5,
        cms_periods=(1, 4, 32, 128),
        self_mod_chunk=8,
        self_mod_chunk_memory=64,
        # disabled per the Hope fidelity verdict: the conv keeps no
        # cross-call state, so chunked training would compute a
        # different operator than the sequence-level conv
        local_conv_window=None,
        train_chunk=64,
    ),
)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def data_paths(data_dir):
    return (
        os.path.join(data_dir, "tokens_gpt2.npy"),
        os.path.join(data_dir, "tokens_gpt2.json"),
    )


def prepare_data(data_dir, num_tokens):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    os.makedirs(data_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(CONTRACT["tokenizer"])
    ds = load_dataset(CONTRACT["dataset"], split=CONTRACT["split"], streaming=True)
    out = np.empty(num_tokens, dtype=np.uint16)
    filled = 0
    docs = 0
    for ex in ds:
        ids = tok(ex.get("text") or "").input_ids + [tok.eos_token_id]
        take = min(len(ids), num_tokens - filled)
        out[filled : filled + take] = np.asarray(ids[:take], dtype=np.uint16)
        filled += take
        docs += 1
        if filled >= num_tokens:
            break
    npy, meta = data_paths(data_dir)
    np.save(npy, out[:filled])
    with open(meta, "w") as f:
        json.dump(
            dict(
                tokenizer=CONTRACT["tokenizer"],
                dataset=CONTRACT["dataset"],
                split=CONTRACT["split"],
                tokens=int(filled),
                documents=docs,
            ),
            f,
            indent=1,
        )
    print(f"{filled} tokens from {docs} documents -> {npy}")


class TokenStream:
    """The one token stream every arm consumes in the same order."""

    def __init__(self, data_dir, seq_len, batch_size):
        npy, meta = data_paths(data_dir)
        self.tokens = np.load(npy, mmap_mode="r")
        self.meta = json.load(open(meta))
        self.seq_len = seq_len
        self.batch_size = batch_size
        n = len(self.tokens)
        val_tokens = int(n * CONTRACT["val_fraction"])
        self.train_end = n - val_tokens
        self.rows_per_step = batch_size
        self.row_len = seq_len + 1  # inputs plus the shifted target

    def train_batch(self, step, device):
        rows = []
        base = step * self.rows_per_step
        for r in range(self.rows_per_step):
            start = ((base + r) * self.row_len) % (self.train_end - self.row_len)
            rows.append(self.tokens[start : start + self.row_len].astype(np.int64))
        return torch.from_numpy(np.stack(rows)).to(device)

    def val_batches(self, device, max_batches=8):
        start = self.train_end
        out = []
        while start + self.row_len <= len(self.tokens) and len(out) < max_batches:
            rows = []
            for _ in range(self.rows_per_step):
                if start + self.row_len > len(self.tokens):
                    break
                rows.append(self.tokens[start : start + self.row_len].astype(np.int64))
                start += self.row_len
            if len(rows) == self.rows_per_step:
                out.append(torch.from_numpy(np.stack(rows)).to(device))
        return out


# ---------------------------------------------------------------------------
# the stack arms: RoPE attention and GatedDeltaNet blocks
# ---------------------------------------------------------------------------


class Rotary(nn.Module):
    def __init__(self, dim_head, max_len=8192, base=10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim_head, 2).float() / dim_head))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, q, k):
        # q, k: (B, H, T, D)
        t = q.shape[-2]
        cos = self.cos[:t].to(q.dtype)
        sin = self.sin[:t].to(q.dtype)

        def rot(x):
            x1, x2 = x[..., 0::2], x[..., 1::2]
            return torch.stack(
                (x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1
            ).flatten(-2)

        return rot(q), rot(k)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.out.weight, std=0.02)
        self.rope = Rotary(self.dim_head)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (
            z.view(b, t, self.heads, self.dim_head).transpose(1, 2) for z in (q, k, v)
        )
        q, k = self.rope(q, k)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(o.transpose(1, 2).reshape(b, t, d))


class MLP(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.up = nn.Linear(dim, mult * dim, bias=False)
        self.down = nn.Linear(mult * dim, dim, bias=False)
        nn.init.normal_(self.up.weight, std=0.02)
        nn.init.normal_(self.down.weight, std=0.02)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, dim, heads, mixer_kind, gdn_heads=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mixer_kind = mixer_kind
        if mixer_kind == "A":
            self.mixer = Attention(dim, heads)
        else:
            from fla.layers.gated_deltanet import GatedDeltaNet

            self.mixer = GatedDeltaNet(
                hidden_size=dim,
                head_dim=64,
                num_heads=gdn_heads or heads,
                mode="chunk",
            )
        self.mlp = MLP(dim)

    def forward(self, x):
        h = self.norm1(x)
        if self.mixer_kind == "A":
            h = self.mixer(h)
        else:
            h = self.mixer(h)
            if isinstance(h, tuple):
                h = h[0]
        x = x + h
        return x + self.mlp(self.norm2(x))


class StackLM(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        layout = cfg["layout"]
        self.embed = nn.Embedding(vocab_size, cfg["dim"])
        kinds = [layout[i % len(layout)] for i in range(cfg["layers"])]
        self.kinds = "".join(kinds)
        self.blocks = nn.ModuleList(
            Block(cfg["dim"], cfg["heads"], k, gdn_heads=cfg.get("gdn_heads"))
            for k in kinds
        )
        self.norm = nn.LayerNorm(cfg["dim"])
        self.head = nn.Linear(cfg["dim"], vocab_size, bias=False)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.head.weight = self.embed.weight

    def forward(self, idx):
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


# ---------------------------------------------------------------------------
# arm construction and per-arm training steps
# ---------------------------------------------------------------------------


def build_arm(name, device):
    cfg = ARMS[name]
    torch.manual_seed(CONTRACT["seed"])
    if cfg["kind"] == "stack":
        model = StackLM(cfg, CONTRACT["vocab_size"]).to(device)
    elif cfg["kind"] == "titans":
        from titans_pytorch import MemoryAsContextTransformer

        model = MemoryAsContextTransformer(
            num_tokens=CONTRACT["vocab_size"],
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
        ).to(device)
    elif cfg["kind"] == "hope":
        from nested_learning.levels import LevelSpec
        from nested_learning.model import HOPEModel, ModelConfig

        model = HOPEModel(
            ModelConfig(
                vocab_size=CONTRACT["vocab_size"],
                dim=cfg["dim"],
                num_layers=cfg["layers"],
                heads=8,
                titan_level=LevelSpec(name="titan", update_period=1),
                cms_levels=tuple(
                    LevelSpec(name=f"cms_p{p}", update_period=p)
                    for p in cfg["cms_periods"]
                ),
                self_mod_chunk_size=cfg["self_mod_chunk"],
                self_mod_chunk_size_memory=cfg["self_mod_chunk_memory"],
                self_mod_local_conv_window=cfg["local_conv_window"],
                block_variant="hope_selfmod",
            )
        ).to(device)
        # the candidate ties its LM head to a default-normal embedding,
        # which explodes the initial logits and, through the teach
        # signal, the fast weights; scaled init is a harness-level
        # intervention recorded here, not an architecture change
        nn.init.normal_(model.embed.weight, std=0.02)
    else:
        raise ValueError(cfg["kind"])
    return model


def param_report(name, model):
    total = sum(p.numel() for p in model.parameters())
    # tied weights are counted once by summing unique data pointers
    seen = set()
    unique = 0
    for p in model.parameters():
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            unique += p.numel()
    embed = 0
    for mod_name, p in model.named_parameters():
        if "embed" in mod_name or "token_emb" in mod_name:
            embed += p.numel()
    return dict(
        arm=name,
        total=total,
        unique=unique,
        embedding=embed,
        non_embedding=unique - embed,
    )


def ce_loss(logits, idx):
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)), idx[:, 1:].reshape(-1)
    )


def step_standard(model, idx):
    logits = model(idx)
    loss = ce_loss(logits, idx)
    loss.backward()
    return loss.item()


def step_titans(model, idx):
    loss = model(idx, return_loss=True)
    loss.backward()
    return loss.item()


def hope_two_pass(model, idx, train_chunk, backward=True):
    """The candidate repository's two-pass stop-gradient protocol with
    boundary-target chunk supervision, scored prequentially: each
    chunk's loss is computed before its own targets feed the memory."""
    from nested_learning.training import compute_teach_signal

    fs = model.init_fast_state()
    b, total_len = idx.shape
    chunks = [
        (s, min(s + train_chunk, total_len)) for s in range(0, total_len, train_chunk)
    ]
    total_loss = 0.0
    total_targets = 0
    weights = []
    for start, end in chunks:
        n_targets = (end - start - 1) + (1 if end < total_len else 0)
        weights.append(max(n_targets, 0))
    denom = float(sum(weights))
    for (start, end), weight in zip(chunks, weights):
        if weight == 0:
            continue
        chunk = idx[:, start:end]
        next_tok = idx[:, end : end + 1] if end < total_len else None
        logits = model(chunk, fast_state=fs)
        if next_tok is not None:
            targets = torch.cat([chunk[:, 1:], next_tok], dim=1)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        else:
            loss = ce_loss(logits, chunk)
        if backward:
            (loss * (weight / denom)).backward()
        teach = compute_teach_signal(model, logits, chunk, next_tokens=next_tok)
        with torch.no_grad():
            model(
                chunk,
                teach_signal=teach,
                fast_state=fs,
                finalize_updates=(end >= total_len),
            )
        total_loss += loss.item() * weight
        total_targets += weight
    return total_loss / total_targets


@torch.no_grad()
def evaluate(name, model, batches, cfg):
    model.eval()
    losses = []
    for idx in batches:
        if cfg["kind"] == "stack":
            losses.append(ce_loss(model(idx), idx).item())
        elif cfg["kind"] == "titans":
            losses.append(model(idx, return_loss=True).item())
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def evaluate_hope(model, batches, cfg):
    # prequential: updates enabled, but every position is scored before
    # its target reaches the memory.  The model stays in train mode
    # deliberately: the candidate clips its continuum-memory deltas
    # only under self.training, and without that clip the eval-mode
    # forward diverges to NaN within two chunks — the trained function
    # is the clipped one, so the clipped one is what gets scored.
    losses = [
        hope_two_pass(model, idx, cfg["train_chunk"], backward=False) for idx in batches
    ]
    return sum(losses) / len(losses) if losses else float("nan")


# ---------------------------------------------------------------------------
# training driver
# ---------------------------------------------------------------------------


def lr_at(step, total_steps):
    warm = CONTRACT["warmup_steps"]
    if step < warm:
        return CONTRACT["lr"] * (step + 1) / warm
    progress = (step - warm) / max(1, total_steps - warm)
    floor = CONTRACT["min_lr_ratio"]
    return CONTRACT["lr"] * (
        floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
    )


def environment_manifest():
    def git_head(path):
        try:
            return subprocess.run(
                ["git", "-C", path, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            return "unknown"

    manifest = dict(
        torch=torch.__version__,
        python=platform.python_version(),
        platform=platform.platform(),
        knlp=git_head(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        cuda_available=torch.cuda.is_available(),
    )
    if torch.cuda.is_available():
        manifest["gpu"] = torch.cuda.get_device_name(0)
    for pkg in ("fla", "titans_pytorch", "nested_learning"):
        try:
            mod = __import__(pkg)
            path = os.path.dirname(os.path.dirname(mod.__file__))
            manifest[pkg] = git_head(path)
        except Exception:  # noqa: BLE001
            pass
    return manifest


def train(
    arm,
    data_dir,
    steps,
    out_dir,
    device_str,
    log_every=10,
    eval_every=0,
    save_checkpoint=False,
):
    device = torch.device(device_str)
    cfg = ARMS[arm]
    stream = TokenStream(data_dir, CONTRACT["seq_len"], CONTRACT["batch_size"])
    model = build_arm(arm, device)
    params = param_report(arm, model)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=CONTRACT["lr"],
        betas=CONTRACT["betas"],
        weight_decay=CONTRACT["weight_decay"],
    )
    torch.manual_seed(CONTRACT["seed"] + 1)
    os.makedirs(out_dir, exist_ok=True)

    run = dict(
        arm=arm,
        config=dict(cfg),
        contract=dict(CONTRACT),
        params=params,
        environment=environment_manifest(),
        data=stream.meta,
        steps_requested=steps,
        history=[],
    )
    run["config"]["cms_periods"] = list(cfg.get("cms_periods", []))
    print(
        f"arm={arm} params={params['unique']:,} "
        f"(non-embedding {params['non_embedding']:,}) device={device}",
        flush=True,
    )

    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    tokens_seen = 0
    val_batches = stream.val_batches(device)
    for step in range(steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, steps)
        idx = stream.train_batch(step, device)
        opt.zero_grad(set_to_none=True)
        if cfg["kind"] == "stack":
            loss = step_standard(model, idx)
        elif cfg["kind"] == "titans":
            loss = step_titans(model, idx)
        else:
            loss = hope_two_pass(model, idx, cfg["train_chunk"])
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONTRACT["grad_clip"])
        opt.step()
        tokens_seen += idx.shape[0] * (idx.shape[1] - 1)
        if step % log_every == 0 or step == steps - 1:
            elapsed = time.time() - t0
            entry = dict(
                step=step,
                loss=loss,
                ppl=math.exp(min(20.0, loss)),
                tokens=tokens_seen,
                tok_s=tokens_seen / elapsed,
                elapsed_s=elapsed,
            )
            if device.type == "cuda":
                entry["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
            run["history"].append(entry)
            print(
                f"  step {step:5d}  loss {loss:.4f}  ppl {entry['ppl']:.2f}  "
                f"tok/s {entry['tok_s']:.0f}",
                flush=True,
            )
        if eval_every and step and step % eval_every == 0:
            val = (
                evaluate_hope(model, val_batches, cfg)
                if cfg["kind"] == "hope"
                else evaluate(arm, model, val_batches, cfg)
            )
            run["history"].append(dict(step=step, val_loss=val))
            print(f"  step {step:5d}  val_loss {val:.4f}", flush=True)

    val = (
        evaluate_hope(model, val_batches, cfg)
        if cfg["kind"] == "hope"
        else evaluate(arm, model, val_batches, cfg)
    )
    run["final"] = dict(
        train_loss=run["history"][-1]["loss"] if run["history"] else float("nan"),
        val_loss=val,
        val_ppl=math.exp(min(20.0, val)),
        tokens=tokens_seen,
        wall_s=time.time() - t0,
    )
    if device.type == "cuda":
        run["final"]["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"  final: val_loss {val:.4f} val_ppl {run['final']['val_ppl']:.2f} "
        f"tokens {tokens_seen} wall {run['final']['wall_s']:.0f}s",
        flush=True,
    )
    out_path = os.path.join(out_dir, f"{arm}.json")
    with open(out_path, "w") as f:
        json.dump(run, f, indent=1)
    print(f"  -> {out_path}", flush=True)
    if save_checkpoint:
        # titans double-registers memory-MLP views; load back with
        # load_state_dict(..., assign=True)
        ckpt_path = os.path.join(out_dir, f"{arm}.pt")
        torch.save(
            dict(
                arm=arm,
                config=dict(cfg),
                contract=dict(CONTRACT),
                model=model.state_dict(),
            ),
            ckpt_path,
        )
        print(f"  -> {ckpt_path}", flush=True)
    return run


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("prepare-data", help="materialize the shared token stream")
    d.add_argument("--data-dir", default="matched-micro-data")
    d.add_argument("--tokens", type=int, default=40_000_000)

    p = sub.add_parser("params", help="parameter report for every arm (CPU)")
    p.add_argument("--arms", default="all")

    t = sub.add_parser("train", help="train one arm under the matched contract")
    t.add_argument("--arm", required=True, choices=sorted(ARMS))
    t.add_argument("--data-dir", default="matched-micro-data")
    t.add_argument("--steps", type=int, default=300)
    t.add_argument("--out-dir", default="matched-micro-runs")
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    t.add_argument("--log-every", type=int, default=10)
    t.add_argument("--eval-every", type=int, default=0)
    t.add_argument(
        "--batch", type=int, default=None, help="override the contract (dry runs)"
    )
    t.add_argument(
        "--seq-len", type=int, default=None, help="override the contract (dry runs)"
    )
    t.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override the contract seed (the sanctioned per-run knob for "
        "multi-seed campaigns; recorded in the run JSON)",
    )
    t.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help="derive --steps from a training-token budget at the effective "
        "batch and sequence length",
    )
    t.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="save final weights as <out-dir>/<arm>.pt next to the run JSON",
    )

    args = ap.parse_args()

    if args.cmd == "prepare-data":
        prepare_data(args.data_dir, args.tokens)
        return 0

    if args.cmd == "params":
        arms = sorted(ARMS) if args.arms == "all" else args.arms.split(",")
        reports = []
        for arm in arms:
            model = build_arm(arm, torch.device("cpu"))
            rep = param_report(arm, model)
            reports.append(rep)
            del model
            print(
                f"{arm:8s} total {rep['unique']:>12,}  "
                f"embedding {rep['embedding']:>12,}  "
                f"non-embedding {rep['non_embedding']:>12,}",
                flush=True,
            )
        base = next((r for r in reports if r["arm"] == "attn"), reports[0])
        for rep in reports:
            drift = (rep["unique"] - base["unique"]) / base["unique"]
            print(f"{rep['arm']:8s} vs attn: {drift:+.1%}")
        return 0

    if args.cmd == "train":
        if args.batch:
            CONTRACT["batch_size"] = args.batch
            CONTRACT["contract_overridden"] = True
        if args.seq_len:
            CONTRACT["seq_len"] = args.seq_len
            CONTRACT["contract_overridden"] = True
        if args.seed is not None:
            CONTRACT["seed"] = args.seed
        if args.token_budget:
            per_step = CONTRACT["batch_size"] * (CONTRACT["seq_len"] - 1)
            args.steps = max(1, args.token_budget // per_step)
            print(
                f"token budget {args.token_budget} at batch "
                f"{CONTRACT['batch_size']} seq {CONTRACT['seq_len']} "
                f"-> {args.steps} steps",
                flush=True,
            )
        train(
            args.arm,
            args.data_dir,
            args.steps,
            args.out_dir,
            args.device,
            log_every=args.log_every,
            eval_every=args.eval_every,
            save_checkpoint=args.save_checkpoint,
        )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
