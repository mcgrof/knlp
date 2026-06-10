"""Trellis-vs-linear-cousins scaling ladder (the "does it cross over at scale" test).

Our toy 4-5M Trellis loses to Gated DeltaNet; the paper (arXiv:2512.23852)
claims Trellis WINS at 125M-1B. That is a testable scaling hypothesis, not a
conclusion. This driver climbs a parameter ladder with a *leading* training
recipe (AdamW betas 0.9/0.95, wd 0.1, linear warmup + cosine decay, grad clip,
near-Chinchilla tokens, gamma=0.1, chunk16 operator). Precision: fp32 master
weights; pass --amp-bf16 for bf16 mixed precision on every model (dense FA2,
fla native, Trellis bf16 inputs with fp32 decay/state -- head-on PPL-neutral).
Trains the same four models -- dense / DeltaNet / Gated DeltaNet /
Trellis -- at each rung on C4, reporting val PPL and the Trellis-minus-GatedDelta
gap vs model size. If that gap closes toward 125M the scale story holds; if it
stays flat the paper does not reproduce at the scale we can reach.

Runs rungs smallest-first and dumps a per-rung JSON the moment each rung
finishes, so the trend is visible as it builds and the run can be called early.

Example:
  python scripts/trellis_ladder.py --dataset c4 --seq_len 2048 \
      --rungs 256x4,384x6,512x8,640x10,768x12,1024x16 \
      --tokens_per_param 20 --out /root/trellis-ladder-results
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis_lm.config import TrellisConfig  # noqa: E402
from trellis_lm.model import build_model  # noqa: E402

# Headline set uses the fla REFERENCE DeltaNet/GatedDeltaNet (short conv + qk-norm
# + output gate) so the linear cousins are not crippled relative to Trellis. The
# minimal-mixer kinds (delta/gated_delta) stay available as a Tier-B reference.
KINDS = ["dense", "delta_ref", "gated_delta_ref", "trellis"]
ALL_KINDS = ["dense", "delta", "gated_delta", "delta_ref", "gated_delta_ref", "trellis"]
# Which kinds MUST run on the fla Triton kernel (silent bmm fallback = unfair).
_FLA_KINDS = {"delta", "gated_delta", "delta_ref", "gated_delta_ref"}


def verify_backends(kinds, cfg, device):
    """Build a 1-layer probe of each kind and confirm the intended compute
    backend is actually active, failing loudly otherwise. The unfairness risk is
    a silent fallback: an fla kind dropping to the bmm chunked kernel (~100x
    slower) or Trellis dropping off its fused Triton path. Returns a per-kind
    backend descriptor dict for the manifest.
    """
    from trellis_lm.linear_baselines_fla import HAS_FLA
    from trellis_lm.linear_baselines_fla_ref import HAS_FLA_REF, FLARefMixer
    from trellis_lm.linear_baselines_fla import FLADeltaNetMixer
    from trellis_lm.trellis_mixer import HAS_TRITON

    desc = {}
    probe_cfg = TrellisConfig.from_dict({**cfg.to_dict(), "n_layers": 1})
    for kind in kinds:
        m = build_model(probe_cfg, kind).to(device)
        blk = m.blocks[0]
        if kind == "dense":
            # DenseTransformerTiny attends via F.scaled_dot_product_attention,
            # which dispatches to the FlashAttention-2 kernel for bf16 on Ampere+.
            assert hasattr(blk, "attn") or hasattr(blk, "mixer"), "dense block shape"
            desc[kind] = "sdpa(flash-capable)"
        elif kind in ("delta", "gated_delta"):
            assert HAS_FLA, f"{kind}: fla Triton op missing -> bmm fallback is unfair"
            assert isinstance(blk.mixer, FLADeltaNetMixer), f"{kind}: wrong mixer"
            desc[kind] = "fla.ops.chunk_(gated_)delta_rule [Triton]"
        elif kind in ("delta_ref", "gated_delta_ref"):
            assert HAS_FLA_REF, f"{kind}: fla reference layers unavailable"
            assert isinstance(blk.mixer, FLARefMixer), f"{kind}: wrong mixer"
            lt = type(blk.mixer.layer).__name__
            want = "GatedDeltaNet" if "gated" in kind else "DeltaNet"
            assert lt == want, f"{kind}: layer is {lt}, want {want}"
            desc[kind] = f"fla.layers.{lt} [Triton + short-conv + qk-norm]"
        elif kind == "trellis":
            assert HAS_TRITON, "trellis: fused Triton state-evolution kernel missing"
            assert cfg.activation == "ln_silu", (
                "trellis: fused Triton path needs activation=ln_silu, "
                f"got {cfg.activation}"
            )
            desc[kind] = "trellis_triton fused fwd+bwd [Triton]"
        else:
            raise ValueError(kind)
        del m
        torch.cuda.empty_cache()
    return desc


def get_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    return tok


# bf16 mixed precision for every model: dense gets FlashAttention-2, fla runs
# native, Trellis takes bf16 inputs while its decay/state/gamma stay fp32 (mixer
# guards those). Validated PPL-neutral for Trellis by the head-on test.
_AMP = False


def _amp_ctx(device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=_AMP)


def stream_text(dataset: str, split: str):
    """Yield raw text strings from a streaming corpus, re-opening forever."""
    from datasets import load_dataset

    while True:
        if dataset == "c4":
            ds = load_dataset("allenai/c4", "en", split=split, streaming=True)
        elif dataset == "wikitext103":
            ds = load_dataset(
                "wikitext", "wikitext-103-raw-v1", split=split, streaming=True
            )
        elif dataset == "pile":
            ds = load_dataset(
                "monology/pile-uncopyrighted", split=split, streaming=True
            )
        else:
            raise ValueError(dataset)
        for ex in ds:
            t = ex.get("text") or ""
            if t:
                yield t
        if split != "train":
            return  # don't loop a finite val stream


def batch_stream(dataset, split, tok, seq_len, micro_batch):
    """Infinite generator of [micro_batch, seq_len] LongTensors packed from text."""
    eos = tok.eos_token_id
    buf, rows = [], []
    for text in stream_text(dataset, split):
        buf.extend(tok(text).input_ids + [eos])
        while len(buf) >= seq_len:
            rows.append(buf[:seq_len])
            buf = buf[seq_len:]
            if len(rows) >= micro_batch:
                yield torch.tensor(rows, dtype=torch.long)
                rows = []


def pack_val(dataset, tok, seq_len, n_rows):
    rows, buf = [], []
    eos = tok.eos_token_id
    for text in stream_text(dataset, "validation"):
        buf.extend(tok(text).input_ids + [eos])
        while len(buf) >= seq_len:
            rows.append(buf[:seq_len])
            buf = buf[seq_len:]
            if len(rows) >= n_rows:
                return torch.tensor(rows, dtype=torch.long)
    return torch.tensor(rows, dtype=torch.long)


def make_cfg(d_model, n_layers, seq_len, args):
    return TrellisConfig(
        vocab_size=args.vocab,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=d_model // 64,
        d_head=64,
        n_slots=args.n_slots,
        max_seq_len=seq_len,
        dtype="fp32",  # fp32 MASTER weights; --amp-bf16 adds bf16 autocast
        # compute (Trellis decay/state stay fp32 via the mixer guard)
        activation="ln_silu",
        alpha_mode="linear",
        beta_mode="scalar_per_head",
        forget_gate=True,
        use_short_conv_qk=True,
        gamma_init=args.gamma_init,
        exact_inner=False,  # chunk16 stale-gradient operator (fast)
        chunk_size=16,
        chunk_refine=0,
    )


def _git_commit():
    import os
    import subprocess

    env = os.environ.get("KNLP_GIT_COMMIT")
    if env:
        return env.strip()
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def write_manifest(out, args, kinds, backend_map, device):
    """Per-run provenance: versions, GPU, git commit, the verified backend per
    kind, and the full hyperparameter set. Lets anyone reproduce the exact run
    and confirm the baselines ran on their intended (Triton) kernels."""
    import platform

    import transformers

    try:
        import triton

        triton_ver = triton.__version__
    except Exception:
        triton_ver = "n/a"
    try:
        import fla

        fla_ver = getattr(fla, "__version__", "installed")
    except Exception:
        fla_ver = "n/a"
    manifest = {
        "git_commit": _git_commit(),
        "gpu": torch.cuda.get_device_name(0),
        "versions": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": triton_ver,
            "transformers": transformers.__version__,
            "fla": fla_ver,
            "python": platform.python_version(),
        },
        "kinds": kinds,
        "backends": backend_map,
        "short_conv_kernel": "pytorch-fallback (use_fast_conv1d=False; math-identical)",
        "precision": "fp32 master weights; bf16 autocast" if args.amp_bf16 else "fp32",
        "hparams": {
            k: getattr(args, k)
            for k in (
                "dataset",
                "seq_len",
                "rungs",
                "tokens_per_param",
                "base_lr",
                "gamma_init",
                "n_slots",
                "eff_tokens",
                "micro_batch",
                "warmup_frac",
                "min_lr_frac",
                "val_rows",
                "vocab",
                "seed",
                "amp_bf16",
                "bf16_inputs",
            )
        },
    }
    fn = out / "manifest.json"
    fn.write_text(json.dumps(manifest, indent=2))
    print(f"  wrote manifest {fn}", flush=True)


def lr_at(step, total, peak, warmup, min_frac):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return peak * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


@torch.no_grad()
def eval_ppl(model, val, micro_batch, device):
    model.eval()
    tot_loss, tot_tok = 0.0, 0
    for i in range(0, val.size(0) - micro_batch + 1, micro_batch):
        idx = val[i : i + micro_batch].to(device)
        with _amp_ctx(device):
            _, loss = model(idx, labels=idx, training=False)
        tot_loss += loss.float().item() * idx.numel()
        tot_tok += idx.numel()
    return math.exp(min(20, tot_loss / max(1, tot_tok)))


def train_one(kind, cfg, gen, val, total_tokens, args, device):
    torch.manual_seed(args.seed)
    model = build_model(cfg, kind).to(device)
    micro = args.micro_batch
    tok_per_micro = micro * cfg.max_seq_len
    accum = max(1, args.eff_tokens // tok_per_micro)
    total_steps = max(1, total_tokens // (accum * tok_per_micro))
    warmup = int(args.warmup_frac * total_steps)
    peak = args.base_lr * math.sqrt(512.0 / cfg.d_model)
    opt = torch.optim.AdamW(
        model.parameters(), lr=peak, betas=(0.9, 0.95), weight_decay=0.1
    )
    model.train()
    t0 = time.time()
    seen_tokens = 0
    for step in range(total_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total_steps, peak, warmup, args.min_lr_frac)
        opt.zero_grad()
        loss_val = 0.0
        for _ in range(accum):
            idx = next(gen).to(device)
            with _amp_ctx(device):
                _, loss = model(idx, labels=idx, training=True)
            (loss / accum).backward()
            loss_val += loss.float().item() / accum
            seen_tokens += idx.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 0 or (step + 1) % args.log_every == 0:
            print(
                f"    {kind} step {step + 1}/{total_steps} "
                f"lr {opt.param_groups[0]['lr']:.2e} loss {loss_val:.3f} "
                f"({seen_tokens / 1e6:.0f}M tok, {(time.time() - t0) / 60:.1f} min)",
                flush=True,
            )
    ppl = eval_ppl(model, val, micro, device)
    total_params = sum(p.numel() for p in model.parameters())
    # Decode-time memory accounting: linear/Trellis keep a BOUNDED recurrent
    # state (independent of context length); dense keeps a KV cache that GROWS
    # with context. Report both so the comparison is on equal footing.
    state_bytes = model.memory_state_bytes(1)
    if kind == "dense":
        # full-context bf16 KV cache for one sequence (K+V, all layers/heads)
        dense_kv = 2 * cfg.n_layers * cfg.n_heads * cfg.d_head * cfg.max_seq_len * 2
    else:
        dense_kv = 0
    res = {
        "kind": kind,
        "val_ppl": ppl,
        "params_total": total_params,
        "params_nonembed": model.get_num_params(),
        "tokens_trained": seen_tokens,
        "total_steps": total_steps,
        "peak_lr": peak,
        "train_min": (time.time() - t0) / 60.0,
        "state_bytes_per_seq": state_bytes,
        "dense_kv_bytes_per_seq_at_maxlen": dense_kv,
    }
    del model, opt
    torch.cuda.empty_cache()
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="c4", choices=["c4", "wikitext103", "pile"])
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument(
        "--rungs",
        default="256x4,384x6,512x8,640x10,768x12,1024x16",
        help="comma list of d_modelxn_layers",
    )
    p.add_argument("--kinds", default=",".join(KINDS))
    p.add_argument("--tokens_per_param", type=float, default=20.0)
    p.add_argument("--base_lr", type=float, default=3e-3)
    p.add_argument("--gamma_init", type=float, default=0.1)
    p.add_argument("--n_slots", type=int, default=64)
    p.add_argument("--eff_tokens", type=int, default=262144)
    p.add_argument("--micro_batch", type=int, default=8)
    p.add_argument("--warmup_frac", type=float, default=0.02)
    p.add_argument("--min_lr_frac", type=float, default=0.1)
    p.add_argument("--val_rows", type=int, default=128)
    p.add_argument("--vocab", type=int, default=50257)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--bf16-inputs",
        action="store_true",
        help="round Trellis write/read/alpha to bf16 (decay/state/gamma stay "
        "fp32) -- the head-on bf16 test vs the default fp32",
    )
    p.add_argument(
        "--amp-bf16",
        action="store_true",
        help="bf16 mixed precision for ALL models (autocast): dense->FA2, fla "
        "native, Trellis bf16 inputs w/ fp32 decay/state. The clean unified path.",
    )
    args = p.parse_args()

    if args.bf16_inputs:
        import trellis_lm.trellis_mixer as _tm

        _tm.BF16_INPUTS = True
        print("BF16_INPUTS=True (Trellis inputs bf16, decay/state fp32)", flush=True)
    if args.amp_bf16:
        global _AMP
        _AMP = True
        print(
            "AMP bf16 ON: dense FA2 / fla native / Trellis bf16-in fp32-state",
            flush=True,
        )

    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = get_tokenizer()
    kinds = args.kinds.split(",")

    # Kernel-fairness gate: verify every kind's compute backend BEFORE spending
    # GPU hours, on a 1-layer probe at the first rung's width. A silent fla->bmm
    # fallback or Trellis dropping its fused kernel would make the comparison
    # unfair; fail loudly here instead.
    first_rung = args.rungs.split(",")[0]
    fd_model, fn_layers = (int(x) for x in first_rung.lower().split("x"))
    probe_cfg = make_cfg(fd_model, fn_layers, args.seq_len, args)
    print("verifying compute backends (kernel-fairness gate) ...", flush=True)
    backend_map = verify_backends(kinds, probe_cfg, device)
    for k, b in backend_map.items():
        print(f"  {k:16s} -> {b}", flush=True)
    write_manifest(out, args, kinds, backend_map, device)

    print(f"packing {args.val_rows} val rows from {args.dataset} ...", flush=True)
    val = pack_val(args.dataset, tok, args.seq_len, args.val_rows)
    print(f"  val rows: {val.size(0)}", flush=True)

    for rung in args.rungs.split(","):
        d_model, n_layers = (int(x) for x in rung.lower().split("x"))
        cfg = make_cfg(d_model, n_layers, args.seq_len, args)
        probe = build_model(cfg, "trellis")
        ptot = sum(p.numel() for p in probe.parameters())
        del probe
        total_tokens = int(args.tokens_per_param * ptot)
        print(
            f"\n=== RUNG d{d_model} L{n_layers}: {ptot / 1e6:.1f}M total params, "
            f"{total_tokens / 1e9:.2f}B tokens budget ===",
            flush=True,
        )
        gen = batch_stream(args.dataset, "train", tok, args.seq_len, args.micro_batch)
        rung_res = {
            "d_model": d_model,
            "n_layers": n_layers,
            "seq_len": args.seq_len,
            "dataset": args.dataset,
            "tokens_per_param": args.tokens_per_param,
            "total_tokens_budget": total_tokens,
            "runs": [],
        }
        for kind in kinds:
            r = train_one(kind, cfg, gen, val, total_tokens, args, device)
            print(
                f"  -> {kind}: ppl {r['val_ppl']:.3f} "
                f"({r['params_total'] / 1e6:.1f}M, {r['train_min']:.1f} min)",
                flush=True,
            )
            rung_res["runs"].append(r)
        ppls = {r["kind"]: r["val_ppl"] for r in rung_res["runs"]}
        if "trellis" in ppls and "gated_delta" in ppls:
            gap = ppls["trellis"] - ppls["gated_delta"]
            rung_res["trellis_minus_gated"] = gap
            print(
                f"  RUNG SUMMARY d{d_model}: trellis {ppls['trellis']:.2f} "
                f"vs gated_delta {ppls['gated_delta']:.2f} -> gap {gap:+.2f} "
                f"({'trellis wins' if gap < 0 else 'gated wins'})",
                flush=True,
            )
        fn = out / f"rung_d{d_model}_L{n_layers}.json"
        fn.write_text(json.dumps(rung_res, indent=2))
        print(f"  wrote {fn}", flush=True)
    print("LADDER_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
