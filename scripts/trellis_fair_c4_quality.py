"""Fair C4-en quality ablation across the linear-attention family.

DeltaProduct wins the synthetic overwrite/binding grid outright, but that grid
does not answer whether its extra Householder rotations help real language
modeling. This harness settles that on loss: it trains dense, DeltaNet
(delta_ref), GatedDeltaNet (gated_delta_ref), and GatedDeltaProduct
(gated_delta_product_ref_nhN) -- optionally Trellis -- at matched width, depth,
and token budget on the SAME packed C4-en token stream, then reports held-out
C4 validation perplexity. Every arm sees the identical training tokens and the
identical validation slice, so the only free variable is the mixer.

This is the reproducible, in-tree form of the on-pod fair-linear quality driver
whose results were archived but whose script was not. It reuses build_model and
the packing logic from trellis_lm so the arms are the exact same modules the
rest of the harness trains.

  python scripts/trellis_fair_c4_quality.py \
      --arms dense,delta_ref,gated_delta_ref,gated_delta_product_ref_nh2 \
      --seeds 0,1,2 --train_tokens 20000000 --seq_len 2048 --batch 8 \
      --d_model 512 --n_layers 10 --n_heads 8 --d_head 64 --n_slots 48 \
      --lr 3e-4 --dtype bf16 --dataset allenai/c4 --dataset_config en \
      --out out/c4_quality/fair.json
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

from trellis_lm.config import TrellisConfig
from trellis_lm.model import build_model


def stream_packed(dataset, dataset_config, split, seq_len, n_seqs):
    """Pack a streaming HF text dataset into n_seqs rows of seq_len gpt2 tokens.

    Identical tokenizer/packing to trellis_lm.eval_ppl.packed so the train and
    validation streams line up with the rest of the harness.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    if dataset_config:
        ds = load_dataset(dataset, dataset_config, split=split, streaming=True)
    else:
        ds = load_dataset(dataset, split=split, streaming=True)
    buf, out = [], []
    for ex in ds:
        buf.extend(tok(ex.get("text") or "").input_ids + [tok.eos_token_id])
        while len(buf) >= seq_len:
            out.append(buf[:seq_len])
            buf = buf[seq_len:]
            if len(out) >= n_seqs:
                return tok.vocab_size, out
    return tok.vocab_size, out


def backend_of(model):
    """Best-effort human-readable mixer/backend for the results table."""
    for attr in ("blocks", "layers", "h"):
        seq = getattr(model, attr, None)
        if seq is None:
            continue
        try:
            block = seq[0]
        except (TypeError, IndexError, KeyError):
            continue
        mixer = getattr(block, "mixer", None) or getattr(block, "attn", block)
        inner = getattr(mixer, "layer", None) or getattr(mixer, "op", mixer)
        return type(inner).__module__ + "." + type(inner).__name__
    return type(model).__name__


# Our Slot-Mixing Delta arms map to a "trellis" model carrying the
# input-conditioned affine write. The gate config is the leading one that
# produced the small-C4 positive (write_mode=input_conditioned, sigmoid gate,
# silu/linear-alpha, gamma=0.05, layer0 gamma mult 0.5, write_l2norm on for
# stability); the scope (per_slot=diagonal vs scalar) is picked by the arm name.
# The exact chunk-16 kernel is numerically identical to the chunk-1 sequential
# path the anchor used but far faster at seq2048.
SMD_ARMS = {
    "smd_diag": "per_slot",
    "smd_scalar": "scalar",
}


def build_cfg(arm, args, vocab):
    """Return (cfg, kind) for an arm. Baselines pass kind=arm; smd arms build a
    trellis cfg with the input-conditioned gate and kind='trellis'."""
    if arm in SMD_ARMS:
        cfg = TrellisConfig(
            vocab_size=vocab,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_head=args.d_head,
            n_slots=args.n_slots,
            max_seq_len=args.seq_len,
            dtype=args.dtype,
            chunk_size=args.chunk_size,
            trellis_write_mode="input_conditioned",
            trellis_input_gate_act="sigmoid",
            trellis_input_gate_scope=SMD_ARMS[arm],
            write_l2norm=True,
            activation="silu",
            alpha_mode="linear",
            beta_mode="scalar_per_head",
            beta_init=0.5,
            value_readout_act="none",
            output_path="current",
            gamma_init=args.smd_gamma_init,
            trellis_layer0_gamma_mult=args.smd_layer0_gamma_mult,
        )
        return cfg, "trellis"
    cfg = TrellisConfig(
        vocab_size=vocab,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_head=args.d_head,
        n_slots=args.n_slots,
        max_seq_len=args.seq_len,
        dtype=args.dtype,
        chunk_size=args.chunk_size,
    )
    return cfg, arm


def train_arm(arm, kind, seed, cfg, train_rows, args, device, dt):
    torch.manual_seed(seed)
    model = build_model(cfg, kind).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    steps = len(train_rows) // args.batch
    t0 = time.time()
    ntok = 0
    last = float("nan")
    for step in range(steps):
        rows = train_rows[step * args.batch : (step + 1) * args.batch]
        idx = torch.tensor(rows, device=device)
        with torch.autocast(
            device_type=device.type, dtype=dt, enabled=args.dtype != "fp32"
        ):
            _, loss = model(idx, labels=idx, training=True)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ntok += idx.numel()
        last = loss.item()
        if not math.isfinite(last):
            return model, {"status": "diverged", "step": step, "loss": last}
        if step % args.log_every == 0 or step == steps - 1:
            tps = ntok / (time.time() - t0)
            print(
                f"    [{arm} s{seed}] step {step:5d}/{steps}  "
                f"loss {last:.4f}  ppl {math.exp(min(20, last)):.2f}  "
                f"tok/s {tps:.0f}",
                flush=True,
            )
    return model, {
        "status": "ok",
        "train_loss": round(last, 5),
        "train_tok_s": round(ntok / (time.time() - t0), 1),
    }


@torch.no_grad()
def val_ppl(model, val_rows, args, device, dt):
    model.eval()
    nll, ntok = 0.0, 0
    for i in range(0, len(val_rows), args.batch):
        idx = torch.tensor(val_rows[i : i + args.batch], device=device)
        with torch.autocast(
            device_type=device.type, dtype=dt, enabled=args.dtype != "fp32"
        ):
            _, loss = model(idx, labels=idx, training=False)
        n = idx.numel() - idx.shape[0]
        nll += loss.item() * n
        ntok += n
    ce = nll / max(1, ntok)
    return round(ce, 5), round(math.exp(min(20, ce)), 4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--arms",
        default="dense,delta_ref,gated_delta_ref,gated_delta_product_ref_nh2",
    )
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--train_tokens", type=int, default=20_000_000)
    p.add_argument("--val_seqs", type=int, default=128)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=10)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_head", type=int, default=64)
    p.add_argument("--n_slots", type=int, default=48)
    p.add_argument("--chunk_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    # Slot-Mixing Delta stability recipe (validated gate window); only used by
    # the smd_* arms.
    p.add_argument("--smd_gamma_init", type=float, default=0.05)
    p.add_argument("--smd_layer0_gamma_mult", type=float, default=0.5)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--log_every", type=int, default=250)
    p.add_argument("--dataset", default="allenai/c4")
    p.add_argument("--dataset_config", default="en")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.dtype
    ]
    arms = [a for a in args.arms.split(",") if a]
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]

    n_train_seqs = args.train_tokens // args.seq_len
    print(
        f"packing {n_train_seqs} train seqs + {args.val_seqs} val seqs "
        f"@ seq_len={args.seq_len} from {args.dataset}/{args.dataset_config}",
        flush=True,
    )
    vocab, train_rows = stream_packed(
        args.dataset, args.dataset_config, "train", args.seq_len, n_train_seqs
    )
    _, val_rows = stream_packed(
        args.dataset,
        args.dataset_config,
        "validation",
        args.seq_len,
        args.val_seqs,
    )
    print(
        f"train_rows={len(train_rows)} ({len(train_rows) * args.seq_len:,} tok) "
        f"val_rows={len(val_rows)} vocab={vocab}",
        flush=True,
    )

    def flush_results(results):
        payload = {"args": vars(args), "results": results}
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(payload, indent=2))

    results = []
    for arm in arms:
        for seed in seeds:
            cfg, kind = build_cfg(arm, args, vocab)
            try:
                model, tr = train_arm(
                    arm, kind, seed, cfg, train_rows, args, device, dt
                )
                row = {
                    "arm": arm,
                    "seed": seed,
                    "params": model.get_num_params(),
                    "mem_state_bytes_per_seq": model.memory_state_bytes(1),
                    "backend": backend_of(model),
                    **tr,
                }
                if tr["status"] == "ok":
                    ce, ppl = val_ppl(model, val_rows, args, device, dt)
                    row["val_nll"] = ce
                    row["val_ppl"] = ppl
                    print(
                        f"  == {arm} s{seed}: val_ppl={ppl} "
                        f"(nll={ce}) train_tok_s={tr['train_tok_s']} "
                        f"state_bytes={row['mem_state_bytes_per_seq']}",
                        flush=True,
                    )
                else:
                    print(
                        f"  == {arm} s{seed}: {tr['status']} "
                        f"@ step {tr.get('step')}",
                        flush=True,
                    )
                del model
            except Exception as e:  # keep the sweep alive; record the failure
                row = {"arm": arm, "seed": seed, "status": "error", "error": str(e)}
                print(f"  == {arm} s{seed}: ERROR {e}", flush=True)
            results.append(row)
            # Persist after every cell so a crash/preemption never loses a
            # completed arm (the training-pod checkpoint-as-you-go rule).
            flush_results(results)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not args.out:
        print(json.dumps({"args": vars(args), "results": results}, indent=2))
    else:
        print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
