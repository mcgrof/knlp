"""Trellis reopening ladder: labeled-semantics C4 comparison driver.

Trains Trellis arms under EXPLICIT training semantics against the fla
reference baselines on one shared packed C4 token stream, and writes a
provenance manifest per cell so no result can be ambiguous about what it
trained. This is a new driver on purpose: the historical ladder's
commands keep their historical meaning.

Trellis arms derive from a named research profile (--trellis_profile,
required whenever a trellis arm runs); the arm name then pins the axes
that vary:

  trellis_repaired_full         profile recipe, full_bilevel
  trellis_repaired_first_order  profile recipe, first_order_detached
  trellis_identity_full         profile recipe, phi=identity (f held at
                                the profile's f), full_bilevel
  gated_delta_ref / delta_ref / dense   fla reference / dense baselines

Every cell records: git commit, profile, resolved state and gradient
modes, the backend the mixer ACTUALLY selected (asserted against the
request before the first optimizer step), phi/f/alpha/readout/output
knobs, beta and gamma parameterization, chunk size, parameter counts,
recurrent-state bytes, tokenizer/dataset with packed-stream hashes,
sequence length, token budget, seed, precision, device, and package
versions. Milestone evals record the crossing curve plus per-layer
state/update diagnostics and gradient norms; a cell aborts on nonfinite
loss, parameters, or gradients.

  python scripts/trellis_reopen_ladder.py \
      --arms trellis_repaired_full,trellis_repaired_first_order,\
trellis_identity_full,gated_delta_ref \
      --trellis_profile repaired_silu_full --chunk_size 16 \
      --seeds 0 --train_tokens 20000000 --seq_len 2048 --batch 4 \
      --d_model 256 --n_layers 4 --n_heads 4 --d_head 64 \
      --eval_milestones 2000000,5000000,10000000,20000000 \
      --out out/trellis_reopen/r0.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from trellis_lm.config import TrellisConfig
from trellis_lm.model import build_model
from trellis_fair_c4_quality import stream_packed, val_ppl

TRELLIS_ARMS = {
    "trellis_repaired_full": {"trellis_outer_gradient_mode": "full_bilevel"},
    "trellis_repaired_first_order": {
        "trellis_outer_gradient_mode": "first_order_detached"
    },
    "trellis_identity_full": {
        "trellis_outer_gradient_mode": "full_bilevel",
        "phi_activation": "identity",
    },
    # matched-mode write-only control: identity under the SAME first-order
    # gradient as the historical candidate, so phi is the only difference
    "trellis_identity_first_order": {
        "trellis_outer_gradient_mode": "first_order_detached",
        "phi_activation": "identity",
    },
}
BASELINE_ARMS = ("dense", "delta_ref", "gated_delta_ref", "delta")


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def rows_sha256(rows):
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r).encode())
    return h.hexdigest()


def package_versions():
    out = {"python": sys.version.split()[0], "torch": torch.__version__}
    for mod in ("transformers", "datasets", "fla", "triton"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def build_arm_cfg(arm, args, vocab):
    dims = dict(
        vocab_size=vocab,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_head=args.d_head,
        max_seq_len=args.seq_len,
        dtype=args.dtype,
    )
    if arm in TRELLIS_ARMS:
        if not args.trellis_profile:
            raise SystemExit(
                f"arm {arm} requires --trellis_profile (named profiles only; "
                "this driver never inherits silent semantics defaults)"
            )
        overrides = {}
        for kv in args.cfg or []:
            key, _, val = kv.partition("=")
            try:
                overrides[key] = json.loads(val)
            except json.JSONDecodeError:
                overrides[key] = val
        cfg = TrellisConfig.profile(
            args.trellis_profile,
            chunk_size=args.chunk_size,
            **dims,
            **TRELLIS_ARMS[arm],
            **overrides,
        )
        return cfg, "trellis"
    if arm not in BASELINE_ARMS:
        raise SystemExit(f"unknown arm {arm}")
    return TrellisConfig(n_slots=args.n_slots, chunk_size=1, **dims), arm


def non_embedding_params(model):
    emb = sum(
        p.numel()
        for name, p in model.named_parameters()
        if "wte" in name or "lm_head" in name
    )
    total = sum(p.numel() for p in model.parameters())
    return total - emb


def cell_manifest(arm, cfg, kind, model, seed, args, hashes, device):
    man = {
        "git_commit": git_commit(),
        "arm": arm,
        "profile": args.trellis_profile if kind == "trellis" else None,
        "seed": seed,
        "seq_len": args.seq_len,
        "batch": args.batch,
        "train_tokens": args.train_tokens,
        "lr": args.lr,
        "precision": args.dtype,
        "device": (torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"),
        "tokenizer": "openai-community/gpt2",
        "dataset": f"{args.dataset}/{args.dataset_config}",
        "train_rows_sha256": hashes["train"],
        "val_rows_sha256": hashes["val"],
        "params_total": sum(p.numel() for p in model.parameters()),
        "params_reported": model.get_num_params(),
        "params_non_embedding": non_embedding_params(model),
        "recurrent_state_bytes": model.memory_state_bytes(1),
        "packages": package_versions(),
    }
    if kind == "trellis":
        sem = cfg.resolve_training_semantics(warn=False)
        man.update(
            {
                "state_mode": sem["state_mode"],
                "outer_gradient_mode": sem["outer_gradient_mode"],
                "phi": cfg.phi_activation or cfg.activation,
                "f": cfg.f_activation or cfg.activation,
                "alpha_mode": cfg.alpha_mode,
                "value_readout_act": cfg.value_readout_act,
                "output_path": cfg.output_path,
                "beta_mode": cfg.beta_mode,
                "beta_init": cfg.beta_init,
                "retention_mode": cfg.trellis_retention_mode,
                "gamma_init": cfg.gamma_init,
                "chunk_size": cfg.chunk_size,
                "n_slots": cfg.n_slots,
                "stabilizer": cfg.trellis_update_stabilizer,
                "residual_update_mix": cfg.residual_update_mix,
            }
        )
    return man


def trellis_backend(model):
    diag = getattr(model.blocks[0].mixer, "last_trellis_diag", None)
    return diag.get("backend") if diag else None


def assert_semantics(model, manifest, idx0):
    """One no-grad forward, then abort if the mixer's selected backend does
    not honor the manifest's gradient mode. Runs BEFORE the first optimizer
    step: a manifest may never claim semantics the backend cannot deliver."""
    with torch.no_grad():
        model(idx0, training=False)
    backend = trellis_backend(model)
    manifest["backend"] = backend
    want = manifest["outer_gradient_mode"]
    ok = (
        (want == "full_bilevel" and backend.endswith("full_bilevel"))
        or (want == "first_order_detached" and backend.endswith("first_order"))
        or (want == "exact_affine" and "affine" in backend)
    )
    if not ok:
        raise RuntimeError(
            f"semantic mismatch: manifest requests {want} but the mixer "
            f"selected backend {backend}"
        )


def layer_diagnostics(model):
    out = []
    for blk in getattr(model, "blocks", []):
        diag = getattr(getattr(blk, "mixer", None), "last_trellis_diag", None)
        if not diag:
            continue
        row = {"layer": diag.get("layer"), "backend": diag.get("backend")}
        for key in ("key_state", "value_state", "key_update", "value_update", "beta"):
            stats = diag.get(key)
            if isinstance(stats, dict):
                row[key] = {k: stats[k] for k in ("rms", "absmax") if k in stats}
        out.append(row)
    return out


def grad_norm_and_finiteness(model):
    total = 0.0
    bad_grads = bad_params = 0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad
            if not torch.isfinite(g).all():
                bad_grads += 1
            else:
                total += g.float().pow(2).sum().item()
        if not torch.isfinite(p).all():
            bad_params += 1
    return math.sqrt(total), bad_grads, bad_params


def train_cell(arm, kind, seed, cfg, rows, val_rows, args, device, dt, hashes):
    torch.manual_seed(seed)
    model = build_model(cfg, kind).to(device)
    manifest = cell_manifest(arm, cfg, kind, model, seed, args, hashes, device)
    if kind == "trellis":
        idx0 = torch.tensor(rows[: args.batch], device=device)
        assert_semantics(model, manifest, idx0)
    else:
        manifest["backend"] = type(model).__name__
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    steps = min(
        len(rows) // args.batch,
        -(-args.train_tokens // (args.batch * args.seq_len)),
    )
    milestones = sorted(int(m) for m in args.eval_milestones.split(",") if m.strip())
    curve, mi, ntok, t0 = [], 0, 0, time.time()
    for step in range(steps):
        idx = torch.tensor(
            rows[step * args.batch : (step + 1) * args.batch], device=device
        )
        with torch.autocast(
            device_type=device.type, dtype=dt, enabled=args.dtype != "fp32"
        ):
            _, loss = model(idx, labels=idx, training=True)
        opt.zero_grad()
        loss.backward()
        gnorm, bad_grads, bad_params = grad_norm_and_finiteness(model)
        if not math.isfinite(loss.item()) or bad_grads or bad_params:
            return (
                model,
                manifest,
                {
                    "status": "diverged",
                    "step": step,
                    "loss": loss.item(),
                    "nonfinite_grad_tensors": bad_grads,
                    "nonfinite_param_tensors": bad_params,
                    "diag": layer_diagnostics(model),
                    "curve": curve,
                },
            )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ntok += idx.numel()
        while mi < len(milestones) and ntok >= milestones[mi]:
            ce, ppl = val_ppl(model, val_rows, args, device, dt)
            curve.append(
                {
                    "tokens": ntok,
                    "val_nll": ce,
                    "val_ppl": ppl,
                    "grad_norm": round(gnorm, 4),
                    "diag": layer_diagnostics(model),
                }
            )
            print(
                f"    [{arm} s{seed}] {ntok:,} tok: val_ppl={ppl} nll={ce} "
                f"gnorm={gnorm:.2f}",
                flush=True,
            )
            model.train()
            mi += 1
        if step % args.log_every == 0 or step == steps - 1:
            tps = ntok / max(1e-9, time.time() - t0)
            print(
                f"    [{arm} s{seed}] step {step:5d}/{steps} "
                f"loss {loss.item():.4f} tok/s {tps:.0f}",
                flush=True,
            )
    ce, ppl = val_ppl(model, val_rows, args, device, dt)
    return (
        model,
        manifest,
        {
            "status": "ok",
            "val_nll": ce,
            "val_ppl": ppl,
            "train_tok_s": round(ntok / max(1e-9, time.time() - t0), 1),
            "curve": curve,
        },
    )


def paired_summary(results, milestones):
    """Per-arm mean val_nll at each milestone plus the reopening's decision
    quantities where the needed arms completed. Per-seed values stay in the
    rows; interval analysis happens offline."""

    def arm_nll(arm, tokens):
        vals = [
            next(
                (pt["val_nll"] for pt in r["curve"] if pt["tokens"] >= tokens),
                None,
            )
            for r in results
            if r.get("arm") == arm and r.get("status") == "ok"
        ]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {}
    for tokens in milestones:
        row = {}
        for arm in {r.get("arm") for r in results}:
            m = arm_nll(arm, tokens)
            if m is not None:
                row[arm] = round(m, 5)
        full = row.get("trellis_repaired_full")
        first = row.get("trellis_repaired_first_order")
        ident = row.get("trellis_identity_full")
        gdn = row.get("gated_delta_ref")
        if full is not None and ident is not None:
            row["nonlinear_gain_nats"] = round(ident - full, 5)
        if full is not None and first is not None:
            row["bilevel_minus_first_order_nats"] = round(full - first, 5)
        if full is not None and gdn is not None:
            row["gap_to_gdn_nats"] = round(full - gdn, 5)
        summary[str(tokens)] = row
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--arms",
        default="trellis_repaired_full,trellis_repaired_first_order,"
        "trellis_identity_full,gated_delta_ref",
    )
    p.add_argument(
        "--trellis_profile",
        default=None,
        choices=sorted(TrellisConfig._PROFILES),
        help="required for trellis arms; semantics are never defaulted here",
    )
    p.add_argument("--seeds", default="0")
    p.add_argument("--train_tokens", type=int, default=20_000_000)
    p.add_argument("--val_seqs", type=int, default=128)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_head", type=int, default=64)
    p.add_argument("--n_slots", type=int, default=48)
    p.add_argument("--chunk_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval_milestones", default="2000000,5000000,10000000,20000000")
    p.add_argument(
        "--cfg",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="extra TrellisConfig override for the trellis arms (repeatable; "
        "values parsed as JSON, else kept as strings); recorded in the "
        "manifest like every other knob",
    )
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--log_every", type=int, default=250)
    p.add_argument("--dataset", default="allenai/c4")
    p.add_argument("--dataset_config", default="en")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.dtype
    ]
    arms = [a for a in args.arms.split(",") if a]
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    milestones = [int(m) for m in args.eval_milestones.split(",") if m.strip()]

    # round the packed stream UP to a whole number of batches covering the
    # token budget, so the final milestone is always reachable (floor packing
    # plus a floored step cap left a 10M request at 9.99M trained tokens)
    steps_target = -(-args.train_tokens // (args.batch * args.seq_len))
    n_train_seqs = steps_target * args.batch
    print(
        f"packing {n_train_seqs} train + {args.val_seqs} val seqs "
        f"@ seq_len={args.seq_len} from {args.dataset}/{args.dataset_config}",
        flush=True,
    )
    vocab, train_rows = stream_packed(
        args.dataset, args.dataset_config, "train", args.seq_len, n_train_seqs
    )
    _, val_rows = stream_packed(
        args.dataset, args.dataset_config, "validation", args.seq_len, args.val_seqs
    )
    hashes = {"train": rows_sha256(train_rows), "val": rows_sha256(val_rows)}
    print(
        f"train_rows={len(train_rows)} val_rows={len(val_rows)} "
        f"train_sha={hashes['train'][:16]} val_sha={hashes['val'][:16]}",
        flush=True,
    )

    results = []

    def flush():
        payload = {
            "args": vars(args),
            "results": results,
            "summary": paired_summary(results, milestones),
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))

    for arm in arms:
        for seed in seeds:
            cfg, kind = build_arm_cfg(arm, args, vocab)
            try:
                model, manifest, tr = train_cell(
                    arm,
                    kind,
                    seed,
                    cfg,
                    train_rows,
                    val_rows,
                    args,
                    device,
                    dt,
                    hashes,
                )
                row = {"arm": arm, "seed": seed, "manifest": manifest, **tr}
                print(
                    f"  == {arm} s{seed}: {tr['status']} "
                    f"val_ppl={tr.get('val_ppl')} backend={manifest.get('backend')}",
                    flush=True,
                )
                del model
            except Exception as e:
                row = {"arm": arm, "seed": seed, "status": "error", "error": str(e)}
                print(f"  == {arm} s{seed}: ERROR {e}", flush=True)
            results.append(row)
            flush()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
