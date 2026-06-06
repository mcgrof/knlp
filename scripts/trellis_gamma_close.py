"""Close out the Trellis inner-lr (gamma) question against Gated DeltaNet.

Context. The matched-scratch comparison (reports/trellis_full_eval.md §13)
left one load-bearing caveat: our Trellis underperforms its linear cousins
partly because the inner-loop step size gamma -- the one knob only Trellis
has, and one the paper never specifies -- was left untuned at 1e-2. The
gamma-sweep-20260605 cell already showed tuning helps and de-noises (chunk16
L2048: 0.01 -> 521.9 +/- 25, 0.1 -> 487.5 +/- 2.2) but had not bottomed out.
This driver finishes that sweep so the verdict is airtight.

Everything matches the gamma-sweep cell exactly: PG19 L2048, 400 steps,
batch 4, seeds {0,1,2}, d256/4L/4H/d64/64slots, lr 3e-3. The PG19 pack is
built once and reused across every gamma and both operators, so the train/val
token streams are byte-identical throughout (a tighter control than the
original sweep, which repacked per gamma).

Phase 1 -- chunk16 (cheap, ~2h). Extend the inner-lr sweep to
gamma in {0.2, 0.3, 0.5, 1.0}. The question: does tuning gamma push Trellis
below Gated DeltaNet (423) in the operator that already FLATTERS Trellis
(chunk16 trains to lower ppl than seq here)? A loss in chunk16 is the
conservative statement, since seq is strictly worse for Trellis.

Phase 2 -- seq (faithful, ~5h). One anchor at the best gamma from the full
chunk16 curve, run in the slow exact-sequential operator, to slot a tuned
Trellis number into the seq-mode table. Combined with the existing
seq gamma=0.01 point (584.5) this gives the seq trend at tuned gamma.

Baselines are NOT recomputed: dense and the linear cousins (DeltaNet /
Gated DeltaNet) ignore chunk_size/exact_inner in their forward, so they are
mode-invariant -- the existing seq numbers (dense ~621, delta 502, gated 423)
already ARE the matched baselines, and chunk16 dense (616) is on file too.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import trellis_firmup as tf  # reuse the exact train/eval/cfg path

# --- fixed cell matching gamma-sweep-20260605 -------------------------------
SEQ_LEN = 2048
STEPS = 400
BATCH = 4
SEEDS = [0, 1, 2]
LR = 3e-3
DIMS = dict(d_model=256, n_layers=4, n_heads=4, d_head=64, n_slots=64)
N_VAL = 48

CHUNK16_NEW_GAMMAS = [0.2, 0.3, 0.5, 1.0]
# existing chunk16 gammas already on disk (read in, not recomputed)
PRIOR_SWEEP = Path("/data/knlp-key-results/trellis-lm/gamma-sweep-20260605")
OUT = Path("/data/knlp-key-results/trellis-lm/gamma-close-20260605")

# matched baselines already measured (mode-invariant for dense/linear)
BASELINES = {
    "seq": {"dense": 620.9, "delta": 502.1, "gated_delta": 423.2},
    "chunk16": {"dense": 615.7, "delta": 502.1, "gated_delta": 423.2},
}


def _args(mode, gamma):
    return SimpleNamespace(
        dataset="pg19",
        seq_len=SEQ_LEN,
        mode=mode,
        seeds=SEEDS,
        models=["trellis"],
        steps=STEPS,
        batch=BATCH,
        lr=LR,
        gamma_init=gamma,
        n_val=N_VAL,
        log_every=100,
        out=None,
        **DIMS,
    )


def _agg(ppls):
    mean = sum(ppls) / len(ppls)
    std = (sum((x - mean) ** 2 for x in ppls) / len(ppls)) ** 0.5
    return {"val_ppl_mean": mean, "val_ppl_std": std, "n": len(ppls), "ppls": ppls}


def run_cell(mode, gamma, train_rows, val_rows, device, vocab):
    """Train 3 seeds of Trellis at (mode, gamma); return per-run + aggregate."""
    out_path = OUT / f"{mode}_gamma_{gamma}.json"
    if out_path.exists():
        d = json.loads(out_path.read_text())
        if d.get("final"):
            print(
                f"[skip] {mode} gamma={gamma} already final "
                f"({d['aggregate']['trellis']['val_ppl_mean']:.1f})",
                flush=True,
            )
            return d
    args = _args(mode, gamma)
    cfg = tf.make_cfg(vocab, SEQ_LEN, mode, args)
    runs = []
    for seed in SEEDS:
        try:
            r = tf.train_one("trellis", seed, cfg, train_rows, val_rows, args, device)
        except RuntimeError as e:  # OOM / numerical -- log and continue
            print(f"  !! {mode} gamma={gamma} seed{seed} FAILED: {e}", flush=True)
            torch.cuda.empty_cache()
            continue
        print(
            f"  -> {mode} gamma={gamma} seed{seed} val_ppl "
            f"{r['val_ppl']:.2f} ({r['train_min']:.1f} min)",
            flush=True,
        )
        runs.append(r)
        _dump(out_path, mode, gamma, runs, final=False)
    _dump(out_path, mode, gamma, runs, final=True)
    return json.loads(out_path.read_text())


def _dump(path, mode, gamma, runs, final):
    ppls = [r["val_ppl"] for r in runs]
    path.write_text(
        json.dumps(
            {
                "dataset": "pg19",
                "seq_len": SEQ_LEN,
                "mode": mode,
                "steps": STEPS,
                "batch": BATCH,
                "lr": LR,
                "gamma_init": gamma,
                "seeds": SEEDS,
                "dims": DIMS,
                "runs": runs,
                "aggregate": {"trellis": _agg(ppls)} if ppls else {},
                "final": final,
            },
            indent=2,
        )
    )


def load_prior_chunk16():
    """Read the existing chunk16 gamma points {0.001,0.01,0.03,0.1}."""
    pts = {}
    for f in sorted(PRIOR_SWEEP.glob("gamma_*.json")):
        d = json.loads(f.read_text())
        if d.get("final") and "trellis" in d.get("aggregate", {}):
            pts[d["gamma_init"]] = d["aggregate"]["trellis"]["val_ppl_mean"]
    return pts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device: {device} "
        f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})",
        flush=True,
    )

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    vocab = tok.vocab_size
    n_train = STEPS * BATCH + BATCH
    print(
        f"[pg19 L{SEQ_LEN}] packing {n_train} train + {N_VAL} val " f"rows once ...",
        flush=True,
    )
    train_rows, val_rows = tf.load_packed("pg19", SEQ_LEN, n_train, N_VAL, tok)
    print(f"  packed train={len(train_rows)} val={len(val_rows)}", flush=True)

    # ---- Phase 1: chunk16 inner-lr extension (decisive, cheap) ----
    print("\n=== Phase 1: chunk16 gamma extension ===", flush=True)
    curve = dict(load_prior_chunk16())
    for g in CHUNK16_NEW_GAMMAS:
        d = run_cell("chunk16", g, train_rows, val_rows, device, vocab)
        if d.get("aggregate"):
            curve[g] = d["aggregate"]["trellis"]["val_ppl_mean"]
    print(
        f"\nchunk16 curve (gamma -> mean ppl): "
        f"{ {k: round(v, 1) for k, v in sorted(curve.items())} }",
        flush=True,
    )
    best_gamma = min(curve, key=curve.get)
    print(
        f"best chunk16 gamma = {best_gamma} ({curve[best_gamma]:.1f} ppl); "
        f"gated_delta baseline = {BASELINES['chunk16']['gated_delta']}",
        flush=True,
    )

    # ---- Phase 2: seq anchor at the best gamma (faithful operator) ----
    print(f"\n=== Phase 2: seq anchor at gamma={best_gamma} ===", flush=True)
    seq_anchor = run_cell("seq", best_gamma, train_rows, val_rows, device, vocab)

    # ---- summary ----
    summary = {
        "cell": {
            "dataset": "pg19",
            "seq_len": SEQ_LEN,
            "steps": STEPS,
            "batch": BATCH,
            "seeds": SEEDS,
            "lr": LR,
            "dims": DIMS,
        },
        "chunk16_trellis_curve": {
            str(k): round(v, 2) for k, v in sorted(curve.items())
        },
        "best_chunk16_gamma": best_gamma,
        "seq_anchor_gamma": best_gamma,
        "seq_trellis_tuned": (
            seq_anchor.get("aggregate", {}).get("trellis", {}).get("val_ppl_mean")
        ),
        "seq_trellis_gamma0p01_existing": 584.5,
        "baselines": BASELINES,
        "verdict_closes_gap": (curve[best_gamma] < BASELINES["chunk16"]["gated_delta"]),
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nwrote {OUT}/SUMMARY.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
