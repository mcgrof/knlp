"""Paired-arm sweep driver for the tool-router semantics gate.

Runs scripts/trellis_tool_router.py across a fixed difficulty ladder,
write-phi set, and both outer-gradient modes with PAIRED seeds (equal
seed = identical episode stream per arm), then summarizes the decision
quantities: per-seed full-minus-detached deltas per phi, the
difference-of-differences against the identity write, and the
promotion-gate checks.

Difficulty cells:

  easy      tools=6   bindings=4   overwrites=1  slots=32   (surplus)
  medium    tools=12  bindings=8   overwrites=3  slots=32   (surplus)
  hard      tools=24  bindings=16  overwrites=5  slots=32   (surplus)
  pressure  tools=12  bindings=64  overwrites=5  slots=16   (4x oversubscribed)

The surplus cells are correctness probes (bind/overwrite/retrieve with
slot headroom). The pressure cell oversubscribes memory so retrieval
REQUIRES lossy compression -- the regime where the write nonlinearity
could earn its keep; phi conclusions about capacity come from this
cell, not the surplus ones.

  python scripts/trellis_router_gate.py run --outdir OUT --steps 300 \
      --seeds 0,1,2,3,4 --cells easy,medium,hard,pressure
  python scripts/trellis_router_gate.py summarize --outdir OUT
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTER = ROOT / "scripts" / "trellis_tool_router.py"

CELLS = {
    "easy": dict(tools=6, bindings=4, overwrites=1, slots=32),
    "medium": dict(tools=12, bindings=8, overwrites=3, slots=32),
    "hard": dict(tools=24, bindings=16, overwrites=5, slots=32),
    "pressure": dict(tools=12, bindings=64, overwrites=5, slots=16),
}
PHIS = ("identity", "silu", "ln_silu")
MODES = ("full_bilevel", "first_order_detached")


def run_name(cell, phi, mode, seed):
    return f"{cell}_{phi}_{mode}_s{seed}"


def cmd_run(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    cells = [c for c in args.cells.split(",") if c]
    py = sys.executable
    total = len(cells) * len(PHIS) * len(MODES) * len(seeds)
    done = 0
    for cell in cells:
        knobs = CELLS[cell]
        for phi in PHIS:
            for mode in MODES:
                for seed in seeds:
                    name = run_name(cell, phi, mode, seed)
                    jsonl = outdir / f"{name}.jsonl"
                    done += 1
                    if jsonl.exists() and args.resume:
                        print(f"[{done}/{total}] {name}: exists, skip", flush=True)
                        continue
                    cmd = [
                        py,
                        str(ROUTER),
                        "--device",
                        args.device,
                        "--steps",
                        str(args.steps),
                        "--batch-size",
                        str(args.batch_size),
                        "--eval-batches",
                        str(args.eval_batches),
                        "--log-every",
                        str(args.log_every),
                        "--seed",
                        str(seed),
                        "--phi",
                        phi,
                        "--outer-gradient-mode",
                        mode,
                        "--tools",
                        str(knobs["tools"]),
                        "--bindings",
                        str(knobs["bindings"]),
                        "--overwrites",
                        str(knobs["overwrites"]),
                        "--slots",
                        str(knobs["slots"]),
                        "--jsonl",
                        str(jsonl),
                        "--output",
                        str(outdir / f"{name}.pt"),
                    ]
                    if seed == seeds[0]:
                        cmd.append("--assert-step0-equivalence")
                    t0 = time.time()
                    r = subprocess.run(cmd, capture_output=True, text=True)
                    dt = time.time() - t0
                    status = "ok" if r.returncode == 0 else f"rc={r.returncode}"
                    print(f"[{done}/{total}] {name}: {status} {dt:.0f}s", flush=True)
                    if r.returncode != 0:
                        (outdir / f"{name}.stderr").write_text(r.stderr)
    return 0


def load_run(outdir, cell, phi, mode, seed):
    p = Path(outdir) / f"{run_name(cell, phi, mode, seed)}.jsonl"
    if not p.exists():
        return None
    train, final = [], None
    for line in p.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("kind") == "train":
            train.append(rec)
        elif rec.get("kind") == "eval":
            final = rec
    if final is None:
        return None
    auc = statistics.mean(r["overwrite"] for r in train) if train else float("nan")
    steps90 = next((r["step"] for r in train if r["overwrite"] >= 0.90), None)
    return {
        "final": final,
        "overwrite_auc": auc,
        "steps_to_090": steps90,
    }


def cmd_summarize(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    cells = [c for c in args.cells.split(",") if c]
    summary = {}
    for cell in cells:
        cs = {}
        # per-seed paired deltas of the primary metric (final overwrite acc)
        deltas = {}
        for phi in PHIS:
            rows = {
                mode: [load_run(args.outdir, cell, phi, mode, s) for s in seeds]
                for mode in MODES
            }
            pairs = [
                (a, b)
                for a, b in zip(rows["full_bilevel"], rows["first_order_detached"])
                if a is not None and b is not None
            ]
            if not pairs:
                continue
            d_over = [
                a["final"]["overwrite"] - b["final"]["overwrite"] for a, b in pairs
            ]
            d_auc = [a["overwrite_auc"] - b["overwrite_auc"] for a, b in pairs]
            d_coll = [
                a["final"]["collateral"] - b["final"]["collateral"] for a, b in pairs
            ]
            deltas[phi] = d_over
            cs[phi] = {
                "n_pairs": len(pairs),
                "full_overwrite": [round(a["final"]["overwrite"], 4) for a, _ in pairs],
                "detached_overwrite": [
                    round(b["final"]["overwrite"], 4) for _, b in pairs
                ],
                "delta_overwrite": [round(d, 4) for d in d_over],
                "delta_overwrite_mean": round(statistics.mean(d_over), 4),
                "delta_auc_mean": round(statistics.mean(d_auc), 4),
                "delta_collateral_mean": round(statistics.mean(d_coll), 4),
                "full_wins": sum(1 for d in d_over if d > 0),
                "steps_to_090_full": [a["steps_to_090"] for a, _ in pairs],
                "steps_to_090_detached": [b["steps_to_090"] for _, b in pairs],
            }
        for phi in ("silu", "ln_silu"):
            if phi in deltas and "identity" in deltas:
                n = min(len(deltas[phi]), len(deltas["identity"]))
                dd = [deltas[phi][i] - deltas["identity"][i] for i in range(n)]
                cs[f"diff_of_diffs_{phi}_minus_identity"] = {
                    "per_seed": [round(d, 4) for d in dd],
                    "mean": round(statistics.mean(dd), 4) if dd else None,
                }
        summary[cell] = cs
    out = json.dumps(summary, indent=2)
    print(out)
    if args.out:
        Path(args.out).write_text(out)
    return 0


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--outdir", required=True)
    r.add_argument("--steps", type=int, default=300)
    r.add_argument("--batch-size", type=int, default=128)
    r.add_argument("--eval-batches", type=int, default=100)
    r.add_argument("--log-every", type=int, default=25)
    r.add_argument("--seeds", default="0,1,2,3,4")
    r.add_argument("--cells", default="easy,medium,hard,pressure")
    r.add_argument("--device", default="cuda")
    r.add_argument("--resume", action="store_true")
    s = sub.add_parser("summarize")
    s.add_argument("--outdir", required=True)
    s.add_argument("--seeds", default="0,1,2,3,4")
    s.add_argument("--cells", default="easy,medium,hard,pressure")
    s.add_argument("--out", default=None)
    args = p.parse_args()
    return cmd_run(args) if args.cmd == "run" else cmd_summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
