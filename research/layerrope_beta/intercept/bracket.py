# SPDX-License-Identifier: MIT
"""Bracket each method's learning rate at full length, then confirm at its own best rate.

The intercept study held every learning rate at the earlier study's pilot
choice. Both LayerRoPE arms had chosen 0.01, the top of that grid, so their
optimum was never bracketed. This follow-up sweeps each compared method's
rate by factors of two on one seed, at the full 2,000 updates, until the
lowest primary CE has a worse rate on both sides. It then trains three
paired seeds of each method at its own best rate. It also retrains the
earlier study's seeds for residual and plain LayerRoPE at beta = 0 to test
whether a seed draw or the hardware explains that row's shift.
"""

import argparse
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from research.layerrope_beta.intercept.analyze import MIN_GAIN, contrast, endpoint
from research.layerrope_beta.intercept.analyze import summarize
from research.layerrope_beta.intercept.plan import job, primary, row, tag
from research.layerrope_beta.intercept.run import REPO, SOURCES, verify, wave
from research.mo_variants.stage_c_common import read, sha, write

START = {"R": 0.003, "M": 0.003, "L": 0.01, "ML": 0.01}
WINNERS = {"L": (-0.5, 0.0), "ML": (-0.1, 0.0)}
MAX_EXTENSIONS = 3
BRACKET_SEED = 77911
CONFIRM_SEEDS = (77912, 77913, 77914)
PREDECESSOR_SEEDS = (77311, 77312, 77313)
# Published three-seed primary CE of the earlier fixed-token study.
PREDECESSOR = {"R": 2.690108, "L-a-0p0-beta-0p0": 2.651701}


def bases():
    rows = [row("R"), row("M")]
    for arm in ("L", "ML"):
        rows += [row(arm, 0.0, 0.0), row(arm, *WINNERS[arm])]
    return rows


def rate(arm, k):
    return float(f"{START[arm] * 2.0**k:.6g}")


def at(base, k):
    r = dict(base, lr=rate(base["arm"], k), k=k, base=base["id"])
    r["id"] = f"{base['id']}-lr-{tag(r['lr'])}"
    return r


def step(values):
    """Return the best ladder index, whether it is bracketed, and the next index to try."""
    score = {k: math.inf if v is None else v for k, v in values.items()}
    tried = sorted(score)
    best = min(tried, key=lambda k: (score[k], abs(k)))
    if math.isinf(score[best]):
        best = tried[0]
    if best == tried[0]:
        k = best - 1
    elif best == tried[-1]:
        k = best + 1
    else:
        return best, True, None
    return best, False, k if abs(k) <= 1 + MAX_EXTENSIONS else None


def freeze():
    return dict(
        schema="layerrope-learning-rate-bracket.v1",
        frozen_before_outcomes=True,
        question="With each method's learning rate bracketed at full length, do tuned LayerRoPE intercepts beat the a=0 setting, and how do residual, mHC and both LayerRoPE arms rank?",
        rows=[b["id"] for b in bases()],
        winners_from="Intercept study screen at fixed rates: LayerRoPE a=-0.5/beta=0, mHC + LayerRoPE a=-0.1/beta=0",
        ladder="rate = start * 2**k; start R/M 0.003, L/ML 0.01; first k in {-1, 0, 1}",
        bracket=dict(
            seed=BRACKET_SEED,
            updates=2000,
            measurement="Curve CE every 100 updates and final clean CE",
            rule=f"Best rate = lowest primary CE; nonfinite or incomplete counts as worse; extend one step past whichever edge is best, at most {MAX_EXTENSIONS} steps beyond the first three; bracketed only when both neighbours ran and are worse",
        ),
        confirmation=dict(
            seeds=list(CONFIRM_SEEDS),
            rows="Every row at its own best rate",
            measurement="Full-gain study endpoints: distributions, 51 KV cache cases and floating/integer interventions",
        ),
        predecessor_check=dict(
            seeds=list(PREDECESSOR_SEEDS),
            rows="R at 0.003 and LayerRoPE a=0/beta=0 at 0.01, the earlier study's rates",
            published_primary=PREDECESSOR,
            reading="Within 0.015 of the published mean: the earlier seeds explain the shift; otherwise the earlier run environment differs",
        ),
        primary="Mean curve CE at updates 1000, 1100, ..., 2000",
        decision_rule=f"Tuned intercepts help an arm if its winner, at its own best rate, beats the a=0/beta=0 row at its own best rate in all three paired seeds with mean gain at least {MIN_GAIN} nats. A method ordering is claimed only under the same 3-of-3 and {MIN_GAIN}-nat test",
        limits=[
            "One seed selects each rate; three fresh seeds confirm",
            "Winners were selected at the old fixed rates, as in the author's procedure",
            "Small scratch-trained story models at depth 32 and width 128",
            "Equal tokens, not equal training time",
        ],
    )


def prepare(root, prepared, checkpoints):
    root, prepared, checkpoints = (
        Path(x).expanduser().resolve() for x in (root, prepared, checkpoints)
    )
    if root.is_relative_to(REPO) or checkpoints.is_relative_to(REPO):
        raise ValueError("Outputs and checkpoints must be outside the checkout")
    manifest = read(prepared / "DATA.json")
    (root / "data").mkdir(parents=True, exist_ok=False)
    for name in ("train.npy", "curve.npy", "final.npy", "DOCUMENTS.json"):
        if sha(prepared / name) != manifest["files"][name]:
            raise ValueError("Changed prepared input: " + name)
        (root / "data" / name).symlink_to(prepared / name)
    shutil.copyfile(prepared / "DATA.json", root / "data/DATA.json")
    count = (len(np.load(root / "data/train.npy", mmap_mode="r")) - 1) // 256
    for seed in (BRACKET_SEED, *CONFIRM_SEEDS, *PREDECESSOR_SEEDS):
        np.save(
            root / f"data/order-{seed}.npy",
            np.random.default_rng(seed).permutation(count).astype(np.int32),
        )
    plan = freeze()
    plan["checkpoint_directory"] = str(checkpoints)
    plan["input_files"] = {
        p.name: sha(p) for p in sorted((root / "data").iterdir()) if p.is_file()
    }
    plan["source_files"] = {
        str(p.relative_to(REPO)): sha(p)
        for directory in SOURCES
        for p in sorted((REPO / "research" / directory).rglob("*.py"))
        if p.name != "modal_app.py"
    }
    for folder in ("jobs", "logs", "language/runs"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    write(root / "PLAN.json", plan)
    return plan


def value(root, r, seed):
    path = root / "language/runs" / job(r, "screen", seed)["id"] / "RESULT.json"
    return primary(read(path)) if path.exists() else None


def pending(root, jobs):
    """Drop finished jobs; a diverged run is a result, not a retry."""
    out = []
    for j in jobs:
        path = root / "language/runs" / j["id"] / "RESULT.json"
        done = path.exists() and read(path).get("status") in (
            "complete",
            "invalid_numerical",
        )
        if not done:
            out.append(j)
    return out


def bracket(root, checkpoints, workers, max_seconds):
    tried = {b["id"]: {-1, 0, 1} for b in bases()}
    while True:
        jobs = [
            job(at(b, k), "screen", BRACKET_SEED)
            for b in bases()
            for k in sorted(tried[b["id"]])
        ]
        wave(root, checkpoints, pending(root, jobs), workers, max_seconds)
        table, grown = {}, False
        for b in bases():
            values = {k: value(root, at(b, k), BRACKET_SEED) for k in tried[b["id"]]}
            best, bracketed, k = step(values)
            table[b["id"]] = dict(
                best_k=best,
                best_lr=rate(b["arm"], best),
                bracketed=bracketed,
                primary_by_lr={
                    str(rate(b["arm"], k)): v for k, v in sorted(values.items())
                },
            )
            if k is not None and k not in tried[b["id"]]:
                tried[b["id"]].add(k)
                grown = True
        if not grown:
            break
    write(root / "BRACKET.json", dict(rows=table, frozen_before_confirmation=True))
    return table


def execute(root, checkpoints, stage, workers, max_seconds):
    root, checkpoints = Path(root).resolve(), Path(checkpoints).resolve()
    verify(root)
    if not torch.cuda.is_available():
        raise RuntimeError("This study trains on one visible GPU")
    began = time.monotonic()
    status = dict(
        status="running",
        stage=stage,
        gpu=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
        started=time.time(),
    )
    write(root / "STATUS.json", status)
    try:
        if stage in ("predecessor", "all"):
            rows = [at(row("R"), 0), at(row("L", 0.0, 0.0), 0)]
            jobs = [job(r, "screen", s) for s in PREDECESSOR_SEEDS for r in rows]
            wave(root, checkpoints, pending(root, jobs), workers, max_seconds)
        if stage in ("bracket", "all"):
            bracket(root, checkpoints, workers, max_seconds)
        if stage in ("confirm", "all"):
            table = read(root / "BRACKET.json")["rows"]
            rows = [at(b, table[b["id"]]["best_k"]) for b in bases()]
            jobs = [job(r, "final", s) for s in CONFIRM_SEEDS for r in rows]
            wave(root, checkpoints, pending(root, jobs), workers, max_seconds)
        status.update(status="complete")
    except BaseException as error:
        status.update(status="incomplete", error=str(error))
        raise
    finally:
        status["seconds"] = time.monotonic() - began
        write(root / "STATUS.json", status)


def analyze(root):
    root = Path(root)
    table = read(root / "BRACKET.json")["rows"]
    rows = {}
    for b in bases():
        r = at(b, table[b["id"]]["best_k"])
        rows[b["id"]] = dict(
            lr=r["lr"],
            bracketed=table[b["id"]]["bracketed"],
            **summarize([endpoint(root, r, s) for s in CONFIRM_SEEDS]),
        )
    decisions = {}
    for arm in ("L", "ML"):
        anchor, winner = row(arm, 0.0, 0.0)["id"], row(arm, *WINNERS[arm])["id"]
        c = contrast(rows, winner, anchor)
        c["helps"] = c["wins"] == c["n"] and c["mean_gain"] >= MIN_GAIN
        decisions[arm] = dict(
            winner_vs_anchor=c,
            anchor_vs_mhc=contrast(rows, anchor, "M"),
            winner_vs_mhc=contrast(rows, winner, "M"),
            anchor_vs_residual=contrast(rows, anchor, "R"),
        )
    decisions["methods"] = dict(M_vs_R=contrast(rows, "M", "R"))
    for group in decisions.values():
        for c in group.values():
            c["ordered"] = c["wins"] == c["n"] and c["mean_gain"] >= MIN_GAIN
    predecessor = {}
    for name, base in (("R", row("R")), ("L-a-0p0-beta-0p0", row("L", 0.0, 0.0))):
        seeds = [value(root, at(base, 0), s) for s in PREDECESSOR_SEEDS]
        if all(v is not None for v in seeds):
            mean = float(np.mean(seeds))
            predecessor[name] = dict(
                by_seed=seeds,
                mean=mean,
                published=PREDECESSOR[name],
                difference=mean - PREDECESSOR[name],
                reproduces=abs(mean - PREDECESSOR[name]) <= 0.015,
            )
    report = dict(
        rule=freeze()["decision_rule"],
        bracket=table,
        rows=rows,
        decisions=decisions,
        predecessor=predecessor,
    )
    write(root / "REPORT.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument(
        "--stage",
        choices=("predecessor", "bracket", "confirm", "all"),
        default="all",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-seconds", type=int, default=7200)
    args = parser.parse_args()
    checkpoints = args.checkpoints or args.root / "checkpoints"
    if args.action == "prepare":
        prepare(args.root, args.prepared, checkpoints)
    elif args.action == "analyze":
        analyze(args.root)
    else:
        execute(args.root, checkpoints, args.stage, args.workers, args.max_seconds)
