"""Command-line stages of the pilot.

  prepare    resolve revisions, load GSM8K-Platinum, write splits
  screen     64 independent attempts per item (batched), one shard
  calibrate  batch-one attempt timing and temperature selection on dev
  freeze     combine calibration into the frozen budget C and arm C's
             temperature (CPU)
  select     choose observed-zero and moderate pilot items (CPU)
  pilot      run arms A/B/C/D/R on the selected items, one shard
  report     analysis (see report.py)

Each GPU stage takes ``--shard i --nshards n`` and writes under
``<out>/shard<i>/``; run one process per GPU.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import platform
import subprocess
import time
from fractions import Fraction

from . import prompts as P
from .arms import ARMS, Budget, Ledger, Runner, derive_seed
from .grader import grade

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_SCREEN = 64
TEMPS = (0.7, 1.0, 1.2)


def jload(path):
    with open(path) as f:
        return json.load(f)


def jdump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # torn final line from an interrupted write
    return out


def shard_of(ids, shard, nshards):
    return [x for k, x in enumerate(sorted(ids)) if k % nshards == shard]


def items_by_id(out):
    return {it["id"]: it for it in jload(os.path.join(out, "items.json"))}


def make_backend(args):
    from .backend import VLLMBackend

    man = jload(os.path.join(args.out, "manifest.json"))
    return VLLMBackend(
        man["model"],
        revision=man["model_revision"],
        gpu_memory_utilization=args.gpu_mem,
    )


# ---------------------------------------------------------------- prepare
def cmd_prepare(args):
    from huggingface_hub import HfApi

    from .data import DATASET, load_items, make_splits

    os.makedirs(args.out, exist_ok=True)
    api = HfApi()
    model_rev = api.model_info(args.model).sha
    data_rev = api.dataset_info(DATASET).sha
    items, info = load_items(revision=data_rev)
    refs = {it["id"]: it.pop("reference_solution") for it in items}
    splits = make_splits(
        items, seed=args.seed, n_dev=32, pilot_fraction=args.pilot_fraction
    )
    jdump(items, os.path.join(args.out, "items.json"))
    jdump(refs, os.path.join(args.out, "refs.json"))
    jdump(splits, os.path.join(args.out, "splits.json"))
    versions = {}
    for mod in ("vllm", "torch", "transformers", "datasets"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception as e:  # noqa: BLE001
            versions[mod] = f"unavailable: {e}"
    try:
        code_rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__), text=True
        ).strip()
    except Exception:
        code_rev = os.environ.get("CODE_REV", "unknown")
    jdump(
        {
            "model": args.model,
            "model_revision": model_rev,
            "dataset": DATASET,
            "dataset_revision": data_rev,
            "dataset_info": info,
            "split_seed": args.seed,
            "pilot_fraction": args.pilot_fraction,
            "split_sizes": {k: len(splits[k]) for k in ("dev", "pilot", "confirm")},
            "screen": {
                "n": N_SCREEN,
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": 4096,
            },
            "prompt_template": P.solve_prompt("{question}"),
            "versions": versions,
            "python": platform.python_version(),
            "code_revision": code_rev,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        os.path.join(args.out, "manifest.json"),
    )
    print(json.dumps({k: len(splits[k]) for k in ("dev", "pilot", "confirm")}))


# ----------------------------------------------------------------- screen
async def _screen(backend, items, refs, path, n, temperature, tag, max_inflight):
    done = {r["item"] for r in read_jsonl(path)}
    todo = [it for it in items if it["id"] not in done]
    sem = asyncio.Semaphore(max_inflight)
    fh = open(path, "a")
    stats = {"gen_tokens": 0, "items": 0}

    async def one(it, k):
        prompt = P.solve_prompt(it["question"])
        P.check_no_gold(prompt, it["id"], refs)
        seed = derive_seed(tag, it["id"], k)
        async with sem:
            r = await backend.generate(prompt, 4096, temperature, 1.0, seed)
        g = grade(r["text"], Fraction(it["gold"]))
        return {
            "k": k,
            "seed": seed,
            "gen_tokens": r["gen_tokens"],
            "prompt_tokens": r["prompt_tokens"],
            "finish": r["finish"],
            "wall": r["t_end"] - r["t_start"],
            **g,
            "text": r["text"],
        }

    async def per_item(it):
        atts = await asyncio.gather(*(one(it, k) for k in range(n)))
        rec = {
            "item": it["id"],
            "temperature": temperature,
            "n": n,
            "n_correct": sum(a["correct"] for a in atts),
            "n_truncated": sum(a["finish"] == "length" for a in atts),
            "gen_tokens": sum(a["gen_tokens"] for a in atts),
            "attempts": atts,
        }
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        stats["gen_tokens"] += rec["gen_tokens"]
        stats["items"] += 1

    t0 = time.monotonic()
    await asyncio.gather(*(per_item(it) for it in todo))
    fh.close()
    stats["wall_seconds"] = time.monotonic() - t0
    return stats


def cmd_screen(args):
    items = items_by_id(args.out)
    refs = jload(os.path.join(args.out, "refs.json"))
    splits = jload(os.path.join(args.out, "splits.json"))
    ids = shard_of(splits[args.split], args.shard, args.nshards)
    if args.limit:
        ids = ids[: args.limit]
    sd = os.path.join(args.out, f"shard{args.shard}")
    os.makedirs(sd, exist_ok=True)
    b = make_backend(args)
    try:
        stats = asyncio.run(
            _screen(
                b,
                [items[i] for i in ids],
                refs,
                os.path.join(sd, f"screen_{args.split}.jsonl"),
                N_SCREEN,
                1.0,
                "screen",
                args.max_inflight,
            )
        )
    finally:
        b.shutdown()
    stats.update({"split": args.split, "shard": args.shard, "n_items": len(ids)})
    with open(os.path.join(sd, "screen_cost.jsonl"), "a") as f:
        f.write(json.dumps(stats) + "\n")
    print(json.dumps(stats))


# -------------------------------------------------------------- calibrate
async def _calibrate(backend, items, refs, sd, per_item, temp_samples, max_inflight):
    # Batch-one timing of the baseline attempt: one call at a time.
    trows = []
    for it in items:
        for k in range(per_item):
            prompt = P.solve_prompt(it["question"])
            P.check_no_gold(prompt, it["id"], refs)
            t0 = time.monotonic()
            r = await backend.generate(
                prompt, 4096, 1.0, 1.0, derive_seed("calib", it["id"], k)
            )
            g = grade(r["text"], Fraction(it["gold"]))
            trows.append(
                {
                    "item": it["id"],
                    "k": k,
                    "seconds": time.monotonic() - t0,
                    "gen_tokens": r["gen_tokens"],
                    "finish": r["finish"],
                    "correct": g["correct"],
                }
            )
    with open(os.path.join(sd, "calib_timing.jsonl"), "w") as f:
        for r in trows:
            f.write(json.dumps(r) + "\n")
    # Temperature screen for arm C (batched; selection uses successes per
    # generated token, so batching does not bias it).
    trow = []
    for temp in TEMPS:
        path = os.path.join(sd, f"calib_temp{temp}.jsonl")
        if os.path.exists(path):
            os.remove(path)
        st = await _screen(
            backend, items, refs, path, temp_samples, temp, f"temp{temp}", max_inflight
        )
        recs = read_jsonl(path)
        trow.append(
            {
                "temperature": temp,
                "correct": sum(r["n_correct"] for r in recs),
                "attempts": sum(r["n"] for r in recs),
                "gen_tokens": sum(r["gen_tokens"] for r in recs),
                "truncated": sum(r["n_truncated"] for r in recs),
                "wall_seconds": st["wall_seconds"],
            }
        )
    jdump(trow, os.path.join(sd, "calib_temps.json"))


def cmd_calibrate(args):
    items = items_by_id(args.out)
    refs = jload(os.path.join(args.out, "refs.json"))
    splits = jload(os.path.join(args.out, "splits.json"))
    ids = shard_of(splits["dev"], args.shard, args.nshards)
    sd = os.path.join(args.out, f"shard{args.shard}")
    os.makedirs(sd, exist_ok=True)
    b = make_backend(args)
    try:
        asyncio.run(
            _calibrate(
                b,
                [items[i] for i in ids],
                refs,
                sd,
                args.per_item,
                args.temp_samples,
                args.max_inflight,
            )
        )
    finally:
        b.shutdown()


def cmd_freeze(args):
    timing = []
    temps: dict[float, dict] = {}
    for sd in sorted(glob.glob(os.path.join(args.out, "shard*"))):
        timing += read_jsonl(os.path.join(sd, "calib_timing.jsonl"))
        p = os.path.join(sd, "calib_temps.json")
        if os.path.exists(p):
            for r in jload(p):
                agg = temps.setdefault(
                    r["temperature"],
                    {"correct": 0, "attempts": 0, "gen_tokens": 0, "truncated": 0},
                )
                for k in agg:
                    agg[k] += r[k]
    if not timing:
        raise SystemExit("no calibration timing found")
    mean_s = sum(r["seconds"] for r in timing) / len(timing)
    mean_tok = sum(r["gen_tokens"] for r in timing) / len(timing)
    rate = {t: v["correct"] / max(1, v["gen_tokens"]) for t, v in temps.items()}
    best = max(rate.values())
    # Ties go to the original setting, then to the lower temperature.
    temp_c = (
        1.0 if rate.get(1.0) == best else min(t for t, r in rate.items() if r == best)
    )
    frozen = {
        "C_seconds": round(N_SCREEN * mean_s, 3),
        "mean_attempt_seconds": mean_s,
        "mean_attempt_gen_tokens": mean_tok,
        "timing_attempts": len(timing),
        "timing_truncated": sum(r["finish"] == "length" for r in timing),
        "token_budget": 32768,
        "temp_c": temp_c,
        "temp_selection": {
            str(t): dict(v, success_per_token=rate[t]) for t, v in temps.items()
        },
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    jdump(frozen, os.path.join(args.out, "frozen.json"))
    print(json.dumps(frozen, indent=1))


# ----------------------------------------------------------------- select
def load_screen(out, split):
    recs = {}
    for p in sorted(glob.glob(os.path.join(out, "shard*", f"screen_{split}.jsonl"))):
        for r in read_jsonl(p):
            recs[r["item"]] = r
    return recs


def cmd_select(args):
    splits = jload(os.path.join(args.out, "splits.json"))
    recs = load_screen(args.out, "pilot")
    missing = set(splits["pilot"]) - set(recs)
    if missing and not args.allow_partial:
        raise SystemExit(f"{len(missing)} pilot items unscreened")
    zeros = sorted(i for i, r in recs.items() if r["n_correct"] == 0)
    mods = sorted(i for i, r in recs.items() if 1 <= r["n_correct"] <= 16)
    mods = sorted(mods, key=lambda i: derive_seed("moderate", i))[: args.n_moderate]
    zeros = sorted(zeros, key=lambda i: derive_seed("zero", i))
    target = zeros[: args.n_zero]
    sel = {
        "observed_zero": target,
        "observed_zero_available": len(zeros),
        "moderate": sorted(mods),
        "moderate_definition": "1 to 16 successes of 64",
        "screened": len(recs),
        "screen_truncation_rate": sum(r["n_truncated"] for r in recs.values())
        / max(1, sum(r["n"] for r in recs.values())),
    }
    jdump(sel, os.path.join(args.out, "selection.json"))
    print(
        json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in sel.items()})
    )


# ------------------------------------------------------------------ pilot
def arm_order(index: int, rep: int) -> list[str]:
    k = (index * 3 + rep) % len(ARMS)
    return list(ARMS[k:] + ARMS[:k])


async def _pilot(runner, jobs, frozen, done, concurrency):
    sem = asyncio.Semaphore(concurrency)
    n = [0]

    async def one(job):
        it, fails, arm, rep, budget = job
        async with sem:
            rec = await runner.run(it, fails, arm, rep, budget, frozen["temp_c"])
        n[0] += 1
        print(
            json.dumps(
                {
                    "done": n[0],
                    "of": len(jobs),
                    "arm": arm,
                    "item": it["id"],
                    "rep": rep,
                    "mode": budget.mode,
                    "solved": rec["solved"],
                    "elapsed": round(rec["elapsed"], 1),
                }
            ),
            flush=True,
        )

    if concurrency == 1:
        for job in jobs:
            await one(job)
    else:
        await asyncio.gather(*(one(j) for j in jobs))


def cmd_pilot(args):
    items = items_by_id(args.out)
    refs = jload(os.path.join(args.out, "refs.json"))
    frozen = jload(os.path.join(args.out, "frozen.json"))
    sel = jload(os.path.join(args.out, "selection.json"))
    screen = load_screen(args.out, "pilot")
    chosen = sel["observed_zero"] + sel["moderate"]
    ids = shard_of(chosen, args.shard, args.nshards)
    sd = os.path.join(args.out, f"shard{args.shard}", f"pilot_{args.mode}")
    ledger = Ledger(sd)
    done = ledger.done_runs()
    if args.mode == "time":
        budget = Budget("time", seconds=frozen["C_seconds"])
    else:
        budget = Budget("tokens", tokens=frozen["token_budget"])
    jobs = []
    order_index = {i: k for k, i in enumerate(sorted(chosen))}
    for i in ids:
        it = items[i]
        fails = [
            a["text"]
            for a in sorted(screen[i]["attempts"], key=lambda a: a["k"])
            if not a["correct"]
        ]
        for rep in range(args.reps):
            for arm in arm_order(order_index[i], rep):
                if args.arms and arm not in args.arms:
                    continue
                key = f"{args.tag}|{i}|{arm}|{rep}|{budget.tag()}"
                if key in done:
                    continue
                jobs.append((it, fails, arm, rep, budget))
    b = make_backend(args)
    runner = Runner(b, ledger, refs, args.tag)
    print(
        json.dumps({"shard": args.shard, "jobs": len(jobs), "skipped_done": len(done)}),
        flush=True,
    )
    try:
        asyncio.run(_pilot(runner, jobs, frozen, done, args.concurrency))
    finally:
        ledger.close()
        b.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--pilot-fraction", type=float, default=0.6)

    for name in ("screen", "calibrate", "pilot"):
        p = sub.add_parser(name)
        p.add_argument("--shard", type=int, default=0)
        p.add_argument("--nshards", type=int, default=1)
        p.add_argument("--gpu-mem", type=float, default=0.85)
        p.add_argument("--max-inflight", type=int, default=512)
        if name == "screen":
            p.add_argument("--split", default="pilot")
            p.add_argument("--limit", type=int, default=0)
        if name == "calibrate":
            p.add_argument("--per-item", type=int, default=2)
            p.add_argument("--temp-samples", type=int, default=16)
        if name == "pilot":
            p.add_argument("--mode", choices=("time", "tokens"), default="time")
            p.add_argument("--reps", type=int, default=3)
            p.add_argument("--tag", default="pilot1")
            p.add_argument("--concurrency", type=int, default=1)
            p.add_argument("--arms", nargs="*", default=None)

    sub.add_parser("freeze")
    p = sub.add_parser("select")
    p.add_argument("--n-zero", type=int, default=60)
    p.add_argument("--n-moderate", type=int, default=20)
    p.add_argument("--allow-partial", action="store_true")

    args = ap.parse_args()
    {
        "prepare": cmd_prepare,
        "screen": cmd_screen,
        "calibrate": cmd_calibrate,
        "freeze": cmd_freeze,
        "select": cmd_select,
        "pilot": cmd_pilot,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
