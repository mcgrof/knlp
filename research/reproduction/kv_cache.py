"""Reuse complete reproduction checkpoints for separate KV-cache cohorts."""

import argparse
import gc
import math
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from research.mo_variants.model import Model as DepthCode
from research.mo_variants.stage_c_common import ROWS, SEEDS, read, sha, write
from research.tinystories_depth.kv_cache import cases, bytes_per_token, evaluate, parity
from research.tinystories_depth.kv_cache_checks import qualify
from research.tinystories_depth.model import Model as Backbone
from research.tinystories_depth.train import state_hash


def inputs(source, cohort):
    """Validate complete, matched inputs before loading any model onto a GPU."""
    source = Path(source).expanduser().resolve()
    jobs = []
    if cohort == "depth":
        data = source / "data/final.npy"
        manifest = read(source / "data/DATA.json")
        expected = manifest["files"]["final.npy"]
        for seed in (75211, 75212, 75213):
            for depth in (4, 16, 32):
                for arm in (("R", "M", "M-O") if depth == 32 else ("R", "M")):
                    jid = f"final-{arm}-D{depth}-seed{seed}"
                    result = read(source / "depth-screen/runs" / jid / "RESULT.json")
                    j = result["metadata"]["job"]
                    if (
                        result["status"] != "complete"
                        or result["completed_steps"] != 2000
                    ):
                        raise ValueError("Incomplete depth checkpoint: " + jid)
                    if (j["arm"], j["depth"], j["seed"], j["lr"]) != (
                        arm,
                        depth,
                        seed,
                        0.003,
                    ):
                        raise ValueError(
                            "Expected the matched .003-learning-rate cohort: " + jid
                        )
                    jobs.append(
                        dict(
                            id=jid,
                            row=arm,
                            arm=arm,
                            depth=depth,
                            seed=seed,
                            step=2000,
                            checkpoint=result["checkpoint"],
                            pairing={
                                k: result["metadata"][k]
                                for k in ("order_sha256", "optimizer_config_sha256")
                            },
                            initial_weights=result["identities"]["common_weights"],
                        )
                    )
        for j in jobs:
            if j["arm"] != "R":
                baseline = next(
                    x
                    for x in jobs
                    if x["arm"] == "R"
                    and x["depth"] == j["depth"]
                    and x["seed"] == j["seed"]
                )
                if (
                    j["pairing"] != baseline["pairing"]
                    or j["initial_weights"] != baseline["initial_weights"]
                ):
                    raise ValueError(
                        "The saved seed pair uses a different recipe: " + j["id"]
                    )
    elif cohort == "confirmation":
        root = source / "confirmation"
        data = root / "data/confirmation/final.npy"
        plan = read(root / "records/STAGE-C-PLAN.json")
        expected = plan["data"]["confirmation_files"]["final.npy"]
        wanted = {(row, seed) for row, _, _ in ROWS for seed in SEEDS}
        seen = set()
        definitions = {row: (variant, mapping) for row, variant, mapping in ROWS}
        for j in plan["jobs"]:
            if (j["row"], j["seed"]) not in wanted or (j["row"], j["seed"]) in seen:
                raise ValueError("Unexpected or duplicated confirmation row/seed")
            seen.add((j["row"], j["seed"]))
            if (j["variant"], j["mapping"]) != definitions[j["row"]]:
                raise ValueError("Changed confirmation row definition")
            result = read(root / "records/runs" / j["id"] / "RESULT.json")
            if result["status"] != "complete" or result["checkpoint"]["step"] != 4000:
                raise ValueError("Incomplete confirmation checkpoint: " + j["id"])
            if j["depth"] != 32 or j["steps"] != 4000 or j["lr"] != 0.003:
                raise ValueError("Changed confirmation recipe: " + j["id"])
            jobs.append(
                dict(
                    id=j["id"],
                    row=j["row"],
                    variant=j["variant"],
                    mapping=j["mapping"],
                    depth=32,
                    seed=j["seed"],
                    step=4000,
                    checkpoint=result["checkpoint"],
                )
            )
        if seen != wanted:
            raise ValueError("Missing confirmation row/seed")
    else:
        raise ValueError("Choose depth or confirmation")
    if sha(data) != expected:
        raise ValueError("The saved evaluation data changed")
    for j in jobs:
        receipt = j["checkpoint"]
        path = Path(receipt["path"]).expanduser()
        if not path.is_file():
            path = (
                source / "depth-screen/checkpoints" / (j["id"] + ".pt")
                if cohort == "depth"
                else source / "confirmation/checkpoints" / j["id"] / path.name
            )
        if not path.is_file() or sha(path) != receipt["sha256"]:
            raise ValueError("Missing or changed checkpoint: " + str(path))
        j["checkpoint"] = {**receipt, "path": str(path.resolve())}
    return jobs, data, expected


def load(job, device, vocab=50257):
    saved = torch.load(
        job["checkpoint"]["path"], map_location="cpu", weights_only=False
    )
    if (
        saved["step"] != job["step"]
        or state_hash(saved["model"]) != job["checkpoint"]["parameter_sha256"]
    ):
        raise ValueError("Checkpoint model-state integrity failure: " + job["id"])
    if job["step"]:
        metadata = saved["metadata"]["job"]
        keys = (
            ("variant", "mapping", "depth", "seed")
            if "variant" in job
            else ("arm", "depth", "seed")
        )
        if any(metadata[key] != job[key] for key in keys):
            raise ValueError("Checkpoint architecture or seed mismatch")
    if "variant" in job:
        model = DepthCode(
            job["variant"],
            job["depth"],
            job["seed"],
            vocab=vocab,
            mapping=job["mapping"],
        )
        model.step = saved["step"]
    else:
        model = Backbone(job["arm"], job["depth"], job["seed"], vocab=vocab)
    model.load_state_dict(saved["model"], strict=True)
    return model.eval().requires_grad_(False).to(device)


def paired(values):
    """Describe seed-level differences; windows are not independent trials."""
    n = len(values)
    critical = {3: 4.30265273, 8: 2.364624251}[n]
    mean = statistics.mean(values)
    half = critical * statistics.stdev(values) / math.sqrt(n)
    nonzero = [x for x in values if x != 0]
    wins = sum(x < 0 for x in nonzero)
    sign_p = sum(
        math.comb(len(nonzero), k) for k in range(wins, len(nonzero) + 1)
    ) / 2 ** len(nonzero)
    return dict(
        mean=mean,
        interval_descriptive=[mean - half, mean + half],
        damage_wins=sum(x < 0 for x in values),
        pairs=values,
        n=n,
        one_sided_sign_p=sign_p,
    )


def summarize(root, records, cohort, smoke=False):
    groups = {}
    for record in records:
        key = f"{record['row']}-D{record['depth']}"
        groups.setdefault(key, []).append(record)
    summary = {}
    for key, members in groups.items():
        n = len(members)
        c = {}
        for case in cases():
            cells = [
                next(x for x in r["cases"] if x["name"] == case["name"])
                for r in members
            ]
            damages = [x["delta_ce"] for x in cells]
            c[case["name"]] = dict(
                mean_ce=statistics.mean(x["ce"] for x in cells),
                mean_damage=statistics.mean(damages),
                min_damage=min(damages),
                max_damage=max(damages),
                passes=sum(x <= 0.1 for x in damages),
                n=n,
                seed_damage={str(r["seed"]): x for r, x in zip(members, damages)},
                ideal_cache_bytes_per_token=cells[0]["storage"]["ideal_total_bytes"],
            )
        summary[key] = dict(
            row=members[0]["row"],
            depth=members[0]["depth"],
            cases=c,
            lowest_passing_bits={
                method: min(
                    [b for b in (8, 4, 3, 2) if c[f"{method}-K{b}V{b}"]["passes"] == n]
                    or [16]
                )
                for method in ("symmetric", "affine")
            },
        )
    contrasts = []
    if not smoke:
        comparisons = (
            [("M", "R", d) for d in (4, 16, 32)]
            if cohort == "depth"
            else [
                (a, b, 32)
                for a, b in (
                    ("V01", "V00"),
                    ("V01", "V02"),
                    ("V01", "V03"),
                    ("V30", "V00"),
                    ("V30", "V30-S"),
                    ("V30", "V30-C"),
                )
            ]
        )
        for a, b, depth in comparisons:
            for case in cases()[1:]:
                va = summary[f"{a}-D{depth}"]["cases"][case["name"]]["seed_damage"]
                vb = summary[f"{b}-D{depth}"]["cases"][case["name"]]["seed_damage"]
                if set(va) != set(vb):
                    raise ValueError("Unpaired seed sets")
                contrasts.append(
                    dict(
                        candidate=a,
                        baseline=b,
                        depth=depth,
                        case=case["name"],
                        primary=case["name"] == "symmetric-K4V4",
                        mean_absolute_ce_difference=summary[f"{a}-D{depth}"]["cases"][
                            case["name"]
                        ]["mean_ce"]
                        - summary[f"{b}-D{depth}"]["cases"][case["name"]]["mean_ce"],
                        **paired([va[s] - vb[s] for s in sorted(va)]),
                    )
                )
        primary = [x for x in contrasts if x["primary"]]
        if cohort == "confirmation":
            previous = 0
            for i, c in enumerate(sorted(primary, key=lambda c: c["one_sided_sign_p"])):
                previous = max(
                    previous, min(1.0, c["one_sided_sign_p"] * (len(primary) - i))
                )
                c["holm_p"] = previous
                c["primary_effect_rule_pass"] = (
                    previous <= 0.05
                    and c["mean"] <= -0.005
                    and c["mean_absolute_ce_difference"] < 0
                )
    report = dict(
        status="complete",
        cohort=cohort,
        scope="synthetic-token smoke" if smoke else "saved-checkpoint cache evaluation",
        scientific_confirmation_allowed=False,
        groups=summary,
        contrasts=contrasts,
        limitations=[
            "Two token-local quantizers and 256-token contexts only",
            "No packed-kernel memory or speed measurement",
            "Keep the three-seed and eight-seed cohorts separate",
            "Retain inherited model numerical flags; cache qualification does not clear them",
        ],
    )
    write(root / "REPORT.json", report)
    text = [
        "# KV-cache fake quantization",
        "",
        "Cohort: " + cohort + (" (SMOKE; no scientific verdict)" if smoke else ""),
        "",
        "Damage is extra CE over the same checkpoint's clean endpoint. The pass criterion is <=0.1 nats on every seed. Compare absolute CE separately. Fixed quantizers do not establish a universal precision boundary.",
        "",
        "| Model | Symmetric K4/V4 damage | Affine K2/V2 damage | Lowest all-seed bits, symmetric / affine |",
        "| --- | ---: | ---: | --- |",
    ]
    for key, g in summary.items():
        text.append(
            f"| {key} | {g['cases']['symmetric-K4V4']['mean_damage']:.6f} | {g['cases']['affine-K2V2']['mean_damage']:.6f} | {g['lowest_passing_bits']['symmetric']} / {g['lowest_passing_bits']['affine']} |"
        )
    text.extend(
        [
            "",
            "All seed pairs, quantizer settings, absolute losses and ideal cache byte estimates are in the generated REPORT.json and cells/. These files belong in the configured external output directory.",
            "",
            "Three-seed comparisons are descriptive. The eight-seed primary sign tests use Holm correction over six contrasts; the other settings are exploratory. No new architecture or numerical qualification verdict is issued by this cache report.",
            "",
        ]
    )
    (root / "REPORT.md").write_text("\n".join(text))
    return report


def run(root, source, cohort, device="cuda", max_seconds=0, smoke=False):
    root = Path(root).expanduser().resolve()
    if root.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("Cache outputs must be outside the source checkout")
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(min(8, int(os.environ.get("OMP_NUM_THREADS", "8"))))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    began = time.monotonic()
    status = dict(
        status="running",
        scientific_confirmation_allowed=False,
        cohort=cohort,
        smoke=smoke,
    )
    write(root / "STATUS.json", status)
    try:
        qualification = qualify(device)
        write(root / "QUALIFICATION.json", qualification)
        if smoke:
            jobs = []
            for row in ("R", "M", "M-O"):
                arm = row
                model = Backbone(arm, 2, 89104, vocab=97)
                path = root / (row + ".pt")
                torch.save(dict(model=model.state_dict(), step=0), path)
                jobs.append(
                    dict(
                        id=row,
                        row=row,
                        arm=arm,
                        depth=2,
                        seed=89104,
                        step=0,
                        checkpoint=dict(
                            path=str(path),
                            sha256=sha(path),
                            parameter_sha256=state_hash(model.state_dict()),
                        ),
                    )
                )
            tokens = np.random.default_rng(89104).integers(0, 97, 33, dtype=np.int64)
            data_sha = "synthetic-token smoke"
        else:
            jobs, data, data_sha = inputs(source, cohort)
            tokens = np.load(data, mmap_mode="r")
            if len(tokens) != 131073:
                raise ValueError("Expected 131072 predicted final-array tokens")
        write(
            root / "PLAN.json",
            dict(
                cohort=cohort,
                jobs=jobs,
                cases=cases(),
                data_sha256=data_sha,
                smoke=smoke,
                precision="Original BF16 activation recipe and FP32 weights/routing/norms/CE/QDQ; TF32 off",
            ),
        )
        window = 16 if smoke else 256
        ids = torch.tensor(
            np.array(tokens[: window + 1], dtype=np.int64), device=device
        )[None, :-1]
        records = []
        for job in jobs:
            if max_seconds and time.monotonic() - began >= max_seconds:
                raise TimeoutError("Cache evaluation elapsed-time limit")
            model = load(job, device, vocab=97 if smoke else 50257)
            original = state_hash(model.state_dict())
            record = dict(
                id=job["id"],
                row=job["row"],
                depth=job["depth"],
                seed=job["seed"],
                checkpoint=job["checkpoint"],
                adapter_parity=parity(model, ids, device),
                status="running",
                cases=[],
            )
            dest = root / "cells" / (job["id"] + ".json")
            for case in cases():
                if max_seconds and time.monotonic() - began >= max_seconds:
                    raise TimeoutError("Cache evaluation elapsed-time limit")
                result = evaluate(
                    model,
                    tokens,
                    case,
                    device,
                    window=window,
                    batch_size=2 if smoke else 32,
                )
                result.update(case)
                result["delta_ce"] = result["ce"] - (
                    record["cases"][0]["ce"] if record["cases"] else result["ce"]
                )
                result["perplexity_ratio"] = math.exp(result["delta_ce"])
                result["storage"] = bytes_per_token(case, depth=job["depth"])
                record["cases"].append(result)
                write(dest, record)
            if state_hash(model.state_dict()) != original:
                raise RuntimeError("Cache evaluation changed model state")
            record.update(status="complete", weights_unchanged=True)
            write(dest, record)
            records.append(record)
            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        summarize(root, records, cohort, smoke)
        status["status"] = "complete"
    except BaseException as exc:
        status.update(status="incomplete", error=str(exc))
        raise
    finally:
        status["seconds"] = time.monotonic() - began
        write(root / "STATUS.json", status)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--input-run", type=Path)
    p.add_argument("--cohort", choices=("depth", "confirmation"), required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--max-seconds", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    run(a.root, a.input_run, a.cohort, a.device, a.max_seconds, a.smoke)
