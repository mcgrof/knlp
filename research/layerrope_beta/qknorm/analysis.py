# SPDX-License-Identifier: MIT
"""Report paired seed effects and separate absolute loss from rounding damage."""

import csv
import math
from pathlib import Path
import statistics

from research.mo_variants.stage_c_common import read, write
from research.tinystories_depth.kv_cache import bytes_per_token


def summary(values):
    values = list(values)
    return dict(
        values=values,
        n=len(values),
        mean=statistics.mean(values) if values else None,
        median=statistics.median(values) if values else None,
        minimum=min(values) if values else None,
        maximum=max(values) if values else None,
    )


def floor(cells, seeds, thresholds):
    """One policy must satisfy every seed; never select codecs per seed."""
    valid = []
    for policy in sorted({c["policy"] for c in cells}):
        selected = [c for c in cells if c["policy"] == policy]
        if (
            len(selected) == len(seeds)
            and {c["seed"] for c in selected} == set(seeds)
            and all(c["ce"] <= thresholds[c["seed"]] + 1e-12 for c in selected)
        ):
            valid.append(dict(policy=policy, bytes=selected[0]["ideal_total_bytes"]))
    return dict(
        minimum_bytes=min((c["bytes"] for c in valid), default=None),
        qualifying_policies=valid,
    )


def analyze(card, root):
    from .run import jobs, name, training_barrier

    root = Path(root)
    p = card["protocol"]
    training = training_barrier(card, root)
    rows = []
    for job in jobs(card):
        directory = root / "evaluation" / name(job)
        r = training[name(job)]["result"]
        for endpoint, available in (
            (
                "time",
                any(f["target_seconds"] == 787.0 for f in r["history"]["landmarks"]),
            ),
            ("tokens", bool(r["history"]["update_landmarks"])),
        ):
            if not available:
                continue
            values = {
                c["name"]: read(directory / (endpoint + "-" + c["name"] + ".json"))
                for c in p["cache_cases"]
            }
            clean = values["clean"]["result"]["ce"]
            base = values["symmetric-K3V3"]["result"]["ce"]
            for policy in p["cache_cases"]:
                v = values[policy["name"]]
                if (
                    v["job"] != job
                    or v["result"]["predicted_tokens"] != p["final_predicted_tokens"]
                ):
                    raise ValueError("Evaluation identity/coverage differs")
                storage = bytes_per_token(policy, **v["cache_geometry"])
                ce = v["result"]["ce"]
                rows.append(
                    dict(
                        endpoint=endpoint,
                        arm=job["arm"],
                        qknorm=job["qknorm"],
                        seed=job["seed"],
                        policy=policy["name"],
                        ce=ce,
                        ppl=math.exp(ce),
                        clean_ce=clean,
                        own_clean_damage=ce - clean,
                        value_bit_sensitivity=(
                            ce - base if policy["name"] == "symmetric-K3V2" else None
                        ),
                        step=v["step"],
                        training_seconds=v["training_seconds"],
                        payload_bytes=storage["payload_bytes"],
                        metadata_bytes=storage["scale_and_offset_bytes"],
                        ideal_total_bytes=storage["ideal_total_bytes"],
                        actual_bf16_tensor_bytes_per_token=storage["bf16_total_bytes"],
                        qknorm_model_parameter_bytes=v["qknorm_model_parameter_bytes"],
                    )
                )
    with (root / "CELLS.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["endpoint"])
        writer.writeheader()
        writer.writerows(rows)
    lookup = {
        (r["endpoint"], r["arm"], r["qknorm"], r["seed"], r["policy"]): r for r in rows
    }
    effects, interactions, floors, reversals = [], [], [], []

    def delta(ep, arm, policy, metric):
        return {
            s: lookup[ep, arm, True, s, policy][metric]
            - lookup[ep, arm, False, s, policy][metric]
            for s in p["final_seeds"]
            if (ep, arm, True, s, policy) in lookup
            and (ep, arm, False, s, policy) in lookup
        }

    for ep in ("time", "tokens"):
        for policy in p["cache_cases"]:
            policy = policy["name"]
            for metric in ("ce", "own_clean_damage"):
                ds = {
                    arm: delta(ep, arm, policy, metric) for arm in ("R", "L", "M", "ML")
                }
                for arm, values in ds.items():
                    stats = summary(values.values())
                    effects.append(
                        dict(
                            endpoint=ep,
                            arm=arm,
                            policy=policy,
                            metric=metric,
                            D=stats,
                            consistent_screen_gain=len(values) == 3
                            and all(v < 0 for v in values.values())
                            and stats["mean"] <= -0.01,
                        )
                    )
                    if arm != "R":
                        interactions.append(
                            dict(
                                endpoint=ep,
                                arm=arm,
                                policy=policy,
                                metric=metric,
                                contrast="I",
                                result=summary(
                                    values[s] - ds["R"][s]
                                    for s in p["final_seeds"]
                                    if s in values and s in ds["R"]
                                ),
                            )
                        )
                interactions.append(
                    dict(
                        endpoint=ep,
                        policy=policy,
                        metric=metric,
                        contrast="J=M-minus-L",
                        result=summary(
                            ds["M"][s] - ds["L"][s]
                            for s in p["final_seeds"]
                            if s in ds["M"] and s in ds["L"]
                        ),
                    )
                )
        references = {
            s: lookup[ep, "R", False, s, "clean"]["ce"] + 0.1
            for s in p["final_seeds"]
            if (ep, "R", False, s, "clean") in lookup
        }
        for arm in p["arms"]:
            cells = [
                r
                for r in rows
                if r["endpoint"] == ep
                and r["arm"] == arm["arm"]
                and r["qknorm"] == arm["qknorm"]
            ]
            own = {r["seed"]: r["clean_ce"] + 0.1 for r in cells}
            own_floor = (
                floor(cells, p["final_seeds"], own)
                if len(own) == 3
                else dict(minimum_bytes=None, qualifying_policies=[])
            )
            common = (
                floor(cells, p["final_seeds"], references)
                if len(references) == 3
                else dict(minimum_bytes=None, qualifying_policies=[])
            )
            floors.append(
                dict(
                    endpoint=ep,
                    arm=arm["name"],
                    own_clean=own_floor,
                    common_quality=common,
                )
            )
            for c in cells:
                s = c["seed"]
                if s in references and c["clean_ce"] > references[s] >= c["ce"]:
                    reversals.append(dict(**c, requires_confirmation=True))
    statuses = {
        k: dict(
            status=v["result"]["status"],
            updates=v["result"]["completed_steps"],
            training_seconds=v["result"]["training_seconds"],
            trainer_process_seconds=v["result"]["attempt_process_seconds"],
            full_process_seconds_upper_bound=v["result"].get(
                "full_process_seconds_upper_bound"
            ),
            exit_poll_interval_seconds=v["result"].get("exit_poll_interval_seconds"),
            peak_allocated_bytes=v["result"].get("peak_allocated_bytes"),
            token_endpoint_available=bool(v["result"]["history"]["update_landmarks"]),
            clipped_updates=sum(
                u.get("clipped", False) for u in v["result"]["history"]["updates"]
            ),
            loss_spikes=sum(
                u.get("loss_spike", False) for u in v["result"]["history"]["updates"]
            ),
        )
        for k, v in training.items()
    }
    stability = []
    for a in ("R", "L", "M", "ML"):
        failed = sum(
            statuses[f"{a}-off-seed{s}"]["status"] == "invalid_numerical"
            for s in p["final_seeds"]
        )
        on_complete = all(
            statuses[f"{a}-on-seed{s}"]["status"] == "complete"
            for s in p["final_seeds"]
        )
        stability.append(
            dict(
                arm=a,
                off_numerical_failures=failed,
                on_all_complete=on_complete,
                confirmation_screen=failed >= 2 and on_complete,
            )
        )
    floor_gains = []
    for ep in ("time", "tokens"):
        for arm in ("R", "L", "M", "ML"):
            a, b = [
                next(
                    f
                    for f in floors
                    if f["endpoint"] == ep and f["arm"] == arm + suffix
                )["common_quality"]["minimum_bytes"]
                for suffix in ("-off", "-on")
            ]
            floor_gains.append(
                dict(
                    endpoint=ep,
                    arm=arm,
                    off_bytes=a,
                    on_bytes=b,
                    screen_gain=b is not None and (a is None or b < a),
                )
            )
    result = dict(
        status="complete",
        scope="Three-seed fixed-recipe screen; no population winner or equivalence",
        metrics="Held-out TinyStories next-token CE in natural-log nats and exp(CE) perplexity; 131072 common targets per endpoint",
        fixed_recipe_limitation=p["fixed_recipe_limitation"],
        cells=len(rows),
        statuses=statuses,
        effects=effects,
        interactions=interactions,
        floors=floors,
        quality_reversals=reversals,
        common_quality_floor_gains=floor_gains,
        stability=stability,
        compression_claim="Ideal packed accounting only; actual cache tensors remain BF16",
        interpretation="Absolute quantized CE, own-clean damage, key-only damage and value-bit sensitivity have distinct meanings",
        automatic_followup=False,
    )
    write(root / "SUMMARY.json", result)
    text = [
        "# QKNorm learning and cache screen",
        "",
        result["scope"] + ".",
        "",
        result["metrics"] + ".",
        "",
        "Learning rates were inherited from the preceding off-only study. Each trajectory gets 787 synchronized training seconds. The 1,000-update checkpoint is a separate endpoint; missing endpoints remain censored.",
        "",
        "| Endpoint | Architecture | QKNorm | Clean CE | K3/V3 CE | K3/V2 CE | Value-bit damage |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for ep in ("time", "tokens"):
        for a in p["arms"]:
            means = []
            for pol in ("clean", "symmetric-K3V3", "symmetric-K3V2"):
                selected = [
                    r["ce"]
                    for r in rows
                    if r["endpoint"] == ep
                    and r["arm"] == a["arm"]
                    and r["qknorm"] == a["qknorm"]
                    and r["policy"] == pol
                ]
                means.append(statistics.mean(selected) if len(selected) == 3 else None)
            fmt = lambda v: "missing" if v is None else f"{v:.6f}"
            damage = None if None in means else means[2] - means[1]
            text.append(
                f"| {ep} | {a['arm']} | {'on' if a['qknorm'] else 'off'} | "
                + " | ".join(map(fmt, means + [damage]))
                + " |"
            )
    text += [
        "",
        "All paired seed effects, ranges, statuses and single-policy floors are in SUMMARY.json and CELLS.csv. An apparent gain requests separate confirmation; this run does not add seeds, temperatures or depths.",
        "",
    ]
    (root / "RESULTS.md").write_text("\n".join(text) + "\n")
    plot(card, training, rows, root)
    return result


def plot(card, training, cells, root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(R="#3366cc", L="#009966", M="#cc6600", ML="#9933cc")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for v in training.values():
        r = v["result"]
        j = r["identity"]["job"]
        frames = r["history"]["landmarks"]
        for ax, key in zip(axes, ("training_seconds", "predicted_tokens")):
            ax.plot(
                [f[key] for f in frames],
                [f["validation"]["ce"] for f in frames],
                color=colors[j["arm"]],
                linestyle="-" if j["qknorm"] else "--",
                alpha=0.65,
                label=(
                    j["name"]
                    if j["seed"] == card["protocol"]["final_seeds"][0]
                    else None
                ),
            )
    for ax, label in zip(
        axes, ("Synchronized training seconds", "Predicted training tokens")
    ):
        ax.set(xlabel=label, ylabel="Validation CE (nats)")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(Path(root) / ("learning-curves." + extension), dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, ep in zip(axes, ("time", "tokens")):
        for a in card["protocol"]["arms"]:
            points = []
            for policy in card["protocol"]["cache_cases"]:
                selected = [
                    c
                    for c in cells
                    if c["endpoint"] == ep
                    and c["arm"] == a["arm"]
                    and c["qknorm"] == a["qknorm"]
                    and c["policy"] == policy["name"]
                ]
                if len(selected) == 3:
                    points.append(
                        (
                            selected[0]["ideal_total_bytes"],
                            statistics.mean(c["ce"] for c in selected),
                        )
                    )
            points.sort()
            frontier = []
            best = float("inf")
            for x, y in points:
                if y < best:
                    frontier.append((x, y))
                    best = y
            if frontier:
                ax.plot(
                    *zip(*frontier),
                    marker="o",
                    color=colors[a["arm"]],
                    linestyle="-" if a["qknorm"] else "--",
                    label=a["name"],
                )
        ax.set(
            xlabel="Ideal packed cache bytes/token",
            ylabel="Mean held-out CE (nats)",
            title=ep + " endpoint",
        )
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(Path(root) / ("ce-byte-frontiers." + extension), dpi=160)
    plt.close(fig)
