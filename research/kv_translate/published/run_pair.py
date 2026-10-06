#!/usr/bin/env python3
"""Run the paired comparison for one source and receiver, stage by stage.

Every stage writes a receipt when it finishes and is skipped when a receipt
from the same configuration already exists, so an instance that disappears
costs the stage in flight. That matters more on a platform whose accelerated
functions can always be preempted, and it is also what makes the per-stage
timings usable: each one is measured once, in isolation, with its own peak
memory.

The timings are the second product of this program. The same stages run on
different providers, and the comparison of what each provider charges for an
identical piece of work is built from these receipts rather than from list
prices.

Nothing here is specific to a provider. It needs a working directory, a
Hugging Face cache and a card.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time

DEFAULT_TASKS = ("hellaswag", "arc_challenge", "winogrande", "mmlu")
METRIC = {
    "hellaswag": "acc_norm",
    "arc_challenge": "acc_norm",
    "winogrande": "acc",
    "mmlu": "acc",
}
FEWSHOT = {"mmlu": 5}
MODES = ("direct", "native", "full_head", "cache_bridge")
SOURCE_NATIVE = "source_native"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)


def identity(cfg):
    keys = (
        "source",
        "source_revision",
        "target",
        "target_revision",
        "sequences",
        "sequence_length",
        "stride",
        "k",
        "lam",
        "calibration_dataset",
        "calibration_config",
        "boundaries",
    )
    blob = json.dumps({k: cfg[k] for k in keys}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def eval_identity(cfg):
    """What a score depends on beyond the fitted mappers."""
    blob = json.dumps(
        {
            "fit": identity(cfg),
            "eval_examples": cfg["eval_examples"],
            "fewshot": FEWSHOT,
            "metric": METRIC,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def usable_processors():
    """Processors this process may actually use, not the host's count.

    A container sees every processor of its host but is held to a quota.
    Threading libraries size their pools from the visible count, and a pool
    several times the quota spends the quota on contention: one stage ran
    seventeen times slower that way.
    """
    seen = len(os.sched_getaffinity(0))
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        if quota != "max":
            return max(1, min(seen, int(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return seen


def device_of(cfg):
    import torch

    return cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")


class Stages:
    """Receipts on disk, one per stage, written atomically."""

    def __init__(self, work, ident, on_stage_done=None):
        self.dir = os.path.join(work, "stages")
        os.makedirs(self.dir, exist_ok=True)
        self.ident = ident
        self.on_stage_done = on_stage_done

    def path(self, name):
        return os.path.join(self.dir, f"{name}.json")

    def done(self, name, ident=None):
        p = self.path(name)
        if not os.path.exists(p):
            return None
        with open(p) as f:
            r = json.load(f)
        if r.get("identity") != (ident or self.ident) or not r.get("completed"):
            return None
        return r

    def run(self, name, fn, ident=None):
        prior = self.done(name, ident)
        if prior is not None:
            log(f"{name}: already complete under this configuration, skipping")
            return prior
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        log(f"{name}: start")
        t0 = time.time()
        detail = fn() or {}
        rec = {
            "stage": name,
            "identity": ident or self.ident,
            "completed": True,
            "seconds": round(time.time() - t0, 2),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "peak_gpu_gib": (
                round(torch.cuda.max_memory_allocated() / 2**30, 2)
                if torch.cuda.is_available()
                else None
            ),
            "peak_host_rss_gib": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2
            ),
            "detail": detail,
        }
        tmp = self.path(name) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rec, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path(name))
        log(f"{name}: done in {rec['seconds']}s")
        if self.on_stage_done:
            self.on_stage_done(name)
        return rec


LOADS = []


def load_model(name, revision, device="cuda"):
    """Load a frozen model and record how long the card waited for it.

    Loading is work the card is billed for and the method does not need, and
    it depends on where the weights are stored, so it is kept apart from the
    stage's own time.
    """
    import torch
    from transformers import AutoModelForCausalLM

    t0 = time.time()
    # placed on the card shard by shard; going through host memory first
    # took four times as long from network storage
    m = AutoModelForCausalLM.from_pretrained(
        name,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    ).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    LOADS.append({"model": name, "seconds": round(time.time() - t0, 2)})
    log(f"loaded {name} in {LOADS[-1]['seconds']}s")
    return m


def record_loads(work):
    if not LOADS:
        return
    with open(os.path.join(work, "LOADS.jsonl"), "a") as f:
        while LOADS:
            f.write(json.dumps(LOADS.pop(0), sort_keys=True) + "\n")


def free(*models):
    import gc

    import torch

    for m in models:
        del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def stage_tokens(cfg, work):
    """Freeze the calibration rows: which documents, which tokens.

    The paper names the corpus and the counts and nothing else, so the
    selection here is ours and is recorded as a reconstruction: the first
    documents in stream order that are long enough, from a pinned revision.
    """
    import numpy as np
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["target"], revision=cfg["target_revision"])
    ds = load_dataset(
        cfg["calibration_dataset"],
        cfg["calibration_config"],
        split="train",
        streaming=True,
        revision=cfg.get("calibration_revision"),
    )
    T, N = cfg["sequence_length"], cfg["sequences"]
    rows, ids, seen = [], [], 0
    for doc in ds:
        seen += 1
        enc = tok.encode(doc["text"], add_special_tokens=False)
        if len(enc) < T:
            continue
        rows.append(enc[:T])
        ids.append(doc.get("id", str(seen)))
        if len(rows) == N:
            break
    arr = np.asarray(rows, dtype=np.int64)
    np.save(os.path.join(work, "calibration_tokens.npy"), arr)
    with open(os.path.join(work, "calibration_ids.json"), "w") as f:
        json.dump(ids, f)
    return {
        "sequences": int(arr.shape[0]),
        "documents_scanned": seen,
        "tokens_sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        "first_id": ids[0],
        "last_id": ids[-1],
    }


def stage_capture(cfg, work, which):
    import numpy as np
    import torch

    from . import attn_repair
    from . import capture as cap

    name, rev = cfg[which], cfg[f"{which}_revision"]
    model = load_model(name, rev, device_of(cfg))
    toks = torch.from_numpy(np.load(os.path.join(work, "calibration_tokens.npy")))
    bounds = None
    if which == "target":
        bounds, info = attn_repair.log_spaced_boundaries(
            cfg["boundaries"][0], cfg["boundaries"][1], cfg["boundaries"][2]
        )
    t0 = time.time()
    meta = cap.capture(
        model,
        toks,
        os.path.join(work, f"trace_{which}"),
        stride=cfg["stride"],
        batch_size=cfg["capture_batch"],
        boundaries=bounds,
    )
    secs = time.time() - t0
    free(model)
    record_loads(work)
    return {
        "geometry": meta["geometry"],
        "rows": meta["rows"],
        "tokens_per_second": round(toks.numel() / secs, 1),
        "boundaries_distinct": len(bounds) if bounds else None,
    }


def stage_select(cfg, work):
    import numpy as np

    from . import fit_pair

    sel, scores = fit_pair.select_layers(
        os.path.join(work, "trace_source"),
        os.path.join(work, "trace_target"),
        cfg["k"],
        lam=cfg["lam"],
        device=device_of(cfg),
    )
    np.save(os.path.join(work, "selector_scores.npy"), scores)
    with open(os.path.join(work, "selected_layers.json"), "w") as f:
        json.dump(sel, f)
    return {
        "best_probe_r2_mean": float(scores.max(1).mean()),
        "first_target_layer": sel[0],
        "last_target_layer": sel[-1],
    }


def stage_fit(cfg, work, method_id):
    from . import fit_pair
    from .methods import CACHE_BRIDGE

    with open(os.path.join(work, "selected_layers.json")) as f:
        sel = json.load(f)
    src, tgt = os.path.join(work, "trace_source"), os.path.join(work, "trace_target")
    weights, winfo = None, None
    if method_id == CACHE_BRIDGE:
        meta = fit_pair.load_meta(tgt)
        width = cfg["k"] * meta["geometry"]["head_dim"]
        weights, winfo = fit_pair.repair_weights(tgt, width)
    fitted = fit_pair.fit(
        method_id, src, tgt, sel, lam=cfg["lam"], device=device_of(cfg), weights=weights
    )
    path = os.path.join(work, f"mapper_{method_id}.safetensors")
    size = fit_pair.save_mapper(
        path, fitted, {"pair": f"{cfg['source']}->{cfg['target']}"}
    )
    out = {
        "feature_width": fitted["feature_width"],
        "coefficients": int(fitted["W"].numel() + fitted["b"].numel()),
        "file_bytes": size,
        "held_in_r2_keys_mean": float(fitted["r2"][:, 0].mean()),
        "held_in_r2_values_mean": float(fitted["r2"][:, 1].mean()),
        "held_in_r2_min": float(fitted["r2"].min()),
    }
    if winfo is not None:
        alphas = [v["alpha"] for v in winfo.values()]
        out["repair_alpha_mean"] = sum(alphas) / len(alphas)
        out["repair_maps_degenerated_to_unweighted"] = sum(
            1 for v in winfo.values() if v["degenerated_to_unweighted"]
        )
        out["repair_maps"] = len(alphas)
    return out


def systematic(n_docs, n_want):
    """Evenly spaced document indices, the same every time."""
    if n_want >= n_docs:
        return list(range(n_docs))
    step = n_docs // n_want
    return [i * step for i in range(n_want)]


def extract_outcomes(samples, metric):
    """Per-example results from the harness's sample log, by subject.

    Accuracy alone cannot say how many examples changed their answer between
    two arms, and that count is what a paired comparison is made of.
    """
    out = {}
    for leaf, rows in (samples or {}).items():
        got = {}
        for row in rows:
            if metric not in row:
                raise RuntimeError(
                    f"{leaf}: an example carries no {metric}; refusing to "
                    "treat a missing result as a wrong answer"
                )
            key = str(row["doc_id"])
            if key in got:
                raise RuntimeError(f"{leaf}: example {key} was scored twice")
            got[key] = float(row[metric])
        if got:
            out[leaf] = got
    if not out:
        raise RuntimeError("the harness returned no per-example results")
    return out


def stage_eval(cfg, work, task, mode, models):
    import inspect

    import lm_eval
    from lm_eval.tasks import TaskManager, get_task_dict
    from transformers import AutoTokenizer

    from . import fit_pair
    from .scoring import PrefixScorer, make_lm

    tok = AutoTokenizer.from_pretrained(cfg["target"], revision=cfg["target_revision"])
    mapper = None
    if mode in ("full_head", "cache_bridge"):
        mid = "full_head_mapping" if mode == "full_head" else "cache_bridge"
        mapper = fit_pair.load_mapper(os.path.join(work, f"mapper_{mid}.safetensors"))
    if mode == SOURCE_NATIVE:
        # the source model answering for itself, from its own cache: what
        # the receiver's route is measured against when the source is the
        # more capable model
        scorer = PrefixScorer(models["source"], "native")
    else:
        scorer = PrefixScorer(
            models["target"], mode, source=models.get("source"), mapper=mapper
        )
    lm = make_lm(scorer, tok)

    tm = TaskManager()
    tdict = get_task_dict([task], tm)
    leaves = {}

    def walk(d):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v)
            else:
                leaves[k if isinstance(k, str) else v.config.task] = v

    walk(tdict)
    per_leaf = max(1, -(-cfg["eval_examples"] // len(leaves)))
    samples = {
        name: systematic(len(t.eval_docs), per_leaf) for name, t in leaves.items()
    }
    kwargs = dict(
        model=lm,
        tasks=[task],
        num_fewshot=FEWSHOT.get(task),
        bootstrap_iters=0,
        log_samples=True,
        task_manager=tm,
    )
    if "samples" in inspect.signature(lm_eval.simple_evaluate).parameters:
        kwargs["samples"] = samples
        sampling = "systematic"
    else:
        kwargs["limit"] = per_leaf
        sampling = "first_n (this harness version has no explicit sample selection)"
    res = lm_eval.simple_evaluate(**kwargs)
    metric = METRIC[task]
    table = res["results"]
    score = None
    for key in (task, *table.keys()):
        row = table.get(key, {})
        for mk in (f"{metric},none", metric):
            if mk in row:
                score = float(row[mk])
                break
        if score is not None:
            break
    if score is None:
        raise RuntimeError(
            f"no {metric} reported for {task}; refusing to report a score"
        )
    n = sum(len(v) for v in samples.values())
    outcomes = extract_outcomes(res.get("samples"), metric)
    flat = [v for rows in outcomes.values() for v in rows.values()]
    pooled = sum(flat) / len(flat)
    if len(flat) != n or abs(pooled - score) > 1e-9:
        raise RuntimeError(
            f"{task}/{mode}: {len(flat)} per-example results averaging "
            f"{pooled:.6f} do not account for the reported {score:.6f} over "
            f"{n} examples"
        )
    odir = os.path.join(work, "outcomes")
    os.makedirs(odir, exist_ok=True)
    tmp = os.path.join(odir, f"{task}.{mode}.json.tmp")
    with open(tmp, "w") as f:
        json.dump(outcomes, f, sort_keys=True)
    os.replace(tmp, os.path.join(odir, f"{task}.{mode}.json"))
    return {
        "task": task,
        "mode": mode,
        "metric": metric,
        "score": score,
        "examples": n,
        "outcomes_recorded": len(flat),
        "leaves": len(leaves),
        "sampling": sampling,
        "num_fewshot": FEWSHOT.get(task, 0),
        "prefixes_built": scorer.prefixes,
        "prefix_tokens": scorer.prefix_tokens,
        "seconds_building_prefixes": round(scorer.seconds_prefix, 2),
        "seconds_scoring": round(scorer.seconds_scoring, 2),
    }


def run(cfg, work, on_stage_done=None, stop_after=None):
    import torch

    threads = min(usable_processors(), 16)
    torch.set_num_threads(threads)
    log(f"using {threads} threads of {os.cpu_count()} visible processors")
    os.makedirs(work, exist_ok=True)
    ident = identity(cfg)
    with open(os.path.join(work, "RUN_CONFIG.json"), "w") as f:
        json.dump({**cfg, "identity": ident}, f, indent=2, sort_keys=True)
    st = Stages(work, ident, on_stage_done)
    from .methods import CACHE_BRIDGE, FULL_HEAD

    plan = [
        ("tokens", lambda: stage_tokens(cfg, work)),
        ("capture_source", lambda: stage_capture(cfg, work, "source")),
        ("capture_target", lambda: stage_capture(cfg, work, "target")),
        ("select", lambda: stage_select(cfg, work)),
        ("fit_full_head", lambda: stage_fit(cfg, work, FULL_HEAD)),
        ("fit_cache_bridge", lambda: stage_fit(cfg, work, CACHE_BRIDGE)),
    ]
    for name, fn in plan:
        st.run(name, fn)
        if stop_after == name:
            return summarise(work)

    eident = eval_identity(cfg)
    pending = [
        (t, m)
        for t in cfg["tasks"]
        for m in cfg["modes"]
        if st.done(f"eval_{t}_{m}", eident) is None
    ]
    if pending:
        models = {}

        def load_both():
            dev = device_of(cfg)
            models["target"] = load_model(cfg["target"], cfg["target_revision"], dev)
            if any(
                m in ("full_head", "cache_bridge", SOURCE_NATIVE) for _, m in pending
            ):
                models["source"] = load_model(
                    cfg["source"], cfg["source_revision"], dev
                )
            return {"loaded": sorted(models)}

        # not skippable: the models have to be resident whenever any cell runs
        t0 = time.time()
        info = load_both()
        log(f"models resident in {time.time() - t0:.1f}s: {info['loaded']}")
        record_loads(work)
        for t, m in pending:
            st.run(
                f"eval_{t}_{m}",
                lambda t=t, m=m: stage_eval(cfg, work, t, m, models),
                ident=eident,
            )
    write_paired(cfg, work)
    return summarise(work)


def write_paired(cfg, work):
    """The paired comparison, when every arm it needs has been scored."""
    from . import paired

    tasks, modes = list(cfg["tasks"]), list(cfg["modes"])
    if paired.BASELINE not in modes or len(modes) < 2:
        return None
    try:
        out = paired.report(work, tasks, modes)
        if "full_head" in modes and "cache_bridge" in modes:
            out["between_arms"] = paired.between(
                work, tasks, "full_head", "cache_bridge"
            )
        if SOURCE_NATIVE in modes:
            # a second comparison, never a substitute for the first
            out["against_source"] = paired.report(
                work, tasks, modes, baseline=SOURCE_NATIVE
            )
    except FileNotFoundError:
        # receipts from a run that predates per-example results
        log("paired comparison skipped: per-example results are not on disk")
        return None
    with open(os.path.join(work, "PAIRED.json"), "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    return out


def summarise(work):
    sdir = os.path.join(work, "stages")
    recs = {}
    for fn in sorted(os.listdir(sdir)):
        if fn.endswith(".json"):
            with open(os.path.join(sdir, fn)) as f:
                recs[fn[:-5]] = json.load(f)
    scores = {}
    rates = {}
    for name, r in recs.items():
        if name.startswith("eval_"):
            d = r["detail"]
            scores.setdefault(d["task"], {})[d["mode"]] = d["score"]
            rates[name] = round(d["examples"] / max(r["seconds"], 1e-9), 3)
    # the transferred arms share the from-cache path with "native", so that
    # is the denominator that isolates the transfer; "direct" is kept beside
    # it because the two differ by the arithmetic's own noise
    retention, retention_direct = {}, {}
    for task, by in scores.items():
        for base_mode, table in (("native", retention), ("direct", retention_direct)):
            base = by.get(base_mode)
            if base:
                table[task] = {m: round(100.0 * v / base, 2) for m, v in by.items()}
    loads = []
    lp = os.path.join(work, "LOADS.jsonl")
    if os.path.exists(lp):
        with open(lp) as f:
            loads = [json.loads(line) for line in f if line.strip()]
    out = {
        "model_loads": loads,
        "model_load_seconds": round(sum(x["seconds"] for x in loads), 1),
        "stage_seconds": {k: v["seconds"] for k, v in recs.items()},
        "stage_peak_gpu_gib": {k: v["peak_gpu_gib"] for k, v in recs.items()},
        "total_stage_seconds": round(sum(v["seconds"] for v in recs.values()), 1),
        "scores": scores,
        "retention_percent_of_native": retention,
        "retention_percent_of_direct": retention_direct,
        "eval_examples_per_second": rates,
    }
    with open(os.path.join(work, "SUMMARY.json"), "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    return out


def default_config():
    return {
        "source": "Qwen/Qwen3-14B",
        "source_revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
        "target": "Qwen/Qwen3-32B",
        "target_revision": "9216db5781bf21249d130ec9da846c4624c16137",
        "sequences": 500,
        "sequence_length": 1024,
        "stride": 4,
        "k": 8,
        "lam": 0.01,
        "boundaries": [12, 1023, 32],
        "calibration_dataset": "HuggingFaceFW/fineweb-edu",
        "calibration_config": "sample-10BT",
        "calibration_revision": None,
        "capture_batch": 8,
        "tasks": list(DEFAULT_TASKS),
        "modes": list(MODES),
        "eval_examples": 500,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", required=True)
    ap.add_argument("--config", default="", help="JSON overriding the defaults")
    ap.add_argument("--stop-after", default=None)
    args = ap.parse_args()
    cfg = default_config()
    if args.config:
        with open(args.config) as f:
            cfg.update(json.load(f))
    out = run(cfg, args.work, stop_after=args.stop_after)
    print("SUMMARY " + json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
