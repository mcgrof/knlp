# SPDX-License-Identifier: GPL-2.0
"""Time the first next token three ways, at three context lengths.

The question is narrow. Given a prompt, how long until the next token is
decided, if the target does the whole prompt itself, versus if a translated
cache is installed instead? The answer depends on something the earlier
measurement left implicit: whether the source key/value cache already exists.
So two translated conditions are measured against one native control.

``target_native``
    The target processes all L prompt tokens and produces the final
    position's vocabulary logits and the next token. Nothing else.

``existing_gpu_source_kv_switch``
    The source key/value cache for the first L-1 tokens is already resident on
    this GPU, built before the timer starts. Inside the timer: de-rotation into
    the content frame, the mapping, re-rotation at the target's positions, the
    cast and layout change to the serving dtype, cache installation, and the
    target's final prompt token through the last-position head and the token
    decision.

``source_prefill_inclusive_translation``
    The same prompt tokens, nothing precomputed. Inside one timer: the source
    backbone over the first L-1 tokens producing its cache, then the entire
    switch path above. The source runs as a cache producer, without a
    vocabulary projection over the whole sequence, because a deployment that
    only needs the cache would not pay for one. A source workload that also
    needs logits would have to add that head work, and this measurement does
    not include it.

The boundary is stated because the earlier gate's boundary was not. That gate
computed the source prefix before starting its clock, so its translated figure
is this file's switch condition and not a source-inclusive one. Nothing here
is tuned to reproduce that number.

Primary measure is synchronized complete-call wall time. Model loading,
tokenisation and placement happen outside every timed region. All three paths
start from the same GPU-resident token ids and end with the next-token
decision available under the same policy.

Env: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.cartridges_cas.scripts.cas_kv_rope import derot, rerot  # noqa: E402
from research.kv_translate import freeze  # noqa: E402
from research.kv_translate.fit import SourceLayout  # noqa: E402
from research.kv_translate.pairs import describe  # noqa: E402
from research.kv_translate.run_a0 import make_cache  # noqa: E402
from research.kv_translate.run_a1 import flat_features  # noqa: E402

CONTRACT = "first_next_token_timing_v1"

BOUNDARY = {
    "target_native": "target over all L prompt tokens, final-position logits "
    "and token decision",
    "existing_gpu_source_kv_switch": "source K/V for L-1 tokens already "
    "resident; timer covers de-rotation, mapping, re-rotation, cast and "
    "layout, cache installation, the target's final prompt token through the "
    "last-position head, and the token decision",
    "source_prefill_inclusive_translation": "same prompt tokens; timer covers "
    "the source backbone over L-1 tokens as a cache producer without a "
    "full-sequence vocabulary projection, then the entire switch path",
    "outside_all_timers": "model load, tokenisation, placement, and the "
    "resident source cache used by the switch condition",
    "note": "an earlier gate computed the source prefix before starting its "
    "clock, so its translated figure is this file's switch condition and not a "
    "source-inclusive one",
}


def sha_file(p):
    try:
        return hashlib.sha256(open(p, "rb").read()).hexdigest()
    except Exception:
        return None


def _git(*a):
    import subprocess

    try:
        return subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), *a],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except Exception:
        return ""


# Captured at import, so a run records the tree it ran from rather than the
# tree someone checks out later. An earlier snapshot recorded a revision whose
# working tree did not contain the runners that produced the results beside it.
REVISION = _git("rev-parse", "HEAD") or "unknown"
DIRTY = bool(_git("status", "--porcelain", "--", ".").strip())
CONDITIONS = (
    "target_native",
    "existing_gpu_source_kv_switch",
    "source_prefill_inclusive_translation",
)
# Predeclared before any sample is taken. Stated here rather than chosen after
# looking at the numbers.
ANALYSIS = {
    "primary_measure": "synchronized complete-call wall time, perf_counter",
    "warmups_per_fixture_path": 3,
    "measured_repetitions_per_fixture_path": 20,
    "schedule": "fixed recorded order, paired and interleaved within each rep",
    "resample_draws": 2000,
    "resample_seed": 0,
    "resample_unit": "paired repetition index across all fixtures of a length",
    "interval": "percentile 2.5 and 97.5 of the resampled ratio of medians",
    "interval_status": "descriptive of repeat timing on these fixed fixtures; "
    "not a workload-population interval and not a coverage guarantee",
    "supported_saving": "upper endpoint of that interval strictly below 1",
    "inherited_p50_threshold": 0.75,
    "inherited_tail_ratio": "translated p95 divided by native p50, threshold "
    "strictly below 1; this is not a p95 over p95 comparison",
}


def log_now(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def paired_ratio_interval(t_ms, n_ms, draws, seed):
    """Resample paired repetitions and describe the spread of the ratio.

    Paired because every translated repetition was taken beside a native one
    under the same schedule, so the pair is the unit that repeats. Resampling
    the two paths independently would discard that and describe a different,
    looser quantity.
    """
    t, n = np.asarray(t_ms, float), np.asarray(n_ms, float)
    assert t.shape == n.shape, "paired samples must be the same length"
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, t.size, size=(draws, t.size))
    r = np.median(t[idx], axis=1) / np.median(n[idx], axis=1)
    return {
        "point": float(np.median(t) / np.median(n)),
        "lo": float(np.percentile(r, 2.5)),
        "hi": float(np.percentile(r, 97.5)),
        "draws": int(draws),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--fixtures", required=True, help="timing_fixtures_v1 file")
    ap.add_argument("--qualification", required=True, help="passed receipt json")
    ap.add_argument("--lengths", default="512,2048,4096")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmups", type=int, default=3)
    ap.add_argument("--dtype", default="bfloat16", help="model and cache")
    ap.add_argument("--solve-dtype", default="float32", help="mapper and features")
    ap.add_argument("--deadline-seconds", type=float, default=0.0)
    ap.add_argument(
        "--extra-block",
        action="store_true",
        help="take the one additional block the plan permits, for a condition "
        "whose interval spans 1 or 0.75 and therefore decides nothing",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="exercise the same timed callables on whatever device is present, "
        "to validate plumbing before renting one. The output is stamped with a "
        "different contract so it can never be read as a measurement.",
    )
    ap.add_argument("--experiment-id", default="", help="binds attempts together")
    ap.add_argument("--attempt-id", default="")
    ap.add_argument(
        "--started-at",
        type=float,
        default=0.0,
        help="unix seconds when the ALLOCATION was requested. The deadline is "
        "measured from here, not from this process, because the cap is spent "
        "on occupied allocation time and provisioning is part of it.",
    )
    ap.add_argument(
        "--require-artifact-sha256",
        default="",
        help="the joint weight hash the plan names. Checked in addition to "
        "agreement with the qualification receipt, because a receipt only "
        "proves the timed artifact is the qualified one, not that either is "
        "the artifact this diagnostic was authorised to measure.",
    )
    ap.add_argument("--source-revision", default="main")
    ap.add_argument("--target-revision", default="main")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t_start = time.time()
    clock_origin = args.started_at if args.started_at > 0 else t_start
    stages = {}

    # ---- hard prerequisite: a passed qualification receipt ----------------
    # A missing, malformed or failing receipt stops the run before any model
    # is loaded. Timing an operator that has not been shown to compute the
    # right thing measures the speed of an unknown function.
    try:
        q = json.load(open(args.qualification))
    except Exception as e:
        raise SystemExit(f"PREREQUISITE: qualification receipt unreadable: {e}")
    # Ordered cheapest-and-most-identifying first: what contract is this, did
    # it pass, then the fields that contract implies. A receipt from another
    # contract should be rejected for being the wrong contract, not for
    # missing a field that only this contract defines.
    if "contract" not in q:
        raise SystemExit("PREREQUISITE: receipt has no 'contract'")
    if q["contract"] != "saved_operator_v1":
        raise SystemExit(f"PREREQUISITE: wrong contract {q['contract']!r}")
    if "passed" not in q:
        raise SystemExit("PREREQUISITE: receipt has no 'passed'")
    if q["passed"] is not True:
        raise SystemExit("PREREQUISITE: qualification did not pass; not timing")
    for field in ("checks", "contract_limits"):
        if field not in q:
            raise SystemExit(f"PREREQUISITE: receipt has no {field!r}")
    # Every field this run will later dereference is checked here, while the
    # cost of stopping is a few seconds. Discovering a thin receipt after two
    # models are resident on a rented card spends allocation time to learn
    # something that was knowable before the allocation was touched at all.
    try:
        qualified_sha = q["checks"]["artifact"]["joint_weight_sha256"]
    except (KeyError, TypeError):
        raise SystemExit(
            "PREREQUISITE: receipt carries no checks.artifact.joint_weight_sha256, "
            "so the timed artifact cannot be bound to a qualified one"
        )
    if not isinstance(qualified_sha, str) or len(qualified_sha) != 64:
        raise SystemExit(f"PREREQUISITE: malformed qualified hash {qualified_sha!r}")
    # A receipt earned under looser limits is not this policy's receipt.
    expected_limits = {
        "operator_matches_reference": 1e-6,
        "operator_is_deterministic": 0.0,
        "serving_cast_within_one_step": 1.001,
    }
    for k, want in expected_limits.items():
        got = q["contract_limits"].get(k)
        if got is None or float(got) > float(want):
            raise SystemExit(
                f"PREREQUISITE: receipt limit {k}={got} is absent or looser than "
                f"the declared {want}"
            )
    for k in expected_limits:
        c = q["checks"].get(k)
        if not isinstance(c, dict) or c.get("passed") is not True:
            raise SystemExit(f"PREREQUISITE: receipt check {k!r} did not pass")
    if q.get("tf32_allowed") not in (False, None):
        raise SystemExit("PREREQUISITE: receipt was earned with TF32 enabled")
    log_now(
        f"qualification receipt accepted: {args.qualification} "
        f"(artifact {qualified_sha[:16]}, all declared checks passed)"
    )

    from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache

    if not torch.cuda.is_available() and not args.dry_run:
        raise SystemExit("PREREQUISITE: no CUDA device")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    sdtype = getattr(torch, args.solve_dtype)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if (
        q.get("operator_dtype") != args.solve_dtype
        or q.get("serving_dtype") != args.dtype
    ):
        raise SystemExit(
            f"PREREQUISITE: receipt precision "
            f"({q.get('operator_dtype')}/{q.get('serving_dtype')}) differs from "
            f"the timed precision ({args.solve_dtype}/{args.dtype})"
        )
    live_gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
    if not args.dry_run and q.get("gpu") != live_gpu:
        raise SystemExit(
            f"PREREQUISITE: qualified on {q.get('gpu')!r} but timing on "
            f"{live_gpu!r}; a qualification belongs to the device that earned it"
        )

    lengths = [int(x) for x in args.lengths.split(",")]

    t0 = time.time()
    fx = torch.load(args.fixtures, map_location="cpu", weights_only=False)
    assert fx["contract"] == "timing_fixtures_v1", "unexpected fixture contract"
    for f in fx["fixtures"]:
        h = hashlib.sha256(f["ids"].numpy().tobytes()).hexdigest()
        if h != f["sha256"]:
            raise SystemExit(f"PREREQUISITE: fixture {f['index']} hash mismatch")
        if f["n_tokens"] < max(lengths):
            raise SystemExit(f"PREREQUISITE: fixture {f['index']} too short")
    log_now(f"{len(fx['fixtures'])} fixtures verified, joint {fx['joint_sha256'][:16]}")

    src = AutoModelForCausalLM.from_pretrained(
        args.source,
        revision=args.source_revision,
        dtype=dtype,
        attn_implementation="sdpa",
    ).to(dev)
    src.eval()
    sg = describe(src, args.source)
    tgt = AutoModelForCausalLM.from_pretrained(
        args.target,
        revision=args.target_revision,
        dtype=dtype,
        attn_implementation="sdpa",
    ).to(dev)
    tgt.eval()
    tg = describe(tgt, args.target)
    layout = SourceLayout(sg.n_layers, sg.n_kv_heads, sg.head_dim)
    mk, mv, man = freeze.load(args.artifact, layout, tg, device=dev)
    mk.cast(sdtype)
    mv.cast(sdtype)
    if man["joint_weight_sha256"] != qualified_sha:
        raise SystemExit("PREREQUISITE: artifact differs from the qualified one")
    if (
        args.require_artifact_sha256
        and man["joint_weight_sha256"] != args.require_artifact_sha256
    ):
        raise SystemExit(
            f"PREREQUISITE: artifact {man['joint_weight_sha256'][:16]} is not the "
            f"required {args.require_artifact_sha256[:16]}"
        )
    receipt_gpu = q.get("gpu")
    live_gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
    if not args.dry_run and receipt_gpu != live_gpu:
        raise SystemExit(
            f"PREREQUISITE: qualified on {receipt_gpu!r} but timing on "
            f"{live_gpu!r}; a qualification belongs to the device that earned it"
        )
    if (
        q.get("operator_dtype") != args.solve_dtype
        or q.get("serving_dtype") != args.dtype
    ):
        raise SystemExit(
            f"PREREQUISITE: receipt precision "
            f"({q.get('operator_dtype')}/{q.get('serving_dtype')}) differs from "
            f"the timed precision ({args.solve_dtype}/{args.dtype})"
        )
    if q.get("tf32_allowed") not in (False, None):
        raise SystemExit("PREREQUISITE: receipt was earned with TF32 enabled")
    stages["load_seconds"] = time.time() - t0
    log_now(
        f"loaded in {stages['load_seconds']:.1f}s; artifact "
        f"{man['joint_weight_sha256'][:16]} matches the qualified one"
    )

    # ---- the three timed callables ---------------------------------------
    @torch.no_grad()
    def native(ids, _resident, ctx):
        logits = tgt(input_ids=ids, logits_to_keep=1).logits[:, -1]
        return logits, logits.argmax(-1)

    @torch.no_grad()
    def switch(ids, resident, ctx):
        keys_post, values, pos_head, last = resident
        xk = flat_features(
            [derot(k.float(), pos_head, sg.rope_theta) for k in keys_post]
        ).to(sdtype)
        xv = flat_features([v.float() for v in values]).to(sdtype)
        keys = [
            rerot(b[0].float(), pos_head, tg.rope_theta).unsqueeze(0).to(dtype)
            for b in mk.apply(xk)
        ]
        vals = [v.to(dtype) for v in mv.apply(xv)]
        logits = tgt(
            input_ids=last,
            attention_mask=torch.ones(1, ctx, dtype=torch.long, device=dev),
            position_ids=torch.arange(ctx - 1, ctx, device=dev).unsqueeze(0),
            past_key_values=make_cache(keys, vals),
            use_cache=True,
        ).logits[:, -1]
        return logits, logits.argmax(-1)

    @torch.no_grad()
    def source_inclusive(ids, resident, ctx):
        # The source runs as a cache producer: the backbone only, so no
        # vocabulary projection over the whole sequence is charged to a path
        # that never reads one.
        cache = DynamicCache()
        src.model(input_ids=ids[:, : ctx - 1], past_key_values=cache, use_cache=True)
        keys_post = [l.keys for l in cache.layers]
        values = [l.values for l in cache.layers]
        pos_head = torch.arange(ctx - 1, device=dev)
        return switch(ids, (keys_post, values, pos_head, ids[:, ctx - 1 :]), ctx)

    paths = {
        "target_native": native,
        "existing_gpu_source_kv_switch": switch,
        "source_prefill_inclusive_translation": source_inclusive,
    }

    def sync():
        if dev == "cuda":
            torch.cuda.synchronize()

    def timed_call(fn, ids, resident, ctx):
        sync()
        t = time.perf_counter()
        out = fn(ids, resident, ctx)
        sync()
        return (time.perf_counter() - t) * 1000.0, out

    def verify_paths(ids, resident, ctx):
        """Check the three timed callables before timing them.

        The spec asks for positions, output shapes and finite values to be
        checked rather than asserted in prose. A switch path that installs a
        cache of the wrong length, or feeds the wrong position, still returns
        a token of the right shape at the right speed, and the timing would
        look fine while measuring the wrong computation.

        Each condition is exercised through the same callable the timer will
        call. Re-implementing the switch body here would check a copy, and a
        copy can satisfy the spec while the timed path quietly does not.
        """
        keys_post, values, pos_head, last = resident
        checks = {
            "resident_cache_tokens": int(keys_post[0].shape[2]),
            "expected_cache_tokens": ctx - 1,
            "fed_position": ctx - 1,
            "fed_token_count": int(last.shape[1]),
            "source_layers": len(keys_post),
            "target_layers": tg.n_layers,
            "conditions_exercised": [],
        }
        checks["cache_length_correct"] = (
            checks["resident_cache_tokens"] == checks["expected_cache_tokens"]
        )
        checks["single_token_fed"] = checks["fed_token_count"] == 1

        # The installed cache length is a property of the mapping, so it is
        # read from the mapper's own output rather than from the callable.
        with torch.no_grad():
            installed = mk.apply(
                flat_features(
                    [derot(k.float(), pos_head, sg.rope_theta) for k in keys_post]
                ).to(sdtype)
            )
        checks["installed_cache_tokens"] = int(installed[0].shape[2])
        checks["installed_layers"] = len(installed)
        del installed

        out = {}
        for name in CONDITIONS:
            logits, tok = paths[name](ids, resident, ctx)
            out[name] = (logits, tok)
            checks["conditions_exercised"].append(name)
            checks[f"{name}_logits_shape"] = list(logits.shape)
            checks[f"{name}_token_shape"] = list(tok.shape)
            checks[f"{name}_logits_finite"] = bool(torch.isfinite(logits).all())
        shapes_seen = {tuple(checks[f"{c}_logits_shape"]) for c in CONDITIONS}
        checks["shapes_agree"] = len(shapes_seen) == 1
        # Not a quality claim. Recorded because a single agreeing argmax is
        # sometimes mistaken for one, and it is cheaper to say so here.
        checks["argmax_agrees_native_vs_switch"] = bool(
            int(out["target_native"][1]) == int(out["existing_gpu_source_kv_switch"][1])
        )
        checks["argmax_agrees_switch_vs_source_inclusive"] = bool(
            int(out["existing_gpu_source_kv_switch"][1])
            == int(out["source_prefill_inclusive_translation"][1])
        )
        checks["argmax_agreement_is_not_quality"] = True
        del out

        bad = [
            k
            for k in (
                ["cache_length_correct", "single_token_fed", "shapes_agree"]
                + [f"{c}_logits_finite" for c in CONDITIONS]
            )
            if not checks[k]
        ]
        if bad:
            raise SystemExit(f"PREFLIGHT at ctx={ctx}: {bad} failed: {checks}")
        if checks["installed_cache_tokens"] != ctx - 1:
            raise SystemExit(
                f"PREFLIGHT at ctx={ctx}: installed cache is "
                f"{checks['installed_cache_tokens']} tokens, expected {ctx - 1}"
            )
        return checks

    raw_path = args.out.replace(".json", ".raw.jsonl")

    def cfg_hash(model_id, revision):
        c = AutoConfig.from_pretrained(model_id, revision=revision)
        return hashlib.sha256(c.to_json_string().encode()).hexdigest()

    def base_record():
        """Everything about the run that does not depend on the samples.

        Written with every partial flush, so a run that stops early still
        carries its identity, its pinning and its provenance rather than
        leaving a bag of numbers nobody can attribute.
        """
        return {
            "contract": CONTRACT + ("_DRY_RUN" if args.dry_run else ""),
            "dry_run": bool(args.dry_run),
            "dry_run_note": (
                "Plumbing validation only. Timings from a dry run are discarded "
                "and never reported as a measurement."
                if args.dry_run
                else None
            ),
            "identity": {
                "experiment_id": args.experiment_id or "unset",
                "attempt_id": args.attempt_id or "unset",
                "clock_origin_unix": clock_origin,
                "clock_origin_is": (
                    "the allocation request"
                    if args.started_at > 0
                    else "this process start; the allocation clock was not passed in"
                ),
                "deadline_seconds_from_origin": args.deadline_seconds or None,
            },
            "code": {
                "revision": REVISION,
                "tree_dirty": DIRTY,
                "runner_sha256": sha_file(__file__),
                "qualifier_sha256": sha_file(
                    os.path.join(os.path.dirname(__file__), "qualify_operator.py")
                ),
            },
            "analysis_policy": {
                **ANALYSIS,
                # Bound to what this invocation actually did, not to the
                # constants above. A policy that cannot disagree with the run
                # records an intention, not a method.
                "warmups_per_fixture_path": args.warmups,
                "measured_repetitions_per_fixture_path": args.reps,
                "n_fixtures": len(fx["fixtures"]),
                "declared_defaults_match_run": bool(
                    args.warmups == ANALYSIS["warmups_per_fixture_path"]
                    and args.reps == ANALYSIS["measured_repetitions_per_fixture_path"]
                ),
            },
            "boundary": BOUNDARY,
            "device": {
                "gpu": (
                    torch.cuda.get_device_name(0) if dev == "cuda" else "cpu (dry run)"
                ),
                "capability": (
                    list(torch.cuda.get_device_capability(0)) if dev == "cuda" else None
                ),
                "total_bytes": (
                    torch.cuda.get_device_properties(0).total_memory
                    if dev == "cuda"
                    else None
                ),
                "peak_alloc_bytes": (
                    int(torch.cuda.max_memory_allocated()) if dev == "cuda" else None
                ),
                "torch": torch.__version__,
                "tf32_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
            },
            "pinning": {
                "source": args.source,
                "target": args.target,
                "source_revision": args.source_revision,
                "target_revision": args.target_revision,
                "source_config_sha256": cfg_hash(args.source, args.source_revision),
                "target_config_sha256": cfg_hash(args.target, args.target_revision),
                "tokenizer": fx.get("tokenizer"),
                "batch_size": 1,
                "attn_implementation": "sdpa",
                "model_and_cache_dtype": args.dtype,
                "mapper_and_feature_dtype": args.solve_dtype,
                "cache_layout": "B,n_kv_heads,T,head_dim",
                "key_frame": "content; re-rotated at the target position on apply",
                "positions": "native 0..L-1; switch installs L-1 and feeds L-1",
                "output_policy": "final-position logits, argmax, left on device",
            },
            "artifact": {
                "path": os.path.abspath(args.artifact),
                "joint_weight_sha256": man["joint_weight_sha256"],
                "n_blocks": man["n_blocks"],
                "required_sha256": args.require_artifact_sha256 or None,
                "matches_required": (
                    man["joint_weight_sha256"] == args.require_artifact_sha256
                    if args.require_artifact_sha256
                    else None
                ),
            },
            "fixtures": {
                "joint_sha256": fx["joint_sha256"],
                "role": fx["role"],
                "split": fx["split"],
                "per_fixture": [
                    {
                        "index": f["index"],
                        "sha256": f["sha256"],
                        "n_tokens": f["n_tokens"],
                    }
                    for f in fx["fixtures"]
                ],
            },
            "qualification_receipt": {
                "path": os.path.abspath(args.qualification),
                "sha256": sha_file(args.qualification),
                "passed": q["passed"],
                "limits": q.get("contract_limits"),
                "measurements": {
                    k: v.get("measured")
                    for k, v in q.get("checks", {}).items()
                    if isinstance(v, dict) and "measured" in v
                },
                "qualified_on_gpu": q.get("gpu"),
                "operator_dtype": q.get("operator_dtype"),
                "serving_dtype": q.get("serving_dtype"),
                "tf32_allowed": q.get("tf32_allowed"),
                "limits_are": "a declared diagnostic acceptance policy, not a "
                "mathematical error bound",
            },
            "stages_seconds": stages,
        }

    results, raw, stopped_after = {}, [], None

    def _flush(results, raw):
        partial = dict(base_record())
        partial["results"] = results
        partial["complete"] = False
        with open(args.out, "w") as fh:
            json.dump(partial, fh, indent=2, sort_keys=True, default=str)
        with open(raw_path, "w") as fh:
            for r in raw:
                fh.write(json.dumps(r) + "\n")

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for ctx in lengths:
        t_len = time.time()
        per_path = {c: [] for c in CONDITIONS}
        # Each translated sample keeps the native sample measured beside it in
        # the same repetition. Without that, taking an extra block for one
        # condition would leave the others compared against a native series of
        # a different length, and the pairing the interval relies on would be
        # silently broken.
        paired_native = {c: [] for c in CONDITIONS if c != "target_native"}
        blocks_taken = {c: 1 for c in CONDITIONS if c != "target_native"}
        per_fixture = {}
        # Timings from perf_counter are finite by construction, so asserting
        # that they are says nothing. What can go non-finite is the model
        # output, so that is what is checked, on every repetition.
        outputs_finite, shapes, logit_shapes, nonfinite = True, {}, {}, []
        for f in fx["fixtures"]:
            ids = f["ids"][:ctx].unsqueeze(0).to(dev)
            # Built outside every timer. This is exactly what the switch
            # condition assumes it is given for free.
            sync()
            t_res = time.time()
            cache = DynamicCache()
            with torch.no_grad():
                src.model(
                    input_ids=ids[:, : ctx - 1], past_key_values=cache, use_cache=True
                )
            resident = (
                [l.keys for l in cache.layers],
                [l.values for l in cache.layers],
                torch.arange(ctx - 1, device=dev),
                ids[:, ctx - 1 :],
            )
            sync()
            stages.setdefault("resident_build_seconds", 0.0)
            stages["resident_build_seconds"] += time.time() - t_res
            stages["resident_build_note"] = (
                "the source prefill the switch condition is given for free, "
                "measured with the device synchronised so it is work and not "
                "kernel-launch time; excluded from every timer by design"
            )

            if f["index"] == 0:
                preflight = verify_paths(ids, resident, ctx)
                log_now(
                    f"  preflight ctx={ctx}: cache "
                    f"{preflight['installed_cache_tokens']} tokens, position "
                    f"{preflight['fed_position']}, "
                    f"{len(preflight['conditions_exercised'])}/3 conditions "
                    f"exercised, logits finite, shapes agree"
                )
            for name in CONDITIONS:
                for _ in range(args.warmups):
                    paths[name](ids, resident, ctx)
            sync()

            for rep in range(args.reps):
                nat_ms = None
                for name in CONDITIONS:  # fixed recorded order, interleaved
                    ms, (logits, out) = timed_call(paths[name], ids, resident, ctx)
                    per_path[name].append(ms)
                    if name == "target_native":
                        nat_ms = ms
                    else:
                        paired_native[name].append(nat_ms)
                    raw.append(
                        {
                            "length": ctx,
                            "fixture": f["index"],
                            "condition": name,
                            "rep": rep,
                            "block": 1,
                            "ms": ms,
                            "paired_native_ms": (
                                None if name == "target_native" else nat_ms
                            ),
                            "token": int(out),
                        }
                    )
                    shapes[name] = list(out.shape)
                    logit_shapes[name] = list(logits.shape)
                    if not bool(torch.isfinite(logits).all()):
                        outputs_finite = False
                        nonfinite.append(
                            {"length": ctx, "fixture": f["index"], "condition": name}
                        )
            for name in CONDITIONS:
                got = [
                    r["ms"]
                    for r in raw
                    if r["length"] == ctx
                    and r["fixture"] == f["index"]
                    and r["condition"] == name
                ]
                per_fixture.setdefault(str(f["index"]), {})[name] = {
                    "n": len(got),
                    "p50_ms": float(np.percentile(got, 50)),
                    "p95_ms": float(np.percentile(got, 95)),
                }
            del resident, cache
            if dev == "cuda":
                torch.cuda.empty_cache()

        def crossing(name):
            iv = paired_ratio_interval(
                per_path[name],
                paired_native[name],
                ANALYSIS["resample_draws"],
                ANALYSIS["resample_seed"],
            )
            return iv["lo"] < 1.0 < iv["hi"] or iv["lo"] < 0.75 < iv["hi"]

        need = [c for c in paired_native if crossing(c)]
        if need and args.extra_block:
            log_now(f"  interval uninformative for {need}; taking one extra block")
            for f in fx["fixtures"]:
                ids = f["ids"][:ctx].unsqueeze(0).to(dev)
                cache2 = DynamicCache()
                with torch.no_grad():
                    src.model(
                        input_ids=ids[:, : ctx - 1],
                        past_key_values=cache2,
                        use_cache=True,
                    )
                resident = (
                    [l.keys for l in cache2.layers],
                    [l.values for l in cache2.layers],
                    torch.arange(ctx - 1, device=dev),
                    ids[:, ctx - 1 :],
                )
                for _ in range(args.warmups):
                    for name in ["target_native"] + need:
                        paths[name](ids, resident, ctx)
                sync()
                for rep in range(args.reps):
                    nat_ms, _nat_out = timed_call(
                        paths["target_native"], ids, resident, ctx
                    )
                    for name in need:
                        ms, (logits, out) = timed_call(paths[name], ids, resident, ctx)
                        if not bool(torch.isfinite(logits).all()):
                            outputs_finite = False
                            nonfinite.append(
                                {
                                    "length": ctx,
                                    "fixture": f["index"],
                                    "condition": name,
                                }
                            )
                        per_path[name].append(ms)
                        paired_native[name].append(nat_ms)
                        raw.append(
                            {
                                "length": ctx,
                                "fixture": f["index"],
                                "condition": name,
                                "rep": rep,
                                "block": 2,
                                "ms": ms,
                                "paired_native_ms": nat_ms,
                                "token": int(out),
                            }
                        )
                del resident, cache2
                if dev == "cuda":
                    torch.cuda.empty_cache()
            for name in need:
                blocks_taken[name] = 2

        nat = per_path["target_native"]
        summary = {
            "preflight": preflight,
            "n_samples_per_condition": len(nat),
            "model_logits_all_finite": bool(outputs_finite),
            "nonfinite_occurrences": nonfinite,
            "logit_shapes": logit_shapes,
            "per_fixture": per_fixture,
            "output_shapes": shapes,
            "conditions": {},
        }
        for name in CONDITIONS:
            s = per_path[name]
            entry = {
                "p50_ms": float(np.percentile(s, 50)),
                "p95_ms": float(np.percentile(s, 95)),
                "mean_ms": float(np.mean(s)),
                "n": len(s),
            }
            if name != "target_native":
                pn = paired_native[name]
                entry["blocks_taken"] = blocks_taken[name]
                entry["paired_native_p50_ms"] = float(np.percentile(pn, 50))
                entry["ratio_p50_over_native_p50"] = entry["p50_ms"] / float(
                    np.percentile(pn, 50)
                )
                entry["ratio_translated_p95_over_native_p50"] = entry["p95_ms"] / float(
                    np.percentile(pn, 50)
                )
                entry["ratio_definitions"] = {
                    "ratio_p50_over_native_p50": "translated p50 / native p50",
                    "ratio_translated_p95_over_native_p50": "translated p95 / "
                    "native p50, the inherited tail criterion; this is not p95/p95",
                }
                entry["paired_interval"] = paired_ratio_interval(
                    s, pn, ANALYSIS["resample_draws"], ANALYSIS["resample_seed"]
                )
                if blocks_taken[name] > 1:
                    entry["paired_interval"]["after_adaptive_sampling"] = True
                hi = entry["paired_interval"]["hi"]
                entry["supported_saving"] = bool(hi < 1.0)
                entry["meets_inherited_p50"] = bool(
                    entry["ratio_p50_over_native_p50"] <= 0.75
                )
                entry["meets_inherited_tail"] = bool(
                    entry["ratio_translated_p95_over_native_p50"] < 1.0
                )
                entry["interval_crosses_one_or_threshold"] = bool(
                    entry["paired_interval"]["lo"] < 1.0 < hi
                    or entry["paired_interval"]["lo"] < 0.75 < hi
                )
            summary["conditions"][name] = entry
        results[str(ctx)] = summary
        stages[f"length_{ctx}_seconds"] = time.time() - t_len
        for name in CONDITIONS:
            e = summary["conditions"][name]
            log_now(
                f"  {ctx:>5} {name:<38s} p50 {e['p50_ms']:8.2f} ms"
                + (
                    ""
                    if name == "target_native"
                    else f"  ratio {e['ratio_p50_over_native_p50']:.3f} "
                    f"[{e['paired_interval']['lo']:.3f},"
                    f"{e['paired_interval']['hi']:.3f}] "
                    f"saving={e['supported_saving']}"
                )
            )
        # Serialised now rather than after the last length. An out-of-memory
        # failure at the longest context would otherwise discard the shorter
        # lengths that were already measured and already paid for.
        _flush(results, raw)
        if args.deadline_seconds and time.time() - clock_origin > args.deadline_seconds:
            log_now("deadline reached; remaining lengths left unmeasured")
            stopped_after = ctx
            break

    out = dict(base_record())
    out["results"] = results
    out["complete"] = True
    out["deadline_stopped_after_length"] = stopped_after
    out["lengths_requested"] = lengths
    out["lengths_measured"] = sorted(int(k) for k in results)
    out["wall_seconds_this_process"] = time.time() - t_start
    out["wall_seconds_note"] = (
        "this process only. It is not occupied allocation time, which includes "
        "provisioning, transfer, qualification, idle and teardown and is "
        "recorded by the allocation record, not here."
    )
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True, default=str)
    with open(raw_path, "w") as fh:
        for r in raw:
            fh.write(json.dumps(r) + "\n")
    # Recorded inside the artifact, so the receipt does not have to guess
    # where the samples went or whether they changed afterwards.
    out["outputs"] = {
        "summary_path": os.path.abspath(args.out),
        "raw_samples_path": os.path.abspath(raw_path),
        "raw_sample_count": len(raw),
        "raw_sha256": hashlib.sha256(open(raw_path, "rb").read()).hexdigest(),
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True, default=str)
    log_now(
        f"wrote {args.out} ({len(raw)} raw samples) in "
        f"{out['wall_seconds_this_process']:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
