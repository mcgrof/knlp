#!/usr/bin/env python3
"""Forecast the receiver-initializer pilot in occupied time, and block if it does not fit.

A reserve measured in occupied time has to be compared against a forecast in
occupied time. The previous accounting in this lane summed stage timers and
compared them against an allocation cap, and understated consumption by roughly
a factor of two; the one allocation since that was clocked create-to-terminate
came in thirty-nine percent above its stage sum. So this refuses to guess the
multiplier: ``--occupancy-multiplier`` has no default, because
``inclusive_forecast`` silently treats a missing one as 1.0, which is exactly
the error it exists to prevent.

It likewise refuses to guess the generation rate. The pilot's evaluation leg is
three hundred and twenty generations -- three initializers plus a floor, across
five checkpoints, over twenty questions -- and its cost is set entirely by how
fast the receiver decodes, which is a property of the device and the attention
path, not of the plan. The previous screen decoded a four-billion-parameter
model at 8.9 tokens per second on an eager path; a compiled path on a faster
card is an order of magnitude away from that. A forecast that assumes either
one is an assertion, so ``--tokens-per-second`` must be measured on the actual
receiver and passed in.

The remaining reserve is reported as an upper bound and never as a budget. The
queue records ``exact_remaining_seconds: null`` and
``failed_A10_and_setup_accounted: false``, so the figure can only shrink.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from research.kv_translate.stage_control import inclusive_forecast  # noqa: E402

RESERVE_TOTAL_S = 14400.0
RECORDED_USED_S = 2914.6
EVAL_GENERATIONS = 320
EVAL_TOKEN_CEILING = 163840
BLOCKED = "CARTRIDGE_RESOURCE_BLOCKED"


def generation_seconds(tokens_per_second, generations, mean_tokens, ceiling_tokens):
    """The evaluation leg, priced two ways.

    The ceiling is what the declared budget permits; the mean is what the
    previous screen actually produced per generation. Both are reported,
    because a forecast against the mean that omits the ceiling is a forecast
    that fails the first time the model rambles.
    """
    at_mean = generations * mean_tokens / tokens_per_second
    at_ceiling = ceiling_tokens / tokens_per_second
    return {
        "tokens_per_second": tokens_per_second,
        "generations": generations,
        "mean_tokens_per_generation": mean_tokens,
        "seconds_at_mean_length": round(at_mean, 1),
        "token_ceiling": ceiling_tokens,
        "seconds_at_token_ceiling": round(at_ceiling, 1),
    }


def build(
    tokens_per_second,
    occupancy_multiplier,
    *,
    mean_tokens=432,
    fit_s=0.0,
    train_s=0.0,
    setup=0.0,
    transfers=0.0,
    diagnosis=0.0,
    export=0.0,
    already_used=RECORDED_USED_S,
    cap=RESERVE_TOTAL_S,
    generations=EVAL_GENERATIONS,
    ceiling_tokens=EVAL_TOKEN_CEILING,
):
    gen = generation_seconds(
        tokens_per_second, generations, mean_tokens, ceiling_tokens
    )
    stages = {
        "fit": float(fit_s),
        "train": float(train_s),
        "evaluate": gen["seconds_at_token_ceiling"],
    }
    fc = inclusive_forecast(
        stages,
        setup=setup,
        transfers=transfers,
        diagnosis=diagnosis,
        export=export,
        already_used=already_used,
        cap=cap,
        occupancy_multiplier=occupancy_multiplier,
    )
    optimistic = inclusive_forecast(
        {**stages, "evaluate": gen["seconds_at_mean_length"]},
        setup=setup,
        transfers=transfers,
        diagnosis=diagnosis,
        export=export,
        already_used=already_used,
        cap=cap,
        occupancy_multiplier=occupancy_multiplier,
    )
    return {
        "contract": "c2_forecast_v1",
        "generation": gen,
        "forecast_at_token_ceiling": fc,
        "forecast_at_mean_length": optimistic,
        "fits": bool(fc["fits"]),
        "fits_only_at_mean_length": bool(optimistic["fits"] and not fc["fits"]),
        "remaining_is_an_upper_bound": True,
        "remaining_note": (
            "the reserve records no exact remaining figure and does not yet "
            "account for failed acquisitions or setup, so the headroom here can "
            "only shrink"
        ),
        "verdict": None if fc["fits"] else BLOCKED,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--tokens-per-second",
        type=float,
        required=True,
        help="measured on the actual receiver; there is deliberately no default",
    )
    ap.add_argument(
        "--occupancy-multiplier",
        type=float,
        required=True,
        help="occupied time per unit of stage time, measured; no default, "
        "because assuming 1.0 is the error this forecast exists to catch",
    )
    ap.add_argument("--mean-tokens", type=int, default=432)
    ap.add_argument("--fit-seconds", type=float, default=0.0)
    ap.add_argument("--train-seconds", type=float, default=0.0)
    ap.add_argument("--setup", type=float, default=0.0)
    ap.add_argument("--transfers", type=float, default=0.0)
    ap.add_argument("--diagnosis", type=float, default=0.0)
    ap.add_argument("--export", type=float, default=0.0)
    ap.add_argument("--already-used", type=float, default=RECORDED_USED_S)
    ap.add_argument("--cap", type=float, default=RESERVE_TOTAL_S)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out = build(
        args.tokens_per_second,
        args.occupancy_multiplier,
        mean_tokens=args.mean_tokens,
        fit_s=args.fit_seconds,
        train_s=args.train_seconds,
        setup=args.setup,
        transfers=args.transfers,
        diagnosis=args.diagnosis,
        export=args.export,
        already_used=args.already_used,
        cap=args.cap,
    )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, sort_keys=True)
            f.write("\n")

    fc = out["forecast_at_token_ceiling"]
    print(
        f"evaluation at {out['generation']['tokens_per_second']} tok/s: "
        f"{out['generation']['seconds_at_token_ceiling']:.0f}s at the ceiling, "
        f"{out['generation']['seconds_at_mean_length']:.0f}s at the mean length"
    )
    print(
        f"occupied forecast {fc['forecast_occupied_s']:.0f}s against "
        f"{fc['remaining_s']:.0f}s remaining (upper bound)"
    )
    if out["fits"]:
        print(f"  fits, headroom {fc['headroom_s']:.0f}s")
        return 0
    print(f"  {BLOCKED}: short by {fc['shortfall_s']:.0f}s")
    if out["fits_only_at_mean_length"]:
        print("  it fits only if no generation reaches the cap, which is not a plan")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
