# SPDX-License-Identifier: GPL-2.0
"""Fold a closure receipt into the lane ledger without disturbing history.

The ledger carries prior rounds' accounting, and those figures are evidence
about what already happened. So this edits only the keys this diagnostic owns,
copies the receipt's numbers rather than restating them, and leaves anything
it cannot source as it found it. Nothing is reset to zero to look tidy.

Run: python research/kv_translate/update_ledger.py --ledger <p> --receipt <p>
"""

from __future__ import annotations

import argparse
import json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    led = json.load(open(args.ledger))
    rec = json.load(open(args.receipt))

    lc = led.setdefault("latency_closure", {})
    acct = rec["accounting"]
    lc["executed"] = True
    lc["state"] = rec["verdicts"]["timing"]
    lc["execution_state"] = rec["identity"]["execution_state"]
    lc["experiment_id"] = rec["identity"]["experiment_id"]
    lc["attempt_id"] = rec["identity"]["attempt_id"]
    lc["closure_receipt"] = args.receipt
    lc["occupied_seconds"] = acct["occupied_seconds"]
    lc["remaining_seconds"] = acct["remaining_seconds"]
    lc["implementation_repairs_used"] = acct["repairs_used"]
    lc["allocation_release"] = rec["cleanup"]["allocation_release"]
    lc["spend_tag"] = acct["spend_tag"]
    lc["billing_receipt"] = acct["billing_receipt"]
    lc["results_by_length_and_condition"] = rec["results_by_length_and_condition"]

    led["state"] = (
        f"RUNTIME_RESCUE_CLOSED; {rec['verdicts']['timing']} + "
        "CARTRIDGE_BLOCKED_INPUTS"
    )
    led["next_state"] = (
        "Runtime rescue is closed. The latency question is answered at the "
        "three sampled lengths on this device under both availability "
        "conditions. Cartridge input recovery continues independently."
    )
    led.setdefault("corrections_this_round", [])
    for c in rec.get("corrections", []):
        if c not in led["corrections_this_round"]:
            led["corrections_this_round"].append(c)

    out = args.out or args.ledger
    json.dump(led, open(out, "w"), indent=2, sort_keys=True, default=str)
    print(f"ledger updated: state={led['state']}")
    print(f"  occupied {acct['occupied_seconds']} of {acct['allowance_seconds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
