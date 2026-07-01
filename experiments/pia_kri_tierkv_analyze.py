# SPDX-License-Identifier: GPL-2.0
"""Post-process the KRI-TierKV kill-test: score only OLD-block needles.

Needles that land in the protected first block or the recent window are
auto-kept by every method and do not test old-block selection, so the
headline is computed on the subset whose block is a genuine old block.
"""

import json
import os
import sys

D = os.path.dirname(__file__)
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else f"{D}/p2.json"))
nb = d["n_blocks"]
recent_n = d["recent_n"]
protected = {0}
recent = set(range(nb - recent_n, nb))
methods = [
    "fifo_old",
    "random_old",
    "kri_d_sum",
    "k_norm",
    "v_norm",
    "quest_lite",
    "oracle",
]

rows = d["per_row"]
# discriminating subset: answerable at full AND fact in an old block
sub = [
    r
    for r in rows
    if r["full"]
    and r["needle_block"] not in protected
    and r["needle_block"] not in recent
    and r["needle_block"] >= 0
]
n = len(sub)


def mean(key):
    return sum(r[key] for r in sub) / n if n else 0.0


acc = {m: round(mean(m + "_acc"), 3) for m in methods}
hit = {m: round(mean(m + "_blockhit"), 3) for m in methods}

fifo, oracle = acc["fifo_old"], acc["oracle"]
kri, quest = acc["kri_d_sum"], acc["quest_lite"]
gap = oracle - fifo
verdict = {
    "n_old_block_needles": n,
    "fifo": fifo,
    "oracle": oracle,
    "kri_d_sum": kri,
    "quest_lite": quest,
    "fifo_to_oracle_gap": round(gap, 3),
    "kri_minus_fifo": round(kri - fifo, 3),
    "kri_closes_gap_frac": round((kri - fifo) / gap, 3) if gap > 1e-9 else 0.0,
    "quest_minus_fifo": round(quest - fifo, 3),
    "quest_closes_gap_frac": round((quest - fifo) / gap, 3) if gap > 1e-9 else 0.0,
    "kri_killed": (kri - fifo) < 0.05 or (gap > 1e-9 and (kri - fifo) / gap < 0.30),
    "quest_survives": (gap > 1e-9 and (quest - fifo) / gap >= 0.50) and quest > fifo,
}
out = {
    "n_total_queries": len(rows),
    "n_strong": sum(r["full"] for r in rows),
    "n_old_block_needles": n,
    "block_budget": f"{1+recent_n+d['k_old']}/{nb}",
    "accuracy_old_block": acc,
    "block_hit_old_block": hit,
    "verdict": verdict,
}
json.dump(out, open(f"{D}/p2_analysis.json", "w"), indent=2)
print(json.dumps(out, indent=2))
