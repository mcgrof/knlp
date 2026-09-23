# SPDX-License-Identifier: GPL-2.0
"""Where the milliseconds in a repaired translated cache actually go.

The stage timing says the repaired path costs more than a native prefill. It
does not say which part costs what, and without that the result reads as a
verdict on one implementation rather than on the operation. So each step is
timed on its own, on the same documents, with the same device timer.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(
    0,
    os.environ.get(
        "KVT_SRC", os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    ),
)

from research.cartridges_cas.scripts.cas_kv_rope import rerot
from research.kv_translate import freeze
from research.kv_translate.fit import SourceLayout
from research.kv_translate.pairs import describe, prefill, to_content_keys
from research.kv_translate.recovery import BLOCK, mask_from_blocks
from research.kv_translate.recovery_runner import make_cache, select
from research.kv_translate.replay import replay_selected
from research.kv_translate.run_a1 import flat_features

CTX = 4096
REPS = 5
BUDGETS = (1, 8, 25, 64, 128, 256)


def main():
    man_path, arm_path, out_path, n_docs = (
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        int(sys.argv[4]),
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    models, geom = {}, {}
    for role, mid in (
        ("source", "Qwen/Qwen2.5-1.5B-Instruct"),
        ("target", "Qwen/Qwen2.5-7B-Instruct"),
    ):
        m = (
            AutoModelForCausalLM.from_pretrained(
                mid, dtype=torch.bfloat16, attn_implementation="sdpa"
            )
            .to(dev)
            .eval()
        )
        for p in m.parameters():
            p.requires_grad_(False)
        models[role], geom[role] = m, describe(m, mid)
    sg, tg = geom["source"], geom["target"]
    layout = SourceLayout(sg.n_layers, sg.n_kv_heads, sg.head_dim)
    mk, mv, _ = freeze.load(arm_path, layout, tg, device=dev)
    mk, mv = mk.cast(torch.float32), mv.cast(torch.float32)
    pos = torch.arange(CTX, device=dev)

    man = json.load(open(man_path))
    docs = [d for d in man["documents"] if d["in_screen"]][:n_docs]
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def t(fn):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        out = []
        for _ in range(REPS):
            ev[0].record()
            fn()
            ev[1].record()
            torch.cuda.synchronize()
            out.append(ev[0].elapsed_time(ev[1]))
        return out

    acc = {}

    def add(name, samples):
        acc.setdefault(name, []).extend(samples)

    for d in docs:
        ids = torch.tensor(d["prompt_ids"], device=dev).unsqueeze(0)
        qids = tok(d["retrieval_query"], return_tensors="pt").input_ids.to(dev)
        n = qids.shape[1]

        add("source_prefill_4096", t(lambda: prefill(models["source"], ids, sg)))
        sp = prefill(models["source"], ids, sg)

        def do_map():
            xk = flat_features(to_content_keys(sp)).float()
            xv = flat_features([v.float() for v in sp.values]).float()
            keys = [
                rerot(b[0].float(), pos, tg.rope_theta).unsqueeze(0).to(torch.bfloat16)
                for b in mk.apply(xk)
            ]
            return keys, [v.to(torch.bfloat16) for v in mv.apply(xv)]

        add("map_apply_and_rerotate", t(do_map))
        keys, vals = do_map()

        add(
            "select_rank_on_mapped",
            t(
                lambda: select(
                    "mapped_attn",
                    64,
                    model=models["target"],
                    mapped=(keys, vals),
                    native=None,
                    query_ids=qids,
                    n_positions=CTX,
                )
            ),
        )

        for k in BUDGETS:
            m = mask_from_blocks(CTX, list(range(0, 256, max(256 // k, 1)))[:k]).to(dev)
            add(
                f"replay_{k:03d}_blocks",
                t(lambda mm=m: replay_selected(models["target"], ids, mm, keys, vals)),
            )

        add(
            "question_over_resident_cache",
            t(
                lambda: models["target"](
                    input_ids=qids,
                    attention_mask=torch.ones(1, CTX + n, dtype=torch.long, device=dev),
                    position_ids=torch.arange(CTX, CTX + n, device=dev).unsqueeze(0),
                    past_key_values=make_cache(keys, vals),
                    use_cache=True,
                ).logits
            ),
        )

        add(
            "native_prefill_plus_question",
            t(
                lambda: models["target"](
                    input_ids=torch.cat([ids, qids], dim=1), logits_to_keep=1
                ).logits
            ),
        )

    out = {
        "contract": "recovery_cost_decomposition_v1",
        "ctx": CTX,
        "documents": [d["doc"] for d in docs],
        "reps": REPS,
        "block_tokens": BLOCK,
        "device": torch.cuda.get_device_name(0),
        "note": (
            "each step timed on its own, same documents, same device "
            "timer; the stage timing measures them composed"
        ),
        "steps": {
            k: {
                "p50_ms": float(np.percentile(v, 50)),
                "p95_ms": float(np.percentile(v, 95)),
                "n": len(v),
            }
            for k, v in acc.items()
        },
    }
    json.dump(out, open(out_path, "w"), indent=2)
    nat = out["steps"]["native_prefill_plus_question"]["p50_ms"]
    print(f"{'step':34s} {'p50_ms':>8s} {'of native':>10s}")
    for k in sorted(out["steps"]):
        s = out["steps"][k]
        print(f"{k:34s} {s['p50_ms']:8.1f} {s['p50_ms'] / nat:10.3f}")


if __name__ == "__main__":
    main()
