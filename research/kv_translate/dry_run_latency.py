# SPDX-License-Identifier: GPL-2.0
"""Exercise the timing path on tiny models before renting a card for it.

A shape error at four thousand tokens on a rented A100 costs the one repair
this diagnostic is allowed. The same error costs nothing here. This builds a
structurally faithful miniature of the pair -- a source with two key/value
heads, a target with four, the same head dimension, the same cache layout --
fits nothing, and runs the real timing runner against it on the CPU.

It validates plumbing, not speed. The runner stamps a dry run with a
different contract precisely so its timings can never be mistaken for the
measurement.

Run: python research/kv_translate/dry_run_latency.py --out-dir <tmp>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.kv_translate import freeze  # noqa: E402
from research.kv_translate.fit import AffineMap, SourceLayout  # noqa: E402
from research.kv_translate.run_a1 import Mapper  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def tiny(n_kv_heads, n_layers=4, head_dim=16, n_heads=4, vocab=256):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=vocab,
        hidden_size=n_heads * head_dim,
        intermediate_size=2 * n_heads * head_dim,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        max_position_embeddings=8192,
        tie_word_embeddings=False,
    )
    m = Qwen2ForCausalLM(cfg)
    m.eval()
    return m, cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ctx", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(0)

    src, scfg = tiny(n_kv_heads=2)
    tgt, tcfg = tiny(n_kv_heads=4)
    src.save_pretrained(os.path.join(args.out_dir, "src"))
    tgt.save_pretrained(os.path.join(args.out_dir, "tgt"))

    layout = SourceLayout(scfg.num_hidden_layers, scfg.num_key_value_heads, 16)
    layers = tuple(range(scfg.num_hidden_layers))
    d_sel = layout.columns_for(layers, 0, False).numel()

    def mapper(kind):
        maps = {}
        for li in range(tcfg.num_hidden_layers):
            for h in range(tcfg.num_key_value_heads):
                maps[(li, h)] = AffineMap(
                    M=torch.randn(d_sel, 16, dtype=torch.float64) * 0.02,
                    b=torch.zeros(16, dtype=torch.float64),
                    layers=layers,
                    head=h,
                    kind=kind,
                    target_layer=li,
                    ridge=1.0,
                    head_local=False,
                    n_calib_tokens=1,
                )

        class G:
            n_layers = tcfg.num_hidden_layers
            n_kv_heads = tcfg.num_key_value_heads
            head_dim = 16

        return Mapper(maps, layout, G(), kind)

    art = os.path.join(args.out_dir, "tiny_arm.pt")
    man = freeze.save(
        art,
        mapper("k"),
        mapper("v"),
        source={"model_id": os.path.join(args.out_dir, "src")},
        target={"model_id": os.path.join(args.out_dir, "tgt")},
        fit_context=args.ctx,
        calib_doc_ids=["dry"],
        dev_doc_ids=["dry"],
        eval_doc_ids=["dry"],
        config={"dry_run": True},
    )

    fx = os.path.join(args.out_dir, "tiny_fixtures.pt")
    import hashlib

    fixtures = []
    for i in range(2):
        t = torch.randint(0, 200, (args.ctx,), dtype=torch.long)
        fixtures.append(
            {
                "index": i,
                "ids": t,
                "n_tokens": args.ctx,
                "documents": [],
                "sha256": hashlib.sha256(t.numpy().tobytes()).hexdigest(),
            }
        )
    torch.save(
        {
            "contract": "timing_fixtures_v1",
            "role": "dry",
            "split": "dry",
            "dataset": "dry",
            "tokenizer": "dry",
            "length": args.ctx,
            "n_fixtures": len(fixtures),
            "fixtures": fixtures,
            "joint_sha256": hashlib.sha256(
                "".join(f["sha256"] for f in fixtures).encode()
            ).hexdigest(),
        },
        fx,
    )

    receipt = os.path.join(args.out_dir, "tiny_qual.json")
    json.dump(
        {
            "contract": "saved_operator_v1",
            "passed": True,
            "gpu": "cpu",
            "operator_dtype": "float32",
            "serving_dtype": "float32",
            "tf32_allowed": False,
            "contract_limits": {
                "operator_matches_reference": 1e-6,
                "operator_is_deterministic": 0.0,
                "serving_cast_within_one_step": 1.001,
            },
            "checks": {
                "artifact": {"joint_weight_sha256": man["joint_weight_sha256"]},
                "operator_matches_reference": {"measured": 0.0, "passed": True},
                "operator_is_deterministic": {"measured": 0.0, "passed": True},
                "serving_cast_within_one_step": {"measured": 0.5, "passed": True},
            },
        },
        open(receipt, "w"),
    )

    out = os.path.join(args.out_dir, "tiny_timing.json")
    r = subprocess.run(
        [
            sys.executable,
            os.path.join(HERE, "time_first_next_token.py"),
            "--source",
            os.path.join(args.out_dir, "src"),
            "--target",
            os.path.join(args.out_dir, "tgt"),
            "--artifact",
            art,
            "--fixtures",
            fx,
            "--qualification",
            receipt,
            "--lengths",
            str(args.ctx),
            "--reps",
            "3",
            "--warmups",
            "1",
            "--dtype",
            "float32",
            "--dry-run",
            "--out",
            out,
        ],
        capture_output=True,
        text=True,
    )
    print(r.stdout[-3000:])
    if r.returncode != 0:
        print(r.stderr[-4000:], file=sys.stderr)
        return r.returncode

    d = json.load(open(out))
    assert d["contract"].endswith("_DRY_RUN"), "a dry run must not claim the contract"
    assert d["dry_run"] is True
    res = d["results"][str(args.ctx)]
    assert res["model_logits_all_finite"], "non-finite model logits"
    assert res["nonfinite_occurrences"] == []
    assert res["per_fixture"], "per-fixture summaries missing"
    assert d["complete"] is True
    assert d["code"]["runner_sha256"], "runner not hashed"
    assert d["analysis_policy"]["n_fixtures"] == 2
    # Native keeps its first block; a translated condition that took the extra
    # block carries twice as many. The total has to add up either way.
    expected = res["n_samples_per_condition"] + sum(
        v["n"] for k, v in res["conditions"].items() if k != "target_native"
    )
    assert d["outputs"]["raw_sample_count"] == expected, (
        d["outputs"]["raw_sample_count"],
        expected,
    )
    assert d["pinning"]["source_config_sha256"] != d["pinning"]["target_config_sha256"]
    for c in (
        "target_native",
        "existing_gpu_source_kv_switch",
        "source_prefill_inclusive_translation",
    ):
        assert c in res["conditions"], f"{c} not measured"
        assert res["output_shapes"][c] == [1], f"{c} shape {res['output_shapes'][c]}"
    print("\nplumbing OK: three conditions ran, shapes [1], all finite")
    print("contract:", d["contract"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
