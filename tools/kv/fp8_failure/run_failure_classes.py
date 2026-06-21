"""Phase 7: the other FP8-K failure classes -- granularity, format, and layer locality. All reuse
the validated FlexKVHarness (K quantized at the post-RoPE interface, V left bf16), so this is a
sweep, not new mechanism. Answers three practical questions on the models that fail K8:

  - granularity: does a finer K scale (per-head -> per-token -> per-channel -> per-group) rescue the
    collapse, and how fine do you have to go? (the deployability lever)
  - format: is the failure DYNAMIC-RANGE (e5m2's extra exponent rescues it) or MANTISSA/precision
    (e5m2 is worse, int8 with a real scale is fine)? fp8-e4m3 vs e5m2 vs int8/int6/int4.
  - layer: is the fragility spread or concentrated? K8 applied to layer subsets (first/second half,
    layer-0-only, all).

--self-test runs the whole sweep on a tiny CPU model.
"""

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # tools/kv
import k_bias_common as kbc  # noqa: E402

from . import architecture_discovery as AD  # noqa: E402
from . import common as C  # noqa: E402
from . import run_smoke as RS  # noqa: E402

# (axis, cell, k_spec)  -- V is bf16 throughout so we isolate K.
GRANULARITY = [
    "fp8:per_tensor",
    "fp8:per_head",
    "fp8:per_token",
    "fp8:per_channel",
    "fp8:per_group:128",
    "fp8:per_group:64",
]
FORMAT = [
    "fp8:per_token",
    "e5m2:per_token",
    "int8:per_token",
    "int6:per_token",
    "int4:per_token",
]


def _layer_sets(n):
    h = max(1, n // 2)
    return {
        "all": None,
        "first_half": set(range(0, h)),
        "second_half": set(range(h, n)),
        "layer0_only": {0},
    }


@torch.no_grad()
def _cell(
    model,
    infos,
    ids_list,
    device,
    base,
    ppl_native,
    nll_native,
    thr,
    k_spec,
    layers=None,
):
    h = kbc.FlexKVHarness(
        model, infos, kbc.parse_spec(k_spec), kbc.parse_spec("bf16"), layers=layers
    )
    h.install()
    try:
        lg = RS._logits(model, ids_list, device)
    finally:
        h.remove()
    mt = RS._metrics_vs_base(base, lg)
    ppl = RS._ppl_from_logits(lg, ids_list)
    nll_inc = 100.0 * (math.log(max(ppl, 1e-9)) - nll_native) / max(nll_native, 1e-9)
    return dict(
        ppl=round(ppl, 4),
        ppl_mult=round(ppl / max(ppl_native, 1e-9), 4),
        nll_increase_pct=round(nll_inc, 4),
        mean_logit_err=round(mt["mean_logit_err"], 5),
        top1_agreement=round(mt["top1_agreement"], 5),
        classification=RS.classify(
            nll_inc, mt["top1_agreement"], ppl / max(ppl_native, 1e-9), thr
        ),
    )


@torch.no_grad()
def run_model(model, infos, ids_list, device, short_name, thr):
    arch = AD.summarize(infos)
    n = arch["n_layers"]
    base = RS._logits(model, ids_list, device)
    ppl_native = RS._ppl_from_logits(base, ids_list)
    nll_native = math.log(max(ppl_native, 1e-9))
    rows = []

    def add(axis, cell, k_spec, layers=None):
        m = _cell(
            model,
            infos,
            ids_list,
            device,
            base,
            ppl_native,
            nll_native,
            thr,
            k_spec,
            layers,
        )
        rows.append(dict(model=short_name, axis=axis, cell=cell, k_spec=k_spec, **m))

    for spec in GRANULARITY:
        add("granularity", spec.split(":", 1)[1], spec)
    for spec in FORMAT:
        add("format", spec.split(":")[0], spec)
    for cell, layers in _layer_sets(n).items():
        add("layer", cell, "fp8:per_tensor", layers)
    return rows, arch


FIELDS = [
    "model",
    "axis",
    "cell",
    "k_spec",
    "ppl",
    "ppl_mult",
    "nll_increase_pct",
    "mean_logit_err",
    "top1_agreement",
    "classification",
]


def _write(out_dir, sn, model_id, rows, arch, manifest):
    C.write_manifest(C.model_csv_path(out_dir, sn, "."), manifest)
    cell_csv = C.model_csv_path(out_dir, sn, "failure_classes.csv")
    for r in rows:
        C.append_row(cell_csv, r, FIELDS)
    with open(C.model_csv_path(out_dir, sn, "card.md"), "w") as f:
        f.write(f"# {model_id} -- FP8-K failure classes (granularity/format/layer)\n\n")
        f.write(
            f"arch: {json.dumps(arch)}\n\n| axis | cell | ppl_mult | top1 | err | class |\n"
        )
        f.write("|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['axis']} | {r['cell']} | {r['ppl_mult']} | {r['top1_agreement']} "
                f"| {r['mean_logit_err']} | {r['classification']} |\n"
            )


def run_one(model_id, sn, out_dir, device, dtype, n, seq_len, seed):
    import gc

    torch.manual_seed(seed)
    model, tok = kbc.load_model(model_id, dtype, device)
    infos = AD.discover(model)
    ids_list = kbc.calib_prompts(tok, n=n, seq_len=seq_len)
    rows, arch = run_model(model, infos, ids_list, device, sn, RS._load_thresholds())
    manifest = C.RunManifest(
        run_id=f"fclass-{sn}-s{seed}",
        model_id=model_id,
        dtype=dtype,
        device=str(device),
        measurement_level="fake_quant_teacher_forced",
        seed=seed,
        eval_dataset="wikitext-103-raw-v1",
    )
    _write(out_dir, sn, model_id, rows, arch, manifest)
    print(f"[fclass] {sn}: granularity/format/layer sweep, {len(rows)} cells")
    for r in rows:
        print(
            f"    {r['axis']:12} {r['cell']:14} ppl×={r['ppl_mult']:<8} top1={r['top1_agreement']:<8} {r['classification']}"
        )
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def self_test(out_dir, device="cpu"):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    m = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            rope_theta=10000.0,
            attn_implementation="sdpa",
            max_position_embeddings=128,
            tie_word_embeddings=False,
        )
    ).eval()
    infos = AD.discover(m)
    ids_list = [torch.randint(0, 64, (24,)).tolist() for _ in range(2)]
    rows, arch = run_model(
        m, infos, ids_list, device, "tiny-qwen2", RS._load_thresholds()
    )
    manifest = C.RunManifest(
        run_id="selftest-fclass",
        model_id="tiny-qwen2",
        dtype="float32",
        device=device,
        measurement_level="self_test",
    )
    _write(out_dir, "tiny-qwen2", "tiny-qwen2", rows, arch, manifest)
    axes = {r["axis"] for r in rows}
    assert axes == {"granularity", "format", "layer"}, axes
    assert len([r for r in rows if r["axis"] == "granularity"]) == len(GRANULARITY)
    assert any(r["cell"] == "e5m2" for r in rows), "format sweep must include e5m2"
    assert any(
        r["cell"] == "layer0_only" for r in rows
    ), "layer sweep must include layer0_only"
    print(f"[self-test] failure-classes OK ({len(rows)} cells across {sorted(axes)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--short-name")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--num-prompts", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.self_test:
        self_test(args.output_dir, device="cpu")
        return
    sn = args.short_name or args.model.split("/")[-1]
    run_one(
        args.model,
        sn,
        args.output_dir,
        args.device,
        args.dtype,
        args.num_prompts,
        args.seq_len,
        args.seed,
    )


if __name__ == "__main__":
    main()
