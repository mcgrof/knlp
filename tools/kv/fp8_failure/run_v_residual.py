"""Phase 5: isolate the V-side residual that keeps K16/V8 from being lossless on Phi-2 (and any model
where K16/V8 is only 'material'). K stays bf16 throughout, so this is purely a V probe. Answers WHY
fp8-V hurts and classifies the residual:

  - granularity: does a finer V scale (per-head -> per-token -> per-channel -> per-group) fix it?
    (=> scale-granularity issue, fixed by a V-specific scale)
  - format: e5m2-V vs e4m3-V vs int8-V (=> dynamic-range vs mantissa)
  - layer: V8 on layer subsets (which layers carry the V sensitivity)
  - real vs artifact: the residual also shows up in the incremental cache (AR K16/V8 divergence),
    so it is not merely a teacher-forced fake-quant artifact -- reported alongside.

Recommendation follows the result: V-specific scale if granularity fixes it; native-V if nothing
does; 'artifact' only if the teacher-forced residual vanishes in the incremental cache.
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
from . import ar_divergence as ARD  # noqa: E402
from . import common as C  # noqa: E402
from . import run_smoke as RS  # noqa: E402

# K bf16 throughout -> isolates V. (axis, cell, v_spec)
V_GRANULARITY = [
    "fp8:per_tensor",
    "fp8:per_head",
    "fp8:per_token",
    "fp8:per_channel",
    "fp8:per_group:128",
]
V_FORMAT = ["fp8:per_token", "e5m2:per_token", "int8:per_token"]


def _layer_sets(n):
    h = max(1, n // 2)
    return {
        "all": None,
        "first_half": set(range(0, h)),
        "second_half": set(range(h, n)),
        "layer0_only": {0},
    }


@torch.no_grad()
def _vcell(
    model,
    infos,
    ids_list,
    device,
    base,
    ppl_native,
    nll_native,
    thr,
    v_spec,
    layers=None,
):
    h = kbc.FlexKVHarness(
        model, infos, kbc.parse_spec("bf16"), kbc.parse_spec(v_spec), layers=layers
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
        ppl_mult=round(ppl / max(ppl_native, 1e-9), 4),
        nll_increase_pct=round(nll_inc, 4),
        mean_logit_err=round(mt["mean_logit_err"], 5),
        top1_agreement=round(mt["top1_agreement"], 5),
        classification=RS.classify(
            nll_inc, mt["top1_agreement"], ppl / max(ppl_native, 1e-9), thr
        ),
    )


@torch.no_grad()
def run_model(model, infos, ids_list, device, short_name, thr, new_tokens=32):
    arch = AD.summarize(infos)
    n = arch["n_layers"]
    base = RS._logits(model, ids_list, device)
    ppl_native = RS._ppl_from_logits(base, ids_list)
    nll_native = math.log(max(ppl_native, 1e-9))
    rows = []

    def add(axis, cell, v_spec, layers=None):
        m = _vcell(
            model,
            infos,
            ids_list,
            device,
            base,
            ppl_native,
            nll_native,
            thr,
            v_spec,
            layers,
        )
        rows.append(dict(model=short_name, axis=axis, cell=cell, v_spec=v_spec, **m))

    for spec in V_GRANULARITY:
        add("v_granularity", spec.split(":", 1)[1], spec)
    for spec in V_FORMAT:
        add("v_format", spec.split(":")[0], spec)
    for cell, layers in _layer_sets(n).items():
        add("v_layer", cell, "fp8:per_token", layers)

    # real-vs-artifact: does the K16/V8 residual survive the incremental cache? (per-token V8)
    ar = ARD.ar_divergence(
        model, infos, ids_list, device, new_tokens, "bf16", "fp8:per_token"
    )
    rows.append(
        dict(
            model=short_name,
            axis="v_incremental",
            cell="ar_k16v8",
            v_spec="fp8:per_token",
            ppl_mult="",
            nll_increase_pct="",
            mean_logit_err="",
            top1_agreement=ar["cumulative_agreement"],
            classification=("diverges" if ar["frac_diverged"] > 0 else "stable"),
        )
    )
    return rows, arch, ar


FIELDS = [
    "model",
    "axis",
    "cell",
    "v_spec",
    "ppl_mult",
    "nll_increase_pct",
    "mean_logit_err",
    "top1_agreement",
    "classification",
]


def _classify_residual(rows):
    """Summarize the residual: does any V granularity/format make K16/V8 tolerant?"""
    gran = [r for r in rows if r["axis"] == "v_granularity"]
    fmt = [r for r in rows if r["axis"] == "v_format"]
    per_tensor = next((r for r in gran if r["cell"] == "per_tensor"), None)
    best_gran = min(
        (r for r in gran if isinstance(r["mean_logit_err"], (int, float))),
        key=lambda r: r["mean_logit_err"],
        default=None,
    )
    fixed_by_gran = best_gran is not None and best_gran["classification"] == "tolerant"
    e5m2 = next((r for r in fmt if r["cell"] == "e5m2"), None)
    fixed_by_e5m2 = e5m2 is not None and e5m2["classification"] == "tolerant"
    if per_tensor and per_tensor["classification"] == "tolerant":
        verdict = "no_residual (V8 already tolerant)"
    elif fixed_by_gran:
        verdict = f"scale_granularity (fixed by V {best_gran['cell']})"
    elif fixed_by_e5m2:
        verdict = "dynamic_range (fixed by e5m2-V)"
    else:
        verdict = (
            "real_V_sensitivity (no V layout/format rescues it -> recommend native V)"
        )
    return verdict


def _write(out_dir, sn, model_id, rows, arch, ar, manifest):
    C.write_manifest(C.model_csv_path(out_dir, sn, "."), manifest)
    cell_csv = C.model_csv_path(out_dir, sn, "v_residual.csv")
    for r in rows:
        C.append_row(cell_csv, r, FIELDS)
    verdict = _classify_residual(rows)
    with open(C.model_csv_path(out_dir, sn, "card.md"), "w") as f:
        f.write(f"# {model_id} -- K16/V8 V-residual diagnosis\n\n")
        f.write(f"arch: {json.dumps(arch)}\n\n**residual verdict: {verdict}**\n\n")
        f.write(f"incremental-cache (AR K16/V8): {json.dumps(ar)}\n\n")
        f.write(
            "| axis | cell | ppl_mult | top1 | err | class |\n|---|---|---|---|---|---|\n"
        )
        for r in rows:
            f.write(
                f"| {r['axis']} | {r['cell']} | {r['ppl_mult']} | {r['top1_agreement']} "
                f"| {r['mean_logit_err']} | {r['classification']} |\n"
            )
    return verdict


def run_one(model_id, sn, out_dir, device, dtype, n, seq_len, new_tokens, seed):
    import gc

    torch.manual_seed(seed)
    model, tok = kbc.load_model(model_id, dtype, device)
    infos = AD.discover(model)
    ids_list = kbc.calib_prompts(tok, n=n, seq_len=seq_len)
    rows, arch, ar = run_model(
        model, infos, ids_list, device, sn, RS._load_thresholds(), new_tokens
    )
    manifest = C.RunManifest(
        run_id=f"vres-{sn}-s{seed}",
        model_id=model_id,
        dtype=dtype,
        device=str(device),
        measurement_level="fake_quant_teacher_forced+hf_dynamic_cache",
        seed=seed,
        eval_dataset="wikitext-103-raw-v1",
    )
    verdict = _write(out_dir, sn, model_id, rows, arch, ar, manifest)
    print(f"[vres] {sn}: residual verdict = {verdict}")
    for r in rows:
        print(
            f"    {r['axis']:14} {r['cell']:14} top1={r['top1_agreement']:<9} {r['classification']}"
        )
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def self_test(out_dir, device="cpu"):
    from transformers import PhiConfig, PhiForCausalLM

    torch.manual_seed(0)
    m = PhiForCausalLM(
        PhiConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            head_dim=8,
            partial_rotary_factor=0.5,
            attn_implementation="sdpa",
            max_position_embeddings=128,
            tie_word_embeddings=False,
        )
    ).eval()
    infos = AD.discover(m)
    ids_list = [torch.randint(0, 64, (24,)).tolist() for _ in range(2)]
    rows, arch, ar = run_model(
        m, infos, ids_list, device, "tiny-phi", RS._load_thresholds(), 8
    )
    manifest = C.RunManifest(
        run_id="selftest-vres",
        model_id="tiny-phi",
        dtype="float32",
        device=device,
        measurement_level="self_test",
    )
    v = _write(out_dir, "tiny-phi", "tiny-phi", rows, arch, ar, manifest)
    axes = {r["axis"] for r in rows}
    assert {"v_granularity", "v_format", "v_layer", "v_incremental"} <= axes, axes
    assert any(r["cell"] == "e5m2" for r in rows), "must include e5m2-V"
    assert any(r["cell"] == "per_channel" for r in rows), "must include per-channel V"
    print(f"[self-test] v-residual OK verdict={v} ({len(rows)} cells)")


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
    ap.add_argument("--new-tokens", type=int, default=48)
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
        args.new_tokens,
        args.seed,
    )


if __name__ == "__main__":
    main()
