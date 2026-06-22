"""Phase 12 / claim C7: a cheap release-time preflight scanner + its held-out evaluation. For a model
it runs the cheap cells on a SMALL prompt sample, classifies the K8/V8 risk, decides K- vs V-dominant,
reads the K-bias stress and the partial-RoPE geometry, and emits a mechanism-matched recommendation
via a heuristic decision tree (a TOOL, not a statistical centerpiece -- per the Codex review).

The metric that matters is the FALSE-SAFE rate: a preflight that declares a fragile model safe is a
production foot-gun. So the scanner also runs a larger confirmation pass and flags any model where the
small preflight sample said 'tolerant' but the confirmation says otherwise. Run on held-out
checkpoints; aggregate the false-safe rate across them.
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

BIAS_STRESS_THRESH = (
    5.0  # |K-bias| above this (relative to ~unit K) flags the bias mechanism
)


@torch.no_grad()
def _cell_class(
    model, infos, ids_list, device, base, ppl_native, nll_native, thr, k, v
):
    h = kbc.FlexKVHarness(model, infos, kbc.parse_spec(k), kbc.parse_spec(v))
    h.install()
    try:
        lg = RS._logits(model, ids_list, device)
    finally:
        h.remove()
    mt = RS._metrics_vs_base(base, lg)
    ppl = RS._ppl_from_logits(lg, ids_list)
    nll_inc = 100.0 * (math.log(max(ppl, 1e-9)) - nll_native) / max(nll_native, 1e-9)
    return (
        RS.classify(nll_inc, mt["top1_agreement"], ppl / max(ppl_native, 1e-9), thr),
        mt["mean_logit_err"],
    )


def _recommend(k8v8_class, k_dominant, has_k_bias, max_kbias, is_partial):
    """The decision tree: mechanism-matched repair + always a K16/V8 fallback."""
    if k8v8_class == "tolerant":
        return "ship K8/V8 (tolerant)"
    if not k_dominant:
        return "V-dominant: native V or a finer V scale; K8/V16 stays safe"
    if has_k_bias and max_kbias >= BIAS_STRESS_THRESH and not is_partial:
        return "K-bias: pre-bias K FP8 (or int8-K); fallback K16/V8"
    if is_partial:
        return "partial-RoPE: subspace-split/per-channel/QK-gauge; fallback K16/V8 (test V too)"
    return "unknown K-side failure: native K; fallback K16/V8 (test V too)"


@torch.no_grad()
def run_model(
    model, infos, tok, device, short_name, thr, n_preflight, n_confirm, seq_len
):
    arch = AD.summarize(infos)
    max_kbias = max(
        (float(i["k_bias"].abs().max()) for i in infos if i.get("k_bias") is not None),
        default=0.0,
    )
    # small preflight sample
    pf = kbc.calib_prompts(tok, n=n_preflight, seq_len=seq_len, seed=0)
    base = RS._logits(model, pf, device)
    ppln = RS._ppl_from_logits(base, pf)
    nlln = math.log(max(ppln, 1e-9))
    k8v8_c, _ = _cell_class(
        model,
        infos,
        pf,
        device,
        base,
        ppln,
        nlln,
        thr,
        "fp8:per_tensor",
        "fp8:per_tensor",
    )
    _, err_k8v16 = _cell_class(
        model, infos, pf, device, base, ppln, nlln, thr, "fp8:per_tensor", "bf16"
    )
    _, err_k16v8 = _cell_class(
        model, infos, pf, device, base, ppln, nlln, thr, "bf16", "fp8:per_tensor"
    )
    k_dominant = (
        err_k8v16 >= err_k16v8
    )  # fp8-K hurts at least as much as fp8-V -> K-side
    rec = _recommend(
        k8v8_c,
        k_dominant,
        bool(arch.get("has_k_bias")),
        max_kbias,
        bool(arch.get("is_partial_rope")),
    )

    # larger confirmation pass -> the false-safe check
    cf = kbc.calib_prompts(tok, n=n_confirm, seq_len=seq_len, seed=1)
    cbase = RS._logits(model, cf, device)
    cppl = RS._ppl_from_logits(cbase, cf)
    cnll = math.log(max(cppl, 1e-9))
    confirm_c, _ = _cell_class(
        model,
        infos,
        cf,
        device,
        cbase,
        cppl,
        cnll,
        thr,
        "fp8:per_tensor",
        "fp8:per_tensor",
    )
    false_safe = (k8v8_c == "tolerant") and (confirm_c != "tolerant")

    return dict(
        model=short_name,
        has_k_bias=bool(arch.get("has_k_bias")),
        is_partial_rope=bool(arch.get("is_partial_rope")),
        max_abs_k_bias=round(max_kbias, 3),
        k_dominant=bool(k_dominant),
        preflight_k8v8_class=k8v8_c,
        confirm_k8v8_class=confirm_c,
        false_safe=bool(false_safe),
        recommended_repair=rec,
        n_preflight=n_preflight,
        n_confirm=n_confirm,
    )


FIELDS = [
    "model",
    "has_k_bias",
    "is_partial_rope",
    "max_abs_k_bias",
    "k_dominant",
    "preflight_k8v8_class",
    "confirm_k8v8_class",
    "false_safe",
    "recommended_repair",
]


def run_one(
    model_id, sn, out_dir, device, dtype, n_preflight, n_confirm, seq_len, seed
):
    import gc

    torch.manual_seed(seed)
    model, tok = kbc.load_model(model_id, dtype, device)
    infos = AD.discover(model)
    row = run_model(
        model,
        infos,
        tok,
        device,
        sn,
        RS._load_thresholds(),
        n_preflight,
        n_confirm,
        seq_len,
    )
    manifest = C.RunManifest(
        run_id=f"preflight-{sn}",
        model_id=model_id,
        dtype=dtype,
        device=str(device),
        measurement_level="fake_quant_teacher_forced",
        seed=seed,
        eval_dataset="wikitext-103-raw-v1",
    )
    C.write_manifest(C.model_csv_path(out_dir, sn, "."), manifest)
    C.append_row(C.model_csv_path(out_dir, sn, "preflight.csv"), row, FIELDS)
    with open(C.model_csv_path(out_dir, sn, "card.md"), "w") as f:
        f.write(
            f"# {model_id} -- FP8-K preflight\n\n```json\n{json.dumps(row, indent=2)}\n```\n"
        )
    print(
        f"[preflight] {sn}: k8v8={row['preflight_k8v8_class']} confirm={row['confirm_k8v8_class']} "
        f"false_safe={row['false_safe']} -> {row['recommended_repair']}"
    )
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def self_test(out_dir, device="cpu"):
    from transformers import PhiConfig, PhiForCausalLM, Qwen2Config, Qwen2ForCausalLM

    common = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        attn_implementation="sdpa",
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    builds = {
        "tiny-qwen2": Qwen2ForCausalLM(
            Qwen2Config(
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                rope_theta=10000.0,
                **common,
            )
        ),
        "tiny-phi": PhiForCausalLM(
            PhiConfig(
                num_attention_heads=4, head_dim=8, partial_rotary_factor=0.5, **common
            )
        ),
    }
    orig = kbc.calib_prompts
    kbc.calib_prompts = lambda tok, n, seq_len, seed=None: kbc._select_chunks(
        list(range(64)) * 400, n, seq_len, seed
    )
    try:
        for sn, m in builds.items():
            m = m.eval().to(device)
            infos = AD.discover(m)
            row = run_model(
                m,
                infos,
                None,
                device,
                sn,
                RS._load_thresholds(),
                n_preflight=4,
                n_confirm=8,
                seq_len=16,
            )
            assert row["preflight_k8v8_class"] in (
                "tolerant",
                "material",
                "catastrophic",
            )
            assert isinstance(row["false_safe"], bool)
            assert (
                isinstance(row["recommended_repair"], str) and row["recommended_repair"]
            )
            print(f"[self-test] preflight {sn} OK -> {row['recommended_repair']}")
    finally:
        kbc.calib_prompts = orig
    # decision-tree unit checks
    assert "pre-bias" in _recommend("catastrophic", True, True, 20.0, False)
    assert "partial-RoPE" in _recommend("material", True, True, 20.0, True)
    assert "native V" in _recommend("material", False, True, 20.0, False)
    print("[self-test] preflight decision-tree OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--short-name")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--n-preflight", type=int, default=8)
    ap.add_argument("--n-confirm", type=int, default=24)
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
        args.n_preflight,
        args.n_confirm,
        args.seq_len,
        args.seed,
    )


if __name__ == "__main__":
    main()
