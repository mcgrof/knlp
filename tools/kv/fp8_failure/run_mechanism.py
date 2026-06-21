"""Mechanism + autoregressive runner: resolves the open phase-8 question (bias vs partial-RoPE
pass-through subspace) and adds the phase-9 AR-divergence, in one pass per model.

For each model it records, all vs the bf16 native baseline:
  - bias test: K8 logit error vs PRE-BIAS K8 (interleaved-aware for GPT-NeoX) -> recovery fraction.
    Read with the |recovery| < 0.05 noise floor (the pre-vs-post-RoPE quant-point move alone gives
    ~0.6% on a biasless model); high recovery = bias-driven, like Qwen ~0.97.
  - subspace test (partial-RoPE only): FP8 the rotary subspace only vs the pass-through subspace, AND
    the CONFOUND-BREAKER -- pass-through quantized with its K-bias removed (passthrough_prebias).
    For partial-RoPE models the bias is EXPECTED to sit on the un-rotated tail, so passthrough_only
    and prebias-recovery both fire on the same cause; only if passthrough_prebias stays high is the
    pass-through DISTRIBUTION genuinely fragile independent of bias. The bias L2 mass split (rotary
    vs pass-through) shows where the bias lives before reading the verdict.
  - AR divergence (measurement_level=hf_dynamic_cache): greedy generate with a PER-TOKEN FP8 cache
    (the real write-time cache, not per_tensor-over-the-whole-cache) for both K8/V16 (the failure
    direction) and K16/V8 (the deployable asym cache). first_divergence_frac is the headline; the
    margin the quant overturned flags numerical tie-flips vs real drift.

--self-test runs the WHOLE thing on tiny CPU models (contiguous Qwen, interleaved NeoX, partial-RoPE
Phi) so the pod only runs proven code.
"""

import argparse
import json
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
from . import mechanism_probe as MP  # noqa: E402
from . import run_smoke as RS  # noqa: E402


def _eval(model, ids_list, device, base, ppl_native, harness):
    harness.install()
    try:
        lg = RS._logits(model, ids_list, device)
    finally:
        harness.remove()
    mt = RS._metrics_vs_base(base, lg)
    return (
        mt["mean_logit_err"],
        mt["top1_agreement"],
        RS._ppl_from_logits(lg, ids_list) / max(ppl_native, 1e-9),
    )


@torch.no_grad()
def run_model(model, infos, ids_list, device, short_name, new_tokens=32):
    arch = AD.summarize(infos)
    interleaved = bool(arch.get("fused_interleaved"))
    is_partial = bool(arch.get("is_partial_rope"))
    rotary_dim = arch.get("rotary_dim")
    base = RS._logits(model, ids_list, device)
    ppl_native = RS._ppl_from_logits(base, ids_list)

    # --- bias test: K8 (failure) vs pre-bias K8 (fix). V bf16 in both -> isolates K. ---
    err_k8, top1_k8, pplx_k8 = _eval(
        model,
        ids_list,
        device,
        base,
        ppl_native,
        kbc.FlexKVHarness(
            model, infos, kbc.parse_spec("fp8:per_tensor"), kbc.parse_spec("bf16")
        ),
    )
    if interleaved:
        preb = MP.InterleavedPrebiasHarness(model, infos)
    else:
        preb = kbc.FlexKVHarness(
            model,
            infos,
            kbc.parse_spec("fp8:per_tensor"),
            kbc.parse_spec("bf16"),
            prebias=True,
        )
    err_preb, top1_preb, _ = _eval(model, ids_list, device, base, ppl_native, preb)
    has_bias = bool(arch.get("has_k_bias"))
    recovery = C.recovery_fraction(err_k8, err_preb, 0.0) if has_bias else None

    # where does the K-bias live -- rotary vs pass-through L2 mass (resolves the confound below).
    bias_l2_rot = bias_l2_pass = None
    pass_bias_by_layer = {}
    if has_bias and is_partial and rotary_dim and rotary_dim < arch["head_dim"]:
        rr, pp = [], []
        for info in infos:
            kb = MP.kbias_per_head(info)  # [n_kv, head_dim]
            if kb is None:
                continue
            rr.append(kb[:, :rotary_dim].pow(2).sum().item())
            pp.append(kb[:, rotary_dim:].pow(2).sum().item())
            pass_bias_by_layer[info["layer_idx"]] = kb[:, rotary_dim:]
        tot = sum(rr) + sum(pp)
        if tot > 0:
            bias_l2_rot = sum(rr) / tot
            bias_l2_pass = sum(pp) / tot

    # --- subspace test (partial-RoPE only): which subspace's FP8 carries the collapse, AND the cell
    # that breaks the bias/subspace confound -- pass-through quantized WITH its bias removed. ---
    err_rot = top1_rot = err_pass = top1_pass = err_pass_pb = top1_pass_pb = None
    if is_partial and rotary_dim and rotary_dim < arch["head_dim"]:
        err_rot, top1_rot, _ = _eval(
            model,
            ids_list,
            device,
            base,
            ppl_native,
            MP.SubspaceKHarness(model, infos, rotary_dim, "rotary_only"),
        )
        err_pass, top1_pass, _ = _eval(
            model,
            ids_list,
            device,
            base,
            ppl_native,
            MP.SubspaceKHarness(model, infos, rotary_dim, "passthrough_only"),
        )
        if has_bias and pass_bias_by_layer:
            err_pass_pb, top1_pass_pb, _ = _eval(
                model,
                ids_list,
                device,
                base,
                ppl_native,
                MP.SubspaceKHarness(
                    model,
                    infos,
                    rotary_dim,
                    "passthrough_prebias",
                    pass_bias_by_layer=pass_bias_by_layer,
                ),
            )

    # --- AR divergence: per-token FP8 (the real write-time cache). K8/V16 (the failure direction)
    # AND K16/V8 (the deployable asym cache) -> certify the shippable config AR-clean. ---
    ar_k8 = ARD.ar_divergence(
        model, infos, ids_list, device, new_tokens, "fp8:per_token", "bf16"
    )
    ar_v8 = ARD.ar_divergence(
        model, infos, ids_list, device, new_tokens, "bf16", "fp8:per_token"
    )

    def r(x, n=5):
        return round(x, n) if isinstance(x, (int, float)) else "NA"

    row = dict(
        model=short_name,
        has_k_bias=has_bias,
        is_partial_rope=is_partial,
        fused_interleaved=interleaved,
        rotary_dim=rotary_dim,
        head_dim=arch.get("head_dim"),
        bias_l2_frac_rotary=r(bias_l2_rot, 4),
        bias_l2_frac_passthrough=r(bias_l2_pass, 4),
        err_k8=r(err_k8),
        top1_k8=r(top1_k8),
        pplx_k8=r(pplx_k8, 4),
        err_prebias=r(err_preb),
        recovery_konly=r(recovery, 4) if recovery is not None else "NA",
        err_rotary_only=r(err_rot),
        top1_rotary_only=r(top1_rot),
        err_passthrough_only=r(err_pass),
        top1_passthrough_only=r(top1_pass),
        err_passthrough_prebias=r(err_pass_pb),
        top1_passthrough_prebias=r(top1_pass_pb),
        ar_k8v16_first_div=ar_k8["first_divergence_frac"],
        ar_k8v16_frac_diverged=ar_k8["frac_diverged"],
        ar_k8v16_margin=(
            ar_k8["margin_at_divergence"]
            if ar_k8["margin_at_divergence"] is not None
            else "NA"
        ),
        ar_k16v8_first_div=ar_v8["first_divergence_frac"],
        ar_k16v8_frac_diverged=ar_v8["frac_diverged"],
    )
    return row


FIELDS = [
    "model",
    "has_k_bias",
    "is_partial_rope",
    "fused_interleaved",
    "rotary_dim",
    "head_dim",
    "bias_l2_frac_rotary",
    "bias_l2_frac_passthrough",
    "err_k8",
    "top1_k8",
    "pplx_k8",
    "err_prebias",
    "recovery_konly",
    "err_rotary_only",
    "top1_rotary_only",
    "err_passthrough_only",
    "top1_passthrough_only",
    "err_passthrough_prebias",
    "top1_passthrough_prebias",
    "ar_k8v16_first_div",
    "ar_k8v16_frac_diverged",
    "ar_k8v16_margin",
    "ar_k16v8_first_div",
    "ar_k16v8_frac_diverged",
]


def _write(out_dir, short_name, model_id, row, manifest):
    C.write_manifest(C.model_csv_path(out_dir, short_name, "."), manifest)
    C.append_row(C.model_csv_path(out_dir, short_name, "mechanism.csv"), row, FIELDS)
    with open(C.model_csv_path(out_dir, short_name, "mechanism_card.md"), "w") as f:
        f.write(
            f"# {model_id} -- FP8-K mechanism + AR\n\n{json.dumps(row, indent=2)}\n"
        )


def run_one(model_id, short_name, out_dir, device, dtype, n, seq_len, new_tokens, seed):
    import gc

    torch.manual_seed(seed)
    model, tok = kbc.load_model(model_id, dtype, device)
    infos = AD.discover(model)
    ids_list = kbc.calib_prompts(tok, n=n, seq_len=seq_len)
    row = run_model(model, infos, ids_list, device, short_name, new_tokens)
    manifest = C.RunManifest(
        run_id=f"mech-{short_name}-s{seed}",
        model_id=model_id,
        dtype=dtype,
        device=str(device),
        measurement_level="fake_quant_teacher_forced+hf_dynamic_cache",
        seed=seed,
        eval_dataset="wikitext-103-raw-v1",
    )
    _write(out_dir, short_name, model_id, row, manifest)
    print(f"[mech] {short_name}: {row}")
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def self_test(out_dir, device="cpu"):
    from transformers import (
        GPTNeoXConfig,
        GPTNeoXForCausalLM,
        PhiConfig,
        PhiForCausalLM,
        Qwen2Config,
        Qwen2ForCausalLM,
    )

    common = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        attn_implementation="sdpa",
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
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
        "tiny-neox": GPTNeoXForCausalLM(
            GPTNeoXConfig(num_attention_heads=4, rotary_pct=0.25, **common)
        ),
    }
    torch.manual_seed(0)
    ids_list = [torch.randint(0, 64, (24,)).tolist() for _ in range(2)]
    for sn, model in builds.items():
        model = model.eval().to(device)
        infos = AD.discover(model)
        row = run_model(model, infos, ids_list, device, sn, new_tokens=8)
        manifest = C.RunManifest(
            run_id=f"selftest-{sn}",
            model_id=sn,
            dtype="float32",
            device=device,
            measurement_level="self_test",
        )
        _write(out_dir, sn, sn, row, manifest)
        # subspace cells (incl the confound-breaking passthrough_prebias) run for partial-RoPE+bias
        if row["is_partial_rope"]:
            assert (
                row["err_passthrough_only"] != "NA"
            ), f"{sn} partial-RoPE must run subspace"
            if row["has_k_bias"]:
                assert (
                    row["err_passthrough_prebias"] != "NA"
                ), f"{sn} partial-RoPE+bias must run passthrough_prebias"
                # bias_l2 split is NA only when the bias is identically zero (tiny models
                # zero-init biases); on real models with learned bias it is populated.
        else:
            assert (
                row["err_passthrough_only"] == "NA"
            ), f"{sn} full-RoPE has no pass-through"
        if row["fused_interleaved"]:
            assert (
                row["recovery_konly"] != "NA"
            ), "NeoX interleaved prebias must produce recovery"
        assert 0.0 <= row["ar_k8v16_first_div"] <= 1.0
        print(
            f"[self-test] {sn} OK partial={row['is_partial_rope']} "
            f"interleaved={row['fused_interleaved']} recovery={row['recovery_konly']} "
            f"bias_l2_pass={row['bias_l2_frac_passthrough']} "
            f"pass_pb_top1={row['top1_passthrough_prebias']} ar_k8={row['ar_k8v16_first_div']}"
        )

    # Causality guard for the AR/decode path: a deliberately-lossy K spec (int2) MUST change the
    # logits -- proves the harness the AR path installs is causal, not a no-op. Deterministic, unlike
    # token-flip on a tiny random model. (Decode-step interception is separately verified.)
    m = builds["tiny-qwen2"]
    infos = AD.discover(m)
    ids = torch.tensor(ids_list[0]).unsqueeze(0).to(device)
    with torch.no_grad():
        base_lg = m(ids).logits.clone()
        hh = kbc.FlexKVHarness(
            m, infos, kbc.parse_spec("int2:per_token"), kbc.parse_spec("bf16")
        )
        hh.install()
        try:
            lossy_lg = m(ids).logits.clone()
        finally:
            hh.remove()
    delta = (base_lg - lossy_lg).abs().max().item()
    assert (
        delta > 1e-3
    ), "lossy-K must change logits -- the cache-quant path is a no-op otherwise"
    print(f"[self-test] AR causality-guard OK (int2-K logit delta={delta:.3g})")
    print("[self-test] mechanism + AR paths exercised; CSVs at", out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--short-name")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--num-prompts", type=int, default=8)
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
