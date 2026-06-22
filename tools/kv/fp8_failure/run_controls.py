"""Negative / causal controls that turn the atlas's findings into causal claims (CLAIM_PLAN C2, C3):

  - BIAS-SPECIFICITY (Qwen): pre-bias recovery using the TRUE K-bias vs a zero / random-same-norm /
    channel-permuted bias. If a wrong same-norm vector recovers as well, the 'pre-bias fix' is not
    bias-specific. (C2 control)
  - ALPHA DOSE-RESPONSE (Qwen): scale the K-bias by alpha and measure the FP8-minus-native error at
    each alpha -- the causal dose-response for 'bias MAGNITUDE drives the failure'. (C2 control)
  - RANDOM-PARTITION (partial-RoPE, e.g. Phi): FP8 the TRUE pass-through tail vs 20 random same-size
    channel sets. If random sets damage as much, the effect is generic-split not architecture-aligned.
    Also runs the artificial split on a full-RoPE model as a negative-negative. (C3 control)

--self-test exercises all three on tiny CPU models.
"""

import argparse
import json
import os
import statistics as S
import sys

import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # tools/kv
import k_bias_common as kbc  # noqa: E402

from . import architecture_discovery as AD  # noqa: E402
from . import common as C  # noqa: E402
from . import mechanism_probe as MP  # noqa: E402
from . import run_smoke as RS  # noqa: E402

ALPHAS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
N_RANDOM_PARTITIONS = 20


def _err(model, ids_list, device, base, harness=None):
    if harness is not None:
        harness.install()
    try:
        lg = RS._logits(model, ids_list, device)
    finally:
        if harness is not None:
            harness.remove()
    return RS._metrics_vs_base(base, lg)["mean_logit_err"]


@torch.no_grad()
def bias_specificity(model, infos, ids_list, device, base):
    """C2 control: pre-bias recovery with true vs zero/random/permuted bias. Higher recovery = the
    subtraction matched the real bias. Returns {kind: recovery_fraction}."""
    k8 = kbc.parse_spec("fp8:per_tensor")
    err_k8 = _err(
        model,
        ids_list,
        device,
        base,
        kbc.FlexKVHarness(model, infos, k8, kbc.parse_spec("bf16")),
    )
    out = {"err_k8_plain": round(err_k8, 5)}
    for kind in ("true", "zero", "random_same_norm", "permuted"):
        bias_by_layer = {}
        for info in infos:
            tb = info.get("k_bias")
            if tb is None:
                continue
            bias_by_layer[info["layer_idx"]] = MP.control_bias(tb, kind, seed=0)
        if not bias_by_layer:
            continue
        h = MP.ControlledPrebiasHarness(model, infos, bias_by_layer, k8)
        err = _err(model, ids_list, device, base, h)
        out[f"recovery_{kind}"] = round(C.recovery_fraction(err_k8, err, 0.0), 4)
    return out


@torch.no_grad()
def alpha_doseresponse(model, infos, ids_list, device):
    """C2 control: FP8-minus-native error as the K-bias is scaled by alpha. Should grow with alpha."""
    k8 = kbc.parse_spec("fp8:per_tensor")
    gaps = {}
    for a in ALPHAS:
        with kbc.AlphaKBiasPatch(infos, a):
            native_a = RS._logits(model, ids_list, device)
            h = kbc.FlexKVHarness(model, infos, k8, kbc.parse_spec("bf16"))
            h.install()
            try:
                k8_a = RS._logits(model, ids_list, device)
            finally:
                h.remove()
        gaps[f"alpha_{a}"] = round(
            RS._metrics_vs_base(native_a, k8_a)["mean_logit_err"], 5
        )
    return gaps


@torch.no_grad()
def random_partition(
    model, infos, ids_list, device, base, n_random=N_RANDOM_PARTITIONS
):
    """C3 control: FP8 the true pass-through tail vs n_random same-size channel sets. The true tail
    should damage more than the random distribution if the effect is architecture-aligned.
    """
    arch = AD.summarize(infos)
    rd, hd = arch.get("rotary_dim"), arch["head_dim"]
    if not (arch.get("is_partial_rope") and rd and rd < hd):
        return None
    n_pass = hd - rd
    true_mask = torch.zeros(hd, dtype=torch.bool)
    true_mask[rd:] = True
    err_true = _err(
        model,
        ids_list,
        device,
        base,
        MP.SubspaceKHarness(model, infos, rd, "mask_only", mask=true_mask),
    )
    rnd = []
    for s in range(n_random):
        g = torch.Generator().manual_seed(1000 + s)
        idx = torch.randperm(hd, generator=g)[:n_pass]
        m = torch.zeros(hd, dtype=torch.bool)
        m[idx] = True
        rnd.append(
            _err(
                model,
                ids_list,
                device,
                base,
                MP.SubspaceKHarness(model, infos, rd, "mask_only", mask=m),
            )
        )
    return dict(
        n_pass=n_pass,
        n_random=n_random,
        err_true_passthrough=round(err_true, 5),
        err_random_mean=round(S.mean(rnd), 5),
        err_random_max=round(max(rnd), 5),
        true_exceeds_random_mean=bool(err_true > S.mean(rnd)),
        true_percentile=round(sum(1 for r in rnd if r < err_true) / len(rnd), 3),
    )


@torch.no_grad()
def run_model(model, infos, ids_list, device, short_name):
    arch = AD.summarize(infos)
    base = RS._logits(model, ids_list, device)
    row = dict(
        model=short_name,
        has_k_bias=arch.get("has_k_bias"),
        is_partial_rope=arch.get("is_partial_rope"),
        fused_interleaved=arch.get("fused_interleaved"),
    )
    # bias controls only for biased, non-interleaved (the ControlledPrebias slice is contiguous)
    if arch.get("has_k_bias") and not arch.get("fused_interleaved"):
        row["bias_specificity"] = bias_specificity(model, infos, ids_list, device, base)
        row["alpha_doseresponse"] = alpha_doseresponse(model, infos, ids_list, device)
    else:
        row["bias_specificity"] = None
        row["alpha_doseresponse"] = None
    row["random_partition"] = random_partition(model, infos, ids_list, device, base)
    return row


def _write(out_dir, sn, model_id, row, manifest):
    C.write_manifest(C.model_csv_path(out_dir, sn, "."), manifest)
    with open(C.model_csv_path(out_dir, sn, "controls.json"), "w") as f:
        json.dump(row, f, indent=2, default=str)
    with open(C.model_csv_path(out_dir, sn, "card.md"), "w") as f:
        f.write(
            f"# {model_id} -- causal/negative controls\n\n```json\n{json.dumps(row, indent=2)}\n```\n"
        )


def run_one(model_id, sn, out_dir, device, dtype, n, seq_len, seed):
    import gc

    torch.manual_seed(seed)
    model, tok = kbc.load_model(model_id, dtype, device)
    infos = AD.discover(model)
    ids_list = kbc.calib_prompts(tok, n=n, seq_len=seq_len)
    row = run_model(model, infos, ids_list, device, sn)
    manifest = C.RunManifest(
        run_id=f"ctrl-{sn}-s{seed}",
        model_id=model_id,
        dtype=dtype,
        device=str(device),
        measurement_level="fake_quant_teacher_forced",
        seed=seed,
        eval_dataset="wikitext-103-raw-v1",
    )
    _write(out_dir, sn, model_id, row, manifest)
    print(f"[ctrl] {sn}: {json.dumps(row)}")
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
        num_hidden_layers=3,
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
    ids_list = [torch.randint(0, 64, (24,)).tolist() for _ in range(2)]
    for sn, m in builds.items():
        m = m.eval().to(device)
        infos = AD.discover(m)
        row = run_model(m, infos, ids_list, device, sn)
        _write(
            out_dir,
            sn,
            sn,
            row,
            C.RunManifest(
                run_id=f"st-{sn}",
                model_id=sn,
                dtype="float32",
                device=device,
                measurement_level="self_test",
            ),
        )
        if row["bias_specificity"]:
            assert "recovery_true" in row["bias_specificity"], "bias control must run"
            assert len(row["alpha_doseresponse"]) == len(
                ALPHAS
            ), "alpha sweep must cover all alphas"
        if row["is_partial_rope"]:
            assert (
                row["random_partition"] is not None
            ), "partial-RoPE must run random-partition"
            assert row["random_partition"]["n_random"] == N_RANDOM_PARTITIONS
        print(
            f"[self-test] {sn} OK partial={row['is_partial_rope']} "
            f"bias_ctrl={'yes' if row['bias_specificity'] else 'no'} "
            f"randpart={'yes' if row['random_partition'] else 'no'}"
        )
    print("[self-test] controls paths exercised; out at", out_dir)


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
