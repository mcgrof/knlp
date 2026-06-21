"""GPT-J atlas + subspace runner -- the partial-RoPE BIASLESS control. Loads GPT-J with
attn_implementation='eager' (its only path) and quantizes via the GPTJKVAttnPatch (the
ALL_ATTENTION_FUNCTIONS harness no-ops GPT-J). FAIL-CLOSED: verify_intercepts() runs first and raises
on a silent no-op, so a 'tolerant' here is real. Answers: does a partial-RoPE model with NO K-bias
show FP8-K fragility, and is it the pass-through subspace? -- the clean separator the biased
Phi/Pythia/NeoX cannot give.

  python3 run_gptj.py --self-test --output-dir /tmp/gptj   # tiny GPT-J, CPU
  python3 run_gptj.py --model EleutherAI/gpt-j-6b --output-dir OUT --device cuda:0
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
from . import gptj_patch as GJ  # noqa: E402
from . import run_smoke as RS  # noqa: E402

CELLS = {
    "k8v8": ("fp8:per_tensor", "fp8:per_tensor", "full"),
    "k16v8": ("bf16", "fp8:per_tensor", "full"),
    "k8v16": ("fp8:per_tensor", "bf16", "full"),
    "k8_rotary_only": ("fp8:per_tensor", "bf16", "rotary_only"),
    "k8_passthrough_only": ("fp8:per_tensor", "bf16", "passthrough_only"),
}


def _load_gptj(model_id, dtype, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dt = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dt, attn_implementation="eager"
    )
    return model.to(device).eval(), tok


@torch.no_grad()
def run_model(model, infos, ids_list, device, short_name, thr):
    arch = AD.summarize(infos)
    rd = arch.get("rotary_dim")
    intercept = GJ.verify_intercepts(
        model, infos, device=device
    )  # FAIL-CLOSED: raises on no-op
    base = RS._logits(model, ids_list, device)
    ppl_native = RS._ppl_from_logits(base, ids_list)
    nll_native = math.log(max(ppl_native, 1e-9))

    rows = []
    for name, (k, v, mode) in CELLS.items():
        h = GJ.GPTJKVAttnPatch(
            model, infos, kbc.parse_spec(k), kbc.parse_spec(v), rotary_dim=rd, mode=mode
        )
        h.install()
        try:
            lg = RS._logits(model, ids_list, device)
        finally:
            h.remove()
        mt = RS._metrics_vs_base(base, lg)
        ppl = RS._ppl_from_logits(lg, ids_list)
        nll_inc = (
            100.0 * (math.log(max(ppl, 1e-9)) - nll_native) / max(nll_native, 1e-9)
        )
        label = RS.classify(
            nll_inc, mt["top1_agreement"], ppl / max(ppl_native, 1e-9), thr
        )
        rows.append(
            dict(
                model=short_name,
                cell=name,
                has_k_bias=arch.get("has_k_bias"),
                rotary_dim=rd,
                head_dim=arch.get("head_dim"),
                ppl=round(ppl, 4),
                ppl_mult=round(ppl / max(ppl_native, 1e-9), 4),
                nll_increase_pct=round(nll_inc, 4),
                mean_logit_err=round(mt["mean_logit_err"], 5),
                top1_agreement=round(mt["top1_agreement"], 5),
                softmax_kl=round(mt["softmax_kl"], 6),
                classification=label,
            )
        )
    return rows, intercept, arch


FIELDS = [
    "model",
    "cell",
    "has_k_bias",
    "rotary_dim",
    "head_dim",
    "ppl",
    "ppl_mult",
    "nll_increase_pct",
    "mean_logit_err",
    "top1_agreement",
    "softmax_kl",
    "classification",
]


def _write(out_dir, sn, model_id, rows, intercept, arch, manifest):
    C.write_manifest(C.model_csv_path(out_dir, sn, "."), manifest)
    cell_csv = C.model_csv_path(out_dir, sn, "cells.csv")
    for r in rows:
        C.append_row(cell_csv, r, FIELDS)
    with open(C.model_csv_path(out_dir, sn, "card.md"), "w") as f:
        f.write(f"# {model_id} -- GPT-J FP8-K (biasless partial-RoPE control)\n\n")
        f.write(
            f"intercept-verify: {json.dumps(intercept)}\narch: {json.dumps(arch)}\n\n"
        )
        f.write(
            "| cell | ppl_mult | top1 | logit_err | class |\n|---|---|---|---|---|\n"
        )
        for r in rows:
            f.write(
                f"| {r['cell']} | {r['ppl_mult']} | {r['top1_agreement']} "
                f"| {r['mean_logit_err']} | {r['classification']} |\n"
            )


def run_one(model_id, sn, out_dir, device, dtype, n, seq_len, seed):
    import gc

    torch.manual_seed(seed)
    model, tok = _load_gptj(model_id, dtype, device)
    infos = AD.discover(model)
    ids_list = kbc.calib_prompts(tok, n=n, seq_len=seq_len)
    rows, intercept, arch = run_model(
        model, infos, ids_list, device, sn, RS._load_thresholds()
    )
    manifest = C.RunManifest(
        run_id=f"gptj-{sn}-s{seed}",
        model_id=model_id,
        dtype=dtype,
        device=str(device),
        measurement_level="fake_quant_teacher_forced",
        seed=seed,
        eval_dataset="wikitext-103-raw-v1",
    )
    _write(out_dir, sn, model_id, rows, intercept, arch, manifest)
    print(f"[gptj] {sn} intercept={intercept}")
    print(f"[gptj] {sn}: {[(r['cell'], r['classification']) for r in rows]}")
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def self_test(out_dir, device="cpu"):
    from transformers import GPTJConfig, GPTJForCausalLM

    torch.manual_seed(0)
    m = GPTJForCausalLM(
        GPTJConfig(
            vocab_size=64,
            n_embd=32,
            n_layer=2,
            n_head=4,
            rotary_dim=4,
            attn_implementation="eager",
            n_positions=128,
            tie_word_embeddings=False,
        )
    ).eval()
    infos = AD.discover(m)
    assert infos[0]["has_k_bias"] is False, "GPT-J must be biasless"
    ids_list = [torch.randint(0, 64, (24,)).tolist() for _ in range(2)]
    rows, intercept, arch = run_model(
        m, infos, ids_list, device, "tiny-gptj", RS._load_thresholds()
    )
    manifest = C.RunManifest(
        run_id="selftest-gptj",
        model_id="tiny-gptj",
        dtype="float32",
        device=device,
        measurement_level="self_test",
    )
    _write(out_dir, "tiny-gptj", "tiny-gptj", rows, intercept, arch, manifest)
    cells = {r["cell"] for r in rows}
    assert (
        "k8_passthrough_only" in cells and "k8_rotary_only" in cells
    ), "subspace cells must run"
    assert intercept["knuke_delta"] > 1e-4, "intercept verify must be causal"
    print(f"[self-test] GPT-J OK intercept={intercept} cells={sorted(cells)}")


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
