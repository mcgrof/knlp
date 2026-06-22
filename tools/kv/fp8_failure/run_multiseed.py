"""Multi-seed core results with bootstrap confidence intervals (stop condition: 3 seeds + CIs, and
raw per-prompt data preserved). For each model it runs the core cells (native / K8V8 / K16V8 / K8V16)
over 3 seeds of DISJOINT wikitext prompts (the calib_prompts seed fix), records PER-PROMPT logit
error and top-1 agreement, and reports a percentile bootstrap CI over the pooled prompts. Replaces
the single-seed point estimates with intervals; one model load, all seeds.
"""

import argparse
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
from . import statistics as ST  # noqa: E402

CORE_CELLS = {
    "k8v8": ("fp8:per_tensor", "fp8:per_tensor"),
    "k16v8": ("bf16", "fp8:per_tensor"),
    "k8v16": ("fp8:per_tensor", "bf16"),
}
PP_FIELDS = ["model", "cell", "seed", "prompt_idx", "logit_err", "top1"]
CI_FIELDS = [
    "model",
    "cell",
    "n_prompts",
    "logit_err_mean",
    "logit_err_lo",
    "logit_err_hi",
    "top1_mean",
    "top1_lo",
    "top1_hi",
]


@torch.no_grad()
def _per_prompt(base, other):
    out = []
    for b, c in zip(base, other):
        out.append(
            (
                (b - c).abs().mean().item(),
                (b.argmax(-1) == c.argmax(-1)).float().mean().item(),
            )
        )
    return out


@torch.no_grad()
def run_model(model, infos, tok, device, short_name, n, seq_len, seeds):
    pp = []  # per-prompt rows
    for seed in seeds:
        ids_list = kbc.calib_prompts(tok, n=n, seq_len=seq_len, seed=seed)
        base = RS._logits(model, ids_list, device)
        for cell, (k, v) in CORE_CELLS.items():
            h = kbc.FlexKVHarness(model, infos, kbc.parse_spec(k), kbc.parse_spec(v))
            h.install()
            try:
                lg = RS._logits(model, ids_list, device)
            finally:
                h.remove()
            for pi, (le, t1) in enumerate(_per_prompt(base, lg)):
                pp.append(
                    dict(
                        model=short_name,
                        cell=cell,
                        seed=seed,
                        prompt_idx=pi,
                        logit_err=round(le, 6),
                        top1=round(t1, 6),
                    )
                )
    # bootstrap CI over the pooled per-prompt values per cell
    ci = []
    for cell in CORE_CELLS:
        le = [r["logit_err"] for r in pp if r["cell"] == cell]
        t1 = [r["top1"] for r in pp if r["cell"] == cell]
        lm, llo, lhi = ST.bootstrap_ci(le, n_boot=2000, seed=0)
        tm, tlo, thi = ST.bootstrap_ci(t1, n_boot=2000, seed=0)
        ci.append(
            dict(
                model=short_name,
                cell=cell,
                n_prompts=len(le),
                logit_err_mean=round(lm, 5),
                logit_err_lo=round(llo, 5),
                logit_err_hi=round(lhi, 5),
                top1_mean=round(tm, 5),
                top1_lo=round(tlo, 5),
                top1_hi=round(thi, 5),
            )
        )
    return pp, ci


def _write(out_dir, sn, model_id, pp, ci, manifest):
    C.write_manifest(C.model_csv_path(out_dir, sn, "."), manifest)
    ppath = C.model_csv_path(out_dir, sn, "per_prompt.csv")
    for r in pp:
        C.append_row(ppath, r, PP_FIELDS)
    cpath = C.model_csv_path(out_dir, sn, "ci.csv")
    for r in ci:
        C.append_row(cpath, r, CI_FIELDS)


def run_one(model_id, sn, out_dir, device, dtype, n, seq_len, seeds, seed):
    import gc

    torch.manual_seed(seed)
    model, tok = kbc.load_model(model_id, dtype, device)
    infos = AD.discover(model)
    pp, ci = run_model(model, infos, tok, device, sn, n, seq_len, seeds)
    manifest = C.RunManifest(
        run_id=f"mseed-{sn}",
        model_id=model_id,
        dtype=dtype,
        device=str(device),
        measurement_level="fake_quant_teacher_forced",
        seed=seed,
        eval_dataset="wikitext-103-raw-v1",
        calib_indices=tuple(seeds),
    )
    _write(out_dir, sn, model_id, pp, ci, manifest)
    print(f"[mseed] {sn}: {len(pp)} per-prompt rows over seeds {seeds}")
    for r in ci:
        print(
            f"    {r['cell']:6} logit_err {r['logit_err_mean']} "
            f"[{r['logit_err_lo']},{r['logit_err_hi']}]  top1 {r['top1_mean']} "
            f"[{r['top1_lo']},{r['top1_hi']}]"
        )
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ci


def self_test(out_dir, device="cpu"):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    m = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
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

    # monkeypatch calib_prompts to synthetic ids (offline; no wikitext download in the self-test)
    orig = kbc.calib_prompts
    kbc.calib_prompts = lambda tok, n, seq_len, seed=None: kbc._select_chunks(
        list(range(64)) * 400, n, seq_len, seed
    )
    try:
        pp, ci = run_model(
            m, infos, None, device, "tiny-qwen2", n=3, seq_len=16, seeds=[0, 1, 2]
        )
    finally:
        kbc.calib_prompts = orig
    _write(
        out_dir,
        "tiny-qwen2",
        "tiny-qwen2",
        pp,
        ci,
        C.RunManifest(
            run_id="st-mseed",
            model_id="tiny-qwen2",
            dtype="float32",
            device=device,
            measurement_level="self_test",
        ),
    )
    assert len(pp) == 3 * 3 * len(CORE_CELLS), "per-prompt rows = seeds*prompts*cells"
    assert all(
        r["logit_err_lo"] <= r["logit_err_mean"] <= r["logit_err_hi"] for r in ci
    )
    # seeded chunk selection must give DIFFERENT prompts per seed (the multi-seed fix)
    ids = list(range(64)) * 400
    assert kbc._select_chunks(ids, 3, 16, seed=0) != kbc._select_chunks(
        ids, 3, 16, seed=1
    )
    assert kbc._select_chunks(ids, 3, 16, seed=None) == kbc._select_chunks(
        ids, 3, 16
    )  # back-compat
    print(
        f"[self-test] multiseed OK ({len(pp)} per-prompt rows, CIs bracket means, prompts disjoint)"
    )


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
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
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
        args.seeds,
        args.seed,
    )


if __name__ == "__main__":
    main()
