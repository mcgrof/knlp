"""Long-context validation (stop condition: long-context for both mechanisms; prevent a 2K-only
paper). Runs the core cells + the pre-bias repair at growing context (2K / 8K / 16K) and asks whether
the K8 failure AND the K16/V8 fix survive positional drift. Memory-light: it stores only the native
next-token argmax per position (not the full [T, vocab] logits, which is ~5 GB at 16K), and computes
each cell's top-1-vs-native and teacher-forced PPL on the fly, so 16K fits a 7B on one 48 GB card.
"""

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # tools/kv
import k_bias_common as kbc  # noqa: E402

from . import architecture_discovery as AD  # noqa: E402
from . import common as C  # noqa: E402
from . import run_smoke as RS  # noqa: E402

CELLS = {
    "native": ("bf16", "bf16", False),
    "k8v8": ("fp8:per_tensor", "fp8:per_tensor", False),
    "k16v8": ("bf16", "fp8:per_tensor", False),
    "k8v16": ("fp8:per_tensor", "bf16", False),
    "prebias_k8v8": ("fp8:per_tensor", "fp8:per_tensor", True),
}
CONTEXTS = [2048, 8192, 16384]
FIELDS = [
    "model",
    "context",
    "cell",
    "ppl",
    "ppl_mult",
    "nll_increase_pct",
    "top1_vs_native",
    "classification",
]


@torch.no_grad()
def _lc_forward(model, ids, device, harness=None):
    """One forward over a long sequence -> (next-token argmax [T-1] on cpu, nll_sum, n_tok). The full
    logits are discarded; only the small argmax vector survives."""
    if harness is not None:
        harness.install()
    try:
        t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        lg = model(t).logits[0, :-1].float()  # [T-1, V] -- transient on GPU
        tgt = t[0, 1:]
        nll = -F.log_softmax(lg, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        return lg.argmax(-1).cpu(), nll.sum().item(), tgt.numel()
    finally:
        if harness is not None:
            harness.remove()


@torch.no_grad()
def run_model(model, infos, tok, device, short_name, thr, contexts, n_by_ctx):
    arch = AD.summarize(infos)
    interleaved = bool(arch.get("fused_interleaved"))
    rows = []
    for ctx in contexts:
        ids_list = kbc.calib_prompts(tok, n=n_by_ctx.get(ctx, 8), seq_len=ctx)
        # native first: store argmax per prompt + native ppl
        nat_argmax, nat_nll, nat_tok = [], 0.0, 0
        for ids in ids_list:
            am, nll, ntok = _lc_forward(model, ids, device)
            nat_argmax.append(am)
            nat_nll += nll
            nat_tok += ntok
        ppl_native = math.exp(nat_nll / max(nat_tok, 1))
        nll_nat = math.log(max(ppl_native, 1e-9))
        for cell, (k, v, prebias) in CELLS.items():
            if cell == "native":
                rows.append(
                    _row(short_name, ctx, cell, ppl_native, 1.0, nll_nat, 1.0, thr)
                )
                continue
            if interleaved and prebias:
                continue
            agree, cnll, ctok = [], 0.0, 0
            for ids, nam in zip(ids_list, nat_argmax):
                h = kbc.FlexKVHarness(
                    model, infos, kbc.parse_spec(k), kbc.parse_spec(v), prebias=prebias
                )
                am, nll, ntok = _lc_forward(model, ids, device, harness=h)
                agree.append((am == nam).float().mean().item())
                cnll += nll
                ctok += ntok
            ppl = math.exp(cnll / max(ctok, 1))
            import statistics as _S

            rows.append(
                _row(
                    short_name,
                    ctx,
                    cell,
                    ppl,
                    ppl / max(ppl_native, 1e-9),
                    nll_nat,
                    _S.mean(agree),
                    thr,
                )
            )
    return rows, arch


def _row(model, ctx, cell, ppl, ppl_mult, nll_nat, top1, thr):
    nll_inc = 100.0 * (math.log(max(ppl, 1e-9)) - nll_nat) / max(nll_nat, 1e-9)
    return dict(
        model=model,
        context=ctx,
        cell=cell,
        ppl=round(ppl, 4),
        ppl_mult=round(ppl_mult, 4),
        nll_increase_pct=round(nll_inc, 4),
        top1_vs_native=round(top1, 5),
        classification=RS.classify(nll_inc, top1, ppl_mult, thr),
    )


def _write(out_dir, sn, model_id, rows, arch, manifest):
    C.write_manifest(C.model_csv_path(out_dir, sn, "."), manifest)
    cell_csv = C.model_csv_path(out_dir, sn, "longctx.csv")
    for r in rows:
        C.append_row(cell_csv, r, FIELDS)
    with open(C.model_csv_path(out_dir, sn, "card.md"), "w") as f:
        f.write(f"# {model_id} -- long-context FP8-K (2K/8K/16K)\n\narch: {arch}\n\n")
        f.write(
            "| ctx | cell | ppl_mult | top1_vs_native | class |\n|---|---|---|---|---|\n"
        )
        for r in rows:
            f.write(
                f"| {r['context']} | {r['cell']} | {r['ppl_mult']} "
                f"| {r['top1_vs_native']} | {r['classification']} |\n"
            )


def run_one(model_id, sn, out_dir, device, dtype, contexts, n_by_ctx, seed):
    import gc

    torch.manual_seed(seed)
    model, tok = kbc.load_model(model_id, dtype, device)
    infos = AD.discover(model)
    rows, arch = run_model(
        model, infos, tok, device, sn, RS._load_thresholds(), contexts, n_by_ctx
    )
    manifest = C.RunManifest(
        run_id=f"lc-{sn}",
        model_id=model_id,
        dtype=dtype,
        device=str(device),
        measurement_level="fake_quant_teacher_forced",
        seed=seed,
        eval_dataset="wikitext-103-raw-v1",
    )
    _write(out_dir, sn, model_id, rows, arch, manifest)
    print(f"[longctx] {sn}: {len(rows)} (ctx,cell) cells")
    for r in rows:
        print(
            f"    ctx={r['context']:<6} {r['cell']:14} ppl×={r['ppl_mult']:<8} "
            f"top1={r['top1_vs_native']:<8} {r['classification']}"
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
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            rope_theta=10000.0,
            attn_implementation="sdpa",
            max_position_embeddings=256,
            tie_word_embeddings=False,
        )
    ).eval()
    infos = AD.discover(m)
    orig = kbc.calib_prompts
    kbc.calib_prompts = lambda tok, n, seq_len, seed=None: kbc._select_chunks(
        list(range(64)) * 400, n, seq_len, seed
    )
    try:
        rows, arch = run_model(
            m,
            infos,
            None,
            device,
            "tiny-qwen2",
            RS._load_thresholds(),
            contexts=[32, 64],
            n_by_ctx={32: 2, 64: 2},
        )
    finally:
        kbc.calib_prompts = orig
    _write(
        out_dir,
        "tiny-qwen2",
        "tiny-qwen2",
        rows,
        arch,
        C.RunManifest(
            run_id="st-lc",
            model_id="tiny-qwen2",
            dtype="float32",
            device=device,
            measurement_level="self_test",
        ),
    )
    ctxs = {r["context"] for r in rows}
    assert ctxs == {32, 64}, ctxs
    nat = [r for r in rows if r["cell"] == "native"]
    assert all(
        r["top1_vs_native"] == 1.0 and r["classification"] == "tolerant" for r in nat
    )
    assert any(r["cell"] == "prebias_k8v8" for r in rows), "prebias cell must run"
    print(f"[self-test] longctx OK ({len(rows)} cells across {sorted(ctxs)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--short-name")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--contexts", type=int, nargs="+", default=CONTEXTS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.self_test:
        self_test(args.output_dir, device="cpu")
        return
    sn = args.short_name or args.model.split("/")[-1]
    # fewer prompts at longer context (the brief: 32@8K, 16@16K)
    n_by_ctx = {2048: 16, 8192: 8, 16384: 6}
    run_one(
        args.model,
        sn,
        args.output_dir,
        args.device,
        args.dtype,
        args.contexts,
        n_by_ctx,
        args.seed,
    )


if __name__ == "__main__":
    main()
