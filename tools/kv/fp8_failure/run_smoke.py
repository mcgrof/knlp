"""End-to-end single-model smoke for the FP8 KV-cache failure atlas -- the ONE report path the
plan builds before any big run. For each model it: discovers architecture, runs the broad-atlas
cells (native / K8V8 / K16V8 / K8V16) as fake-quant teacher-forced + a real next-token PPL, adds
the pre-bias recovery cell, runs the Phase-6 QK-gauge activation smoke (native score-invariance on
a REAL model + the gauged-vs-ungauged FP8-K delta), classifies every cell against the frozen
thresholds, and writes a per-model CSV (incremental, never silent-overwrite) + manifest + a
human-readable card.

The point of this script before the pod: `--self-test` runs the WHOLE pipeline on tiny random-weight
CPU models (no GPU, no download), so the pod only ever executes already-proven code. Smoke params
come from configs/kv/fp8_failure_datasets.yaml (smoke tier) unless overridden.

  python3 run_smoke.py --self-test --output-dir /tmp/fp8smoke      # CPU, proves the paths
  python3 run_smoke.py --model Qwen/Qwen2.5-7B --output-dir OUT --device cuda:0
"""

import argparse
import json
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
from . import qk_gauge_equalization as G  # noqa: E402

# The broad-atlas cells (brief Phase 2): K/V independent fmt. native == bf16/bf16 baseline.
ATLAS_CELLS = {
    "native": dict(k="bf16", v="bf16", prebias=False),
    "k8v8": dict(k="fp8:per_tensor", v="fp8:per_tensor", prebias=False),
    "k16v8": dict(k="bf16", v="fp8:per_tensor", prebias=False),
    "k8v16": dict(k="fp8:per_tensor", v="bf16", prebias=False),
    "prebias_k8v8": dict(
        k="fp8:per_tensor", v="fp8:per_tensor", prebias=True
    ),  # the bias fix
}


@torch.no_grad()
def _logits(model, ids_list, device):
    return [
        model(torch.tensor(ids).unsqueeze(0).to(device)).logits[0].float().cpu()
        for ids in ids_list
    ]


def _metrics_vs_base(base, other):
    """Per-cell quality vs the native baseline: logit err, top-1/5 agreement, NLL delta."""
    me, mx, t1, t5, nll = [], [], [], [], []
    for b, c in zip(base, other):
        d = (b - c).abs()
        me.append(d.mean().item())
        mx.append(d.max().item())
        t1.append((b.argmax(-1) == c.argmax(-1)).float().mean().item())
        b5 = b.topk(5, -1).indices
        c1 = c.argmax(-1, keepdim=True)
        t5.append((b5 == c1).any(-1).float().mean().item())
        tgt = b.argmax(-1)
        nb = -F.log_softmax(b, -1).gather(-1, tgt.unsqueeze(-1)).mean().item()
        nc = -F.log_softmax(c, -1).gather(-1, tgt.unsqueeze(-1)).mean().item()
        nll.append(nc - nb)
    import statistics as _S

    return dict(
        mean_logit_err=_S.mean(me),
        max_logit_err=max(mx),
        top1_agreement=_S.mean(t1),
        top5_agreement=_S.mean(t5),
        nll_delta=_S.mean(nll),
    )


@torch.no_grad()
def _ppl(model, ids_list, device, harness=None):
    """Real next-token PPL over the prompts' own targets (no dataset download; works in self-test).
    If harness is given it is installed around the forward so the quantized cache is exercised.
    """
    if harness is not None:
        harness.install()
    tot_nll, tot_tok = 0.0, 0
    try:
        for ids in ids_list:
            t = torch.tensor(ids).unsqueeze(0).to(device)
            lp = torch.log_softmax(model(t).logits[0, :-1].float(), dim=-1)
            tgt = t[0, 1:]
            tot_nll += -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item()
            tot_tok += tgt.numel()
    finally:
        if harness is not None:
            harness.remove()
    return math.exp(tot_nll / max(tot_tok, 1))


@torch.no_grad()
def capture_k_amax(model, infos, ids_list, device):
    """Per-(kv_head, channel) |K| amax over the prompts, post-RoPE, via a read-only attention hook.
    Feeds the QK-gauge D construction. Returns {layer_idx: [n_kv, head_dim]}."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    impl = model.config._attn_implementation
    orig = ALL_ATTENTION_FUNCTIONS[impl]
    by_mod = {id(i["attn_module"]): i for i in infos}
    acc = {}

    def hook(module, q, k, v, attention_mask, scaling=None, dropout=0.0, **kw):
        info = by_mod.get(id(module))
        if info is not None:
            a = k.abs().amax(dim=(0, 2)).float().cpu()  # [n_kv, head_dim]
            li = info["layer_idx"]
            acc[li] = a if li not in acc else torch.maximum(acc[li], a)
        return orig(
            module, q, k, v, attention_mask, dropout=dropout, scaling=scaling, **kw
        )

    ALL_ATTENTION_FUNCTIONS[impl] = hook
    try:
        for ids in ids_list:
            model(torch.tensor(ids).unsqueeze(0).to(device))
    finally:
        ALL_ATTENTION_FUNCTIONS[impl] = orig
    return acc


def classify(nll_increase_pct, top1, ppl_mult, thr):
    """First band hit, catastrophic -> material -> tolerant (brief's frozen bands)."""
    cat = thr["catastrophic"]
    mat = thr["material"]
    tol = thr["tolerant"]
    if (
        nll_increase_pct > cat["nll_increase_pct_min"]
        or ppl_mult > cat["ppl_multiplier_min"]
        or top1 < cat["top1_agreement_max"]
    ):
        return "catastrophic"
    if (
        nll_increase_pct >= mat["nll_increase_pct_range"][0]
        or top1 < tol["top1_agreement_min"]
    ):
        return "material"
    return "tolerant"


@torch.no_grad()
def run_model(model, infos, ids_list, device, thr, short_name):
    """Core e2e pipeline for one already-loaded model. Returns (rows, gauge, arch)."""
    arch = AD.summarize(infos)
    base = _logits(model, ids_list, device)
    ppl_native = _ppl(model, ids_list, device)
    nll_native = math.log(max(ppl_native, 1e-9))

    rows = []
    cell_err = {}
    for name, cfg in ATLAS_CELLS.items():
        k, v = kbc.parse_spec(cfg["k"]), kbc.parse_spec(cfg["v"])
        h = kbc.FlexKVHarness(model, infos, k, v, prebias=cfg["prebias"])
        h.install()
        try:
            lg = _logits(model, ids_list, device)
            ppl = _ppl(model, ids_list, device)
        finally:
            h.remove()
        mt = _metrics_vs_base(base, lg)
        cell_err[name] = mt["mean_logit_err"]
        nll_cell = math.log(max(ppl, 1e-9))
        nll_inc_pct = 100.0 * (nll_cell - nll_native) / max(nll_native, 1e-9)
        ppl_mult = ppl / max(ppl_native, 1e-9)
        label = classify(nll_inc_pct, mt["top1_agreement"], ppl_mult, thr)
        rows.append(
            dict(
                model=short_name,
                cell=name,
                measurement_level="fake_quant_teacher_forced",
                ppl=round(ppl, 4),
                ppl_mult=round(ppl_mult, 4),
                nll_increase_pct=round(nll_inc_pct, 4),
                mean_logit_err=round(mt["mean_logit_err"], 5),
                max_logit_err=round(mt["max_logit_err"], 5),
                top1_agreement=round(mt["top1_agreement"], 5),
                top5_agreement=round(mt["top5_agreement"], 5),
                classification=label,
            )
        )

    # recovery fraction of the pre-bias fix on the K8/V8 failure
    rec = C.recovery_fraction(cell_err["k8v8"], cell_err["prebias_k8v8"], 0.0)

    # ---- Phase-6 QK-gauge activation smoke (the novel piece, on a REAL model) ----
    amax = capture_k_amax(model, infos, ids_list, device)
    D_by_layer = {
        li: G.make_D_from_target_amax(a) for li, a in amax.items()
    }  # equalize K channel range
    # (a) native invariance: gauge with NO quant must leave logits identical (it's a true gauge)
    with G.QKGaugeProbe(model, infos, D_by_layer=D_by_layer, quant_k=None):
        lg_inv = _logits(model, ids_list, device)
    inv_maxabs = max((a - b).abs().max().item() for a, b in zip(base, lg_inv))
    # (b) does the gauge reduce FP8-K error?  ungauged fp8-K vs gauged fp8-K, both vs baseline
    hk = kbc.FlexKVHarness(
        model, infos, kbc.parse_spec("fp8:per_tensor"), kbc.parse_spec("bf16")
    )
    hk.install()
    try:
        lg_fp8k = _logits(model, ids_list, device)
    finally:
        hk.remove()
    err_ungauged = _metrics_vs_base(base, lg_fp8k)["mean_logit_err"]
    with G.QKGaugeProbe(model, infos, D_by_layer=D_by_layer, quant_k="fp8:per_tensor"):
        lg_gauged = _logits(model, ids_list, device)
    err_gauged = _metrics_vs_base(base, lg_gauged)["mean_logit_err"]

    gauge = dict(
        model=short_name,
        native_invariance_maxabs=round(inv_maxabs, 6),
        fp8k_err_ungauged=round(err_ungauged, 5),
        fp8k_err_gauged=round(err_gauged, 5),
        gauge_reduces_err=bool(err_gauged < err_ungauged),
        prebias_recovery_fraction=round(rec, 4),
    )
    return rows, gauge, arch


CELL_FIELDS = [
    "model",
    "cell",
    "measurement_level",
    "ppl",
    "ppl_mult",
    "nll_increase_pct",
    "mean_logit_err",
    "max_logit_err",
    "top1_agreement",
    "top5_agreement",
    "classification",
]
GAUGE_FIELDS = [
    "model",
    "native_invariance_maxabs",
    "fp8k_err_ungauged",
    "fp8k_err_gauged",
    "gauge_reduces_err",
    "prebias_recovery_fraction",
]


def _load_thresholds():
    from . import configs as CFG

    return CFG.load("thresholds")["classification"]


def _write_outputs(out_dir, short_name, model_id, rows, gauge, arch, manifest):
    C.write_manifest(C.model_csv_path(out_dir, short_name, "."), manifest)
    cell_csv = C.model_csv_path(out_dir, short_name, "cells.csv")
    for r in rows:
        C.append_row(cell_csv, r, CELL_FIELDS)
    C.append_row(
        C.model_csv_path(out_dir, short_name, "qk_gauge.csv"), gauge, GAUGE_FIELDS
    )
    card = C.model_csv_path(out_dir, short_name, "card.md")
    with open(card, "w") as f:
        f.write(f"# {model_id} -- FP8 KV-cache smoke\n\n")
        f.write(f"arch: {json.dumps(arch)}\n\n")
        f.write("| cell | ppl | ppl_mult | nll_inc% | top1 | logit_err | class |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['cell']} | {r['ppl']} | {r['ppl_mult']} | {r['nll_increase_pct']} "
                f"| {r['top1_agreement']} | {r['mean_logit_err']} | {r['classification']} |\n"
            )
        f.write(
            f"\nQK-gauge: native_invariance_maxabs={gauge['native_invariance_maxabs']} "
            f"(want ~0), fp8k err ungauged={gauge['fp8k_err_ungauged']} -> "
            f"gauged={gauge['fp8k_err_gauged']} (reduces={gauge['gauge_reduces_err']}); "
            f"prebias recovery={gauge['prebias_recovery_fraction']}\n"
        )
    return cell_csv


def run_one(model_id, short_name, out_dir, device, dtype, n, seq_len, seed):
    torch.manual_seed(seed)
    model, tok = kbc.load_model(model_id, dtype, device)
    infos = AD.discover(model)
    ids_list = kbc.calib_prompts(tok, n=n, seq_len=seq_len)
    thr = _load_thresholds()
    rows, gauge, arch = run_model(model, infos, ids_list, device, thr, short_name)
    manifest = C.RunManifest(
        run_id=f"smoke-{short_name}-s{seed}",
        model_id=model_id,
        dtype=dtype,
        device=str(device),
        measurement_level="fake_quant_teacher_forced",
        seed=seed,
        eval_dataset="wikitext-103-raw-v1",
    )
    _write_outputs(out_dir, short_name, model_id, rows, gauge, arch, manifest)
    print(f"[smoke] {short_name}: {[ (r['cell'], r['classification']) for r in rows ]}")
    print(f"[smoke] {short_name} gauge: {gauge}")
    return rows, gauge


def self_test(out_dir, device="cpu"):
    """Run the FULL pipeline on tiny random-weight CPU models -- no GPU, no download. Proves every
    code path (discovery, atlas cells, prebias, QK-gauge probe, capture, classify, writers).
    """
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
    builds = {
        "tiny-qwen2-biased": Qwen2ForCausalLM(
            Qwen2Config(
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                rope_theta=10000.0,
                **common,
            )
        ),
        "tiny-phi-partialrope": PhiForCausalLM(
            PhiConfig(
                num_attention_heads=4, head_dim=8, partial_rotary_factor=0.5, **common
            )
        ),
    }
    thr = _load_thresholds()
    torch.manual_seed(0)
    ids_list = [torch.randint(0, 64, (24,)).tolist() for _ in range(3)]
    ok = True
    for sn, model in builds.items():
        model = model.eval().to(device)
        infos = AD.discover(model)
        rows, gauge, arch = run_model(model, infos, ids_list, device, thr, sn)
        manifest = C.RunManifest(
            run_id=f"selftest-{sn}",
            model_id=sn,
            dtype="float32",
            device=device,
            measurement_level="fake_quant_teacher_forced",
        )
        _write_outputs(out_dir, sn, sn, rows, gauge, arch, manifest)
        # invariants the self-test asserts:
        native = [r for r in rows if r["cell"] == "native"][0]
        assert native["classification"] == "tolerant", "native must be tolerant"
        assert abs(native["mean_logit_err"]) < 1e-4, "native vs base must be ~0"
        assert (
            gauge["native_invariance_maxabs"] < 1e-3
        ), "QK-gauge must be score-invariant"
        print(
            f"[self-test] {sn} OK  native={native['classification']} "
            f"inv={gauge['native_invariance_maxabs']:.2e}"
        )
    print("[self-test] all paths exercised; CSVs at", out_dir)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--short-name")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--num-prompts", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
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
