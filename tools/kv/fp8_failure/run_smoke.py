"""End-to-end single-model smoke for the FP8 KV-cache failure atlas -- the ONE report path the
plan builds before any big run. For each model it: discovers architecture, runs the broad-atlas
cells (native / K8V8 / K16V8 / K8V16) as fake-quant teacher-forced + a real next-token PPL, adds
the pre-bias recovery cell, runs the Phase-6 QK-gauge activation smoke (native score-invariance on
a REAL model + the gauged-vs-ungauged FP8-K delta), classifies every cell against the frozen
thresholds, and writes a per-model CSV (incremental, never silent-overwrite) + manifest + a
human-readable card.

The point of this script before the pod: `--self-test` runs the WHOLE pipeline on tiny random-weight
CPU models (no GPU, no download), so the pod only ever executes already-proven code. Smoke params
are CLI-driven (--num-prompts/--seq-len); the datasets.yaml smoke tier is the reference, not loaded.

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
    # K-only prebias (V bf16 in both this and k8v16) -> recovery isolates K, no V-FP8 in the
    # denominator. On a biasless model prebias degenerates to K-native, so recovery is N/A there.
    "prebias_k8v16": dict(k="fp8:per_tensor", v="bf16", prebias=True),
}


@torch.no_grad()
def _logits(model, ids_list, device):
    return [
        model(torch.tensor(ids).unsqueeze(0).to(device)).logits[0].float().cpu()
        for ids in ids_list
    ]


def _metrics_vs_base(base, other):
    """Per-cell quality vs the native baseline: logit err, top-1/5 agreement, NLL delta, and the
    softmax KL(base || other) (a frozen-metrics headline; attributes degradation at the
    distribution level, not just the argmax)."""
    me, mx, t1, t5, nll, kl = [], [], [], [], [], []
    for b, c in zip(base, other):
        d = (b - c).abs()
        me.append(d.mean().item())
        mx.append(d.max().item())
        t1.append((b.argmax(-1) == c.argmax(-1)).float().mean().item())
        b5 = b.topk(5, -1).indices
        c1 = c.argmax(-1, keepdim=True)
        t5.append((b5 == c1).any(-1).float().mean().item())
        tgt = b.argmax(-1)
        lpb = F.log_softmax(b, -1)
        lpc = F.log_softmax(c, -1)
        nb = -lpb.gather(-1, tgt.unsqueeze(-1)).mean().item()
        nc = -lpc.gather(-1, tgt.unsqueeze(-1)).mean().item()
        nll.append(nc - nb)
        kl.append((lpb.exp() * (lpb - lpc)).sum(-1).mean().item())
    import statistics as _S

    return dict(
        mean_logit_err=_S.mean(me),
        max_logit_err=max(mx),
        top1_agreement=_S.mean(t1),
        top5_agreement=_S.mean(t5),
        nll_delta=_S.mean(nll),
        softmax_kl=_S.mean(kl),
    )


def _ppl_from_logits(logits_list, ids_list):
    """Next-token PPL from already-computed logits (avoids a second forward per cell)."""
    tot_nll, tot_tok = 0.0, 0
    for lg, ids in zip(logits_list, ids_list):
        lp = torch.log_softmax(lg[:-1].float(), dim=-1)
        tgt = torch.tensor(ids[1:])
        tot_nll += -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item()
        tot_tok += tgt.numel()
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


def _ratio(t):
    """max/median of a flattened tensor -- how much outlier remains after gauging."""
    a = t.float().abs().flatten()
    med = a.median().clamp(min=1e-9)
    return (a.max() / med).item()


# clamp sweep for the gauge D: (0.25,4.0) is the cautious default; real Qwen post-RoPE K outliers
# run 15-60x median, so the cautious clamp can't equalize them -> sweep up to (near-)unclamped so a
# `gauge_reduces_err=False` is attributable to the gauge, not to a clamp artifact.
GAUGE_CLAMPS = [(0.25, 4.0), (0.1, 10.0), (0.05, 20.0), (1e-6, 1e6)]


@torch.no_grad()
def run_model(model, infos, ids_list, device, thr, short_name):
    """Core e2e pipeline for one already-loaded model.
    Returns (rows, gauge_summary, gauge_sweep, arch)."""
    arch = AD.summarize(infos)
    has_k_bias = bool(arch.get("has_k_bias"))
    # The prebias path slices the K-bias out of the fused QKV output as a CONTIGUOUS block; that is
    # wrong for an interleaved-per-head QKV (GPT-NeoX/Pythia), so skip the prebias cells there rather
    # than quantize the wrong channels and report a meaningless recovery.
    interleaved = bool(arch.get("fused_interleaved"))
    n_tokens = sum(max(len(ids) - 1, 0) for ids in ids_list)
    base = _logits(model, ids_list, device)
    ppl_native = _ppl_from_logits(base, ids_list)
    nll_native = math.log(max(ppl_native, 1e-9))
    base_maxabs = max(b.abs().max().item() for b in base)

    rows = []
    cell_err = {}
    for name, cfg in ATLAS_CELLS.items():
        if interleaved and cfg["prebias"]:
            continue  # prebias path can't address an interleaved K -> skip (recovery -> NA)
        k, v = kbc.parse_spec(cfg["k"]), kbc.parse_spec(cfg["v"])
        h = kbc.FlexKVHarness(model, infos, k, v, prebias=cfg["prebias"])
        h.install()
        try:
            lg = _logits(model, ids_list, device)
        finally:
            h.remove()
        mt = _metrics_vs_base(base, lg)
        cell_err[name] = mt["mean_logit_err"]
        ppl = _ppl_from_logits(lg, ids_list)
        nll_inc_pct = (
            100.0 * (math.log(max(ppl, 1e-9)) - nll_native) / max(nll_native, 1e-9)
        )
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
                softmax_kl=round(mt["softmax_kl"], 6),
                classification=label,
            )
        )

    # K-only recovery (V bf16 in both arms): isolates K, no unrepairable V-FP8 in the denominator.
    # Meaningful only when there IS a K bias AND the prebias cell actually ran (not interleaved).
    if has_k_bias and not interleaved and "prebias_k8v16" in cell_err:
        rec_konly = C.recovery_fraction(
            cell_err["k8v16"], cell_err["prebias_k8v16"], 0.0
        )
    else:
        rec_konly = None  # N/A: no bias to repair, or interleaved QKV (prebias slice untrustworthy)

    # ---- Phase-6 QK-gauge activation smoke (the novel piece). Isolated in try/except so an exotic
    # architecture that breaks the gauge still keeps this model's atlas cells (the headline). ----
    try:
        gauge_summary, gauge_sweep = _qk_gauge_section(
            model, infos, ids_list, device, base, base_maxabs, short_name
        )
    except Exception as e:  # noqa: BLE001 -- never let the gauge sink the atlas row
        gauge_summary = dict(
            model=short_name, gauge_error=f"{type(e).__name__}: {str(e)[:120]}"
        )
        gauge_sweep = []
    gauge_summary["has_k_bias"] = has_k_bias
    gauge_summary["n_calib_tokens"] = n_tokens
    gauge_summary["prebias_recovery_fraction_konly"] = (
        round(rec_konly, 4) if rec_konly is not None else "NA"
    )
    return rows, gauge_summary, gauge_sweep, arch


@torch.no_grad()
def _qk_gauge_section(model, infos, ids_list, device, base, base_maxabs, short_name):
    """Phase-6 QK-gauge: native invariance + per-tensor/per-channel FP8-K baselines + clamp sweep.
    Returns (gauge_summary, gauge_sweep). Raises if the model's arch breaks the probe.
    """
    amax = capture_k_amax(model, infos, ids_list, device)
    ungauged_ratio = (
        sum(_ratio(a) for a in amax.values()) / max(len(amax), 1) if amax else 0.0
    )
    # (a) native invariance with NO quant must leave logits identical -- a TRUE gauge. On bf16 the
    # residual is rounding noise that scales with logit magnitude, so report it RELATIVE to base.
    D_default = {li: G.make_D_from_target_amax(a) for li, a in amax.items()}
    with G.QKGaugeProbe(model, infos, D_by_layer=D_default, quant_k=None):
        lg_inv = _logits(model, ids_list, device)
    inv_maxabs = max((a - b).abs().max().item() for a, b in zip(base, lg_inv))

    # (b) the HONEST free competitor to the gauge is per-channel FP8-K (no gauge, no Q-side cost).
    def _kfp8_err(spec):
        h = kbc.FlexKVHarness(
            model, infos, kbc.parse_spec(spec), kbc.parse_spec("bf16")
        )
        h.install()
        try:
            return _metrics_vs_base(base, _logits(model, ids_list, device))[
                "mean_logit_err"
            ]
        finally:
            h.remove()

    err_pertensor = _kfp8_err(
        "fp8:per_tensor"
    )  # the failure baseline the gauge must beat
    err_perchannel = _kfp8_err(
        "fp8:per_channel"
    )  # the free competitor it must also beat

    # (c) clamp sweep: gauged per-tensor FP8-K error vs both baselines, + realized outlier ratio.
    gauge_sweep = []
    for cl in GAUGE_CLAMPS:
        D = {li: G.make_D_from_target_amax(a, clamp=cl) for li, a in amax.items()}
        post_ratio = (
            sum(_ratio(D[li] * amax[li]) for li in amax) / max(len(amax), 1)
            if amax
            else 0.0
        )
        with G.QKGaugeProbe(model, infos, D_by_layer=D, quant_k="fp8:per_tensor"):
            err_g = _metrics_vs_base(base, _logits(model, ids_list, device))[
                "mean_logit_err"
            ]
        gauge_sweep.append(
            dict(
                model=short_name,
                clamp=f"{cl[0]}-{cl[1]}",
                fp8k_err_gauged=round(err_g, 5),
                gauge_beats_pertensor=bool(err_g < err_pertensor),
                gauge_beats_perchannel=bool(err_g < err_perchannel),
                post_gauge_amax_ratio=round(post_ratio, 3),
            )
        )

    best = min(gauge_sweep, key=lambda r: r["fp8k_err_gauged"])
    gauge_summary = dict(
        model=short_name,
        base_logit_maxabs=round(base_maxabs, 4),
        native_invariance_maxabs=round(inv_maxabs, 6),
        native_invariance_rel=round(inv_maxabs / max(base_maxabs, 1e-9), 6),
        ungauged_amax_ratio=round(ungauged_ratio, 3),
        fp8k_err_pertensor=round(err_pertensor, 5),
        fp8k_err_perchannel=round(err_perchannel, 5),
        fp8k_err_gauged_best=round(best["fp8k_err_gauged"], 5),
        gauge_best_clamp=best["clamp"],
        gauge_beats_pertensor=best["gauge_beats_pertensor"],
        gauge_beats_perchannel=best["gauge_beats_perchannel"],
    )
    return gauge_summary, gauge_sweep


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
    "softmax_kl",
    "classification",
]
GAUGE_SUMMARY_FIELDS = [
    "model",
    "has_k_bias",
    "n_calib_tokens",
    "base_logit_maxabs",
    "native_invariance_maxabs",
    "native_invariance_rel",
    "ungauged_amax_ratio",
    "fp8k_err_pertensor",
    "fp8k_err_perchannel",
    "fp8k_err_gauged_best",
    "gauge_best_clamp",
    "gauge_beats_pertensor",
    "gauge_beats_perchannel",
    "prebias_recovery_fraction_konly",
    "gauge_error",
]
GAUGE_SWEEP_FIELDS = [
    "model",
    "clamp",
    "fp8k_err_gauged",
    "gauge_beats_pertensor",
    "gauge_beats_perchannel",
    "post_gauge_amax_ratio",
]


def _load_thresholds():
    from . import configs as CFG

    return CFG.load("thresholds")["classification"]


def _write_outputs(out_dir, short_name, model_id, rows, summary, sweep, arch, manifest):
    C.write_manifest(C.model_csv_path(out_dir, short_name, "."), manifest)
    cell_csv = C.model_csv_path(out_dir, short_name, "cells.csv")
    for r in rows:
        C.append_row(cell_csv, r, CELL_FIELDS)
    C.append_row(
        C.model_csv_path(out_dir, short_name, "qk_gauge.csv"),
        summary,
        GAUGE_SUMMARY_FIELDS,
    )
    sweep_csv = C.model_csv_path(out_dir, short_name, "qk_gauge_sweep.csv")
    for r in sweep:
        C.append_row(sweep_csv, r, GAUGE_SWEEP_FIELDS)
    card = C.model_csv_path(out_dir, short_name, "card.md")
    with open(card, "w") as f:
        f.write(f"# {model_id} -- FP8 KV-cache smoke\n\n")
        f.write(f"arch: {json.dumps(arch)}\n\n")
        f.write(
            "| cell | ppl | ppl_mult | nll_inc% | top1 | logit_err | kl | class |\n"
        )
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['cell']} | {r['ppl']} | {r['ppl_mult']} | {r['nll_increase_pct']} "
                f"| {r['top1_agreement']} | {r['mean_logit_err']} | {r['softmax_kl']} "
                f"| {r['classification']} |\n"
            )
        f.write(
            f"\nQK-gauge (has_k_bias={summary['has_k_bias']}, "
            f"n_calib_tokens={summary['n_calib_tokens']}):\n"
            f"- native invariance: maxabs={summary['native_invariance_maxabs']} "
            f"rel={summary['native_invariance_rel']} (bf16 residual ~ rel*|logit|, expected; "
            f"a TRUE gauge has rel ~ 0)\n"
            f"- FP8-K err: per-tensor={summary['fp8k_err_pertensor']} (the failure baseline), "
            f"per-channel={summary['fp8k_err_perchannel']} (the free competitor), "
            f"gauged-best={summary['fp8k_err_gauged_best']} @clamp {summary['gauge_best_clamp']}\n"
            f"- gauge beats per-tensor={summary['gauge_beats_pertensor']}, "
            f"beats per-channel={summary['gauge_beats_perchannel']} "
            f"(K outlier ratio ungauged={summary['ungauged_amax_ratio']})\n"
            f"- K-only prebias recovery={summary['prebias_recovery_fraction_konly']} "
            f"(NA on biasless control -- do not cite as bias-fix evidence)\n"
        )
    return cell_csv


def run_one(
    model_id, short_name, out_dir, device, dtype, n, seq_len, seed, device_map=None
):
    import gc

    torch.manual_seed(seed)
    # device_map="auto" shards a too-big model (e.g. Qwen2.5-72B) across GPU+CPU. max_memory forces
    # CPU (not disk/meta) overflow so biases stay real tensors; activations ride accelerate hooks to
    # the GPU during the forward, so the interface harness + gauge still work (atlas cells only --
    # AR generation over an offloaded 72B would be hours, so the gauge/AR caller should keep n small).
    if device_map:
        model, tok = kbc.load_model(model_id, dtype, device, device_map=device_map)
    else:
        model, tok = kbc.load_model(model_id, dtype, device)
    infos = AD.discover(model)
    ids_list = kbc.calib_prompts(tok, n=n, seq_len=seq_len)
    thr = _load_thresholds()
    rows, summary, sweep, arch = run_model(
        model, infos, ids_list, device, thr, short_name
    )
    manifest = C.RunManifest(
        run_id=f"smoke-{short_name}-s{seed}",
        model_id=model_id,
        dtype=dtype,
        device=str(device),
        measurement_level="fake_quant_teacher_forced",
        seed=seed,
        eval_dataset="wikitext-103-raw-v1",
    )
    _write_outputs(out_dir, short_name, model_id, rows, summary, sweep, arch, manifest)
    print(f"[smoke] {short_name}: {[(r['cell'], r['classification']) for r in rows]}")
    print(f"[smoke] {short_name} gauge: {summary}")
    # free the model before the next subprocess/model (cheap insurance even with the per-model
    # subprocess driver; an in-process loop would otherwise stack and OOM a 24GB card).
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, summary


def self_test(out_dir, device="cpu"):
    """Run the FULL pipeline on tiny random-weight CPU models -- no GPU, no download. Proves every
    code path (discovery, atlas cells, prebias, QK-gauge probe, capture, classify, writers).
    """
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
        # GPT-NeoX/Pythia: gpt_neox.layers path + interleaved fused QKV (prebias must be skipped) +
        # partial RoPE from the rope dict. Proves the phase-8 legacy path GPU-free.
        "tiny-neox-interleaved": GPTNeoXForCausalLM(
            GPTNeoXConfig(num_attention_heads=4, rotary_pct=0.25, **common)
        ),
    }
    thr = _load_thresholds()
    torch.manual_seed(0)
    ids_list = [torch.randint(0, 64, (24,)).tolist() for _ in range(3)]
    ok = True
    for sn, model in builds.items():
        model = model.eval().to(device)
        infos = AD.discover(model)
        rows, summary, sweep, arch = run_model(model, infos, ids_list, device, thr, sn)
        manifest = C.RunManifest(
            run_id=f"selftest-{sn}",
            model_id=sn,
            dtype="float32",
            device=device,
            measurement_level="fake_quant_teacher_forced",
        )
        _write_outputs(out_dir, sn, sn, rows, summary, sweep, arch, manifest)
        # invariants the self-test asserts (fp32 CPU -> the gauge is exact, rel ~ 0):
        native = [r for r in rows if r["cell"] == "native"][0]
        assert native["classification"] == "tolerant", "native must be tolerant"
        assert abs(native["mean_logit_err"]) < 1e-4, "native vs base must be ~0"
        assert (
            summary["native_invariance_rel"] < 1e-4
        ), "QK-gauge must be score-invariant (fp32)"
        assert len(sweep) == len(GAUGE_CLAMPS), "clamp sweep must emit a row per clamp"
        cells_run = {r["cell"] for r in rows}
        if arch.get("fused_interleaved"):
            # NeoX: prebias cells skipped (interleaved QKV), recovery NA, but atlas + gauge present
            assert (
                "prebias_k8v16" not in cells_run
            ), "interleaved model must skip prebias"
            assert summary["prebias_recovery_fraction_konly"] == "NA"
            assert (
                "k8v8" in cells_run
            ), "interleaved model must still run the atlas cells"
        else:
            assert "prebias_k8v16" in cells_run, "K-only prebias cell must run"
        print(
            f"[self-test] {sn} OK  native={native['classification']} "
            f"inv_rel={summary['native_invariance_rel']:.2e} "
            f"recovery_konly={summary['prebias_recovery_fraction_konly']} "
            f"sweep={len(sweep)}"
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
    ap.add_argument(
        "--device-map",
        default=None,
        help="'auto' to shard a too-big model (72B) across GPU+CPU via accelerate",
    )
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
        device_map=args.device_map,
    )


if __name__ == "__main__":
    main()
