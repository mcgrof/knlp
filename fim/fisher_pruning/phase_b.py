"""Phase B: SOAP-native pruning signals on the SOAP-trained model.

Adds two arms harvested for free from a SOAP checkpoint's optimizer
state, evaluated alongside the phase-A calibration arms (recomputed
on the SOAP model so the comparison is within-model):

  soap_kron   |w| * (diag(L)_i * diag(R)_j + eps)^0.25 from the
              Shampoo preconditioner accumulators GG = [L, R]
              (EMA of G G^T / G^T G over minibatch gradients) —
              the Shampoo analogue of the K-FAC kron_diag arm.
  soap_rot_v  |w| * (v_ws + eps)^0.25 where v_ws projects SOAP's
              rotated-basis exp_avg_sq back to weight coordinates:
              g = QL g' QR^T with g' decorrelated in the eigenbasis
              gives E[g_ij^2] ~= ((QL^2) @ v' @ (QR^2)^T)_ij —
              the Shampoo analogue of bitter7's Adam-state score.

The plain bitter7 arm is excluded here: SOAP's exp_avg_sq lives in
the rotated basis, so the raw fetch phase A used would be
meaningless on this checkpoint.

Usage (calibration first, reusing the phase-A command):

  python3 fim/fisher_pruning/phase_a.py calibrate  --config CFG_B
  python3 fim/fisher_pruning/phase_b.py prune-eval --config CFG_B

Design contract: knlp-key-results/fisher-factored-pruning-20260820/
PLAN.md, Amendment 1.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fim.fisher_pruning import phase_a  # noqa: E402
from fim.fisher_pruning.kfac_capture import (  # noqa: E402
    default_target_names,
    global_masks,
    per_layer_mask,
)
from fim.fisher_pruning.train_optimizer import build_optimizer  # noqa: E402
from fim.reciprocal_attention.gpt2_spectral_ra import (  # noqa: E402
    BinData,
    _jsonl,
    _sha256_file,
    _sha256_json,
    evaluate,
)

EPS = phase_a.EPS


def soap_native_scores(ckpt, model) -> Dict[str, Dict[str, torch.Tensor]]:
    """soap_kron and soap_rot_v from a SOAP checkpoint's state."""
    if ckpt.get("optimizer_state") is None:
        raise RuntimeError("checkpoint has no optimizer state")
    tcfg = ckpt["train_config"]["train"]
    if tcfg.get("optimizer") != "soap":
        raise RuntimeError("checkpoint was not trained with SOAP")
    optimizer = build_optimizer(model, tcfg)
    optimizer.load_state_dict(ckpt["optimizer_state"])
    params = dict(model.named_parameters())
    n_layer = ckpt["train_config"]["model"]["n_layer"]
    out: Dict[str, Dict[str, torch.Tensor]] = {"soap_kron": {}, "soap_rot_v": {}}
    for name in default_target_names(n_layer):
        p = params[name + ".weight"]
        state = optimizer.state.get(p)
        if not state or "GG" not in state or state.get("Q") is None:
            raise RuntimeError(f"no SOAP factors for {name}")
        gg = state["GG"]
        q = state["Q"]
        if len(gg) != 2 or any(len(m) == 0 for m in gg) or any(len(m) == 0 for m in q):
            raise RuntimeError(f"unexpected SOAP factor layout for {name}")
        w = p.detach().float().cpu()
        diag_l = torch.diagonal(gg[0]).float().cpu()
        diag_r = torch.diagonal(gg[1]).float().cpu()
        out["soap_kron"][name] = w.abs() * (torch.outer(diag_l, diag_r) + EPS) ** 0.25
        v_rot = state["exp_avg_sq"].detach().float().cpu()
        ql2 = q[0].float().cpu().pow(2)
        qr2 = q[1].float().cpu().pow(2)
        v_ws = ql2 @ v_rot @ qr2.T
        out["soap_rot_v"][name] = w.abs() * (v_ws.clamp(min=0) + EPS) ** 0.25
    return out


def muon_native_scores(ckpt, model) -> Dict[str, Dict[str, torch.Tensor]]:
    """muon_momentum: |w| * |momentum_buffer|^0.5 — the free-state
    signal a Muon checkpoint offers. Muon keeps no second moment, so
    this is the closest analogue of the Adam-state score: the EMA'd
    gradient magnitude in place of the EMA'd squared gradient."""
    if ckpt.get("optimizer_state") is None:
        raise RuntimeError("checkpoint has no optimizer state")
    tcfg = ckpt["train_config"]["train"]
    if tcfg.get("optimizer") != "muon":
        raise RuntimeError("checkpoint was not trained with Muon")
    optimizer = build_optimizer(model, tcfg)
    optimizer.load_state_dict(ckpt["optimizer_state"])
    params = dict(model.named_parameters())
    n_layer = ckpt["train_config"]["model"]["n_layer"]
    out: Dict[str, Dict[str, torch.Tensor]] = {"muon_momentum": {}}
    for name in default_target_names(n_layer):
        p = params[name + ".weight"]
        state = optimizer.state.get(p)
        if not state or "momentum_buffer" not in state:
            raise RuntimeError(f"no momentum buffer for {name}")
        w = p.detach().float().cpu()
        mom = state["momentum_buffer"].detach().float().cpu()
        out["muon_momentum"][name] = w.abs() * (mom.abs() + EPS) ** 0.5
    return out


def native_scores(ckpt, model) -> Dict[str, Dict[str, torch.Tensor]]:
    opt_name = ckpt["train_config"]["train"].get("optimizer")
    if opt_name == "soap":
        return soap_native_scores(ckpt, model)
    if opt_name == "muon":
        return muon_native_scores(ckpt, model)
    raise RuntimeError(f"no native score arms for optimizer {opt_name!r}")


def cmd_prune_eval(cfg: Dict, device: str, out_dir: Path) -> None:
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    log = results_dir / "phaseB.jsonl"
    ckpt, model = phase_a._load_ckpt(cfg)
    calib = torch.load(
        out_dir / "calib" / "calib.pt", map_location="cpu", weights_only=False
    )
    scores = phase_a.build_scores(
        cfg, ckpt, model, calib["factors"], include_bitter7=False
    )
    scores.update(native_scores(ckpt, model))
    pristine = {k: v.clone() for k, v in ckpt["model_state"].items()}
    model.to(device)
    model.eval()
    block = ckpt["train_config"]["model"]["block_size"]
    val_data = BinData(cfg["data"]["val_bin"], block)
    ecfg = cfg["eval"]
    _jsonl(
        log,
        phase_a._manifest(
            cfg,
            {
                "event": "start",
                "ckpt_sha256": _sha256_file(cfg["ckpt"]),
                "calib_manifest_hash": _sha256_json(calib["manifest"]),
                "device": device,
                "arms": sorted(scores),
            },
        ),
    )

    def eval_cell(tag: Dict) -> None:
        for seed in ecfg["seeds"]:
            val_loss = evaluate(
                model,
                val_data,
                ecfg["n_batches"],
                ecfg["batch_size"],
                device,
                seed=seed,
            )
            _jsonl(
                log,
                dict(
                    tag,
                    event="eval",
                    eval_seed=seed,
                    val_loss=val_loss,
                    val_ppl=math.exp(val_loss),
                ),
            )
            print(
                f"{tag} seed={seed} loss={val_loss:.4f} "
                f"ppl={math.exp(val_loss):.2f}",
                flush=True,
            )

    phase_a._apply_masks(model, pristine, {})
    eval_cell({"arm": "dense", "sparsity": 0.0, "scope": "none"})

    scopes: Dict[str, List[float]] = cfg["scopes"]
    t0 = time.time()
    for arm, arm_scores in scores.items():
        for scope, sparsities in scopes.items():
            for sparsity in sparsities:
                if scope == "per_layer":
                    masks = {
                        n: per_layer_mask(s, sparsity) for n, s in arm_scores.items()
                    }
                else:
                    masks = global_masks(arm_scores, sparsity)
                phase_a._apply_masks(model, pristine, masks)
                eval_cell(
                    {
                        "arm": arm,
                        "sparsity": sparsity,
                        "scope": scope,
                        "actual_sparsity": phase_a._actual_sparsity(masks),
                    }
                )

    overlap = {}
    arm_names = sorted(scores)
    masks50 = {
        a: {n: per_layer_mask(s, 0.5) for n, s in scores[a].items()} for a in arm_names
    }
    for i, a in enumerate(arm_names):
        for b in arm_names[i + 1 :]:
            inter = 0
            union = 0
            for n in masks50[a]:
                pa = ~masks50[a][n]
                pb = ~masks50[b][n]
                inter += (pa & pb).sum().item()
                union += (pa | pb).sum().item()
            overlap[f"{a}|{b}"] = inter / max(union, 1)
    _jsonl(
        log,
        {
            "event": "mask_overlap_jaccard_50pct",
            "overlap": overlap,
            "elapsed_s": time.time() - t0,
        },
    )
    _jsonl(log, {"event": "done", "elapsed_s": time.time() - t0})
    print("done", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prune-eval"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    cmd_prune_eval(cfg, args.device, Path(cfg["out_dir"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
