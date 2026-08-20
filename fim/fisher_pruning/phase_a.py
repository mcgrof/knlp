"""Factored-Fisher pruning signal head-to-head (phase A).

Compares unstructured one-shot pruning scores on one trained GPT-2
checkpoint whose Adam state was saved:

  random      floor control
  magnitude   |w|
  bitter7     |w| * (exp_avg_sq + eps)^0.25   (Adam state, history)
  calib_diag  |w| * (D + eps)^0.25            (exact per-token diag
                                               empirical Fisher from a
                                               calibration pass)
  kron_diag   |w| * (G_ii * A_jj + eps)^0.25  (per-dimension i.e.
                                               rank-1 Kronecker
                                               diagonal — the factored
                                               FIM simplification)
  kfac_obs    w^2 / [(G+l)^-1_ii (A+l)^-1_jj] (OBS-style; the only
                                               arm using off-diagonal
                                               curvature)

Masks are per-layer uniform sparsity (primary) with global pooling as
a secondary scope. Eval is the spectral harness fixed-batch-set val
loss, three disjoint batch sets per cell for a noise floor.

Design contract: knlp-key-results/fisher-factored-pruning-20260820/
PLAN.md. Two commands:

  python3 fim/fisher_pruning/phase_a.py calibrate  --config CFG
  python3 fim/fisher_pruning/phase_a.py prune-eval --config CFG
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fim.fisher_pruning.kfac_capture import (  # noqa: E402
    FactorAccumulator,
    damped_inverse_diag,
    default_target_names,
    global_masks,
    per_layer_mask,
)
from fim.reciprocal_attention.gpt2_spectral_ra import (  # noqa: E402
    BinData,
    _git_commit,
    _git_dirty,
    _jsonl,
    _sha256_file,
    _sha256_json,
    build_model,
    evaluate,
    make_optimizer,
)

EPS = 1e-8


def _own_code_sha256() -> str:
    h = hashlib.sha256()
    for name in ("kfac_capture.py", "phase_a.py"):
        h.update((Path(__file__).parent / name).read_bytes())
    return h.hexdigest()


def _manifest(cfg: Dict, extra: Dict) -> Dict:
    man = {
        "code_commit": _git_commit(),
        "code_dirty": _git_dirty(),
        "lane_code_sha256": _own_code_sha256(),
        "config_hash": _sha256_json(cfg),
        "config": cfg,
        "created_unix": time.time(),
    }
    man.update(extra)
    return man


def _load_ckpt(cfg: Dict):
    ckpt = torch.load(cfg["ckpt"], map_location="cpu", weights_only=False)
    model = build_model(ckpt["train_config"]["model"], selection={})
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected[:5]}")
    real_missing = [m for m in missing if "attn.bias" not in m]
    if real_missing:
        raise RuntimeError(f"missing checkpoint keys: {real_missing[:5]}")
    return ckpt, model


def cmd_calibrate(cfg: Dict, device: str, out_dir: Path, force: bool) -> None:
    calib_path = out_dir / "calib" / "calib.pt"
    if calib_path.exists() and not force:
        raise SystemExit(f"{calib_path} exists; pass --force to overwrite")
    calib_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt, model = _load_ckpt(cfg)
    model.to(device)
    model.eval()  # Fisher of the deployed (dropout-free) function
    n_layer = ckpt["train_config"]["model"]["n_layer"]
    block = ckpt["train_config"]["model"]["block_size"]
    acc = FactorAccumulator(model, default_target_names(n_layer))
    acc.attach()
    data = BinData(cfg["data"]["train_bin"], block)
    ccfg = cfg["calibration"]
    gen = torch.Generator().manual_seed(ccfg["seed"])
    t0 = time.time()
    for i in range(ccfg["n_batches"]):
        x, y = data.batch(ccfg["batch_size"], gen, device)
        _, loss = model(x, y)  # fp32 on purpose: measurement pass
        model.zero_grad(set_to_none=True)
        loss.backward()
        acc.count_batch(x.numel())
        if i % 10 == 0:
            print(f"calib batch {i} loss {loss.item():.4f}", flush=True)
    acc.detach()
    factors = acc.finalize()
    model.zero_grad(set_to_none=True)
    payload = {
        "factors": factors,
        "n_tokens": acc.n_tokens,
        "manifest": _manifest(
            cfg,
            {
                "event": "calibrate",
                "ckpt_sha256": _sha256_file(cfg["ckpt"]),
                "train_bin_sha256": _sha256_file(cfg["data"]["train_bin"]),
                "elapsed_s": time.time() - t0,
            },
        ),
    }
    tmp = calib_path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(calib_path)
    print(f"saved {calib_path} ({acc.n_tokens} tokens)", flush=True)


def _bitter7_exp_avg_sq(ckpt, model) -> Dict[str, torch.Tensor]:
    """exp_avg_sq per target weight, via the harness optimizer path."""
    if ckpt.get("optimizer_state") is None:
        raise RuntimeError("checkpoint has no optimizer state")
    optimizer = make_optimizer(model, ckpt["train_config"]["train"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    out = {}
    params = dict(model.named_parameters())
    n_layer = ckpt["train_config"]["model"]["n_layer"]
    for name in default_target_names(n_layer):
        p = params[name + ".weight"]
        state = optimizer.state.get(p)
        if not state or "exp_avg_sq" not in state:
            raise RuntimeError(f"no exp_avg_sq for {name}")
        out[name] = state["exp_avg_sq"].detach().float().cpu()
    return out


def build_scores(
    cfg: Dict, ckpt, model, factors: Dict
) -> Dict[str, Dict[str, torch.Tensor]]:
    """arm -> {module name -> score tensor [out, in]} (cpu fp32)."""
    n_layer = ckpt["train_config"]["model"]["n_layer"]
    names = default_target_names(n_layer)
    weights = {
        n: dict(model.named_parameters())[n + ".weight"].detach().float().cpu()
        for n in names
    }
    v_adam = _bitter7_exp_avg_sq(ckpt, model)
    rng = torch.Generator().manual_seed(cfg.get("random_arm_seed", 1234))
    scores: Dict[str, Dict[str, torch.Tensor]] = {
        "random": {},
        "magnitude": {},
        "bitter7": {},
        "calib_diag": {},
        "kron_diag": {},
    }
    kappas = cfg.get("obs_kappas", [1e-2, 1e-3])
    for kappa in kappas:
        scores[f"kfac_obs_k{kappa:g}"] = {}
    for n in names:
        w = weights[n]
        f = factors[n]
        diag_g = torch.diagonal(f["G"]).float()
        diag_a = torch.diagonal(f["A"]).float()
        scores["random"][n] = torch.rand(w.shape, generator=rng)
        scores["magnitude"][n] = w.abs()
        scores["bitter7"][n] = w.abs() * (v_adam[n] + EPS) ** 0.25
        scores["calib_diag"][n] = w.abs() * (f["D"].float() + EPS) ** 0.25
        scores["kron_diag"][n] = w.abs() * (torch.outer(diag_g, diag_a) + EPS) ** 0.25
        for kappa in kappas:
            inv_g = damped_inverse_diag(f["G"], kappa).float()
            inv_a = damped_inverse_diag(f["A"], kappa).float()
            denom = torch.clamp(torch.outer(inv_g, inv_a), min=1e-30)
            scores[f"kfac_obs_k{kappa:g}"][n] = w.pow(2) / denom
    return scores


def _apply_masks(model, pristine: Dict, masks: Dict[str, torch.Tensor]):
    model.load_state_dict(pristine, strict=False)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, mask in masks.items():
            p = params[name + ".weight"]
            p.mul_(mask.to(device=p.device, dtype=p.dtype))


def _actual_sparsity(masks: Dict[str, torch.Tensor]) -> float:
    total = sum(m.numel() for m in masks.values())
    kept = sum(m.sum().item() for m in masks.values())
    return 1.0 - kept / total


def cmd_prune_eval(cfg: Dict, device: str, out_dir: Path) -> None:
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    log = results_dir / "phaseA.jsonl"
    ckpt, model = _load_ckpt(cfg)
    calib = torch.load(
        out_dir / "calib" / "calib.pt", map_location="cpu", weights_only=False
    )
    factors = calib["factors"]
    scores = build_scores(cfg, ckpt, model, factors)
    pristine = {k: v.clone() for k, v in ckpt["model_state"].items()}
    model.to(device)
    model.eval()
    block = ckpt["train_config"]["model"]["block_size"]
    val_data = BinData(cfg["data"]["val_bin"], block)
    ecfg = cfg["eval"]
    _jsonl(
        log,
        _manifest(
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

    _apply_masks(model, pristine, {})
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
                _apply_masks(model, pristine, masks)
                eval_cell(
                    {
                        "arm": arm,
                        "sparsity": sparsity,
                        "scope": scope,
                        "actual_sparsity": _actual_sparsity(masks),
                    }
                )

    # Pairwise pruned-set agreement at 50% per-layer: how different
    # the arms' decisions actually are.
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
    ap.add_argument("command", choices=["calibrate", "prune-eval"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out_dir = Path(cfg["out_dir"])
    if args.command == "calibrate":
        cmd_calibrate(cfg, args.device, out_dir, args.force)
    else:
        cmd_prune_eval(cfg, args.device, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
