"""Phase B trainer: GPT-2 baseline with a pluggable optimizer.

Mirrors the spectral harness train-baseline loop (same data stream,
LR schedule shape, bf16 autocast, fork-ready checkpoints with full
optimizer state) but builds the optimizer by name:

  adamw  torch.optim.AdamW via the harness make_optimizer
  soap   vendored SOAP (fim/fisher_pruning/soap.py, MIT, from
         github.com/nikhilvyas/SOAP) — Adam in Shampoo's eigenbasis
  muon   vendored Muon (fim/fisher_pruning/muon.py, MIT, from
         github.com/KellerJordan/Muon) — nesterov momentum
         orthogonalized by a 5-step Newton-Schulz iteration every
         step, applied to the transformer matmul weights only;
         embeddings and 1-D params get the bundled aux Adam.

The learning-rate schedule is applied as a relative cosine
multiplier per parameter group, so optimizers with different base
learning rates per group (Muon vs its aux Adam) keep their ratios.

Checkpoints keep the optimizer state so optimizer-native factors
(SOAP's GG/Q, Muon's momentum buffer) can be harvested later as
pruning signals.

In-training unstructured pruning (the bitter7 idea generalized —
score with the state the optimizer already maintains, adding no
new variables) is enabled by a "prune" config block:

  "prune": {"target_sparsity": 0.5, "start_step": 2000,
            "end_step": 16000, "interval": 500,
            "signal": "state"}          # or "magnitude"

Sparsity follows the cubic schedule of Zhu & Gupta (2017) between
start and end; masks are per-layer over the transformer matmul
weights, recomputed every `interval` steps from the LIVE optimizer
state ("state" maps to: AdamW -> |w|*(exp_avg_sq)^0.25 (= bitter7),
SOAP -> |w|*(rotated exp_avg_sq projected back)^0.25, Muon ->
|w|*|momentum|^0.5), and re-applied after every optimizer step, so
pruned weights stay zero and pruning is monotone. An optional
"final_eval" block runs the lane's fixed-batch eval at the end.

Run configs are archived in
knlp-key-results/fisher-factored-pruning-20260820/configs/.

  python3 fim/fisher_pruning/train_optimizer.py --config CFG
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fim.fisher_pruning.muon import SingleDeviceMuonWithAuxAdam  # noqa: E402
from fim.fisher_pruning.soap import SOAP  # noqa: E402
from fim.fisher_pruning.batched_soap import BatchedSOAP  # noqa: E402
from fim.reciprocal_attention.gpt2_spectral_ra import (  # noqa: E402
    BinData,
    _amp_ctx,
    _jsonl,
    build_model,
    cosine_lr,
    evaluate,
    make_optimizer,
)
from fim.fisher_pruning.phase_a import _manifest  # noqa: E402


def build_optimizer(model: nn.Module, cfg: dict):
    name = cfg.get("optimizer", "adamw")
    optimizer = None
    if name == "adamw":
        optimizer = make_optimizer(model, cfg)
    elif name == "muon":
        # Muon: the transformer matmul weights. Aux Adam: embeddings
        # (2-D, weight-decayed) and 1-D params (no decay). The tied
        # lm_head shares wte's parameter.
        muon_p, adam_decay, adam_nodecay = [], [], []
        for pname, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if pname.startswith("transformer.h.") and p.dim() >= 2:
                muon_p.append(p)
            elif p.dim() >= 2:
                adam_decay.append(p)
            else:
                adam_nodecay.append(p)
        optimizer = SingleDeviceMuonWithAuxAdam(
            [
                {
                    "params": muon_p,
                    "use_muon": True,
                    "lr": cfg.get("muon_lr", 0.02),
                    "momentum": cfg.get("muon_momentum", 0.95),
                    "weight_decay": cfg.get("muon_weight_decay", 0.01),
                },
                {
                    "params": adam_decay,
                    "use_muon": False,
                    "lr": cfg["lr"],
                    "betas": tuple(cfg.get("betas", (0.9, 0.95))),
                    "eps": 1e-8,
                    "weight_decay": cfg.get("weight_decay", 0.1),
                },
                {
                    "params": adam_nodecay,
                    "use_muon": False,
                    "lr": cfg["lr"],
                    "betas": tuple(cfg.get("betas", (0.9, 0.95))),
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                },
            ]
        )
    elif name == "soap":
        decay, no_decay = [], []
        for _, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        optimizer = SOAP(
            [
                {
                    "params": decay,
                    "weight_decay": cfg.get("weight_decay", 0.1),
                },
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=cfg["lr"],
            betas=tuple(cfg.get("betas", (0.95, 0.95))),
            precondition_frequency=cfg.get("precondition_frequency", 10),
        )
    elif name == "batched_soap":
        # The reference algorithm, unchanged, but parameters that share
        # a shape are held as one stacked tensor so the projections and
        # the eigenbasis refresh issue batched linear algebra instead of
        # one call per parameter. Same config keys as "soap".
        decay, no_decay = [], []
        for _, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        optimizer = BatchedSOAP(
            [
                {
                    "params": decay,
                    "weight_decay": cfg.get("weight_decay", 0.1),
                },
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=cfg["lr"],
            betas=tuple(cfg.get("betas", (0.95, 0.95))),
            precondition_frequency=cfg.get("precondition_frequency", 10),
        )
    elif name == "heavyball_soap":
        # Batched, compiled SOAP from the HeavyBall library. Two
        # settings decide what algorithm this actually is:
        # max_precond_dim below a weight's larger dimension makes the
        # preconditioning ONE-SIDED for that weight, and
        # precise_zeroth_power_mode="newtonschulz" replaces the QR
        # refresh. Both are faster and neither is the reference
        # algorithm, so quality has to be measured, not assumed.
        import heavyball
        import heavyball.utils as hbu

        if cfg.get("heavyball_newtonschulz", True):
            hbu.precise_zeroth_power_mode = "newtonschulz"
        if cfg.get("heavyball_compile_mode_none", True):
            # ROCm: whole-step compilation measured far slower.
            hbu.compile_mode = None
        optimizer = heavyball.SOAP(
            model.parameters(),
            lr=cfg["lr"],
            betas=tuple(cfg.get("betas", (0.9, 0.95))),
            weight_decay=cfg.get("weight_decay", 0.1),
            precondition_frequency=cfg.get("precondition_frequency", 10),
            max_precond_dim=cfg.get("max_precond_dim", 2048),
            merge_dims=cfg.get("merge_dims", True),
            storage_dtype=cfg.get("storage_dtype", "float32"),
        )
    elif name == "kfac":
        # K-FAC-preconditioned AdamW: gpauloski/kfac-pytorch hooks
        # every nn.Linear, accumulates Kronecker factors during
        # forward/backward, and preconditioner.step() rewrites
        # p.grad in place before AdamW consumes it. The
        # preconditioner rides on the optimizer object so the train
        # loop and checkpointing can reach it.
        from kfac.preconditioner import KFACPreconditioner

        optimizer = make_optimizer(model, cfg)
        optimizer.kfac_preconditioner = KFACPreconditioner(
            model,
            factor_update_steps=cfg.get("kfac_factor_update_steps", 10),
            inv_update_steps=cfg.get("kfac_inv_update_steps", 100),
            damping=cfg.get("kfac_damping", 0.003),
            factor_decay=cfg.get("kfac_factor_decay", 0.95),
            kl_clip=cfg.get("kfac_kl_clip", 0.001),
            lr=lambda x: optimizer.param_groups[0]["lr"],
            update_factors_in_hook=False,
            grad_scaler=None,
        )
    if optimizer is None:
        raise ValueError(f"unknown optimizer {name!r}")
    return optimizer


def base_learning_rates(optimizer, fallback: float) -> list:
    """Per-group learning rates as configured, held OUTSIDE the
    optimizer.

    The schedule is applied as a relative multiplier so groups with
    different base rates (Muon and its auxiliary Adam) keep their
    ratio. Stashing the base in the group dict is not safe: HeavyBall
    keeps its own "base_lr" entry and resynchronizes it from "lr"
    whenever the two differ, so a stashed base would be overwritten by
    the scaled rate and the schedule would compound geometrically.
    """
    return [g.get("lr", fallback) for g in optimizer.param_groups]


def apply_lr_multiplier(optimizer, base_lrs: list, mult: float) -> None:
    for group, base in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base * mult


def cubic_sparsity(step: int, pcfg: dict) -> float:
    """Zhu & Gupta (2017) gradual schedule: 0 before start_step,
    target after end_step, cubic ramp between."""
    s0, s1 = pcfg["start_step"], pcfg["end_step"]
    target = pcfg["target_sparsity"]
    if step < s0:
        return 0.0
    if step >= s1:
        return target
    frac = (step - s0) / (s1 - s0)
    return target * (1 - (1 - frac) ** 3)


def live_state_scores(
    optimizer,
    opt_name: str,
    targets: dict,
    signal: str,
    q: float = 0.25,
    muon_mode: str = "momentum",
) -> dict:
    """Pruning scores from the optimizer's CURRENT state — the
    free variables it already maintains, nothing extra beyond
    transient scratch at the mask event.

    The whole family is score_q = |w| * (F + eps)^q where F is a
    squared-gradient-scale statistic: q=0 is magnitude, q=0.25 is
    bitter7's damped exponent, q=0.5 ranks identically to the
    diagonal second-order deletion cost w^2 * F.

    F per optimizer: AdamW uses exp_avg_sq; SOAP uses its rotated
    exp_avg_sq projected back through the squared eigenbases; Muon
    keeps no second moment, so F is the square of either the raw
    momentum buffer (muon_mode="momentum") or the actual
    Newton-Schulz-orthogonalized update recomputed from that buffer
    (muon_mode="update" — deterministic from state, no gradient
    needed, transient only); the K-FAC-preconditioned arm uses the
    live Kronecker factor diagonals, F = diag(G)_i * diag(A)_j
    (bias coordinate stripped from A). signal="obs" (K-FAC only)
    instead ranks by the Optimal-Brain-Surgeon deletion cost
    w^2 / [(G+dI)^-1_ii (A+dI)^-1_jj] from the damped factor
    inverses.
    """
    from fim.fisher_pruning.muon import zeropower_via_newtonschulz5
    from fim.fisher_pruning.phase_a import EPS

    kfac_factors = {}
    if opt_name == "kfac" and signal in ("state", "obs"):
        pre = optimizer.kfac_preconditioner
        for module, (mod_name, layer) in pre._layers.items():
            if layer.a_factor is None or layer.g_factor is None:
                raise RuntimeError(f"no K-FAC factors yet for {mod_name}")
            kfac_factors[mod_name] = (layer.a_factor, layer.g_factor)

    out = {}
    for name, p in targets.items():
        w = p.detach()
        if signal == "magnitude":
            out[name] = w.abs()
            continue
        if opt_name == "kfac":
            a_fac, g_fac = kfac_factors[name]
            in_d = p.shape[1]
            if signal == "obs":
                from fim.fisher_pruning.kfac_capture import damped_inverse_diag

                inv_a = damped_inverse_diag(a_fac[:in_d, :in_d].float().cpu(), 1e-2)
                inv_g = damped_inverse_diag(g_fac.float().cpu(), 1e-2)
                denom = torch.clamp(
                    torch.outer(inv_g.float(), inv_a.float()), min=1e-30
                )
                out[name] = (w.detach().float().cpu().pow(2) / denom).to(w.device)
                continue
            a_diag = torch.diagonal(a_fac)[:in_d].to(w.dtype)
            g_diag = torch.diagonal(g_fac).to(w.dtype)
            f_stat = torch.outer(g_diag, a_diag).clamp(min=0)
            out[name] = w.abs() * (f_stat + EPS) ** q
            continue
        state = optimizer.state.get(p)
        if not state:
            raise RuntimeError(f"no optimizer state yet for {name}")
        if opt_name == "adamw":
            f_stat = state["exp_avg_sq"]
        elif opt_name in ("soap", "batched_soap"):
            # BatchedSOAP mirrors per-parameter views of its stacked
            # state into optimizer.state[p], so the layout is the same.
            v_rot = state["exp_avg_sq"]
            qb = state["Q"]
            ql2 = qb[0].to(w.dtype).pow(2)
            qr2 = qb[1].to(w.dtype).pow(2)
            f_stat = (ql2 @ v_rot.to(w.dtype) @ qr2.T).clamp(min=0)
        elif opt_name == "muon":
            mom = state["momentum_buffer"]
            if muon_mode == "update":
                upd = zeropower_via_newtonschulz5(mom.float(), steps=5)
                upd = upd * max(1, mom.size(-2) / mom.size(-1)) ** 0.5
                f_stat = upd.to(w.dtype).pow(2)
            else:
                f_stat = mom.pow(2)
        else:
            raise ValueError(f"no state signal for optimizer {opt_name!r}")
        out[name] = w.abs() * (f_stat + EPS) ** q
    return out


def state_byte_accounting(model, optimizer, masks: dict) -> dict:
    """Byte accounting per CLOUD_WAVE_PLAN: what is model, gradient,
    native optimizer state, and pruning-only persistent state. The
    boolean masks ARE persistent pruning-only state and are reported
    as such — 'free' refers to the score, not the mask."""

    def _tensor_bytes(obj):
        if torch.is_tensor(obj):
            return obj.numel() * obj.element_size()
        if isinstance(obj, dict):
            return sum(_tensor_bytes(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(_tensor_bytes(v) for v in obj)
        return 0

    acct = {
        "weights_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "gradients_bytes": sum(
            p.grad.numel() * p.grad.element_size()
            for p in model.parameters()
            if p.grad is not None
        ),
        "optimizer_state_bytes": _tensor_bytes(dict(optimizer.state)),
        "pruning_persistent_bytes": sum(
            m.numel() * m.element_size() for m in masks.values()
        ),
    }
    if torch.cuda.is_available():
        acct["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        acct["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    return acct


def _mask_hash(masks: dict) -> str:
    import hashlib

    h = hashlib.sha256()
    for name in sorted(masks):
        h.update(name.encode())
        h.update(masks[name].cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _prune_targets(model: nn.Module, n_layer: int) -> dict:
    from fim.fisher_pruning.kfac_capture import default_target_names

    params = dict(model.named_parameters())
    return {n: params[n + ".weight"] for n in default_target_names(n_layer)}


def _save_ckpt(path: Path, model, optimizer, step, batch_counter, cfg, masks):
    """Atomic checkpoint incl. the pruning masks, so a resumed run
    keeps pruned weights pruned between reload and the next mask
    event (weights alone would revive under gradient updates)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "batch_counter": batch_counter,
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "prune_masks": {n: m.cpu() for n, m in masks.items()},
            "train_config": cfg,
        },
        tmp,
    )
    tmp.replace(path)


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _pct(values, frac: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(frac * len(s)))]


def cmd_train(cfg: dict, device: str, resume: bool = False) -> None:
    from collections import deque

    from fim.fisher_pruning.kfac_capture import per_layer_mask

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "train.jsonl"
    torch.manual_seed(cfg["seed"])
    model = build_model(cfg["model"], selection={})
    model.to(device)
    train_data = BinData(cfg["data"]["train_bin"], cfg["model"]["block_size"])
    val_data = BinData(cfg["data"]["val_bin"], cfg["model"]["block_size"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    val_gen_seed = cfg["seed"] + 1000
    tcfg = cfg["train"]
    optimizer = build_optimizer(model, tcfg)
    base_lrs = base_learning_rates(optimizer, tcfg["lr"])
    pcfg = cfg.get("prune")
    masks = {}
    prune_targets = {}
    if pcfg:
        prune_targets = _prune_targets(model, cfg["model"]["n_layer"])
    batch_counter = 0
    start_step = 0
    ckpt_latest = out_dir / "ckpt_latest.pt"
    if resume and ckpt_latest.exists():
        ck = torch.load(ckpt_latest, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        torch.set_rng_state(ck["rng_torch"])
        if ck.get("rng_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["rng_cuda"])
        masks = {
            n: m.to(device=prune_targets[n].device, dtype=torch.bool)
            for n, m in ck.get("prune_masks", {}).items()
        }
        batch_counter = ck["batch_counter"]
        start_step = ck["step"] + 1
        train_data.skip(batch_counter, tcfg["batch_size"], gen)
    t0 = time.time()
    tokens_per_step = tcfg["batch_size"] * cfg["model"]["block_size"]
    step_ms = deque(maxlen=200)
    opt_ms = deque(maxlen=200)
    prev_pruned = None
    _jsonl(
        log,
        _manifest(
            cfg,
            {
                "event": "start" if start_step == 0 else "resume",
                "run_id": out_dir.name,
                "device": device,
                "optimizer": type(optimizer).__name__,
                "start_step": start_step,
            },
        ),
    )
    for step in range(start_step, tcfg["max_steps"]):
        lr = cosine_lr(step, tcfg)
        apply_lr_multiplier(optimizer, base_lrs, lr / tcfg["lr"])
        x, y = train_data.batch(tcfg["batch_size"], gen, device)
        batch_counter += 1
        _sync(device)
        t_step = time.perf_counter()
        with _amp_ctx(device):
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.get("clip", 1.0))
        _sync(device)
        t_opt = time.perf_counter()
        if hasattr(optimizer, "kfac_preconditioner"):
            optimizer.kfac_preconditioner.step()
        optimizer.step()
        _sync(device)
        opt_ms.append((time.perf_counter() - t_opt) * 1e3)
        if pcfg:
            if (
                pcfg["start_step"] <= step
                and step % pcfg["interval"] == 0
                and cubic_sparsity(step, pcfg) > 0
            ):
                sparsity = cubic_sparsity(step, pcfg)
                t_score = time.perf_counter()
                scores = live_state_scores(
                    optimizer,
                    tcfg.get("optimizer", "adamw"),
                    prune_targets,
                    pcfg.get("signal", "state"),
                    q=pcfg.get("q", 0.25),
                    muon_mode=pcfg.get("muon_mode", "momentum"),
                )
                _sync(device)
                score_ms = (time.perf_counter() - t_score) * 1e3
                t_mask = time.perf_counter()
                masks = {
                    n: per_layer_mask(s.float().cpu(), sparsity).to(s.device)
                    for n, s in scores.items()
                }
                mask_ms = (time.perf_counter() - t_mask) * 1e3
                kept = sum(m.sum().item() for m in masks.values())
                total = sum(m.numel() for m in masks.values())
                pruned_now = {n: ~m for n, m in masks.items()}
                churn = None
                if prev_pruned is not None:
                    changed = sum(
                        (pruned_now[n] ^ prev_pruned[n]).sum().item()
                        for n in pruned_now
                    )
                    churn = changed / total
                    monotone = all(
                        bool((prev_pruned[n] & masks[n]).sum() == 0) for n in pruned_now
                    )
                else:
                    monotone = True
                prev_pruned = pruned_now
                mag_masks = {
                    n: per_layer_mask(
                        prune_targets[n].detach().abs().float().cpu(), sparsity
                    )
                    for n in prune_targets
                }
                inter = sum(
                    ((~mag_masks[n]).to(pruned_now[n].device) & pruned_now[n])
                    .sum()
                    .item()
                    for n in pruned_now
                )
                union = sum(
                    ((~mag_masks[n]).to(pruned_now[n].device) | pruned_now[n])
                    .sum()
                    .item()
                    for n in pruned_now
                )
                nonfinite = sum(
                    (~torch.isfinite(s)).sum().item() for s in scores.values()
                )
                sample = torch.cat(
                    [
                        s.detach().float().reshape(-1)[::97].cpu()
                        for s in scores.values()
                    ]
                )
                event = {
                    "event": "prune",
                    "step": step,
                    "target_sparsity": sparsity,
                    "actual_sparsity": 1 - kept / total,
                    "score_ms": score_ms,
                    "mask_ms": mask_ms,
                    "mask_churn": churn,
                    "mask_monotone": monotone,
                    "jaccard_vs_magnitude": inter / max(union, 1),
                    "score_nonfinite": nonfinite,
                    "score_q10": float(torch.quantile(sample, 0.10)),
                    "score_q50": float(torch.quantile(sample, 0.50)),
                    "score_q90": float(torch.quantile(sample, 0.90)),
                    "mask_hash": _mask_hash(masks),
                    "per_layer_sparsity": {
                        n: float(1 - masks[n].float().mean()) for n in masks
                    },
                }
                if len(prev_pruned) and churn is None:
                    event["accounting"] = state_byte_accounting(model, optimizer, masks)
                _jsonl(log, event)
            if masks:
                with torch.no_grad():
                    for n, m in masks.items():
                        prune_targets[n].mul_(m.to(prune_targets[n].dtype))
        _sync(device)
        step_ms.append((time.perf_counter() - t_step) * 1e3)
        if step % tcfg.get("log_interval", 50) == 0:
            elapsed = time.time() - t0
            _jsonl(
                log,
                {
                    "event": "train",
                    "step": step,
                    "loss": float(loss.item()),
                    # what the schedule asked for, and what the
                    # optimizer actually holds: a mismatch means the
                    # optimizer rewrote the rate behind the schedule.
                    "lr_scheduled": lr,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "elapsed_s": elapsed,
                    "tokens_seen": tokens_per_step * (step + 1),
                    "tokens_per_s": tokens_per_step * (step + 1 - start_step) / elapsed,
                    "step_ms_p50": _pct(step_ms, 0.50),
                    "step_ms_p99": _pct(step_ms, 0.99),
                    "optimizer_step_ms_p50": _pct(opt_ms, 0.50),
                },
            )
        if step % tcfg.get("eval_interval", 500) == 0 or step == tcfg["max_steps"] - 1:
            val_loss = evaluate(
                model,
                val_data,
                tcfg.get("eval_batches", 20),
                tcfg["batch_size"],
                device,
                seed=val_gen_seed,
            )
            _jsonl(
                log,
                {
                    "event": "eval",
                    "step": step,
                    "val_loss": val_loss,
                    "val_ppl": math.exp(val_loss),
                },
            )
        if (step > 0 and step % tcfg.get("ckpt_interval", 5000) == 0) or step == tcfg[
            "max_steps"
        ] - 1:
            _save_ckpt(
                out_dir / f"ckpt_step{step}.pt",
                model,
                optimizer,
                step,
                batch_counter,
                cfg,
                masks,
            )
            _save_ckpt(ckpt_latest, model, optimizer, step, batch_counter, cfg, masks)
        if tcfg.get("halt_after_step") is not None and step >= tcfg["halt_after_step"]:
            # Simulated preemption for resume testing: stop AFTER the
            # checkpoint block, without touching the LR schedule the
            # way a shortened max_steps would.
            _save_ckpt(ckpt_latest, model, optimizer, step, batch_counter, cfg, masks)
            _jsonl(log, {"event": "halted", "step": step})
            return
    fcfg = cfg.get("final_eval")
    if fcfg:
        for seed in fcfg["seeds"]:
            val_loss = evaluate(
                model,
                val_data,
                fcfg["n_batches"],
                tcfg["batch_size"],
                device,
                seed=seed,
            )
            _jsonl(
                log,
                {
                    "event": "final_eval",
                    "eval_seed": seed,
                    "val_loss": val_loss,
                    "val_ppl": math.exp(val_loss),
                },
            )
    if pcfg and masks:
        kept = sum(m.sum().item() for m in masks.values())
        total = sum(m.numel() for m in masks.values())
        _jsonl(
            log,
            {
                "event": "final_sparsity",
                "actual_sparsity": 1 - kept / total,
                "accounting": state_byte_accounting(model, optimizer, masks),
            },
        )
    done = {"event": "done", "elapsed_s": time.time() - t0}
    if torch.cuda.is_available():
        done["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        done["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    _jsonl(log, done)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="continue from out_dir/ckpt_latest.pt if present",
    )
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    cmd_train(cfg, args.device, resume=args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
