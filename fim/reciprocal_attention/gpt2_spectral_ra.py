"""spectral_delta_ra lane: GPT-2 (model_knlp) harness.

Implements, beside the immutable legacy RA in gpt2/model_knlp.py:

  * SpectralDeltaAttention — baseline causal attention plus a gated
    per-head correction built from the reciprocal delta
    delta = SDPA(k,q,v) - SDPA(q,k,v), applied in head space BEFORE
    concatenation and the shared output projection. No learned
    d_head -> d_model RA projection exists in this variant. Gate
    modes: "spectral" (frozen orthonormal U, per-mode beta),
    "scalar_delta", "coordinate_diag", "standard_extra_lowrank"
    (same parameters applied to the standard output — a capacity
    control), and "none" (exact baseline / audit capture).
    All betas are beta_max * tanh(raw_beta) with raw_beta == 0 at
    init, so every variant is exactly baseline at initialization.

  * audit mode — measurement only: counterfactual capture of the
    reciprocal delta R and the output gradient G = dL/dY_std on
    selected heads, accumulation of the signed reciprocal credit
    operator H = -sym(E[G^T R]) and its diagnostic companions
    (see spectral_credit.py), permutation nulls, split-half
    stability, and candidate basis bundles.

  * train-baseline mode — plain GPT-2 training from tokenized .bin
    files with fork-ready checkpoints (model/optimizer/RNG/loader
    position).

  * oracle mode — freeze every base parameter of a checkpoint and
    train only the gates; per-step beta trajectory logging; RA-on
    vs RA-off and per-mode ablations on held-out batches.

Design contract: knlp-key-results/ra-spectral-modes-20260810/PLAN.md.
The signed operator H is NOT a Fisher Information Matrix.
No torch.compile anywhere in this lane.
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fim.reciprocal_attention import spectral_credit as sc  # noqa: E402
from gpt2.model_knlp import (  # noqa: E402
    GPT2_KNLP,
    GPT2_KNLP_Config,
    CausalSelfAttention_KNLP,
)

GATE_MODES = (
    "none",
    "scalar_delta",
    "spectral",
    "coordinate_diag",
    "standard_extra_lowrank",
)


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------


class SpectralDeltaAttention(CausalSelfAttention_KNLP):
    """Baseline attention + mode-gated reciprocal-delta correction.

    The baseline path is pristine (no legacy RA, no SDPA gate). For
    the selected heads of this layer the reciprocal head output is
    computed with swapped Q/K under the same causal mask, and the
    correction is applied per head before concatenation:

        delta      = SDPA(k,q,v) - SDPA(q,k,v)         [B, S, T, d]
        beta       = beta_max * tanh(raw_beta)
        spectral:            corr = ((delta @ U) * beta) @ U^T
        scalar_delta:        corr = beta * delta
        coordinate_diag:     corr = beta * delta        (beta per feature)
        standard_extra_lowrank: corr = ((std @ U) * beta) @ U^T

    U is a frozen buffer (never a Parameter): no gradient, no
    optimizer state. Eigenvector sign flips leave the output
    unchanged because U appears an even number of times.
    """

    def __init__(
        self,
        config: GPT2_KNLP_Config,
        selected_heads: List[int],
        gate_mode: str = "none",
        rank: int = 0,
        beta_max: float = 1.0,
    ):
        super().__init__(config, use_sdpa_gate=False, use_ra=False)
        if gate_mode not in GATE_MODES:
            raise ValueError(f"unknown gate_mode {gate_mode!r}")
        if len(set(selected_heads)) != len(selected_heads):
            raise ValueError("duplicate selected heads")
        for h in selected_heads:
            if not 0 <= h < self.n_head:
                raise ValueError(f"head {h} out of range")
        self.gate_mode = gate_mode
        self.rank = rank
        self.beta_max = beta_max
        self.audit_capture = False
        self.capture_scores = False
        self._audit: Dict[str, torch.Tensor] = {}
        n_sel = len(selected_heads)
        self.register_buffer(
            "selected_heads",
            torch.tensor(sorted(selected_heads), dtype=torch.long),
            persistent=True,
        )
        if gate_mode in ("spectral", "standard_extra_lowrank"):
            if not 1 <= rank <= self.head_dim:
                raise ValueError(f"rank {rank} out of range")
            self.register_buffer(
                "basis_u",
                torch.zeros(n_sel, self.head_dim, rank),
                persistent=True,
            )
            self.raw_beta = nn.Parameter(torch.zeros(n_sel, rank))
        elif gate_mode == "scalar_delta":
            self.register_buffer("basis_u", None, persistent=False)
            self.raw_beta = nn.Parameter(torch.zeros(n_sel, 1))
        elif gate_mode == "coordinate_diag":
            self.register_buffer("basis_u", None, persistent=False)
            self.raw_beta = nn.Parameter(torch.zeros(n_sel, self.head_dim))
        else:  # none
            self.register_buffer("basis_u", None, persistent=False)
            self.register_parameter("raw_beta", None)

    def set_basis(self, u: torch.Tensor) -> None:
        """Install a frozen [n_sel, d_head, r] orthonormal basis."""
        if self.basis_u is None:
            raise RuntimeError(f"gate_mode {self.gate_mode!r} takes no basis")
        if u.shape != self.basis_u.shape:
            raise ValueError(
                f"basis shape {tuple(u.shape)} != {tuple(self.basis_u.shape)}"
            )
        with torch.no_grad():
            self.basis_u.copy_(u.to(self.basis_u.dtype))

    def forward(self, x, mechint_kv_mask=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        yb = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )  # [B, nh, T, d]

        sel = self.selected_heads
        need_rec = sel.numel() > 0 and (
            self.audit_capture
            or self.gate_mode in ("scalar_delta", "spectral", "coordinate_diag")
        )
        need_std_gate = sel.numel() > 0 and self.gate_mode == "standard_extra_lowrank"

        if self.audit_capture:
            self._audit = {}
            yb.register_hook(self._store_grad)

        rec = None
        if need_rec:
            rec_ctx = torch.no_grad() if self.audit_capture else nullcontext()
            with rec_ctx:
                rec = F.scaled_dot_product_attention(
                    k[:, sel],  # swapped: keys act as queries
                    q[:, sel],  # swapped: queries act as keys
                    v[:, sel],
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=True,
                )  # [B, S, T, d]

        if self.audit_capture and rec is not None:
            std_sel = yb[:, sel]
            self._audit["delta"] = (rec - std_sel).detach()
            self._audit["q_sel"] = q[:, sel].detach()
            self._audit["k_sel"] = k[:, sel].detach()
            if self.capture_scores:
                self._audit["scores_premask"] = (
                    q[:, sel] @ k[:, sel].transpose(-2, -1)
                ).detach() / math.sqrt(self.head_dim)

        if self.raw_beta is not None and (need_rec or need_std_gate):
            beta = self.beta_max * torch.tanh(self.raw_beta)
            std_sel = yb[:, sel]
            if self.gate_mode == "scalar_delta":
                corr = beta[None, :, None, :] * (rec - std_sel)
            elif self.gate_mode == "coordinate_diag":
                corr = beta[None, :, None, :] * (rec - std_sel)
            elif self.gate_mode == "spectral":
                delta = rec - std_sel
                modes = torch.einsum("bstd,sdr->bstr", delta, self.basis_u)
                corr = torch.einsum(
                    "bstr,sdr->bstd", modes * beta[None, :, None, :], self.basis_u
                )
            else:  # standard_extra_lowrank
                modes = torch.einsum("bstd,sdr->bstr", std_sel, self.basis_u)
                corr = torch.einsum(
                    "bstr,sdr->bstd", modes * beta[None, :, None, :], self.basis_u
                )
            yb = yb.clone()
            yb[:, sel] = std_sel + corr

        y = yb.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

    def _store_grad(self, grad: torch.Tensor) -> None:
        self._audit["g_full"] = grad.detach()


def build_model(
    model_cfg: Dict,
    selection: Dict[int, List[int]],
    gate_mode: str = "none",
    rank: int = 0,
    beta_max: float = 1.0,
) -> GPT2_KNLP:
    """Baseline GPT2_KNLP with SpectralDeltaAttention on selected layers.

    selection maps layer index -> list of head indices. Layers not in
    the selection keep the stock (baseline) attention module.
    """
    cfg = GPT2_KNLP_Config(use_sdpa_gate=False, use_ra=False, **model_cfg)
    model = GPT2_KNLP(cfg)
    for layer_idx, heads in selection.items():
        if not 0 <= int(layer_idx) < cfg.n_layer:
            raise ValueError(f"layer {layer_idx} out of range")
        block = model.transformer.h[int(layer_idx)]
        new_attn = SpectralDeltaAttention(
            cfg,
            selected_heads=list(heads),
            gate_mode=gate_mode,
            rank=rank,
            beta_max=beta_max,
        )
        new_attn.load_state_dict(block.attn.state_dict(), strict=False)
        block.attn = new_attn
    return model


def spectral_modules(model: GPT2_KNLP) -> Dict[int, SpectralDeltaAttention]:
    out = {}
    for i, block in enumerate(model.transformer.h):
        if isinstance(block.attn, SpectralDeltaAttention):
            out[i] = block.attn
    return out


def gate_parameters(model: GPT2_KNLP) -> List[nn.Parameter]:
    return [
        m.raw_beta for m in spectral_modules(model).values() if m.raw_beta is not None
    ]


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------


class BinData:
    """nanoGPT-style uint16 token bins with a seeded batch stream.

    The batch counter makes the stream forkable: saving (seed,
    counter) and re-seeding reproduces the exact remaining order.
    """

    def __init__(self, path: str, block_size: int):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.path = path

    def batch(
        self, batch_size: int, gen: torch.Generator, device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(
            0,
            len(self.data) - self.block_size - 1,
            (batch_size,),
            generator=gen,
        )
        x = torch.stack(
            [
                torch.from_numpy(self.data[i : i + self.block_size].astype(np.int64))
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    self.data[i + 1 : i + 1 + self.block_size].astype(np.int64)
                )
                for i in ix
            ]
        )
        if device.startswith("cuda"):
            return (
                x.pin_memory().to(device, non_blocking=True),
                y.pin_memory().to(device, non_blocking=True),
            )
        return x.to(device), y.to(device)


# ----------------------------------------------------------------------
# Shared plumbing
# ----------------------------------------------------------------------


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _sha256_file(path: str, max_bytes: int = 1 << 24) -> str:
    h = hashlib.sha256()
    p = Path(path)
    with open(p, "rb") as fh:
        h.update(fh.read(max_bytes))
    h.update(str(p.stat().st_size).encode())
    return h.hexdigest()


def _sha256_json(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()


def _jsonl(path: Path, event: Dict) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")


def _selection_from_cfg(sel_cfg: Dict) -> Dict[int, List[int]]:
    return {int(k): [int(h) for h in v] for k, v in sel_cfg.items()}


def _amp_ctx(device: str, precision: str = "bf16"):
    """bf16 autocast for training; fp32 for measurement passes.

    The audit defaults to fp32: delta = rec - std is a difference of
    two near-equal tensors, and bf16's ~8 mantissa bits would inject
    relative noise into exactly the quantity whose spectrum is being
    measured. Training keeps bf16 (matches the trusted lanes).
    """
    if device.startswith("cuda") and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _manifest(cfg: Dict, extra: Dict) -> Dict:
    man = {
        "code_commit": _git_commit(),
        "config_hash": _sha256_json(cfg),
        "config": cfg,
        "created_unix": time.time(),
    }
    man.update(extra)
    return man


# ----------------------------------------------------------------------
# train-baseline
# ----------------------------------------------------------------------


def cosine_lr(step: int, cfg: Dict) -> float:
    warmup = cfg["warmup_steps"]
    max_steps = cfg["max_steps"]
    lr = cfg["lr"]
    min_lr = cfg.get("min_lr", lr / 10)
    if step < warmup:
        return lr * (step + 1) / warmup
    frac = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * min(frac, 1.0)))


def make_optimizer(model: nn.Module, cfg: Dict) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.get("weight_decay", 0.1)},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg["lr"],
        betas=tuple(cfg.get("betas", (0.9, 0.95))),
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    step: int,
    batch_counter: int,
    cfg: Dict,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer else None,
            "step": step,
            "batch_counter": batch_counter,
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "train_config": cfg,
            "code_commit": _git_commit(),
        },
        path,
    )


def cmd_train_baseline(cfg: Dict, device: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "train.jsonl"
    torch.manual_seed(cfg["seed"])
    model = build_model(cfg["model"], selection={})
    model.to(device)
    train_data = BinData(cfg["data"]["train_bin"], cfg["model"]["block_size"])
    val_data = BinData(cfg["data"]["val_bin"], cfg["model"]["block_size"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    val_gen_seed = cfg["seed"] + 1000
    optimizer = make_optimizer(model, cfg["train"])
    tcfg = cfg["train"]
    batch_counter = 0
    t0 = time.time()
    tokens_per_step = tcfg["batch_size"] * cfg["model"]["block_size"]
    _jsonl(log, _manifest(cfg, {"event": "start", "device": device}))
    for step in range(tcfg["max_steps"]):
        lr = cosine_lr(step, tcfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = train_data.batch(tcfg["batch_size"], gen, device)
        batch_counter += 1
        with _amp_ctx(device):
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.get("clip", 1.0))
        optimizer.step()
        if step % tcfg.get("log_interval", 50) == 0:
            elapsed = time.time() - t0
            _jsonl(
                log,
                {
                    "event": "train",
                    "step": step,
                    "loss": float(loss.item()),
                    "lr": lr,
                    "elapsed_s": elapsed,
                    "tokens_per_s": tokens_per_step * (step + 1) / elapsed,
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
        if (step > 0 and step % tcfg.get("ckpt_interval", 1000) == 0) or step == tcfg[
            "max_steps"
        ] - 1:
            save_checkpoint(
                out_dir / f"ckpt_step{step}.pt",
                model,
                optimizer,
                step,
                batch_counter,
                cfg,
            )
            save_checkpoint(
                out_dir / "ckpt_latest.pt", model, optimizer, step, batch_counter, cfg
            )
    _jsonl(log, {"event": "done", "elapsed_s": time.time() - t0})


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: BinData,
    n_batches: int,
    batch_size: int,
    device: str,
    seed: int,
    precision: str = "bf16",
) -> float:
    """Mean loss over a FIXED batch set (same seed -> same batches)."""
    was_training = model.training
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    total = 0.0
    for _ in range(n_batches):
        x, y = data.batch(batch_size, gen, device)
        with _amp_ctx(device, precision):
            _, loss = model(x, y)
        total += float(loss.item())
    if was_training:
        model.train()
    return total / n_batches


# ----------------------------------------------------------------------
# audit
# ----------------------------------------------------------------------


def load_model_from_checkpoint(
    ckpt_path: str,
    selection: Dict[int, List[int]],
    gate_mode: str = "none",
    rank: int = 0,
    beta_max: float = 1.0,
    device: str = "cpu",
) -> Tuple[GPT2_KNLP, Dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt["train_config"]["model"]
    model = build_model(model_cfg, selection, gate_mode, rank, beta_max)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    unexpected = [k for k in unexpected if "attn.bias" not in k]
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected}")
    for key in missing:
        if not ("raw_beta" in key or "basis_u" in key or "selected_heads" in key):
            raise RuntimeError(f"missing checkpoint key: {key}")
    model.to(device)
    return model, ckpt


def cmd_audit(cfg: Dict, device: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    selection = _selection_from_cfg(cfg["selection"]["heads"])
    controls = _selection_from_cfg(cfg["selection"].get("control_heads", {}))
    merged: Dict[int, List[int]] = {}
    for src in (selection, controls):
        for layer, heads in src.items():
            merged.setdefault(layer, [])
            merged[layer] = sorted(set(merged[layer]) | set(heads))
    model, ckpt = load_model_from_checkpoint(
        cfg["checkpoint"], merged, gate_mode="none", device=device
    )
    model.eval()
    mods = spectral_modules(model)
    for mod in mods.values():
        mod.audit_capture = True
    d_head = model.config.n_embd // model.config.n_head

    acc: Dict[str, sc.HeadCreditAccumulator] = {}
    qk_acc: Dict[str, sc.QKAsymmetryAccumulator] = {}
    s_ratios: Dict[str, List[float]] = {}
    head_keys: Dict[str, Tuple[int, int, int]] = {}
    for layer, mod in mods.items():
        for slot, head in enumerate(mod.selected_heads.tolist()):
            key = f"L{layer}H{head}"
            acc[key] = sc.HeadCreditAccumulator(d=d_head, retain_raw=True)
            qk_acc[key] = sc.QKAsymmetryAccumulator(d=d_head)
            s_ratios[key] = []
            head_keys[key] = (layer, head, slot)

    data = BinData(cfg["data"]["train_bin"], model.config.block_size)
    gen = torch.Generator().manual_seed(cfg["calibration"]["seed"])
    n_batches = cfg["calibration"]["batches"]
    batch_size = cfg["calibration"]["batch_size"]
    score_batches = cfg["calibration"].get("score_batches", 2)

    audit_precision = cfg["calibration"].get("precision", "fp32")
    for b in range(n_batches):
        for mod in mods.values():
            mod.capture_scores = b < score_batches
        x, y = data.batch(batch_size, gen, device)
        model.zero_grad(set_to_none=True)
        with _amp_ctx(device, audit_precision):
            _, loss = model(x, y)
        loss.backward()
        for layer, mod in mods.items():
            audit = mod._audit
            g_full = audit["g_full"].float()  # [B, nh, T, d]
            delta = audit["delta"].float()  # [B, S, T, d]
            for slot, head in enumerate(mod.selected_heads.tolist()):
                key = f"L{layer}H{head}"
                g = g_full[:, head].reshape(-1, d_head)
                r = delta[:, slot].reshape(-1, d_head)
                acc[key].update(g, r)
                qk_acc[key].update(
                    audit["q_sel"][:, slot].float().reshape(-1, d_head),
                    audit["k_sel"][:, slot].float().reshape(-1, d_head),
                )
                if "scores_premask" in audit:
                    s_ratios[key].append(
                        sc.s_pm_ratio(audit["scores_premask"][:, slot])
                    )
            mod._audit = {}
    model.zero_grad(set_to_none=True)

    trusted_keys = {
        f"L{layer}H{head}" for layer, heads in selection.items() for head in heads
    }
    n_perm = cfg["calibration"].get("n_perm", 100)
    null_rows = cfg["calibration"].get("null_rows", 65536)
    results: Dict[str, Dict] = {}
    bases_by_source: Dict[str, Dict[str, torch.Tensor]] = {}
    lams_by_source: Dict[str, Dict[str, torch.Tensor]] = {}
    block = model.config.block_size
    for key in sorted(acc):
        fin = acc[key].finalize()
        raw = acc[key].raw()
        g_raw, r_raw = raw["G"], raw["R"]
        # Row indices are b*T + t, so whole sequences are contiguous
        # blocks of block_size rows. Subsample at SEQUENCE granularity
        # so the block-permutation null keeps its structure.
        n_seq_total = g_raw.shape[0] // block
        n_seq_keep = max(1, min(n_seq_total, null_rows // block))
        seq_keep = torch.randperm(
            n_seq_total, generator=torch.Generator().manual_seed(0)
        )[:n_seq_keep]
        keep = (seq_keep[:, None] * block + torch.arange(block)[None, :]).reshape(-1)
        g_null, r_null = g_raw[keep], r_raw[keep]
        lam, u = sc.sym_eig_by_abs(fin["H"])
        gr_norm_mean = float(fin["gr_norm_mean"])
        entry = {
            "role": "trusted" if key in trusted_keys else "control",
            "N": int(fin["N"]),
            "signed": sc.signed_spectrum_stats(lam, gr_norm_mean),
            "psd": {
                name: sc.psd_spectrum_stats(fin[name]) for name in ("C_z", "C_r", "C_g")
            },
            # Gate-relevant null: sequence-block permutation (the
            # conservative one; within-sequence autocorrelation
            # survives on both sides). Row-level kept for reference.
            "permutation_null": sc.permutation_null(
                g_null, r_null, n_perm=n_perm, seed=1, rows_per_block=block
            ),
            "permutation_null_rowlevel": sc.permutation_null(
                g_null, r_null, n_perm=n_perm, seed=1
            ),
            "split_half": sc.split_half_overlap(
                g_raw, r_raw, seed=2, rows_per_block=block
            ),
            "split_half_seed2": sc.split_half_overlap(
                g_raw, r_raw, seed=3, rows_per_block=block
            ),
            "split_half_rowlevel": sc.split_half_overlap(g_raw, r_raw, seed=2),
            "s_pm_ratio_mean": (
                float(np.mean(s_ratios[key])) if s_ratios[key] else None
            ),
        }
        qk = qk_acc[key].finalize()
        entry["qk_asym"] = {
            "rho_asym": qk["rho_asym"],
            "singular_values": [float(x) for x in qk["singular_values"]],
        }
        entry["eigenvalues"] = [float(x) for x in lam]
        results[key] = entry
        # candidate bases for the oracle
        for source, mat in (
            ("signed_credit", None),
            ("elementwise_credit_second_moment", fin["C_z"]),
            ("reciprocal_activation", fin["C_r"]),
            ("gradient_covariance", fin["C_g"]),
        ):
            if source == "signed_credit":
                u_s, lam_s = u, lam
            else:
                lam_s, u_s = sc.sym_eig_by_abs(mat)
            bases_by_source.setdefault(source, {})[key] = u_s
            lams_by_source.setdefault(source, {})[key] = lam_s
        bases_by_source.setdefault("qk_asymmetry", {})[key] = qk["U"].to(torch.float64)
        lams_by_source.setdefault("qk_asymmetry", {})[key] = qk["singular_values"].to(
            torch.float64
        )
        bases_by_source.setdefault("random", {})[key] = sc.haar_random_basis(
            d_head, d_head, seed=abs(hash(key)) % (2**31)
        )
        lams_by_source.setdefault("random", {})[key] = torch.zeros(
            d_head, dtype=torch.float64
        )

    gates = evaluate_audit_gates(results, cfg)
    meta_common = {
        "model_commit": _git_commit(),
        "checkpoint": cfg["checkpoint"],
        "checkpoint_sha256": _sha256_file(cfg["checkpoint"]),
        "config_hash": _sha256_json(cfg),
        "dataset": cfg["data"]["train_bin"],
        "dataset_sha256": _sha256_file(cfg["data"]["train_bin"]),
        "calibration_seed": cfg["calibration"]["seed"],
        "calibration_tokens": int(n_batches * batch_size * model.config.block_size),
        "audit_precision": audit_precision,
        "d_head": d_head,
        "creation_command": " ".join(sys.argv),
    }
    for source, u_map in bases_by_source.items():
        sc.save_basis(
            out_dir / f"basis_{source}",
            u_map,
            lams_by_source[source],
            dict(meta_common, basis_source=source),
        )
    (out_dir / "audit_results.json").write_text(
        json.dumps(
            {
                "manifest": meta_common,
                "gates": gates,
                "heads": results,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    print(json.dumps(gates, indent=2))


def evaluate_audit_gates(results: Dict[str, Dict], cfg: Dict) -> Dict:
    """Preregistered survival conditions (PLAN.md Stage 2 hard gate)."""
    trusted = {k: v for k, v in results.items() if v["role"] == "trusted"}
    controls = {k: v for k, v in results.items() if v["role"] == "control"}

    def frac(items, pred):
        items = list(items)
        return sum(1 for x in items if pred(x)) / max(1, len(items))

    beats_null = frac(trusted.values(), lambda v: v["permutation_null"]["exceeds_p95"])
    low_rank = frac(
        trusted.values(), lambda v: v["signed"]["top4_mass_fraction"] >= 0.60
    )
    stable = frac(
        trusted.values(),
        lambda v: max(
            v["split_half"].get("split_half_overlap_r1", 0.0),
            v["split_half"].get("split_half_overlap_r2", 0.0),
            v["split_half"].get("split_half_overlap_r4", 0.0),
        )
        >= 0.60,
    )
    cancel = frac(
        trusted.values(),
        lambda v: v["signed"]["cancellation_ratio"] >= 0.50
        and v["permutation_null"]["exceeds_p95"],
    )
    control_beats_null = frac(
        controls.values(), lambda v: v["permutation_null"]["exceeds_p95"]
    )
    thresholds = cfg.get("gates", {})
    min_frac = thresholds.get("min_trusted_fraction", 0.5)
    gates = {
        "trusted_frac_beats_perm_null_p95": beats_null,
        "trusted_frac_top4_mass_ge_060": low_rank,
        "trusted_frac_split_half_ge_060": stable,
        "trusted_frac_cancellation_ge_050_and_nonnull": cancel,
        "control_frac_beats_perm_null_p95": control_beats_null,
        "pass_existence": beats_null >= min_frac,
        "pass_low_rank": low_rank >= min_frac,
        "pass_stability": stable >= min_frac,
        "pass_cancellation": cancel > 0.0,
        "controls_distinct": control_beats_null < beats_null,
    }
    gates["pass_all"] = all(
        gates[k]
        for k in (
            "pass_existence",
            "pass_low_rank",
            "pass_stability",
            "pass_cancellation",
            "controls_distinct",
        )
    )
    return gates


# ----------------------------------------------------------------------
# oracle
# ----------------------------------------------------------------------


def _load_arm_basis(model: GPT2_KNLP, basis_dir: str, rank: int) -> None:
    bundle = sc.load_basis(Path(basis_dir))
    u_map = bundle["U_by_layer_head"]
    for layer, mod in spectral_modules(model).items():
        if mod.basis_u is None:
            continue
        stacked = []
        for head in mod.selected_heads.tolist():
            key = f"L{layer}H{head}"
            if key not in u_map:
                raise KeyError(f"basis for {key} not in {basis_dir}")
            stacked.append(u_map[key][:, :rank])
        mod.set_basis(torch.stack(stacked))


def cmd_oracle(cfg: Dict, device: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    arm = cfg["arm"]
    gate_mode = arm["gate_mode"]
    rank = arm.get("rank", 0)
    selection = _selection_from_cfg(cfg["selection"]["heads"])
    torch.manual_seed(cfg["seed"])
    model, ckpt = load_model_from_checkpoint(
        cfg["checkpoint"],
        selection,
        gate_mode=gate_mode,
        rank=rank,
        beta_max=arm.get("beta_max", 1.0),
        device=device,
    )
    if gate_mode in ("spectral", "standard_extra_lowrank"):
        _load_arm_basis(model, arm["basis_dir"], rank)
    for p in model.parameters():
        p.requires_grad_(False)
    gates = gate_parameters(model)
    for p in gates:
        p.requires_grad_(True)
    if gate_mode != "none" and not gates:
        raise RuntimeError("no gate parameters found")
    log = out_dir / f"oracle_{cfg['arm']['name']}_seed{cfg['seed']}.jsonl"
    _jsonl(
        log,
        _manifest(
            cfg,
            {
                "event": "start",
                "checkpoint_sha256": _sha256_file(cfg["checkpoint"]),
                "device": device,
                "n_gate_params": sum(p.numel() for p in gates),
            },
        ),
    )
    train_data = BinData(cfg["data"]["train_bin"], model.config.block_size)
    val_data = BinData(cfg["data"]["val_bin"], model.config.block_size)
    ocfg = cfg["oracle"]
    gen = torch.Generator().manual_seed(cfg["seed"] + 7)
    optimizer = torch.optim.Adam(gates, lr=ocfg["lr"]) if gates else None
    val_seed = cfg["seed"] + 9999

    # Oracle-scale effects can be ~1e-3 nats; measure in fp32 by
    # default (config-overridable) so bf16 forward noise never sits
    # inside the effect being compared.
    oracle_precision = ocfg.get("precision", "fp32")

    def val_loss() -> float:
        return evaluate(
            model,
            val_data,
            ocfg.get("eval_batches", 20),
            ocfg["batch_size"],
            device,
            seed=val_seed,
            precision=oracle_precision,
        )

    base_val = val_loss()
    _jsonl(log, {"event": "eval", "step": -1, "val_loss": base_val})
    model.train()
    for step in range(ocfg["steps"]):
        x, y = train_data.batch(ocfg["batch_size"], gen, device)
        with _amp_ctx(device, oracle_precision):
            _, loss = model(x, y)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.cat([p.grad.reshape(-1) for p in gates]).norm())
            optimizer.step()
        else:
            gnorm = 0.0
        if step % ocfg.get("log_interval", 10) == 0:
            _jsonl(
                log,
                {
                    "event": "train",
                    "step": step,
                    "loss": float(loss.item()),
                    "gate_grad_norm": gnorm,
                    "betas": [
                        [float(b) for b in p.detach().cpu().reshape(-1)] for p in gates
                    ],
                },
            )
        if step > 0 and step % ocfg.get("eval_interval", 100) == 0:
            _jsonl(log, {"event": "eval", "step": step, "val_loss": val_loss()})
    final_on = val_loss()
    saved = [p.detach().clone() for p in gates]
    with torch.no_grad():
        for p in gates:
            p.zero_()
    final_off = val_loss()
    with torch.no_grad():
        for p, s in zip(gates, saved):
            p.copy_(s)
    per_mode = []
    if gates and gate_mode in ("spectral", "standard_extra_lowrank"):
        n_modes = gates[0].shape[1]
        for mode_idx in range(n_modes):
            backup = [p.detach().clone() for p in gates]
            with torch.no_grad():
                for p in gates:
                    p[:, mode_idx] = 0.0
            per_mode.append({"mode": mode_idx, "val_loss_without_mode": val_loss()})
            with torch.no_grad():
                for p, s in zip(gates, backup):
                    p.copy_(s)
    _jsonl(
        log,
        {
            "event": "final",
            "val_loss_start": base_val,
            "val_loss_gates_on": final_on,
            "val_loss_gates_off": final_off,
            "per_mode_ablation": per_mode,
            "final_betas": [
                [float(b) for b in p.detach().cpu().reshape(-1)] for p in gates
            ],
        },
    )
    print(
        json.dumps(
            {
                "arm": arm["name"],
                "seed": cfg["seed"],
                "val_start": base_val,
                "val_on": final_on,
                "val_off": final_off,
            },
            indent=2,
        )
    )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train-baseline", "audit", "oracle"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out or cfg.get("out_dir", "out/spectral_ra"))
    if args.mode == "train-baseline":
        cmd_train_baseline(cfg, device, out_dir)
    elif args.mode == "audit":
        cmd_audit(cfg, device, out_dir)
    else:
        cmd_oracle(cfg, device, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
