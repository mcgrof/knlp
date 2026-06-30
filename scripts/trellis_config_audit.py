#!/usr/bin/env python3
"""Print a compact Trellis-vs-GatedDeltaNet config/resource audit."""

from __future__ import annotations

import argparse
import json
from typing import Any

from trellis_lm.config import TrellisConfig
from trellis_lm.model import build_model


def model_params(cfg: TrellisConfig, kind: str) -> int | str:
    try:
        return int(build_model(cfg, kind).get_num_params())
    except Exception as exc:
        return f"unavailable: {exc!r}"


def state_bytes(cfg: TrellisConfig, kind: str, batch_size: int) -> int | str:
    try:
        return int(build_model(cfg, kind).memory_state_bytes(batch_size))
    except Exception as exc:
        return f"unavailable: {exc!r}"


def cfg_summary(cfg: TrellisConfig, kind: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "kind": kind,
        "params": model_params(cfg, kind),
        "state_bytes_per_sequence": state_bytes(cfg, kind, 1),
        "d_model": cfg.d_model,
        "layers": cfg.n_layers,
        "heads": cfg.n_heads,
        "head_dim": cfg.d_head,
        "M_or_state_slots": cfg.n_slots if kind == "trellis" else None,
        "expand_v": args.gdn_expand_v if "gated_delta" in kind else None,
        "short_conv_qk": cfg.use_short_conv_qk,
        "short_conv_v": cfg.use_short_conv_v,
        "conv_kernel": cfg.conv_kernel,
        "output_path": cfg.output_path if kind == "trellis" else None,
        "value_readout_act": cfg.value_readout_act if kind == "trellis" else None,
        "activation": cfg.activation if kind == "trellis" else None,
        "alpha_mode": cfg.alpha_mode if kind == "trellis" else None,
        "beta_init": cfg.beta_init if kind == "trellis" else None,
        "gamma_init": cfg.gamma_init if kind == "trellis" else None,
        "trellis_update_stabilizer": (
            cfg.trellis_update_stabilizer if kind == "trellis" else None
        ),
        "trellis_innovation_rms_cap": (
            cfg.trellis_innovation_rms_cap if kind == "trellis" else None
        ),
        "trellis_delta_ratio_cap": (
            cfg.trellis_delta_ratio_cap if kind == "trellis" else None
        ),
        "trellis_state_rms_floor": (
            cfg.trellis_state_rms_floor if kind == "trellis" else None
        ),
        "trellis_layer0_gamma_mult": (
            cfg.trellis_layer0_gamma_mult if kind == "trellis" else None
        ),
        "backend": args.trellis_backend if kind == "trellis" else args.gdn_backend,
        "training_tokens": args.train_tokens,
        "lr": args.lr,
        "batch": args.batch,
        "seq_len": args.seq_len,
    }


def build_cfg(args: argparse.Namespace) -> TrellisConfig:
    return TrellisConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_head=args.d_head,
        n_slots=args.n_slots,
        max_seq_len=args.seq_len,
        chunk_size=args.chunk_size,
        exact_inner=args.chunk_size <= 1,
        activation=args.activation,
        alpha_mode=args.alpha_mode,
        beta_init=args.beta_init,
        gamma_init=args.gamma_init,
        output_path=args.output_path,
        use_short_conv_v=args.use_short_conv_v,
        trellis_update_stabilizer=args.trellis_update_stabilizer,
        trellis_innovation_rms_cap=args.trellis_innovation_rms_cap,
        trellis_delta_ratio_cap=args.trellis_delta_ratio_cap,
        trellis_state_rms_floor=args.trellis_state_rms_floor,
        trellis_layer0_gamma_mult=args.trellis_layer0_gamma_mult,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, default=50257)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=10)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--d-head", type=int, default=64)
    p.add_argument("--n-slots", type=int, default=48)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--chunk-size", type=int, default=16)
    p.add_argument("--train-tokens", type=int, default=6_000_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--activation", default="silu")
    p.add_argument("--alpha-mode", default="linear")
    p.add_argument("--beta-init", type=float, default=0.99)
    p.add_argument("--gamma-init", type=float, default=0.005)
    p.add_argument("--output-path", default="paper")
    p.add_argument(
        "--use-short-conv-v",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--trellis-update-stabilizer", default="none")
    p.add_argument("--trellis-innovation-rms-cap", type=float, default=0.0)
    p.add_argument("--trellis-delta-ratio-cap", type=float, default=0.0)
    p.add_argument("--trellis-state-rms-floor", type=float, default=1e-3)
    p.add_argument("--trellis-layer0-gamma-mult", type=float, default=1.0)
    p.add_argument("--trellis-backend", default="triton_state_evolution")
    p.add_argument("--gdn-kind", default="gated_delta_ref")
    p.add_argument("--gdn-expand-v", type=float, default=1.0)
    p.add_argument("--gdn-backend", default="fla.layers.GatedDeltaNet")
    args = p.parse_args()

    cfg = build_cfg(args)
    rows = {
        "trellis": cfg_summary(cfg, "trellis", args),
        "gdn": cfg_summary(cfg, args.gdn_kind, args),
    }
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
