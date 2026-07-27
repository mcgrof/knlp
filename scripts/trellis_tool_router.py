#!/usr/bin/env python3
"""
Tiny episodic tool router using the existing knlp Trellis memory operator.

The slow model weights learn HOW to use memory across synthetic episodes.

At evaluation:
  * all model parameters are frozen;
  * every episode receives completely new random aliases;
  * alias -> tool bindings are written through the exact running-state
    Trellis recurrence;
  * queries read the two Trellis memories;
  * one binding is overwritten and queried again.

Run from the knlp repository root:

    python scripts/trellis_tool_router.py --device cuda

Clean write-nonlinearity control:

    python scripts/trellis_tool_router.py --device cuda --phi identity

The identity run keeps the inter-pass function fixed as LN-SiLU. It changes
only the memory-write phi, unlike our regrettably educational earlier ablation.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Run from anywhere: put the repo root (parent of scripts/) on the path so the
# in-tree trellis_lm package imports without an editable install, matching the
# sibling trellis_* scripts.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis_lm.activations import identity, ln_silu  # noqa: E402
from trellis_lm.trellis_memory import run_trellis_memory  # noqa: E402


@dataclass(frozen=True)
class RouterConfig:
    alias_dim: int = 32
    d_model: int = 64

    n_heads: int = 1
    d_head: int = 32
    n_slots: int = 32

    n_tools: int = 6
    n_bindings: int = 4

    beta_floor: float = 0.98
    beta_ceiling: float = 0.999
    gamma_init: float = 0.05

    phi: str = "ln_silu"
    exact_inner_backward: bool = False


@dataclass
class FastMemoryState:
    """
    Request-local fast state.

    Shapes:
        key:   [B, H, M, D]
        value: [B, H, M, D]
    """

    key: torch.Tensor
    value: torch.Tensor


@dataclass
class EpisodeBatch:
    aliases: torch.Tensor
    tools: torch.Tensor
    overwrite_index: torch.Tensor
    replacement_tool: torch.Tensor


class TrellisToolRouter(nn.Module):
    """
    Two-memory Trellis router.

    Key memory learns:

        alias/address vector -> internal slot code

    Value memory learns:

        tool vector -> the same internal slot code

    Query:

        alias -> key memory -> code
              -> fixed inter-pass LN-SiLU
              -> value memory -> reconstructed tool vector
              -> tied tool classifier
    """

    def __init__(self, cfg: RouterConfig) -> None:
        super().__init__()

        self.cfg = cfg
        self.H = cfg.n_heads
        self.D = cfg.d_head
        self.M = cfg.n_slots

        self.alias_encoder = nn.Sequential(
            nn.Linear(cfg.alias_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.RMSNorm(cfg.d_model),
        )

        self.key_proj = nn.Linear(
            cfg.d_model,
            cfg.n_heads * cfg.d_head,
            bias=False,
        )
        self.query_proj = nn.Linear(
            cfg.d_model,
            cfg.n_heads * cfg.d_head,
            bias=False,
        )
        self.alpha_proj = nn.Linear(
            cfg.d_model,
            cfg.n_heads * cfg.n_slots,
            bias=False,
        )
        self.beta_proj = nn.Linear(
            cfg.d_model,
            cfg.n_heads,
            bias=True,
        )

        # Tool vectors serve as both value-memory writes and classifier anchors.
        self.tool_embeddings = nn.Embedding(
            cfg.n_tools,
            cfg.d_model,
        )
        self.tool_value_proj = nn.Linear(
            cfg.d_model,
            cfg.n_heads * cfg.d_head,
            bias=False,
        )

        # Positive per-head inner update strength.
        gamma_raw = math.log(math.expm1(cfg.gamma_init))
        self.gamma_raw = nn.Parameter(torch.full((cfg.n_heads,), gamma_raw))

        # Temperature for tied tool-vector classification.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

        if cfg.phi == "ln_silu":
            self.write_phi = ln_silu
        elif cfg.phi == "identity":
            self.write_phi = identity
        else:
            raise ValueError(f"Unsupported phi {cfg.phi!r}; use ln_silu or identity")

        # Keep f fixed across phi ablations.
        self.inter_pass_f = ln_silu

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Queries and keys should begin in approximately the same address space.
        with torch.no_grad():
            self.query_proj.weight.copy_(self.key_proj.weight)

        # Begin with long retention rather than beta ~= 0.5.
        midpoint = (self.cfg.beta_floor + self.cfg.beta_ceiling) / 2.0
        normalized = (midpoint - self.cfg.beta_floor) / (
            self.cfg.beta_ceiling - self.cfg.beta_floor
        )
        beta_bias = math.log(normalized / (1.0 - normalized))

        nn.init.zeros_(self.beta_proj.weight)
        nn.init.constant_(self.beta_proj.bias, beta_bias)

        nn.init.normal_(self.alpha_proj.weight, std=0.02)
        nn.init.normal_(self.tool_embeddings.weight, std=0.02)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> FastMemoryState:
        memory = torch.zeros(
            batch_size,
            self.H,
            self.M,
            self.D,
            device=device,
            dtype=dtype,
        )

        return FastMemoryState(
            key=memory,
            value=memory.clone(),
        )

    def _split_heads(
        self,
        tensor: torch.Tensor,
        width: int,
    ) -> torch.Tensor:
        """
        Convert:

            [B, T, H * width] -> [B, H, T, width]
        """
        batch, tokens, _ = tensor.shape

        return (
            tensor.view(batch, tokens, self.H, width).permute(0, 2, 1, 3).contiguous()
        )

    def _project_aliases(
        self,
        aliases: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        aliases: [B, T, alias_dim]

        Returns:
            keys:    [B, H, T, D]
            queries: [B, H, T, D]
            alpha:   [B, H, T, M]
            beta:    [B, H, T, 1]
        """
        hidden = self.alias_encoder(aliases)

        keys = F.normalize(
            self._split_heads(self.key_proj(hidden), self.D),
            dim=-1,
        )
        queries = F.normalize(
            self._split_heads(self.query_proj(hidden), self.D),
            dim=-1,
        )

        # Paper-style linear target code.
        alpha = self._split_heads(
            self.alpha_proj(hidden),
            self.M,
        )

        beta_logits = self.beta_proj(hidden)
        beta = self.cfg.beta_floor + (
            self.cfg.beta_ceiling - self.cfg.beta_floor
        ) * torch.sigmoid(beta_logits)

        # [B,T,H] -> [B,H,T,1]
        beta = beta.permute(0, 2, 1).unsqueeze(-1).contiguous()

        return keys, queries, alpha, beta

    def _tool_bank(self) -> torch.Tensor:
        """
        Return one normalized value vector per tool.

        Shape:
            [N_tools, H, D]
        """
        projected = self.tool_value_proj(self.tool_embeddings.weight)

        return F.normalize(
            projected.view(
                self.cfg.n_tools,
                self.H,
                self.D,
            ),
            dim=-1,
        )

    def _project_tools(
        self,
        tool_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        tool_ids: [B,T]

        Returns:
            [B,H,T,D]
        """
        bank = self._tool_bank()

        return bank[tool_ids].permute(0, 2, 1, 3).contiguous()

    def effective_gamma(self) -> torch.Tensor:
        return F.softplus(self.gamma_raw)

    def bind(
        self,
        state: FastMemoryState,
        aliases: torch.Tensor,
        tool_ids: torch.Tensor,
    ) -> FastMemoryState:
        """
        Write one or more bindings through the exact sequential Trellis path.

        aliases:
            [B,T,alias_dim]

        tool_ids:
            [B,T]

        Each token's nonlinear correction is computed from the CURRENT live
        memory, not a chunk-start approximation.
        """
        if aliases.ndim != 3:
            raise ValueError(f"aliases must be [B,T,A], got {tuple(aliases.shape)}")
        if tool_ids.shape != aliases.shape[:2]:
            raise ValueError("tool_ids must match aliases' [B,T] dimensions")

        keys, _, alpha, beta = self._project_aliases(aliases)
        values = self._project_tools(tool_ids)
        gamma = self.effective_gamma()

        # The exact operator performs write-before-read. We only need the
        # resulting state during binding, so its read vectors are zero.
        dummy_key_reads = torch.zeros_like(keys)

        dummy_value_reads = torch.zeros(
            aliases.shape[0],
            self.H,
            aliases.shape[1],
            self.M,
            device=aliases.device,
            dtype=aliases.dtype,
        )

        _, key_state = run_trellis_memory(
            write=keys,
            read=dummy_key_reads,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            phi=self.write_phi,
            read_mode="M_q",
            training=self.training,
            exact_inner=self.cfg.exact_inner_backward,
            M_init=state.key,
            return_state=True,
        )

        _, value_state = run_trellis_memory(
            write=values,
            read=dummy_value_reads,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            phi=self.write_phi,
            read_mode="M_T_r",
            training=self.training,
            exact_inner=self.cfg.exact_inner_backward,
            M_init=state.value,
            return_state=True,
        )

        return FastMemoryState(
            key=key_state,
            value=value_state,
        )

    def query(
        self,
        state: FastMemoryState,
        aliases: torch.Tensor,
    ) -> torch.Tensor:
        """
        Read without modifying memory.

        aliases:
            [B,Q,alias_dim]

        Returns:
            tool logits [B,Q,N_tools]
        """
        _, queries, _, _ = self._project_aliases(aliases)

        # Key-memory read:
        # [B,H,M,D] x [B,H,Q,D] -> [B,H,Q,M]
        key_code = torch.einsum(
            "bhmd,bhqd->bhqm",
            state.key,
            queries,
        )

        # Keep this fixed when comparing write phi variants.
        read_code = self.inter_pass_f(key_code)

        # Value-memory read:
        # [B,H,M,D]^T x [B,H,Q,M] -> [B,H,Q,D]
        retrieved = torch.einsum(
            "bhmd,bhqm->bhqd",
            state.value,
            read_code,
        )
        retrieved = F.normalize(retrieved, dim=-1)

        tool_bank = self._tool_bank()

        # Tied classifier: compare reconstructed vectors to tool write vectors.
        logits = torch.einsum(
            "bhqd,nhd->bqn",
            retrieved,
            tool_bank,
        )
        logits = logits / self.H

        scale = self.logit_scale.exp().clamp(
            min=1.0,
            max=100.0,
        )

        return logits * scale


def make_episode_batch(
    cfg: RouterConfig,
    *,
    batch_size: int,
    device: torch.device,
) -> EpisodeBatch:
    """
    Create completely new continuous aliases for every episode.

    The model cannot memorize alias -> tool mappings in its slow parameters,
    because the alias vectors and assignments are regenerated every batch.
    """
    aliases = torch.randn(
        batch_size,
        cfg.n_bindings,
        cfg.alias_dim,
        device=device,
    )
    aliases = F.normalize(aliases, dim=-1)

    tools = torch.randint(
        low=0,
        high=cfg.n_tools,
        size=(batch_size, cfg.n_bindings),
        device=device,
    )

    overwrite_index = torch.randint(
        low=0,
        high=cfg.n_bindings,
        size=(batch_size,),
        device=device,
    )

    rows = torch.arange(batch_size, device=device)
    old_tool = tools[rows, overwrite_index]

    # Draw uniformly from all tools except the current one.
    replacement_tool = torch.randint(
        low=0,
        high=cfg.n_tools - 1,
        size=(batch_size,),
        device=device,
    )
    replacement_tool = replacement_tool + (replacement_tool >= old_tool).long()

    return EpisodeBatch(
        aliases=aliases,
        tools=tools,
        overwrite_index=overwrite_index,
        replacement_tool=replacement_tool,
    )


def episode_forward(
    model: TrellisToolRouter,
    episode: EpisodeBatch,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Returns:
        logits_before: [B,K,N]
        logits_after:  [B,K,N]
        targets_after: [B,K]
    """
    batch_size = episode.aliases.shape[0]
    device = episode.aliases.device

    state = model.initial_state(
        batch_size,
        device=device,
        dtype=episode.aliases.dtype,
    )

    # Learn all new bindings through fast memory.
    state = model.bind(
        state,
        episode.aliases,
        episode.tools,
    )

    logits_before = model.query(
        state,
        episode.aliases,
    )

    # Overwrite one binding per episode.
    rows = torch.arange(batch_size, device=device)

    overwritten_alias = episode.aliases[
        rows,
        episode.overwrite_index,
    ].unsqueeze(1)

    replacement_tool = episode.replacement_tool.unsqueeze(1)

    state = model.bind(
        state,
        overwritten_alias,
        replacement_tool,
    )

    targets_after = episode.tools.clone()
    targets_after[
        rows,
        episode.overwrite_index,
    ] = episode.replacement_tool

    logits_after = model.query(
        state,
        episode.aliases,
    )

    return logits_before, logits_after, targets_after


@torch.no_grad()
def score_episode(
    episode: EpisodeBatch,
    logits_before: torch.Tensor,
    logits_after: torch.Tensor,
    targets_after: torch.Tensor,
) -> dict[str, float]:
    rows = torch.arange(
        episode.aliases.shape[0],
        device=episode.aliases.device,
    )

    predictions_before = logits_before.argmax(dim=-1)
    predictions_after = logits_after.argmax(dim=-1)

    overwrite_correct = (
        predictions_after[
            rows,
            episode.overwrite_index,
        ]
        == episode.replacement_tool
    )

    collateral_mask = torch.ones_like(
        episode.tools,
        dtype=torch.bool,
    )
    collateral_mask[
        rows,
        episode.overwrite_index,
    ] = False

    return {
        "before": (predictions_before == episode.tools).float().mean().item(),
        "after": (predictions_after == targets_after).float().mean().item(),
        "overwrite": overwrite_correct.float().mean().item(),
        "collateral": (
            predictions_after[collateral_mask] == targets_after[collateral_mask]
        )
        .float()
        .mean()
        .item(),
    }


@torch.no_grad()
def evaluate(
    model: TrellisToolRouter,
    cfg: RouterConfig,
    *,
    device: torch.device,
    batch_size: int,
    batches: int = 20,
) -> dict[str, float]:
    model.eval()

    totals = {
        "before": 0.0,
        "after": 0.0,
        "overwrite": 0.0,
        "collateral": 0.0,
    }

    for _ in range(batches):
        episode = make_episode_batch(
            cfg,
            batch_size=batch_size,
            device=device,
        )

        outputs = episode_forward(model, episode)
        metrics = score_episode(
            episode,
            *outputs,
        )

        for name, value in metrics.items():
            totals[name] += value

    return {name: value / batches for name, value in totals.items()}


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    cfg = RouterConfig(
        alias_dim=args.alias_dim,
        d_model=args.d_model,
        n_heads=args.heads,
        d_head=args.head_dim,
        n_slots=args.slots,
        n_tools=args.tools,
        n_bindings=args.bindings,
        gamma_init=args.gamma_init,
        phi=args.phi,
        exact_inner_backward=args.exact_inner_backward,
    )

    model = TrellisToolRouter(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    chance = 1.0 / cfg.n_tools

    print(f"device:                {device}")
    print(f"write phi:             {cfg.phi}")
    print(f"inter-pass f:          ln_silu")
    print(f"tools / chance:        {cfg.n_tools} / {chance:.3f}")
    print(f"bindings per episode:  {cfg.n_bindings}")
    print(f"exact inner backward:  {cfg.exact_inner_backward}")
    print()

    for step in range(1, args.steps + 1):
        model.train()

        episode = make_episode_batch(
            cfg,
            batch_size=args.batch_size,
            device=device,
        )

        (
            logits_before,
            logits_after,
            targets_after,
        ) = episode_forward(model, episode)

        loss_before = F.cross_entropy(
            logits_before.flatten(0, 1),
            episode.tools.flatten(),
        )
        loss_after = F.cross_entropy(
            logits_after.flatten(0, 1),
            targets_after.flatten(),
        )

        # Overwriting is the more interesting behavior, so weight it slightly.
        loss = loss_before + args.overwrite_weight * loss_after

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.grad_clip,
        )

        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"nonfinite gradient at step {step}: {grad_norm}")

        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            metrics = score_episode(
                episode,
                logits_before.detach(),
                logits_after.detach(),
                targets_after,
            )

            gamma = model.effective_gamma().detach().cpu().tolist()

            print(
                f"step={step:5d} "
                f"loss={loss.item():.4f} "
                f"before={metrics['before']:.3f} "
                f"after={metrics['after']:.3f} "
                f"overwrite={metrics['overwrite']:.3f} "
                f"collateral={metrics['collateral']:.3f} "
                f"gamma={gamma}"
            )

    metrics = evaluate(
        model,
        cfg,
        device=device,
        batch_size=args.batch_size,
        batches=args.eval_batches,
    )

    print("\nHeld-out episodes, model weights frozen:")
    for name, value in metrics.items():
        print(f"  {name:12s}: {value:.4f}")

    checkpoint = {
        "config": cfg.__dict__,
        "model": model.state_dict(),
        "metrics": metrics,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)

    print(f"\nSaved: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=128)

    parser.add_argument("--tools", type=int, default=6)
    parser.add_argument("--bindings", type=int, default=4)
    parser.add_argument("--alias-dim", type=int, default=32)

    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--slots", type=int, default=32)

    parser.add_argument(
        "--phi",
        choices=("ln_silu", "identity"),
        default="ln_silu",
    )
    parser.add_argument("--gamma-init", type=float, default=0.05)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--overwrite-weight", type=float, default=1.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # This changes outer training gradients, not the forward-time memory rule.
    parser.add_argument(
        "--exact-inner-backward",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument(
        "--output",
        default="results/trellis_tool_router.pt",
    )

    args = parser.parse_args()

    if args.tools < 2:
        parser.error("--tools must be at least 2")
    if args.bindings < 1:
        parser.error("--bindings must be positive")
    if args.gamma_init <= 0:
        parser.error("--gamma-init must be positive")

    return args


def main() -> None:
    args = parse_args()

    random.seed(0)
    torch.manual_seed(0)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    torch.set_float32_matmul_precision("high")

    train(args)


if __name__ == "__main__":
    main()
