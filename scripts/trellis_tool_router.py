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

Outer gradient semantics (SEQUENTIAL path only): --outer-gradient-mode
selects how the outer loss differentiates through the inner VJP.
"full_bilevel" (default) keeps the inner VJP in the graph -- exact, the
correctness default. "first_order_detached" cuts z before the inner VJP --
the historical fast mode. Forward is identical either way; only the
backward differs (--assert-step0-equivalence verifies both claims).

Two memory regimes, both driven by the existing --bindings/--slots flags:

  * surplus (bindings <= slots, the default): every binding can own a
    slot, so this is the correctness probe -- errors are semantics bugs,
    not capacity limits;
  * oversubscribed (bindings > slots, e.g. --bindings 64 --slots 16):
    the capacity probe -- the memory must compress under pressure, where
    the write nonlinearity could earn its keep.

Paired arms: episodes are drawn from a dedicated generator seeded from
--seed, decoupled from model-init RNG, so two arms with the same seed but
different phi or gradient mode see identical episode streams. --jsonl
appends per-step training curves and a final eval record for offline
comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Run from anywhere: put the repo root (parent of scripts/) on the path so the
# in-tree trellis_lm package imports without an editable install, matching the
# sibling trellis_* scripts.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis_lm.activations import get_activation, ln_silu  # noqa: E402
from trellis_lm.trellis_memory import run_trellis_memory  # noqa: E402

OUTER_GRADIENT_MODES = ("full_bilevel", "first_order_detached")
PHI_CHOICES = ("ln_silu", "silu", "identity")


@dataclass(frozen=True)
class RouterConfig:
    alias_dim: int = 32
    d_model: int = 64

    n_heads: int = 1
    d_head: int = 32
    n_slots: int = 32

    n_tools: int = 6
    n_bindings: int = 4
    n_overwrites: int = 1

    beta_floor: float = 0.98
    beta_ceiling: float = 0.999
    gamma_init: float = 0.05

    phi: str = "ln_silu"
    # Outer training gradient through the inner VJP: "full_bilevel" keeps it
    # in the graph (exact_inner=True), "first_order_detached" cuts it. The
    # forward-time memory rule is unaffected.
    outer_gradient_mode: str = "full_bilevel"
    # Deprecated: the pre-split boolean spelling of outer_gradient_mode
    # (True == full_bilevel). Kept only so old checkpoint configs still
    # load; the mode string is authoritative everywhere.
    exact_inner_backward: bool = False
    # tie_f_to_phi reproduces the earlier confounded ablation: a single knob
    # drove BOTH the memory-write phi and the inter-pass map f, so choosing
    # phi=identity silently linearized f too. The faithful control (default,
    # False) holds f fixed at LN-SiLU so the write-nonlinearity ablation changes
    # ONLY the write. Set True to measure how much the tie distorts the result.
    tie_f_to_phi: bool = False


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
    """
    Shapes:
        aliases:          [B, K, alias_dim]
        tools:            [B, K]
        overwrite_index:  [B, N_ow]  applied in column order; last write wins
        replacement_tool: [B, N_ow]
    """

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

        if cfg.phi not in PHI_CHOICES:
            raise ValueError(f"Unsupported phi {cfg.phi!r}; use one of {PHI_CHOICES}")
        self.write_phi = get_activation(cfg.phi)

        if cfg.outer_gradient_mode not in OUTER_GRADIENT_MODES:
            raise ValueError(
                f"Unsupported outer_gradient_mode {cfg.outer_gradient_mode!r}; "
                f"use one of {OUTER_GRADIENT_MODES}"
            )

        # Faithful control (default): f is held at LN-SiLU while phi varies, so
        # the write-nonlinearity ablation isolates the write. tie_f_to_phi
        # reproduces the earlier confound by driving f from the same knob as phi.
        self.inter_pass_f = self.write_phi if cfg.tie_f_to_phi else ln_silu

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
        exact_inner = self.cfg.outer_gradient_mode == "full_bilevel"

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
            exact_inner=exact_inner,
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
            exact_inner=exact_inner,
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
    generator: torch.Generator | None = None,
) -> EpisodeBatch:
    """
    Create completely new continuous aliases for every episode.

    The model cannot memorize alias -> tool mappings in its slow parameters,
    because the alias vectors and assignments are regenerated every batch.

    All draws happen on CPU through `generator` (decoupled from model-init
    RNG) and move to `device` afterwards, so arms with the same seed see
    identical episodes on any device.
    """
    aliases = torch.randn(
        batch_size,
        cfg.n_bindings,
        cfg.alias_dim,
        generator=generator,
    )
    aliases = F.normalize(aliases, dim=-1)

    tools = torch.randint(
        low=0,
        high=cfg.n_tools,
        size=(batch_size, cfg.n_bindings),
        generator=generator,
    )

    # Sequential overwrites: each picks a binding index and a replacement
    # tool different from that binding's CURRENT tool at that point. Later
    # overwrites may hit the same index -- last write wins.
    rows = torch.arange(batch_size)
    current = tools.clone()

    overwrite_index = torch.empty(batch_size, cfg.n_overwrites, dtype=torch.long)
    replacement_tool = torch.empty_like(overwrite_index)

    for j in range(cfg.n_overwrites):
        index = torch.randint(
            low=0,
            high=cfg.n_bindings,
            size=(batch_size,),
            generator=generator,
        )
        old_tool = current[rows, index]

        # Draw uniformly from all tools except the current one.
        replacement = torch.randint(
            low=0,
            high=cfg.n_tools - 1,
            size=(batch_size,),
            generator=generator,
        )
        replacement = replacement + (replacement >= old_tool).long()

        current[rows, index] = replacement
        overwrite_index[:, j] = index
        replacement_tool[:, j] = replacement

    return EpisodeBatch(
        aliases=aliases.to(device),
        tools=tools.to(device),
        overwrite_index=overwrite_index.to(device),
        replacement_tool=replacement_tool.to(device),
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

    # Apply overwrites sequentially: each is a single-token bind against the
    # live intermediate state, mirroring how they were sampled.
    rows = torch.arange(batch_size, device=device)
    targets_after = episode.tools.clone()

    for j in range(episode.overwrite_index.shape[1]):
        index = episode.overwrite_index[:, j]
        replacement = episode.replacement_tool[:, j]

        overwritten_alias = episode.aliases[rows, index].unsqueeze(1)

        state = model.bind(
            state,
            overwritten_alias,
            replacement.unsqueeze(1),
        )

        # In-order assignment: a later overwrite of the same index wins.
        targets_after[rows, index] = replacement

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

    # Mark every index touched by any overwrite; duplicates collapse to one
    # mask entry, and targets_after already carries the LAST written value.
    overwritten = torch.zeros_like(episode.tools, dtype=torch.bool)
    overwritten[rows.unsqueeze(1), episode.overwrite_index] = True

    after_correct = predictions_after == targets_after
    untouched = ~overwritten

    return {
        "before": (predictions_before == episode.tools).float().mean().item(),
        "after": after_correct.float().mean().item(),
        "overwrite": after_correct[overwritten].float().mean().item(),
        "collateral": (
            after_correct[untouched].float().mean().item()
            if untouched.any()
            else float("nan")
        ),
    }


@torch.no_grad()
def evaluate(
    model: TrellisToolRouter,
    cfg: RouterConfig,
    *,
    device: torch.device,
    batch_size: int,
    batches: int = 20,
    generator: torch.Generator | None = None,
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
            generator=generator,
        )

        outputs = episode_forward(model, episode)
        metrics = score_episode(
            episode,
            *outputs,
        )

        for name, value in metrics.items():
            totals[name] += value

    return {name: value / batches for name, value in totals.items()}


def check_step0_equivalence(
    model: TrellisToolRouter,
    cfg: RouterConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[bool, bool]:
    """
    Cheap pre-training probe of the outer-gradient toggle.

    Forward must be mode-independent: a weight-tied twin built with the
    OPPOSITE outer_gradient_mode must produce allclose outputs on one
    identical episode batch. Backward must not be: after one loss+backward
    on each, at least one parameter grad must differ, or the toggle is dead.

    Returns (forward_ok, grads_differ).
    """
    other = (
        "first_order_detached"
        if cfg.outer_gradient_mode == "full_bilevel"
        else "full_bilevel"
    )
    # RouterConfig is frozen: build a second config for the twin.
    cfg_other = replace(
        cfg,
        outer_gradient_mode=other,
        exact_inner_backward=(other == "full_bilevel"),
    )
    twin = TrellisToolRouter(cfg_other).to(device)
    twin.load_state_dict(model.state_dict())

    # Dedicated generator: the probe must not consume the training episode
    # stream, or enabling the flag would break cross-arm pairing.
    probe_generator = torch.Generator().manual_seed(args.seed)
    episode = make_episode_batch(
        cfg,
        batch_size=min(args.batch_size, 8),
        device=device,
        generator=probe_generator,
    )

    model.train()
    twin.train()

    outputs = {}
    for name, net in (("main", model), ("twin", twin)):
        logits_before, logits_after, targets_after = episode_forward(net, episode)

        loss = F.cross_entropy(
            logits_before.flatten(0, 1),
            episode.tools.flatten(),
        ) + args.overwrite_weight * F.cross_entropy(
            logits_after.flatten(0, 1),
            targets_after.flatten(),
        )

        net.zero_grad(set_to_none=True)
        loss.backward()

        outputs[name] = (logits_before.detach(), logits_after.detach())

    forward_ok = torch.allclose(
        outputs["main"][0], outputs["twin"][0], rtol=1e-5, atol=1e-5
    ) and torch.allclose(outputs["main"][1], outputs["twin"][1], rtol=1e-5, atol=1e-5)

    grads_differ = False
    for p_main, p_twin in zip(model.parameters(), twin.parameters()):
        g_main = p_main.grad if p_main.grad is not None else torch.zeros_like(p_main)
        g_twin = p_twin.grad if p_twin.grad is not None else torch.zeros_like(p_twin)
        if (g_main - g_twin).abs().max().item() > 0:
            grads_differ = True
            break

    model.zero_grad(set_to_none=True)

    return forward_ok, grads_differ


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
        n_overwrites=args.overwrites,
        gamma_init=args.gamma_init,
        phi=args.phi,
        outer_gradient_mode=args.outer_gradient_mode,
        # Deprecated mirror of the mode, kept coherent for old readers.
        exact_inner_backward=(args.outer_gradient_mode == "full_bilevel"),
        tie_f_to_phi=args.tie_f_phi,
    )

    model = TrellisToolRouter(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    chance = 1.0 / cfg.n_tools
    f_name = cfg.phi if cfg.tie_f_to_phi else "ln_silu"

    # Surplus (bindings <= slots) probes correctness; oversubscribed probes
    # compression under capacity pressure.
    regime = (
        "oversubscribed (capacity probe)"
        if cfg.n_bindings > cfg.n_slots
        else "surplus (correctness probe)"
    )

    print(f"device:                {device}")
    print(f"seed:                  {args.seed}")
    print(f"write phi:             {cfg.phi}")
    print(
        f"inter-pass f:          {f_name}"
        f"{'  (TIED to phi -- confound)' if cfg.tie_f_to_phi else '  (fixed)'}"
    )
    print(f"tools / chance:        {cfg.n_tools} / {chance:.3f}")
    print(f"bindings per episode:  {cfg.n_bindings}")
    print(f"overwrites per episode:{cfg.n_overwrites:2d}")
    print(f"slots:                 {cfg.n_slots}  [{regime}]")
    print(f"outer gradient mode:   {cfg.outer_gradient_mode}")
    print()

    if args.assert_step0_equivalence:
        forward_ok, grads_differ = check_step0_equivalence(model, cfg, args, device)
        passed = forward_ok and grads_differ
        print(
            f"step0 equivalence: forward_allclose={forward_ok} "
            f"grads_differ={grads_differ} -> {'PASS' if passed else 'FAIL'}"
        )
        if not passed:
            sys.exit(1)

    # Episode stream generator: dedicated CPU generator seeded AFTER model
    # construction, so arms sharing a seed get identical episodes no matter
    # how model init consumed the global RNG.
    episode_generator = torch.Generator().manual_seed(args.seed)

    jsonl_file = None
    if args.jsonl:
        jsonl_path = Path(args.jsonl)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        # Line-buffered append so curves survive interruption.
        jsonl_file = open(jsonl_path, "a", buffering=1)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def peak_mem_mb() -> float:
        if device.type == "cuda":
            return torch.cuda.max_memory_allocated(device) / 2**20
        return 0.0

    start_time = time.monotonic()

    for step in range(1, args.steps + 1):
        model.train()

        episode = make_episode_batch(
            cfg,
            batch_size=args.batch_size,
            device=device,
            generator=episode_generator,
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

            if jsonl_file is not None:
                record = {
                    "kind": "train",
                    "step": step,
                    "loss": loss.item(),
                    "before": metrics["before"],
                    "after": metrics["after"],
                    "overwrite": metrics["overwrite"],
                    "collateral": metrics["collateral"],
                    "gamma": gamma,
                    "wall_s": time.monotonic() - start_time,
                    "peak_mem_mb": peak_mem_mb(),
                }
                jsonl_file.write(json.dumps(record) + "\n")

    metrics = evaluate(
        model,
        cfg,
        device=device,
        batch_size=args.batch_size,
        batches=args.eval_batches,
        generator=episode_generator,
    )

    print("\nHeld-out episodes, model weights frozen:")
    for name, value in metrics.items():
        print(f"  {name:12s}: {value:.4f}")

    if jsonl_file is not None:
        record = {
            "kind": "eval",
            **metrics,
            "wall_s": time.monotonic() - start_time,
            "peak_mem_mb": peak_mem_mb(),
            "config": {
                "outer_gradient_mode": cfg.outer_gradient_mode,
                "phi": cfg.phi,
                "f": f_name,
                "tie_f_phi": cfg.tie_f_to_phi,
                "seed": args.seed,
                "tools": cfg.n_tools,
                "bindings": cfg.n_bindings,
                "overwrites": cfg.n_overwrites,
                "slots": cfg.n_slots,
                "alias_dim": cfg.alias_dim,
                "d_model": cfg.d_model,
                "heads": cfg.n_heads,
                "head_dim": cfg.d_head,
                "gamma_init": cfg.gamma_init,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "eval_batches": args.eval_batches,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "overwrite_weight": args.overwrite_weight,
                "device": str(device),
            },
        }
        jsonl_file.write(json.dumps(record) + "\n")
        jsonl_file.close()

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

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--tools", type=int, default=6)
    parser.add_argument("--bindings", type=int, default=4)
    parser.add_argument("--overwrites", type=int, default=1)
    parser.add_argument("--alias-dim", type=int, default=32)

    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--slots", type=int, default=32)

    parser.add_argument(
        "--phi",
        choices=PHI_CHOICES,
        default="ln_silu",
    )
    parser.add_argument("--gamma-init", type=float, default=0.05)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--overwrite-weight", type=float, default=1.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # This changes outer training gradients, not the forward-time memory rule.
    parser.add_argument(
        "--outer-gradient-mode",
        choices=OUTER_GRADIENT_MODES,
        default=None,
        help="outer gradient through the inner VJP; default full_bilevel",
    )
    # Deprecated boolean spelling of --outer-gradient-mode; only takes
    # effect when --outer-gradient-mode is not given.
    parser.add_argument(
        "--exact-inner-backward",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="deprecated: use --outer-gradient-mode",
    )
    # Reproduce the earlier confounded ablation (f driven by the same knob as
    # phi). Default off = the faithful control (f fixed at LN-SiLU).
    parser.add_argument(
        "--tie-f-phi",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument(
        "--output",
        default="results/trellis_tool_router.pt",
    )
    parser.add_argument(
        "--jsonl",
        default=None,
        help="append per-step train records and a final eval record here",
    )
    parser.add_argument(
        "--assert-step0-equivalence",
        action="store_true",
        help="pre-training check: forward mode-independent, backward not",
    )

    args = parser.parse_args()

    if args.tools < 2:
        parser.error("--tools must be at least 2")
    if args.bindings < 1:
        parser.error("--bindings must be positive")
    if args.overwrites < 1:
        parser.error("--overwrites must be positive")
    if args.gamma_init <= 0:
        parser.error("--gamma-init must be positive")

    # Resolve the outer gradient mode. The mode flag wins; the deprecated
    # boolean maps True -> full_bilevel, False -> first_order_detached.
    if args.outer_gradient_mode is None:
        if args.exact_inner_backward is not None:
            args.outer_gradient_mode = (
                "full_bilevel" if args.exact_inner_backward else "first_order_detached"
            )
            print(
                "note: --[no-]exact-inner-backward is deprecated; use "
                f"--outer-gradient-mode {args.outer_gradient_mode}"
            )
        else:
            args.outer_gradient_mode = "full_bilevel"

    return args


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.set_float32_matmul_precision("high")

    train(args)


if __name__ == "__main__":
    main()
