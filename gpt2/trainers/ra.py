"""
RA Trainer

Trainer for Reciprocal Attention with compute routing.
Supports two modes: Baseline (standard GPT-2) vs RA (router enabled).

The inductive bias: route compute based on |x - E(x)| to use cheap attention
for easy tokens and full attention for hard tokens.
"""

import os
import sys
import math
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.nn import functional as F

# Add parent to path for imports
parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, parent_dir)

from gpt2.model import GPT, GPTConfig, Block, LayerNorm, MLP, CausalSelfAttention
from ra import (
    RAConfig,
    RAAttention,
    ContextShiftGate,
    ContextRouter,
    RoutedMixer,
    WarmupScheduler,
)
from lib.optimizers import create_optimizer
from .base import BaseGPT2Trainer


class RATransformerBlock(nn.Module):
    """
    Transformer block with RA attention and routing.

    In phase 1 (warmup): Uses all heads combined, no routing overhead.
    In phase 2 (routing): Routes between FULL and RA head groups.
    """

    def __init__(self, gpt_config, ra_config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.ra_config = ra_config

        # Pre-norm
        self.ln_1 = LayerNorm(gpt_config.n_embd, bias=gpt_config.bias)
        self.ln_2 = LayerNorm(gpt_config.n_embd, bias=gpt_config.bias)

        # RA attention (no internal LN - we handle it)
        self.attn = RAAttention(ra_config)

        # MLP
        self.mlp = MLP(gpt_config)

        # Routing components
        self.shift_gate = ContextShiftGate()
        self.router = ContextRouter(ra_config)
        self.mixer = RoutedMixer()

        # Phase tracking
        self.phase1 = True

        # Router stats for logging
        self.last_router_probs = None

    def forward(self, x, tok_emb=None):
        """
        Args:
            x: [B, T, D] hidden state
            tok_emb: [B, T, D] token embeddings for routing

        Returns:
            out: [B, T, D] output hidden state
        """
        # Pre-norm for attention
        x_norm = self.ln_1(x)

        if self.phase1:
            # Phase 1: Use all heads combined, no routing
            # Get both outputs but combine them simply
            out_full, out_ra, _ = self.attn(x_norm)
            # Weight by head count ratio
            alpha = self.attn.n_ra / self.attn.n_heads
            attn_out = (1 - alpha) * out_full + alpha * out_ra
            self.last_router_probs = None
        else:
            # Phase 2: Route based on contextual hardness
            if tok_emb is None:
                raise ValueError("tok_emb required for phase 2 routing")

            # Get both head group outputs
            out_full, out_ra, _ = self.attn(x_norm)

            # Compute routing probabilities
            shift = self.shift_gate(x_norm, tok_emb)
            probs = self.router(x_norm, tok_emb, shift)

            # Mix according to router
            attn_out = self.mixer(x_norm, out_ra, out_full, probs)

            # Store for logging
            self.last_router_probs = {
                k: v.detach().mean().item() for k, v in probs.items()
            }

        # Residual + MLP
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))

        return x


class RAGPT(nn.Module):
    """GPT with RA attention blocks."""

    def __init__(self, gpt_config, ra_config):
        super().__init__()
        self.config = gpt_config
        self.ra_config = ra_config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(gpt_config.vocab_size, gpt_config.n_embd),
                wpe=nn.Embedding(gpt_config.block_size, gpt_config.n_embd),
                drop=nn.Dropout(gpt_config.dropout),
                h=nn.ModuleList(
                    [
                        RATransformerBlock(gpt_config, ra_config, i)
                        for i in range(gpt_config.n_layer)
                    ]
                ),
                ln_f=LayerNorm(gpt_config.n_embd, bias=gpt_config.bias),
            )
        )
        # Fixed: correct dimensions
        self.lm_head = nn.Linear(gpt_config.n_embd, gpt_config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Init weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("c_proj_full.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * gpt_config.n_layer)
                )

        # Report parameters
        n_params = sum(p.numel() for p in self.parameters())
        n_params -= self.transformer.wpe.weight.numel()
        print(f"Number of parameters: {n_params/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size

        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Embeddings
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Forward through blocks
        for block in self.transformer.h:
            x = block(x, tok_emb=tok_emb)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def set_phase(self, phase1):
        """Set phase for all RA blocks."""
        for block in self.transformer.h:
            block.phase1 = phase1

    def get_router_stats(self) -> Dict[str, float]:
        """Get average router probabilities across all layers."""
        stats = {"p_none": 0.0, "p_ra": 0.0, "p_full": 0.0, "p_both": 0.0}
        count = 0

        for block in self.transformer.h:
            if block.last_router_probs is not None:
                for k in stats:
                    key = k.replace("p_", "")
                    if key in block.last_router_probs:
                        stats[k] += block.last_router_probs[key]
                count += 1

        if count > 0:
            for k in stats:
                stats[k] /= count

        return stats


class RATrainer(BaseGPT2Trainer):
    """
    Trainer for RA with compute routing.

    Supports two steps:
      - Step 0: Baseline (standard GPT-2 with CausalSelfAttention)
      - Step 1: RA with routing (router decides compute tier)
    """

    def __init__(self, args, config, ablation_step: Optional[str] = None):
        """
        Initialize RA trainer.

        Args:
            args: Command-line arguments
            config: Config object
            ablation_step: "0" for baseline, "1" for RA
        """
        # Configure ablation step
        self.ablation_step = ablation_step or getattr(args, "ra_step", "0")
        self._configure_step(args, self.ablation_step)

        super().__init__(args, config)

        # Initialize model
        self.model = self.create_model()
        self.raw_model = self.model.module if self.ddp else self.model

        # Initialize optimizer
        (
            self.optimizer,
            self.scheduler,
            self.gradient_clip_norm,
            self.spam_state,
            self.adamwprune_state,
        ) = self.create_optimizer()

        # Setup mixed precision
        self.setup_mixed_precision()

        # Track phase transition
        self.transitioned = False

        # Warmup scheduler (only for RA mode)
        if self.args.enable_routing:
            self.warmup_scheduler = WarmupScheduler(
                threshold=args.warmup_loss_drop,
                min_evals=2,
            )
        else:
            self.warmup_scheduler = None

    def _configure_step(self, args, step: str):
        """Configure args based on ablation step."""
        # RA configuration
        args.ra_head_frac = getattr(args, "ra_head_frac", 0.25)
        args.router_hidden = getattr(args, "router_hidden", 16)
        args.router_bias_full = getattr(args, "router_bias_full", -1.0)
        args.warmup_loss_drop = getattr(args, "warmup_loss_drop", 0.15)
        args.compute_penalty_weight = getattr(args, "compute_penalty_weight", 0.01)

        if step == "0":
            # Baseline: standard GPT-2
            args.enable_routing = False
            args.use_baseline = True
            print(f"Step {step}: Baseline (standard GPT-2)")
        elif step == "1":
            # RA with routing
            args.enable_routing = True
            args.use_baseline = False
            print(f"Step {step}: RA with routing")
        else:
            raise ValueError(f"Unknown step: {step}. Use '0' or '1'.")

    def create_model(self):
        """Create GPT model (baseline or RA)."""
        # GPT config
        gpt_config = GPTConfig(
            block_size=self.config.GPT2_BLOCK_SIZE,
            vocab_size=self.config.TOKENIZER_VOCAB_SIZE,
            n_layer=self.config.GPT2_N_LAYER,
            n_head=self.config.GPT2_N_HEAD,
            n_embd=self.config.GPT2_N_EMBD,
        )

        args = self.args

        if args.use_baseline:
            # True baseline: standard GPT-2
            model = GPT(gpt_config)
            print("Created baseline GPT-2")
        else:
            # RA config
            ra_config = RAConfig(
                d_model=gpt_config.n_embd,
                n_heads=gpt_config.n_head,
                block_size=gpt_config.block_size,
                ra_head_frac=args.ra_head_frac,
                router_hidden=args.router_hidden,
                router_bias_full=args.router_bias_full,
                warmup_loss_drop=args.warmup_loss_drop,
            )

            model = RAGPT(gpt_config, ra_config)

            # Start in phase 1 (warmup)
            model.set_phase(phase1=True)

            # Log configuration
            n_ra = max(1, int(round(args.ra_head_frac * gpt_config.n_head)))
            n_ra = min(n_ra, gpt_config.n_head - 1)
            n_full = gpt_config.n_head - n_ra

            print(f"Created RAGPT with routing")
            print(f"  Heads: {n_full} FULL + {n_ra} RA = {gpt_config.n_head} total")

        # Move to device
        model = model.to(self.device)

        # Compile if requested
        if getattr(self.config, "COMPILE_MODEL", False):
            print("Compiling model...")
            model = torch.compile(model)

        # Setup DDP if needed
        if self.ddp:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
            )

        return model

    def train_step(self, batch):
        """Execute one training step."""
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        # Forward pass
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits, loss = self.raw_model(x, y)

        # Backward pass
        self.scaler.scale(loss).backward()

        # Gradient clipping
        if self.gradient_clip_norm > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_norm
            )

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        # LR scheduler step
        if self.scheduler is not None:
            self.scheduler.step()

        # Get router stats for logging
        router_stats = {}
        if hasattr(self.raw_model, "get_router_stats"):
            router_stats = self.raw_model.get_router_stats()

        return {
            "loss": loss.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
            **router_stats,
        }

    def on_eval(self, eval_loss, eval_metrics):
        """
        Called after each evaluation.

        Checks for phase transition if routing is enabled.
        """
        if not self.args.enable_routing:
            return

        if self.transitioned:
            return

        # Check for transition
        if self.warmup_scheduler.should_transition(eval_loss):
            print(f"\n{'='*60}")
            print("PHASE TRANSITION: Enabling routing")
            print(f"  Eval loss: {eval_loss:.4f}")
            print(f"  Initial loss: {self.warmup_scheduler.initial_loss:.4f}")
            rel_drop = (
                self.warmup_scheduler.initial_loss - eval_loss
            ) / self.warmup_scheduler.initial_loss
            print(f"  Relative drop: {rel_drop*100:.1f}%")
            print(f"{'='*60}\n")

            self.raw_model.set_phase(phase1=False)
            self.transitioned = True

    def get_run_name(self):
        """Get run name for logging."""
        step = self.ablation_step
        mode = "baseline" if step == "0" else "ra_routing"
        optimizer = getattr(self.args, "optimizer", "adamw")
        return f"gpt2_{optimizer}_{mode}"

    def get_step_description(self):
        """Get human-readable step description."""
        if self.ablation_step == "0":
            return "Baseline (standard GPT-2)"
        else:
            return f"RA with routing (ra_frac={self.args.ra_head_frac})"
