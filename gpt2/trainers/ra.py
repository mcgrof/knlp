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
from .vanilla import VanillaGPT2Trainer


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
            attn_out = self.mixer(out_ra, out_full, probs)

            # Store for logging (soft averages and hard counts)
            p_ra = probs["ra"].detach()
            p_full = probs["full"].detach()
            total_tokens = p_ra.numel()
            ra_tokens = (p_ra > 0.5).sum().item()

            self.last_router_probs = {
                "ra": p_ra.mean().item(),
                "full": p_full.mean().item(),
                "ra_token_pct": ra_tokens / total_tokens if total_tokens > 0 else 0.0,
                "total_tokens": total_tokens,
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

    def get_num_params(self, non_embedding=True):
        """Return number of parameters (excluding position embeddings by default)."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def get_router_stats(self) -> Dict[str, float]:
        """Get average router stats across all layers."""
        stats = {"p_ra": 0.0, "p_full": 0.0, "ra_token_pct": 0.0}
        count = 0

        for block in self.transformer.h:
            if block.last_router_probs is not None:
                stats["p_ra"] += block.last_router_probs.get("ra", 0.0)
                stats["p_full"] += block.last_router_probs.get("full", 0.0)
                stats["ra_token_pct"] += block.last_router_probs.get(
                    "ra_token_pct", 0.0
                )
                count += 1

        if count > 0:
            for k in stats:
                stats[k] /= count

        return stats


class RATrainer(VanillaGPT2Trainer):
    """
    Trainer for RA with compute routing.

    Supports two steps:
      - Step 0: Baseline (standard GPT-2 with CausalSelfAttention)
      - Step 1: RA with routing (router decides compute tier)

    Inherits from VanillaGPT2Trainer to get the training loop,
    overrides train_step and adds router stats logging.
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

        # Parent class (VanillaGPT2Trainer) handles model, optimizer, and mixed precision
        # via polymorphism - self.create_model() resolves to RATrainer.create_model()
        super().__init__(args, config)

        # Track phase transition
        self.transitioned = False

        # Warmup scheduler (only for RA mode)
        if self.args.enable_routing:
            self.warmup_scheduler = WarmupScheduler(
                threshold=args.warmup_loss_drop,
                min_evals=2,
            )

            # Skip RA warmup if requested - enable routing immediately
            if args.skip_ra_warmup:
                self.raw_model.set_phase(phase1=False)
                self.transitioned = True
                print("Skipping RA warmup - routing enabled from start")
        else:
            self.warmup_scheduler = None

    def _configure_step(self, args, step: str):
        """Configure args based on ablation step."""
        # RA configuration
        args.ra_head_frac = getattr(args, "ra_head_frac", 0.25)
        args.router_hidden = getattr(args, "router_hidden", 16)
        args.router_bias_full = getattr(args, "router_bias_full", -1.0)
        args.warmup_loss_drop = getattr(args, "warmup_loss_drop", 0.05)
        args.compute_penalty_weight = getattr(args, "compute_penalty_weight", 0.01)
        args.skip_ra_warmup = getattr(args, "skip_ra_warmup", False)

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
        # Get config from model name (gpt2, gpt2-medium, etc.)
        gpt_config = GPTConfig.from_name(self.args.model_name)
        # Override block_size if specified
        if hasattr(self.config, "GPT2_BLOCK_SIZE"):
            gpt_config.block_size = self.config.GPT2_BLOCK_SIZE

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

    def log_metrics(self, metrics_dict):
        """Override to add router stats to logged metrics."""
        # Get the actual model (handle torch.compile wrapper)
        model = self.raw_model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

        # Add router stats if available
        if hasattr(model, "get_router_stats"):
            router_stats = model.get_router_stats()
            metrics_dict.update(router_stats)

        # Call parent's log_metrics
        super().log_metrics(metrics_dict)

    def estimate_loss(self):
        """Override to call on_eval after evaluation for phase transition."""
        losses = super().estimate_loss()

        # Call on_eval for phase transition check
        if self.master_process:
            self.on_eval(losses["val"], {})

        return losses

    def run_inference_benchmark(self, num_tokens=100, num_runs=5):
        """
        Run inference benchmarks after training.

        Measures:
        - TTFT (Time To First Token)
        - KV cache size reduction estimate
        - Throughput (tokens/sec)
        """
        import time

        if not self.master_process:
            return {}

        print("\n" + "=" * 60)
        print("INFERENCE BENCHMARK")
        print("=" * 60)

        # Get the actual model
        model = self.raw_model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

        model.eval()

        # Use a sample prompt
        prompt_text = "The quick brown fox jumps over the lazy"
        # Simple tokenization (space-split for demo, real would use tokenizer)
        prompt_ids = torch.randint(
            0, model.config.vocab_size, (1, 8), device=self.device
        )

        ttft_times = []
        total_times = []
        ra_token_pcts = []

        with torch.no_grad():
            for run in range(num_runs):
                generated = prompt_ids.clone()

                # Measure TTFT (time to first token)
                torch.cuda.synchronize()
                t0 = time.perf_counter()

                # Generate first token
                logits, _ = model(generated)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

                torch.cuda.synchronize()
                ttft = time.perf_counter() - t0
                ttft_times.append(ttft)

                # Generate remaining tokens
                for _ in range(num_tokens - 1):
                    logits, _ = model(generated)
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)

                torch.cuda.synchronize()
                total_time = time.perf_counter() - t0
                total_times.append(total_time)

                # Get router stats for KV cache estimation
                if hasattr(model, "get_router_stats"):
                    stats = model.get_router_stats()
                    ra_token_pcts.append(stats.get("ra_token_pct", 0.0))

        # Compute metrics
        avg_ttft = sum(ttft_times) / len(ttft_times) * 1000  # ms
        avg_total = sum(total_times) / len(total_times)
        throughput = num_tokens / avg_total

        # KV cache reduction estimate
        # RA tokens use 3 heads instead of 12, so KV cache is 3/12 = 25% for those
        avg_ra_pct = sum(ra_token_pcts) / len(ra_token_pcts) if ra_token_pcts else 0.0
        # Weighted average: ra_pct * 0.25 + (1 - ra_pct) * 1.0
        kv_cache_ratio = avg_ra_pct * 0.25 + (1 - avg_ra_pct) * 1.0
        kv_cache_reduction = (1 - kv_cache_ratio) * 100

        print(f"TTFT: {avg_ttft:.2f} ms")
        print(f"Throughput: {throughput:.1f} tokens/sec")
        print(f"RA token %: {avg_ra_pct*100:.1f}%")
        print(f"KV cache reduction: {kv_cache_reduction:.1f}%")

        # Qualitative metrics
        print("\n--- Generation Quality ---")

        # Generate longer sample for quality analysis
        sample_ids = torch.randint(
            0, model.config.vocab_size, (1, 8), device=self.device
        )
        generated_tokens = []

        with torch.no_grad():
            gen = sample_ids.clone()
            for _ in range(200):
                logits, _ = model(gen)
                # Sample with temperature for diversity
                probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
                next_token = torch.multinomial(probs, 1)
                gen = torch.cat([gen, next_token], dim=1)
                generated_tokens.append(next_token.item())

        # Compute repetition metrics
        def calc_distinct_n(tokens, n):
            if len(tokens) < n:
                return 1.0
            ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
            return len(set(ngrams)) / len(ngrams) if ngrams else 1.0

        distinct_1 = calc_distinct_n(generated_tokens, 1)
        distinct_2 = calc_distinct_n(generated_tokens, 2)
        distinct_3 = calc_distinct_n(generated_tokens, 3)

        # Repetition rate (how many tokens are same as previous)
        repetitions = sum(
            1
            for i in range(1, len(generated_tokens))
            if generated_tokens[i] == generated_tokens[i - 1]
        )
        rep_rate = repetitions / len(generated_tokens) if generated_tokens else 0

        print(f"Distinct-1: {distinct_1:.3f} (unique unigrams)")
        print(f"Distinct-2: {distinct_2:.3f} (unique bigrams)")
        print(f"Distinct-3: {distinct_3:.3f} (unique trigrams)")
        print(f"Repetition rate: {rep_rate:.3f} (consecutive repeats)")

        # Self-perplexity: model's perplexity on its own generation
        with torch.no_grad():
            gen_input = gen[:, :-1]
            gen_target = gen[:, 1:]
            # Pass targets to get full sequence logits (not just last token)
            logits, loss = model(gen_input, gen_target)
            self_ppl = math.exp(min(loss.item(), 20))

        print(f"Self-perplexity: {self_ppl:.2f}")

        # Run lm-eval benchmarks if enabled
        lm_eval_results = {}
        if getattr(self.args, "run_lm_eval", False):
            lm_eval_results = self._run_lm_eval(model, generated_tokens)

        # Quality assessment
        if distinct_2 < 0.3:
            quality = "Poor (high repetition)"
        elif distinct_2 < 0.5:
            quality = "Fair"
        elif distinct_2 < 0.7:
            quality = "Good"
        else:
            quality = "Excellent (high diversity)"

        print(f"Quality: {quality}")
        print("=" * 60 + "\n")

        metrics = {
            "inference/ttft_ms": avg_ttft,
            "inference/throughput_tps": throughput,
            "inference/ra_token_pct": avg_ra_pct,
            "inference/kv_cache_reduction_pct": kv_cache_reduction,
            "inference/distinct_1": distinct_1,
            "inference/distinct_2": distinct_2,
            "inference/distinct_3": distinct_3,
            "inference/rep_rate": rep_rate,
            "inference/self_ppl": self_ppl,
        }
        # Add lm-eval results
        metrics.update(lm_eval_results)

        # Log to trackers
        self.log_metrics(metrics)

        # Generate and log text samples to W&B
        if "wandb" in self.trackers:
            try:
                import wandb
                import tiktoken

                enc = tiktoken.get_encoding("gpt2")

                prompts = [
                    "The quick brown fox",
                    "In the beginning",
                    "Once upon a time",
                    "The scientist discovered",
                    "It was a dark and stormy",
                    "The future of AI",
                    "When I was young",
                    "The most important thing",
                    "Yesterday I learned",
                    "The world needs more",
                ]

                samples = []
                with torch.no_grad():
                    for prompt in prompts:
                        prompt_ids = torch.tensor(
                            [enc.encode(prompt)], device=self.device
                        )
                        gen = prompt_ids.clone()

                        # Generate 50 tokens
                        for _ in range(50):
                            if gen.size(1) >= model.config.block_size:
                                break
                            logits, _ = model(gen)
                            probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
                            next_token = torch.multinomial(probs, 1)
                            gen = torch.cat([gen, next_token], dim=1)

                        output_text = enc.decode(gen[0].tolist())
                        samples.append([prompt, output_text])

                # Log as W&B table
                table = wandb.Table(columns=["Prompt", "Generated Text"], data=samples)
                wandb.log({"inference/text_samples": table})
                print(f"Logged {len(samples)} text samples to W&B")

            except Exception as e:
                print(f"Warning: Failed to log text samples: {e}")

        model.train()
        return metrics

    def _run_lm_eval(self, model, generated_tokens):
        """Run lm-eval benchmarks on the model."""
        try:
            from lm_eval import evaluator
            from lm_eval.api.model import LM
            from lm_eval.api.instance import Instance
            import numpy as np
            import tiktoken
        except ImportError as e:
            print(f"(Install dependencies: pip install lm-eval tiktoken) - {e}")
            return {}

        print("\n--- lm-eval Benchmarks ---")

        # Get tasks from args
        tasks = getattr(self.args, "lm_eval_tasks", "hellaswag").split(",")
        tasks = [t.strip() for t in tasks]

        # Load GPT-2 tokenizer
        enc = tiktoken.get_encoding("gpt2")

        # Create a wrapper for our model
        class RAModelWrapper(LM):
            def __init__(wrapper_self, model, device, config, tokenizer):
                super().__init__()
                wrapper_self._model = model
                wrapper_self._device = device
                wrapper_self._config = config
                wrapper_self._tokenizer = tokenizer
                wrapper_self.batch_size_per_gpu = 1

            @property
            def eot_token_id(wrapper_self):
                return wrapper_self._tokenizer.eot_token

            @property
            def max_length(wrapper_self):
                return wrapper_self._config.block_size

            @property
            def max_gen_toks(wrapper_self):
                return 256

            @property
            def batch_size(wrapper_self):
                return 1

            @property
            def device(wrapper_self):
                return wrapper_self._device

            def tok_encode(wrapper_self, string, **kwargs):
                return wrapper_self._tokenizer.encode(
                    string, allowed_special={"<|endoftext|>"}
                )

            def tok_decode(wrapper_self, tokens, **kwargs):
                return wrapper_self._tokenizer.decode(tokens)

            def _loglikelihood_tokens(wrapper_self, requests, disable_tqdm=False):
                results = []
                for context, continuation in requests:
                    ctx_tensor = torch.tensor([context], device=wrapper_self._device)
                    with torch.no_grad():
                        logits, _ = wrapper_self._model(ctx_tensor)
                    # Compute log likelihood of continuation
                    log_probs = F.log_softmax(
                        logits[0, -len(continuation) - 1 : -1], dim=-1
                    )
                    ll = sum(
                        log_probs[i, continuation[i]].item()
                        for i in range(min(len(continuation), log_probs.size(0)))
                    )
                    results.append((ll, True))
                return results

            def loglikelihood(wrapper_self, requests):
                new_reqs = []
                for req in requests:
                    context = wrapper_self.tok_encode(req.args[0])
                    continuation = wrapper_self.tok_encode(req.args[1])
                    new_reqs.append((context, continuation))
                return wrapper_self._loglikelihood_tokens(new_reqs)

            def loglikelihood_rolling(wrapper_self, requests):
                results = []
                for req in requests:
                    tokens = wrapper_self.tok_encode(req.args[0])
                    if len(tokens) < 2:
                        results.append((0.0, True))
                        continue
                    ctx = tokens[:-1]
                    cont = tokens[1:]
                    ll, _ = wrapper_self._loglikelihood_tokens([(ctx, cont)])[0]
                    results.append((ll, True))
                return results

            def generate_until(wrapper_self, requests):
                results = []
                for req in requests:
                    context = wrapper_self.tok_encode(req.args[0])
                    gen_kwargs = req.args[1]
                    max_gen = gen_kwargs.get("max_gen_toks", 100)

                    ctx_tensor = torch.tensor(
                        [context[-wrapper_self.max_length :]],
                        device=wrapper_self._device,
                    )
                    generated = []

                    with torch.no_grad():
                        for _ in range(max_gen):
                            logits, _ = wrapper_self._model(ctx_tensor)
                            next_token = logits[0, -1].argmax().item()
                            generated.append(next_token)
                            ctx_tensor = torch.cat(
                                [
                                    ctx_tensor,
                                    torch.tensor(
                                        [[next_token]], device=wrapper_self._device
                                    ),
                                ],
                                dim=1,
                            )
                            if ctx_tensor.size(1) > wrapper_self.max_length:
                                ctx_tensor = ctx_tensor[:, -wrapper_self.max_length :]

                    results.append(wrapper_self.tok_decode(generated))
                return results

        # Create wrapper and run evaluation
        try:
            wrapper = RAModelWrapper(model, self.device, model.config, enc)

            # Get limit from config (None = all samples)
            limit = getattr(self.args, "lm_eval_limit", None)
            if limit:
                print(f"Running lm-eval with limit={limit} samples per task")

            results = evaluator.simple_evaluate(
                model=wrapper,
                tasks=tasks,
                num_fewshot=0,
                batch_size=1,
                device=str(self.device),
                limit=limit,
            )

            # Extract metrics
            lm_eval_metrics = {}
            for task_name, task_results in results.get("results", {}).items():
                for metric_name, value in task_results.items():
                    if isinstance(value, (int, float)) and not metric_name.endswith(
                        "_stderr"
                    ):
                        key = f"lm_eval/{task_name}_{metric_name}"
                        lm_eval_metrics[key] = value
                        print(f"{task_name}/{metric_name}: {value:.4f}")

            return lm_eval_metrics

        except Exception as e:
            print(f"lm-eval failed: {e}")
            import traceback

            traceback.print_exc()
            return {}

    def on_train_end(self):
        """Run inference benchmark before trackers finish."""
        # Run inference benchmark for all steps (baseline and routing)
        if not getattr(self.args, "dry_run", False):
            self.run_inference_benchmark()
