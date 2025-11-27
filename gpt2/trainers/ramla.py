"""
RAMLA Trainer - RA + MLA Learning Rate Ablation

Handles all architecture variants for the LR ablation study:
  B0, B1       - Baseline GPT-2 at 6e-4 / 1.2e-3
  MLA0, MLA1   - MLA at 6e-4 / 1.2e-3
  RA0, RA1     - RA routing at 6e-4 / 1.2e-3
  RAMLA0, RAMLA1 - RA+MLA at 6e-4 / 1.2e-3
  RAMLAKV0, RAMLAKV1 - RA+MLA+KVSplice at 6e-4 / 1.2e-3
"""

import sys
import os

# Add parent to path
parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, parent_dir)

from typing import Optional, Dict
import torch
import torch.nn.functional as F
from gpt2.trainers.vanilla import VanillaGPT2Trainer
# RATrainer removed with deprecated RA implementation


# Learning rates for ablation
LR_STANDARD = 6e-4
LR_AGGRESSIVE = 1.2e-3


def parse_step(step: str) -> Dict:
    """
    Parse ablation step name into configuration.

    Returns:
        dict with keys: arch, lr, lr_name
    """
    step = step.upper()

    # Support legacy naming with 0/1 suffixes for backwards compatibility
    # New naming uses just the architecture name without learning rate suffix
    if step.endswith(("0", "1")):
        # Legacy: strip suffix, always use standard LR
        base = step[:-1]
    else:
        # New style: no suffix needed
        base = step

    lr = LR_STANDARD
    lr_name = "standard"

    # Map step names to architecture identifiers
    arch_map = {
        "B": "baseline",  # GPT2 baseline
        "MLA": "mla",  # GPT2_MLA
        "MLAKV2": "mlakv2",  # GPT2_MLA_KV2
        "MLAKV2M": "mlakv2mlp",  # GPT2_MLA_KV2M
        "MLAKV": "mlakv",  # GPT2_MLA_KV
        "RA": "ra",  # GPT2_RA (router-based)
        "RALEARN": "ra_learned",  # GPT2_RA_Learned (blend both paths)
        "RANONE": "ra_fixed_none",  # GPT2_RA_Fixed pattern=none
        "RAEARLY": "ra_fixed_early",  # GPT2_RA_Fixed pattern=early
        "RALATE": "ra_fixed_late",  # GPT2_RA_Fixed pattern=late
        "RAALL": "ra_fixed_all",  # GPT2_RA_Fixed pattern=all
        "RAMLA": "ramla",  # GPT2_MLA_RA
        "RAMLAKV": "ramlakv",  # GPT2_MLA_RA_KV
        "RAMLAKVM": "ramlakvm",  # GPT2_MLA_RA_KVM
    }

    if base in arch_map:
        arch = arch_map[base]
    else:
        raise ValueError(f"Unknown architecture: {base} (from step: {step})")

    return {
        "arch": arch,
        "lr": lr,
        "lr_name": lr_name,
    }


class RAMLATrainer(VanillaGPT2Trainer):
    """
    Trainer for RA-MLA LR ablation study.

    Dynamically selects architecture based on step name and configures
    appropriate learning rate.
    """

    def __init__(self, args, config, ablation_step: str = "B0"):
        self.ablation_step = ablation_step
        self.step_config = parse_step(ablation_step)

        # Override learning rate based on step
        args.learning_rate = self.step_config["lr"]

        print(
            f"Step {ablation_step}: {self.step_config['arch']} architecture, "
            f"{self.step_config['lr_name']} LR ({self.step_config['lr']:.1e})"
        )

        # Initialize based on architecture
        if self.step_config["arch"] == "baseline":
            # Standard GPT-2
            super().__init__(args, config)

        elif self.step_config["arch"] == "ra":
            # RA routing - delegate to RATrainer
            # We need to use composition here since RATrainer has different model
            self._ra_trainer = RATrainer(args, config, ablation_step="1")
            self.model = self._ra_trainer.model
            self.optimizer = self._ra_trainer.optimizer
            self.scheduler = self._ra_trainer.scheduler
            self.ctx = self._ra_trainer.ctx
            self.scaler = self._ra_trainer.scaler
            self.args = args
            self.config = config
            self.trackers = self._ra_trainer.trackers

        elif self.step_config["arch"] == "ra_learned":
            # Pure GPT-2 with learned reciprocal attention (no MLA, no router)
            super().__init__(args, config)
            self._setup_ra_learned_model()

        elif self.step_config["arch"].startswith("ra_fixed_"):
            # GPT2_RA_Fixed with predetermined pattern
            super().__init__(args, config)
            pattern = self.step_config["arch"].replace("ra_fixed_", "")
            self._setup_ra_fixed_model(pattern)

        elif self.step_config["arch"] in [
            "mla",
            "mlakv",
            "mlakv2",
            "mlakv2mlp",
            "ramla",
            "ramlakv",
            "ramlakvm",
        ]:
            # MLA-based architectures
            super().__init__(args, config)
            # Replace model with MLA variant
            self._setup_mla_model()

        else:
            raise ValueError(f"Unknown architecture: {self.step_config['arch']}")

    def _setup_ra_learned_model(self):
        """DEPRECATED: GPT2_RA_Learned removed with old RA implementation."""
        raise NotImplementedError(
            "GPT2_RA_Learned deprecated. Old RA implementation (Q/K swap within layers) "
            "was fundamentally flawed. See docs/archive/ra-old.md for details."
        )
        from gpt2.model import GPTConfig

        # Create GPTConfig
        gpt_config = GPTConfig.from_name(self.args.model_name)
        gpt_config.block_size = self.args.block_size
        gpt_config.dropout = self.args.dropout
        gpt_config.bias = getattr(self.args, "bias", True)

        # Create model with proper GPTConfig
        self.model = GPT2_RA_Learned(gpt_config)
        print(f"Created GPT2_RA_Learned with gradient flow")
        print(f"  Params: {self.model.get_num_params():,}")
        print(f"  Training: blends both standard and reciprocal paths")
        print(f"  Inference: uses learned threshold per layer")

        # Move to device and setup optimizer
        self.model.to(self.args.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

        # Rebuild scheduler for new optimizer
        if hasattr(self, "scheduler") and self.scheduler is not None:
            from torch.optim.lr_scheduler import CosineAnnealingLR

            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=getattr(self.args, "max_iters", 10000),
                eta_min=getattr(self.args, "min_lr", 6e-5),
            )

    def _setup_ra_fixed_model(self, pattern: str):
        """DEPRECATED: GPT2_RA_Fixed removed with old RA implementation."""
        raise NotImplementedError(
            "GPT2_RA_Fixed deprecated. Old RA implementation (Q/K swap within layers) "
            "was fundamentally flawed. See docs/archive/ra-old.md for details."
        )
        from gpt2.model import GPTConfig

        # Create GPTConfig
        gpt_config = GPTConfig.from_name(self.args.model_name)
        gpt_config.block_size = self.args.block_size
        gpt_config.dropout = self.args.dropout
        gpt_config.bias = getattr(self.args, "bias", True)

        # Create model with fixed pattern
        self.model = GPT2_RA_Fixed(gpt_config, pattern=pattern)
        print(f"Created GPT2_RA_Fixed with pattern={pattern}")
        print(f"  Params: {self.model.get_num_params():,}")
        stats = self.model.get_pattern_stats()
        print(
            f"  Pattern: {stats['n_reciprocal']} reciprocal, {stats['n_standard']} standard"
        )
        print(f"  Reciprocal layers: {stats['reciprocal_layers']}")

        # Move to device and setup optimizer
        self.model.to(self.args.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

        # Rebuild scheduler for new optimizer
        if hasattr(self, "scheduler") and self.scheduler is not None:
            from torch.optim.lr_scheduler import CosineAnnealingLR

            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=getattr(self.args, "max_iters", 10000),
                eta_min=getattr(self.args, "min_lr", 6e-5),
            )

    def _setup_mla_model(self):
        """Replace model with MLA variant."""
        import torch
        from gpt2.mla import (
            MLA_Config,
            GPT2_MLA,
            GPT2_MLA_KV,
            GPT2_MLA_KV2,
            GPT2_MLA_KV2M,
        )

        arch = self.step_config["arch"]

        # Get MLA config from args or defaults
        d_latent = getattr(self.args, "mla_d_latent", 256)
        compression_ratio = getattr(self.args, "mla_compression_ratio", 0.5)

        cfg = MLA_Config(
            d_model=768,
            n_heads=12,
            head_dim=64,
            d_latent=d_latent,
            block_size=self.args.block_size,
            n_layers=12,
        )

        # Create appropriate full GPT model
        if arch == "mla":
            self.model = GPT2_MLA(cfg)
            print(f"Created GPT2_MLA with d_latent={d_latent}")
        elif arch == "mlakv":
            self.model = GPT2_MLA_KV(cfg, compression_ratio=compression_ratio)
            print(f"Created GPT2_MLA_KV (MLA + KVSplice, no RA)")
            print(f"  Compression: {self.model.get_compression_stats()}")
        elif arch == "mlakv2":
            self.model = GPT2_MLA_KV2(cfg, compression_ratio=compression_ratio)
            print(f"Created GPT2_MLA_KV2 (MLA + 2-latent: Q direct, K/V compressed)")
            print(f"  Compression: {self.model.get_compression_stats()}")
        elif arch == "mlakv2mlp":
            mlp_d_latent = getattr(self.args, "mlpsplice_d_latent", 256)
            self.model = GPT2_MLA_KV2M(
                cfg, compression_ratio=compression_ratio, mlp_d_latent=mlp_d_latent
            )
            print(f"Created GPT2_MLA_KV2M (2-latent + MLPSplice)")
            print(f"  Compression: {self.model.get_compression_stats()}")
        elif arch in ("ramla", "ramlakv", "ramlakvm"):
            raise NotImplementedError(
                f"Architecture '{arch}' deprecated. RA+MLA combinations used broken RA "
                "implementation (Q/K swap within layers). Use pure MLA variants instead: "
                "mla, mlakv, mlakv2, mlakv2mlp"
            )

        # Move to device
        self.model = self.model.to(self.args.device)

        # Store config for balance loss
        self._mla_config = cfg

        print(f"Number of parameters: {self.model.get_num_params()/1e6:.2f}M")

        # Recreate optimizer with new model
        self._setup_optimizer()

    def _setup_optimizer(self):
        """Setup optimizer for the MLA model."""
        import torch

        # Use AdamW with weight decay
        param_dict = {
            pn: p for pn, p in self.model.named_parameters() if p.requires_grad
        }

        # Separate weight decay and no-decay params
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": self.args.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        self.optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.args.learning_rate,
            betas=(0.9, 0.95),
        )

        print(f"Created optimizer with LR={self.args.learning_rate:.1e}")

    def train(self):
        """Run training."""
        if self.step_config["arch"] == "ra":
            # Use RATrainer's train method
            return self._ra_trainer.train()
        else:
            # For MLAKV/RAMLAKV, set up periodic metrics logging
            if self.step_config["arch"] in ["mlakv", "ramlakv"]:
                self._setup_kvsplice_logging()

            # Use base trainer (on_train_end will be called before wandb finishes)
            return super().train()

    def on_train_end(self):
        """Log final metrics before trackers finish."""
        # Skip for RA architecture (uses its own trainer)
        if self.step_config["arch"] == "ra":
            return

        # Log final metrics based on architecture
        if self.step_config["arch"] in ["mlakv", "ramlakv"]:
            self._log_kvsplice_metrics()

        # Log Fisher metrics for all architectures that support it
        self._log_fisher_metrics()

        # Log KV cache memory metrics
        self._log_kv_cache_metrics()

        # Run lm-eval if requested
        if getattr(self.args, "run_lm_eval", False):
            self._run_lm_eval()

        # Generate text samples
        self._log_text_samples()

    def _log_text_samples(self):
        """Generate and log text samples to W&B."""
        import tiktoken

        prompts = [
            "The meaning of life is",
            "In a shocking discovery, scientists found that",
            "Once upon a time in a land far away,",
            "The best way to learn programming is",
        ]

        enc = tiktoken.get_encoding("gpt2")
        device = next(self.model.parameters()).device

        samples = []
        print("\n--- Text Generation Samples ---")

        self.model.eval()
        with torch.no_grad():
            for prompt in prompts:
                tokens = enc.encode(prompt)
                tokens = torch.tensor(
                    tokens, dtype=torch.long, device=device
                ).unsqueeze(0)

                # Generate
                for _ in range(50):  # max 50 new tokens
                    if tokens.size(1) > 1024:
                        tokens = tokens[:, -1024:]

                    logits, _ = self.model(tokens)
                    logits = logits[:, -1, :] / 0.8  # temperature

                    # Top-k sampling
                    top_k = 40
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    tokens = torch.cat([tokens, next_token], dim=1)

                    if next_token.item() == enc.eot_token:
                        break

                output = enc.decode(tokens[0].tolist())
                samples.append({"prompt": prompt, "output": output})
                print(f"\nPrompt: {prompt}")
                print(f"Output: {output[:200]}...")

        # Log to W&B as table
        if "wandb" in self.trackers:
            try:
                import wandb

                table = wandb.Table(columns=["prompt", "generated_text"])
                for s in samples:
                    table.add_data(s["prompt"], s["output"])
                wandb.log({"text_samples": table})
            except Exception as e:
                print(f"Warning: Failed to log text samples to wandb: {e}")

    def _log_kv_cache_metrics(self):
        """Log KV cache memory metrics to W&B."""
        # Get model config
        if hasattr(self.model, "cfg"):
            cfg = self.model.cfg
            n_layers = cfg.n_layers
            n_heads = cfg.n_heads
            head_dim = cfg.head_dim
            d_latent = cfg.d_latent
        else:
            return

        # Test sequence lengths
        seq_lengths = [512, 1024, 2048, 4096]
        batch_size = 1

        metrics = {}

        for seq_len in seq_lengths:
            # Standard KV cache: K and V for each layer
            standard_cache_bytes = (
                batch_size * n_layers * 2 * n_heads * seq_len * head_dim * 2
            )
            standard_cache_mb = standard_cache_bytes / 1024**2

            # Check if model uses KVSplice
            if hasattr(self.model, "compression_ratio"):
                compression_ratio = self.model.compression_ratio
                d_compressed = int(d_latent * compression_ratio)
                compressed_cache_bytes = (
                    batch_size * n_layers * seq_len * d_compressed * 2
                )
                actual_cache_mb = compressed_cache_bytes / 1024**2
                cache_type = "kvsplice"
                savings_pct = (1 - actual_cache_mb / standard_cache_mb) * 100
            else:
                actual_cache_mb = standard_cache_mb
                cache_type = "standard"
                savings_pct = 0.0

            metrics[f"kv_cache/seq{seq_len}_standard_mb"] = standard_cache_mb
            metrics[f"kv_cache/seq{seq_len}_actual_mb"] = actual_cache_mb
            metrics[f"kv_cache/seq{seq_len}_savings_pct"] = savings_pct

        # Summary metrics
        metrics["kv_cache/type"] = 1.0 if cache_type == "kvsplice" else 0.0
        if hasattr(self.model, "compression_ratio"):
            metrics["kv_cache/compression_ratio"] = self.model.compression_ratio

        # Print summary
        print("\n--- KV Cache Memory ---")
        print(f"  Cache type: {cache_type}")
        for seq_len in seq_lengths:
            actual = metrics[f"kv_cache/seq{seq_len}_actual_mb"]
            savings = metrics[f"kv_cache/seq{seq_len}_savings_pct"]
            if savings > 0:
                print(f"  seq={seq_len}: {actual:.1f} MB ({savings:.0f}% savings)")
            else:
                print(f"  seq={seq_len}: {actual:.1f} MB")

        # Log to W&B
        step = getattr(self, "iter_num", None)
        if "wandb" in self.trackers:
            try:
                import wandb

                wandb.log(metrics, step=step)
            except Exception as e:
                print(f"Warning: Failed to log kv_cache metrics to wandb: {e}")

    @torch.no_grad()
    def estimate_loss(self):
        """Override to log Fisher metrics at each evaluation."""
        # Call base class evaluation
        losses = super().estimate_loss()

        # Log Fisher metrics during training (not just at the end)
        # This tracks curvature evolution: early=bumpy, mid=peaks, late=flat
        if hasattr(self.model, "compute_fisher_metrics"):
            try:
                batch_size = 4
                seq_len = 128
                device = next(self.model.parameters()).device
                x = torch.randint(0, 50257, (batch_size, seq_len), device=device)

                metrics = self.model.compute_fisher_metrics(x, n_samples=64, topk=8)

                if metrics:
                    step = getattr(self, "iter_num", None)
                    if "wandb" in self.trackers:
                        try:
                            import wandb

                            wandb.log(metrics, step=step)
                        except Exception:
                            pass
            except Exception:
                pass  # Don't interrupt training for metric failures

        return losses

    def _setup_kvsplice_logging(self):
        """Set up KVSplice metrics logging callback."""
        # Store original log_interval for periodic logging
        self._kvsplice_log_interval = getattr(self.args, "eval_interval", 50)

    def _log_kvsplice_metrics(self):
        """Log KVSplice compression metrics to trackers."""
        if not hasattr(self.model, "get_kvsplice_metrics"):
            return

        metrics = self.model.get_kvsplice_metrics()

        # Print summary
        print("\n--- KVSplice Compression Metrics ---")
        print(
            f"  Compression ratio: {metrics.get('kvsplice/compression_ratio', 'N/A')}"
        )
        print(
            f"  Memory reduction: {metrics.get('kvsplice/memory_reduction_pct', 'N/A'):.1f}%"
        )
        if "kvsplice/avg_reconstruction_error" in metrics:
            print(
                f"  Avg reconstruction error: {metrics['kvsplice/avg_reconstruction_error']:.6f}"
            )
        print(
            f"  Reciprocal layers: {metrics.get('kvsplice/reciprocal_layers', 'N/A')}"
        )
        print(f"  Standard layers: {metrics.get('kvsplice/standard_layers', 'N/A')}")

        # Log to trackers
        step = getattr(self, "iter_num", None)
        if "trackio" in self.trackers:
            try:
                import trackio

                trackio.log(metrics)
            except Exception as e:
                print(f"Warning: Failed to log kvsplice metrics to trackio: {e}")

        if "wandb" in self.trackers:
            try:
                import wandb

                wandb.log(metrics, step=step)
            except Exception as e:
                print(f"Warning: Failed to log kvsplice metrics to wandb: {e}")

    def _log_fisher_metrics(self):
        """
        Log Fisher Information Matrix spectrum metrics.

        The FIM eigenvalues reveal the curvature geometry of attention (SPDA paper).
        Lower eigmax and better conditioning indicate smoother optimization.
        """
        if not hasattr(self.model, "compute_fisher_metrics"):
            return

        # Get a small batch for Fisher computation (use shorter sequence for speed)
        try:
            # Create a small dummy batch
            batch_size = 4
            seq_len = 128  # Shorter for O(T^3) eigendecomposition
            device = next(self.model.parameters()).device

            # Use random tokens for Fisher computation
            x = torch.randint(0, 50257, (batch_size, seq_len), device=device)

            metrics = self.model.compute_fisher_metrics(x, n_samples=64, topk=8)

            if metrics:
                # Print summary
                print("\n--- Fisher Information Metrics ---")
                # Find layer aggregates
                for key, value in metrics.items():
                    if "eigmax_mean" in key:
                        layer_idx = key.split("/")[1]
                        print(f"  {layer_idx} eigmax_mean: {value:.6f}")

                # Log to trackers
                step = getattr(self, "iter_num", None)
                if "trackio" in self.trackers:
                    try:
                        import trackio

                        trackio.log(metrics)
                    except Exception as e:
                        print(f"Warning: Failed to log fisher metrics to trackio: {e}")

                if "wandb" in self.trackers:
                    try:
                        import wandb

                        wandb.log(metrics, step=step)
                    except Exception as e:
                        print(f"Warning: Failed to log fisher metrics to wandb: {e}")

        except Exception as e:
            print(f"  (Fisher metrics computation failed: {e})")

    def _run_lm_eval(self):
        """Run lm-eval benchmarks on the model."""
        try:
            from lm_eval import evaluator
            from lm_eval.api.model import LM
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
        model = self.model
        device = self.args.device

        # Create a wrapper for our model
        class MLAModelWrapper(LM):
            def __init__(wrapper_self, model, device, tokenizer, block_size):
                super().__init__()
                wrapper_self._model = model
                wrapper_self._device = device
                wrapper_self._tokenizer = tokenizer
                wrapper_self._block_size = block_size
                wrapper_self.batch_size_per_gpu = 1

            @property
            def eot_token_id(wrapper_self):
                return wrapper_self._tokenizer.eot_token

            @property
            def max_length(wrapper_self):
                return wrapper_self._block_size

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
            wrapper = MLAModelWrapper(model, device, enc, self.args.block_size)

            # Get limit from config (None = all samples)
            limit = getattr(self.args, "lm_eval_limit", None)
            if limit:
                print(f"Running lm-eval with limit={limit} samples per task")

            results = evaluator.simple_evaluate(
                model=wrapper,
                tasks=tasks,
                num_fewshot=0,
                batch_size=1,
                device=str(device),
                limit=limit,
            )

            # Extract and print metrics
            lm_eval_metrics = {}
            for task_name, task_results in results.get("results", {}).items():
                for metric_name, value in task_results.items():
                    if isinstance(value, (int, float)) and not metric_name.endswith(
                        "_stderr"
                    ):
                        key = f"lm_eval/{task_name}_{metric_name}"
                        lm_eval_metrics[key] = value
                        print(f"{task_name}/{metric_name}: {value:.4f}")

            # Log to trackers
            step = getattr(self, "iter_num", None)
            if "trackio" in self.trackers:
                try:
                    import trackio

                    trackio.log(lm_eval_metrics)
                except Exception as e:
                    print(f"Warning: Failed to log lm_eval metrics to trackio: {e}")

            if "wandb" in self.trackers:
                try:
                    import wandb

                    wandb.log(lm_eval_metrics, step=step)
                except Exception as e:
                    print(f"Warning: Failed to log lm_eval metrics to wandb: {e}")

            return lm_eval_metrics

        except Exception as e:
            print(f"lm-eval failed: {e}")
            import traceback

            traceback.print_exc()
            return {}

    def run_dry_run(self, exit_on_completion=True):
        """Run architecture validation."""
        if self.step_config["arch"] == "ra":
            return self._ra_trainer.run_dry_run(exit_on_completion=exit_on_completion)
        else:
            return super().run_dry_run(exit_on_completion=exit_on_completion)


class RAMLACoordinator:
    """Coordinates running multiple RAMLA ablation steps."""

    def __init__(self, args, config, steps):
        self.args = args
        self.config = config
        self.steps = steps

    def run(self):
        """Run all ablation steps sequentially."""
        for step in self.steps:
            print(f"\n{'=' * 80}")
            print(f"Running RAMLA ablation step: {step}")
            print(f"{'=' * 80}\n")

            trainer = RAMLATrainer(self.args, self.config, ablation_step=step)

            if getattr(self.args, "dry_run", False):
                status = trainer.run_dry_run(exit_on_completion=False)
                if status != 0:
                    print(f"✗ Step {step} failed validation")
                    import sys

                    sys.exit(1)
            else:
                trainer.train()

            print(f"\nCompleted RAMLA ablation step: {step}")
