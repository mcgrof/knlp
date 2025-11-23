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
from gpt2.trainers.ra import RATrainer


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

    # Determine learning rate from suffix
    if step.endswith("1"):
        lr = LR_AGGRESSIVE
        lr_name = "aggressive"
        base = step[:-1]
    elif step.endswith("0"):
        lr = LR_STANDARD
        lr_name = "standard"
        base = step[:-1]
    else:
        raise ValueError(f"Step must end with 0 or 1: {step}")

    # Determine architecture from prefix
    if base == "B":
        arch = "baseline"
    elif base == "MLA":
        arch = "mla"
    elif base == "RA":
        arch = "ra"
    elif base == "RAMLA":
        arch = "ramla"
    elif base == "RAMLAKV":
        arch = "ramlakv"
    elif base == "SBA":
        arch = "sba"
    elif base == "SBASS":
        arch = "sba_ss"  # SBA with shared+skew
    elif base == "SBAKV":
        arch = "sba_kv"  # SBA with K=V tying
    else:
        raise ValueError(f"Unknown architecture prefix: {base}")

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

        elif self.step_config["arch"] in ["mla", "ramla", "ramlakv", "sba", "sba_ss", "sba_kv"]:
            # MLA-based architectures (including SBA variants)
            super().__init__(args, config)
            # Replace model with MLA/SBA variant
            self._setup_mla_model()

        else:
            raise ValueError(f"Unknown architecture: {self.step_config['arch']}")

    def _setup_mla_model(self):
        """Replace model with MLA/RAMLA/RAMLAKV/SBA variant."""
        import torch
        from ra import RA_MLA_Config, MLAGPT, RAMLAGPT, RAMLAKV_GPT, SBAGPT

        arch = self.step_config["arch"]

        # Get MLA config from args or defaults
        d_latent = getattr(self.args, "mla_d_latent", 256)
        compression_ratio = getattr(self.args, "mla_compression_ratio", 0.5)

        cfg = RA_MLA_Config(
            d_model=768,
            n_heads=12,
            head_dim=64,
            d_latent=d_latent,
            block_size=self.args.block_size,
            n_layers=12,
        )

        # Create appropriate full GPT model
        if arch == "mla":
            self.model = MLAGPT(cfg)
            print(f"Created MLAGPT with d_latent={d_latent}")
        elif arch == "ramla":
            self.model = RAMLAGPT(cfg)
            print(f"Created RAMLAGPT with d_latent={d_latent}")
            print(
                f"  Layer directions: {self.model.get_alternation_distribution()[:3].tolist()}..."
            )
        elif arch == "ramlakv":
            self.model = RAMLAKV_GPT(cfg, compression_ratio=compression_ratio)
            print(f"Created RAMLAKV_GPT")
            print(f"  Compression: {self.model.get_compression_stats()}")
        elif arch == "sba":
            self.model = SBAGPT(cfg, kv_mode="separate")
            print(f"Created SBAGPT with d_latent={d_latent}, kv_mode=separate")
            print(f"  Alpha distribution: {self.model.get_alpha_distribution()[:3].tolist()}...")
        elif arch == "sba_ss":
            self.model = SBAGPT(cfg, kv_mode="shared_skew")
            print(f"Created SBAGPT with d_latent={d_latent}, kv_mode=shared_skew")
            print(f"  Alpha distribution: {self.model.get_alpha_distribution()[:3].tolist()}...")
        elif arch == "sba_kv":
            self.model = SBAGPT(cfg, kv_mode="k_eq_v")
            print(f"Created SBAGPT with d_latent={d_latent}, kv_mode=k_eq_v")
            print(f"  Alpha distribution: {self.model.get_alpha_distribution()[:3].tolist()}...")

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
            # For RAMLAKV, set up periodic metrics logging
            if self.step_config["arch"] == "ramlakv":
                self._setup_kvsplice_logging()

            # Use base trainer
            result = super().train()

            # Log final metrics based on architecture
            if self.step_config["arch"] == "ramlakv":
                self._log_kvsplice_metrics()
            elif self.step_config["arch"] in ["sba", "sba_ss", "sba_kv"]:
                self._log_sba_metrics()

            # Run lm-eval if requested
            if getattr(self.args, "run_lm_eval", False):
                self._run_lm_eval()

            return result

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
        print(f"  Compression ratio: {metrics.get('kvsplice/compression_ratio', 'N/A')}")
        print(f"  Memory reduction: {metrics.get('kvsplice/memory_reduction_pct', 'N/A'):.1f}%")
        if "kvsplice/avg_reconstruction_error" in metrics:
            print(f"  Avg reconstruction error: {metrics['kvsplice/avg_reconstruction_error']:.6f}")
        print(f"  Reciprocal layers: {metrics.get('kvsplice/reciprocal_layers', 'N/A')}")
        print(f"  Standard layers: {metrics.get('kvsplice/standard_layers', 'N/A')}")

        # Log to trackers
        if hasattr(self, "trackers") and self.trackers:
            for tracker in self.trackers:
                if hasattr(tracker, "log"):
                    tracker.log(metrics)

    def _log_sba_metrics(self):
        """Log SBA metrics to trackers."""
        if not hasattr(self.model, "get_sba_metrics"):
            return

        metrics = self.model.get_sba_metrics()

        # Print summary
        print("\n--- SBA Attention Metrics ---")
        print(f"  Alpha mean: {metrics.get('sba/alpha_mean', 'N/A'):.3f}")
        print(f"  Alpha std: {metrics.get('sba/alpha_std', 'N/A'):.3f}")
        print(f"  Forward-dominant layers: {metrics.get('sba/forward_dominant_layers', 'N/A')}")
        print(f"  Reverse-dominant layers: {metrics.get('sba/reverse_dominant_layers', 'N/A')}")
        print(f"  Mixed layers: {metrics.get('sba/mixed_layers', 'N/A')}")

        # Log to trackers
        if hasattr(self, "trackers") and self.trackers:
            for tracker in self.trackers:
                if hasattr(tracker, "log"):
                    tracker.log(metrics)

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

            # Log to trackers if available
            if hasattr(self, 'trackers') and self.trackers:
                for tracker in self.trackers:
                    if hasattr(tracker, 'log'):
                        tracker.log(lm_eval_metrics)

            return lm_eval_metrics

        except Exception as e:
            print(f"lm-eval failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def run_dry_run(self):
        """Run architecture validation."""
        if self.step_config["arch"] == "ra":
            return self._ra_trainer.run_dry_run()
        else:
            return super().run_dry_run()


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
                trainer.run_dry_run()
            else:
                trainer.train()

            print(f"\nCompleted RAMLA ablation step: {step}")
