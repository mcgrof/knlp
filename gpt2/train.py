"""
Unified GPT-2 Training Dispatcher

Main entry point for all GPT-2 training.
Dispatches to appropriate trainer based on architecture/mode.

Usage:
    # Vanilla GPT-2
    python gpt2/train.py --architecture vanilla

    # Unified RA single step
    python gpt2/train.py --architecture unified-ra --ra-step V1

    # Ablation study
    python gpt2/train.py --architecture unified-ra --ablation-mode --ablation-steps V0,V1,V3
"""

import os
import sys
import argparse

# Add parent to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import config first (needed for env setup)
try:
    from config import Config

    config = Config()
except ImportError:
    config = None
    print("Warning: config.py not found, using defaults")


def create_argument_parser():
    """Create argument parser with all options."""
    parser = argparse.ArgumentParser(description="Unified GPT-2 Training")

    # Architecture selection
    parser.add_argument(
        "--architecture",
        type=str,
        default="vanilla",
        choices=["vanilla", "unified-ra"],
        help="Training architecture (vanilla=standard GPT-2, unified-ra=V-series RA ablations)",
    )

    # Ablation mode
    parser.add_argument(
        "--ablation-mode",
        action="store_true",
        help="Run ablation study (multiple steps sequentially)",
    )
    parser.add_argument(
        "--ablation-steps",
        type=str,
        default="V0,V1,V3",
        help="Comma-separated ablation steps (e.g., 'V0,V1,V3,V7,V9')",
    )
    parser.add_argument(
        "--ra-step",
        type=str,
        default="V1",
        help="Single RA ablation step to run (when not in ablation mode)",
    )

    # Model configuration
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        help="GPT-2 model size",
    )
    parser.add_argument("--block-size", type=int, default=1024, help="Context length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--bias", action="store_true", default=True, help="Use bias")

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="finewebedu",
        choices=["shakespeare", "finewebedu", "openwebtext"],
        help="Dataset",
    )
    parser.add_argument(
        "--data-dir", type=str, default="./gpt2/data", help="Data directory"
    )

    # Training
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--max-iters", type=int, default=10000, help="Maximum iterations"
    )
    parser.add_argument(
        "--max-time",
        type=int,
        default=0,
        help="Maximum training time in seconds (0=no limit)",
    )

    # Optimizer
    parser.add_argument(
        "--learning-rate", type=float, default=6e-4, help="Learning rate"
    )
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=200, help="Warmup steps")
    parser.add_argument(
        "--decay-lr", action="store_true", default=True, help="Use LR decay"
    )
    parser.add_argument(
        "--min-lr", type=float, default=6e-5, help="Minimum learning rate"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamwspam",
        choices=["sgd", "adam", "adamw", "adamwspam", "adamwprune"],
        help="Optimizer",
    )

    # Pruning (vanilla only)
    parser.add_argument(
        "--pruning-method",
        type=str,
        default="none",
        choices=["none", "magnitude", "movement", "state"],
        help="Pruning method",
    )
    parser.add_argument(
        "--target-sparsity", type=float, default=0.5, help="Target sparsity"
    )
    parser.add_argument(
        "--pruning-warmup", type=int, default=1000, help="Pruning warmup steps"
    )

    # SPAM optimizer config
    parser.add_argument("--spam-theta", type=float, default=50.0, help="SPAM theta")
    parser.add_argument(
        "--spam-interval", type=int, default=1000, help="SPAM reset interval"
    )
    parser.add_argument(
        "--spam-warmup-steps", type=int, default=1000, help="SPAM warmup steps"
    )
    parser.add_argument(
        "--spam-enable-clip", action="store_true", help="Enable SPAM gradient clipping"
    )

    # AdamWPrune config
    parser.add_argument(
        "--adamwprune-variant", type=str, default="bitter0", help="AdamWPrune variant"
    )

    # Logging and evaluation
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval")
    parser.add_argument(
        "--eval-interval", type=int, default=100, help="Evaluation interval"
    )
    parser.add_argument(
        "--eval-samples", type=int, default=200, help="Evaluation samples"
    )

    # Checkpointing
    parser.add_argument(
        "--save-checkpoint", action="store_true", help="Save checkpoints"
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=1000, help="Checkpoint interval"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./output", help="Output directory"
    )

    # System
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile()")
    parser.add_argument(
        "--flash-attention",
        action="store_true",
        help="Enable Flash Attention (ignored, for backward compatibility)",
    )
    parser.add_argument(
        "--json-output", type=str, default="", help="JSON output file for metrics"
    )

    # Tracking
    parser.add_argument(
        "--tracker",
        type=str,
        default="none",
        help="Experiment tracker (none, trackio, wandb)",
    )
    parser.add_argument(
        "--tracker-project", type=str, default="", help="Tracker project name"
    )
    parser.add_argument(
        "--tracker-run-name", type=str, default="", help="Tracker run name"
    )

    # DDP
    parser.add_argument(
        "--ddp-find-unused-params",
        action="store_true",
        default=True,
        help="DDP find unused params",
    )

    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Load config if available
    if config is not None:
        # Config overrides could go here
        pass

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Dispatch to appropriate trainer
    if args.ablation_mode:
        # Ablation study mode
        from gpt2.trainers import AblationCoordinator, RATrainer

        steps = [s.strip() for s in args.ablation_steps.split(",")]
        print(f"Running ablation study with {len(steps)} steps: {steps}")

        coordinator = AblationCoordinator(args, config, steps)
        coordinator.run()

    elif args.architecture == "unified-ra":
        # Single Unified RA run
        from gpt2.trainers import RATrainer

        print(f"Running RA trainer (step {args.ra_step})")
        trainer = RATrainer(args, config, ablation_step=args.ra_step)
        trainer.train()

    else:  # vanilla
        # Standard GPT-2 training
        from gpt2.trainers import VanillaGPT2Trainer

        print("Running Vanilla GPT-2 trainer")
        trainer = VanillaGPT2Trainer(args, config)
        trainer.train()

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
