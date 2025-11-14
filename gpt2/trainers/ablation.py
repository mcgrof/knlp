"""
Ablation Study Coordinator

Manages running multiple ablation steps sequentially or in parallel.
Handles results aggregation and reporting.
"""

from typing import List, Dict
import os


class AblationCoordinator:
    """
    Coordinator for ablation studies.

    Manages:
    - Sequential execution of ablation steps
    - Results collection and aggregation
    - Parallel job support (via external scheduler)
    - Step descriptions and metadata
    """

    def __init__(self, args, config, steps: List[str]):
        """
        Initialize ablation coordinator.

        Args:
            args: Command-line arguments
            config: Config object
            steps: List of ablation steps to run (e.g., ["V0", "V1", "V3"])
        """
        self.args = args
        self.config = config
        self.steps = steps
        self.results = {}

        # Step descriptions (V-series)
        self.step_descriptions = {
            "V0": "Baseline GPT-2 (standard SDPA)",
            "V1": "Unified RA (R=4, folded layout)",
            "V2": "Unified RA + Self-Restart",
            "V3": "Unified RA + R-MLP (basic)",
            "V4": "Unified RA + R-MLP + Mixer",
            "V5": "Unified RA + R-MLP + Gates",
            "V6": "Unified RA + R-MLP + Mixer + Gates",
            "V7": "Unified RA (R=8, higher capacity)",
            "V8": "Unified RA (R=8) + Self-Restart",
            "V9": "Unified RA (R=2, minimal capacity)",
            "V10": "Unified RA + Self-Restart + 6x MLP",
            "V11": "R-MLP delayed activation (75 steps)",
            "V12": "Unified RA delayed activation (75 steps)",
            "V13": "R-MLP golden ratio delayed",
            "V14": "V11 + KV pruning (golden)",
            "V15": "V13 + KV pruning (learned)",
            "V16": "Unified RA (R=4, per-head gates, variance-guided activation)",
            "V17": "R-MLP basic (R_ff=64) + KV pruning + variance-guided",
            "V18": "R-MLP golden (R_ff=1152) + learned KV pruning + variance-guided",
            "V19": "V-only pruning baseline",
        }

    def run(self):
        """
        Run all ablation steps.

        Returns:
            Dictionary with results for each step
        """
        import time

        for i, step in enumerate(self.steps):
            print(f"\n{'='*70}")
            print(f"Ablation step {i+1}/{len(self.steps)}: {step}")
            print(f"Description: {self.step_descriptions.get(step, 'Unknown')}")
            print(f"{'='*70}\n")

            step_start_time = time.time()

            # Create trainer for this step
            from .ra import RATrainer

            trainer = RATrainer(self.args, self.config, ablation_step=step)

            # Run training or dry-run
            if getattr(self.args, "dry_run", False):
                trainer.run_dry_run()
            else:
                trainer.train()

            # Collect results
            step_time = time.time() - step_start_time
            self.results[step] = {
                "best_val_loss": trainer.best_val_loss,
                "training_time": step_time,
                "final_iter": trainer.iter_num,
            }

            print(f"\nStep {step} complete in {step_time/60:.2f} minutes")
            print(f"Best val loss: {trainer.best_val_loss:.4f}\n")

        # Print summary
        print(f"\n{'='*70}")
        print("ABLATION STUDY COMPLETE")
        print(f"{'='*70}")
        for step, result in self.results.items():
            print(
                f"{step}: val_loss={result['best_val_loss']:.4f}, "
                f"time={result['training_time']/60:.1f}min"
            )
        print(f"{'='*70}\n")

        return self.results

    def save_results(self, output_dir: str):
        """
        Save aggregated results.

        Args:
            output_dir: Directory to save results
        """
        # TODO: Implement results saving
        pass
