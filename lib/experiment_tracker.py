# SPDX-License-Identifier: MIT
"""
Generalized experiment tracking for W&B and Trackio.

Provides a unified interface for initializing and logging to multiple
experiment tracking backends simultaneously.
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Unified interface for experiment tracking with W&B and/or Trackio."""

    def __init__(
        self,
        tracker_list: str = "",
        project: str = "training",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        resume: str = "allow",
    ):
        """
        Initialize experiment trackers.

        Args:
            tracker_list: Comma-separated list of trackers (wandb,trackio)
            project: Project name for tracking
            run_name: Optional custom run name
            config: Configuration dictionary to log
            resume: Resume mode (allow, must, never)
        """
        self.enabled_trackers: List[str] = []
        self.wandb_run = None
        self.trackio_active = False

        if not tracker_list:
            logger.info("No experiment trackers enabled")
            return

        trackers = [t.strip().lower() for t in tracker_list.split(",")]
        config = config or {}

        # Initialize W&B
        if "wandb" in trackers:
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=project,
                    name=run_name,
                    config=config,
                    resume=resume,
                )
                self.enabled_trackers.append("wandb")
                logger.info(f"Initialized WandB tracking for project: {project}")
            except ImportError:
                logger.warning("wandb not installed, skipping W&B tracking")
            except Exception as e:
                logger.warning(f"Failed to initialize W&B: {e}")

        # Initialize Trackio
        if "trackio" in trackers:
            try:
                import trackio

                trackio.init(
                    project=project,
                    config=config,
                    resume=resume,
                )
                self.trackio_active = True
                self.enabled_trackers.append("trackio")
                logger.info(f"Initialized Trackio tracking for project: {project}")
            except ImportError:
                logger.warning("trackio not installed, skipping Trackio tracking")
            except Exception as e:
                logger.warning(f"Failed to initialize Trackio: {e}")

        if self.enabled_trackers:
            logger.info(
                f"Active experiment trackers: {', '.join(self.enabled_trackers)}"
            )

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics to all enabled trackers.

        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number for metrics
        """
        if not self.enabled_trackers:
            return

        # Log to W&B
        if self.wandb_run is not None:
            try:
                import wandb

                if step is not None:
                    wandb.log(metrics, step=step)
                else:
                    wandb.log(metrics)
            except Exception as e:
                logger.warning(f"Failed to log to W&B: {e}")

        # Log to Trackio
        if self.trackio_active:
            try:
                import trackio

                trackio.log(metrics)
            except Exception as e:
                logger.warning(f"Failed to log to Trackio: {e}")

    def finish(self):
        """Finish tracking and close all trackers."""
        if self.wandb_run is not None:
            try:
                import wandb

                wandb.finish()
                logger.info("Closed W&B run")
            except Exception as e:
                logger.warning(f"Failed to finish W&B: {e}")

        if self.trackio_active:
            try:
                import trackio

                trackio.finish()
                logger.info("Closed Trackio run")
            except Exception as e:
                logger.warning(f"Failed to finish Trackio: {e}")

    def is_enabled(self) -> bool:
        """Check if any trackers are enabled."""
        return len(self.enabled_trackers) > 0

    @property
    def trackers(self) -> List[str]:
        """Get list of enabled tracker names."""
        return self.enabled_trackers.copy()
