"""
Vanilla GPT-2 Trainer

Standard GPT-2 training without architectural modifications.
Supports AdamW, AdamWSPAM, and AdamWPrune optimizers.
"""

from .base import BaseGPT2Trainer


class VanillaGPT2Trainer(BaseGPT2Trainer):
    """
    Trainer for standard GPT-2 models.

    Implements:
    - Standard GPT-2 model initialization
    - Optimizer setup (AdamW, AdamWSPAM, AdamWPrune)
    - Standard training loop
    - Pruning evaluation (when using AdamWPrune)
    """

    def __init__(self, args, config):
        super().__init__(args, config)
        # TODO: Implement vanilla trainer initialization

    def create_model(self):
        """Create standard GPT-2 model."""
        # TODO: Extract from train.py
        pass

    def create_optimizer(self):
        """Create optimizer (AdamW, AdamWSPAM, or AdamWPrune)."""
        # TODO: Extract from train.py
        pass

    def train_step(self, X, Y):
        """Perform single training step."""
        # TODO: Extract from train.py
        pass

    def setup_data(self):
        """Setup data loading."""
        # TODO: Extract from train.py
        pass

    def get_batch(self, split):
        """Get batch of data."""
        # TODO: Extract from train.py
        pass
