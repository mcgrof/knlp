"""
Base GPT-2 Trainer

Provides common functionality for all GPT-2 training variants.
Subclasses implement specific architecture initialization and
training loop customization.
"""

import os
import sys
import time
import math
import json
from typing import Dict, Tuple, Optional
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)

from gpt2.model import GPT, GPTConfig


class BaseGPT2Trainer:
    """
    Base trainer for GPT-2 variants.

    Provides common functionality:
    - Data loading (get_batch)
    - Learning rate scheduling
    - Loss estimation
    - Checkpoint management
    - DDP setup
    - Metrics tracking

    Subclasses override:
    - create_model(): Model initialization
    - create_optimizer(): Optimizer setup
    - train_step(): Single training iteration
    - should_save_checkpoint(): Custom checkpoint logic
    """

    def __init__(self, args, config):
        """
        Initialize base trainer.

        Args:
            args: Parsed command-line arguments
            config: Config object from config.py
        """
        self.args = args
        self.config = config

        # Device setup
        self.setup_device()

        # DDP setup (if applicable)
        self.setup_ddp()

        # Data loading
        self.setup_data()

        # Model and optimizer (to be created by subclass)
        self.model = None
        self.optimizer = None
        self.scaler = None

        # Training state
        self.iter_num = 0
        self.best_val_loss = float('inf')

        # Metrics
        self.metrics = {}

    def setup_device(self):
        """Setup device (CPU/CUDA) and dtype."""
        self.device = self.args.device
        self.dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

        # Set up precision context
        self.ptdtype = {
            'float32': torch.float32,
            'bfloat16': torch.bfloat16,
            'float16': torch.float16
        }[self.dtype]
        self.ctx = nullcontext() if self.device == 'cpu' else torch.amp.autocast(device_type=self.device, dtype=self.ptdtype)

    def setup_ddp(self):
        """Setup Distributed Data Parallel if applicable."""
        self.ddp = int(os.environ.get('RANK', -1)) != -1
        if self.ddp:
            init_process_group(backend='nccl')
            self.ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.ddp_world_size = int(os.environ['WORLD_SIZE'])
            self.device = f'cuda:{self.ddp_local_rank}'
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0
            self.seed_offset = self.ddp_rank
            assert self.args.gradient_accumulation_steps % self.ddp_world_size == 0
            self.args.gradient_accumulation_steps //= self.ddp_world_size
        else:
            self.master_process = True
            self.seed_offset = 0
            self.ddp_world_size = 1

    def setup_data(self):
        """Setup data loading (to be implemented by subclass if needed)."""
        pass

    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch of data.

        Args:
            split: 'train' or 'val'

        Returns:
            (x, y) tensors
        """
        # Load data
        if split == "train":
            data_path = os.path.join(self.args.data_dir, self.args.dataset, "train.bin")
        else:
            data_path = os.path.join(self.args.data_dir, self.args.dataset, "val.bin")

        data = np.memmap(data_path, dtype=np.uint16, mode="r")

        # Generate random positions
        ix = torch.randint(len(data) - self.args.block_size, (self.args.batch_size,))

        # Create batch
        x = torch.stack(
            [torch.from_numpy((data[i : i + self.args.block_size]).astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [
                torch.from_numpy((data[i + 1 : i + 1 + self.args.block_size]).astype(np.int64))
                for i in ix
            ]
        )

        if self.device == "cuda":
            # Pin arrays for async transfer
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(
                self.device, non_blocking=True
            )
        else:
            x, y = x.to(self.device), y.to(self.device)

        return x, y

    def get_lr(self, it: int) -> float:
        """
        Compute learning rate for iteration.

        Args:
            it: Current iteration number

        Returns:
            Learning rate
        """
        # Linear warmup
        if it < self.args.warmup_steps:
            return self.args.learning_rate * it / self.args.warmup_steps
        # Cosine decay
        if it > self.args.max_iters:
            return self.args.min_lr
        decay_ratio = (it - self.args.warmup_steps) / (self.args.max_iters - self.args.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.args.min_lr + coeff * (self.args.learning_rate - self.args.min_lr)

    @torch.no_grad()
    def estimate_loss(self) -> Dict[str, float]:
        """
        Estimate loss on train and val splits.

        Returns:
            Dictionary with 'train' and 'val' losses
        """
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.args.eval_samples)
            for k in range(self.args.eval_samples):
                X, Y = self.get_batch(split)
                with self.ctx:
                    logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def save_checkpoint(self, checkpoint_path: str):
        """
        Save training checkpoint.

        Args:
            checkpoint_path: Path to save checkpoint
        """
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_num': self.iter_num,
            'best_val_loss': self.best_val_loss,
            'config': self.args,
        }
        if self.scaler is not None:
            checkpoint['scaler'] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_num = checkpoint['iter_num']
        self.best_val_loss = checkpoint['best_val_loss']
        if 'scaler' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler'])

    # Abstract methods to be implemented by subclasses

    def create_model(self) -> nn.Module:
        """Create and return the model. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement create_model()")

    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create and return the optimizer. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement create_optimizer()")

    def train_step(self, X: torch.Tensor, Y: torch.Tensor) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            X: Input tensor
            Y: Target tensor

        Returns:
            Dictionary with metrics (at minimum 'loss')
        """
        raise NotImplementedError("Subclass must implement train_step()")

    def should_save_checkpoint(self) -> bool:
        """
        Determine if checkpoint should be saved at current iteration.

        Returns:
            True if checkpoint should be saved
        """
        return self.iter_num % self.args.checkpoint_interval == 0
