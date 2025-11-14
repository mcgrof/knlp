"""
GPT-2 Training Module

Provides trainer classes for different GPT-2 variants:
- VanillaGPT2Trainer: Standard GPT-2 training
- UnifiedRATrainer: RA (V-series) ablation studies

All trainers inherit from BaseGPT2Trainer which provides common
functionality like data loading, checkpointing, and DDP setup.
"""

from .base import BaseGPT2Trainer
from .vanilla import VanillaGPT2Trainer
from .ra import RATrainer
from .ablation import AblationCoordinator

__all__ = [
    "BaseGPT2Trainer",
    "VanillaGPT2Trainer",
    "RATrainer",
    "AblationCoordinator",
]
