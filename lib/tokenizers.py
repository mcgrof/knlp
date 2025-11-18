"""
Image Tokenizers for Hierarchical Tiering

Implements PCA-based and spline-based tokenization strategies for
image inputs. Designed to work with hierarchical memory tiering by
creating natural importance hierarchies in the representation space.

Key idea: Not all dimensions are equally important. Compress images
into principal components, then tier components by variance explained
or by spline coefficient importance.
"""

import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import IncrementalPCA


class PCAImageTokenizer:
    """
    PCA-based compression for image inputs.

    Reduces image dimensionality while preserving most variance.
    Creates natural tiering hierarchy: top components capture more
    variance and should stay in fast memory.

    Example for MNIST:
        Original: 28×28 = 784 dimensions
        PCA(n=64): Compress to 64 dimensions (captures ~95% variance)
        Top 16 components: 80% variance → keep hot
        Next 32 components: 15% variance → warm tier
        Last 16 components: 5% variance → cold tier
    """

    def __init__(self, n_components: int = 64, whiten: bool = False):
        """
        Initialize PCA tokenizer.

        Args:
            n_components: Number of principal components to keep
            whiten: Whether to whiten components (normalize variance)
        """
        self.n_components = n_components
        self.whiten = whiten
        self._ipca = IncrementalPCA(n_components=n_components, whiten=whiten)
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.fitted = False

    def fit(self, images: np.ndarray) -> "PCAImageTokenizer":
        """
        Fit PCA on training images.

        Args:
            images: Training images, shape [N, H, W] or [N, C, H, W]

        Returns:
            self (for chaining)
        """
        # Flatten images to 2D
        if images.ndim == 4:  # [N, C, H, W]
            N, C, H, W = images.shape
            flat_images = images.reshape(N, C * H * W)
        elif images.ndim == 3:  # [N, H, W]
            N, H, W = images.shape
            flat_images = images.reshape(N, H * W)
        else:
            flat_images = images

        # Center data
        self.mean_ = np.mean(flat_images, axis=0)
        centered = flat_images - self.mean_

        # Compute covariance matrix and eigenvectors
        # Use SVD for numerical stability
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        # Components are rows of Vt (right singular vectors)
        self.components_ = Vt[: self.n_components]

        # Explained variance
        n_samples = flat_images.shape[0]
        self.explained_variance_ = (S**2) / (n_samples - 1)
        self.explained_variance_ = self.explained_variance_[: self.n_components]

        total_var = np.sum((S**2) / (n_samples - 1))
        self.explained_variance_ratio_ = self.explained_variance_ / total_var

        # Whitening transform (normalize by std dev)
        if self.whiten:
            self.components_ /= np.sqrt(self.explained_variance_[:, np.newaxis])

        self.fitted = True
        return self

    def partial_fit(self, images: np.ndarray) -> "PCAImageTokenizer":
        """
        Incrementally fit PCA on a batch of images.

        Use this for memory-efficient fitting on large datasets.
        Call multiple times with different batches.

        Args:
            images: Batch of images, shape [N, H, W] or [N, C, H, W]

        Returns:
            self (for chaining)
        """
        # Flatten images to 2D
        if images.ndim == 4:  # [N, C, H, W]
            N, C, H, W = images.shape
            flat_images = images.reshape(N, C * H * W)
        elif images.ndim == 3:  # [N, H, W]
            N, H, W = images.shape
            flat_images = images.reshape(N, H * W)
        else:
            flat_images = images

        # Fit incrementally using sklearn
        self._ipca.partial_fit(flat_images)

        # Update our attributes to match sklearn's state
        self.mean_ = self._ipca.mean_
        self.components_ = self._ipca.components_
        self.explained_variance_ = self._ipca.explained_variance_
        self.explained_variance_ratio_ = self._ipca.explained_variance_ratio_
        self.fitted = True

        return self

    def transform(self, images: np.ndarray) -> np.ndarray:
        """
        Transform images to PCA space.

        Args:
            images: Images to transform, same shape as fit()

        Returns:
            PCA codes, shape [N, n_components]
        """
        if not self.fitted:
            raise RuntimeError("Must call fit() before transform()")

        # Flatten images
        if images.ndim == 4:  # [N, C, H, W]
            N, C, H, W = images.shape
            flat_images = images.reshape(N, C * H * W)
        elif images.ndim == 3:  # [N, H, W]
            N, H, W = images.shape
            flat_images = images.reshape(N, H * W)
        else:
            flat_images = images

        # Center and project
        centered = flat_images - self.mean_
        codes = np.dot(centered, self.components_.T)

        return codes

    def inverse_transform(self, codes: np.ndarray) -> np.ndarray:
        """
        Reconstruct images from PCA codes.

        Args:
            codes: PCA codes, shape [N, n_components]

        Returns:
            Reconstructed images, shape [N, D] where D is original dimension
        """
        if not self.fitted:
            raise RuntimeError("Must call fit() before inverse_transform()")

        # Project back and add mean
        reconstructed = np.dot(codes, self.components_) + self.mean_

        return reconstructed

    def get_component_importance(self) -> Dict[int, float]:
        """
        Get importance score for each component (variance explained).

        Returns:
            Dictionary mapping component index to importance score
        """
        if not self.fitted:
            raise RuntimeError("Must call fit() first")

        return {i: var for i, var in enumerate(self.explained_variance_ratio_)}

    def get_tier_assignments(
        self, hbm_threshold: float = 0.3, cpu_threshold: float = 0.5
    ) -> Dict[int, str]:
        """
        Assign tiers to PCA components based on variance explained.

        Args:
            hbm_threshold: Fraction of components to keep in HBM (hot tier)
            cpu_threshold: Fraction to keep in CPU tier (warm tier)

        Returns:
            Dictionary mapping component index to tier name
        """
        if not self.fitted:
            raise RuntimeError("Must call fit() first")

        n_hbm = int(self.n_components * hbm_threshold)
        n_cpu = int(self.n_components * cpu_threshold)

        tier_assignments = {}
        for i in range(self.n_components):
            if i < n_hbm:
                tier_assignments[i] = "HBM"
            elif i < n_cpu:
                tier_assignments[i] = "CPU"
            else:
                tier_assignments[i] = "SSD"

        return tier_assignments

    def save(self, path: str):
        """Save tokenizer state to disk."""
        state = {
            "n_components": self.n_components,
            "whiten": self.whiten,
            "mean_": self.mean_,
            "components_": self.components_,
            "explained_variance_": self.explained_variance_,
            "explained_variance_ratio_": self.explained_variance_ratio_,
            "fitted": self.fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> "PCAImageTokenizer":
        """Load tokenizer state from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)

        tokenizer = cls(n_components=state["n_components"], whiten=state["whiten"])
        tokenizer.mean_ = state["mean_"]
        tokenizer.components_ = state["components_"]
        tokenizer.explained_variance_ = state["explained_variance_"]
        tokenizer.explained_variance_ratio_ = state["explained_variance_ratio_"]
        tokenizer.fitted = state["fitted"]

        return tokenizer


class SplinePCATokenizer(PCAImageTokenizer):
    """
    PCA tokenizer with spline-based trajectory representation.

    Extends PCA tokenizer to track how principal components evolve
    during training and represent trajectories as splines (control
    points + coefficients). This enables:

    1. Temporal tiering: Frequently updated components stay hot
    2. Trajectory compression: Store splines instead of full history
    3. Online learning: Update control points, not full model

    Example:
        tokenizer = SplinePCATokenizer(n_components=64, n_control_points=8)
        tokenizer.fit(train_images)

        # During training, track component evolution
        for epoch in range(epochs):
            codes = tokenizer.transform(batch_images)
            tokenizer.record_trajectory(codes, epoch)

        # Fit splines to trajectories
        tokenizer.fit_splines()

        # Get tier assignments based on trajectory variance
        tier_assignments = tokenizer.get_spline_tier_assignments()
    """

    def __init__(
        self, n_components: int = 64, n_control_points: int = 8, whiten: bool = False
    ):
        """
        Initialize spline-PCA tokenizer.

        Args:
            n_components: Number of principal components
            n_control_points: Number of spline control points per component
            whiten: Whether to whiten components
        """
        super().__init__(n_components=n_components, whiten=whiten)
        self.n_control_points = n_control_points
        self.trajectories = []  # History of PCA codes over training
        self.spline_coeffs = None
        self.spline_fitted = False

    def record_trajectory(self, codes: np.ndarray, step: int):
        """
        Record PCA codes at a training step.

        Args:
            codes: PCA codes for batch, shape [N, n_components]
            step: Training step number
        """
        # Store mean codes for this step
        mean_codes = np.mean(codes, axis=0)
        self.trajectories.append((step, mean_codes))

    def fit_splines(self):
        """
        Fit cubic splines to component trajectories.

        After training, this compresses the trajectory history into
        a compact spline representation with control points.
        """
        if not self.trajectories:
            raise RuntimeError("No trajectory data recorded")

        from scipy.interpolate import UnivariateSpline

        # Extract steps and codes
        steps = np.array([s for s, _ in self.trajectories])
        codes = np.array([c for _, c in self.trajectories])  # [n_steps, n_components]

        # Fit spline for each component
        self.spline_coeffs = []
        for i in range(self.n_components):
            component_trajectory = codes[:, i]

            # Fit cubic spline with specified number of knots
            spline = UnivariateSpline(steps, component_trajectory, k=3)

            # Extract spline representation
            self.spline_coeffs.append(
                {
                    "knots": spline.get_knots(),
                    "coeffs": spline.get_coeffs(),
                    "degree": 3,
                }
            )

        self.spline_fitted = True

    def get_trajectory_variance(self) -> Dict[int, float]:
        """
        Compute variance of each component's trajectory.

        Components with high temporal variance are actively learning
        and should stay in fast memory. Low-variance components are
        stable and can be offloaded.

        Returns:
            Dictionary mapping component index to trajectory variance
        """
        if not self.trajectories:
            raise RuntimeError("No trajectory data recorded")

        # Extract codes
        codes = np.array([c for _, c in self.trajectories])  # [n_steps, n_components]

        # Compute variance over time for each component
        variances = np.var(codes, axis=0)

        return {i: var for i, var in enumerate(variances)}

    def get_spline_tier_assignments(
        self, hbm_threshold: float = 0.3, cpu_threshold: float = 0.5
    ) -> Dict[int, str]:
        """
        Assign tiers based on trajectory variance (not just explained variance).

        High trajectory variance = actively learning = keep hot
        Low trajectory variance = stable = offload cold

        Args:
            hbm_threshold: Fraction of components to keep in HBM
            cpu_threshold: Fraction to keep in CPU tier

        Returns:
            Dictionary mapping component index to tier name
        """
        # Get trajectory variances
        traj_variances = self.get_trajectory_variance()

        # Sort components by trajectory variance (descending)
        sorted_comps = sorted(traj_variances.items(), key=lambda x: x[1], reverse=True)

        # Assign tiers by percentile
        n_hbm = int(self.n_components * hbm_threshold)
        n_cpu = int(self.n_components * cpu_threshold)

        tier_assignments = {}
        for rank, (comp_idx, var) in enumerate(sorted_comps):
            if rank < n_hbm:
                tier_assignments[comp_idx] = "HBM"
            elif rank < n_cpu:
                tier_assignments[comp_idx] = "CPU"
            else:
                tier_assignments[comp_idx] = "SSD"

        return tier_assignments

    def save(self, path: str):
        """Save tokenizer and spline state to disk."""
        state = {
            "n_components": self.n_components,
            "n_control_points": self.n_control_points,
            "whiten": self.whiten,
            "mean_": self.mean_,
            "components_": self.components_,
            "explained_variance_": self.explained_variance_,
            "explained_variance_ratio_": self.explained_variance_ratio_,
            "fitted": self.fitted,
            "trajectories": self.trajectories,
            "spline_coeffs": self.spline_coeffs,
            "spline_fitted": self.spline_fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> "SplinePCATokenizer":
        """Load tokenizer and spline state from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)

        tokenizer = cls(
            n_components=state["n_components"],
            n_control_points=state["n_control_points"],
            whiten=state["whiten"],
        )
        tokenizer.mean_ = state["mean_"]
        tokenizer.components_ = state["components_"]
        tokenizer.explained_variance_ = state["explained_variance_"]
        tokenizer.explained_variance_ratio_ = state["explained_variance_ratio_"]
        tokenizer.fitted = state["fitted"]
        tokenizer.trajectories = state.get("trajectories", [])
        tokenizer.spline_coeffs = state.get("spline_coeffs", None)
        tokenizer.spline_fitted = state.get("spline_fitted", False)

        return tokenizer
