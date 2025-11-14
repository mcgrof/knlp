"""
LeNet-5 Model with Optional PCA Tokenizer Support

Standard LeNet-5 architecture with extensions to support:
- PCA-based input tokenization
- Spline-based trajectory compression
- Hierarchical memory tiering

The model can operate in three modes:
1. Standard: Raw images → LeNet-5 (vanilla)
2. PCA: Raw images → PCA encoder → LeNet-5 (dimensionality reduction)
3. Spline-PCA: Raw images → PCA → Spline trajectory → LeNet-5 (temporal compression)
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.tokenizers import PCAImageTokenizer, SplinePCATokenizer


class LeNet5(nn.Module):
    """
    Standard LeNet-5 architecture for MNIST.

    Architecture:
        Conv1: 1 → 6 channels (5×5 kernel)
        Pool1: MaxPool 2×2
        Conv2: 6 → 16 channels (5×5 kernel)
        Pool2: MaxPool 2×2
        FC1: 400 → 120
        FC2: 120 → 84
        FC3: 84 → 10 (output)
    """

    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


class LeNet5WithPCA(nn.Module):
    """
    LeNet-5 with PCA-based input tokenization.

    Instead of processing raw 28×28 images, this model first projects
    them into PCA space (e.g., 784→64 dimensions), then feeds the
    compressed representation through a modified LeNet-5.

    Benefits:
    - Dimensionality reduction (12x compression for n_components=64)
    - Natural tiering hierarchy (tier by variance explained)
    - Potentially better generalization (compression as regularization)

    Usage:
        model = LeNet5WithPCA(num_classes=10, n_components=64)
        model.fit_pca(train_images)  # Fit PCA on training data
        outputs = model(test_images)  # Uses PCA transform internally
    """

    def __init__(self, num_classes=10, n_components=64, whiten=False):
        """
        Initialize LeNet-5 with PCA tokenizer.

        Args:
            num_classes: Number of output classes
            n_components: Number of PCA components to keep
            whiten: Whether to whiten PCA components
        """
        super(LeNet5WithPCA, self).__init__()
        self.n_components = n_components
        self.tokenizer = PCAImageTokenizer(n_components=n_components, whiten=whiten)

        # Modified architecture: PCA codes → FC layers → output
        # Skip convolutional layers since PCA already extracts features
        self.fc = nn.Linear(n_components, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def fit_pca(self, images):
        """
        Fit PCA tokenizer on training images.

        Args:
            images: Training images, shape [N, H, W] or [N, C, H, W]
        """
        # Convert torch tensor to numpy if needed
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()

        self.tokenizer.fit(images)

    def forward(self, x):
        """
        Forward pass with PCA tokenization.

        Args:
            x: Input images, shape [N, 1, H, W]

        Returns:
            Output logits, shape [N, num_classes]
        """
        # Transform to PCA space
        device = x.device
        x_np = x.cpu().numpy()
        codes = self.tokenizer.transform(x_np)
        codes_tensor = torch.from_numpy(codes).float().to(device)

        # Feed through FC layers
        out = self.fc(codes_tensor)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

    def get_tier_assignments(
        self, hbm_threshold=0.3, cpu_threshold=0.5
    ) -> dict[int, str]:
        """
        Get tier assignments for PCA components.

        Returns:
            Dictionary mapping component index to tier name
        """
        return self.tokenizer.get_tier_assignments(hbm_threshold, cpu_threshold)


class LeNet5WithSplinePCA(nn.Module):
    """
    LeNet-5 with spline-based PCA trajectory tokenization.

    Extends PCA tokenization to track how principal components evolve
    during training and compress trajectories as splines. This enables:

    1. Temporal tiering: Tier by component update frequency
    2. Trajectory compression: Store splines instead of full history
    3. Online learning: Update control points, not full model

    Usage:
        model = LeNet5WithSplinePCA(num_classes=10, n_components=64)
        model.fit_pca(train_images)

        # During training, record trajectories
        for epoch, batch in enumerate(train_loader):
            outputs = model(batch)
            model.record_trajectory(batch, epoch)

        # After training, fit splines
        model.fit_splines()

        # Get tier assignments based on trajectory variance
        tier_assignments = model.get_spline_tier_assignments()
    """

    def __init__(
        self, num_classes=10, n_components=64, n_control_points=8, whiten=False
    ):
        """
        Initialize LeNet-5 with spline-PCA tokenizer.

        Args:
            num_classes: Number of output classes
            n_components: Number of PCA components
            n_control_points: Number of spline control points per component
            whiten: Whether to whiten PCA components
        """
        super(LeNet5WithSplinePCA, self).__init__()
        self.n_components = n_components
        self.tokenizer = SplinePCATokenizer(
            n_components=n_components,
            n_control_points=n_control_points,
            whiten=whiten,
        )

        # Modified architecture: PCA codes → FC layers → output
        self.fc = nn.Linear(n_components, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def fit_pca(self, images):
        """Fit PCA tokenizer on training images."""
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        self.tokenizer.fit(images)

    def forward(self, x):
        """Forward pass with PCA tokenization."""
        device = x.device
        x_np = x.cpu().numpy()
        codes = self.tokenizer.transform(x_np)
        codes_tensor = torch.from_numpy(codes).float().to(device)

        out = self.fc(codes_tensor)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

    def record_trajectory(self, x, step):
        """Record PCA codes at training step for trajectory fitting."""
        x_np = x.cpu().numpy()
        codes = self.tokenizer.transform(x_np)
        self.tokenizer.record_trajectory(codes, step)

    def fit_splines(self):
        """Fit cubic splines to component trajectories."""
        self.tokenizer.fit_splines()

    def get_spline_tier_assignments(
        self, hbm_threshold=0.3, cpu_threshold=0.5
    ) -> dict[int, str]:
        """
        Get tier assignments based on trajectory variance.

        Components with high temporal variance are actively learning
        and stay in fast memory. Stable components are offloaded.
        """
        return self.tokenizer.get_spline_tier_assignments(hbm_threshold, cpu_threshold)


def create_lenet5_model(
    num_classes=10, tokenizer_type="none", n_components=64, n_control_points=8
):
    """
    Factory function to create LeNet-5 model with optional tokenizer.

    Args:
        num_classes: Number of output classes
        tokenizer_type: "none", "pca", or "spline-pca"
        n_components: Number of PCA components (if using tokenizer)
        n_control_points: Number of spline control points (if using spline-pca)

    Returns:
        LeNet-5 model instance
    """
    if tokenizer_type == "none":
        return LeNet5(num_classes=num_classes)
    elif tokenizer_type == "pca":
        return LeNet5WithPCA(num_classes=num_classes, n_components=n_components)
    elif tokenizer_type == "spline-pca":
        return LeNet5WithSplinePCA(
            num_classes=num_classes,
            n_components=n_components,
            n_control_points=n_control_points,
        )
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
