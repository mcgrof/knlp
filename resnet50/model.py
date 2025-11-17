#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ResNet-50 model implementation for ImageNet."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.tokenizers import PCAImageTokenizer, SplinePCATokenizer


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet-50/101/152."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet50(nn.Module):
    """ResNet-50 model."""

    def __init__(self, num_classes=1000):
        super(ResNet50, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet-50 layers: [3, 4, 6, 3]
        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def resnet50(num_classes=1000):
    """Create ResNet-50 model."""
    return ResNet50(num_classes=num_classes)


class ResNet50WithPCA(nn.Module):
    """
    ResNet-50 with PCA-based input tokenization.

    Instead of processing raw 32×32×3 CIFAR-100 images directly,
    this model first projects them into PCA space (e.g., 3072→256
    dimensions), then feeds the compressed representation through
    fully connected layers before the standard ResNet blocks.

    Benefits:
    - Dimensionality reduction (~12x compression for n_components=256)
    - Natural tiering hierarchy (tier by variance explained)
    - Memory reduction potential

    Usage:
        model = ResNet50WithPCA(num_classes=100, n_components=256)
        model.fit_pca(train_images)
        outputs = model(test_images)
    """

    def __init__(self, num_classes=100, n_components=256, whiten=False):
        """
        Initialize ResNet-50 with PCA tokenizer.

        Args:
            num_classes: Number of output classes
            n_components: Number of PCA components to keep
            whiten: Whether to whiten PCA components
        """
        super(ResNet50WithPCA, self).__init__()
        self.n_components = n_components
        self.tokenizer = PCAImageTokenizer(n_components=n_components, whiten=whiten)

        # PCA expansion layer to match ResNet input channels
        # Transform PCA codes back to 3-channel representation
        self.pca_expand = nn.Sequential(nn.Linear(n_components, 3 * 32 * 32), nn.ReLU())

        # Standard ResNet-50
        self.resnet = ResNet50(num_classes=num_classes)

    def fit_pca(self, images):
        """
        Fit PCA tokenizer on training images.

        Args:
            images: Training images, shape [N, C, H, W]
        """
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        self.tokenizer.fit(images)

    def forward(self, x):
        """
        Forward pass with PCA tokenization.

        Args:
            x: Input images, shape [N, 3, H, W]

        Returns:
            Output logits, shape [N, num_classes]
        """
        # Transform to PCA space
        device = x.device
        batch_size = x.shape[0]
        x_np = x.cpu().numpy()
        codes = self.tokenizer.transform(x_np)
        codes_tensor = torch.from_numpy(codes).float().to(device)

        # Expand back to image dimensions
        x_expanded = self.pca_expand(codes_tensor)
        x_expanded = x_expanded.view(batch_size, 3, 32, 32)

        # Feed through ResNet
        return self.resnet(x_expanded)

    def get_tier_assignments(
        self, hbm_threshold=0.3, cpu_threshold=0.5
    ) -> dict[int, str]:
        """Get tier assignments for PCA components."""
        return self.tokenizer.get_tier_assignments(hbm_threshold, cpu_threshold)


class ResNet50WithSplinePCA(nn.Module):
    """
    ResNet-50 with spline-based PCA trajectory tokenization.

    Extends PCA tokenization to track how principal components evolve
    during training and compress trajectories as splines.

    Usage:
        model = ResNet50WithSplinePCA(num_classes=100, n_components=256)
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
        self, num_classes=100, n_components=256, n_control_points=8, whiten=False
    ):
        """
        Initialize ResNet-50 with spline-PCA tokenizer.

        Args:
            num_classes: Number of output classes
            n_components: Number of PCA components
            n_control_points: Number of spline control points per component
            whiten: Whether to whiten PCA components
        """
        super(ResNet50WithSplinePCA, self).__init__()
        self.n_components = n_components
        self.tokenizer = SplinePCATokenizer(
            n_components=n_components,
            n_control_points=n_control_points,
            whiten=whiten,
        )

        # PCA expansion layer
        self.pca_expand = nn.Sequential(nn.Linear(n_components, 3 * 32 * 32), nn.ReLU())

        # Standard ResNet-50
        self.resnet = ResNet50(num_classes=num_classes)

    def fit_pca(self, images):
        """Fit PCA tokenizer on training images."""
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        self.tokenizer.fit(images)

    def forward(self, x):
        """Forward pass with PCA tokenization."""
        device = x.device
        batch_size = x.shape[0]
        x_np = x.cpu().numpy()
        codes = self.tokenizer.transform(x_np)
        codes_tensor = torch.from_numpy(codes).float().to(device)

        # Expand back to image dimensions
        x_expanded = self.pca_expand(codes_tensor)
        x_expanded = x_expanded.view(batch_size, 3, 32, 32)

        return self.resnet(x_expanded)

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
        """Get tier assignments based on trajectory variance."""
        return self.tokenizer.get_spline_tier_assignments(hbm_threshold, cpu_threshold)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = resnet50(num_classes=1000)
    print(f"ResNet-50 parameters: {count_parameters(model):,}")

    # Test forward pass with ImageNet-sized input
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
