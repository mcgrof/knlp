# SPDX-License-Identifier: GPL-2.0
"""Sequence-axis (temporal) transforms for sealed KV blocks.

A sealed block holds B consecutive tokens of one layer/head as a
matrix X of shape [B, d] (tokens along dim -2, head dims along -1).
Every transform here acts along the token axis:

    Y = R @ X

where R is [B, B]. Orthogonal transforms preserve the Frobenius norm
and admit the transformed-attention identity

    q @ K.T == (q @ K_tilde.T) @ R        (K_tilde = R @ K)
    a @ V   == (a @ R.T) @ V_tilde        (V_tilde = R @ V)

so attention can consume transformed blocks without reconstructing
the original token vectors. The anchor-delta baseline (CacheGen
style) is affine, not orthogonal, and is included only as a
prior-art baseline.

Transforms are deterministic given their configuration. Data-adaptive
transforms (per-block PCA, power-iteration Householder) record the
per-block basis metadata they would have to ship in an artifact via
``metadata_floats()``.
"""

from __future__ import annotations

import math

import torch


def householder_vector(u):
    """Stable Householder vector v with H(v) @ u = sign * e0.

    H(v) = I - 2 v v^T / (v^T v). Uses the LAPACK sign choice
    v = u + sign(u0) * e0 so the construction never cancels:
    |v[0]| = |u0| + 1 >= 1 for unit u. Returns (v, sign) where
    sign = -sign(u0) is the sign of the e0 image.
    """
    u = u / u.norm()
    s = 1.0 if float(u[0]) >= 0.0 else -1.0
    v = u.clone()
    v[0] = v[0] + s
    return v, -s


def apply_householder(x, v):
    """Apply H(v) along the token axis of x with rank-one updates.

    x: [..., B, d]; v: [B]. Never materializes the [B, B] matrix.
    """
    vv = torch.dot(v, v)
    proj = torch.matmul(v.unsqueeze(0), x)
    return x - (2.0 / vv) * v.unsqueeze(-1) * proj


def householder_matrix(v):
    """Dense [B, B] matrix of H(v), for tests and small-B fallbacks."""
    b = v.shape[0]
    eye = torch.eye(b, dtype=v.dtype, device=v.device)
    return eye - (2.0 / torch.dot(v, v)) * torch.outer(v, v)


class TemporalTransform:
    """Base class: an invertible transform along the token axis."""

    name = "base"
    is_orthogonal = True

    def __init__(self, block_size):
        self.block_size = block_size

    def fit(self, x):
        """Adapt to a block (no-op for fixed transforms)."""
        return self

    def forward(self, x):
        raise NotImplementedError

    def inverse(self, y):
        raise NotImplementedError

    def matrix(self, dtype=torch.float32, device="cpu"):
        """Dense [B, B] R with forward(x) == R @ x, or None."""
        return None

    def metadata_floats(self):
        """Per-block basis floats an artifact must carry."""
        return 0


class Identity(TemporalTransform):
    name = "identity"

    def forward(self, x):
        return x

    def inverse(self, y):
        return y

    def matrix(self, dtype=torch.float32, device="cpu"):
        return torch.eye(self.block_size, dtype=dtype, device=device)


class AnchorDelta(TemporalTransform):
    """CacheGen-style baseline: token 0 is the anchor, every other
    token is stored as a residual against that same anchor. Affine,
    not orthogonal; no transitive delta chain."""

    name = "anchor_delta"
    is_orthogonal = False

    def forward(self, x):
        y = x.clone()
        y[1:] = x[1:] - x[0:1]
        return y

    def inverse(self, y):
        x = y.clone()
        x[1:] = y[1:] + y[0:1]
        return x


class DCHouseholder(TemporalTransform):
    """Single reflector mapping the DC mode ones(B)/sqrt(B) to e0.

    Row 0 of the output carries the block-wide common component
    (scaled block mean); the remaining rows carry differences.
    """

    name = "dc_householder"

    def __init__(self, block_size):
        super().__init__(block_size)
        u = torch.full((block_size,), 1.0 / math.sqrt(block_size))
        self.v, self.sign = householder_vector(u)

    def forward(self, x):
        v = self.v.to(dtype=x.dtype, device=x.device)
        return apply_householder(x, v)

    def inverse(self, y):
        return self.forward(y)

    def matrix(self, dtype=torch.float32, device="cpu"):
        return householder_matrix(self.v.to(dtype=dtype, device=device))


class RandomHouseholder(TemporalTransform):
    """Negative control: reflector for a random unit direction."""

    name = "random_householder"

    def __init__(self, block_size, seed=0):
        super().__init__(block_size)
        self.seed = seed
        gen = torch.Generator().manual_seed(seed)
        u = torch.randn(block_size, generator=gen)
        self.v, self.sign = householder_vector(u)

    forward = DCHouseholder.forward
    inverse = DCHouseholder.inverse
    matrix = DCHouseholder.matrix


class DCT2(TemporalTransform):
    """Orthonormal DCT-II along the token axis."""

    name = "dct2"

    def __init__(self, block_size):
        super().__init__(block_size)
        b = block_size
        n = torch.arange(b, dtype=torch.float64)
        k = n.unsqueeze(1)
        mat = torch.cos(math.pi * (n + 0.5) * k / b) * math.sqrt(2.0 / b)
        mat[0] = mat[0] / math.sqrt(2.0)
        self._mat = mat

    def forward(self, x):
        return self.matrix(x.dtype, x.device) @ x

    def inverse(self, y):
        return self.matrix(y.dtype, y.device).transpose(-1, -2) @ y

    def matrix(self, dtype=torch.float32, device="cpu"):
        return self._mat.to(dtype=dtype, device=device)


class Haar(TemporalTransform):
    """Orthonormal Haar transform; requires B to be a power of two."""

    name = "haar"

    def __init__(self, block_size):
        super().__init__(block_size)
        if block_size & (block_size - 1):
            raise ValueError("Haar requires a power-of-two block size")
        mat = torch.eye(1, dtype=torch.float64)
        while mat.shape[0] < block_size:
            n = mat.shape[0]
            top = torch.kron(mat, torch.tensor([[1.0, 1.0]], dtype=torch.float64))
            bot = torch.kron(
                torch.eye(n, dtype=torch.float64),
                torch.tensor([[1.0, -1.0]], dtype=torch.float64),
            )
            mat = torch.cat([top, bot], dim=0) / math.sqrt(2.0)
        self._mat = mat

    forward = DCT2.forward
    inverse = DCT2.inverse

    def matrix(self, dtype=torch.float32, device="cpu"):
        return self._mat.to(dtype=dtype, device=device)


class Hadamard(TemporalTransform):
    """Flat orthonormal Hadamard (Sylvester) transform; requires B a
    power of two. This is the KVLinC-ablation control: a token-axis
    rotation with no energy compaction. KVLinC (arXiv:2510.05373)
    found exactly this rotation amplifies quantization noise under
    uniform bits; KV-House keeps it to test whether per-mode bit
    allocation changes that verdict."""

    name = "hadamard"

    def __init__(self, block_size):
        super().__init__(block_size)
        if block_size & (block_size - 1):
            raise ValueError("Hadamard requires a power-of-two block size")
        mat = torch.ones(1, 1, dtype=torch.float64)
        while mat.shape[0] < block_size:
            mat = torch.cat([torch.cat([mat, mat], 1), torch.cat([mat, -mat], 1)], 0)
        self._mat = mat / math.sqrt(block_size)

    forward = DCT2.forward
    inverse = DCT2.inverse

    def matrix(self, dtype=torch.float32, device="cpu"):
        return self._mat.to(dtype=dtype, device=device)


class PCAOracle(TemporalTransform):
    """Per-block KLT upper bound: the left singular basis of X.

    forward() uses the basis fitted on the most recent fit() call.
    This is a control, not a deployable codec: shipping the basis
    costs B*B floats per block per layer/head.
    """

    name = "pca_oracle"

    def __init__(self, block_size):
        super().__init__(block_size)
        self._mat = None

    def fit(self, x):
        u, _, _ = torch.linalg.svd(x.to(torch.float64), full_matrices=True)
        self._mat = u.transpose(-1, -2).contiguous()
        return self

    def forward(self, x):
        return self._mat.to(dtype=x.dtype, device=x.device) @ x

    def inverse(self, y):
        return self._mat.to(dtype=y.dtype, device=y.device).transpose(-1, -2) @ y

    def matrix(self, dtype=torch.float32, device="cpu"):
        return None if self._mat is None else self._mat.to(dtype=dtype, device=device)

    def metadata_floats(self):
        return self.block_size * self.block_size


class CorpusPCA(TemporalTransform):
    """Deployable KLT: temporal eigenbasis of the mean token-axis
    second moment over a calibration corpus. The basis is shared, so
    per-block metadata is zero; the calibration identity must be part
    of the codec cache key."""

    name = "corpus_pca"

    def __init__(self, block_size):
        super().__init__(block_size)
        self._mat = None

    def fit_corpus(self, blocks):
        cov = torch.zeros(self.block_size, self.block_size, dtype=torch.float64)
        for x in blocks:
            x64 = x.to(torch.float64)
            cov = cov + x64 @ x64.transpose(-1, -2)
        _, vecs = torch.linalg.eigh(cov / max(len(blocks), 1))
        self._mat = vecs.flip(-1).transpose(-1, -2).contiguous()
        return self

    forward = PCAOracle.forward
    inverse = PCAOracle.inverse
    matrix = PCAOracle.matrix


class PowerIterHouseholder(TemporalTransform):
    """One data-driven reflector: estimate the leading temporal
    singular vector of the block by power iteration on X X^T, then
    reflect it onto e0. Ships B floats of metadata per block."""

    name = "poweriter_householder"

    def __init__(self, block_size, iters=8):
        super().__init__(block_size)
        self.iters = iters
        self.v = None
        self.sign = None

    def fit(self, x):
        x32 = x.to(torch.float32)
        a = x32 @ x32.transpose(-1, -2)
        u = torch.full(
            (self.block_size,),
            1.0 / math.sqrt(self.block_size),
            dtype=a.dtype,
            device=a.device,
        )
        for _ in range(self.iters):
            u = a @ u
            u = u / (u.norm() + 1e-30)
        # v lives on CPU like every other transform's parameters;
        # forward() casts to the input's device
        self.v, self.sign = householder_vector(u.cpu())
        return self

    def forward(self, x):
        v = self.v.to(dtype=x.dtype, device=x.device)
        return apply_householder(x, v)

    def inverse(self, y):
        return self.forward(y)

    def matrix(self, dtype=torch.float32, device="cpu"):
        if self.v is None:
            return None
        return householder_matrix(self.v.to(dtype=dtype, device=device))

    def metadata_floats(self):
        return self.block_size


TRANSFORMS = {
    cls.name: cls
    for cls in (
        Identity,
        AnchorDelta,
        DCHouseholder,
        RandomHouseholder,
        DCT2,
        Haar,
        Hadamard,
        PCAOracle,
        CorpusPCA,
        PowerIterHouseholder,
    )
}


def make_transform(name, block_size, **kwargs):
    return TRANSFORMS[name](block_size, **kwargs)
