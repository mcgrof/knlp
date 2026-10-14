# SPDX-License-Identifier: MIT
"""MNIST tensors, fixed noise images and named random streams.

MNIST is kept at its native 28x28 resolution and both splits use the same
ToTensor() + Normalize((0.5,), (0.5,)) transform as the reference script,
so pixels lie in [-1, 1], the same range as the uniform noise images.

Every random quantity is drawn from a CPU torch.Generator seeded by
stream_seed(model_key, stream_name), a SHA-256 derivation that does not
depend on Python's per-process hash() randomization or on what other
streams consumed. Architectures therefore see identical common weights,
noise and minibatch orders for a given model key.
"""

import hashlib
import os

import torch

PIXELS = 28 * 28


def stream_seed(model_key, stream):
    digest = hashlib.sha256(f"{model_key}/{stream}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def generator(model_key, stream):
    g = torch.Generator(device="cpu")
    g.manual_seed(stream_seed(model_key, stream))
    return g


def load_mnist(data_dir):
    """Return (train_x [60000,784], train_y, test_x [10000,784], test_y)."""
    cache = os.path.join(data_dir, "subliminal_mnist_28x28_norm05.pt")
    if os.path.exists(cache):
        return torch.load(cache)
    from torchvision import datasets, transforms

    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    out = []
    for train in (True, False):
        ds = datasets.MNIST(data_dir, train=train, download=True, transform=tfm)
        xs = torch.stack([ds[i][0] for i in range(len(ds))]).reshape(len(ds), PIXELS)
        ys = torch.tensor([int(ds.targets[i]) for i in range(len(ds))])
        out += [xs.contiguous(), ys]
    tensors = tuple(out)
    tmp = cache + ".tmp"
    torch.save(tensors, tmp)
    os.replace(tmp, cache)
    return tensors


def make_noise(model_key, n, pixels=PIXELS, stream="noise"):
    """The model's fixed noise images, uniform in [-1, 1).

    stream="heldout_noise" gives a separate set used only for diagnostics.
    """
    g = generator(model_key, stream)
    return torch.rand(n, pixels, generator=g) * 2 - 1


def epoch_order(model_key, stage, epoch, n):
    """Minibatch order for one epoch of a training stage (a permutation)."""
    return torch.randperm(n, generator=generator(model_key, f"{stage}/e{epoch}"))


def tensor_digest(t):
    """SHA-256 of a tensor's bytes, for noise and checkpoint identity."""
    t = t.detach().to("cpu").contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(t.shape)).encode())
    h.update(str(t.dtype).encode())
    h.update(t.numpy().tobytes())
    return h.hexdigest()
