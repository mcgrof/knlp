"""Run several packed sequences through the reader in one pass without
changing the cartridge objective.

The library trains on one packed 2,048-token sequence per forward pass
and accumulates gradients over the global batch, so the GPU sees a
single sequence at a time. ``merge_batches`` concatenates m packed
sequences into one sequence of m * 2,048 tokens. Each keeps its own
element ids (offset so no two sequences share one), its own position
ids and its own distillation targets, so the block mask of the
library's FlexAttention keeps them apart exactly as separate passes
would.

The library's loss for one pass is the mean over that pass's target
entries, divided by the number of accumulated passes. Pooling m
sequences into one mean would weight long sequences more, so
``entry_weights`` gives each entry of sub-sequence s the weight
1 / (m * n_s): the weighted sum over a merged pass, divided by
global_batch / m accumulated passes, equals the sum of the m
per-sequence means divided by global_batch, the original objective.
"""

import torch

ELEMENT_OFFSET = 1 << 16


def merge_batches(subs):
    """Concatenate DatasetBatch objects that are each padded to the same
    packed length. Returns one DatasetBatch with ``_cr_sub_entries``
    holding the target-entry count of each sub-sequence."""
    from cartridges.datasets import DatasetBatch

    seq = subs[0].input_ids.shape[0]
    assert all(b.input_ids.shape[0] == seq for b in subs), "unequal packed lengths"
    assert all(b.loss_weight is None for b in subs), "loss weights unsupported"
    tc = subs[0].token_counts
    for b in subs[1:]:
        tc = tc + b.token_counts
    out = DatasetBatch(
        input_ids=torch.cat([b.input_ids for b in subs]),
        element_ids=torch.cat(
            [b.element_ids + k * ELEMENT_OFFSET for k, b in enumerate(subs)]
        ),
        position_ids=torch.cat([b.position_ids for b in subs]),
        metadata=[m for b in subs for m in b.metadata],
        token_counts=tc,
        topk_logprobs=torch.cat([b.topk_logprobs for b in subs]),
        topk_token_ids=torch.cat([b.topk_token_ids for b in subs]),
        topk_token_idxs=torch.cat(
            [b.topk_token_idxs + k * seq for k, b in enumerate(subs)]
        ),
    )
    out._cr_sub_entries = [int(b.topk_token_idxs.numel()) for b in subs]
    return out


def entry_weights(counts, m, device=None, dtype=torch.float32):
    """Per-entry weights 1 / (m * n_s) for sub-sequence entry counts."""
    return torch.cat([torch.full((n,), 1.0 / (m * n), dtype=dtype) for n in counts]).to(
        device
    )


def install(train_dataset_cls, train_module, m):
    """Merge every m consecutive packed sequences of a TrainDataset (not
    its evaluation subclasses) and weight the loss through the library's
    ``_cr_entry_loss`` hook. The caller divides the library's global
    batch by m."""
    assert hasattr(
        train_module, "_cr_entry_loss"
    ), "MICRO_SEQS needs the position_weights.py library patch (the loss hook)"
    orig_len = train_dataset_cls.__len__
    orig_get = train_dataset_cls.__getitem__

    def _len(self):
        n = orig_len(self)
        return -(-n // m) if type(self) is train_dataset_cls else n

    def _get(self, i):
        if type(self) is not train_dataset_cls:
            return orig_get(self, i)
        n = orig_len(self)
        return merge_batches(
            [orig_get(self, j) for j in range(i * m, min((i + 1) * m, n))]
        )

    train_dataset_cls.__len__ = _len
    train_dataset_cls.__getitem__ = _get

    def _entry_loss(batch, ce_by_token):
        counts = getattr(batch, "_cr_sub_entries", None)
        if counts is None:  # an unmerged pass, e.g. held-out evaluation
            return ce_by_token.mean()
        w = entry_weights(counts, m, ce_by_token.device, ce_by_token.dtype)
        return (ce_by_token * w).sum()

    train_module._cr_entry_loss = _entry_loss
