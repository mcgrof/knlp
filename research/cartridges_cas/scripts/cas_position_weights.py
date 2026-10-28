#!/usr/bin/env python3
"""Per-position loss weights for cartridge training, keyed to the
trainer's own packed batches.

The cartridge objective averages the top-20 distillation loss over
every stored entry of every assistant token. The document changes the
teacher's next-token distribution at only a minority of those positions
(the dates, drug names, lab values); the rest the frozen model predicts
without the record. Measured in August, the objective's weight is
uncorrelated with where the document matters. This module lets a run
put its weight where the document is.

A position is *informative* when the frozen model's most likely next
token with no cartridge differs from the teacher's most likely token
(the teacher had the record in context). That is the same mask
``cas_cart_loss.py`` reports as the informative/rest split, computed
once over the training parquet with the trainer's exact packing. The
builder writes one mask per packed batch, keyed by a hash of the batch's
token ids and stored entry layout, so the trainer looks its batch up by
content rather than by index and aborts on any mismatch.

Weight modes (entry weights are normalized so their mean over the whole
training set is 1, keeping the loss scale of the uniform objective):

    informative  1 on informative entries, FLOOR on the rest
    rest         0 on informative entries, 1 on the rest (the control)
    uniform      1 everywhere (bit-identical to the unpatched loss)

Builder env:
    MODEL     Qwen/Qwen3-8B (default)
    PARQUET   the training parquet the run will use
    SEED      packing seed (default 42, the trainer's)
    OUT       output .npz
    DEVICE    cuda (default)

Prints ``POSW,<key>,<value>`` lines and ``POSW_DONE``. The trainer side
(``load_weights``, ``entry_loss``) needs only numpy and torch.
"""

import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

SEQ_LEN = 2048
TOP_K = 20


def batch_key(batch):
    """Content hash of a packed batch: token ids plus the stored entry
    layout. Two batches with the same key have the same entries in the
    same order, so a mask built on one applies to the other."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(batch.input_ids.cpu().numpy()).tobytes())
    h.update(np.ascontiguousarray(batch.topk_token_idxs.cpu().numpy()).tobytes())
    h.update(np.ascontiguousarray(batch.topk_token_ids.cpu().numpy()).tobytes())
    return h.hexdigest()[:24]


def teacher_argmax(batch, n_pos):
    """Teacher's most likely token at each position (-1 without entries)."""
    pos = batch.topk_token_idxs - 1
    lp = batch.topk_logprobs
    best = torch.full((n_pos,), -float("inf"))
    best = best.scatter_reduce(0, pos, lp, reduce="amax", include_self=True)
    is_best = lp == best[pos]
    out = torch.full((n_pos,), -1, dtype=torch.long)
    out[pos[is_best]] = batch.topk_token_ids[is_best]
    return out


def mode_weights(mode, floor, n_info, n_total):
    """(w_info, w_rest) before normalization, and the normalizer that
    makes the mean entry weight 1 over the training set."""
    if mode == "informative":
        wi, wr = 1.0, float(floor)
    elif mode == "rest":
        wi, wr = 0.0, 1.0
    elif mode == "uniform":
        wi, wr = 1.0, 1.0
    else:
        raise ValueError(mode)
    mean = (wi * n_info + wr * (n_total - n_info)) / max(1, n_total)
    return wi / mean, wr / mean


class EntryWeights:
    """Masks per batch key plus the two scalar weights for a mode."""

    def __init__(self, masks, w_info, w_rest, meta):
        self.masks = masks
        self.w_info = float(w_info)
        self.w_rest = float(w_rest)
        self.meta = meta
        self.hits = 0

    def weights_for(self, batch, device, dtype):
        key = batch_key(batch)
        if key not in self.masks:
            raise KeyError(f"no position mask for batch {key}; wrong parquet or seed")
        self.hits += 1
        m = torch.from_numpy(self.masks[key]).to(device)
        return torch.where(
            m,
            torch.tensor(self.w_info, device=device, dtype=dtype),
            torch.tensor(self.w_rest, device=device, dtype=dtype),
        )

    def entry_loss(self, batch, ce_by_token):
        """Weighted mean of the per-entry losses. With uniform weights
        this is ``(1 * ce).mean()``, bit-identical to ``ce.mean()``."""
        w = self.weights_for(batch, ce_by_token.device, ce_by_token.dtype)
        assert w.shape == ce_by_token.shape, (w.shape, ce_by_token.shape)
        return (w * ce_by_token).mean()


def load_weights(path, mode, floor=0.1):
    """Load a builder .npz and fix the weights for ``mode``."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    masks = {k[2:]: z[k].astype(bool) for k in z.files if k.startswith("m_")}
    wi, wr = mode_weights(mode, floor, meta["n_info_entries"], meta["n_entries"])
    meta = dict(meta, mode=mode, floor=floor, w_info=wi, w_rest=wr)
    return EntryWeights(masks, wi, wr, meta)


def main():
    os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/tmp/cas_position_weights")
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    from transformers import AutoTokenizer
    from cartridges.datasets import TrainDataset, DataSource
    from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM

    model_name = os.environ.get("MODEL", "Qwen/Qwen3-8B")
    parquet = os.environ["PARQUET"]
    seed = int(os.environ.get("SEED", "42"))
    out = os.environ["OUT"]
    device = os.environ.get("DEVICE", "cuda")

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name)
    ds = TrainDataset.Config(
        data_sources=[DataSource(path=parquet, type="local")],
        top_k_logits=TOP_K,
        packed_seq_length=SEQ_LEN,
        packing_mode="truncate",
    ).instantiate(tokenizer=tok, seed=seed)
    print(f"POSW,packed,{len(ds)}", flush=True)
    model = (
        FlexQwen3ForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        .to(device)
        .to(torch.bfloat16)
    )
    model.eval()

    masks = {}
    n_entries = n_info = n_pos = n_info_pos = 0
    with torch.no_grad():
        for b in range(len(ds)):
            batch = ds[b]
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outp = model(
                    input_ids=batch.input_ids.to(device),
                    seq_ids=batch.element_ids.to(device),
                    position_ids=batch.position_ids.to(device),
                    use_cache=False,
                    past_key_values=None,
                )
            model_arg = outp.logits[0].argmax(dim=-1).cpu()
            t_arg = teacher_argmax(batch, SEQ_LEN)
            pos = batch.topk_token_idxs - 1
            info_pos = (t_arg >= 0) & (model_arg != t_arg)
            mask = info_pos[pos].numpy()
            masks[batch_key(batch)] = mask
            n_entries += mask.size
            n_info += int(mask.sum())
            has = torch.zeros(SEQ_LEN, dtype=torch.bool)
            has[pos] = True
            n_pos += int(has.sum())
            n_info_pos += int((has & info_pos).sum())
            if b % 100 == 0:
                print(
                    f"POSW,progress,{b}/{len(ds)},{time.time() - t0:.0f}s", flush=True
                )
            del outp
    meta = {
        "model": model_name,
        "parquet": os.path.abspath(parquet),
        "seed": seed,
        "seq_len": SEQ_LEN,
        "n_packed": len(ds),
        "n_entries": n_entries,
        "n_info_entries": n_info,
        "n_positions": n_pos,
        "n_info_positions": n_info_pos,
        "definition": "informative = frozen model argmax without cartridge "
        "differs from teacher argmax over the stored top-20",
    }
    np.savez_compressed(
        out, meta=json.dumps(meta), **{f"m_{k}": v for k, v in masks.items()}
    )
    for k in (
        "n_packed",
        "n_entries",
        "n_info_entries",
        "n_positions",
        "n_info_positions",
    ):
        print(f"POSW,{k},{meta[k]}")
    print(f"POSW,info_fraction_entries,{n_info / max(1, n_entries):.4f}")
    print(f"POSW,info_fraction_positions,{n_info_pos / max(1, n_pos):.4f}")
    print(f"POSW_DONE {out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
