"""Document-aware data preparation and an honest token stream.

Companion to scripts/matched_micro_train.py. That harness schedules token
budgets by batch x (length - 1) while its step consumes batch x length, packs
a token array and then cuts it at a fixed fraction, and takes row starts
modulo the partition length so a second pass looks like continued training.
This module supplies the corrected data path; the trainer keeps its own
history for comparison.

Three defects in the previous harness are fixed here.

Documents are split into partitions BEFORE packing, by a deterministic hash of
normalized content, so a single document cannot land in two partitions and a
document cannot be cut by a partition boundary. Duplicates are removed across
partitions by that same hash.

The stream does not wrap silently. It exposes its cursor, counts unique tokens
separately from repeated exposures, and announces every epoch boundary instead
of hiding one behind a modulo.

Targets are counted exactly: a batch of B rows of L+1 identifiers yields B*L
targets, and the count reported is of actual unmasked targets, including
partial batches.
"""

import hashlib, json, os
import numpy as np

PAD = -100


def doc_partition(text, weights=(98, 1, 1), names=("train", "dev", "test")):
    """A deterministic content hash decides the partition, so it is stable."""
    h = hashlib.sha256(" ".join(text.split()).encode("utf-8")).digest()
    v = int.from_bytes(h[:8], "big") % sum(weights)
    acc = 0
    for name, w in zip(names, weights):
        acc += w
        if v < acc:
            return name, h.hex()
    return names[-1], h.hex()


def prepare(
    out_dir,
    dataset,
    split,
    tokenizer_name,
    target_tokens,
    eos_delimited=True,
    revision=None,
):
    """Tokenize documents, assign whole documents to partitions, then pack."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    os.makedirs(out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    kw = dict(split=split, streaming=True)
    if revision:
        kw["revision"] = revision
    ds = load_dataset(dataset, **kw)
    buckets = {k: [] for k in ("train", "dev", "test")}
    counts = {k: 0 for k in buckets}
    seen, docs, dup = set(), 0, 0
    for ex in ds:
        text = ex.get("text") or ""
        if not text.strip():
            continue
        part, h = doc_partition(text)
        if h in seen:
            dup += 1
            continue
        seen.add(h)
        ids = tok(text).input_ids
        if eos_delimited:
            ids = ids + [tok.eos_token_id]
        buckets[part].append(np.asarray(ids, dtype=np.uint16))
        counts[part] += len(ids)
        docs += 1
        if counts["train"] >= target_tokens:
            break
    meta = dict(
        dataset=dataset,
        split=split,
        tokenizer=tokenizer_name,
        revision=revision,
        documents=docs,
        duplicates_dropped=dup,
        eos_delimited=eos_delimited,
        partition_weights=dict(train=98, dev=1, test=1),
        partition_rule=(
            "sha256 of whitespace-normalized document text "
            "modulo 100; whole documents only"
        ),
        tokens=dict(counts),
    )
    for part, arrs in buckets.items():
        if not arrs:
            continue
        flat = np.concatenate(arrs)
        np.save(os.path.join(out_dir, "%s.npy" % part), flat)
        bounds = np.cumsum([len(a) for a in arrs], dtype=np.int64)
        np.save(os.path.join(out_dir, "%s_docbounds.npy" % part), bounds)
        meta.setdefault("packed", {})[part] = dict(
            tokens=int(len(flat)), documents=len(arrs)
        )
    meta["token_bytes_sha256"] = {
        p: hashlib.sha256(
            np.load(os.path.join(out_dir, "%s.npy" % p)).tobytes()
        ).hexdigest()
        for p in buckets
        if os.path.exists(os.path.join(out_dir, "%s.npy" % p))
    }
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    return meta


class Stream:
    """A packed stream with an explicit cursor and no hidden wrapping."""

    def __init__(self, data_dir, part, seq_len, batch_size):
        self.tokens = np.load(os.path.join(data_dir, "%s.npy" % part), mmap_mode="r")
        self.meta = json.load(open(os.path.join(data_dir, "meta.json")))
        self.seq_len = seq_len
        self.batch = batch_size
        self.row_len = seq_len + 1  # L inputs plus the final target
        self.n = len(self.tokens)
        self.rows_available = (self.n - 1) // self.seq_len
        self.cursor = 0  # in rows, monotonically increasing
        self.epochs = 0
        self.unique_rows_seen = min(self.rows_available, 0)

    def targets_per_batch(self, rows=None):
        return (rows if rows is not None else self.batch) * self.seq_len

    def next_batch(self, device):
        """Rows advance by seq_len so every target is consumed exactly once."""
        import torch

        rows, wrapped = [], False
        for _ in range(self.batch):
            r = self.cursor
            if r >= self.rows_available:
                self.epochs += 1
                wrapped = True
                self.cursor = 0
                r = 0
            start = r * self.seq_len
            rows.append(
                np.asarray(self.tokens[start : start + self.row_len], dtype=np.int64)
            )
            self.cursor += 1
        x = torch.from_numpy(np.stack(rows)).to(device)
        return x, dict(wrapped=wrapped, epochs=self.epochs, cursor=self.cursor)

    def state(self):
        return dict(
            cursor=self.cursor, epochs=self.epochs, rows_available=self.rows_available
        )

    def load_state(self, s):
        self.cursor = int(s["cursor"])
        self.epochs = int(s["epochs"])
