"""Packed-token dataloader using the GPT-2 tokenizer.

The default smoke dataset is TinyStories; the default serious dataset
is FineWeb-Edu sample-10BT (streaming). The CLI surface is the same:

    --dataset_name <hf-name>
    --dataset_config <name>
    --text_column <text>
    --streaming true|false
    --train_split / --val_split

Packing: tokens from many documents are concatenated and chunked into
`seq_len`-token training contexts. EOS separates documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional

import torch
from torch.utils.data import IterableDataset


@dataclass
class DataConfig:
    dataset_name: str = "roneneldan/TinyStories"
    dataset_config: Optional[str] = None
    text_column: str = "text"
    streaming: bool = False
    train_split: str = "train"
    val_split: str = "validation"
    seq_len: int = 1024
    eos_token_id: Optional[int] = None  # default = tokenizer's eos
    val_max_documents: int = 256
    val_from_train_tail: int = 256
    # Tokenizer to use. Must match the model's vocab to avoid OOB
    # token IDs at the embedding/loss layers. Default is GPT-2's
    # tokenizer; pass SmolLM2's name when training/eval'ing SmolLM2.
    tokenizer_name: str = "openai-community/gpt2"


class PackedTextDataset(IterableDataset):
    """Streams text, tokenizes with the GPT-2 tokenizer, packs to seq_len.

    Each item is a dict {"input_ids": [seq_len], "labels": [seq_len]}.
    Labels = input_ids; the cross-entropy in the model handles the
    shift-by-1.

    Distinct documents are separated by EOS.
    """

    def __init__(self, ds, tokenizer, seq_len: int, text_column: str = "text",
                 eos_token_id: Optional[int] = None) -> None:
        super().__init__()
        self.ds = ds
        self.tok = tokenizer
        self.seq_len = seq_len
        self.text_column = text_column
        self.eos_token_id = eos_token_id if eos_token_id is not None else (
            tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 50256
        )

    def __iter__(self) -> Iterator[dict]:
        buf: List[int] = []
        for ex in self.ds:
            text = ex.get(self.text_column)
            if not text:
                continue
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            buf.extend(ids)
            buf.append(self.eos_token_id)
            while len(buf) >= self.seq_len:
                chunk = buf[: self.seq_len]
                buf = buf[self.seq_len :]
                t = torch.tensor(chunk, dtype=torch.long)
                yield {"input_ids": t, "labels": t.clone()}


def load_dataset_safe(cfg: DataConfig, split: str):
    """Wraps datasets.load_dataset to handle the streaming/non-streaming
    case and TinyStories' (lack of) validation split.
    """
    from datasets import load_dataset

    if cfg.dataset_config:
        ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split=split, streaming=cfg.streaming)
    else:
        ds = load_dataset(cfg.dataset_name, split=split, streaming=cfg.streaming)
    return ds


def get_train_val_streams(cfg: DataConfig, tokenizer):
    train_ds = load_dataset_safe(cfg, cfg.train_split)
    try:
        val_ds = load_dataset_safe(cfg, cfg.val_split)
    except Exception:
        # Fall back to slicing the tail of train for validation.
        if cfg.streaming:
            # Cannot slice a streaming dataset by index. Take the first
            # N documents of train, deterministic.
            val_ds = load_dataset_safe(cfg, cfg.train_split)
            val_ds = val_ds.take(cfg.val_from_train_tail)
        else:
            full = load_dataset_safe(cfg, cfg.train_split)
            n = min(len(full), cfg.val_from_train_tail)
            val_ds = full.select(range(len(full) - n, len(full)))

    train = PackedTextDataset(train_ds, tokenizer, cfg.seq_len, cfg.text_column, cfg.eos_token_id)
    val = PackedTextDataset(val_ds, tokenizer, cfg.seq_len, cfg.text_column, cfg.eos_token_id)
    return train, val


def collate(batch: List[dict]) -> dict:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch], dim=0),
        "labels": torch.stack([b["labels"] for b in batch], dim=0),
    }


def get_tokenizer(name: str = "openai-community/gpt2"):
    """Return the tokenizer for the named model.

    For GPT-2-family names we still use GPT2TokenizerFast (the
    historical path). For everything else (SmolLM2 etc.) we fall
    through to AutoTokenizer which picks the right concrete class.
    Vocab mismatch between data and model is the canonical KRI-FT
    foot-gun: don't tokenize with GPT-2 and feed into SmolLM2.
    """
    if "gpt2" in name.lower():
        from transformers import GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained(name)
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    # We always tokenize whole documents then pack into seq_len chunks
    # ourselves. The model_max_length warning is unhelpful in this flow.
    tok.model_max_length = int(1e30)
    return tok
