#!/usr/bin/env python3
"""
Prepare datasets for GPT-2 training.
Downloads and processes text datasets into binary format.
"""

import os
import sys
import numpy as np
import requests
from pathlib import Path
import tiktoken


def download_shakespeare():
    """Download and prepare Shakespeare dataset."""
    data_dir = Path("gpt2/data/shakespeare")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download Shakespeare text
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    input_file = data_dir / "input.txt"
    if not input_file.exists():
        print(f"Downloading Shakespeare dataset from {url}...")
        response = requests.get(url)
        with open(input_file, "w") as f:
            f.write(response.text)
        print(f"Downloaded to {input_file}")
    else:
        print(f"Shakespeare dataset already exists at {input_file}")

    # Read the text
    with open(input_file, "r") as f:
        text = f.read()

    # Tokenize using GPT-2 tokenizer
    print("Tokenizing dataset...")
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode(text, allowed_special={"<|endoftext|>"})
    print(f"Total tokens: {len(train_ids):,}")

    # Split into train and val (90/10 split)
    split_idx = int(len(train_ids) * 0.9)
    train_data = np.array(train_ids[:split_idx], dtype=np.uint16)
    val_data = np.array(train_ids[split_idx:], dtype=np.uint16)

    # Save as binary files
    train_file = data_dir / "train.bin"
    val_file = data_dir / "val.bin"

    train_data.tofile(train_file)
    val_data.tofile(val_file)

    print(f"Saved {len(train_data):,} training tokens to {train_file}")
    print(f"Saved {len(val_data):,} validation tokens to {val_file}")
    print("Dataset preparation complete!")


def download_finewebedu():
    """Download and prepare FineWebEdu dataset."""
    from datasets import load_dataset

    data_dir = Path("gpt2/data/finewebedu")
    data_dir.mkdir(parents=True, exist_ok=True)

    train_file = data_dir / "train.bin"
    val_file = data_dir / "val.bin"

    # Check if already processed
    if train_file.exists() and val_file.exists():
        print(f"FineWebEdu dataset already exists at {data_dir}")
        return

    print("Downloading FineWebEdu dataset (this may take a while)...")
    # Load a subset of FineWebEdu for reasonable training time
    # Using 'sample-10BT' subset which is ~10B tokens
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True
    )

    print("Tokenizing FineWebEdu dataset...")
    enc = tiktoken.get_encoding("gpt2")

    # Process in chunks to avoid memory issues
    max_tokens = 100_000_000  # 100M tokens for reasonable size
    all_tokens = []
    token_count = 0

    for example in dataset:
        if token_count >= max_tokens:
            break
        text = example["text"]
        tokens = enc.encode(text, allowed_special={"<|endoftext|>"})
        all_tokens.extend(tokens)
        token_count += len(tokens)

        if token_count % 1_000_000 == 0:
            print(f"Processed {token_count:,} tokens...")

    print(f"Total tokens collected: {len(all_tokens):,}")

    # Split into train and val (90/10 split)
    split_idx = int(len(all_tokens) * 0.9)
    train_data = np.array(all_tokens[:split_idx], dtype=np.uint16)
    val_data = np.array(all_tokens[split_idx:], dtype=np.uint16)

    # Save as binary files
    train_data.tofile(train_file)
    val_data.tofile(val_file)

    print(f"Saved {len(train_data):,} training tokens to {train_file}")
    print(f"Saved {len(val_data):,} validation tokens to {val_file}")
    print("FineWebEdu dataset preparation complete!")

    # Explicitly cleanup dataset to avoid threading issues during shutdown
    del dataset


def download_openwebtext():
    """Download and prepare OpenWebText dataset."""
    from datasets import load_dataset

    data_dir = Path("gpt2/data/openwebtext")
    data_dir.mkdir(parents=True, exist_ok=True)

    train_file = data_dir / "train.bin"
    val_file = data_dir / "val.bin"

    # Check if already processed
    if train_file.exists() and val_file.exists():
        print(f"OpenWebText dataset already exists at {data_dir}")
        return

    print("Downloading OpenWebText dataset (this may take a while)...")
    # Load OpenWebText dataset
    dataset = load_dataset("openwebtext", split="train", streaming=True)

    print("Tokenizing OpenWebText dataset...")
    enc = tiktoken.get_encoding("gpt2")

    # Process in chunks to avoid memory issues
    max_tokens = 100_000_000  # 100M tokens for reasonable size
    all_tokens = []
    token_count = 0

    for example in dataset:
        if token_count >= max_tokens:
            break
        text = example["text"]
        tokens = enc.encode(text, allowed_special={"<|endoftext|>"})
        all_tokens.extend(tokens)
        token_count += len(tokens)

        if token_count % 1_000_000 == 0:
            print(f"Processed {token_count:,} tokens...")

    print(f"Total tokens collected: {len(all_tokens):,}")

    # Split into train and val (90/10 split)
    split_idx = int(len(all_tokens) * 0.9)
    train_data = np.array(all_tokens[:split_idx], dtype=np.uint16)
    val_data = np.array(all_tokens[split_idx:], dtype=np.uint16)

    # Save as binary files
    train_data.tofile(train_file)
    val_data.tofile(val_file)

    print(f"Saved {len(train_data):,} training tokens to {train_file}")
    print(f"Saved {len(val_data):,} validation tokens to {val_file}")
    print("OpenWebText dataset preparation complete!")

    # Explicitly cleanup dataset to avoid threading issues during shutdown
    del dataset


def download_tinystories():
    """Download and prepare TinyStories dataset."""
    from datasets import load_dataset

    data_dir = Path("gpt2/data/tinystories")
    data_dir.mkdir(parents=True, exist_ok=True)

    train_file = data_dir / "train.bin"
    val_file = data_dir / "val.bin"

    # Check if already processed
    if train_file.exists() and val_file.exists():
        print(f"TinyStories dataset already exists at {data_dir}")
        return

    print("Downloading TinyStories dataset...")
    # Load TinyStories dataset from HuggingFace
    # Using the full dataset which is compact (~2.1M stories)
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    print(f"Loaded {len(dataset):,} stories")
    print("Tokenizing TinyStories dataset...")
    enc = tiktoken.get_encoding("gpt2")

    # Process all stories
    all_tokens = []
    for i, example in enumerate(dataset):
        text = example["text"]
        tokens = enc.encode(text, allowed_special={"<|endoftext|>"})
        # Add end of text token between stories
        tokens.append(enc.encode("<|endoftext|>")[0])
        all_tokens.extend(tokens)

        if (i + 1) % 100000 == 0:
            print(f"Processed {i+1:,} stories, {len(all_tokens):,} tokens...")

    print(f"Total tokens collected: {len(all_tokens):,}")

    # Split into train and val (95/5 split - larger train set for small dataset)
    split_idx = int(len(all_tokens) * 0.95)
    train_data = np.array(all_tokens[:split_idx], dtype=np.uint16)
    val_data = np.array(all_tokens[split_idx:], dtype=np.uint16)

    # Save as binary files
    train_data.tofile(train_file)
    val_data.tofile(val_file)

    print(f"Saved {len(train_data):,} training tokens to {train_file}")
    print(f"Saved {len(val_data):,} validation tokens to {val_file}")
    print("TinyStories dataset preparation complete!")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Prepare datasets for GPT-2 training")
    parser.add_argument(
        "--dataset",
        type=str,
        default="shakespeare",
        choices=["shakespeare", "finewebedu", "openwebtext", "tinystories"],
        help="Dataset to prepare",
    )

    args = parser.parse_args()

    if args.dataset == "shakespeare":
        download_shakespeare()
    elif args.dataset == "finewebedu":
        download_finewebedu()
    elif args.dataset == "openwebtext":
        download_openwebtext()
    elif args.dataset == "tinystories":
        download_tinystories()
    else:
        print(f"Unknown dataset: {args.dataset}")
        sys.exit(1)


if __name__ == "__main__":
    main()
