"""Download pinned public data and reserve document-disjoint evaluation inputs."""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from research.tinystories_depth.data import DATA_REV, TOKEN_REV, prepare
from research.mo_variants.prepare_data import prepare as confirmation_data


def build(root, cache):
    root, cache = Path(root), Path(cache).expanduser()
    hub = cache / "hub"
    dataset = Path(
        snapshot_download(
            "roneneldan/TinyStories",
            repo_type="dataset",
            revision=DATA_REV,
            allow_patterns=["data/*.parquet"],
            cache_dir=hub,
        )
    )
    tokenizer = Path(
        snapshot_download(
            "gpt2",
            revision=TOKEN_REV,
            allow_patterns=[
                "tokenizer.json",
                "tokenizer_config.json",
                "config.json",
                "vocab.json",
                "merges.txt",
            ],
            cache_dir=hub,
        )
    )
    prepare(root / "data", hub=hub)
    inputs = root / "inputs"
    inputs.mkdir()
    validation = sorted((dataset / "data").glob("validation-*.parquet"))
    if len(validation) != 1:
        raise ValueError("Expected one pinned validation shard")
    shutil.copyfile(validation[0], inputs / "validation.parquet")
    shutil.copyfile(tokenizer / "tokenizer.json", inputs / "tokenizer.json")
    (root / "records").mkdir(exist_ok=True)
    confirmation_data(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache", required=True)
    args = parser.parse_args()
    build(args.root, args.cache)
