"""Bind the public confirmation recipe to newly prepared arrays and source."""

import argparse
from pathlib import Path

from research.mo_variants.stage_c_common import read, sha, write

REPO = Path(__file__).resolve().parents[2]


def prepare(root, data, checkpoints=None):
    root, data = Path(root), Path(data).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "records").mkdir()
    (root / "logs").mkdir()
    (root / "code").symlink_to(REPO, target_is_directory=True)
    (root / "data").symlink_to(data, target_is_directory=True)
    if checkpoints:
        destination = Path(checkpoints).expanduser() / root.parent.name / "confirmation"
        destination.mkdir(parents=True, exist_ok=False)
        (root / "checkpoints").symlink_to(destination, target_is_directory=True)
    files = {}
    for directory in ("mo_variants", "tinystories_depth"):
        for path in sorted((REPO / "research" / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                files[str(path.relative_to(REPO))] = sha(path)
    write(root / "records/STAGE-C-SOURCE.json", {"files": files})
    manifest = read(data / "DATA.json")
    inherited = {
        name: sha(data / name) for name in ("train.npy", "curve.npy", "DOCUMENTS.json")
    }
    for name, digest in inherited.items():
        if digest != manifest["files"][name]:
            raise ValueError("Prepared data changed: " + name)
    write(
        root / "records/DATA.json",
        {
            "verified_inherited_files": inherited,
            "confirmation_files": {
                p.name: sha(p) for p in (data / "confirmation").iterdir()
            },
            "training_and_order_files": {
                p.name: sha(p) for p in data.glob("order-*.npy")
            },
            "dataset_revision": manifest["dataset_revision"],
            "tokenizer_revision": manifest["tokenizer_revision"],
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoints")
    args = parser.parse_args()
    prepare(args.root, args.data, args.checkpoints)
