"""Fail before confirmation when the requested GPU topology is unavailable."""

import argparse
import torch


def check(count, h100=False):
    if not torch.cuda.is_available() or torch.cuda.device_count() < count:
        raise RuntimeError(f"This recipe requires {count} visible GPUs")
    names = [torch.cuda.get_device_name(i) for i in range(count)]
    if h100 and any("H100" not in name for name in names):
        raise RuntimeError("The fixed confirmation runtime requires two H100 GPUs")
    print(names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--h100", action="store_true")
    args = parser.parse_args()
    check(args.count, args.h100)
