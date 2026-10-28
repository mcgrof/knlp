#!/usr/bin/env python3
"""Route the trainer's per-entry loss through a replaceable hook.

``cartridges.train`` reduces the per-entry distillation losses with a
plain ``.mean()`` inside its training loop, so there is no way to weight
positions without editing the loop. This patch replaces that one
expression with a call to a module-level function ``_cr_entry_loss``
whose default body is the same ``.mean()``; ``cas_train_isolated.py``
swaps the function in when ``POS_WEIGHTS`` is set (see
``scripts/cas_position_weights.py``). The held-out evaluation keeps its
own unweighted formula, so every arm's held-out loss stays comparable.
Runs without ``POS_WEIGHTS`` execute the identical arithmetic.

Usage:
    position_weights.py --cart-root /path/to/cartridges

Idempotent; prints one status line.
"""

import pathlib
import sys

_root = "/root/cartridges"
if "--cart-root" in sys.argv:
    _root = sys.argv[sys.argv.index("--cart-root") + 1]
TRAIN = pathlib.Path(_root) / "cartridges" / "train.py"

OLD_LOSS = "                    loss = (ce_by_token.mean() / accumulate_grad_steps)\n"
NEW_LOSS = "                    loss = (_cr_entry_loss(batch, ce_by_token) / accumulate_grad_steps)\n"
ANCHOR = "class CacheAndModel(nn.Module):\n"
HOOK = """# knlp: per-entry loss hook, see patches/position_weights.py
def _cr_entry_loss(batch, ce_by_token):
    return ce_by_token.mean()


"""


def main():
    src = TRAIN.read_text()
    if "_cr_entry_loss" in src:
        print(f"position_weights: already applied to {TRAIN}")
        return 0
    if src.count(OLD_LOSS) != 1 or src.count(ANCHOR) != 1:
        print(f"position_weights: anchor not found in {TRAIN}", file=sys.stderr)
        return 1
    src = src.replace(OLD_LOSS, NEW_LOSS).replace(ANCHOR, HOOK + ANCHOR)
    TRAIN.write_text(src)
    print(f"position_weights: applied to {TRAIN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
