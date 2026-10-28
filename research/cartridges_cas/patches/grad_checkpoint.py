#!/usr/bin/env python3
"""Env-gated activation recomputation for the library's Qwen3 decoder.

The pinned cartridges model declares gradient-checkpointing support but
never uses it: ``FlexQwen3Model.forward`` runs the decoder layers in a
plain loop, so one packed 2048-token micro-batch keeps every layer's
activations for the backward pass. On Qwen3-8B that is about 33 GiB
with the bf16 weights and does not fit a 24 GiB GPU.

This patch wraps each decoder layer call in ``torch.utils.checkpoint``
(non-reentrant) when ``CARTRIDGES_GRAD_CKPT=1`` is set and gradients
are enabled. Recomputation changes no arithmetic: the loss and the
cartridge gradient equal the unpatched ones up to bf16 kernel
nondeterminism, which ``scripts/cas_grad_ckpt_check.py`` measures on
the target GPU. Evaluation under ``no_grad`` and every run without the
variable are byte-for-byte the unpatched code path.

Usage:
    grad_checkpoint.py --cart-root /path/to/cartridges

Idempotent; prints one status line.
"""

import pathlib
import sys

_root = "/root/cartridges"
if "--cart-root" in sys.argv:
    _root = sys.argv[sys.argv.index("--cart-root") + 1]
MODEL = pathlib.Path(_root) / "cartridges" / "models" / "qwen" / "modeling_qwen3.py"

OLD_LOOP = """        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            batch = decoder_layer(batch)
"""

NEW_LOOP = """        # knlp: activation recomputation, see patches/grad_checkpoint.py
        _ckpt = (
            os.environ.get("CARTRIDGES_GRAD_CKPT") == "1" and torch.is_grad_enabled()
        )
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if _ckpt:
                hidden = torch.utils.checkpoint.checkpoint(
                    lambda h, _l=decoder_layer, _b=batch: _l(
                        _b.update(hidden_states=h)
                    ).hidden_states,
                    batch.hidden_states,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
                batch = batch.update(hidden_states=hidden)
            else:
                batch = decoder_layer(batch)
"""

OLD_IMPORT = "import torch\nfrom torch import nn\n"
NEW_IMPORT = "import torch\nimport torch.utils.checkpoint\nfrom torch import nn\n"


def main():
    src = MODEL.read_text()
    if "CARTRIDGES_GRAD_CKPT" in src:
        print(f"grad_checkpoint: already applied to {MODEL}")
        return 0
    if src.count(OLD_LOOP) != 1 or src.count(OLD_IMPORT) != 1:
        print(f"grad_checkpoint: anchor not found in {MODEL}", file=sys.stderr)
        return 1
    src = src.replace(OLD_IMPORT, NEW_IMPORT).replace(OLD_LOOP, NEW_LOOP)
    assert "import os\n" in src
    MODEL.write_text(src)
    print(f"grad_checkpoint: applied to {MODEL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
