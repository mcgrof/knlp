"""Round-trip equivalence: custom GPT2KRI <-> HuggingFace GPT2LMHeadModel.

We start from HF GPT-2, load into GPT2KRI, save a synthetic
"trained" checkpoint (we just save the same weights), export to HF,
reload HF, and check that:

   custom.forward(x)  ==  hf_exported.forward(x)

within fp32 tolerance.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.export_hf_gpt2 import export  # noqa: E402
from src.model_gpt2_kri import GPT2KRI  # noqa: E402
from src.utils import pick_device, set_seed  # noqa: E402


def main(hf_name: str = "openai-community/gpt2") -> int:
    from dataclasses import asdict

    from transformers import GPT2LMHeadModel

    set_seed(0)
    device = pick_device()
    print(f"device={device}")

    print(f"loading HF {hf_name} ...")
    custom = GPT2KRI.from_hf_gpt2(hf_name).to(device).eval()

    # Save a "trained" checkpoint that is just the loaded weights.
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "ckpt.pt"
        torch.save({
            "model": custom.state_dict(),
            "cfg": asdict(custom.cfg),
            "step": 0,
            "args": {},
        }, ckpt_path)

        out_dir = Path(tmp) / "hf_export"
        print("exporting to HF format ...")
        export(str(ckpt_path), base_model=hf_name, output_dir=str(out_dir))

        print("reloading exported HF model ...")
        hf_reloaded = GPT2LMHeadModel.from_pretrained(str(out_dir)).to(device).eval()

        # Compare logits.
        ids = torch.randint(0, custom.cfg.vocab_size, (2, 64), device=device)
        with torch.no_grad():
            cl, _ = custom(ids)
            hl = hf_reloaded(ids).logits

        diff = (cl.float() - hl.float()).abs()
        max_abs = diff.max().item()
        print(f"max_abs_diff(custom vs reloaded HF) = {max_abs:.3e}")
        tol = 1e-3
        if max_abs > tol:
            print(f"FAIL: max_abs {max_abs:.3e} > tol {tol:.0e}")
            return 1

    print("PASS: round-trip equivalence within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
