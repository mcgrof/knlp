"""Verify custom GPT-2 dense logits match HuggingFace GPT-2 before training.

The acceptance criterion stated in the project plan:

    max_abs_error < 1e-3 in fp32

We tolerate a small numerical drift only if explained. Running this on
ROCm shows whether the GPT-2 weight transfer is correct end-to-end.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_gpt2_kri import GPT2KRI  # noqa: E402
from src.utils import pick_device, set_seed  # noqa: E402


def main(hf_name: str = "openai-community/gpt2") -> int:
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    set_seed(0)
    device = pick_device()
    print(f"device={device}")

    print(f"loading HF {hf_name} ...")
    hf = GPT2LMHeadModel.from_pretrained(hf_name).to(device).eval()
    tok = GPT2TokenizerFast.from_pretrained(hf_name)
    text = "The quick brown fox jumps over the lazy dog. Three little pigs went to market."
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    # Pad to a deterministic length and stack a small batch.
    ids = torch.cat([ids, ids[:, :8]], dim=1)
    ids = ids.repeat(2, 1)
    print(f"input_ids shape={tuple(ids.shape)}")

    print("loading custom GPT2KRI from HF weights ...")
    custom = GPT2KRI.from_hf_gpt2(hf).to(device).eval()
    print(f"custom params: {custom.num_params():,}")

    with torch.no_grad():
        hf_logits = hf(ids).logits.float()
        custom_logits, _ = custom(ids)
        custom_logits = custom_logits.float()

    diff = (hf_logits - custom_logits).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    print(f"max_abs_diff={max_abs:.3e}  mean_abs_diff={mean_abs:.3e}")

    # On HIP/ROCm, expect ~1e-5 in fp32; allow a slack of 1e-3.
    tol = 1e-3
    if max_abs > tol:
        print(f"FAIL: max_abs_diff {max_abs:.3e} > tol {tol:.0e}")
        return 1

    # Sanity: argmax token agreement
    hf_top = hf_logits.argmax(-1)
    cs_top = custom_logits.argmax(-1)
    agree = (hf_top == cs_top).float().mean().item()
    print(f"argmax_agreement={agree:.4f}")
    if agree < 0.999:
        print(f"FAIL: argmax agreement {agree:.4f} < 0.999")
        return 1

    print("PASS: custom GPT-2 matches HF GPT-2 within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
