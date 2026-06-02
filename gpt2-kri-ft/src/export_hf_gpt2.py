"""Export a trained GPT2KRI checkpoint back to HuggingFace GPT2LMHeadModel.

The exported directory is a normal HF model directory:
    config.json, pytorch_model.bin (or safetensors), tokenizer files.

CLI:
    python -m src.export_hf_gpt2 \
        --checkpoint runs/kri-gpt2-small/checkpoint_final.pt \
        --base_model openai-community/gpt2 \
        --output_dir runs/kri-gpt2-small/hf

`--base_model` is used only to copy the tokenizer over and to derive
the HF config defaults.

Numerical convention:
    For each Conv1D weight (c_attn, c_proj, c_fc) we transpose
    `[out, in]` back to `[in, out]`. Biases are kept as-is. The HF
    `transformer.h.<i>.attn.bias` causal-mask buffer is not in our
    state and HF will reconstruct it on load.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_gpt2_kri import GPT2Config, GPT2KRI  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--base_model", type=str, default="openai-community/gpt2")
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def custom_to_hf_state_dict(custom_sd: dict, n_layer: int) -> dict:
    """Map our `GPT2KRI` state dict to the HF GPT-2 state dict layout.

    Linear -> Conv1D weight transpose. Biases unchanged.
    Add the `transformer.` prefix. Causal buffers omitted.
    """
    conv1d_keys = {
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    }
    out = {}
    for k, v in custom_sd.items():
        if k.endswith("causal_bias"):
            continue
        if k == "lm_head.weight":
            # HF ties this to wte; leave it implicit, but also emit it
            # so non-tying HF inference paths still work.
            out["lm_head.weight"] = v
            continue
        hk = f"transformer.{k}"
        for suffix in conv1d_keys:
            if hk.endswith(suffix):
                v = v.t().contiguous()
                break
        out[hk] = v
    return out


def export(checkpoint_path: str, base_model: str, output_dir: str) -> None:
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config as HFGPT2Config

    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = GPT2Config(**ck["cfg"])

    hf_cfg = HFGPT2Config.from_pretrained(base_model)
    # Overwrite fields where our trained config diverges (unlikely for
    # the GPT-2 small fine-tune, but defensive).
    hf_cfg.vocab_size = cfg.vocab_size
    hf_cfg.n_positions = cfg.n_positions
    hf_cfg.n_embd = cfg.n_embd
    hf_cfg.n_layer = cfg.n_layer
    hf_cfg.n_head = cfg.n_head
    hf_cfg.layer_norm_epsilon = cfg.layer_norm_epsilon

    hf_model = GPT2LMHeadModel(hf_cfg)
    hf_sd = hf_model.state_dict()
    out_sd = custom_to_hf_state_dict(ck["model"], cfg.n_layer)

    # Load into HF model
    missing = []
    for k, v in hf_sd.items():
        if k in out_sd:
            if v.shape != out_sd[k].shape:
                raise RuntimeError(f"shape mismatch on {k}: {tuple(v.shape)} vs {tuple(out_sd[k].shape)}")
            hf_sd[k] = out_sd[k]
        elif k.endswith(".attn.bias") or k.endswith(".attn.masked_bias"):
            continue  # HF will regenerate these
        else:
            missing.append(k)
    hf_model.load_state_dict(hf_sd, strict=False)
    # tie embeddings explicitly
    hf_model.tie_weights()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    hf_model.save_pretrained(out)
    tok = GPT2TokenizerFast.from_pretrained(base_model)
    tok.save_pretrained(out)
    print(f"exported HF GPT-2 model to {out}")
    if missing:
        print(f"  (note) HF keys not overwritten by checkpoint: {missing[:5]}{'...' if len(missing)>5 else ''}")


def main() -> int:
    args = parse_args()
    export(args.checkpoint, args.base_model, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
