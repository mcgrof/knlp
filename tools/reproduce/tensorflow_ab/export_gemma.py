# SPDX-License-Identifier: MIT
"""Export a tokenizer-free KerasHub Gemma SavedModel and deterministic feeds."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_case(case_name: str) -> tuple[int, int]:
    batch, sequence = case_name.split("_")
    return int(batch[1:]), int(sequence[1:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="gemma_2b_en")
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--feeds-output", type=Path, required=True)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["b1_s32", "b4_s32", "b1_s128"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, default=256000)
    args = parser.parse_args()

    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    import keras_hub  # noqa: PLC0415
    import tensorflow as tf  # noqa: PLC0415

    model = keras_hub.models.CausalLM.from_preset(args.preset)

    class ExportModule(tf.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        @tf.function(
            input_signature=[
                tf.TensorSpec([None, None], tf.float32, name="token_ids"),
                tf.TensorSpec([None, None], tf.float32, name="padding_mask"),
            ]
        )
        def serve(self, token_ids, padding_mask):
            hidden = self.backbone(
                {
                    "token_ids": token_ids,
                    "padding_mask": padding_mask,
                },
                training=False,
            )
            logits = self.backbone.token_embedding(hidden, reverse=True)
            return {"output_0": logits}

    module = ExportModule(model.backbone)
    concrete = module.serve.get_concrete_function()
    tf.saved_model.save(
        module,
        str(args.model_output),
        signatures={"serve": concrete},
    )

    args.feeds_output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    manifest = {
        "preset": args.preset,
        "seed": args.seed,
        "vocab_size": args.vocab_size,
        "cases": {},
    }
    for case_name in args.cases:
        batch, sequence = parse_case(case_name)
        token_ids = rng.integers(
            0,
            args.vocab_size,
            size=(batch, sequence),
            dtype=np.int32,
        ).astype(np.float32)
        padding_mask = np.ones((batch, sequence), dtype=np.float32)
        token_path = args.feeds_output / f"token_ids_{case_name}.bin"
        mask_path = args.feeds_output / f"padding_mask_{case_name}.bin"
        token_ids.tofile(token_path)
        padding_mask.tofile(mask_path)
        manifest["cases"][case_name] = {
            "shape": [batch, sequence],
            "token_ids": str(token_path),
            "padding_mask": str(mask_path),
        }
    (args.feeds_output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
