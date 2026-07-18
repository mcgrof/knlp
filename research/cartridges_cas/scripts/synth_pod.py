#!/usr/bin/env python3
"""Pod synthesis: LongHealth self-study data with Qwen3-8B served by vLLM (OpenAI-compatible
endpoint) via the repo's OpenAIClient -- the clean CUDA serving path (no LocalHFClient version
rot). Per-patient so each doc gets its own training data (the CAS 'split' step). Env:
  PATIENT      one patient id (default patient_01)  -> one doc = one cartridge's data
  NUM_SAMPLES  conversations for this patient (default 250)
  VLLM_URL     base_url of the vLLM OpenAI endpoint (default http://localhost:8000/v1)
Run once per patient; corpus is reused for both isolated and joint (mixed-visibility) training."""
import os
from pathlib import Path

os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/synth_out")
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.openai import OpenAIClient
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.data.longhealth.resources import LongHealthResource
from cartridges.utils.wandb import WandBConfig

PATIENT = os.environ.get("PATIENT", "patient_01")
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "250"))
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1")
pstr = PATIENT.replace("patient_", "p")

client = OpenAIClient.Config(
    model_name="Qwen/Qwen3-8B",
    base_url=VLLM_URL,
    api_key="EMPTY",  # vLLM ignores it; client requires a value
)

config = SynthesizeConfig(
    synthesizer=SelfStudySynthesizer.Config(
        client=client,
        max_rounds=1,
        prob_thinking=0.2,
        use_tools_a=False,
        use_tools_b=False,
        tools=[],
        num_top_logprobs=20,
        min_prob_mass=0.998,  # richer distillation targets (with flatten edge fix)
        resources=[
            LongHealthResource.Config(
                seed_prompts=[
                    "structuring",
                    "summarization",
                    "question",
                    "use_case",
                    "creative",
                ],
                patient_ids=[PATIENT],
            )
        ],
    ),
    num_samples=NUM_SAMPLES,
    batch_size=16,
    max_num_batches_in_parallel=32,
    output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
    name=FormatStringVariable(f"synth_qwen3_8b_lh_{pstr}_n{{num_samples}}"),
    run_id=FormatStringVariable(f"synth_qwen3_8b_lh_{pstr}_n{{num_samples}}"),
    wandb=WandBConfig(tags=["synth", "qwen3-8b", "longhealth", "pod"]),
    upload_to_wandb=False,
    save_wandb_preview=False,
    upload_to_hf=False,
)

if __name__ == "__main__":
    pydrantic.main([config])
