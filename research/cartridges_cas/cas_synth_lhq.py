#!/usr/bin/env python3
"""Sacrificial diagnostic (ChatGPT-Pro): can a cartridge learn LongHealth-style
access AT ALL when trained on the exact target question distribution?

CONTAMINATING -- a diagnostic, never a reproduction score. For one patient we
paraphrase its OWN LongHealth questions (train split only) with the 14B bot-A,
have the 8B teacher answer each grounded in the full record (with top-20
distillation logprobs), and train a cartridge on those. We then evaluate on the
patient's HELD-OUT LongHealth questions (never paraphrased, never seen).

Interpretation:
  * held-out >> 0.5  -> the cartridge CAN learn LongHealth access; our generic
    synthesis is simply misaligned to LongHealth's operations.
  * held-out ~ 0.5   -> even matching the exact task distribution doesn't help;
    the wall is deeper than question style (points at implementation / the
    authors' unpublished specifics).

Same working decoupled self_study pipeline as cas_synth_diverse (correct parquet
+ logprobs), only the seed changes: bot A rephrases a train-split LongHealth MC
question verbatim-in-meaning; bot B answers it grounded in the record.

Env: PATIENT, TRAIN_Q_END (questions [0,TRAIN_Q_END) are train; rest held out;
default 15), NUM_SAMPLES, CLIENT_A/B_URL/MODEL, PROB_THINKING, MAXTOK_B, MAX_BATCHES.
"""

import os
import random

os.environ.setdefault("CARTRIDGES_DIR", os.path.expanduser("~/cartridges"))
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", os.path.expanduser("~/cas_out"))
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.openai import OpenAIClient
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.data.longhealth.resources import LongHealthResource
from cartridges.data.longhealth.utils import load_longhealth_dataset
from cartridges.utils.wandb import WandBConfig

PATIENT = os.environ.get("PATIENT", "patient_01")
TRAIN_Q_END = int(os.environ.get("TRAIN_Q_END", "15"))
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "1500"))
CLIENT_A_URL = os.environ.get("CLIENT_A_URL", "http://localhost:8105/v1")
CLIENT_A_MODEL = os.environ.get("CLIENT_A_MODEL", "qwen3-14b-qgen")
CLIENT_B_URL = os.environ.get("CLIENT_B_URL", "http://localhost:8005/v1")
CLIENT_B_MODEL = os.environ.get("CLIENT_B_MODEL", "qwen3-8b-ans")
PROB_THINKING = float(os.environ.get("PROB_THINKING", "1.0"))
MAXTOK_B = int(os.environ.get("MAXTOK_B", "2048"))
pstr = PATIENT.replace("patient_", "p")

# Build the train-split LongHealth questions (with options) for this patient.
_pat = load_longhealth_dataset([PATIENT])[0]
_train_q = _pat.questions[:TRAIN_Q_END]


def _fmt(q):
    return (
        f"{q.question}\n"
        f"A. {q.answer_a}\nB. {q.answer_b}\nC. {q.answer_c}\n"
        f"D. {q.answer_d}\nE. {q.answer_e}"
    )


REPHRASE = (
    "Rephrase the following multiple-choice question in your own words. Preserve "
    "the exact clinical meaning and ALL five options, keeping every specific name, "
    "date, dosage, and numeric value intact. Output only the rephrased question "
    "followed by its five options (A-E), nothing else.\n\n{q}"
)


class LongHealthQResource(LongHealthResource):
    """Seeds bot A with a rephrase of a train-split LongHealth question, so the
    synthetic training distribution matches the benchmark's own question forms."""

    class Config(LongHealthResource.Config):
        pass

    async def sample_prompt(self, batch_size: int):
        ctx, _ = await super().sample_prompt(batch_size)
        seeds = [
            REPHRASE.format(q=_fmt(random.choice(_train_q))) for _ in range(batch_size)
        ]
        return ctx, seeds


client_a = OpenAIClient.Config(
    model_name=CLIENT_A_MODEL, base_url=CLIENT_A_URL, api_key="EMPTY"
)
client_b = OpenAIClient.Config(
    model_name=CLIENT_B_MODEL, base_url=CLIENT_B_URL, api_key="EMPTY"
)

config = SynthesizeConfig(
    synthesizer=SelfStudySynthesizer.Config(
        client=client_a,
        client_b=client_b,
        max_rounds=1,
        temperature_a=0.7,
        max_completion_tokens_a=1024,
        prob_thinking=PROB_THINKING,
        temperature_b=0.0,
        max_completion_tokens_b=MAXTOK_B,
        use_tools_a=False,
        use_tools_b=False,
        tools=[],
        num_top_logprobs=20,
        min_prob_mass=0.998,
        resources=[LongHealthQResource.Config(patient_ids=[PATIENT])],
    ),
    num_samples=NUM_SAMPLES,
    batch_size=16,
    max_num_batches_in_parallel=int(os.environ.get("MAX_BATCHES", "8")),
    worker_timeout=int(os.environ.get("WORKER_TIMEOUT", str(40 * 60))),
    output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
    name=FormatStringVariable(f"synth_lhq_14bq_8ba_{pstr}_n{{num_samples}}"),
    run_id=FormatStringVariable(f"synth_lhq_14bq_8ba_{pstr}_n{{num_samples}}"),
    wandb=WandBConfig(tags=["synth", "lhq-sacrificial", "longhealth"]),
    upload_to_wandb=False,
    save_wandb_preview=False,
    upload_to_hf=False,
)

if __name__ == "__main__":
    pydrantic.main([config])
