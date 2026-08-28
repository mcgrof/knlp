#!/usr/bin/env python3
"""CAS diverse-FORM self-study.

Same decoupled pipeline as cas_synth_decoupled.py (bot A = Qwen3-14B question
generator, bot B = Qwen3-8B answer/distillation teacher, via the library
SelfStudySynthesizer with the client_b decouple patch), but bot A is instructed
to generate questions of specific HARD COMPOSITIONAL FORMS instead of the
library's generic single-fact "question" seed (which just asks "generate a
question ... include details").

Motivation (2026-07-28 coverage audit + failure-mode inspection): the trained
cartridge FAILS negation, temporal-ordering, entity-value-binding, and cross-note
comparison questions even though the underlying facts ARE present in the
self-study corpus (wrong-question answer-fact coverage 0.94 ~ right-question
0.96). So the gap is question-form diversity / generalization, not fact coverage,
and not answer quality (thinking-on was flat). The forms below are GENERAL
clinical question types (CAS Appendix I "mix factual recall, comparison,
reasoning, detail-oriented" + standard negation/temporal/binding), generated from
the NOTES, never from the eval questions -- benchmark-blind.

Env mirrors cas_synth_decoupled.py: PATIENT, NUM_SAMPLES, CLIENT_A/B_URL/MODEL,
PROB_THINKING, MAXTOK_B, MAX_BATCHES, WORKER_TIMEOUT.
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
from cartridges.utils.wandb import WandBConfig

# General clinical question FORMS (benchmark-blind). Each convo draws one, so the
# corpus practices every retrieval form rather than only flat factual recall.
FORMS = [
    "a factual-recall question about one specific value, date, dose, medication, "
    "lab result, or diagnosis",
    "a comparison question that contrasts two medications, visits, timepoints, "
    "lab values, or diagnoses",
    "a temporal-ordering question about the sequence or order in which events, "
    "examinations, or medication changes occurred",
    "a negation or exception question -- e.g. which statement is NOT accurate, "
    "what was ruled out, or what was discontinued and why",
    "an entity-value binding question that links a specific medication or finding "
    "to its EXACT dose, timing, date, or result",
    "a multi-fact reasoning question that requires combining several details from "
    "the note to answer",
]
SEED_TMPL = (
    "Generate {form} based on the record above. Include the specific ids, names, "
    "dates, and numerical values needed so it is clear what you are asking about "
    "and it can be answered closed-book. Output only the single question, with no "
    "other text or explanation."
)


class LongHealthDiverseResource(LongHealthResource):
    """LongHealthResource whose seed prompt forces a random hard question FORM,
    instead of the library's generic single-fact 'question' seed."""

    class Config(LongHealthResource.Config):
        pass

    async def sample_prompt(self, batch_size: int):
        # reuse the parent's context (patient/note sampling); replace only the
        # seed prompts with form-directed instructions.
        ctx, _ = await super().sample_prompt(batch_size)
        seeds = [SEED_TMPL.format(form=random.choice(FORMS)) for _ in range(batch_size)]
        return ctx, seeds


PATIENT = os.environ.get("PATIENT", "patient_01")
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "2000"))
CLIENT_A_URL = os.environ.get("CLIENT_A_URL", "http://localhost:8105/v1")
CLIENT_A_MODEL = os.environ.get("CLIENT_A_MODEL", "qwen3-14b-qgen")
CLIENT_B_URL = os.environ.get("CLIENT_B_URL", "http://localhost:8005/v1")
CLIENT_B_MODEL = os.environ.get("CLIENT_B_MODEL", "qwen3-8b-ans")
PROB_THINKING = float(os.environ.get("PROB_THINKING", "1.0"))
MAXTOK_B = int(os.environ.get("MAXTOK_B", "2048"))
pstr = PATIENT.replace("patient_", "p")

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
        temperature_a=0.6,
        max_completion_tokens_a=1024,
        prob_thinking=PROB_THINKING,
        temperature_b=0.0,
        max_completion_tokens_b=MAXTOK_B,
        use_tools_a=False,
        use_tools_b=False,
        tools=[],
        num_top_logprobs=20,
        min_prob_mass=0.998,
        resources=[LongHealthDiverseResource.Config(patient_ids=[PATIENT])],
    ),
    num_samples=NUM_SAMPLES,
    batch_size=16,
    max_num_batches_in_parallel=int(os.environ.get("MAX_BATCHES", "8")),
    worker_timeout=int(os.environ.get("WORKER_TIMEOUT", str(40 * 60))),
    output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
    name=FormatStringVariable(f"synth_diverse_14bq_8ba_{pstr}_n{{num_samples}}"),
    run_id=FormatStringVariable(f"synth_diverse_14bq_8ba_{pstr}_n{{num_samples}}"),
    wandb=WandBConfig(
        tags=["synth", "diverse", "qwen3-14bq", "qwen3-8ba", "longhealth"]
    ),
    upload_to_wandb=False,
    save_wandb_preview=False,
    upload_to_hf=False,
)

if __name__ == "__main__":
    pydrantic.main([config])
