#!/usr/bin/env python3
"""CAS hard-negative entity-binding self-study (Pro strategy #3, 2026-07-29).

Same decoupled pipeline as cas_synth_diverse.py (bot A = Qwen3-14B question
generator, bot B = Qwen3-8B answer/distillation teacher, via the library
SelfStudySynthesizer). The ONLY change vs the diverse arm is bot A's seed: it
must build a five-option multiple-choice question whose four WRONG options are
other REAL values of the SAME type drawn from the SAME patient's record.

Motivation: CAS's own analysis reports cartridges retrieve valid values but
attach them to the WRONG entity. Open-ended QA (our diverse arm) does not punish
that -- a roughly-correct list scores low loss. LongHealth's options are
same-patient plausible alternatives, so the cartridge must distinguish the
correct patient-specific relation from near-identical distractors. This arm makes
the TRAINING distribution force that discrimination: same-patient same-type hard
negatives, bidirectional (entity->value and value->entity), and minimal pairs
(before/after, confirmed/ruled-out). Distractors are drawn by the 14B generator
from the record in context -- benchmark-BLIND (the eval questions are never used;
that was the separate sacrificial arm). This tests whether reshaping the training
distribution toward hard-negative binding closes the isolated-cartridge gap.

Env mirrors cas_synth_diverse.py: PATIENT, NUM_SAMPLES, CLIENT_A/B_URL/MODEL,
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

# Hard-negative binding MODES (benchmark-blind). Each convo draws one so the
# corpus practices forward, inverse, and minimal-pair discrimination rather than
# only one binding direction. Every mode demands SAME-PATIENT SAME-TYPE
# distractors, which is the whole point (entity/value confusion, not coverage).
MODES = [
    "an ENTITY-to-VALUE question: pick one specific medication, lab test, "
    "procedure, or diagnosis from the record and ask for its EXACT dose, result, "
    "date, or status",
    "the INVERSE VALUE-to-ENTITY question: pick one specific dose, lab result, "
    "date, or finding and ask WHICH medication, test, procedure, or diagnosis it "
    "belongs to",
    "a DATE-binding question: ask on what date a specific event occurred, or "
    "which event occurred on a specific date",
    "a RESULT-binding question: ask what result belonged to a specific test, or "
    "which test produced a specific result",
    "a POLARITY minimal-pair question: ask which condition was CONFIRMED (vs one "
    "that was considered but RULED OUT), or which treatment WAS administered (vs "
    "one that was NOT)",
    "a TEMPORAL minimal-pair question: ask for a dose/finding BEFORE vs AFTER a "
    "specific procedure or visit, where the two answers differ",
]
SEED_TMPL = (
    "Using ONLY the record above, write {mode}. Format it as a single "
    "multiple-choice question with exactly five options labelled A, B, C, D, E. "
    "CRITICAL: the ONE correct option must be exactly right per the record, and "
    "ALL FOUR wrong options must be OTHER REAL values of the SAME TYPE that "
    "actually appear elsewhere in THIS patient's record (e.g. other real doses, "
    "other real dates, other real test results, other real medications). Never "
    "invent a value and never use a generic or obviously-wrong option -- every "
    "distractor must be a genuine same-type value from this record so the "
    "question is a hard discrimination. Include the specific ids, names, dates, "
    "and numbers needed to make the target unambiguous. Output only the question "
    "and its five options A-E, with no answer key and no other text."
)


class LongHealthHardNegResource(LongHealthResource):
    """LongHealthResource whose seed forces a hard-negative binding MCQ with
    same-patient same-type distractors, instead of the generic 'question' seed."""

    class Config(LongHealthResource.Config):
        pass

    async def sample_prompt(self, batch_size: int):
        ctx, _ = await super().sample_prompt(batch_size)
        seeds = [SEED_TMPL.format(mode=random.choice(MODES)) for _ in range(batch_size)]
        return ctx, seeds


PATIENT = os.environ.get("PATIENT", "patient_01")
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "1500"))
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
        resources=[LongHealthHardNegResource.Config(patient_ids=[PATIENT])],
    ),
    num_samples=NUM_SAMPLES,
    batch_size=16,
    max_num_batches_in_parallel=int(os.environ.get("MAX_BATCHES", "8")),
    worker_timeout=int(os.environ.get("WORKER_TIMEOUT", str(40 * 60))),
    output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
    name=FormatStringVariable(f"synth_hardneg_14bq_8ba_{pstr}_n{{num_samples}}"),
    run_id=FormatStringVariable(f"synth_hardneg_14bq_8ba_{pstr}_n{{num_samples}}"),
    wandb=WandBConfig(
        tags=["synth", "hardneg", "qwen3-14bq", "qwen3-8ba", "longhealth"]
    ),
    upload_to_wandb=False,
    save_wandb_preview=False,
    upload_to_hf=False,
)

if __name__ == "__main__":
    pydrantic.main([config])
