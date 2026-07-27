#!/usr/bin/env python3
"""CAS self-study with a DECOUPLED question model (bot A) and answer model
(bot B), the Cartridges-at-Scale (arXiv:2606.04557) recipe where the data
generator is a stronger, separate model than the student teacher.

Bot A (questions) is served at CLIENT_A_URL; bot B (answers + top-logprob
distillation targets) at CLIENT_B_URL. This uses the repo's SelfStudy
synthesizer with the `client_b` decouple patch: bot A keeps `self.client`,
bot B routes through `self.client_b`. Per-patient so each document gets its
own training corpus (the CAS split step); run once per patient. The four
non-creative seed rounds are used (question / structuring / summarization /
use_case); creative is excluded per the paper's footnote 12. The answer
model runs with thinking on, matching the paper's M_A.

Env:
  PATIENT       one patient id (default patient_01)
  NUM_SAMPLES   conversations for this patient (default 2000)
  CLIENT_A_URL  bot A (q-gen) endpoint  (default http://localhost:8006/v1)
  CLIENT_A_MODEL bot A served-model-name (default qwen3-14b-qgen)
  CLIENT_B_URL  bot B (answer) endpoint (default http://localhost:8005/v1)
  CLIENT_B_MODEL bot B served-model-name (default qwen3-8b-ans)
"""

import os

os.environ.setdefault("CARTRIDGES_DIR", "/home/mcgrof/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/home/mcgrof/cas_out")
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
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "2000"))
CLIENT_A_URL = os.environ.get("CLIENT_A_URL", "http://localhost:8006/v1")
CLIENT_A_MODEL = os.environ.get("CLIENT_A_MODEL", "qwen3-14b-qgen")
CLIENT_B_URL = os.environ.get("CLIENT_B_URL", "http://localhost:8005/v1")
CLIENT_B_MODEL = os.environ.get("CLIENT_B_MODEL", "qwen3-8b-ans")
pstr = PATIENT.replace("patient_", "p")

# Bot A: the stronger question generator (Qwen3-14B).
client_a = OpenAIClient.Config(
    model_name=CLIENT_A_MODEL,
    base_url=CLIENT_A_URL,
    api_key="EMPTY",
)
# Bot B: the answer / distillation teacher (Qwen3-8B), the model the
# cartridge is trained to imitate.
client_b = OpenAIClient.Config(
    model_name=CLIENT_B_MODEL,
    base_url=CLIENT_B_URL,
    api_key="EMPTY",
)

config = SynthesizeConfig(
    synthesizer=SelfStudySynthesizer.Config(
        client=client_a,
        client_b=client_b,
        max_rounds=1,
        temperature_a=0.6,
        max_completion_tokens_a=1024,
        # Match the iso5-baseline answer config (synth_pod) so the ONLY variable
        # vs baseline is the question generator (14B decoupled bot A). Always-on
        # thinking (prob_thinking=1.0, max 2048) is ~14 h/patient here -- too slow
        # for the pilot; keep the baseline's 0.2 mix and 1024 cap.
        prob_thinking=0.2,
        temperature_b=0.0,
        max_completion_tokens_b=1024,
        use_tools_a=False,
        use_tools_b=False,
        tools=[],
        num_top_logprobs=20,
        min_prob_mass=0.998,
        resources=[
            LongHealthResource.Config(
                seed_prompts=[
                    "structuring",
                    "summarization",
                    "question",
                    "use_case",
                ],
                patient_ids=[PATIENT],
            )
        ],
    ),
    num_samples=NUM_SAMPLES,
    batch_size=16,
    # 16 x 16 = 256 concurrent = the top of the library's recommended
    # batch_size*num_batches range (128-256, see synthesize.py). Safe on bot B's
    # ~370k-token KV cache now that answers are short (prob_thinking=0.2, 1024
    # cap). 16x32=512 thrashed the cache; 16x8 with always-thinking was too slow.
    max_num_batches_in_parallel=16,
    worker_timeout=15 * 60,
    output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
    name=FormatStringVariable(f"synth_decoupled_14bq_8ba_{pstr}_n{{num_samples}}"),
    run_id=FormatStringVariable(f"synth_decoupled_14bq_8ba_{pstr}_n{{num_samples}}"),
    wandb=WandBConfig(
        tags=["synth", "decoupled", "qwen3-14bq", "qwen3-8ba", "longhealth"]
    ),
    upload_to_wandb=False,
    save_wandb_preview=False,
    upload_to_hf=False,
)

if __name__ == "__main__":
    pydrantic.main([config])
