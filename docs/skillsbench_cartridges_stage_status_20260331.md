# SkillsBench cartridges stage status — 2026-03-31

## Durable artifact root
Full SkillsBench/cartridge artifact bundle is being collected into:
- `/data/knlp-key-results/cartridges/skillsbench-cartridges-20260331T202500Z/`

This bundle includes:
- `tier1-results/`
- `cartridges/`
- evaluation harness files
- exoplanet skill text

## Code landed in knlp
Copied into:
- `/data/knlp/tools/skillsbench_cartridges/`

Files:
- `eval_cartridge.py`
- `eval_codegen_retry_v2.py`
- `train_cartridge.py`
- `run_exoplanet.sh`
- `run_task.sh`
- `run_powerlifting_eval.py`
- `exoplanet_all_skills.txt`

## What was tested
### Citation-check at 7B
- full-skill: `5/9`
- no-skill: `5/9`
- all cartridge variants: `3/9`

Interpretation:
- bad first discriminant
- task is API/tool gated in a way that single-shot completion cannot realize
- cartridge KV injection also caused parroting / generation corruption on this task

### Exoplanet-detection-period at 32B AWQ
Key result summary from the preserved A100 lane:
- oracle: `4/4`
- full-skill (32B): `3/4`
- no-skill (32B): `3/4`
- cartridge default-75%: `1/4`
- cartridge default-50%: `0/4`
- cartridge sci-50%: `0/4`

## Main finding
The exoplanet run produced the first useful cartridge signal:

- **full-skill and cartridge variants changed the models approach**
