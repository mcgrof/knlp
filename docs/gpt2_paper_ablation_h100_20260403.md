# GPT-2 matched paper ablation on 1xH100

This document records the exact GPT-2 matched ablation lane used to compare the old FIM-trace selector against the newer attention-probability trace selector on a single H100. The point of the lane was not to redesign the training harness. It was to reuse the matched GPT-2 reproducer and add the attention-derived arm in the same budgeted setting.

The experiment uses three arms in the same harness family:

- baseline with RA disabled
- Arm A with RA enabled and layer selection driven by regular FIM trace
- Arm B with RA enabled and layer selection driven by attention-layer eigmax

The GPT-2 harness is `fim/reciprocal_attention/gpt2_matched.py`. The concrete configs live in `fim/reciprocal_attention/configs/` and the selector definitions live alongside them:

- `gpt2_baseline_1xh100.json`
- `gpt2_arm_a_fimtrace_1xh100.json`
- `gpt2_arm_b_eigmax_1xh100.json`
- smoke variants for each arm
- `ra_ablation_gpt2_arm_a_fimtrace.json`
- `ra_ablation_gpt2_arm_b_eigmax.json`

The selector budget matches the paper-fim GPT-2 lineage: eight total heads with the same RA mixing coefficients (`alpha_std=0.9375`, `alpha_rec=0.0625`). Arm A uses the older FIM-trace layer selector. Arm B uses the newer attention-probability trace proxy (`attn_layer_eigmax`) and the same head selector (`max_eigenvalue`).

The provenance of the newer selector matters and should be stated plainly. It is generated from the matched harness attention-statistics summary, not from an unrelated side metric. The generation logic lives in `fim/reciprocal_attention/gen_layer_ablation_configs.py`. Arm A selects candidate layers from `per_layer_traces` with `layer_selector="fim_trace"`. Arm B instead computes a per-layer score as the mean of the per-head maximum eigenvalues and uses `layer_selector="attn_layer_eigmax"`. The head-selection rule after that is still `max_eigenvalue` in both arms. So the ablation keeps the RA budget and head-ranking rule fixed while changing the layer-selection source from regular FIM trace to an attention-probability-derived eigmax proxy.

The empirical conclusion from the completed H100 run is straightforward. In this GPT-2 124M setting, regular FIM trace outperformed the attention-probability trace selector. Arm A beat Arm B on the clean seed-1337 run and also beat Arm B on the two follow-on seeds (42 and 7). The attention-derived selector did not beat the old FIM-trace selector in this matched GPT-2 comparison.

That does not mean the newer functional-trace-affinity direction is useless. The current read is narrower: at this GPT-2 / ~150M-ish scale, the attention-probability trace proxy is weaker than regular FIM trace in the matched 1-hour ablation. The value of functional trace affinities may emerge later, after this regime, when the model and behavior are rich enough for that signal to pay for itself.

The durable result bundle and seed-by-seed summary live in `knlp-key-results` under `paper-fim/gpt2-h100-paper-ablation/`.
