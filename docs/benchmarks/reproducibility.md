# Reproducibility Checklist

This document covers everything required to make a fused KV
quantization benchmark run reproducible by a third party. Use it
as a pre-submission gate: every item must be satisfied before
results are published or archived.

For the full evaluation protocol, see the
[runbook](../fused_kv_benchmark_runbook.md). For a quick first
run, see the [quickstart](quickstart.md).

---

## 1. Environment Pinning

These artifacts must be present in every results directory. Without
them, a third party cannot reproduce the run.

### Required artifacts

| Artifact | How to generate | Why it matters |
|----------|----------------|----------------|
| `vllm_commit.txt` | `git log --oneline -1` in the vLLM source tree | Pins the exact serving framework version. Different commits can change KV cache layout, kernel dispatch, or scheduling. |
| `collect_env.txt` | `python -c "import vllm; vllm.utils.collect_env()"` | Records PyTorch version, CUDA/ROCm version, GPU driver, and Python version in one file. |
| `common.env` | Hand-written (see [quickstart](quickstart.md#step-2-write-a-shared-config)) | Ensures FP16 and FUSED runs use identical model, TP degree, max-model-len, dtype, and seed. |
| `config_diff.txt` | Hand-written | Documents the single flag difference between FP16 and FUSED. If more than one flag differs, the comparison is invalid. |

### Verification

```bash
# After collecting artifacts, verify:
test -f $RESULTS_DIR/vllm_commit.txt || echo "MISSING: vllm_commit.txt"
test -f $RESULTS_DIR/collect_env.txt || echo "MISSING: collect_env.txt"
test -f $RESULTS_DIR/common.env      || echo "MISSING: common.env"
test -f $RESULTS_DIR/config_diff.txt || echo "MISSING: config_diff.txt"
```

---

## 2. Controlled Variables

The FP16 and FUSED runs must be identical in every respect except
the KV cache quantization flag. Check each item:

- [ ] **Model weights**: Same HuggingFace model ID or local
  checkpoint path
- [ ] **Tensor-parallel degree**: Same TP value
- [ ] **max-model-len**: Same maximum sequence length
- [ ] **gpu-memory-utilization**: Same fraction (typically 0.90)
- [ ] **dtype**: Same dtype setting (typically `auto`)
- [ ] **Seed**: Same random seed (typically 42)
- [ ] **GPU hardware**: Same GPU model, same driver version, same
  number of GPUs
- [ ] **vLLM commit**: Same commit hash for both runs
- [ ] **torch.compile status**: Both enabled or both disabled
- [ ] **Dataset**: Same evaluation data, same preprocessing

If any of these differ between FP16 and FUSED, the comparison
is invalid and must be re-run.

---

## 3. Benchmark Completeness

### Full suite (top-venue submission)

All 11 phases from the runbook must be completed:

- [ ] Phase 1: Startup time (3 runs per config, report median)
- [ ] Phase 2: lm-eval standard tasks (MMLU, HellaSwag, ARC,
  WinoGrande, GSM8K, TruthfulQA)
- [ ] Phase 3: Latency at batch=1 (sanity check)
- [ ] Phase 4: NIAH (retrieval heatmap)
- [ ] Phase 5: Latency full sweep (batch x length matrix)
- [ ] Phase 6: Throughput (saturated engine)
- [ ] Phase 7: GuideLLM sweep (open-loop serving)
- [ ] Phase 8: Serving rate sweep (Poisson-arrival clients)
- [ ] Phase 9: RULER (synthetic long-context tasks)
- [ ] Phase 10: LongBench (real-world long-document tasks)
- [ ] Phase 11: InfiniteBench (100K+ token stress tests)

### Minimal subset (workshop paper or technical report)

At minimum, these four must be completed (runbook Section 6):

- [ ] lm-eval (mmlu + hellaswag + arc_challenge, 5-shot)
- [ ] Latency (batch=1 and batch=8 at input_len=512 and 8192)
- [ ] Throughput (input_len=512, output_len=128, 1000 prompts)
- [ ] NIAH (context lengths 4096, 16384, 32768; 5 depth intervals)

---

## 4. Pass/Fail Criteria

Apply these thresholds before claiming results. They are defined
in the runbook (Section 7) and repeated here for convenience.

### Accuracy (lm-eval)

| Delta | Verdict |
|-------|---------|
| Within 1% on all tasks | PASS |
| Within 3% with justification | Acceptable |
| Beyond 3% on any task | FAIL --- debug before proceeding |

### NIAH retrieval

| Result | Verdict |
|--------|---------|
| 100% at all (depth, length) pairs | PASS |
| Any cell below 95% | Investigate --- likely cache corruption |

### Latency and throughput

| Claim | Requirement |
|-------|-------------|
| "Fused is faster" | At least 5% improvement at one or more operating points |
| "Serving improvement" | Measurable gains at realistic request rates (not just saturated throughput) |
| "No regression" | No statistically significant latency increase at any operating point |

### Long-context (RULER, LongBench, InfiniteBench)

| Result | Verdict |
|--------|---------|
| FUSED accuracy slope matches FP16 | PASS |
| FUSED degrades faster than FP16 with length | Quantization errors compounding --- investigate |

---

## 5. Result Artifacts

### Required JSON outputs

Every benchmark tool produces JSON. All of these must be saved:

| Benchmark | Expected output files |
|-----------|-----------------------|
| GuideLLM | `guidellm/{fp16,fused}_sweep.json`, `guidellm/{fp16,fused}_sweep_long.json` |
| Latency | `bench/{fp16,fused}_latency_b{1,4,8,16,32}.log` |
| Throughput | `bench/{fp16,fused}_throughput.log` |
| Serving | `bench/{fp16,fused}_serving_rr{1,2,4,8,16}.json` |
| Startup | `bench/{fp16,fused}_startup.log` (3 runs each) |
| lm-eval | `lm_eval/{fp16,fused}_standard.json` |
| NIAH | `niah/{fp16,fused}/` (per-cell results) |
| RULER | `ruler/{fp16,fused}/` (per-task, per-length results) |
| LongBench | `longbench/{fp16,fused}/predictions/`, `longbench/{fp16,fused}/scores/` |
| InfiniteBench | `infinitebench/{fp16,fused}/` |

### Directory structure

See the [README](README.md#result-directory-structure) for the
full tree layout.

---

## 6. Archival Procedure

Results must be committed to the key-results repo, not the code
repo. The code repo stays slim.

```bash
# Copy results
cp -a $RESULTS_DIR/ /data/knlp-key-results/fused_kv_bench/

# Commit to key-results repo
cd /data/knlp-key-results
git add fused_kv_bench/
git commit -m "bench: fused KV results $(date +%Y%m%d)"
```

Include in the commit message:
- Model name and size
- GPU type and count
- vLLM commit hash
- Whether this is a full suite or minimal subset run

---

## 7. Common Mistakes

These are failure modes observed in prior benchmark runs:

**Missing `add_bos_token=True` in lm-eval.** Most causal LMs
condition on BOS. Omitting this flag silently degrades scores,
making FP16 look worse than it is and masking FUSED regressions.

**Comparing across vLLM commits.** KV cache layout and kernel
dispatch change between commits. FP16 on commit A vs FUSED on
commit B is not a valid comparison.

**Cherry-picking operating points.** Report the full sweep data.
Claiming "2x speedup at batch=32" while hiding a regression at
batch=1 is misleading. The runbook requires full sweep data to
be available.

**Measuring NIAH with too few depth intervals.** Quantization
errors can be depth-sensitive. Use at least 5 depth intervals
(0%, 25%, 50%, 75%, 100%) for a meaningful heatmap.

**Forgetting to kill the vLLM server between configs.** Running
FUSED benchmarks against an FP16 server (or vice versa) produces
invalid results. Always restart the server when switching configs.

**Running long-context benchmarks without enough VRAM.** NIAH at
131072 tokens requires substantial KV cache memory. If the server
OOMs silently, retrieval scores drop to zero and look like a
quantization failure. Check server logs for OOM errors.

---

## 8. Pre-Submission Summary

Before submitting or publishing results, confirm:

- [ ] All environment artifacts present (Section 1)
- [ ] Controlled variables verified (Section 2)
- [ ] Required benchmarks completed (Section 3)
- [ ] All pass/fail criteria evaluated (Section 4)
- [ ] All JSON artifacts saved (Section 5)
- [ ] Results archived to key-results repo (Section 6)
- [ ] No common mistakes apply (Section 7)
- [ ] Results reproducible from committed configs and pinned vLLM

When all items are checked, the results are submission-ready.
