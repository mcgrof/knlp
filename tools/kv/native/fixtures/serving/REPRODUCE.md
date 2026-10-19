# Replicate the dense Qwen3 8K HTTP study

Use the [validated H100 installation](https://knlp.io/asymmetric-install.html)
with the `current` recipe first. This procedure adds the HTTP production,
quality, serial-parity and fixed-plan checks. It uses one H100 SXM 80 GB,
22 CPU threads, FP16 weights/compute and a 40 GiB persistent cache budget.
It can take several GPU-hours. The two context studies have separate designs;
these commands reproduce **8K**, not the earlier batched 1K experiment.

The CPU readers in the current checkout are later analysis tools. Run the
GPU entry points from their measured source, in a clean separate worktree:

```sh
git fetch origin
git checkout --detach 749a566aed06bb7d642087f310bdcb9bb73d926e
TOOLS_SOURCE="$PWD"
STACK="$PWD/stack"                    # Existing validated installation
MEASURED_SOURCE="$PWD/../knlp-http-measured"
CAPTURES="$PWD/../http-replication-captures"
PLAN="$PWD/../http-replication-plan"
git worktree add --detach "$MEASURED_SOURCE" \
  0e34f048a82420a3a7b7a5d6d27254d4cd7692b8
mkdir "$CAPTURES"
export OMP_NUM_THREADS=22 MKL_NUM_THREADS=22
export FLASHINFER_WORKSPACE_BASE="$STACK/jit"
```

Keep the installed FlashInfer pin
`800a550f8b736e49f8cfa136b82bfaf6d9d32711`. Candidate vLLM is
`276305b29e0d11f95ada503ba76323ab24992e59`; both native controls use
`c21c0878210695db7b108cdc285aaf0cf890190a`. The installation uses editable
Python sources over the verified compatible binary wheel. Switch that source
checkout only between completed engine processes. The server verifies its
required source pin before launching.

For offline execution, download the **complete** model snapshot first. A
weights-only cache can fail vLLM's outer snapshot check before engine startup:

```sh
"$STACK/env/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-8B",
                  revision="b968826d9c46dd6066d109eabc6255188de91218")
PY
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

Use the same `HF_HOME` for the download and all engines if you override it.
Model files remain separate downloads under the publisher's license. The
public fixtures contain the original effective scales and tokenized tasks.

## Qualify each arm on the replication machine

Run sequentially, with no other GPU workload. These pilots are excluded from
confirmation. Preserve every output and failure; never mix captures from
changed source, scales, runtime policy or model revision.

```sh
cd "$MEASURED_SOURCE"
for arm in candidate native_fi native_fa3; do
  case "$arm" in
    candidate) pin=276305b29e0d11f95ada503ba76323ab24992e59 ;;
    *) pin=c21c0878210695db7b108cdc285aaf0cf890190a ;;
  esac
  git -C "$STACK/vllm" checkout --detach "$pin" || exit 1
  "$STACK/env/bin/python" -m tools.kv.native.serving_pilot \
    --install "$STACK" --format fp16 --arm "$arm" \
    --length 8192 --cache-bytes 42949672960 \
    --output "$CAPTURES/fp16-$arm-pilot" || exit 1
  "$STACK/env/bin/python" -m tools.kv.native.serving_quality \
    --install "$STACK" --format fp16 --arm "$arm" \
    --length 8192 --cache-bytes 42949672960 \
    --output "$CAPTURES/fp16-$arm-http-quality" || exit 1
  "$STACK/env/bin/python" -m tools.kv.native.serving_reference reference \
    --install "$STACK" --format fp16 --arm "$arm" \
    --length 8192 --cache-bytes 42949672960 --sequential-reference \
    --output "$CAPTURES/fp16-$arm-sequential-reference" || exit 1
done
```

The heavy pilot checks run inside the actual HTTP worker. The quality capture
includes NLL, retrieval, all 1,319 GSM8K questions and 32 serial HTTP parity
requests. The reference command supplies both in-process prompt orders. A
successful capture alone is not a quality decision.

Return to the current tools checkout and let the independent CPU audits decide
whether this replication qualifies. The output directory must not exist:

```sh
cd "$TOOLS_SOURCE"
"$STACK/env/bin/python" -m tools.kv.native.serving_replay_plan \
  --design tools/kv/native/fixtures/serving/fp16-8k/DESIGN.json \
  --catalog tools/kv/native/fixtures/serving/fp16-8k/catalog.json.gz \
  --manifest tools/kv/native/fixtures/current/fp16/qwen.json.gz \
  --captures "$CAPTURES" --prefix fp16 --production-tag pilot \
  --source-tree "$MEASURED_SOURCE" --output "$PLAN"
```

This command restores exact trace bytes, reruns production and quality readers,
checks matching source/protocol identities, and emits 33 plans plus their
SHA-256 values in `REPLICATION.json`. It launches no GPU work. Failure leaves
the audit evidence; it does not issue qualified plans. Historical pass flags
are not inputs to this command.

## Execute and read the frozen study

Record the generated plan hashes before timing. Follow `DESIGN.json`:
blocks 0 through 9 in order, each block's listed three-arm order, then the
three descriptive arm runs. Each plan already fixes its cell order, original
arrival trace, warmup, timeout and no-retry policy. Switch to the appropriate
vLLM pin before each arm, as above, and run from `MEASURED_SOURCE`:

```sh
# Example: the first frozen arm. Use the hash emitted in REPLICATION.json.
cd "$MEASURED_SOURCE"
git -C "$STACK/vllm" checkout --detach \
  276305b29e0d11f95ada503ba76323ab24992e59
"$STACK/env/bin/python" -m tools.kv.native.serving_trial \
  --plan "$PLAN/plan-0-candidate.json" \
  --plan-sha256 SHA256_FROM_REPLICATION_JSON \
  --install "$STACK" --output "$CAPTURES/fp16-8k-0-candidate"
```

Use output names `fp16-8k-BLOCK-ARM`, including `descriptive` as the final
block label. Retain nonzero exit status and partial output for every failed
launch; never replace it with another block. The reader supports a failed
launch receipt at `captures/jobs/launch-fp16-8k-BLOCK-ARM-result.json` with
`{"passed": false, "returncode": ACTUAL_EXIT_STATUS}`. Write such a receipt
only for an observed failure; absent evidence remains pending.

After all scheduled runs finish, read the fixed family from the current tools
checkout. Incomplete primary blocks get no reduced-subset inference:

```sh
cd "$TOOLS_SOURCE"
"$STACK/env/bin/python" -m tools.kv.native.serving_confirmation \
  --root "$CAPTURES" --plan "$PLAN" --prefix fp16-8k \
  --output "$CAPTURES/confirmation-audit.json"
"$STACK/env/bin/python" -m tools.kv.native.serving_admission_audit \
  --root "$CAPTURES" --confirmation "$CAPTURES/confirmation-audit.json" \
  --prefix fp16-8k --output "$CAPTURES/admission-audit.json"
```

The admission supplement verifies the final reader's scheduler hashes and
separates newly admitted prompts, their mixing with decode, and recomputation
after preemption. It describes scheduling; it does not infer kernel dispatch
or add performance comparisons. It requires a complete scheduled study,
with failed launches retained, and uses every available captured cell.

This is an exact-workload replication, including the historical absolute
arrival rates. A different machine may place those rates at a different point
on its load curve. Selecting new rates, counts, sources or execution policies
requires a separately frozen study; do not relabel it as this exact design.
