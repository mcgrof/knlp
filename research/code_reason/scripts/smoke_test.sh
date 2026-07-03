#!/usr/bin/env bash
# Ticket 1 acceptance: both defconfigs load through the real Kconfig path and
# the generated config.json carries the expected flags. No models, no network.
# Restores any pre-existing .config so it does not clobber your working config.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

GEN="python3 research/code_reason/scripts/gen_config_json.py"
OUT="build/code_reason/config.json"
BAK=".config.code_reason_smoke_bak"
fail=0

[ -f .config ] && cp .config "$BAK"
restore() { [ -f "$BAK" ] && mv "$BAK" .config || rm -f .config; }
trap restore EXIT

assert() {  # assert <flag> <expected> <label>
  local got
  got=$(python3 -c "import json,sys; d=json.load(open('$OUT'))['flags']; print(d.get('$1', 'MISSING'))")
  if [ "$got" != "$2" ]; then
    echo "  FAIL: $3 -> $1 = $got (want $2)"; fail=1
  else
    echo "  ok:   $3 -> $1 = $got"
  fi
}

assert_off() {  # passes if the flag is off: False OR gated-out (MISSING)
  local got
  got=$(python3 -c "import json,sys; d=json.load(open('$OUT'))['flags']; print(d.get('$1', 'MISSING'))")
  if [ "$got" != "False" ] && [ "$got" != "MISSING" ]; then
    echo "  FAIL: $2 -> $1 = $got (want off: False or gated-out)"; fail=1
  else
    echo "  ok:   $2 -> $1 = $got (off)"
  fi
}

echo "=== [paper] make defconfig-code-reason-paper ==="
make defconfig-code-reason-paper >/dev/null 2>&1 || { echo "  FAIL: defconfig apply"; exit 1; }
$GEN --config .config --out "$OUT" >/dev/null || { echo "  FAIL: gen_config_json"; exit 1; }
assert CONFIG_CODE_REASON True "paper"
assert CONFIG_CODE_REASON_PAPER_REPRO True "paper"
assert CONFIG_CODE_REASON_AUGMENTED False "paper"
assert_off CONFIG_CODE_REASON_ADDENDUMS "paper"
assert_off CONFIG_CODE_REASON_ADDENDUM_COCCINELLE "paper"
assert CONFIG_CODE_REASON_NO_TARGET_TEST_EXECUTION True "paper"
assert CONFIG_CODE_REASON_NO_GIT_HISTORY True "paper"

echo "=== [augmented] make defconfig-code-reason ==="
make defconfig-code-reason >/dev/null 2>&1 || { echo "  FAIL: defconfig apply"; exit 1; }
$GEN --config .config --out "$OUT" >/dev/null || { echo "  FAIL: gen_config_json"; exit 1; }
assert CONFIG_CODE_REASON_AUGMENTED True "augmented"
assert CONFIG_CODE_REASON_ADDENDUMS True "augmented"
assert CONFIG_CODE_REASON_ADDENDUM_AST_RUNTIME True "augmented"
assert CONFIG_CODE_REASON_ADDENDUM_COCCINELLE True "augmented"
assert CONFIG_CODE_REASON_ADDENDUM_A_VS_BLB True "augmented"

echo "=== [manifest] build smoke dataset from the augmented config.json ==="
MAN="build/code_reason/smoke_manifest.jsonl"
if python3 research/code_reason/datasets/manifest_builder.py \
     --config "$OUT" --out "$MAN" >/dev/null 2>&1; then
  n=$(python3 -c "print(sum(1 for _ in open('$MAN')))")
  if [ "$n" -ge 1 ]; then
    echo "  ok:   manifest -> $n rows from config-enabled tasks"
  else
    echo "  FAIL: manifest produced 0 rows"; fail=1
  fi
else
  echo "  FAIL: manifest_builder"; fail=1
fi
python3 research/code_reason/datasets/manifest_builder.py --self-test \
  >/dev/null 2>&1 && echo "  ok:   manifest_builder self-test" \
  || { echo "  FAIL: manifest_builder self-test"; fail=1; }

echo "=== [e2e] manifest -> run -> score -> report (offline mock) ==="
E2E="build/code_reason/e2e"
rm -rf "$E2E"
if python3 - "$MAN" "$OUT" "$E2E" <<'PY' >/dev/null 2>&1
import json, sys
sys.path[:0] = [
    "research/code_reason/datasets", "research/code_reason/runners",
    "research/code_reason/reports", "research/code_reason/addendums",
    "research/code_reason/tools",
]
from runner import run_manifest
from metrics import score_run
from report import write_report
man, cfg, rd = sys.argv[1], sys.argv[2], sys.argv[3]
flags = json.load(open(cfg))["flags"]
rows = [json.loads(x) for x in open(man)]
ds_dir = "research/code_reason/datasets/fixtures/smoke"
run_manifest(rows, flags, rd, ds_dir, config_path=cfg, repo_root=".")
score_run(rd, ds_dir, rows)
write_report(rd)
PY
then
  [ -f "$E2E/report.md" ] && [ -f "$E2E/metrics_summary.json" ] \
    && echo "  ok:   e2e report + metrics written" \
    || { echo "  FAIL: e2e artifacts missing"; fail=1; }
else
  echo "  FAIL: e2e pipeline"; fail=1
fi

if [ "$fail" -eq 0 ]; then echo "SMOKE PASS"; else echo "SMOKE FAIL"; fi
exit "$fail"
