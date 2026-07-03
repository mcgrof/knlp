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

if [ "$fail" -eq 0 ]; then echo "SMOKE PASS"; else echo "SMOKE FAIL"; fi
exit "$fail"
