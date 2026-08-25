#!/bin/bash
# Reproduce the AgentFloor tool-use capability ladder from the published corpus.
#
# No GPU and no model inference: this scores the authors' released run corpus
# and rebuilds their headline table. It exists to establish that the ladder is
# a usable foundation before any compute is spent extending it.
set -euo pipefail

ROOT="${1:-$HOME/agentfloor}"
REPO="https://github.com/rkarmaka/AgentFloor.git"
DATA="https://github.com/rkarmaka/AgentFloor/releases/download/v1.0-data"
HERE="$(cd "$(dirname "$0")" && pwd)"

[ -d "$ROOT" ] || git clone -q "$REPO" "$ROOT"
cd "$ROOT"

[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q PyYAML jsonschema openai

mkdir -p results
[ -f /tmp/agentfloor-runs-v1.tar.gz ] || \
  curl -sL -o /tmp/agentfloor-runs-v1.tar.gz "$DATA/agentfloor-runs-v1.tar.gz"
[ -f results/llm_judge_cache.jsonl ] || \
  curl -sL -o results/llm_judge_cache.jsonl "$DATA/llm_judge_cache.jsonl"
tar -xzf /tmp/agentfloor-runs-v1.tar.gz

# The released metrics code cannot score its own corpus; see the patch note.
git apply --check "$HERE/metrics-none-ci.patch" 2>/dev/null \
  && git apply "$HERE/metrics-none-ci.patch" \
  && echo "applied metrics-none-ci.patch" \
  || echo "patch already applied or upstream fixed it"

AGENTFLOOR_LLM_JUDGE=1 .venv/bin/python runs/run_metrics.py results/ \
  --subset paper_baseline
