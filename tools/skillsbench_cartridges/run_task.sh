#!/bin/bash
# Direct benchmark harness v2 — runs a task without Docker
# Usage: run_task.sh <task_dir> <arm>
#   arm: oracle | no-skill | full-skill
# Output: /workspace/tier1-results/<task>/<arm>/

set -eo pipefail
TASK_DIR="$1"
ARM="$2"
TASK_NAME=$(basename "$TASK_DIR")

RESULT_DIR="/workspace/tier1-results/${TASK_NAME}/${ARM}"
mkdir -p "$RESULT_DIR" "$RESULT_DIR/logs"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting $TASK_NAME / $ARM"

# --- 1. Set up environment from Dockerfile ---
python3 /workspace/setup_env.py "$TASK_DIR" "$ARM" 2>&1 | tee "$RESULT_DIR/logs/setup.log"

# --- 2. Install test deps (scan test files for imports) ---
TEST_DIR="$TASK_DIR/tests"
if [ -d "$TEST_DIR" ]; then
    # Extract imports from test files and install common ones
    EXTRA_DEPS=$(python3 -c "
import ast, sys
from pathlib import Path
test_dir = Path('$TEST_DIR')
imports = set()
for f in test_dir.glob('*.py'):
    try:
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split('.')[0])
    except: pass
# Map known import names to pip packages
mapping = {
    'polars': 'polars', 'rapidfuzz': 'rapidfuzz', 'openpyxl': 'openpyxl',
    'fastexcel': 'fastexcel', 'yaml': 'pyyaml', 'PIL': 'pillow',
    'cv2': 'opencv-python', 'sklearn': 'scikit-learn', 'bs4': 'beautifulsoup4',
    'lxml': 'lxml', 'pandas': 'pandas', 'numpy': 'numpy',
}
stdlib = {'json','re','os','sys','pathlib','zipfile','xml','io','typing','collections',
          'math','subprocess','shutil','tempfile','unittest','dataclasses','enum',
          'functools','itertools','copy','textwrap','hashlib','datetime','time','struct',
          'csv','glob','abc','string','operator','contextlib','warnings','ast','inspect',
          'importlib','difflib','configparser','argparse','logging','traceback','pprint',
          'statistics','fractions','decimal','random','secrets','uuid','base64','hmac',
          'socket','http','urllib','email','mimetypes','ftplib','smtplib','xmlrpc',
          'ctypes','multiprocessing','threading','concurrent','asyncio','signal','platform'}
needed = []
for imp in imports:
    if imp in stdlib or imp == 'pytest':
        continue
    pkg = mapping.get(imp, imp)
    needed.append(pkg)
print(' '.join(sorted(set(needed))))
" 2>/dev/null)
    pip install pytest pytest-json-ctrf $EXTRA_DEPS -q 2>/dev/null || true
fi

# Also install any deps from test.sh (look for uv add / pip install)
if [ -f "$TEST_DIR/test.sh" ]; then
    UV_DEPS=$(grep -oP 'uv add \K\S+' "$TEST_DIR/test.sh" 2>/dev/null | tr '\n' ' ')
    if [ -n "$UV_DEPS" ]; then
        echo "Installing test.sh deps: $UV_DEPS"
        pip install $UV_DEPS -q 2>/dev/null || true
    fi
fi

# --- 3. Copy test ground truth files ---
if [ -d "$TEST_DIR" ]; then
    rm -rf /tests
    mkdir -p /tests
    cp -r "$TEST_DIR/"* /tests/ 2>/dev/null || true
fi

# --- 4. Run solution/agent ---
if [ "$ARM" = "oracle" ]; then
    echo "Running oracle solution..."
    cd /root
    bash "$TASK_DIR/solution/solve.sh" > "$RESULT_DIR/logs/agent_stdout.log" 2>&1 || {
        echo "WARNING: Oracle exited with code $?, last 20 lines:"
        tail -20 "$RESULT_DIR/logs/agent_stdout.log"
    }
fi

# --- 5. Run tests ---
echo "Running tests..."
mkdir -p /logs/verifier
cd /root
python3 -m pytest /tests/test_outputs.py -rA -v --tb=short 2>&1 | tee "$RESULT_DIR/logs/test_output.log" || true

# --- 6. Collect results ---
read PASSED FAILED TOTAL <<< $(python3 -c "
lines = open('$RESULT_DIR/logs/test_output.log').readlines()
p = sum(1 for l in lines if ' PASSED' in l and '::' in l)
f = sum(1 for l in lines if ' FAILED' in l and '::' in l)
e = sum(1 for l in lines if ' ERROR' in l and '::' in l)
print(p, f + e, p + f + e)
" 2>/dev/null || echo "0 0 0")

if [ "$TOTAL" -eq 0 ]; then
    echo "RESULT: ERROR (no tests collected)"
    echo "{\"reward\": 0.0, \"pass\": false, \"passed\": 0, \"failed\": 0, \"total\": 0, \"pass_rate\": 0.0, \"error\": \"no_tests_collected\"}" > "$RESULT_DIR/result.json"
elif [ "$PASSED" -eq "$TOTAL" ]; then
    echo "RESULT: PASS ($PASSED/$TOTAL tests passed)"
    echo "{\"reward\": 1.0, \"pass\": true, \"passed\": $PASSED, \"failed\": 0, \"total\": $TOTAL, \"pass_rate\": 1.0}" > "$RESULT_DIR/result.json"
else
    RATE=$(python3 -c "print(round($PASSED / max($TOTAL, 1), 3))")
    echo "RESULT: FAIL ($PASSED/$TOTAL tests passed, rate=$RATE)"
    echo "{\"reward\": 0.0, \"pass\": false, \"passed\": $PASSED, \"failed\": $FAILED, \"total\": $TOTAL, \"pass_rate\": $RATE}" > "$RESULT_DIR/result.json"
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Done $TASK_NAME / $ARM"
cat "$RESULT_DIR/result.json"
