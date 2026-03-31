#!/bin/bash
# Run exoplanet-detection-period oracle
TASK_DIR=/workspace/skillsbench/tasks/exoplanet-detection-period
RESULT_DIR=/workspace/tier1-results/exoplanet-detection-period/oracle
mkdir -p "$RESULT_DIR/logs"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting exoplanet-detection-period / oracle"

# Environment already set up, just verify data and deps
ls /root/data/ 2>/dev/null || {
    echo "Copying data..."
    mkdir -p /root/data
    cp -r "$TASK_DIR/environment/data/"* /root/data/
}
python3 -c "import lightkurve; print('lightkurve OK')" 2>&1

# Run oracle
echo "Running oracle solution..."
cd /root
bash "$TASK_DIR/solution/solve.sh" > "$RESULT_DIR/logs/agent_stdout.log" 2>&1 || {
    echo "WARNING: Oracle exited with code $?"
    tail -30 "$RESULT_DIR/logs/agent_stdout.log"
}

# Run tests
echo "Running tests..."
rm -rf /tests
mkdir -p /tests /logs/verifier
cp "$TASK_DIR/tests/"* /tests/
pip install pytest pytest-json-ctrf -q 2>/dev/null || true

cd /root
python3 -m pytest /tests/test_outputs.py -rA -v --tb=short 2>&1 | tee "$RESULT_DIR/logs/test_output.log" || true

# Count results
python3 << 'PYEOF'
import json
result_dir = "/workspace/tier1-results/exoplanet-detection-period/oracle"
lines = open(f"{result_dir}/logs/test_output.log").readlines()
p = sum(1 for l in lines if " PASSED" in l and "::" in l)
f = sum(1 for l in lines if " FAILED" in l and "::" in l)
e = sum(1 for l in lines if " ERROR" in l and "::" in l)
total = p + f + e
rate = round(p / max(total, 1), 3)
result = {
    "reward": 1.0 if p == total and total > 0 else 0.0,
    "pass": p == total and total > 0,
    "passed": p,
    "failed": f + e,
    "total": total,
    "pass_rate": rate
}
with open(f"{result_dir}/result.json", "w") as fh:
    json.dump(result, fh)
print(f"RESULT: {'PASS' if result['pass'] else 'FAIL'} ({p}/{total}, rate={rate})")
print(json.dumps(result))
PYEOF
