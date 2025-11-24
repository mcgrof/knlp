#!/bin/bash
# SPDX-License-Identifier: MIT
#
# next-defconfig-wait.sh - Queue next defconfig test after current completes
#
# Waits for any running training job to complete, then pulls latest code
# and starts the specified defconfig test. Useful for batching jobs on
# cloud computing where you want to queue the next experiment while
# the current one runs.
#
# Usage:
#   ./scripts/next-defconfig-wait.sh gpt2/defconfigs/gpt2-ra-ablation
#   ./scripts/next-defconfig-wait.sh gpt2/defconfigs/gpt2-kvsplice-ablation
#
# The script will:
#   1. Verify the defconfig exists
#   2. Wait for any running 'train' process to complete
#   3. Pull latest code from origin/main
#   4. Load the defconfig and run the experiment
#
# Note: This is a simple single-job queuing mechanism. For more complex
# batching with multiple queued jobs and job management, use a dedicated
# job scheduler or the upcoming Python batch script.

set -e

DEFCONFIG="$1"

if [ -z "$DEFCONFIG" ]; then
    echo "Usage: $0 <defconfig-path>"
    echo "Example: $0 gpt2/defconfigs/gpt2-ra-ablation"
    exit 1
fi

# Verify defconfig exists
if [ ! -f "$DEFCONFIG" ]; then
    echo "Error: defconfig not found: $DEFCONFIG"
    exit 1
fi

# Extract defconfig name for make target
DEFCONFIG_NAME=$(basename "$DEFCONFIG")
echo "Queued defconfig: $DEFCONFIG_NAME"
echo "Will run: make defconfig-$DEFCONFIG_NAME && make YES=1"
echo ""

# Wait for any training process to complete
echo "Checking for running training processes..."
while true; do
    if pgrep -f "train" > /dev/null 2>&1; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Training in progress, waiting 30s..."
        sleep 30
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') - No training process found"
        echo "Waiting 2 minutes before starting (in case job just finished)..."
        sleep 120

        # Double-check no new training started
        if pgrep -f "train" > /dev/null 2>&1; then
            echo "Training started during wait, continuing to monitor..."
            continue
        fi

        break
    fi
done

echo ""
echo "Pulling latest code from origin/main..."
git fetch origin
git reset --hard origin/main

echo ""
echo "Loading defconfig: $DEFCONFIG_NAME"
make defconfig-"$DEFCONFIG_NAME"

echo ""
echo "Starting experiment..."
make YES=1
