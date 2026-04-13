#!/usr/bin/env bash
# submit_all_local.sh
# Runs run_training_data_gen.sh locally in a loop over starting_id.
# Jobs run sequentially by default; set MAX_JOBS > 1 for limited parallelism.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER_SCRIPT="$SCRIPT_DIR/run_training_data_gen.sh"

STEP=1000
START=0
END=30000    # exclusive upper bound
MAX_JOBS=15   # increase to run N jobs in parallel (be mindful of RAM/CPU)

job_count=0

for starting_id in $(seq "$START" "$STEP" $(( END - STEP ))); do
    echo "[$(date '+%H:%M:%S')] Starting batch starting_id=${starting_id} ..."
    STARTING_ID_OVERRIDE="$starting_id" STEP_OVERRIDE="$STEP" bash "$INNER_SCRIPT" &

    job_count=$(( job_count + 1 ))
    if (( job_count >= MAX_JOBS )); then
        wait -n 2>/dev/null || wait   # wait for any one job to finish
        job_count=$(( job_count - 1 ))
    fi
done

wait  # wait for remaining jobs
echo "All $(( (END - START) / STEP )) batches complete."
