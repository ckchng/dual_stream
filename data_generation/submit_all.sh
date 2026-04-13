#!/usr/bin/env bash
# submit_all.sh
# Submits run_training_data_gen.sh as individual SLURM jobs,
# iterating starting_id from 0 to 30000 (exclusive) in steps of STEP.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER_SCRIPT="$SCRIPT_DIR/run_training_data_gen.sh"

STEP=1000
START=1000
END=30000   # exclusive upper bound

for starting_id in $(seq "$START" "$STEP" $(( END - STEP ))); do
    # Override STARTING_ID inside the inner script via env variable,
    # then submit as a separate SLURM job.
    sbatch --export=ALL,STARTING_ID_OVERRIDE="$starting_id",STEP_OVERRIDE="$STEP" \
           --job-name="datagen_${starting_id}" \
           "$INNER_SCRIPT"
done

echo "Submitted $(( (END - START) / STEP )) jobs (starting_id ${START} to $(( END - STEP )) in steps of ${STEP})."
