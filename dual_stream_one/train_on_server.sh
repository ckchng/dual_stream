#!/usr/bin/env bash
# train_on_server.sh
# Submits each training run as a separate SLURM job via run_train_job.sh.
#
# PREREQUISITE: In main.py, uncomment the load_parser line:
#   config = load_parser(config)
#
# Usage: bash train_on_server.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER_SCRIPT="$SCRIPT_DIR/run_train_job.sh"

# ─── Shared settings ───────────────────────────────────────────────────────────
TRAIN_DATAROOT="/hpcfs/users/a1775493/ck/SDA_OTE/dual_stream/data/snr_1_32_len_200_for_m1/"
VAL_DATAROOT="/hpcfs/users/a1775493/ck/SDA_OTE/val/"
SAVE_BASE="/hpcfs/users/a1775493/ck/SDA_OTE/dual_stream/dual_stream_one/save/snr_1_32_len_200_for_m1/single_class"
NUM_CLASS=1
LOSS_TYPE="ohem_bce"

# ─── Configurations ────────────────────────────────────────────────────────────
# Format: "dataset:model:save_subdir:run_start:run_end"
CONFIGS=(
    "customdualmask:bisenetv2dualmaskguidedv2:both:1:2"
    # "custom:bisenetv2:rt:1:5"
)

# ─── Submit ────────────────────────────────────────────────────────────────────
total=0
for cfg in "${CONFIGS[@]}"; do
    IFS=':' read -r dataset model subdir run_start run_end <<< "$cfg"
    for run in $(seq "$run_start" "$run_end"); do
        total=$((total + 1))
        save_dir="${SAVE_BASE}/${subdir}_run${run}/"
        sbatch \
            --export=ALL,DATASET="$dataset",MODEL="$model",SAVE_DIR="$save_dir",NUM_CLASS="$NUM_CLASS",LOSS_TYPE="$LOSS_TYPE",TRAIN_DATAROOT="$TRAIN_DATAROOT",VAL_DATAROOT="$VAL_DATAROOT" \
            --job-name="train_${model}_${subdir}_run${run}" \
            "$INNER_SCRIPT"
        echo "Submitted: dataset=${dataset} model=${model} subdir=${subdir} run=${run}"
    done
done

echo "Submitted ${total} job(s)."
