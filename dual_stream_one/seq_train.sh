#!/usr/bin/env bash
#SBATCH -p a100cpu
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --time=48:00:00
#SBATCH --mem=32GB


set -e

PYTHON="python"
SCRIPT="main.py"
WORKDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORKDIR"

TRAIN_DATAROOT="/home/ckchng/Documents/SDA_ODA/LMA_data/snr_1_32_len_200_for_m1/"
VAL_DATAROOT="/home/ckchng/Documents/SDA_ODA/LMA_data/snr_1_32_len_200_for_m1/"
SAVE_BASE="/home/ckchng/Documents/dual_stream/dual_stream_one/save/snr_1_32_len_200_for_m1/single_class"
NUM_CLASS=1
LOSS_TYPE="ohem_bce"

# ─── Configurations ────────────────────────────────────────────────────────────
# Format: "dataset:model:save_subdir:run_start:run_end"
CONFIGS=(
    # "customdualmask:bisenetv2dualmaskguidedv2:both:1:5"
    "custom:bisenetv2:rt:4:5"
)

# ─── Run ───────────────────────────────────────────────────────────────────────
total=0
for cfg in "${CONFIGS[@]}"; do
    IFS=':' read -r dataset model subdir run_start run_end <<< "$cfg"
    for run in $(seq "$run_start" "$run_end"); do
        total=$((total + 1))
    done
done

current=0
for cfg in "${CONFIGS[@]}"; do
    IFS=':' read -r dataset model subdir run_start run_end <<< "$cfg"
    for run in $(seq "$run_start" "$run_end"); do
        current=$((current + 1))
        save_dir="${SAVE_BASE}/${subdir}_run${run}/"
        echo "=========================================="
        echo "Run ${current}/${total}: dataset=${dataset} model=${model} run=${run}"
        echo "  save_dir=${save_dir}"
        echo "=========================================="
        EXTRA_ARGS=()
        [ -n "$TRAIN_DATAROOT" ] && EXTRA_ARGS+=(--train_dataroot "$TRAIN_DATAROOT")
        [ -n "$VAL_DATAROOT" ]   && EXTRA_ARGS+=(--val_dataroot   "$VAL_DATAROOT")
        $PYTHON $SCRIPT \
            --dataset   "$dataset" \
            --num_class "$NUM_CLASS" \
            --model     "$model" \
            --loss_type "$LOSS_TYPE" \
            --save_dir  "$save_dir" \
            "${EXTRA_ARGS[@]}"
        echo "Run ${current}/${total} complete."
    done
done

echo "All ${total} training runs complete."
