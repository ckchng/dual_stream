#!/usr/bin/env bash
#SBATCH -p a100cpu
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --time=48:00:00
#SBATCH --mem=32GB

set -euo pipefail

PYTHON="/hpcfs/users/a1775493/ck/conda_env/ds/bin/python"
SCRIPT="/hpcfs/users/a1775493/ck/SDA_OTE/dual_stream/dual_stream_one/main.py"

# Receive parameters injected by train_on_server.sh via --export
DATAROOT="${DATAROOT:-}"
TRAIN_DATAROOT="${TRAIN_DATAROOT:-}"
VAL_DATAROOT="${VAL_DATAROOT:-}"
TRAIN_MASK_ROOT="${TRAIN_MASK_ROOT:-}"
VAL_MASK_ROOT="${VAL_MASK_ROOT:-}"
DATASET="${DATASET}"
NUM_CLASS="${NUM_CLASS}"
MODEL="${MODEL}"
LOSS_TYPE="${LOSS_TYPE}"
SAVE_DIR="${SAVE_DIR}"

echo "Training: dataset=${DATASET} model=${MODEL} save_dir=${SAVE_DIR}"

EXTRA_ARGS=()
[ -n "$DATAROOT" ]        && EXTRA_ARGS+=(--dataroot        "$DATAROOT")
[ -n "$TRAIN_DATAROOT" ]  && EXTRA_ARGS+=(--train_dataroot  "$TRAIN_DATAROOT")
[ -n "$VAL_DATAROOT" ]    && EXTRA_ARGS+=(--val_dataroot    "$VAL_DATAROOT")
[ -n "$TRAIN_MASK_ROOT" ] && EXTRA_ARGS+=(--train_mask_root "$TRAIN_MASK_ROOT")
[ -n "$VAL_MASK_ROOT" ]   && EXTRA_ARGS+=(--val_mask_root   "$VAL_MASK_ROOT")

"$PYTHON" "$SCRIPT" \
    --dataset   "$DATASET" \
    --num_class "$NUM_CLASS" \
    --model     "$MODEL" \
    --loss_type "$LOSS_TYPE" \
    --save_dir  "$SAVE_DIR" \
    "${EXTRA_ARGS[@]}"

echo "Done: ${SAVE_DIR}"
