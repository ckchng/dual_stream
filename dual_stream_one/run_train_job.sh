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
DATAROOT="${DATAROOT}"
DATASET="${DATASET}"
NUM_CLASS="${NUM_CLASS}"
MODEL="${MODEL}"
LOSS_TYPE="${LOSS_TYPE}"
SAVE_DIR="${SAVE_DIR}"

echo "Training: dataset=${DATASET} model=${MODEL} save_dir=${SAVE_DIR}"

"$PYTHON" "$SCRIPT" \
    --dataroot  "$DATAROOT" \
    --dataset   "$DATASET" \
    --num_class "$NUM_CLASS" \
    --model     "$MODEL" \
    --loss_type "$LOSS_TYPE" \
    --save_dir  "$SAVE_DIR"

echo "Done: ${SAVE_DIR}"
