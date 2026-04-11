#!/usr/bin/env bash
#SBATCH -p a100cpu
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --time=48:00:00
#SBATCH --mem=32GB



set -euo pipefail

# PYTHON=/home/ckchng/conda_env/pose_estimation/bin/python
PYTHON=/hpcfs/users/a1775493/ck/conda_env/dual_stream/bin/python
SCRIPT="/hpcfs/users/a1775493/ck/SDA_OTE/dual_stream/data_generation/training_data_gen_for_RT.py"

# ── IO ─────────────────────────────────────────────────────────────────────────
# IMG_DIR="/home/ckchng/Documents/SDA_ODA/LMA_data/background_patches_with_new_model"
IMG_DIR=/hpcfs/users/a1775493/ck/SDA_OTE/dual_stream/data/bg_data/
OUTPUT_DIR="/hpcfs/users/a1775493/ck/SDA_OTE/dual_stream/data/tmp/"

# ── Range selection ────────────────────────────────────────────────────────────
STARTING_ID=100
STEP=100
NUM_ANGLES=192
NUM_RHOS=416

# ── Rho caps (comment out both lines to disable capping) ──────────────────────
# RHO_MIN_CAP=-144
# RHO_MAX_CAP=143

# ── Hyperparameters ────────────────────────────────────────────────────────────
SNR_MIN_1=1.6;   SNR_MAX_1=2.5
SNR_MIN_2=1.25;  SNR_MAX_2=2.0

SIGMA_MIN_1=0.75; SIGMA_MAX_1=1.3
SIGMA_MIN_2=1.25; SIGMA_MAX_2=2.0

LENGTH_MIN_1=100; LENGTH_MAX_1=400
LENGTH_MIN_2=50;  LENGTH_MAX_2=600
LENGTH_MIN_3=601; LENGTH_MAX_3=1000

SNR_RATIO=1.0
SIGMA_RATIO=1.0
LENGTH_RATIO_1=1.0
LENGTH_RATIO_2=0.66

LC_WIDTH=3
MAX_NUM_STREAK=1
SCALE_FLAG=1
LINE_MASK_THICKNESS=5

# ── Build rho-cap args conditionally ──────────────────────────────────────────
RHO_ARGS=""
# Uncomment the two lines below to enable rho capping:
# RHO_ARGS="--rho_min_cap ${RHO_MIN_CAP} --rho_max_cap ${RHO_MAX_CAP}"

# ── Run ────────────────────────────────────────────────────────────────────────
"$PYTHON" "$SCRIPT" \
    --img_dir        "$IMG_DIR" \
    --output_dir     "$OUTPUT_DIR" \
    --starting_id    "$STARTING_ID" \
    --step           "$STEP" \
    --num_angles     "$NUM_ANGLES" \
    --num_rhos       "$NUM_RHOS" \
    --snr_min_1      "$SNR_MIN_1" \
    --snr_max_1      "$SNR_MAX_1" \
    --snr_min_2      "$SNR_MIN_2" \
    --snr_max_2      "$SNR_MAX_2" \
    --sigma_min_1    "$SIGMA_MIN_1" \
    --sigma_max_1    "$SIGMA_MAX_1" \
    --sigma_min_2    "$SIGMA_MIN_2" \
    --sigma_max_2    "$SIGMA_MAX_2" \
    --length_min_1   "$LENGTH_MIN_1" \
    --length_max_1   "$LENGTH_MAX_1" \
    --length_min_2   "$LENGTH_MIN_2" \
    --length_max_2   "$LENGTH_MAX_2" \
    --length_min_3   "$LENGTH_MIN_3" \
    --length_max_3   "$LENGTH_MAX_3" \
    --snr_ratio      "$SNR_RATIO" \
    --sigma_ratio    "$SIGMA_RATIO" \
    --length_ratio_1 "$LENGTH_RATIO_1" \
    --length_ratio_2 "$LENGTH_RATIO_2" \
    --lc_width       "$LC_WIDTH" \
    --max_num_streak "$MAX_NUM_STREAK" \
    --scale_flag     "$SCALE_FLAG" \
    --line_mask_thickness "$LINE_MASK_THICKNESS" \
    $RHO_ARGS
