#!/bin/bash
# Sequential training script — 6 configurations from settings.txt
#
# PREREQUISITE: In main.py, uncomment the load_parser line:
#   config = load_parser(config)
#
# Usage: bash train_sequential.sh

set -e  # Exit immediately on any failure

PYTHON="python"
SCRIPT="main.py"
WORKDIR="$(cd "$(dirname "$0")" && pwd)"

cd "$WORKDIR"

# ─── Setting 1 ────────────────────────────────────────────────────────────────
echo "=========================================="
echo "Starting training: Setting 1 (customdualmask, 2-class, bisenetv2dualmaskguidedv2, ohem)"
echo "=========================================="
$PYTHON $SCRIPT \
    --dataroot "/home/ckchng/Documents/SDA_ODA/LMA_data/snr_1_25_wo_borders/" \
    --dataset "customdualmask" \
    --num_class 2 \
    --model "bisenetv2dualmaskguidedv2" \
    --loss_type "ohem" \
    --save_dir "/home/ckchng/Documents/dual_stream_two/save/bg_50_no_crop/snr_1_25_wo_borders/two_classes/both_run2/"

echo "=========================================="
echo "Starting training: Setting 2 (customdualmask, 2-class, bisenetv2dualmaskguidedv2, ohem)"
echo "=========================================="
$PYTHON $SCRIPT \
    --dataroot "/home/ckchng/Documents/SDA_ODA/LMA_data/snr_1_25_wo_borders/" \
    --dataset "customdualmask" \
    --num_class 2 \
    --model "bisenetv2dualmaskguidedv2" \
    --loss_type "ohem" \
    --save_dir "/home/ckchng/Documents/dual_stream_two/save/bg_50_no_crop/snr_1_25_wo_borders/two_classes/both_run3/"

echo "=========================================="
echo "Starting training: Setting 3 (custom, 2-class, bisenetv2, ohem)"
echo "=========================================="
$PYTHON $SCRIPT \
    --dataroot "/home/ckchng/Documents/SDA_ODA/LMA_data/snr_1_25_wo_borders/" \
    --dataset "custom" \
    --num_class 2 \
    --model "bisenetv2" \
    --loss_type "ohem" \
    --save_dir "/home/ckchng/Documents/dual_stream_two/save/bg_50_no_crop/snr_1_25_wo_borders/two_classes/rt_run2/"

echo "=========================================="
echo "Starting training: Setting 4 (custom, 2-class, bisenetv2, ohem)"
echo "=========================================="
$PYTHON $SCRIPT \
    --dataroot "/home/ckchng/Documents/SDA_ODA/LMA_data/snr_1_25_wo_borders/" \
    --dataset "custom" \
    --num_class 2 \
    --model "bisenetv2" \
    --loss_type "ohem" \
    --save_dir "/home/ckchng/Documents/dual_stream_two/save/bg_50_no_crop/snr_1_25_wo_borders/two_classes/rt_run3/"
