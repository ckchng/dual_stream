#!/bin/bash
# ============================================================
# predict_multi_model.sh – run predict_multi_model.py
#
# Runs two evaluation passes:
#   1. both_run* dirs  →  bisenetv2dualmaskguidedv2
#   2. rt_run* dirs    →  bisenetv2
#
# Results for each model are saved under:
#   $OUTPUT_ROOT/<model-dir-name>/
# ============================================================

set -e

# ── Shared base directory ────────────────────────────────────
SAVE_ROOT="/home/ckchng/Documents/dual_stream/dual_stream_one/save/snr_1_32_len_200_for_m1/single_class/"

# ── Checkpoint filename inside each model dir ────────────────
CKPT_FILENAME="best.pth"

# ── Shared output root (per-model sub-dirs created automatically) ─
OUTPUT_ROOT="$SAVE_ROOT/"

# ── Paths ────────────────────────────────────────────────────
IMG_DIR="/media/ckchng/internal2TB/FILTERED_IMAGES/"
ANNO_JSON="/home/ckchng/Documents/SDA_ODA/LMA_data/testing_data_label_with_dual_single_stage_and_rt_two_stage_labeled_merged_labels.json"

# ── Shared settings ───────────────────────────────────────────
ENCODER=""                  # leave empty if not used (smp models only)
DECODER=""                  # leave empty if not used (smp models only)
NUM_CLASSES=1
USE_AUX=true
DEVICE="auto"               # auto | cpu | cuda | cuda:0
BATCH_SIZE=128
SCALE=1.0
MEAN="0.30566086 0.30566086 0.30566086"
STD="0.21072077 0.21072077 0.21072077"
MEAN2="0.34827731 0.34827731 0.34827731"
STD2="0.16927711 0.16927711 0.16927711"
TILE_SIZE=288
TILE_STRIDE=144
NUM_ANGLES=192
NUM_RHOS=416
# RHO_MIN_CAP=-144
# RHO_MAX_CAP=143
SEP=true
SEP_PARAMS="3.0 6 6.0 0.6 6.0 0.1"
PALETTE="cityscapes"
BLEND=true
BLEND_ALPHA=0.3

# ── Per-group config ──────────────────────────────────────────
BOTH_MODEL="bisenetv2dualmaskguidedv2"
BOTH_DIRS=("both_run1" "both_run2" "both_run3" "both_run4" "both_run5")
# BOTH_DIRS=("both_run1")

RT_MODEL="bisenetv2"
# RT_DIRS=("rt_run1")
RT_DIRS=("rt_run1" "rt_run2" "rt_run3" "rt_run4" "rt_run5")

# ============================================================
cd "$(dirname "$0")"

run_predict() {
    local model="$1"
    shift
    local dirs=("$@")

    local full_dirs=()
    for d in "${dirs[@]}"; do
        full_dirs+=("$SAVE_ROOT/$d")
    done

    echo ""
    echo "============================================================"
    echo "Running predict_multi_model.py"
    echo "  Model  : $model"
    echo "  Dirs   : ${dirs[*]}"
    echo "  Output : $OUTPUT_ROOT"
    echo "  SEP    : $SEP"
    echo "============================================================"

    local ARGS=(
        --model         "$model"
        --ckpt-filename "$CKPT_FILENAME"
        --model-dirs    "${full_dirs[@]}"
        --img-dir       "$IMG_DIR"
        --anno-json     "$ANNO_JSON"
        --output-root   "$OUTPUT_ROOT"
        --num-classes   "$NUM_CLASSES"
        --device        "$DEVICE"
        --batch-size    "$BATCH_SIZE"
        --scale         "$SCALE"
        --mean          $MEAN
        --std           $STD
        --mean2         $MEAN2
        --std2          $STD2
        --tile-size     "$TILE_SIZE"
        --tile-stride   "$TILE_STRIDE"
        --num_angles    "$NUM_ANGLES"
        --num_rhos      "$NUM_RHOS"
        --sep-params    $SEP_PARAMS
        --palette       "$PALETTE"
        --blend-alpha   "$BLEND_ALPHA"
    )

    [ -n "$ENCODER" ]      && ARGS+=(--encoder "$ENCODER")
    [ -n "$DECODER" ]      && ARGS+=(--decoder "$DECODER")
    [ -n "$RHO_MIN_CAP" ]  && ARGS+=(--rho-min-cap "$RHO_MIN_CAP")
    [ -n "$RHO_MAX_CAP" ]  && ARGS+=(--rho-max-cap "$RHO_MAX_CAP")
    [ "$USE_AUX" = true ]  && ARGS+=(--use-aux)
    [ "$BLEND"    = true ] && ARGS+=(--blend)
    [ "$SEP"      = true ] && ARGS+=(--sep true) || ARGS+=(--sep false)

    python predict_multi_model.py "${ARGS[@]}"
}

# ── Run 1: SEP=true ──────────────────────────────────────────
SEP=true
# run_predict "$BOTH_MODEL" "${BOTH_DIRS[@]}"
run_predict "$RT_MODEL"   "${RT_DIRS[@]}"

# ── Run 2: SEP=false ─────────────────────────────────────────
SEP=false
# run_predict "$BOTH_MODEL" "${BOTH_DIRS[@]}"
# run_predict "$RT_MODEL"   "${RT_DIRS[@]}"

echo ""
echo "Done. Results saved under: $OUTPUT_ROOT"
