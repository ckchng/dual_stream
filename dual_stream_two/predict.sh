#!/bin/bash
# ============================================================
# predict.sh – run predict.py across multiple checkpoints
#
# Set SAVE_ROOT to the shared base directory.
# Each entry in EXPERIMENTS is just the subfolder name under SAVE_ROOT.
# The script automatically resolves:
#   ckpt       → $SAVE_ROOT/<name>/best.pth
#   output_dir → $SAVE_ROOT/<name>/single_stage/vis
# ============================================================

# ── Shared base directory ────────────────────────────────────
SAVE_ROOT="/home/ckchng/Documents/dual_stream/dual_stream_two/save/bg_50_no_crop/snr_1_25_wo_borders/two_classes"

# ── Checkpoint filename & output sub-path ────────────────────
CKPT_FILENAME="best.pth"
OUTPUT_SUBDIR="single_stage/vis"

# ── Experiments to evaluate (subfolder names under SAVE_ROOT) ─
EXPERIMENTS=(
    "both_run2"
    "both_run3"
    "rt_run2"
    "rt_run3"
    # Add more experiment names below:
    # "both_run3"
    # "another_experiment"
)

# ── Paths ────────────────────────────────────────────────────
IMG_DIR="/media/ckchng/internal2TB/FILTERED_IMAGES/"
ANNO_JSON="/home/ckchng/Documents/SDA_ODA/LMA_data/testing_data_label_with_dual_single_stage_and_rt_two_stage_labeled_merged_labels.json"

# ── Model ────────────────────────────────────────────────────
MODEL="bisenetv2dualmaskguidedv2"
ENCODER=""                  # leave empty if not used (smp models only)
DECODER=""                  # leave empty if not used (smp models only)
NUM_CLASSES=2
USE_AUX=true                # set to true/false

# ── Device & batch ───────────────────────────────────────────
DEVICE="auto"               # auto | cpu | cuda | cuda:0
BATCH_SIZE=128

# ── Preprocessing ────────────────────────────────────────────
SCALE=1.0
MEAN="0.39509313 0.39509313 0.39509313"
STD="0.17064099 0.17064099 0.17064099"
MEAN2="0.34827731 0.34827731 0.34827731"
STD2="0.16927711 0.16927711 0.16927711"

# ── Tiling ───────────────────────────────────────────────────
TILE_SIZE=288
TILE_STRIDE=144             # overlap = tile_size - tile_stride

# ── RT transform ─────────────────────────────────────────────
NUM_ANGLES=192
NUM_RHOS=288
RHO_MIN_CAP=-144
RHO_MAX_CAP=143

# ── Star/blob separation ─────────────────────────────────────
SEP=true                    # set to true/false
# sep_params: thresh_sigma  minarea  elong_max  r_eff_min  r_eff_max  hough_thresh
SEP_PARAMS="3.0 6 5.5 0.6 6.0 0.1"

# ── Visualisation ────────────────────────────────────────────
PALETTE="cityscapes"        # binary | cityscapes | none
BLEND=true                  # set to true/false
BLEND_ALPHA=0.3

# ============================================================
# Run over each checkpoint
# ============================================================
cd "$(dirname "$0")"

total=${#EXPERIMENTS[@]}
idx=0

for exp in "${EXPERIMENTS[@]}"; do
    idx=$((idx + 1))
    CKPT="$SAVE_ROOT/$exp/$CKPT_FILENAME"
    OUTPUT_DIR="$SAVE_ROOT/$exp/$OUTPUT_SUBDIR"

    echo ""
    echo "============================================================"
    echo "[$idx/$total] Checkpoint : $CKPT"
    echo "             Output dir  : $OUTPUT_DIR"
    echo "============================================================"

    ARGS=(
        --model "$MODEL"
        --ckpt "$CKPT"
        --img-dir "$IMG_DIR"
        --anno-json "$ANNO_JSON"
        --output-dir "$OUTPUT_DIR"
        --num-classes "$NUM_CLASSES"
        --device "$DEVICE"
        --batch-size "$BATCH_SIZE"
        --scale "$SCALE"
        --mean $MEAN
        --std $STD
        --mean2 $MEAN2
        --std2 $STD2
        --tile-size "$TILE_SIZE"
        --tile-stride "$TILE_STRIDE"
        --num_angles "$NUM_ANGLES"
        --num_rhos "$NUM_RHOS"
        --rho-min-cap "$RHO_MIN_CAP"
        --rho-max-cap "$RHO_MAX_CAP"
        --sep-params $SEP_PARAMS
        --palette "$PALETTE"
        --blend-alpha "$BLEND_ALPHA"
    )

    [ -n "$ENCODER" ]      && ARGS+=(--encoder "$ENCODER")
    [ -n "$DECODER" ]      && ARGS+=(--decoder "$DECODER")
    [ "$USE_AUX" = true ]  && ARGS+=(--use-aux)
    [ "$BLEND" = true ]    && ARGS+=(--blend)
    [ "$SEP" = true ]      && ARGS+=(--sep true) || ARGS+=(--sep false)

    python predict.py "${ARGS[@]}"
done

echo ""
echo "All $total checkpoint(s) done."
