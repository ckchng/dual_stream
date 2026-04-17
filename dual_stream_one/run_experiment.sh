#!/bin/bash
# run_experiment.sh — train all configs, then evaluate together
#
# Usage: bash run_experiment.sh

set -e
cd "$(dirname "$0")"

echo "=========================================="
echo "STEP 1: Sequential training"
echo "=========================================="
bash seq_train.sh

echo "=========================================="
echo "STEP 2: Multi-model prediction"
echo "=========================================="
bash predict_multi_model.sh

echo "=========================================="
echo "Experiment complete."
echo "=========================================="
