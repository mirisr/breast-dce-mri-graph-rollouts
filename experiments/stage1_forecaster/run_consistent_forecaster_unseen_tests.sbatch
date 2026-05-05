#!/bin/bash
#SBATCH --job-name=cg-fc-unseen
#SBATCH --output=experiments/stage1_forecaster/logs/cg-unseen-%A_%a.out
#SBATCH --error=experiments/stage1_forecaster/logs/cg-unseen-%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpul40q
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-6

# Parallel held-out generalization tests for the consistent-graph forecaster.
#
# This runs seven independent experiments:
#   - hold out ACRIN, train on ISPY2
#   - hold out ISPY2, train on ACRIN
#   - hold out each molecular subtype
#   - hold out a locked stratified 15% test set
#
# Each array task:
#   1. trains from scratch on the split's train list, using random internal val
#   2. evaluates the best checkpoint once on the held-out test list
#
# We intentionally avoid warm-starting from existing 5-fold Stage-1 checkpoints
# because those checkpoints may have seen patients in these held-out sets.

set -euo pipefail

cd "$HOME/3DGCNN"
mkdir -p experiments/stage1_forecaster/logs

export LD_LIBRARY_PATH="$HOME/miniforge3/envs/3dgcnn/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2
PY="$HOME/miniforge3/envs/3dgcnn/bin/python"

SPLIT_DIR="reports/unseen_forecaster_splits"
MANIFEST="$SPLIT_DIR/manifest.json"

if [[ ! -f "$MANIFEST" ]]; then
  "$PY" experiments/stage1_forecaster/make_unseen_forecaster_splits.py \
      --cohort datasets/ispy2/cohort.parquet \
      --patient-list reports/all_patients_4visit.txt \
      --graphs-root datasets/ispy2/graphs_consistent \
      --out-dir "$SPLIT_DIR"
fi

read -r NAME TRAIN_LIST TEST_LIST N_TRAIN N_TEST <<< "$("$PY" - <<PYEOF
import json
from pathlib import Path
idx = int("$SLURM_ARRAY_TASK_ID")
manifest = json.loads(Path("$MANIFEST").read_text())
item = manifest[idx]
print(item["name"], item["train_list"], item["test_list"], item["n_train"], item["n_test"])
PYEOF
)"

OUT_DIR="runs/consistent_forecaster_unseen/${NAME}"
mkdir -p "$OUT_DIR"

echo "[cg-fc-unseen] task=$SLURM_ARRAY_TASK_ID name=$NAME train=$N_TRAIN test=$N_TEST start=$(date -Iseconds)"

srun "$PY" experiments/stage1_forecaster/train_consistent_forecaster.py \
    --graphs-root  datasets/ispy2/graphs_consistent \
    --patient-list "$TRAIN_LIST" \
    --out-dir      "$OUT_DIR" \
    --pretrained   experiments/stage1_forecaster/runs/no_unseen_pretrain.pt \
    --hidden       64 \
    --num-layers   2 \
    --epochs       400 \
    --lr           3e-4 \
    --patience     60 \
    --freeze-backbone-epochs 0

srun "$PY" experiments/stage1_forecaster/eval_consistent_forecaster.py \
    --checkpoint   "$OUT_DIR/best.pt" \
    --graphs-root  datasets/ispy2/graphs_consistent \
    --test-list    "$TEST_LIST" \
    --train-list   "$TRAIN_LIST" \
    --out-json     "$OUT_DIR/heldout_eval.json"

echo "[cg-fc-unseen] task=$SLURM_ARRAY_TASK_ID name=$NAME end=$(date -Iseconds)"
