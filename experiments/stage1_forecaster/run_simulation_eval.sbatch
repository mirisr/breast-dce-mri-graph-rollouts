#!/bin/bash
#SBATCH --job-name=cg-sim-eval
#SBATCH --output=experiments/stage1_forecaster/logs/cg-sim-eval-%A.out
#SBATCH --error=experiments/stage1_forecaster/logs/cg-sim-eval-%A.err
#SBATCH --time=02:00:00
#SBATCH --partition=kimq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Run simulation evaluation on registered/consistent graphs with the
# consistent forecaster checkpoints.
#
# Examples:
#   sbatch experiments/stage1_forecaster/run_simulation_eval.sbatch
#
#   RUNS_DIR=runs/consistent_forecaster_unseen \
#   PATIENT_LIST=reports/unseen_forecaster_splits/locked_stratified_test_test.txt \
#   OUT_DIR=reports/simulation_eval_locked \
#   sbatch experiments/stage1_forecaster/run_simulation_eval.sbatch

set -euo pipefail

cd "$HOME/3DGCNN"
mkdir -p experiments/stage1_forecaster/logs

export LD_LIBRARY_PATH="$HOME/miniforge3/envs/3dgcnn/lib:${LD_LIBRARY_PATH:-}"
PY="$HOME/miniforge3/envs/3dgcnn/bin/python"

RUNS_DIR="${RUNS_DIR:-runs/consistent_forecaster_5fold}"
GRAPHS_ROOT="${GRAPHS_ROOT:-datasets/ispy2/graphs_consistent}"
OUT_DIR="${OUT_DIR:-reports/simulation_eval}"
FOLDS="${FOLDS:-datasets/ispy2/folds.parquet}"
COHORT="${COHORT:-datasets/ispy2/cohort.parquet}"
CKPT_NAME="${CKPT_NAME:-best.pt}"
PATIENT_LIST="${PATIENT_LIST:-}"
LIMIT="${LIMIT:-0}"

echo "[cg-sim-eval] runs=$RUNS_DIR graphs=$GRAPHS_ROOT out=$OUT_DIR start=$(date -Iseconds)"
CMD=(
  "$PY" experiments/stage1_forecaster/run_simulation_eval.py
  --runs-dir "$RUNS_DIR"
  --graphs-root "$GRAPHS_ROOT"
  --out-dir "$OUT_DIR"
  --folds "$FOLDS"
  --cohort "$COHORT"
  --checkpoint-name "$CKPT_NAME"
  --limit "$LIMIT"
)
if [[ -n "$PATIENT_LIST" ]]; then
  CMD+=(--patient-list "$PATIENT_LIST")
fi

srun "${CMD[@]}"
echo "[cg-sim-eval] end=$(date -Iseconds)"
