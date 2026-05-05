#!/bin/bash
#SBATCH --job-name=cg-bio-eval
#SBATCH --output=experiments/stage1_forecaster/logs/cg-bio-eval-%A_%a.out
#SBATCH --error=experiments/stage1_forecaster/logs/cg-bio-eval-%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --partition=kimq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-4

set -euo pipefail

cd "$HOME/3DGCNN"
mkdir -p experiments/stage1_forecaster/logs

export LD_LIBRARY_PATH="$HOME/miniforge3/envs/3dgcnn/lib:${LD_LIBRARY_PATH:-}"
PY="$HOME/miniforge3/envs/3dgcnn/bin/python"

VARIANT_NAMES=(
  "bio_ftv005_alive001"
  "bio_ftv010_alive002"
  "bio_ftv020_alive005"
  "bio_ftv010_alive000"
  "bio_ftv000_alive002"
)

TAG="${VARIANT_NAMES[$SLURM_ARRAY_TASK_ID]}"
RUNS_ROOT="${RUNS_ROOT:-runs/consistent_forecaster_v2_bio}"
REPORTS_ROOT="${REPORTS_ROOT:-reports/consistent_forecaster_v2_bio_eval}"
RUNS_DIR="${RUNS_ROOT}/${TAG}"
OUT_DIR="${REPORTS_ROOT}/${TAG}"

GRAPHS_ROOT="${GRAPHS_ROOT:-datasets/ispy2/graphs_consistent}"
FOLDS="${FOLDS:-datasets/ispy2/folds.parquet}"
COHORT="${COHORT:-datasets/ispy2/cohort.parquet}"
CKPT_NAME="${CKPT_NAME:-best.pt}"
LIMIT="${LIMIT:-0}"

echo "[cg-bio-eval] tag=${TAG} runs=${RUNS_DIR} out=${OUT_DIR} start=$(date -Iseconds)"
srun "$PY" experiments/stage1_forecaster/run_simulation_eval.py \
  --runs-dir "$RUNS_DIR" \
  --graphs-root "$GRAPHS_ROOT" \
  --out-dir "$OUT_DIR" \
  --folds "$FOLDS" \
  --cohort "$COHORT" \
  --checkpoint-name "$CKPT_NAME" \
  --limit "$LIMIT"
echo "[cg-bio-eval] tag=${TAG} end=$(date -Iseconds)"
