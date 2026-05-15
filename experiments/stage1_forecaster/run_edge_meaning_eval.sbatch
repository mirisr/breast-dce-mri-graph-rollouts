#!/bin/bash
#SBATCH --job-name=edge-meaning-eval
#SBATCH --output=experiments/stage1_forecaster/logs/edge-meaning-eval-%A_%a.out
#SBATCH --error=experiments/stage1_forecaster/logs/edge-meaning-eval-%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpul40q
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-10%3

set -euo pipefail

cd "$HOME/3DGCNN"
mkdir -p experiments/stage1_forecaster/logs

export LD_LIBRARY_PATH="$HOME/miniforge3/envs/3dgcnn/lib:${LD_LIBRARY_PATH:-}"
PY="$HOME/miniforge3/envs/3dgcnn/bin/python"

TAGS=(
  "endpoint_full_k8"
  "no_edges"
  "spatial_k4"
  "spatial_k16"
  "radial_k8"
  "feature_all_k8"
  "feature_pe_ser_k8"
  "feature_volume_k8"
  "hybrid_a25_k8"
  "hybrid_a50_k8"
  "hybrid_a75_k8"
)

DATASET="${DATASET:-breast}"
TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"

if [[ "$DATASET" == "synthetic" ]]; then
  RUNS_ROOT="${RUNS_ROOT:-runs/edge_meaning_synthetic}"
  REPORTS_ROOT="${REPORTS_ROOT:-reports/edge_meaning_synthetic_eval}"
  GRAPHS_ROOT="${GRAPHS_ROOT:-datasets/synthetic_ftv_graphs/graphs}"
  FOLDS="${FOLDS:-datasets/synthetic_ftv_graphs/folds.parquet}"
  COHORT="${COHORT:-datasets/synthetic_ftv_graphs/cohort.parquet}"
else
  RUNS_ROOT="${RUNS_ROOT:-runs/edge_meaning_breast}"
  REPORTS_ROOT="${REPORTS_ROOT:-reports/edge_meaning_breast_eval}"
  GRAPHS_ROOT="${GRAPHS_ROOT:-datasets/ispy2/graphs_consistent}"
  FOLDS="${FOLDS:-datasets/ispy2/folds.parquet}"
  COHORT="${COHORT:-datasets/ispy2/cohort.parquet}"
fi

RUNS_DIR="${RUNS_ROOT}/${TAG}"
OUT_DIR="${REPORTS_ROOT}/${TAG}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-best.pt}"
LIMIT="${LIMIT:-0}"

echo "[edge-meaning-eval] dataset=${DATASET} tag=${TAG} runs=${RUNS_DIR} out=${OUT_DIR} start=$(date -Iseconds)"
srun "$PY" experiments/stage1_forecaster/run_simulation_eval.py \
  --runs-dir "$RUNS_DIR" \
  --graphs-root "$GRAPHS_ROOT" \
  --out-dir "$OUT_DIR" \
  --folds "$FOLDS" \
  --cohort "$COHORT" \
  --checkpoint-name "$CHECKPOINT_NAME" \
  --limit "$LIMIT"
echo "[edge-meaning-eval] dataset=${DATASET} tag=${TAG} end=$(date -Iseconds)"
