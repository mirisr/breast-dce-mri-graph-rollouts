#!/bin/bash
#SBATCH --job-name=edge-attr-mc
#SBATCH --output=experiments/consistent_rollout/logs/edge-attr-mc-%A_%a.out
#SBATCH --error=experiments/consistent_rollout/logs/edge-attr-mc-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpul40q
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-6%3

set -euo pipefail

cd "$HOME/3DGCNN"
mkdir -p experiments/consistent_rollout/logs

export LD_LIBRARY_PATH="$HOME/miniforge3/envs/3dgcnn/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
PY="$HOME/miniforge3/envs/3dgcnn/bin/python"

TAGS=(
  "radial_geo_k8"
  "radial_bio_k8"
  "radial_bio_k4"
  "radial_bio_k16"
  "spatial_k4_bio"
  "hybrid_a50_bio_k8"
  "hybrid_a75_bio_k8"
)

DATASET="${DATASET:-breast}"
TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"

if [[ "$DATASET" == "synthetic" ]]; then
  RUNS_ROOT="${RUNS_ROOT:-runs/edge_attr_meaning_synthetic}"
  OUT_ROOT="${OUT_ROOT:-reports/edge_attr_meaning_synthetic_mc}"
  GRAPHS_ROOT="${GRAPHS_ROOT:-datasets/synthetic_ftv_graphs/graphs}"
  FOLDS="${FOLDS:-datasets/synthetic_ftv_graphs/folds.parquet}"
  COHORT="${COHORT:-datasets/synthetic_ftv_graphs/cohort.parquet}"
  PATIENT_LIST="${PATIENT_LIST:-datasets/synthetic_ftv_graphs/patient_list.txt}"
else
  RUNS_ROOT="${RUNS_ROOT:-runs/edge_attr_meaning_breast}"
  OUT_ROOT="${OUT_ROOT:-reports/edge_attr_meaning_breast_mc}"
  GRAPHS_ROOT="${GRAPHS_ROOT:-datasets/ispy2/graphs_consistent}"
  FOLDS="${FOLDS:-datasets/ispy2/folds.parquet}"
  COHORT="${COHORT:-datasets/ispy2/cohort.parquet}"
  PATIENT_LIST="${PATIENT_LIST:-}"
fi

RUNS_DIR="${RUNS_ROOT}/${TAG}"
OUT_DIR="${OUT_ROOT}/${TAG}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-best.pt}"
N_MC="${N_MC:-128}"
METRIC_DRAWS="${METRIC_DRAWS:-32}"
PATIENT_LIMIT="${PATIENT_LIMIT:-0}"
EPS_PCR="${EPS_PCR:-0.1}"
INTERVAL="${INTERVAL:-0.90}"
SEED="${SEED:-20260513}"
DEVICE="${DEVICE:-auto}"

echo "[edge-attr-mc] dataset=${DATASET} tag=${TAG} runs=${RUNS_DIR} out=${OUT_DIR} n_mc=${N_MC} metric_draws=${METRIC_DRAWS} start=$(date -Iseconds)"
CMD=(
  "$PY" experiments/consistent_rollout/run_conditional_mc.py
  --runs-dir "$RUNS_DIR"
  --checkpoint-name "$CHECKPOINT_NAME"
  --graphs-root "$GRAPHS_ROOT"
  --folds "$FOLDS"
  --cohort "$COHORT"
  --out-dir "$OUT_DIR"
  --n-mc "$N_MC"
  --metric-draws "$METRIC_DRAWS"
  --patient-limit "$PATIENT_LIMIT"
  --eps-pcr "$EPS_PCR"
  --interval "$INTERVAL"
  --residual-stratify-by "none"
  --seed "$SEED"
  --device "$DEVICE"
)

if [[ -n "$PATIENT_LIST" ]]; then
  CMD+=(--patient-list "$PATIENT_LIST")
fi

srun "${CMD[@]}"
echo "[edge-attr-mc] dataset=${DATASET} tag=${TAG} end=$(date -Iseconds)"
