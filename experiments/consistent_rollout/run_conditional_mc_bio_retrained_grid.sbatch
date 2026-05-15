#!/bin/bash
#SBATCH --job-name=cg-bio-mc
#SBATCH --output=experiments/consistent_rollout/logs/cg-bio-mc-%A_%a.out
#SBATCH --error=experiments/consistent_rollout/logs/cg-bio-mc-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpul40q
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-2

# Conditional Monte Carlo for the retrained consistent-graph rollout models.
# Submit from ~/3DGCNN on Cradle:
#
#   sbatch experiments/consistent_rollout/run_conditional_mc_bio_retrained_grid.sbatch
#
# Quick smoke:
#
#   N_MC=8 PATIENT_LIMIT=3 OUT_ROOT=reports/conditional_mc_bio_retrained_smoke \
#     sbatch experiments/consistent_rollout/run_conditional_mc_bio_retrained_grid.sbatch

set -euo pipefail

cd "$HOME/3DGCNN"
mkdir -p experiments/consistent_rollout/logs

export LD_LIBRARY_PATH="$HOME/miniforge3/envs/3dgcnn/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
PY="$HOME/miniforge3/envs/3dgcnn/bin/python"

TAGS=(
  "bio_ftv020_alive005"
  "bio_ftv010_alive000"
  "bio_ftv010_alive002"
)

TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"
RUNS_ROOT="${RUNS_ROOT:-runs/consistent_forecaster_v2_bio}"
OUT_ROOT="${OUT_ROOT:-reports/conditional_mc_bio_retrained}"
RUNS_DIR="${RUNS_ROOT}/${TAG}"
OUT_DIR="${OUT_ROOT}/${TAG}"

CHECKPOINT_NAME="${CHECKPOINT_NAME:-best.pt}"
GRAPHS_ROOT="${GRAPHS_ROOT:-datasets/ispy2/graphs_consistent}"
FOLDS="${FOLDS:-datasets/ispy2/folds.parquet}"
COHORT="${COHORT:-datasets/ispy2/cohort.parquet}"
N_MC="${N_MC:-256}"
METRIC_DRAWS="${METRIC_DRAWS:-0}"
PATIENT_LIMIT="${PATIENT_LIMIT:-0}"
EPS_PCR="${EPS_PCR:-0.1}"
INTERVAL="${INTERVAL:-0.90}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"
PATIENT_LIST="${PATIENT_LIST:-}"
RESIDUAL_STRATIFY_BY="${RESIDUAL_STRATIFY_BY:-none}"

echo "[cg-bio-mc] tag=${TAG} start=$(date -Iseconds)"
echo "[cg-bio-mc] runs=${RUNS_DIR} graphs=${GRAPHS_ROOT} out=${OUT_DIR} n_mc=${N_MC} metric_draws=${METRIC_DRAWS} residual_stratify_by=${RESIDUAL_STRATIFY_BY}"

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
  --residual-stratify-by "$RESIDUAL_STRATIFY_BY"
  --seed "$SEED"
  --device "$DEVICE"
)

if [[ -n "$PATIENT_LIST" ]]; then
  CMD+=(--patient-list "$PATIENT_LIST")
fi

srun "${CMD[@]}"
echo "[cg-bio-mc] tag=${TAG} end=$(date -Iseconds)"
