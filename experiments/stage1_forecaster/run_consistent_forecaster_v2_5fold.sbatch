#!/bin/bash
#SBATCH --job-name=cg-fc-v2-5fold
#SBATCH --output=experiments/stage1_forecaster/logs/cg-v2-%A_%a.out
#SBATCH --error=experiments/stage1_forecaster/logs/cg-v2-%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpul40q
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-4

set -euo pipefail

cd "$HOME/3DGCNN"
mkdir -p experiments/stage1_forecaster/logs

export LD_LIBRARY_PATH="$HOME/miniforge3/envs/3dgcnn/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2
PY="$HOME/miniforge3/envs/3dgcnn/bin/python"

FOLD="${SLURM_ARRAY_TASK_ID}"
GRAPH_ROOT="${GRAPH_ROOT:-datasets/ispy2/graphs_consistent}"
PATIENT_LIST="${PATIENT_LIST:-reports/all_patients_4visit.txt}"
FOLDS_PARQUET="${FOLDS_PARQUET:-datasets/ispy2/folds.parquet}"
PRETRAIN_BASE="${PRETRAIN_BASE:-experiments/stage1_forecaster/runs/bio_gated_5fold}"

EPOCHS="${EPOCHS:-180}"
PATIENCE="${PATIENCE:-40}"
FREEZE_EPOCHS="${FREEZE_EPOCHS:-30}"
LR="${LR:-3e-4}"
HIDDEN="${HIDDEN:-64}"
NUM_LAYERS="${NUM_LAYERS:-2}"
K_SPATIAL="${K_SPATIAL:-8}"

# Variant flags
MODE="${MODE:-B}" # B, C, D
EPS_MAX="${EPS_MAX:-0.5}"
EPS_WARM="${EPS_WARM:-60}"
EPS_ANNEAL="${EPS_ANNEAL:-60}"
LAMBDA_CLOUD="${LAMBDA_CLOUD:-1.0}"
LAMBDA_CLOUD_FEAT="${LAMBDA_CLOUD_FEAT:-0.0}"
LAMBDA_HORIZON="${LAMBDA_HORIZON:-0.0}"
TAG="${TAG:-v2_sched_samp}"

EXTRA=()
if [[ "$MODE" == "B" ]]; then
  EXTRA+=(--scheduled-sampling --eps-max "$EPS_MAX" --eps-warmup-epochs "$EPS_WARM" --eps-anneal-epochs "$EPS_ANNEAL")
elif [[ "$MODE" == "C" ]]; then
  EXTRA+=(--scheduled-sampling --eps-max "$EPS_MAX" --eps-warmup-epochs "$EPS_WARM" --eps-anneal-epochs "$EPS_ANNEAL" --lambda-horizon "${LAMBDA_HORIZON:-0.5}")
elif [[ "$MODE" == "D" ]]; then
  EXTRA+=(--scheduled-sampling --eps-max "$EPS_MAX" --eps-warmup-epochs "$EPS_WARM" --eps-anneal-epochs "$EPS_ANNEAL" --lambda-horizon "${LAMBDA_HORIZON:-0.5}" --random-start)
else
  echo "Unsupported MODE=$MODE (expected B/C/D)"; exit 2
fi

OUT_DIR="runs/consistent_forecaster_v2/${TAG}/fold${FOLD}"
mkdir -p "$OUT_DIR"

echo "[cg-fc-v2-5fold] mode=$MODE tag=$TAG fold=$FOLD out=$OUT_DIR start=$(date -Iseconds)"
srun "$PY" experiments/stage1_forecaster/train_consistent_forecaster_v2.py \
  --graphs-root "$GRAPH_ROOT" \
  --patient-list "$PATIENT_LIST" \
  --folds-parquet "$FOLDS_PARQUET" \
  --fold "$FOLD" \
  --out-dir "$OUT_DIR" \
  --pretrained "${PRETRAIN_BASE}/fold${FOLD}/best.pt" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --freeze-backbone-epochs "$FREEZE_EPOCHS" \
  --lr "$LR" \
  --hidden "$HIDDEN" \
  --num-layers "$NUM_LAYERS" \
  --k-spatial "$K_SPATIAL" \
  --lambda-cloud "$LAMBDA_CLOUD" \
  --lambda-cloud-feat "$LAMBDA_CLOUD_FEAT" \
  "${EXTRA[@]}"
echo "[cg-fc-v2-5fold] mode=$MODE tag=$TAG fold=$FOLD end=$(date -Iseconds)"
