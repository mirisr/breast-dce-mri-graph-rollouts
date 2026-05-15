#!/bin/bash
#SBATCH --job-name=cg-bio-ablate
#SBATCH --output=experiments/stage1_forecaster/logs/cg-bio-ablate-%A_%a.out
#SBATCH --error=experiments/stage1_forecaster/logs/cg-bio-ablate-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpul40q
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-9%5

set -euo pipefail

cd "$HOME/3DGCNN"
mkdir -p experiments/stage1_forecaster/logs

export LD_LIBRARY_PATH="$HOME/miniforge3/envs/3dgcnn/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2
PY="$HOME/miniforge3/envs/3dgcnn/bin/python"

VARIANT_NAMES=(
  "bio_ftv020_alive005_no_sched"
  "bio_ftv020_alive005_no_edges"
)
EDGE_MODES=("full" "none")
SCHEDULED_FLAGS=("0" "1")

TASK_ID="${SLURM_ARRAY_TASK_ID}"
N_FOLDS=5
VARIANT_IDX=$(( TASK_ID / N_FOLDS ))
FOLD=$(( TASK_ID % N_FOLDS ))

TAG="${VARIANT_NAMES[$VARIANT_IDX]}"
EDGE_MODE="${EDGE_MODES[$VARIANT_IDX]}"
SCHEDULED_FLAG="${SCHEDULED_FLAGS[$VARIANT_IDX]}"

GRAPH_ROOT="${GRAPH_ROOT:-datasets/ispy2/graphs_consistent}"
PATIENT_LIST="${PATIENT_LIST:-reports/all_patients_4visit.txt}"
FOLDS_PARQUET="${FOLDS_PARQUET:-datasets/ispy2/folds.parquet}"
RUNS_ROOT="${RUNS_ROOT:-runs/consistent_forecaster_v2_bio_graph_ablation}"
PRETRAIN_BASE="${PRETRAIN_BASE:-runs/consistent_forecaster_v2/v2_sched_samp}"

EPOCHS="${EPOCHS:-120}"
PATIENCE="${PATIENCE:-45}"
FREEZE_EPOCHS="${FREEZE_EPOCHS:-8}"
LR="${LR:-1e-4}"
HIDDEN="${HIDDEN:-64}"
NUM_LAYERS="${NUM_LAYERS:-2}"
K_SPATIAL="${K_SPATIAL:-8}"

EPS_MAX="${EPS_MAX:-0.5}"
EPS_WARM="${EPS_WARM:-10}"
EPS_ANNEAL="${EPS_ANNEAL:-60}"
BIO_WARM="${BIO_WARM:-0}"
BIO_ANNEAL="${BIO_ANNEAL:-20}"
SELECTION_FTV_WEIGHT="${SELECTION_FTV_WEIGHT:-0.02}"
SELECTION_ALIVE_WEIGHT="${SELECTION_ALIVE_WEIGHT:-0.001}"
PATIENT_LIMIT="${PATIENT_LIMIT:-0}"

if [[ "${GRAPH_ABLATION_SMOKE:-0}" == "1" ]]; then
  EPOCHS="${SMOKE_EPOCHS:-2}"
  PATIENCE="${SMOKE_PATIENCE:-2}"
  PATIENT_LIMIT="${SMOKE_PATIENT_LIMIT:-12}"
  RUNS_ROOT="${SMOKE_RUNS_ROOT:-runs/consistent_forecaster_v2_bio_graph_ablation_smoke}"
fi

OUT_DIR="${RUNS_ROOT}/${TAG}/fold${FOLD}"
mkdir -p "$OUT_DIR"

echo "[cg-bio-ablate] task=${TASK_ID} variant=${TAG} fold=${FOLD} edge_mode=${EDGE_MODE} scheduled=${SCHEDULED_FLAG} out=${OUT_DIR} start=$(date -Iseconds)"

CMD=(
  "$PY" experiments/stage1_forecaster/train_consistent_forecaster_v2.py
  --graphs-root "$GRAPH_ROOT"
  --patient-list "$PATIENT_LIST"
  --folds-parquet "$FOLDS_PARQUET"
  --fold "$FOLD"
  --out-dir "$OUT_DIR"
  --pretrained "${PRETRAIN_BASE}/fold${FOLD}/best.pt"
  --epochs "$EPOCHS"
  --patience "$PATIENCE"
  --freeze-backbone-epochs "$FREEZE_EPOCHS"
  --lr "$LR"
  --hidden "$HIDDEN"
  --num-layers "$NUM_LAYERS"
  --k-spatial "$K_SPATIAL"
  --edge-mode "$EDGE_MODE"
  --eps-max "$EPS_MAX"
  --eps-warmup-epochs "$EPS_WARM"
  --eps-anneal-epochs "$EPS_ANNEAL"
  --lambda-ftv "0.20"
  --lambda-alive-mass "0.05"
  --lambda-alive-bce "0.05"
  --bio-warmup-epochs "$BIO_WARM"
  --bio-anneal-epochs "$BIO_ANNEAL"
  --selection-ftv-weight "$SELECTION_FTV_WEIGHT"
  --selection-alive-weight "$SELECTION_ALIVE_WEIGHT"
  --patient-limit "$PATIENT_LIMIT"
)

if [[ "$SCHEDULED_FLAG" == "1" ]]; then
  CMD+=(--scheduled-sampling)
fi

srun "${CMD[@]}"
echo "[cg-bio-ablate] task=${TASK_ID} variant=${TAG} fold=${FOLD} end=$(date -Iseconds)"
