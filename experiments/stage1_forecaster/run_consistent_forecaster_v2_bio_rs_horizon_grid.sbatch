#!/bin/bash
#SBATCH --job-name=cg-bio-rs
#SBATCH --output=experiments/stage1_forecaster/logs/cg-bio-rs-%A_%a.out
#SBATCH --error=experiments/stage1_forecaster/logs/cg-bio-rs-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpul40q
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-14%5

set -euo pipefail

cd "$HOME/3DGCNN"
mkdir -p experiments/stage1_forecaster/logs

export LD_LIBRARY_PATH="$HOME/miniforge3/envs/3dgcnn/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2
PY="$HOME/miniforge3/envs/3dgcnn/bin/python"

# Priority corrected grid: the primary candidate and the two controls already
# selected for retrained conditional MC. The previous bio grid omitted
# random-start; this run keeps a separate output root.
VARIANT_NAMES=(
  "bio_ftv020_alive005"
  "bio_ftv010_alive000"
  "bio_ftv010_alive002"
)
LAMBDA_FTVS=("0.20" "0.10" "0.10")
LAMBDA_ALIVE_MASSES=("0.05" "0.00" "0.02")
LAMBDA_ALIVE_BCES=("0.05" "0.00" "0.05")

TASK_ID="${SLURM_ARRAY_TASK_ID}"
N_FOLDS=5
VARIANT_IDX=$(( TASK_ID / N_FOLDS ))
FOLD=$(( TASK_ID % N_FOLDS ))

TAG="${VARIANT_NAMES[$VARIANT_IDX]}"
LAMBDA_FTV="${LAMBDA_FTVS[$VARIANT_IDX]}"
LAMBDA_ALIVE_MASS="${LAMBDA_ALIVE_MASSES[$VARIANT_IDX]}"
LAMBDA_ALIVE_BCE="${LAMBDA_ALIVE_BCES[$VARIANT_IDX]}"

GRAPH_ROOT="${GRAPH_ROOT:-datasets/ispy2/graphs_consistent}"
PATIENT_LIST="${PATIENT_LIST:-reports/all_patients_4visit.txt}"
FOLDS_PARQUET="${FOLDS_PARQUET:-datasets/ispy2/folds.parquet}"
RUNS_ROOT="${RUNS_ROOT:-runs/consistent_forecaster_v2_bio_rs_horizon}"
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
LAMBDA_HORIZON="${LAMBDA_HORIZON:-0.5}"
SELECTION_FTV_WEIGHT="${SELECTION_FTV_WEIGHT:-0.02}"
SELECTION_ALIVE_WEIGHT="${SELECTION_ALIVE_WEIGHT:-0.001}"
PATIENT_LIMIT="${PATIENT_LIMIT:-0}"

if [[ "${BIO_RS_SMOKE:-0}" == "1" ]]; then
  EPOCHS="${SMOKE_EPOCHS:-2}"
  PATIENCE="${SMOKE_PATIENCE:-2}"
  PATIENT_LIMIT="${SMOKE_PATIENT_LIMIT:-12}"
  RUNS_ROOT="${SMOKE_RUNS_ROOT:-runs/consistent_forecaster_v2_bio_rs_horizon_smoke}"
fi

OUT_DIR="${RUNS_ROOT}/${TAG}/fold${FOLD}"
mkdir -p "$OUT_DIR"

echo "[cg-bio-rs] task=${TASK_ID} variant=${TAG} fold=${FOLD} out=${OUT_DIR} start=$(date -Iseconds)"
echo "[cg-bio-rs] pretrained=${PRETRAIN_BASE}/fold${FOLD}/best.pt"
echo "[cg-bio-rs] random_start=1 lambda_horizon=${LAMBDA_HORIZON} lambda_ftv=${LAMBDA_FTV} lambda_alive_mass=${LAMBDA_ALIVE_MASS} lambda_alive_bce=${LAMBDA_ALIVE_BCE}"

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
  --scheduled-sampling \
  --eps-max "$EPS_MAX" \
  --eps-warmup-epochs "$EPS_WARM" \
  --eps-anneal-epochs "$EPS_ANNEAL" \
  --random-start \
  --lambda-horizon "$LAMBDA_HORIZON" \
  --lambda-ftv "$LAMBDA_FTV" \
  --lambda-alive-mass "$LAMBDA_ALIVE_MASS" \
  --lambda-alive-bce "$LAMBDA_ALIVE_BCE" \
  --bio-warmup-epochs "$BIO_WARM" \
  --bio-anneal-epochs "$BIO_ANNEAL" \
  --selection-ftv-weight "$SELECTION_FTV_WEIGHT" \
  --selection-alive-weight "$SELECTION_ALIVE_WEIGHT" \
  --patient-limit "$PATIENT_LIMIT"

echo "[cg-bio-rs] task=${TASK_ID} variant=${TAG} fold=${FOLD} end=$(date -Iseconds)"
