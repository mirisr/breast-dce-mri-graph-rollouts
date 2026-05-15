#!/bin/bash
#SBATCH --job-name=edge-meaning-train
#SBATCH --output=experiments/stage1_forecaster/logs/edge-meaning-train-%A_%a.out
#SBATCH --error=experiments/stage1_forecaster/logs/edge-meaning-train-%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpul40q
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-54%3

set -euo pipefail

cd "$HOME/3DGCNN"
mkdir -p experiments/stage1_forecaster/logs

export LD_LIBRARY_PATH="$HOME/miniforge3/envs/3dgcnn/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
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
EDGE_MODES=(
  "full"
  "none"
  "full"
  "full"
  "radial_knn"
  "feature_knn_all"
  "feature_knn_pe_ser"
  "feature_knn_volume"
  "hybrid_spatial_feature_a25"
  "hybrid_spatial_feature_a50"
  "hybrid_spatial_feature_a75"
)
K_SPATIALS=("8" "8" "4" "16" "8" "8" "8" "8" "8" "8" "8")

DATASET="${DATASET:-breast}"
N_FOLDS=5
TASK_ID="${SLURM_ARRAY_TASK_ID}"
VARIANT_IDX=$(( TASK_ID / N_FOLDS ))
FOLD=$(( TASK_ID % N_FOLDS ))

TAG="${TAGS[$VARIANT_IDX]}"
EDGE_MODE="${EDGE_MODES[$VARIANT_IDX]}"
K_SPATIAL="${K_SPATIALS[$VARIANT_IDX]}"

if [[ "$DATASET" == "synthetic" ]]; then
  GRAPH_ROOT="${GRAPH_ROOT:-datasets/synthetic_ftv_graphs/graphs}"
  PATIENT_LIST="${PATIENT_LIST:-datasets/synthetic_ftv_graphs/patient_list.txt}"
  FOLDS_PARQUET="${FOLDS_PARQUET:-datasets/synthetic_ftv_graphs/folds.parquet}"
  RUNS_ROOT="${RUNS_ROOT:-runs/edge_meaning_synthetic}"
  PRETRAINED="${PRETRAINED:-runs/nonexistent_synthetic_pretrain.pt}"
  EPOCHS="${EPOCHS:-100}"
  PATIENCE="${PATIENCE:-25}"
  FREEZE_EPOCHS="${FREEZE_EPOCHS:-0}"
  LR="${LR:-3e-4}"
  EPS_WARM="${EPS_WARM:-5}"
  EPS_ANNEAL="${EPS_ANNEAL:-35}"
else
  GRAPH_ROOT="${GRAPH_ROOT:-datasets/ispy2/graphs_consistent}"
  PATIENT_LIST="${PATIENT_LIST:-reports/all_patients_4visit.txt}"
  FOLDS_PARQUET="${FOLDS_PARQUET:-datasets/ispy2/folds.parquet}"
  RUNS_ROOT="${RUNS_ROOT:-runs/edge_meaning_breast}"
  PRETRAINED="${PRETRAINED:-runs/consistent_forecaster_v2/v2_sched_samp/fold${FOLD}/best.pt}"
  EPOCHS="${EPOCHS:-120}"
  PATIENCE="${PATIENCE:-45}"
  FREEZE_EPOCHS="${FREEZE_EPOCHS:-8}"
  LR="${LR:-1e-4}"
  EPS_WARM="${EPS_WARM:-10}"
  EPS_ANNEAL="${EPS_ANNEAL:-60}"
fi

HIDDEN="${HIDDEN:-64}"
NUM_LAYERS="${NUM_LAYERS:-2}"
EPS_MAX="${EPS_MAX:-0.5}"
BIO_WARM="${BIO_WARM:-0}"
BIO_ANNEAL="${BIO_ANNEAL:-20}"
PATIENT_LIMIT="${PATIENT_LIMIT:-0}"

OUT_DIR="${RUNS_ROOT}/${TAG}/fold${FOLD}"
mkdir -p "$OUT_DIR"

echo "[edge-meaning-train] dataset=${DATASET} task=${TASK_ID} variant=${TAG} fold=${FOLD} edge_mode=${EDGE_MODE} k=${K_SPATIAL} start=$(date -Iseconds)"

srun "$PY" experiments/stage1_forecaster/train_consistent_forecaster_v2.py \
  --graphs-root "$GRAPH_ROOT" \
  --patient-list "$PATIENT_LIST" \
  --folds-parquet "$FOLDS_PARQUET" \
  --fold "$FOLD" \
  --out-dir "$OUT_DIR" \
  --pretrained "$PRETRAINED" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --freeze-backbone-epochs "$FREEZE_EPOCHS" \
  --lr "$LR" \
  --hidden "$HIDDEN" \
  --num-layers "$NUM_LAYERS" \
  --k-spatial "$K_SPATIAL" \
  --edge-mode "$EDGE_MODE" \
  --scheduled-sampling \
  --eps-max "$EPS_MAX" \
  --eps-warmup-epochs "$EPS_WARM" \
  --eps-anneal-epochs "$EPS_ANNEAL" \
  --lambda-ftv "0.20" \
  --lambda-alive-mass "0.05" \
  --lambda-alive-bce "0.05" \
  --bio-warmup-epochs "$BIO_WARM" \
  --bio-anneal-epochs "$BIO_ANNEAL" \
  --selection-ftv-weight "0.02" \
  --selection-alive-weight "0.001" \
  --patient-limit "$PATIENT_LIMIT"

echo "[edge-meaning-train] dataset=${DATASET} task=${TASK_ID} variant=${TAG} fold=${FOLD} end=$(date -Iseconds)"
