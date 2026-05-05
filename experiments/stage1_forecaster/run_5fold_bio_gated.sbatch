#!/bin/bash
#SBATCH --job-name=fc-bio-gate
#SBATCH --output=experiments/stage1_forecaster/runs/logs/biogate-%A_%a.out
#SBATCH --error=experiments/stage1_forecaster/runs/logs/biogate-%A_%a.err
#SBATCH --time=02:30:00
#SBATCH --partition=gpul40q
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-4

# S1.8 -- Stage-1 forecaster 5-fold with biological edge attrs + edge gating.

set -euo pipefail
cd "$HOME/3DGCNN"
mkdir -p experiments/stage1_forecaster/runs/logs

export LD_LIBRARY_PATH="$HOME/miniforge3/envs/3dgcnn/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2
PY="$HOME/miniforge3/envs/3dgcnn/bin/python"

GRAPH_ROOT="datasets/ispy2/graphs_v3_full"
RUN_TAG="bio_gated_5fold"
OUT_DIR="experiments/stage1_forecaster/runs/${RUN_TAG}/fold${SLURM_ARRAY_TASK_ID}"

"$PY" scripts/validate_graph_schema.py \
    --graphs-root "$GRAPH_ROOT" --require-habitat --expected-dim 33

echo "[$(date)] starting fold ${SLURM_ARRAY_TASK_ID}"

"$PY" experiments/stage1_forecaster/train_forecaster.py \
    --graphs-root "$GRAPH_ROOT" \
    --out-dir "$OUT_DIR" \
    --fold "$SLURM_ARRAY_TASK_ID" \
    --match-method nn \
    --num-layers 2 \
    --temporal-skip-hops 1 2 3 \
    --hidden 64 \
    --epochs 60 \
    --lr 2e-3 \
    --weight-decay 1e-4 \
    --lambda-pos 1.0 \
    --lambda-feat 0.5 \
    --lambda-alive 1.0 \
    --k-spatial 8 \
    --k-temporal 4 \
    --edge-mode mixed+attr \
    --edge-attr-mode bio \
    --use-edge-gating \
    --device cuda

echo "[$(date)] done fold ${SLURM_ARRAY_TASK_ID}"
