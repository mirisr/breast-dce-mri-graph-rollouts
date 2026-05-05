#!/bin/bash
#SBATCH --job-name=s1-fc
#SBATCH --output=experiments/stage1_forecaster/runs/logs/slurm-%A_%a.out
#SBATCH --error=experiments/stage1_forecaster/runs/logs/slurm-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpul40q
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-4

set -euo pipefail

cd "$HOME/3DGCNN"
mkdir -p experiments/stage1_forecaster/runs/logs

export LD_LIBRARY_PATH="$HOME/miniforge3/envs/3dgcnn/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2
PY="$HOME/miniforge3/envs/3dgcnn/bin/python"

FOLD="${SLURM_ARRAY_TASK_ID}"
# Match-method is optional: set MATCH=nn via --export to run the ablation.
MATCH="${MATCH:-sinkhorn}"
GRAPH_ROOT="${GRAPH_ROOT:-datasets/ispy2/graphs}"
EDGE_MODE="${EDGE_MODE:-geometric}"
SPATIAL_ALPHA="${SPATIAL_ALPHA:-0.7}"
# Tag the output directory so sinkhorn and nn runs don't collide.
OUT_DIR="experiments/stage1_forecaster/runs/${MATCH}_L2_skips-1-2-3/fold${FOLD}"
mkdir -p "$OUT_DIR"

echo "[s1-fc] fold=$FOLD match=$MATCH out=$OUT_DIR start=$(date -Iseconds)"
srun "$PY" experiments/stage1_forecaster/train_forecaster.py \
    --out-dir "$OUT_DIR" \
    --fold "$FOLD" \
    --match-method "$MATCH" \
    --graphs-root "$GRAPH_ROOT" \
    --num-layers 2 \
    --temporal-skip-hops 1 2 3 \
    --edge-mode "$EDGE_MODE" \
    --spatial-alpha "$SPATIAL_ALPHA" \
    --hidden 64 \
    --epochs 60 \
    --batch-accum 8 \
    --lr 2e-3 \
    --weight-decay 1e-4 \
    --lambda-pos 1.0 \
    --lambda-feat 0.5 \
    --lambda-alive 1.0 \
    --seed 42 \
    --device cuda
echo "[s1-fc] fold=$FOLD end=$(date -Iseconds)"
