#!/bin/bash
#SBATCH --job-name=cg-forecaster
#SBATCH --output=experiments/stage1_forecaster/logs/cg-%j.out
#SBATCH --error=experiments/stage1_forecaster/logs/cg-%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

# Consistent-graph forecaster prototype (30-patient set, S1.8 warm start).
set -euo pipefail
cd "$HOME/3DGCNN"
mkdir -p experiments/stage1_forecaster/logs

export LD_LIBRARY_PATH="$HOME/miniforge3/envs/3dgcnn/lib:${LD_LIBRARY_PATH:-}"
PY="$HOME/miniforge3/envs/3dgcnn/bin/python"

echo "[$(date)] Starting consistent-graph forecaster training"
"$PY" experiments/stage1_forecaster/train_consistent_forecaster.py \
    --graphs-root  datasets/ispy2/graphs_consistent \
    --patient-list reports/prototype_patients.txt \
    --out-dir      runs/consistent_forecaster \
    --pretrained   experiments/stage1_forecaster/runs/bio_gated_5fold/fold0/best.pt \
    --epochs 400 \
    --lr 3e-4 \
    --hidden 64 \
    --num-layers 2 \
    --patience 60 \
    --freeze-backbone-epochs 30

echo "[$(date)] Done"
