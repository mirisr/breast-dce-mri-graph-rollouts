#!/bin/bash
#SBATCH --job-name=s1-diag
#SBATCH --output=experiments/stage1_forecaster/runs/logs/diag-%j.out
#SBATCH --error=experiments/stage1_forecaster/runs/logs/diag-%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=gpuq
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Thread A2 + D — Diagnostic evaluation of existing S1.8 checkpoints.
# No training. Computes:
#   D1. Calibration: ECE, Brier per fold
#   D2. Trivial baselines: copy-paste, uniform-shrink
#   D3. Volume-aware metrics: top-K Dice / IoU
#   D4. Per-pCR and per-cohort breakdown
#   A2. ISPY2-only vs ACRIN-only split of the existing val metrics

set -euo pipefail
cd "$HOME/3DGCNN"
mkdir -p experiments/stage1_forecaster/runs/logs reports

export LD_LIBRARY_PATH="$HOME/miniforge3/envs/3dgcnn/lib:${LD_LIBRARY_PATH:-}"
PY="$HOME/miniforge3/envs/3dgcnn/bin/python"

TIMESTAMP=$(date +%Y%m%d_%H%M)
OUT="reports/s1_diagnostics_${TIMESTAMP}.md"

echo "[$(date)] running Stage-1 diagnostics"

"$PY" experiments/stage1_forecaster/run_diagnostics.py \
    --runs-dir experiments/stage1_forecaster/runs/bio_gated_5fold \
    --graphs-root datasets/ispy2/graphs_v3_full \
    --out "$OUT"

echo "[$(date)] done — report at $OUT"
