#!/bin/bash
#SBATCH --job-name=cg-build-all
#SBATCH --output=experiments/preprocessing/logs/cg-build-%A_%a.out
#SBATCH --error=experiments/preprocessing/logs/cg-build-%A_%a.err
#SBATCH --time=00:15:00
#SBATCH --partition=kimq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-759%50

# Build consistent-graph .pt files for ALL registered patients.
# Each task handles one patient; takes ~1 min per patient (CPU-only).
#
# Prerequisites:
#   - Registration completed: datasets/ispy2/registered/{pid}/ must exist
#   - Derived data available: datasets/ispy2/derived_v2/{pid}/ must exist
#   - Run after run_register_all.sbatch (or use --dependency=afterok:<REG_JOB>)
#
# Submit (after registration completes):
#   NREADY=$(wc -l < reports/all_patients_4visit_ready.txt)
#   sbatch --array=0-$((NREADY-1)) \
#          --dependency=afterok:<REGISTRATION_JOB_ID> \
#          experiments/preprocessing/run_build_consistent_graphs_all.sbatch

set -euo pipefail
cd "$HOME/3DGCNN"
mkdir -p experiments/preprocessing/logs datasets/ispy2/graphs_consistent

export LD_LIBRARY_PATH="$HOME/miniforge3/envs/3dgcnn/lib:${LD_LIBRARY_PATH:-}"
PY="$HOME/miniforge3/envs/3dgcnn/bin/python"

PATIENT_LIST="${PATIENT_LIST:-reports/all_patients_4visit_ready.txt}"
if [ ! -f "$PATIENT_LIST" ]; then
    echo "Patient list not found: $PATIENT_LIST"
    exit 1
fi

LINE_NUM=$(($SLURM_ARRAY_TASK_ID + 1))
PID=$(sed -n "${LINE_NUM}p" "$PATIENT_LIST")

if [ -z "$PID" ]; then
    echo "ERROR: no patient at line $LINE_NUM"
    exit 1
fi

OUT_PT="datasets/ispy2/graphs_consistent/$PID.pt"
if [ -f "$OUT_PT" ]; then
    echo "[$(date)] $PID already built — skipping"
    exit 0
fi

echo "[$(date)] Building consistent graph for $PID (array task $SLURM_ARRAY_TASK_ID)"
"$PY" experiments/preprocessing/build_consistent_graphs.py \
    --patient    "$PID" \
    --derived    datasets/ispy2/derived \
    --registered datasets/ispy2/registered \
    --out-root   datasets/ispy2/graphs_consistent
echo "[$(date)] Done $PID"
