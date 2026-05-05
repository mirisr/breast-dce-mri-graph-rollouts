#!/bin/bash
#SBATCH --job-name=ants-reg-all
#SBATCH --output=experiments/preprocessing/logs/reg-all-%A_%a.out
#SBATCH --error=experiments/preprocessing/logs/reg-all-%A_%a.err
#SBATCH --time=00:45:00
#SBATCH --partition=kimq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-759%50

# ANTs SyN deformable registration for ALL patients with 4 visits.
# Each array task processes one patient (T1, T2, T3 -> T0).
#
# Prerequisites on HPC:
#   1. Run make_full_patient_list.py to produce reports/all_patients_4visit_ready.txt
#      (patients whose derived_v2 data is complete)
#   2. Set --array upper bound to (N_READY - 1). Inspect with:
#        wc -l reports/all_patients_4visit_ready.txt
#
# Submit:
#   NREADY=$(wc -l < reports/all_patients_4visit_ready.txt)
#   sbatch --array=0-$((NREADY-1)) experiments/preprocessing/run_register_all.sbatch

set -euo pipefail
cd "$HOME/3DGCNN"
mkdir -p experiments/preprocessing/logs

export LD_LIBRARY_PATH="$HOME/miniforge3/envs/3dgcnn/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4
PY="$HOME/miniforge3/envs/3dgcnn/bin/python"

# Use the 'ready' list (only patients with complete derived_v2 data).
# Fall back to the full list if the ready list doesn't exist yet.
PATIENT_LIST="${PATIENT_LIST:-reports/all_patients_4visit_ready.txt}"
if [ ! -f "$PATIENT_LIST" ]; then
    echo "Patient list not found: $PATIENT_LIST"
    echo "Run: python experiments/preprocessing/make_full_patient_list.py --derived datasets/ispy2/derived_v2"
    exit 1
fi

LINE_NUM=$(($SLURM_ARRAY_TASK_ID + 1))
PID=$(sed -n "${LINE_NUM}p" "$PATIENT_LIST")

if [ -z "$PID" ]; then
    echo "ERROR: no patient at line $LINE_NUM in $PATIENT_LIST"
    exit 1
fi

# Skip if already registered (all three warp outputs present)
REG_OUT="datasets/ispy2/registered/$PID"
if [ -f "$REG_OUT/T1_t0sv_in_T1_space.npz" ] && \
   [ -f "$REG_OUT/T2_t0sv_in_T2_space.npz" ] && \
   [ -f "$REG_OUT/T3_t0sv_in_T3_space.npz" ]; then
    echo "[$(date)] $PID already registered — skipping"
    exit 0
fi

echo "[$(date)] Registering $PID (array task $SLURM_ARRAY_TASK_ID)"
"$PY" experiments/preprocessing/register_visits.py \
    --patient  "$PID" \
    --derived  datasets/ispy2/derived \
    --out-root datasets/ispy2/registered
echo "[$(date)] Done $PID"
