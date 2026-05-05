#!/bin/bash
#SBATCH --job-name=ants-reg
#SBATCH --output=experiments/preprocessing/logs/reg-%A_%a.out
#SBATCH --error=experiments/preprocessing/logs/reg-%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpuq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-29

# Deformable registration (ANTs SyN) for the digital-twin prototype.
# Each array task processes one patient (T1, T2, T3 -> T0).
# Reads the patient ID at line $SLURM_ARRAY_TASK_ID of prototype_patients.txt.

set -euo pipefail
cd "$HOME/3DGCNN"
mkdir -p experiments/preprocessing/logs

export LD_LIBRARY_PATH="$HOME/miniforge3/envs/3dgcnn/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4
PY="$HOME/miniforge3/envs/3dgcnn/bin/python"

PATIENT_LIST=reports/prototype_patients.txt
LINE_NUM=$(($SLURM_ARRAY_TASK_ID + 1))
PID=$(sed -n "${LINE_NUM}p" "$PATIENT_LIST")

if [ -z "$PID" ]; then
    echo "ERROR: no patient at line $LINE_NUM"
    exit 1
fi

echo "[$(date)] Registering $PID (task $SLURM_ARRAY_TASK_ID)"
"$PY" experiments/preprocessing/register_visits.py --patient "$PID"
echo "[$(date)] Done $PID"
