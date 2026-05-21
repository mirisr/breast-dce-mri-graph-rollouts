#!/bin/bash
#SBATCH --job-name=nact4-paper-srcmc
#SBATCH --output=reports/breast_mri_nact_external/logs/nact4-paper-srcmc-%j.out
#SBATCH --error=reports/breast_mri_nact_external/logs/nact4-paper-srcmc-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=kimq,gpua30q,gpul40q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euo pipefail

cd "$HOME/3DGCNN"
mkdir -p reports/breast_mri_nact_external/logs

export LD_LIBRARY_PATH="$HOME/miniforge3/envs/3dgcnn/lib:${LD_LIBRARY_PATH:-}"
PY="$HOME/miniforge3/envs/3dgcnn/bin/python"

echo "[nact4-paper-srcmc] start=$(date -Iseconds)"
"$PY" experiments/breast_nact_external/run_external_source_residual_mc.py \
  --external-det-root "${EXTERNAL_DET_ROOT:-reports/breast_mri_nact_external/deterministic_4visit_paper_models}" \
  --out-dir "${OUT_DIR:-reports/breast_mri_nact_external/source_residual_mc_4visit_paper_models}" \
  --n-mc "${N_MC:-256}" \
  --seed "${SEED:-20260521}" \
  --models \
    bio_ftv020_alive005 \
    no_edges \
    radial_bio_k8 \
    hybrid_a50_bio_k8
echo "[nact4-paper-srcmc] end=$(date -Iseconds)"
