#!/bin/bash
#SBATCH --job-name=nact4-paper-eval
#SBATCH --output=reports/breast_mri_nact_external/logs/nact4-paper-eval-%A_%a.out
#SBATCH --error=reports/breast_mri_nact_external/logs/nact4-paper-eval-%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpul40q,gpua30q,kimq
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --array=0-3%3

set -euo pipefail

cd "$HOME/3DGCNN"
mkdir -p reports/breast_mri_nact_external/logs

export LD_LIBRARY_PATH="$HOME/miniforge3/envs/3dgcnn/lib:${LD_LIBRARY_PATH:-}"
PY="$HOME/miniforge3/envs/3dgcnn/bin/python"

EVAL_TABLE_DIR="${EVAL_TABLE_DIR:-datasets/breast_mri_nact_pilot/eval_tables_4visit}"
FOLDS="$EVAL_TABLE_DIR/external_folds_ensemble.parquet"
COHORT="$EVAL_TABLE_DIR/external_cohort.parquet"
PATIENT_LIST="${PATIENT_LIST:-reports/breast_mri_nact_external/patients_4visit.txt}"

MODEL_NAMES=(
  "bio_ftv020_alive005"
  "no_edges"
  "radial_bio_k8"
  "hybrid_a50_bio_k8"
)
RUN_DIRS=(
  "runs/consistent_forecaster_v2_bio/bio_ftv020_alive005"
  "runs/edge_meaning_breast/no_edges"
  "runs/edge_attr_meaning_breast/radial_bio_k8"
  "runs/edge_attr_meaning_breast/hybrid_a50_bio_k8"
)

IDX="${SLURM_ARRAY_TASK_ID}"
MODEL_NAME="${MODEL_NAMES[$IDX]}"
RUNS_DIR="${RUN_DIRS[$IDX]}"
GRAPH_ROOT="${GRAPH_ROOT:-datasets/breast_mri_nact_pilot/graphs_consistent_4visit}"
OUT_ROOT="${OUT_ROOT:-reports/breast_mri_nact_external/deterministic_4visit_paper_models}"
OUT_DIR="${OUT_ROOT}/${MODEL_NAME}"

echo "[nact4-paper-eval] model=$MODEL_NAME runs=$RUNS_DIR graphs=$GRAPH_ROOT start=$(date -Iseconds)"
"$PY" experiments/stage1_forecaster/run_simulation_eval.py \
  --runs-dir "$RUNS_DIR" \
  --graphs-root "$GRAPH_ROOT" \
  --folds "$FOLDS" \
  --cohort "$COHORT" \
  --patient-list "$PATIENT_LIST" \
  --out-dir "$OUT_DIR" \
  --checkpoint-name best.pt \
  --limit "${LIMIT:-0}"
echo "[nact4-paper-eval] model=$MODEL_NAME end=$(date -Iseconds)"
