# Endpoint-Calibrated Graph Rollouts for Breast DCE-MRI

This repository contains code, configurations, environment notes, launch
scripts, notebooks, and data-access instructions for endpoint-calibrated spatial
graph rollouts for breast DCE-MRI functional tumor volume (FTV) forecasting.

The public repository intentionally does not bundle paper result data, generated
figures, generated tables, manuscript PDFs, processed graph tensors, model
checkpoints, Monte Carlo outputs, or patient-level derived files. The source
DCE-MRI collections are publicly available through their custodial imaging
archives, and the scripts in this repository document how the derived graph and
evaluation artifacts are rebuilt after data access.

## Current Model

The manuscript model is `hybrid_a50_bio_k8`, reported in the paper as
`Hybrid-Edge k=8`. It combines endpoint and active-burden calibration with a
hybrid spatial-feature graph neighborhood and radial imaging-feature edge
attributes.

The `bio` substring in run tags is a historical internal label for these
attributes and endpoint-burden losses, not a biological-marker claim. The older
`bio_ftv020_alive005` model is retained as a historical Endpoint+Active
calibration baseline, not the final publication model.

## Repository Layout

```text
configs/                      retained model configuration files
cradle/                       cluster setup and launch notes
data/                         data-access instructions and manifests only
docs/                         reproducibility notes
environment/                  minimal dependency notes
experiments/preprocessing/    graph construction and preprocessing entry points
experiments/stage1_forecaster training, deterministic eval, edge ablations
experiments/consistent_rollout residual MC evaluation and Slurm wrappers
experiments/breast_nact_external external stress-test helper scripts
models/                       checkpoint manifest only
notebooks/                    analysis notebooks and plotting helpers
scripts/                      graph validation and data-preparation utilities
src/lsgc/                     graph model implementation
```

The compatibility link `lsgc -> src/lsgc` is intentional and preserves the
development import path used by the training and evaluation scripts.

## Data Status

This repository does not include raw DCE-MRI image volumes or derived
patient-level graph artifacts. Source imaging collections should be obtained
through TCIA and related custodial archives under the applicable data-use terms.

The public release excludes:

- raw DICOM/NIfTI data and segmentation masks;
- processed graph tensors and patient-level cohort/split parquet files;
- generated paper tables, figures, and manuscript PDFs;
- model checkpoints and training logs;
- residual Monte Carlo samples and patient-level evaluation outputs.

See `data/DATA_MANIFEST.md` and `data/ispy2/README.md` for data-access and
rebuild notes.

## Key Code Paths

| Path | Purpose |
| --- | --- |
| `configs/hybrid_a50_bio_k8.json` | Retained Hybrid-Edge model configuration. |
| `experiments/stage1_forecaster/train_consistent_forecaster_v2.py` | Current rollout training entry point with endpoint losses, dynamic edge modes, and optional edge attributes. |
| `experiments/stage1_forecaster/edge_modes.py` | Dynamic no-edge, spatial, radial, feature, hybrid, and radial imaging-feature graph construction utilities. |
| `experiments/consistent_rollout/run_conditional_mc.py` | Conditional residual Monte Carlo evaluator. |
| `experiments/breast_nact_external/` | Independent Breast-MRI-NACT-Pilot audit, preprocessing, and external evaluation helpers. |
| `scripts/build_graphs_v3_lite.py` | Lightweight graph build helper. |
| `scripts/validate_graph_schema.py` | Graph schema validation helper. |
| `models/MODEL_MANIFEST.md` | Checkpoint packaging expectations; no weights are bundled. |

## Reproducibility Notes

The included code is sufficient to audit the implementation and rerun the
pipeline after obtaining the source imaging data and rebuilding the derived
graph tensors. Full numerical reproduction requires locally generated graph
artifacts, held-out fold assignments, trained checkpoints, and evaluation
outputs.

Generated artifacts should remain outside git unless a separate release decision
is made for a specific archive. The `.gitignore` is configured to keep raw data,
derived data, result outputs, paper outputs, and checkpoint files out of the
public repository by default.
