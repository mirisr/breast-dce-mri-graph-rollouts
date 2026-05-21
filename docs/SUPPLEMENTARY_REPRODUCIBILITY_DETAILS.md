# Supplementary Reproducibility Details

This file is the reviewer-facing audit trail for the manuscript package. It
records the model names, data partitions, evaluation paths, and result files
that support the claims made in the paper.

## Scope

The repository supports three levels of reproducibility.

1. Manuscript audit from included aggregate tables, figures, and derived
   summaries.
2. Re-running deterministic and residual Monte Carlo evaluation when graph
   tensors and checkpoints are available.
3. Re-training the retained graph-family models from graph tensors under the
   five-fold held-out-patient protocol.

Raw DCE-MRI volumes, raw segmentations, and controlled-access source files are
not redistributed. They must be obtained through the appropriate TCIA data
access mechanism and processed under the relevant data-use terms.

## Model Tags

| Paper label | Repository tag | Role |
| --- | --- | --- |
| Original graph rollout | `v2_sched_samp` / `consistent_forecaster_v2` | Scheduled-sampling graph rollout baseline. |
| Endpoint+Active | `bio_ftv020_alive005` | Endpoint-calibrated comparator with FTV and active-burden losses. |
| No-edge endpoint | `no_edges` | Node-local endpoint-calibrated control. |
| Radial imaging-feature k=8 | `radial_bio_k8` | Radial-neighborhood graph with radial imaging-feature edge attributes. |
| Hybrid-Edge k=8 | `hybrid_a50_bio_k8` | Final retained manuscript graph model. |

The `bio` substring in repository tags is a historical internal label for
radial imaging-feature attributes or endpoint/active-burden losses; it is not a
claim that the edge variables are biological biomarkers.

The final retained configuration is recorded in
`configs/hybrid_a50_bio_k8.json`. The paper-facing external stress-test table
uses only the four endpoint-calibrated paper-family models:
`bio_ftv020_alive005`, `no_edges`, `radial_bio_k8`, and
`hybrid_a50_bio_k8`.

## Data Partitions

### Internal Source Cohort

The main source-cohort evaluation uses 758 four-visit graph-bearing patients
from the I-SPY2/ACRIN-derived breast DCE-MRI graph set. The held-out evaluation
uses five patient-level folds. Each patient is scored only by the model
checkpoint from the fold in which that patient was held out.

Primary source files:

| Path | Purpose |
| --- | --- |
| `data/ispy2/cohort.parquet` | Derived cohort metadata used by evaluation scripts. |
| `data/ispy2/folds.parquet` | Five-fold held-out-patient split assignment. |
| `data/ispy2/cohort_audit.md` | Cohort construction audit. |
| `data/ispy2/folds_audit.md` | Split audit. |
| `paper/tables/split_leakage_audit.csv` | Manuscript-facing leakage-control summary. |

The compatibility links `datasets/ispy2 -> data/ispy2` and
`reports -> results` preserve the paths used by the training and evaluation
scripts.

### Independent NACT Stress Test

The independent stress test uses graph-ready Breast-MRI-NACT-Pilot patients
with four visits available after the external preprocessing gate. The reported
T0-to-T3 result contains 11 patients. One additional V4 candidate failed the
support check after segmentation mapping and was excluded before scoring.

This should be described as a preliminary independent external stress test, not
as external validation. The cohort is independent of the source training cohort,
but the sample is too small to claim powered clinical generalization.

Paper-facing aggregate table:

| Path | Purpose |
| --- | --- |
| `paper/tables/external_nact_stress_test_t0t3.csv` | Four-model T0-to-T3 external stress-test summary used by the manuscript. |

The reported T0-to-T3 external values are:

| Model | Deterministic MAE (mL) | Deterministic bias (mL) | MC mean MAE (mL) | Raw 90% coverage | Conformal 90% coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Endpoint+Active | 20.17 | +19.71 | 23.51 | 0.455 | 0.455 |
| No-edge endpoint | 19.21 | +18.82 | 22.11 | 0.455 | 0.455 |
| Radial imaging-feature k=8 | 16.70 | +16.36 | 19.31 | 0.455 | 0.455 |
| Hybrid-Edge k=8 | 12.40 | +12.13 | 15.01 | 0.545 | 0.545 |

The key claim supported by this table is relative transfer behavior within the
paper-family graph models: `Hybrid-Edge k=8` has the lowest deterministic MAE,
lowest MC-mean MAE, and highest coverage among the four reported models, while
all four models show positive bias on this small external cohort.

## Graph Construction

Each patient is represented as a registration-consistent tumor graph. Baseline
tumor support is partitioned into compact 3D SLIC supervoxels. Registration
transports the baseline supervoxel partition to later visits so each node
tracks the same baseline tumor region across visits.

Each node stores six local imaging-burden features:

- `log(1 + voxel count)`;
- volume in mL;
- percent enhancement mean;
- percent enhancement standard deviation;
- signal enhancement ratio mean;
- signal enhancement ratio standard deviation.

Centroid positions are represented in millimeters after subtracting the
visit-level tumor centroid. The active-node label records whether the
transported baseline supervoxel still has tumor support at the target visit.

The final retained model rebuilds within-visit graph edges dynamically. For
`hybrid_a50_bio_k8`, candidate neighbors are ranked by an equal-weight
combination of normalized spatial distance and normalized feature distance.
The eight lowest-score neighbors are retained and symmetrized. Radial
imaging-feature edge attributes encode centroid distance, radial-shell
relation, visit gap, temporal-edge status, and imaging-feature contrast.

Implementation files:

| Path | Purpose |
| --- | --- |
| `experiments/preprocessing/build_consistent_graphs.py` | Internal source graph construction entry point. |
| `experiments/stage1_forecaster/edge_modes.py` | No-edge, spatial, radial, hybrid, and radial imaging-feature edge construction. |
| `scripts/validate_graph_schema.py` | Graph schema validation helper. |
| `experiments/breast_nact_external/` | External Breast-MRI-NACT-Pilot audit, download, derivation, and evaluation helpers. |

## Training Protocol

All retained graph-family models use the scheduled-sampling consistent graph
forecaster with two LSGC layers, hidden size 64, and five held-out patient
folds. The endpoint-calibrated family starts from the scheduled-sampling
baseline checkpoint for each fold, freezes the backbone for 8 epochs, and then
fine-tunes with AdamW at learning rate `1e-4`.

The retained endpoint loss settings are:

| Setting | Value |
| --- | ---: |
| `lambda_ftv` | 0.20 |
| `lambda_alive_mass` | 0.05 |
| `lambda_alive_bce` | 0.05 |
| `eps_max` | 0.50 |
| `eps_warmup_epochs` | 10 |
| `eps_anneal_epochs` | 60 |
| `bio_warmup_epochs` | 0 |
| `bio_anneal_epochs` | 20 |
| `selection_ftv_weight` | 0.02 |
| `selection_alive_weight` | 0.001 |

Training launchers:

| Path | Purpose |
| --- | --- |
| `experiments/stage1_forecaster/run_consistent_forecaster_v2_bio_grid.sbatch` | Endpoint and active-burden calibration grid. |
| `experiments/stage1_forecaster/run_edge_meaning_grid.sbatch` | Edge-neighborhood ablation training. |
| `experiments/stage1_forecaster/run_edge_attribute_meaning_grid.sbatch` | Edge-attribute ablation training for the retained model family. |
| `experiments/stage1_forecaster/train_consistent_forecaster_v2.py` | Training entry point. |

## Deterministic Evaluation

Deterministic evaluation computes future graph states and scalar FTV readouts
from held-out fold checkpoints.

Internal evaluation:

```bash
python experiments/stage1_forecaster/run_simulation_eval.py \
  --runs-dir runs/edge_attr_meaning_breast/hybrid_a50_bio_k8 \
  --graphs-root datasets/ispy2/graphs_consistent \
  --folds datasets/ispy2/folds.parquet \
  --cohort datasets/ispy2/cohort.parquet \
  --out-dir reports/edge_attr_meaning_breast_eval/hybrid_a50_bio_k8 \
  --checkpoint-name best.pt
```

External deterministic source-fold ensemble:

```bash
sbatch cradle/run_breast_nact_external_4visit_paper_models_eval.sbatch
python experiments/breast_nact_external/summarize_external_deterministic.py \
  --root reports/breast_mri_nact_external/deterministic_4visit_paper_models \
  --out-dir reports/breast_mri_nact_external/tables_paper_models \
  --markdown-out reports/breast_mri_nact_external/external_4visit_paper_models_deterministic_summary.md
```

## Residual Monte Carlo And Conformal Calibration

The residual Monte Carlo layer is a fixed empirical wrapper around each
deterministic rollout center. It samples held-out source-cohort residuals
within the same conditioning bucket and excludes the target patient from the
calibration pool before sampling or conformal expansion.

The original baseline-to-Endpoint+Active comparison used `N_MC=256`. The larger
edge-ablation screen used `N_MC=128` and `METRIC_DRAWS=32` to keep all
graph-family models comparable under the same uncertainty layer. The external
NACT stress test used source-cohort residual pools only. NACT residuals were
used only for final scoring, not for MC or conformal calibration.

Internal retained model MC:

```bash
sbatch experiments/consistent_rollout/run_edge_attribute_meaning_conditional_mc.sbatch
```

External source-residual MC:

```bash
sbatch cradle/run_breast_nact_external_4visit_paper_models_source_residual_mc.sbatch
```

The source-residual MC script defaults to the four paper-family external
models. Additional historical or exploratory model keys remain available in
the script for local audits, but they are not part of the manuscript-facing
external stress-test table.

## Scalar And Hybrid Controls

The manuscript separates the graph-state claim from the scalar endpoint claim.
Scalar and hybrid FTV baselines use only observed FTV history through the
conditioning visit. The hybrid scalar readout combines retained graph FTV with
last-observed and baseline FTV using target-fold-excluded training.

Main scalar-control artifact:

| Path | Purpose |
| --- | --- |
| `paper/tables/scalar_vs_graph_mc_t3.csv` | T0/T1/T2-to-T3 scalar, graph, and hybrid MC comparison. |

This table supports the statement that task-adapted scalar and hybrid centers
remain strong for scalar FTV alone, while the graph model contributes a
structured tumor-state forecast and active-node dynamics.

## Manuscript Table And Figure Map

| Manuscript item | Supporting artifact |
| --- | --- |
| Synthetic mechanism table | `paper/tables/ablation_t0t3_mc.csv` and `results/bio_ftv_synthetic_ablation_analysis/` |
| Internal main model comparison | `paper/tables/latest_model_comparison_t0_t3.csv` |
| Edge-neighborhood and edge-attribute ablations | `paper/tables/latest_model_collection_t0_t3.csv`, `paper/tables/latest_model_subtype_t0_t3.csv`, and edge MC result roots |
| Split and leakage audit | `paper/tables/split_leakage_audit.csv` |
| Scalar and hybrid controls | `paper/tables/scalar_vs_graph_mc_t3.csv` |
| Calibration by burden and subtype | `paper/tables/calibration_by_t3_burden_quartile.csv`, `paper/tables/calibration_subgroup_robustness.csv` |
| MRI-burden monitoring readouts | `paper/tables/clinical_burden_threshold_readouts.csv`, `paper/tables/decision_curve_t3_burden.csv` |
| External NACT stress test | `paper/tables/external_nact_stress_test_t0t3.csv` |

Paper figures and tables can be regenerated from the included derived result
roots with:

```bash
python paper/make_manuscript_support.py
```

The supplementary material can be regenerated with:

```bash
python paper/make_supplementary_material.py
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary_material.tex
```

The paper PDF can be rebuilt with:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Claims And Boundaries

The supported manuscript claim is an endpoint-calibrated graph rollout for
future MRI imaging-burden forecasting. The method forecasts future tumor graph
state, active-node state, and FTV readouts. It does not claim treatment
counterfactual optimization, pathology response prediction, or clinical
deployment readiness.

The external NACT stress test supports a transfer check on independent
graph-ready patients. It should not be presented as definitive external
clinical validation because the reported external cohort has 11 patients.

The strongest scalar FTV readouts are included because they define the boundary
of the graph claim: graph structure is not claimed to dominate every scalar
FTV-only baseline. The retained graph model is justified by the combined
endpoint calibration, uncertainty behavior, active-node behavior, and
graph-native tumor-state forecast.
