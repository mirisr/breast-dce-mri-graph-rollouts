# Endpoint-Calibrated Graph Rollouts for Breast DCE-MRI

This repository packages the code, derived analysis artifacts, figures, tables,
and manuscript source for endpoint-calibrated spatial graph rollouts for breast
DCE-MRI functional tumor volume (FTV) forecasting.

The current manuscript model is `hybrid_a50_bio_k8`, reported in the paper as
`Hybrid-Edge k=8`. It combines endpoint and active-burden calibration with a
hybrid spatial-feature graph neighborhood and radial-biologic edge attributes.
The older `bio_ftv020_alive005` model is retained as the historical
Endpoint+Active calibration baseline, not the final publication model.

## Current Paper

The manuscript package is under `paper/`.

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The current compiled manuscript is also included as:

```text
paper/bio_ftv020_mc_manuscript.pdf
```

## Main Claims Supported By This Package

- Synthetic graph controls test when message passing helps.
- Endpoint and active-burden losses correct the low-FTV center of the original
  graph rollout.
- The retained Hybrid-Edge model improves held-out T0-to-T3 FTV error,
  coverage, CRPS, and conformal interval width relative to the original graph
  rollout.
- Scalar and hybrid FTV baselines define the boundary of the graph claim: scalar
  centers remain strong for scalar FTV, while the graph model contributes
  structured tumor-state forecasts and active-node dynamics.
- MRI-burden threshold readouts are reported as imaging-burden monitoring
  outputs, not pathology response claims.

## Repository Layout

```text
data/                         derived graph metadata and graph tensors
experiments/stage1_forecaster training, deterministic eval, edge ablations
experiments/consistent_rollout residual MC evaluation and Slurm wrappers
models/                       model manifest and checkpoint packaging notes
notebooks/                    analysis notebooks and plotting helpers
paper/                        IEEE Access manuscript source, figures, tables
results/                      derived result tables, MC outputs, and summaries
docs/                         reproducibility notes and historical planning notes
environment/                  minimal dependency notes
cradle/                       cluster setup and launch notes
```

The compatibility links are intentional:

```text
lsgc -> src/lsgc
reports -> results
datasets/ispy2 -> ../data/ispy2
```

They preserve the development paths used by the training, evaluation, and
manuscript-support scripts.

## Current Key Artifacts

| Path | Purpose |
| --- | --- |
| `paper/main.tex` | Current IEEE Access manuscript source. |
| `paper/figures/` | Current paper figures, including synthetic, calibration, edge-ablation, and clinical burden panels. |
| `paper/tables/` | Current paper-facing CSV tables. |
| `paper/make_manuscript_support.py` | Regenerates paper tables and support figures from `results/`. |
| `experiments/stage1_forecaster/train_consistent_forecaster_v2.py` | Current rollout training entry point with endpoint losses, dynamic edge modes, and optional edge attributes. |
| `experiments/stage1_forecaster/edge_modes.py` | Dynamic no-edge, spatial, radial, feature, hybrid, and radial-biologic graph construction utilities. |
| `experiments/stage1_forecaster/run_edge_meaning_grid.sbatch` | Stage-one edge-neighborhood ablation launcher. |
| `experiments/stage1_forecaster/run_edge_attribute_meaning_grid.sbatch` | Stage-two edge-attribute ablation launcher. |
| `experiments/consistent_rollout/run_conditional_mc.py` | Conditional residual Monte Carlo evaluator. |
| `notebooks/breast_cohort_mc_and_graph_ablation_results.ipynb` | Current breast cohort and synthetic ablation comparison notebook. |
| `notebooks/breast_edge_attribute_publication_model_results.ipynb` | Edge-attribute publication-model analysis notebook. |
| `notebooks/clinical_burden_monitoring_results.ipynb` | MRI-burden monitoring readout notebook. |

## Current Result Roots

The manuscript-support code reads from these result roots:

```text
results/conditional_mc_consistent_rollout/
results/conditional_mc_bio_retrained/
results/edge_meaning_breast_mc/
results/edge_attr_meaning_breast_mc/
results/edge_meaning_synthetic_spatial_field_mc/
results/edge_attr_meaning_synthetic_spatial_field_mc/
results/bio_ftv_latest_job_analysis/
results/bio_ftv_synthetic_ablation_analysis/
results/bio_ftv_clinical_burden_monitoring/
results/bio_ftv_real_stratified_ablation/
```

The final retained model result used by the manuscript is:

```text
results/edge_attr_meaning_breast_mc/hybrid_a50_bio_k8/
```

## Data Status

This repository does not include raw DCE-MRI image volumes. The source imaging
collections are public/controlled-access TCIA resources, and users should obtain
those datasets through TCIA under the relevant data-use terms.

This working repository currently includes derived graph tensors and
patient-level derived result files. Before making the repository public, review
`RELEASE_AUDIT.md` and decide whether to keep those derived patient-level files
in the public release or replace them with aggregate paper tables plus
instructions for rebuilding the derived artifacts from TCIA data.

## Reproducibility Notes

The current paper can be audited from the included derived result files and
paper tables. Re-running the full model training path requires the graph
artifacts, fold assignments, and the training scripts in `experiments/`.

The final Hybrid-Edge training/evaluation scripts and MC outputs are included,
but checkpoint files are not currently mirrored into this local publication
package. Before claiming that the public repo contains trained weights, mirror
the intended checkpoint families into `models/` and update
`models/MODEL_MANIFEST.md`.

## Public Release Rule

Do not make this repository public until:

1. the release audit has been completed;
2. the data and model manifests match what will actually be distributed;
3. no raw or restricted imaging files are present;
4. no local credentials, cluster-only paths, or scratch logs are present;
5. the README and manuscript code-availability statement point to the final
   public repository URL.
