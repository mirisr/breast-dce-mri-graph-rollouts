# Latest Cradle Result Analysis

Date: 2026-05-14

## Cradle Status

- Live Slurm queue snapshot: no active jobs for `irisseaman01`.
- Slurm accounting is disabled on Cradle, so `sacct` cannot provide completed-job history.
- Expected raw `reports/real_stratified_mc/...` roots were not present locally, but the derived stratified analysis tables are present under `reports/bio_ftv_real_stratified_ablation`.
- Synthetic neighbor-coupled and latent spatial-field evaluation/MC summaries are present and timestamped from this morning's runs.

## Real Cohort: Overall T0 to T3

The edge-aware models are clearly better than the older endpoint-calibrated model and the original graph rollout.

| Model | MC FTV MAE | CRPS | Raw 90% Cov. | Raw Width | Conformal Width | MC Chamfer |
|---|---:|---:|---:|---:|---:|---:|
| Original graph | 15.26 | 8.35 | 0.766 | 56.87 | 59.51 | 3.66 |
| Endpoint-calibrated | 7.61 | 5.16 | 0.876 | 24.42 | 27.95 | 3.49 |
| Radial-bio k=8 | 5.52 | 3.95 | 0.872 | 20.05 | 23.29 | 3.46 |
| Hybrid-bio k=8 | 5.56 | 4.00 | 0.877 | 19.21 | 22.38 | 3.45 |

Interpretation:

- `radial_bio_k8` is marginally best for scalar MC FTV MAE.
- `hybrid_a50_bio_k8` remains the best single publication model because it has the best balance: nearly identical scalar error, better raw coverage, narrower intervals, and slightly better sampled-cloud geometry.
- The right paper claim is not that one topology wins every scalar metric; it is that edge-aware message passing improves the full forecasting stack over the previous endpoint-calibrated model.

## Real Cohort: Collection Robustness

`hybrid_a50_bio_k8` is stable across both data sources.

| Collection | N | Hybrid-bio MC FTV MAE | CRPS | Raw 90% Cov. | Raw Width | Conformal Width |
|---|---:|---:|---:|---:|---:|---:|
| ACRIN-6698 | 202 | 4.55 | 2.93 | 0.881 | 18.61 | 21.78 |
| ISPY2 | 556 | 5.93 | 4.39 | 0.876 | 19.42 | 22.59 |

Interpretation:

- Performance is better on ACRIN-6698 and harder on ISPY2, but the interval behavior stays similar.
- This supports a real-data robustness claim across two sources without needing separate dataset-specific retraining.

## Real Cohort: Subtype Robustness

`hybrid_a50_bio_k8` is strongest or near strongest across subtypes.

| Subtype | N | Hybrid-bio MC FTV MAE | CRPS | Raw 90% Cov. | Raw Width | Conformal Width |
|---|---:|---:|---:|---:|---:|---:|
| HR+/HER2+ | 128 | 4.90 | 3.13 | 0.906 | 19.59 | 22.76 |
| HR+/HER2- | 312 | 5.24 | 3.77 | 0.869 | 18.97 | 22.14 |
| TNBC | 261 | 5.86 | 4.26 | 0.866 | 19.33 | 22.50 |
| HR-/HER2+ | 57 | 7.52 | 6.01 | 0.912 | 19.08 | 22.25 |

Interpretation:

- HR-/HER2+ remains the hardest subgroup by scalar error.
- Coverage is not the failure mode for HR-/HER2+; absolute error is. That points to center calibration/tail behavior rather than interval widening alone.

## Synthetic: Node-Local Cohort

This cohort does not strongly reward message passing because the data-generating process is mostly node-local.

Key result:

- `no_edges` has the best deterministic FTV MAE and alive-count MAE.
- Edge-aware models can improve sampled geometry, but they do not clearly improve scalar FTV.

Interpretation:

- This is a useful negative control. When the generative process does not require neighbor information, message passing should not be expected to dominate.

## Synthetic: Neighbor-Coupled Cohort

The first neighbor-coupled generator shows a modest graph benefit, mostly in MC geometry rather than scalar FTV.

`hybrid_a50_bio_k8` versus `no_edges`:

- deterministic FTV MAE improves by 0.44 mL.
- CRPS improves by 0.20.
- raw interval width narrows by 1.02 mL.
- MC Chamfer improves by 0.62 mm.
- MC FTV MAE is 0.40 mL worse than `no_edges`.

Interpretation:

- Neighbor information helps some graph-state and uncertainty metrics, but this generator is not strong enough to make the edge-aware model win every layer.
- It is not the best synthetic demonstration for the manuscript.

## Synthetic: Latent Spatial-Field Cohort

This is the strongest synthetic evidence that neighbors can be meaningful.

`hybrid_a50_bio_k8` versus `no_edges`:

- deterministic FTV MAE improves by 2.06 mL.
- MC FTV MAE improves by 5.43 mL.
- CRPS improves by 3.53.
- raw interval width narrows by 24.14 mL.
- MC Chamfer improves by 0.41 mm.

Interpretation:

- This is the cleanest synthetic result for the paper.
- When tumor dynamics contain a hidden smooth local resistance field and node features expose only a noisy local proxy, message passing helps recover spatially coherent response.
- This provides the right synthetic-first argument: graph neighborhoods matter when the data contain local spatial dependence that cannot be fully read from each node alone.

## Main Takeaway

The current evidence supports a stronger paper organization:

1. Synthetic controlled validation first:
   - Node-local synthetic data: message passing is not automatically beneficial.
   - Latent spatial-field synthetic data: message passing becomes useful when true local tumor-state dependence exists.
2. Real cohort second:
   - `hybrid_a50_bio_k8` is the retained model because it improves the full stack relative to the old endpoint-calibrated model.
   - It is robust across ACRIN-6698, ISPY2, and subtypes.
3. Discussion:
   - The method works best when data have persistent graph nodes, consistent longitudinal imaging features, and local spatial/anatomical dependence that neighbors can denoise or propagate.
   - If the data are node-local or neighborhood definitions are noisy, endpoint calibration still helps scalar FTV, but message passing contributes less.

