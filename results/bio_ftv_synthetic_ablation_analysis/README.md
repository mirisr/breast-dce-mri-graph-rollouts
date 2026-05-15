# Synthetic Validation and MC Ablation Analysis

## Interpretation Notes

- Synthetic endpoint calibration improves endpoint error and uncertainty while largely preserving spatial rollout metrics.
- Synthetic no-edge training improves deterministic scalar FTV in this run, but its MC endpoint score is worse than the full endpoint-calibrated model and its spatial metrics are worse.
- The synthetic tail-shift regime remains harder and under-covered after narrowing, which demonstrates the support/exchangeability condition.
- MC simulations are an evaluation/calibration layer, not a separate trained model family. The report separates the standalone MC experiment lineage from MC evaluation of graph ablations.
- In the MC evaluation of graph ablations, the breast no-scheduled-sampling output is numerically identical to the full endpoint model to rounding. The selected full-model checkpoint was epoch 6, before scheduled sampling became active, so this ablation does not isolate scheduled sampling.
- The breast no-edge ablation improves scalar endpoint metrics with nearly unchanged spatial MC metrics. This suggests graph message passing is not currently helping scalar FTV after endpoint calibration, although the graph-state representation remains necessary for spatial/state outputs.

## Main Synthetic Result: T0 to T3

| label                                |   n_patients |   det_ftv_mae_ml |   mc_mean_ftv_mae_ml |   crps_ftv_ml |   conformal_coverage90 |   conformal_width90_ml |   swd_mm_mc |   chamfer_mm_mc |   alive_abs_err_mc |
|:-------------------------------------|-------------:|-----------------:|---------------------:|--------------:|-----------------------:|-----------------------:|------------:|----------------:|-------------------:|
| Geometry + scheduled sampling        |          240 |           10.09  |               10.638 |         7.072 |                  0.988 |                 57.354 |       2.927 |           4.408 |             14.931 |
| Endpoint FTV + alive calibration     |          240 |            8.608 |                8.107 |         5.661 |                  0.971 |                 35.022 |       2.966 |           4.405 |             14.889 |
| Endpoint calibration, no graph edges |          240 |            6.564 |                8.649 |         5.823 |                  0.962 |                 36.943 |       3.363 |           4.885 |             14.707 |

## Synthetic Support-Regime Stress Test

| label                                | support_regime   |   n_patients |   mc_mean_ftv_mae_ml |   crps_ftv_ml |   conformal_coverage90 |   conformal_width90_ml |
|:-------------------------------------|:-----------------|-------------:|---------------------:|--------------:|-----------------------:|-----------------------:|
| Geometry + scheduled sampling        | in_support       |          221 |                6.512 |         4.067 |                  1     |                 51.768 |
| Geometry + scheduled sampling        | tail_shift       |           19 |               58.63  |        42.023 |                  0.842 |                122.331 |
| Endpoint FTV + alive calibration     | in_support       |          221 |                5.911 |         4.302 |                  0.995 |                 28.685 |
| Endpoint FTV + alive calibration     | tail_shift       |           19 |               33.639 |        21.469 |                  0.684 |                108.734 |
| Endpoint calibration, no graph edges | in_support       |          221 |                7.747 |         5.065 |                  0.982 |                 32.107 |
| Endpoint calibration, no graph edges | tail_shift       |           19 |               19.147 |        14.636 |                  0.737 |                 93.193 |

## Standalone Breast MC Experiment Lineage: T0 to T3

| label                            |   n_patients |   det_ftv_mae_ml |   mc_mean_ftv_mae_ml |   crps_ftv_ml |   raw_coverage90 |   conformal_coverage90 |   conformal_width90_ml |   swd_mm_mc |   chamfer_mm_mc |   alive_abs_err_mc |
|:---------------------------------|-------------:|-----------------:|---------------------:|--------------:|-----------------:|-----------------------:|-----------------------:|------------:|----------------:|-------------------:|
| Original consistent rollout MC   |          758 |           12.878 |               15.262 |         8.354 |            0.766 |                  0.901 |                 59.515 |       1.32  |           3.656 |             40.783 |
| Endpoint FTV 0.10, no alive mass |          758 |            5.995 |                7.306 |         5.145 |            0.875 |                  0.901 |                 26.923 |       1.214 |           3.498 |              7.005 |
| Endpoint FTV 0.10 + light alive  |          758 |            5.987 |                7.525 |         5.186 |            0.883 |                  0.901 |                 29.138 |       1.205 |           3.485 |              2.704 |
| Endpoint FTV 0.20 + alive 0.05   |          758 |            5.881 |                7.609 |         5.156 |            0.876 |                  0.901 |                 27.949 |       1.207 |           3.485 |              2.692 |

## MC Evaluation of Breast Graph Ablations: T0 to T3

| label                 |   n_patients |   det_ftv_mae_ml |   mc_mean_ftv_mae_ml |   crps_ftv_ml |   raw_coverage90 |   conformal_coverage90 |   conformal_width90_ml |   swd_mm_mc |   chamfer_mm_mc |   alive_abs_err_mc |
|:----------------------|-------------:|-----------------:|---------------------:|--------------:|-----------------:|-----------------------:|-----------------------:|------------:|----------------:|-------------------:|
| Full endpoint model   |          758 |            5.881 |                7.609 |         5.156 |            0.876 |                  0.901 |                 27.949 |       1.207 |           3.485 |              2.692 |
| No scheduled sampling |          758 |            5.881 |                7.609 |         5.156 |            0.876 |                  0.901 |                 27.949 |       1.207 |           3.485 |              2.692 |
| No graph edges        |          758 |            5.28  |                6.541 |         4.525 |            0.894 |                  0.901 |                 23.93  |       1.211 |           3.493 |              2.654 |

## Conditioning-to-T3 Tables

Synthetic:

| label                                | cohort             |   n_patients |   mc_mean_ftv_mae_ml |   crps_ftv_ml |   conformal_width90_ml |   conformal_coverage90 |
|:-------------------------------------|:-------------------|-------------:|---------------------:|--------------:|-----------------------:|-----------------------:|
| Geometry + scheduled sampling        | synthetic_T0_to_T3 |          240 |               10.638 |         7.072 |                 57.354 |                  0.988 |
| Geometry + scheduled sampling        | synthetic_T1_to_T3 |          240 |                8.213 |         5.426 |                 46.493 |                  0.979 |
| Geometry + scheduled sampling        | synthetic_T2_to_T3 |          240 |                7.876 |         4.964 |                 41.07  |                  0.988 |
| Endpoint FTV + alive calibration     | synthetic_T0_to_T3 |          240 |                8.107 |         5.661 |                 35.022 |                  0.971 |
| Endpoint FTV + alive calibration     | synthetic_T1_to_T3 |          240 |                3.841 |         2.617 |                 18.449 |                  0.954 |
| Endpoint FTV + alive calibration     | synthetic_T2_to_T3 |          240 |                1.899 |         1.314 |                 10.207 |                  0.954 |
| Endpoint calibration, no graph edges | synthetic_T0_to_T3 |          240 |                8.649 |         5.823 |                 36.943 |                  0.962 |
| Endpoint calibration, no graph edges | synthetic_T1_to_T3 |          240 |                4.423 |         3.07  |                 22.93  |                  0.975 |
| Endpoint calibration, no graph edges | synthetic_T2_to_T3 |          240 |                2.147 |         1.511 |                 12.036 |                  0.971 |

Standalone breast MC experiments:

| label                            | cohort          |   n_patients |   mc_mean_ftv_mae_ml |   crps_ftv_ml |   conformal_width90_ml |   conformal_coverage90 |
|:---------------------------------|:----------------|-------------:|---------------------:|--------------:|-----------------------:|-----------------------:|
| Original consistent rollout MC   | breast_T0_to_T3 |          758 |               15.262 |         8.354 |                 59.515 |                  0.901 |
| Original consistent rollout MC   | breast_T1_to_T3 |          758 |               14.875 |         8.098 |                 60.144 |                  0.901 |
| Original consistent rollout MC   | breast_T2_to_T3 |          758 |               13.802 |         7.49  |                 60.25  |                  0.901 |
| Endpoint FTV 0.10, no alive mass | breast_T0_to_T3 |          758 |                7.306 |         5.145 |                 26.923 |                  0.901 |
| Endpoint FTV 0.10, no alive mass | breast_T1_to_T3 |          758 |                7.17  |         4.875 |                 31.004 |                  0.901 |
| Endpoint FTV 0.10, no alive mass | breast_T2_to_T3 |          758 |                5.887 |         4.016 |                 24.559 |                  0.901 |
| Endpoint FTV 0.10 + light alive  | breast_T0_to_T3 |          758 |                7.525 |         5.186 |                 29.138 |                  0.901 |
| Endpoint FTV 0.10 + light alive  | breast_T1_to_T3 |          758 |                7.136 |         4.898 |                 30.448 |                  0.901 |
| Endpoint FTV 0.10 + light alive  | breast_T2_to_T3 |          758 |                5.806 |         4.029 |                 24.834 |                  0.901 |
| Endpoint FTV 0.20 + alive 0.05   | breast_T0_to_T3 |          758 |                7.609 |         5.156 |                 27.949 |                  0.901 |
| Endpoint FTV 0.20 + alive 0.05   | breast_T1_to_T3 |          758 |                7.185 |         4.889 |                 29.613 |                  0.901 |
| Endpoint FTV 0.20 + alive 0.05   | breast_T2_to_T3 |          758 |                5.802 |         4.025 |                 24.193 |                  0.901 |

MC evaluation of breast graph ablations:

| label                 | cohort          |   n_patients |   mc_mean_ftv_mae_ml |   crps_ftv_ml |   conformal_width90_ml |   conformal_coverage90 |
|:----------------------|:----------------|-------------:|---------------------:|--------------:|-----------------------:|-----------------------:|
| Full endpoint model   | breast_T0_to_T3 |          758 |                7.609 |         5.156 |                 27.949 |                  0.901 |
| Full endpoint model   | breast_T1_to_T3 |          758 |                7.185 |         4.889 |                 29.613 |                  0.901 |
| Full endpoint model   | breast_T2_to_T3 |          758 |                5.802 |         4.025 |                 24.193 |                  0.901 |
| No scheduled sampling | breast_T0_to_T3 |          758 |                7.609 |         5.156 |                 27.949 |                  0.901 |
| No scheduled sampling | breast_T1_to_T3 |          758 |                7.185 |         4.889 |                 29.613 |                  0.901 |
| No scheduled sampling | breast_T2_to_T3 |          758 |                5.802 |         4.025 |                 24.193 |                  0.901 |
| No graph edges        | breast_T0_to_T3 |          758 |                6.541 |         4.525 |                 23.93  |                  0.901 |
| No graph edges        | breast_T1_to_T3 |          758 |                6.396 |         4.392 |                 26.899 |                  0.901 |
| No graph edges        | breast_T2_to_T3 |          758 |                5.332 |         3.776 |                 22.372 |                  0.901 |

## Figure Files

- `figures/synthetic_t0_t3_endpoint.png`
- `figures/synthetic_t0_t3_coverage_width.png`
- `figures/synthetic_support_t0_t3.png`
- `figures/synthetic_conditioning_t3.png`
- `figures/breast_mc_lineage_t0_t3_endpoint.png`
- `figures/breast_mc_lineage_t0_t3_coverage_width.png`
- `figures/breast_mc_lineage_conditioning_t3.png`
- `figures/breast_ablation_t0_t3_endpoint.png`
- `figures/breast_ablation_t0_t3_coverage_width.png`
- `figures/breast_ablation_conditioning_t3.png`
