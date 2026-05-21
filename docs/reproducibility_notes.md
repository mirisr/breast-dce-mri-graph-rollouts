# Reproducibility Notes

## Current Retained Model

The current manuscript model is:

```text
hybrid_a50_bio_k8
```

It is reported as `Hybrid-Edge k=8`. It extends the Endpoint+Active calibration
model with a hybrid spatial-feature neighborhood and radial-biologic edge
attributes.

The retained configuration is recorded in:

```text
configs/hybrid_a50_bio_k8.json
```

For the full reviewer-facing audit trail, including manuscript table paths,
external stress-test details, and launch commands, see:

```text
docs/SUPPLEMENTARY_REPRODUCIBILITY_DETAILS.md
```

For the manuscript-facing supplementary tables, see:

```text
paper/supplementary_material.pdf
paper/supplementary_material.tex
```

## Main Evaluation Path

The main held-out evaluation is T0-to-T3 FTV forecasting on the graph-bearing
I-SPY2/ACRIN cohort. The primary metrics are:

- MC-mean FTV MAE;
- CRPS;
- raw 90% coverage;
- conformal 90% interval width.

Secondary analyses include deterministic bias/MAE, active-node error, SWD,
Chamfer, Dice, final-visit horizon checks, subtype/source/burden strata, and
MRI-burden threshold readouts.

The independent Breast-MRI-NACT-Pilot stress test is reported only for the four
paper-family endpoint-calibrated models:

```text
Endpoint+Active
No-edge endpoint
Radial-biologic k=8
Hybrid-Edge k=8
```

The paper-facing external table is:

```text
paper/tables/external_nact_stress_test_t0t3.csv
```

It should be interpreted as preliminary external validation on 11 graph-ready
patients, not as powered clinical external validation.

## Conditional Monte Carlo Design

The conditional MC sampler is a fixed residual wrapper around each deterministic
rollout center. This isolates whether model changes improve the center under a
constant uncertainty layer.

Final model-selection edge-ablation settings:

```text
N_MC=128
METRIC_DRAWS=32
SEED=20260513
residual_stratify_by=none
interval=0.90
```

The original baseline-to-Endpoint+Active comparison used:

```text
N_MC=256
METRIC_DRAWS=0
start visits: T0, T1, T2
target visits: later visits through T3
```

The sampler uses residual buckets by `(start_visit, predicted_visit)`, excludes
the target patient from calibration residuals, samples active masks with Gumbel
top-k, and reports raw and conformal FTV intervals.

## Main Result Roots

```text
results/conditional_mc_consistent_rollout/
results/conditional_mc_bio_retrained/
results/edge_meaning_breast_mc/
results/edge_attr_meaning_breast_mc/
results/edge_meaning_synthetic_spatial_field_mc/
results/edge_attr_meaning_synthetic_spatial_field_mc/
```

The current retained model MC outputs are:

```text
results/edge_attr_meaning_breast_mc/hybrid_a50_bio_k8/
```

## Historical Model Role

`bio_ftv020_alive005` is still included because it is the historical
Endpoint+Active calibration comparator, but it is no longer the final retained
publication model.
