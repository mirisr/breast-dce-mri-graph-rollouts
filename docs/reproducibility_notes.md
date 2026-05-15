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
