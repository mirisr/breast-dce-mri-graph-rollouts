# Reproducibility Notes

## Current Retained Model

The retained manuscript model is:

```text
hybrid_a50_bio_k8
```

It is reported as `Hybrid-Edge k=8`. It extends the Endpoint+Active calibration
model with a hybrid spatial-feature neighborhood and radial imaging-feature
edge attributes. The `bio` substring in run tags is a historical internal label
for these attributes and endpoint-burden losses, not a biological-marker claim.

The retained configuration is recorded in:

```text
configs/hybrid_a50_bio_k8.json
```

## Public Repository Scope

This public repository provides code and rebuild instructions. It does not
bundle generated paper data, paper figures, paper tables, processed graph
tensors, trained checkpoints, residual Monte Carlo samples, or patient-level
evaluation outputs.

To reproduce numerical results, obtain the public source imaging collections
from their custodial archives, rebuild the graph tensors locally, train or
restore fold-specific checkpoints, and then run the deterministic and residual
Monte Carlo evaluation scripts.

## Evaluation Design

The main held-out evaluation is T0-to-T3 FTV forecasting on the graph-bearing
I-SPY2/ACRIN cohort. The primary metrics are:

- MC-mean FTV MAE;
- CRPS;
- raw 90% coverage;
- conformal 90% interval width.

Secondary analyses include deterministic bias/MAE, active-node error, SWD,
Chamfer, Dice, final-visit horizon checks, subtype/source/burden strata, and
MRI-burden threshold readouts.

The independent Breast-MRI-NACT-Pilot analysis should be interpreted as a
preliminary external stress test, not as powered clinical external validation.

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

## Expected Local Output Roots

When the pipeline is rerun locally, generated outputs are expected under local
ignored paths such as:

```text
data/ispy2/graphs_consistent/
models/<model_tag>/fold*/best.pt
results/
paper/
```

These paths are intentionally ignored by git in the public release.
