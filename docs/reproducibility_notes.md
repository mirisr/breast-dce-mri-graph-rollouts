# Reproducibility Notes

## Retained Model

The retained paper model is:

```text
bio_ftv020_alive005
```

The retained model was selected because it improved deterministic T3 FTV
centering while preserving graph geometry and alive-supervoxel behavior.

## Conditional Monte Carlo Design

The conditional MC sampler was run without sampler modifications for the
retrained models. This isolates whether the improved deterministic FTV center
also improves probabilistic calibration.

Key settings:

```text
N_MC=256
METRIC_DRAWS=0
SEED=42
start visits: T0, T1, T2
target visits: later visits through T3
```

The sampler uses residual buckets by `(start_visit, predicted_visit)`, excludes
the target patient from calibration residuals, samples alive masks with Gumbel
top-k, and reports both raw and conformal FTV intervals.

## Main Comparison

The main comparison is:

```text
baseline_v2_sched_samp
bio_ftv020_alive005
bio_ftv010_alive000
bio_ftv010_alive002
```

The key question is whether the retained model improves T0-to-T3 probabilistic
FTV calibration while preserving spatial graph quality.

