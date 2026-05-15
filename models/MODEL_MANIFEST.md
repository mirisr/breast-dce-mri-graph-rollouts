# Model Manifest

This directory currently records the checkpoint families needed for full
reproducibility. It does not currently include checkpoint files.

## Checkpoint Families To Mirror

```text
models/baseline_v2_sched_samp/
models/bio_ftv010_alive000/
models/bio_ftv010_alive002/
models/bio_ftv020_alive005/
```

Each model directory is organized by fold:

```text
fold0/best.pt
fold1/best.pt
fold2/best.pt
fold3/best.pt
fold4/best.pt
```

When mirrored from the training environment, folds may also include `last.pt`
and `train_log.json`.

## Current Manuscript Model

The current retained manuscript model is:

```text
hybrid_a50_bio_k8
```

It is reported in the paper as `Hybrid-Edge k=8`. The model uses:

- Endpoint+Active FTV and active-burden calibration;
- hybrid spatial-feature neighborhoods with equal spatial/feature weighting;
- radial-biologic edge attributes;
- scheduled sampling with the same held-out patient fold protocol.

The training, deterministic evaluation, and conditional MC scripts for this
model are included under `experiments/`. The MC outputs used by the paper are
included under:

```text
results/edge_attr_meaning_breast_mc/hybrid_a50_bio_k8/
```

## Checkpoint Packaging Gap

No checkpoint files are currently mirrored into this local publication package.
Before claiming that a public release contains trained weights, mirror the
intended checkpoint families into `models/`. The final Hybrid-Edge fold
checkpoints should be placed at:

```text
models/hybrid_a50_bio_k8/fold0/best.pt
models/hybrid_a50_bio_k8/fold1/best.pt
models/hybrid_a50_bio_k8/fold2/best.pt
models/hybrid_a50_bio_k8/fold3/best.pt
models/hybrid_a50_bio_k8/fold4/best.pt
```

Until then, this repository contains the current scripts and final MC outputs,
but not trained weights.

## Historical Models

`bio_ftv020_alive005` remains scientifically important as the first
Endpoint+Active calibration model, but it is no longer the final retained paper
model.
