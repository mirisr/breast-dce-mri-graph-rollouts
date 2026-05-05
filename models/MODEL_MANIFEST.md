# Model Manifest

This directory contains the checkpoints and training logs used for the retained
paper experiments.

## Included Models

```text
models/baseline_v2_sched_samp/
models/bio_ftv010_alive000/
models/bio_ftv010_alive002/
models/bio_ftv020_alive005/
```

## Primary Model

```text
models/bio_ftv020_alive005/
```

This is the retained biology-calibrated scheduled-sampling rollout model used
as the primary candidate in the paper.

## Control Models

```text
models/bio_ftv010_alive000/
models/bio_ftv010_alive002/
```

These are the retained-model controls used in the conditional MC comparison.

## Baseline Model

```text
models/baseline_v2_sched_samp/
```

This is the earlier scheduled-sampling rollout model used as the pre-retraining
baseline.

## Checkpoint Structure

Each model directory contains fold-level checkpoints:

```text
fold0/best.pt
fold1/best.pt
fold2/best.pt
fold3/best.pt
fold4/best.pt
```

Most directories also include `last.pt` and `train_log.json` files.

