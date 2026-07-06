# Model Manifest

This public repository does not include trained checkpoint files, fold-specific
training logs, or model output artifacts.

## Current Manuscript Model

The retained manuscript model is:

```text
hybrid_a50_bio_k8
```

It is reported in the paper as `Hybrid-Edge k=8`. The configuration is kept at:

```text
configs/hybrid_a50_bio_k8.json
```

## Expected Local Checkpoint Layout

After retraining or restoring weights in a local working environment, checkpoint
families should use this layout:

```text
models/<model_tag>/fold0/best.pt
models/<model_tag>/fold1/best.pt
models/<model_tag>/fold2/best.pt
models/<model_tag>/fold3/best.pt
models/<model_tag>/fold4/best.pt
```

Generated checkpoint files and logs are ignored by git in the public release.
