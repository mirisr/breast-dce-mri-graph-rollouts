# Data Manifest

This public repository does not include raw breast DCE-MRI image volumes,
segmentation masks, processed graph tensors, cohort parquet files, fold parquet
files, patient-level residual Monte Carlo outputs, or generated paper tables.

## Source Data

The source DCE-MRI imaging collections are publicly available through their
custodial imaging archives, including TCIA-hosted I-SPY2 and ACRIN-6698
resources. Users should obtain the imaging data directly from those archives and
follow the applicable data-use terms.

## Included Here

```text
data/DATA_MANIFEST.md
data/ispy2/README.md
```

These files document the expected data layout and rebuild process. They are not
data extracts.

## Generated Locally

After obtaining the source imaging data, local preprocessing scripts can
generate:

```text
data/ispy2/cohort.parquet
data/ispy2/folds.parquet
data/ispy2/graphs_consistent/*.pt
results/
paper/
models/*/fold*/best.pt
```

These generated artifacts are intentionally ignored by git in the public
release. Keep them local unless a separate release decision explicitly allows a
specific artifact to be distributed.
