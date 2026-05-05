# Data Manifest

This private repository includes derived data products needed to reproduce the
paper analyses and figures. It intentionally excludes the raw DCE-MRI image
volumes.

## Included

```text
data/ispy2/cohort.parquet
data/ispy2/folds.parquet
data/ispy2/cohort_audit.md
data/ispy2/folds_audit.md
data/ispy2/arm_audit.md
data/ispy2/graphs_consistent/*.pt
```

The graph tensor directory contains 758 patient-level longitudinal graph files.
Each file stores the derived supervoxel graph representation used by the
forecaster: node features, positions, visit offsets, alive labels, and related
metadata.

## Not Included

Raw MRI image volumes, raw segmentation masks, raw DICOM/NIfTI files, and large
intermediate preprocessing outputs are not included in this repository.

## Derived Results

The deterministic and Monte Carlo result tables are under:

```text
results/
```

These include patient-level derived metrics and MC draws used for analysis and
plot generation.

## Privacy and Sharing

This repository should remain private unless the data packaging is revised to
remove restricted patient-level derived files and replace them with public-safe
aggregate summaries and external data-access instructions.

