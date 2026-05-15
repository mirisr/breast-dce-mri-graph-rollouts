# Data Manifest

This repository does not include raw breast DCE-MRI image volumes. Source
imaging data should be obtained through TCIA under the applicable I-SPY2 and
ACRIN-6698 data-use terms.

## Included In This Working Repository

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
forecaster: node features, positions, visit offsets, active-node labels, and
related metadata.

## Not Included

Raw MRI image volumes, raw segmentation masks, raw DICOM/NIfTI files, and large
intermediate preprocessing outputs are excluded.

## Derived Results

The deterministic, Monte Carlo, ablation, and clinical MRI-burden result tables
are under:

```text
results/
```

These include aggregate tables, figures, JSON summaries, patient-level derived
metrics, and MC draws used for analysis and paper generation.

## Public Release Caution

The source imaging collections are public or controlled-access, but the derived
graph tensors and patient-level result files are still patient-level derived
artifacts. Before making this repository public, decide whether those files
should remain in the release or be replaced by:

- aggregate paper tables and figures;
- scripts for rebuilding derived graphs after TCIA data access;
- clear TCIA data-access instructions.

The safest public release is code, aggregate paper artifacts, and rebuild
instructions, with patient-level derived files distributed only if allowed by
the relevant data-use terms and institutional review.
