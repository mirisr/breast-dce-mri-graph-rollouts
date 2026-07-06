# I-SPY2 / ACRIN-6698 Data Notes

This directory is reserved for locally rebuilt I-SPY2 and ACRIN-6698 derived
artifacts. The public repository does not track the generated cohort files,
fold files, graph tensors, raw image volumes, or paper result outputs.

## Source Collections

- I-SPY2 / I-SPY 2 Breast DCE-MRI Trial:
  <https://doi.org/10.7937/TCIA.D8Z0-9T85>
- ACRIN-6698 breast MRI resources distributed through TCIA.

Obtain the source imaging data from the custodial archive and follow the
applicable data-use terms before running preprocessing.

## Expected Local Layout

```text
data/ispy2/
|-- raw/                       local DICOM mirror, ignored by git
|-- derived/                   local NIfTI/preprocessed outputs, ignored by git
|-- cohort.parquet             generated cohort table, ignored by git
|-- folds.parquet              generated held-out fold table, ignored by git
`-- graphs_consistent/         generated graph tensors, ignored by git
```

The training and evaluation scripts also support the historical compatibility
path:

```text
datasets/ispy2 -> ../data/ispy2
```

## Rebuild Notes

Use the scripts under `experiments/preprocessing/` and `scripts/` to rebuild
series indexes, graph tensors, and schema checks after data access. Generated
data products should remain local unless a separate distribution decision
explicitly allows releasing them.
