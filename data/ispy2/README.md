# I-SPY 2 Dataset Layout

Target collection: **ISPY2 / I-SPY 2 Breast DCE-MRI Trial**
([TCIA DOI 10.7937/TCIA.D8Z0-9T85](https://doi.org/10.7937/TCIA.D8Z0-9T85)).
For full cohort-1 analyses we combine with **ACRIN-6698** (DWI data).

## Directory layout

```
datasets/ispy2/
├── README.md                # this file
├── manifests/               # TCIA .tcia manifest files (committed)
│   ├── ISPY2-719-patients.tcia
│   └── ISPY2-Cohort1-incl-ACRIN6698.tcia
├── metadata/                # clinical + series indexes (committed)
│   ├── ISPY2-Cohort-1-Clinical-Data.xlsx         # pCR, HR, HER2, Arm, ... (N=985)
│   ├── Multi-feature-MRI-NACT-Data.xlsx          # published radiomic baselines (N=384)
│   ├── ISPY2-719-patients-nbia-digest.xlsx       # TCIA-side per-series summary
│   ├── ISPY2-Cohort1-nbia-digest.xlsx
│   ├── ispy2_series_index.csv                    # NBIA REST -> series-level index
│   └── acrin6698_series_index.csv
├── docs/                    # data dictionaries + analysis-mask docs (committed)
├── scripts/                 # build / filter / download pipeline (committed)
│   ├── build_series_index.py
│   ├── filter_series.py
│   ├── pick_pilot_patients.py
│   └── download_ispy2.py
├── pilot/                   # pilot download (gitignored)
├── raw/                     # full DICOM mirror, run on Cradle (gitignored)
└── derived/                 # NIfTI / preprocessed outputs (gitignored)
```

## Pipeline

1. `scripts/build_series_index.py` — query the NBIA REST API (`tcia_utils`) for
   both ISPY2 and ACRIN-6698 collections, save one row per SeriesInstanceUID
   with `FileSize`, `SeriesDescription`, `StudyInstanceUID`, `PatientID`, etc.
   **Already run; output is checked in under `metadata/`.**
2. `scripts/filter_series.py` — apply a subset filter
   (default: cropped DCE + Analysis Mask, uni- and bi-lateral) and join with
   the digest XLSX to tag each series with its timepoint (T0/T1/T2/T3) and
   with clinical `pCR` / `HR` / `HER2` / `Arm`. Writes `metadata/filtered_manifest.csv`.
3. `scripts/pick_pilot_patients.py` — select a small pilot set (e.g. one
   `pCR=1` + one `pCR=0` patient with all four timepoints and a consistent
   scanner) from the filtered manifest.
4. `scripts/download_ispy2.py` — resumable series downloader, organizes DICOMs
   under `raw/<PatientID>/<Timepoint>/<SeriesDescription>/<SeriesInstanceUID>/`.

## Patient-ID mapping (important)

NBIA `PatientID` is a prefixed string (`ISPY2-349639`, `ACRIN-6698-373346`); the
clinical spreadsheet's `Patient_ID` is a bare integer (`349639`). Strip the
prefix before joining. This yields full coverage for 985 Cohort-1 patients
(719 ISPY2 + 266 ACRIN-6698). The remaining ~119 patients present in NBIA's
ACRIN-6698 listing are not enrolled in I-SPY 2 and have no clinical labels;
`filter_series.py` drops them by default.

## Scope sizing (as queried from NBIA, exact FileSize)

| Scope | Patients | Series | Size |
| --- | --- | --- | --- |
| Pilot (2 patients, full 4-timepoint coverage, minimal filter) | 2 | 16 | 2.32 GB |
| Minimal filter (cropped DCE + Analysis Mask, all clinical patients) | 985 | 7,354 | 499 GB |
| All VOLSER derivatives, Cohort 1 | ~1,100 | ~17,500 | ~750 GB |
| Full Cohort 1 (TCIA's recommendation) | ~1,100 | 51,158 | 2.59 TB |

Numbers are re-derived at the top of `scripts/filter_series.py`.

## Analysis Mask decoding (important)

The FTV `Analysis Mask` DICOM SEG objects are **inverse bit-encoded masks**
(per `docs/Analysis-mask-files-description.docx`). A voxel value of `0` means
the voxel was *included* in the measured FTV — that is, `tumor = (mask == 0)`.
Nonzero bits indicate why the voxel was excluded:

| value | meaning |
|------:|---------|
| 1     | PE threshold failed (percent-enhancement below site-specific threshold) |
| 2     | MNC (3D minimum-neighbor-count) connectivity filter failed |
| 8     | Background mask (pre-contrast background / fat) |
| 16    | (undocumented but observed — treat as informational) |
| 32    | Outside manual rectangular VOI |
| 64    | Inside manual OMIT region |

## Pilot download results (see `notebooks/ispy2/session1/`)

| PatientID     | pCR | HR | HER2 | Scanner             | T0 FTV | T1   | T2   | T3   |
|---------------|-----|----|------|---------------------|--------|------|------|------|
| ISPY2-825363  | 1   | -  | +    | SIEMENS Verio (3T)  | 7.1    | 2.9  | 0.0  | 0.07 |
| ISPY2-212307  | 0   | +  | -    | SIEMENS Symphony (1.5T) | 43.4 | 25.2 | 4.1  | 1.14 |

(FTV in mL, decoded from Analysis Mask.) Responder goes to effectively zero FTV
by T2-T3, non-responder retains measurable residual disease — consistent with
pCR labels.

Key heterogeneity already seen across just 2 patients:

- slice thickness: 1.0 vs 2.5 mm
- field strength: 1.5 vs 3.0 T
- in-plane pixel spacing: 0.73-0.94 mm (varies within a patient across timepoints)

Any featurization that pools across the full cohort will need explicit
resampling to a common voxel grid and intensity normalization that's robust to
scanner class.
