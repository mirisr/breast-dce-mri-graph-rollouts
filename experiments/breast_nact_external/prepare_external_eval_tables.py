#!/usr/bin/env python3
"""Prepare external folds/cohort tables for source-fold ensemble evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-list", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("datasets/breast_mri_nact_pilot/eval_tables_4visit"))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--collection", default="Breast-MRI-NACT-Pilot")
    args = parser.parse_args()

    patients = [ln.strip() for ln in args.patient_list.read_text().splitlines() if ln.strip()]
    if not patients:
        raise RuntimeError(f"No patients in {args.patient_list}")

    fold_rows = [
        {"patient_id": pid, "fold": fold}
        for fold in range(int(args.n_folds))
        for pid in patients
    ]
    cohort_rows = [
        {
            "patient_id": pid,
            "collection": args.collection,
            "pCR": pd.NA,
            "subtype": "NACT-Pilot",
        }
        for pid in patients
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_parquet(args.out_dir / "external_folds_ensemble.parquet", index=False)
    pd.DataFrame(cohort_rows).to_parquet(args.out_dir / "external_cohort.parquet", index=False)
    print(
        f"[nact-eval-tables] patients={len(patients)} folds={args.n_folds} out={args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
