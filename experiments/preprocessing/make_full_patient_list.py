#!/usr/bin/env python3
"""Generate the full patient list for consistent-graph scaling.

Selects all patients with visit_count == 4 from datasets/ispy2/cohort.parquet,
optionally filtered to ISPY2-only.  Writes two files:

  reports/all_patients_4visit.txt       -- all 760 (ISPY2 + ACRIN) with 4 visits
  reports/ispy2_patients_4visit.txt     -- 557 ISPY2-only with 4 visits

Also cross-checks against derived_v2/ to emit a 'ready' sub-list so the
registration job only touches patients whose preprocessed data already exists.

Usage:
    python experiments/preprocessing/make_full_patient_list.py
    python experiments/preprocessing/make_full_patient_list.py --derived datasets/ispy2/derived_v2
    python experiments/preprocessing/make_full_patient_list.py --ispy2-only
"""
import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).parent.parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort",  type=Path,
                    default=REPO / "datasets/ispy2/cohort.parquet")
    ap.add_argument("--derived", type=Path,
                    default=REPO / "datasets/ispy2/derived",
                    help="Path to derived root. Use derived_v2 for bio-enriched features "
                         "(prototype only); derived covers the full cohort. "
                         "Both have the same schema (pe_ser.npz, supervoxel_labels.npz, meta.json).")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "reports")
    ap.add_argument("--ispy2-only", action="store_true",
                    help="Only include ISPY2-* patients (exclude ACRIN-6698).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.cohort)
    df4 = df[df["visit_count"] == 4].copy()

    all_ids  = df4["patient_id"].tolist()
    ispy2_ids = df4[df4["collection"] == "ISPY2"]["patient_id"].tolist()

    def write_list(path: Path, ids: list[str]) -> None:
        path.write_text("\n".join(ids) + "\n")
        print(f"Wrote {len(ids):>4d} patients -> {path}")

    write_list(args.out_dir / "all_patients_4visit.txt",   all_ids)
    write_list(args.out_dir / "ispy2_patients_4visit.txt", ispy2_ids)

    # If derived_v2 exists, check which patients have ALL four visit dirs
    if args.derived.exists():
        visits = ("T0", "T1", "T2", "T3")
        ready = [
            pid for pid in all_ids
            if all((args.derived / pid / v / "pe_ser.npz").exists() for v in visits)
        ]
        ispy2_ready = [p for p in ready if p.startswith("ISPY2-")]
        write_list(args.out_dir / "all_patients_4visit_ready.txt",   ready)
        write_list(args.out_dir / "ispy2_patients_4visit_ready.txt", ispy2_ready)
        print(f"\n{len(ready)}/{len(all_ids)} patients have complete derived_v2 data")
        print(f"{len(ispy2_ready)}/{len(ispy2_ids)} are ISPY2")
    else:
        print(f"\nNote: derived_v2 not found at {args.derived}; skipping readiness check.")
        print("Run on the HPC where derived_v2 is available to get the *_ready.txt lists.")


if __name__ == "__main__":
    main()
