#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--download-log", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    log = pd.read_csv(args.download_log) if args.download_log.exists() else pd.DataFrame()
    ok_uids = set(log.loc[log.get("status", "") == "ok", "SeriesInstanceUID"].astype(str)) if not log.empty else set()

    by_patient: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for _, row in manifest.iterrows():
        pid = str(row["PatientID"])
        tp = row.get("Timepoint")
        tp = tp if isinstance(tp, str) and tp else "Tunk"
        uid = str(row["SeriesInstanceUID"])
        desc = str(row["SeriesDescription"])
        if uid not in ok_uids:
            continue
        if "ADC:" in desc:
            by_patient[pid][tp].add("adc")
        elif "DWI MASK:" in desc:
            by_patient[pid][tp].add("mask")

    rows = []
    for pid, tps in sorted(by_patient.items()):
        for tp, have in sorted(tps.items()):
            rows.append({"patient_id": pid, "timepoint": tp, "has_adc": int("adc" in have), "has_mask": int("mask" in have)})
    cov = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        fh.write("# Phase 2b DWI Coverage Audit\n\n")
        fh.write(f"- Patients with any downloaded DWI artifact: {cov['patient_id'].nunique() if not cov.empty else 0}\n")
        if not cov.empty:
            fh.write(f"- Timepoint counts: {cov.groupby('timepoint')['patient_id'].nunique().to_dict()}\n")
            fh.write(f"- ADC available rows: {int(cov['has_adc'].sum())}\n")
            fh.write(f"- DWI mask available rows: {int(cov['has_mask'].sum())}\n")
            fh.write("\n## Sample rows\n\n")
            fh.write(cov.head(20).to_csv(index=False))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
