#!/usr/bin/env python3
"""Select patients with complete successful derived visits from a QC table."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qc", type=Path, required=True)
    parser.add_argument("--visits", nargs="+", default=["V1", "V2", "V3", "V4"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclusions-out", type=Path, default=None)
    args = parser.parse_args()

    qc = pd.read_csv(args.qc)
    required = [str(v) for v in args.visits]
    ok: list[str] = []
    exclusions: list[dict[str, str]] = []
    for patient_id, group in qc.groupby("patient_id", sort=True):
        by_visit = {str(r.visit): str(r.status) for r in group.itertuples(index=False)}
        bad = [v for v in required if by_visit.get(v) != "ok"]
        if bad:
            exclusions.append(
                {
                    "patient_id": str(patient_id),
                    "excluded_visits": ";".join(bad),
                    "statuses": ";".join(f"{v}:{by_visit.get(v, 'missing')}" for v in bad),
                }
            )
        else:
            ok.append(str(patient_id))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(ok) + ("\n" if ok else ""), encoding="utf-8")
    exclusions_out = args.exclusions_out or args.out.with_suffix(".exclusions.csv")
    pd.DataFrame(exclusions).to_csv(exclusions_out, index=False)
    print(
        f"[nact-complete-patients] ok={len(ok)} excluded={len(exclusions)} "
        f"out={args.out} exclusions={exclusions_out}"
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
