#!/usr/bin/env python3
"""Prepare Breast-MRI-NACT-Pilot transfer manifests from the metadata audit."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROLE_PRIORITY = {
    "dce_original": 0,
    "pe_map": 1,
    "ser_map": 2,
    "pe_seg": 3,
    "tissue_seg": 4,
    "voi_pe_seg": 5,
    "voi_tissue_seg": 6,
    "standard_report": 7,
}


def select_patients(patient_counts: pd.DataFrame, track: str, limit: int) -> list[str]:
    if track == "four":
        mask = patient_counts["has_ge4_graph_ready"].fillna(False)
        sort_cols = ["n_graph_derived_ready_visits", "patient_id"]
        ascending = [False, True]
    elif track == "three":
        mask = patient_counts["has_ge3_graph_ready"].fillna(False)
        sort_cols = ["n_graph_derived_ready_visits", "patient_id"]
        ascending = [False, True]
    else:
        raise ValueError(f"Unknown track: {track}")
    selected = patient_counts[mask].sort_values(sort_cols, ascending=ascending)
    if int(limit) > 0:
        selected = selected.head(int(limit))
    return selected["patient_id"].astype(str).tolist()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=Path("reports/breast_mri_nact_external/audit"))
    parser.add_argument("--track", choices=("four", "three"), default="four")
    parser.add_argument("--limit", type=int, default=0, help="0 means all eligible patients.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--patient-list-out", type=Path, default=None)
    parser.add_argument(
        "--roles",
        nargs="+",
        default=[
            "dce_original",
            "pe_map",
            "ser_map",
            "pe_seg",
            "tissue_seg",
            "voi_pe_seg",
            "voi_tissue_seg",
            "standard_report",
        ],
    )
    args = parser.parse_args()

    patient_counts = pd.read_csv(args.audit_dir / "patient_visit_counts.csv")
    selected_series = pd.read_csv(args.audit_dir / "selected_series_by_role.csv")
    patients = select_patients(patient_counts, args.track, args.limit)
    if not patients:
        raise RuntimeError(f"No eligible patients for track={args.track}")

    out = args.out or args.audit_dir / f"{args.track}_visit_manifest.csv"
    patient_list_out = args.patient_list_out or args.audit_dir / f"patients_{args.track}_visit.txt"
    roles = set(args.roles)
    manifest = selected_series[
        selected_series["PatientID"].astype(str).isin(patients)
        & selected_series["role"].astype(str).isin(roles)
    ].copy()
    if args.track == "four":
        manifest = manifest[manifest["Timepoint"].astype(str).isin(["V1", "V2", "V3", "V4"])].copy()
    elif args.track == "three":
        manifest = manifest[manifest["Timepoint"].astype(str).isin(["V1", "V2", "V3"])].copy()
    if manifest.empty:
        raise RuntimeError("Transfer manifest is empty")

    manifest["patient_order"] = manifest["PatientID"].astype(str).map({pid: i for i, pid in enumerate(patients)})
    manifest["role_order"] = manifest["role"].map(ROLE_PRIORITY).fillna(99).astype(int)
    manifest = manifest.sort_values(["patient_order", "Timepoint", "role_order"]).drop(
        columns=["patient_order", "role_order"]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    patient_list_out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)
    patient_list_out.write_text("\n".join(patients) + "\n", encoding="utf-8")

    total_gb = float(manifest["FileSize"].fillna(0).sum()) / 1e9
    print(
        f"[nact-transfer-manifest] track={args.track} patients={len(patients)} "
        f"series={len(manifest)} gb={total_gb:.3f} out={out}"
    )
    print(f"[nact-transfer-manifest] patient_list={patient_list_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
