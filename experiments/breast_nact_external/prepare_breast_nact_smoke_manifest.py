#!/usr/bin/env python3
"""Prepare a small Breast-MRI-NACT-Pilot smoke-download manifest."""
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


def _select_patients(patient_counts: pd.DataFrame, n_three: int, n_four: int) -> list[str]:
    four = (
        patient_counts[patient_counts["has_ge4_graph_ready"]]
        .sort_values(["n_graph_derived_ready_visits", "patient_id"], ascending=[False, True])
        .head(int(n_four))["patient_id"]
        .astype(str)
        .tolist()
    )
    used = set(four)
    three = (
        patient_counts[patient_counts["has_ge3_graph_ready"] & ~patient_counts["patient_id"].astype(str).isin(used)]
        .sort_values(["n_graph_derived_ready_visits", "patient_id"], ascending=[False, True])
        .head(int(n_three))["patient_id"]
        .astype(str)
        .tolist()
    )
    return four + three


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=Path("reports/breast_mri_nact_external/audit"))
    parser.add_argument("--out", type=Path, default=Path("reports/breast_mri_nact_external/audit/smoke_manifest.csv"))
    parser.add_argument("--n-three", type=int, default=2)
    parser.add_argument("--n-four", type=int, default=2)
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
    selected_patients = _select_patients(patient_counts, args.n_three, args.n_four)
    if not selected_patients:
        raise RuntimeError("No graph-ready patients available for smoke manifest")

    roles = set(args.roles)
    manifest = selected_series[
        selected_series["PatientID"].astype(str).isin(selected_patients)
        & selected_series["role"].astype(str).isin(roles)
    ].copy()
    if manifest.empty:
        raise RuntimeError("Smoke manifest is empty")
    manifest["patient_smoke_order"] = manifest["PatientID"].astype(str).map(
        {pid: i for i, pid in enumerate(selected_patients)}
    )
    manifest["role_order"] = manifest["role"].map(ROLE_PRIORITY).fillna(99).astype(int)
    manifest = manifest.sort_values(["patient_smoke_order", "Timepoint", "role_order"]).drop(
        columns=["patient_smoke_order", "role_order"]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out, index=False)

    selected_path = args.out.with_name("smoke_patients.txt")
    selected_path.write_text("\n".join(selected_patients) + "\n", encoding="utf-8")

    total_gb = float(manifest["FileSize"].fillna(0).sum()) / 1e9
    print(
        f"[nact-smoke-manifest] patients={len(selected_patients)} "
        f"series={len(manifest)} gb={total_gb:.3f} out={args.out}"
    )
    print(f"[nact-smoke-manifest] patients={','.join(selected_patients)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
