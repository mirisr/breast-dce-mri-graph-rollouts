#!/usr/bin/env python3
"""Audit Breast-MRI-NACT-Pilot series for graph-transfer feasibility."""
from __future__ import annotations

import argparse
import json
import re
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
NBIA_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeries"
DEFAULT_COLLECTION = "Breast-MRI-NACT-Pilot"


def _safe_int(value: Any) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(float(value))
    except Exception:
        return 0


def _text(row: pd.Series, *cols: str) -> str:
    return " ".join(str(row.get(c, "")) for c in cols if pd.notna(row.get(c, ""))).lower()


def _fetch_series(collection: str, timeout: int = 90) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({"Collection": collection})
    req = urllib.request.Request(f"{NBIA_URL}?{query}", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", None)
        if isinstance(reason, ssl.SSLCertVerificationError):
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
                payload = resp.read().decode("utf-8")
        else:
            raise
    return json.loads(payload) if payload.strip() else []


def _add_visit_labels(df: pd.DataFrame) -> pd.DataFrame:
    studies = (
        df[["PatientID", "StudyInstanceUID", "StudyDate"]]
        .drop_duplicates()
        .sort_values(["PatientID", "StudyDate", "StudyInstanceUID"], kind="mergesort")
    )
    order: dict[tuple[str, str], int] = {}
    for _, group in studies.groupby("PatientID", sort=False):
        for i, row in enumerate(group.itertuples(index=False), start=1):
            order[(str(row.PatientID), str(row.StudyInstanceUID))] = i
    df = df.copy()
    df["visit"] = [
        f"V{order.get((str(r.PatientID), str(r.StudyInstanceUID)), 0)}"
        for r in df.itertuples(index=False)
    ]
    return df


def classify_series(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in [
        "Collection",
        "PatientID",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "StudyDate",
        "SeriesDate",
        "StudyDesc",
        "SeriesDescription",
        "ProtocolName",
        "Modality",
        "BodyPartExamined",
        "Manufacturer",
        "ManufacturerModelName",
        "ImageCount",
        "FileSize",
        "SeriesNumber",
    ]:
        if col not in df.columns:
            df[col] = ""
    df = _add_visit_labels(df)
    df["image_count"] = df["ImageCount"].map(_safe_int)
    df["file_size_bytes"] = df["FileSize"].map(_safe_int)
    text = df.apply(lambda r: _text(r, "SeriesDescription", "ProtocolName", "StudyDesc"), axis=1)
    desc = df["SeriesDescription"].fillna("").astype(str).str.lower()
    modality = df["Modality"].fillna("").astype(str).str.upper()

    is_mr = modality.eq("MR")
    is_seg = modality.eq("SEG")
    is_sr = modality.eq("SR")
    derived_map = desc.str.contains(r":\s*(?:ser|pe1)\b", regex=True)
    non_dynamic = desc.str.contains("locator|localizer|scout|diffusion|dwi|adc|t2|fse|pjn|summary", regex=True)

    df["is_original_dce"] = (
        is_mr
        & ~derived_map
        & ~non_dynamic
        & (
            desc.str.contains("dynamic|3dfgre|3d.*fgre|spgr|gradient echo", regex=True)
            | (df["image_count"] >= 120)
        )
    )
    df["is_ser_map"] = is_mr & desc.str.contains(r":\s*ser\b", regex=True)
    df["is_pe_map"] = is_mr & desc.str.contains(r":\s*pe1\b", regex=True)
    df["is_pe_seg"] = is_seg & desc.eq("pe segmentation thresh=70")
    df["is_tissue_seg"] = is_seg & desc.eq("breast tissue segmentation")
    df["is_voi_pe_seg"] = is_seg & desc.eq("voi pe segmentation thresh=70")
    df["is_voi_tissue_seg"] = is_seg & desc.eq("voi breast tissue segmentation")
    df["is_standard_report"] = is_sr | text.str.contains("standard breast imaging report", regex=False)
    df["is_graph_relevant"] = (
        df["is_original_dce"]
        | df["is_ser_map"]
        | df["is_pe_map"]
        | df["is_pe_seg"]
        | df["is_tissue_seg"]
        | df["is_voi_pe_seg"]
        | df["is_voi_tissue_seg"]
        | df["is_standard_report"]
    )
    return df


def _select_one(group: pd.DataFrame, flag: str) -> pd.Series | None:
    pool = group[group[flag].fillna(False)].copy()
    if pool.empty:
        return None
    pool = pool.sort_values(["image_count", "file_size_bytes", "SeriesNumber"], ascending=False)
    return pool.iloc[0]


def build_visit_inventory(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    roles = {
        "dce_original": "is_original_dce",
        "ser_map": "is_ser_map",
        "pe_map": "is_pe_map",
        "pe_seg": "is_pe_seg",
        "tissue_seg": "is_tissue_seg",
        "voi_pe_seg": "is_voi_pe_seg",
        "voi_tissue_seg": "is_voi_tissue_seg",
        "standard_report": "is_standard_report",
    }
    for (patient_id, visit), group in df.groupby(["PatientID", "visit"], sort=True):
        row: dict[str, Any] = {
            "patient_id": patient_id,
            "visit": visit,
            "study_uid": str(group["StudyInstanceUID"].iloc[0]),
            "study_date": str(group["StudyDate"].iloc[0]),
            "n_series": int(len(group)),
            "visit_file_size_gb": round(float(group["file_size_bytes"].sum()) / 1e9, 4),
        }
        for role, flag in roles.items():
            selected = _select_one(group, flag)
            row[f"has_{role}"] = bool(selected is not None)
            row[f"{role}_series_uid"] = "" if selected is None else str(selected["SeriesInstanceUID"])
            row[f"{role}_series_description"] = "" if selected is None else str(selected["SeriesDescription"])
            row[f"{role}_file_size_bytes"] = 0 if selected is None else int(selected["file_size_bytes"])
            row[f"{role}_image_count"] = 0 if selected is None else int(selected["image_count"])
        row["graph_derived_ready"] = bool(row["has_ser_map"] and row["has_pe_map"] and row["has_pe_seg"])
        row["exact_transfer_ready"] = bool(row["graph_derived_ready"] and row["has_dce_original"])
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["patient_id", "visit"]).reset_index(drop=True)


def build_patient_counts(visit_inventory: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for patient_id, group in visit_inventory.groupby("patient_id", sort=True):
        graph = group[group["graph_derived_ready"]].copy()
        exact = group[group["exact_transfer_ready"]].copy()
        rows.append(
            {
                "patient_id": patient_id,
                "n_visits": int(group["visit"].nunique()),
                "visits": ";".join(group["visit"].tolist()),
                "n_graph_derived_ready_visits": int(len(graph)),
                "graph_derived_ready_visits": ";".join(graph["visit"].tolist()),
                "n_exact_transfer_ready_visits": int(len(exact)),
                "exact_transfer_ready_visits": ";".join(exact["visit"].tolist()),
                "has_ge3_graph_ready": bool(len(graph) >= 3),
                "has_ge4_graph_ready": bool(len(graph) >= 4),
                "has_ge3_exact_ready": bool(len(exact) >= 3),
                "has_ge4_exact_ready": bool(len(exact) >= 4),
            }
        )
    return pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)


def write_role_selection(df: pd.DataFrame, visit_inventory: pd.DataFrame, out_dir: Path) -> None:
    role_rows: list[dict[str, Any]] = []
    role_flags = [
        ("dce_original", "is_original_dce"),
        ("ser_map", "is_ser_map"),
        ("pe_map", "is_pe_map"),
        ("pe_seg", "is_pe_seg"),
        ("tissue_seg", "is_tissue_seg"),
        ("voi_pe_seg", "is_voi_pe_seg"),
        ("voi_tissue_seg", "is_voi_tissue_seg"),
        ("standard_report", "is_standard_report"),
    ]
    selected = set()
    for row in visit_inventory.itertuples(index=False):
        for role, _flag in role_flags:
            uid = getattr(row, f"{role}_series_uid")
            if uid:
                selected.add((str(row.patient_id), str(row.visit), role, str(uid)))
    for patient_id, visit, role, uid in sorted(selected):
        src = df[df["SeriesInstanceUID"].astype(str).eq(uid)].iloc[0]
        role_rows.append(
            {
                "Collection": src["Collection"],
                "PatientID": patient_id,
                "Timepoint": visit,
                "role": role,
                "StudyInstanceUID": src["StudyInstanceUID"],
                "SeriesInstanceUID": uid,
                "Modality": src["Modality"],
                "SeriesDescription": src["SeriesDescription"],
                "SeriesNumber": src["SeriesNumber"],
                "ImageCount": src["ImageCount"],
                "FileSize": src["FileSize"],
            }
        )
    roles = pd.DataFrame(role_rows)
    roles.to_csv(out_dir / "selected_series_by_role.csv", index=False)
    roles[roles["role"].isin(["dce_original", "ser_map", "pe_map"])].to_csv(
        out_dir / "dce_derived_series_selection.csv",
        index=False,
    )
    roles[roles["role"].str.contains("seg|report", regex=True)].to_csv(
        out_dir / "roi_series_selection.csv",
        index=False,
    )


def write_summary(out_dir: Path, df: pd.DataFrame, visit_inventory: pd.DataFrame, patient_counts: pd.DataFrame) -> None:
    def table_text(frame: pd.DataFrame) -> str:
        try:
            return frame.to_markdown(index=False)
        except ImportError:
            return "```\n" + frame.to_string(index=False) + "\n```"

    lines: list[str] = []
    lines.append("# Breast-MRI-NACT-Pilot External Pilot Metadata Audit")
    lines.append("")
    lines.append(f"Run timestamp: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append("")
    lines.append("## Series-Level Summary")
    lines.append("")
    series_summary = pd.DataFrame(
        [
            {
                "patients": int(df["PatientID"].nunique()),
                "studies": int(df["StudyInstanceUID"].nunique()),
                "series": int(df["SeriesInstanceUID"].nunique()),
                "mr_series": int(df["Modality"].astype(str).str.upper().eq("MR").sum()),
                "seg_series": int(df["Modality"].astype(str).str.upper().eq("SEG").sum()),
                "sr_series": int(df["Modality"].astype(str).str.upper().eq("SR").sum()),
                "total_file_size_gb": round(float(df["file_size_bytes"].sum()) / 1e9, 3),
            }
        ]
    )
    lines.append(table_text(series_summary))
    lines.append("")
    lines.append("## Role Counts")
    lines.append("")
    role_counts = pd.DataFrame(
        [
            {"role": role, "series": int(df[flag].sum())}
            for role, flag in [
                ("original DCE", "is_original_dce"),
                ("SER map", "is_ser_map"),
                ("PE map", "is_pe_map"),
                ("PE segmentation", "is_pe_seg"),
                ("breast tissue segmentation", "is_tissue_seg"),
                ("VOI PE segmentation", "is_voi_pe_seg"),
                ("VOI breast tissue segmentation", "is_voi_tissue_seg"),
                ("standard report", "is_standard_report"),
            ]
        ]
    )
    lines.append(table_text(role_counts))
    lines.append("")
    lines.append("## Patient Visit Readiness")
    lines.append("")
    readiness = pd.DataFrame(
        [
            {
                "patients": int(patient_counts["patient_id"].nunique()),
                "patients_ge3_graph_ready": int(patient_counts["has_ge3_graph_ready"].sum()),
                "patients_ge4_graph_ready": int(patient_counts["has_ge4_graph_ready"].sum()),
                "patients_ge3_exact_ready": int(patient_counts["has_ge3_exact_ready"].sum()),
                "patients_ge4_exact_ready": int(patient_counts["has_ge4_exact_ready"].sum()),
                "graph_ready_visits": int(visit_inventory["graph_derived_ready"].sum()),
                "exact_ready_visits": int(visit_inventory["exact_transfer_ready"].sum()),
            }
        ]
    )
    lines.append(table_text(readiness))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- The graph-derived-ready gate requires SER, PE, and PE segmentation at a visit.")
    lines.append("- The exact-transfer-ready gate also requires an original DCE series for image-level QC.")
    lines.append("- Continue to smoke download only with patients passing these gates.")
    lines.append("")
    lines.append("## Files Written")
    lines.append("")
    for name in [
        "series_manifest.csv",
        "visit_inventory.csv",
        "patient_visit_counts.csv",
        "selected_series_by_role.csv",
        "dce_derived_series_selection.csv",
        "roi_series_selection.csv",
        "feasibility_summary.md",
    ]:
        lines.append(f"- `{name}`")
    lines.append("")
    (out_dir / "feasibility_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--out-dir", type=Path, default=REPO / "reports" / "breast_mri_nact_external" / "audit")
    parser.add_argument("--metadata-dir", type=Path, default=REPO / "datasets" / "breast_mri_nact_pilot" / "metadata")
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.metadata_dir.mkdir(parents=True, exist_ok=True)

    print(f"[nact-audit] querying {args.collection!r}")
    rows = _fetch_series(args.collection, timeout=args.timeout)
    if not rows:
        raise RuntimeError(f"No rows returned for {args.collection}")
    df = classify_series(pd.DataFrame(rows))
    visit_inventory = build_visit_inventory(df)
    patient_counts = build_patient_counts(visit_inventory)

    df.to_csv(args.out_dir / "series_manifest.csv", index=False)
    df.to_csv(args.metadata_dir / "breast_mri_nact_pilot_series_index.csv", index=False)
    visit_inventory.to_csv(args.out_dir / "visit_inventory.csv", index=False)
    patient_counts.to_csv(args.out_dir / "patient_visit_counts.csv", index=False)
    write_role_selection(df, visit_inventory, args.out_dir)
    write_summary(args.out_dir, df, visit_inventory, patient_counts)

    print(
        "[nact-audit] "
        f"patients={df['PatientID'].nunique()} series={len(df)} "
        f"ge3_graph={int(patient_counts['has_ge3_graph_ready'].sum())} "
        f"ge4_graph={int(patient_counts['has_ge4_graph_ready'].sum())} "
        f"ge4_exact={int(patient_counts['has_ge4_exact_ready'].sum())}"
    )
    print(f"[nact-audit] wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
