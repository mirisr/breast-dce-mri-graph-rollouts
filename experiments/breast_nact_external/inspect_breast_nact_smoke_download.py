#!/usr/bin/env python3
"""Inspect downloaded Breast-MRI-NACT-Pilot smoke series."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any
import re

import pandas as pd

try:
    import pydicom
except Exception:  # pragma: no cover
    pydicom = None


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def first_dicom(path: Path) -> Path | None:
    for p in sorted(path.rglob("*")):
        if not p.is_file() or p.name.startswith("._"):
            continue
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
        except Exception:
            continue
        if safe_str(getattr(ds, "SOPInstanceUID", "")):
            return p
    return None


def pixel_probe(path: Path) -> dict[str, Any]:
    if pydicom is None:
        return {"pixel_read_status": "pydicom_unavailable"}
    try:
        ds = pydicom.dcmread(path, force=True)
    except Exception as exc:
        return {"pixel_read_status": f"read_error:{type(exc).__name__}:{exc}"}
    has_pixel = "PixelData" in ds
    out: dict[str, Any] = {
        "has_pixel_data": bool(has_pixel),
        "rows": int(getattr(ds, "Rows", 0) or 0),
        "columns": int(getattr(ds, "Columns", 0) or 0),
        "number_of_frames": int(getattr(ds, "NumberOfFrames", 1) or 1),
    }
    if not has_pixel:
        out["pixel_read_status"] = "no_pixel_data"
        return out
    try:
        arr = ds.pixel_array
        out["pixel_read_status"] = "ok"
        out["pixel_shape"] = "x".join(str(x) for x in getattr(arr, "shape", ()))
        out["pixel_dtype"] = safe_str(getattr(arr, "dtype", ""))
    except Exception as exc:
        out["pixel_read_status"] = f"pixel_error:{type(exc).__name__}:{exc}"
    return out


def referenced_uids(ds: Any) -> str:
    found: set[str] = set()

    def visit(obj: Any, depth: int = 0) -> None:
        if depth > 8:
            return
        if not hasattr(obj, "__iter__"):
            return
        try:
            iterator = obj
        except Exception:
            return
        for elem in iterator:
            value = getattr(elem, "value", None)
            keyword = safe_str(getattr(elem, "keyword", ""))
            if keyword in {"ReferencedSeriesInstanceUID", "SeriesInstanceUID"} and value:
                found.add(str(value))
            if isinstance(value, (list, tuple)):
                for item in value:
                    visit(item, depth + 1)
            elif hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
                visit(value, depth + 1)

    visit(ds)
    return ";".join(sorted(found)[:30])


SERIES_UID_RE = re.compile(r"^\d+(?:\.\d+)+$")


def inspect_series_dir(series_dir: Path, raw_root: Path) -> dict[str, Any] | None:
    if pydicom is None:
        raise RuntimeError("pydicom is required for smoke inspection")
    dcm = first_dicom(series_dir)
    if dcm is None:
        return None
    try:
        ds = pydicom.dcmread(dcm, stop_before_pixels=True, force=True)
    except Exception as exc:
        return {
            "series_dir": str(series_dir.relative_to(raw_root)),
            "status": f"read_error:{type(exc).__name__}:{exc}",
        }
    parts = series_dir.relative_to(raw_root).parts
    if len(parts) < 5 or not SERIES_UID_RE.match(parts[-1]):
        return None
    row: dict[str, Any] = {
        "series_dir": str(series_dir.relative_to(raw_root)),
        "collection_dir": parts[0] if len(parts) > 0 else "",
        "patient_dir": parts[1] if len(parts) > 1 else "",
        "visit_dir": parts[2] if len(parts) > 2 else "",
        "series_description_dir": parts[3] if len(parts) > 3 else "",
        "dicom_files": sum(1 for p in series_dir.rglob("*") if p.is_file()),
        "patient_id": safe_str(getattr(ds, "PatientID", "")),
        "study_uid": safe_str(getattr(ds, "StudyInstanceUID", "")),
        "series_uid": safe_str(getattr(ds, "SeriesInstanceUID", "")),
        "sop_class_uid": safe_str(getattr(ds, "SOPClassUID", "")),
        "modality": safe_str(getattr(ds, "Modality", "")),
        "series_description": safe_str(getattr(ds, "SeriesDescription", "")),
        "protocol_name": safe_str(getattr(ds, "ProtocolName", "")),
        "study_date": safe_str(getattr(ds, "StudyDate", "")),
        "series_date": safe_str(getattr(ds, "SeriesDate", "")),
        "pixel_spacing": safe_str(getattr(ds, "PixelSpacing", "")),
        "slice_thickness": safe_str(getattr(ds, "SliceThickness", "")),
        "image_position_patient": safe_str(getattr(ds, "ImagePositionPatient", "")),
        "image_orientation_patient": safe_str(getattr(ds, "ImageOrientationPatient", "")),
        "referenced_series_uids": referenced_uids(ds),
        "status": "ok",
    }
    row.update(pixel_probe(dcm))
    return row


def write_summary(out_dir: Path, rows: pd.DataFrame) -> None:
    def table_text(frame: pd.DataFrame) -> str:
        try:
            return frame.to_markdown(index=False)
        except ImportError:
            return "```\n" + frame.to_string(index=False) + "\n```"

    lines: list[str] = ["# Breast-MRI-NACT-Pilot Smoke Download Inspection", ""]
    if rows.empty:
        lines.append("No readable DICOM series were found.")
        (out_dir / "smoke_download_inspection.md").write_text("\n".join(lines), encoding="utf-8")
        return

    summary = (
        rows.groupby(["modality", "pixel_read_status"], dropna=False)
        .agg(series=("series_uid", "nunique"), patients=("patient_id", "nunique"))
        .reset_index()
    )
    lines.append("## Pixel Readiness")
    lines.append("")
    lines.append(table_text(summary))
    lines.append("")

    visit_counts = (
        rows.groupby(["patient_id", "visit_dir", "modality"], dropna=False)
        .agg(series=("series_uid", "nunique"))
        .reset_index()
        .pivot_table(index=["patient_id", "visit_dir"], columns="modality", values="series", fill_value=0)
        .reset_index()
    )
    lines.append("## Patient/Visit Modalities")
    lines.append("")
    lines.append(table_text(visit_counts))
    lines.append("")

    role_counts = defaultdict(int)
    for desc, modality in zip(rows["series_description"].str.lower(), rows["modality"].str.upper()):
        if modality == "MR" and ": pe1" in desc:
            role_counts["pe_map"] += 1
        if modality == "MR" and ": ser" in desc:
            role_counts["ser_map"] += 1
        if modality == "SEG" and desc == "pe segmentation thresh=70":
            role_counts["pe_seg"] += 1
        if modality == "SEG" and desc == "breast tissue segmentation":
            role_counts["tissue_seg"] += 1
    lines.append("## Required Role Counts")
    lines.append("")
    lines.append(table_text(pd.DataFrame([dict(role_counts)])))
    lines.append("")
    (out_dir / "smoke_download_inspection.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("datasets/breast_mri_nact_pilot/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/breast_mri_nact_external/smoke_download_inspection"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    if not args.raw_root.exists():
        raise FileNotFoundError(args.raw_root)
    for series_dir in sorted(p for p in args.raw_root.rglob("*") if p.is_dir()):
        row = inspect_series_dir(series_dir, args.raw_root)
        if row is not None:
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "series_inspection.csv", index=False)
    if not df.empty:
        (args.out_dir / "series_inspection.json").write_text(
            json.dumps(df.to_dict(orient="records"), indent=2),
            encoding="utf-8",
        )
    write_summary(args.out_dir, df)
    print(
        f"[nact-smoke-inspect] series={len(df)} "
        f"patients={df['patient_id'].nunique() if not df.empty else 0} out={args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
