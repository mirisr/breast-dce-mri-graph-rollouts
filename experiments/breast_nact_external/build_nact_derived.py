#!/usr/bin/env python3
"""Build Breast-MRI-NACT-Pilot derived PE/SER/support bundles."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def dicom_files(series_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(series_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("._"):
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception:
            continue
        if str(getattr(ds, "SOPInstanceUID", "")):
            files.append(path)
    return files


def _orientation(ds: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    orient = np.asarray(getattr(ds, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0]), dtype=np.float32)
    row = orient[:3]
    col = orient[3:6]
    normal = np.cross(row, col)
    norm = np.linalg.norm(normal)
    if norm > 0:
        normal = normal / norm
    return row, col, normal.astype(np.float32)


def _slice_position(ds: Any, normal: np.ndarray) -> float:
    ipp = np.asarray(getattr(ds, "ImagePositionPatient", [0, 0, 0]), dtype=np.float32)
    return float(np.dot(ipp, normal))


def load_mr_series(series_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    records: list[tuple[float, int, Path, Any]] = []
    for path in dicom_files(series_dir):
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        _, _, normal = _orientation(ds)
        inst = int(getattr(ds, "InstanceNumber", len(records)))
        records.append((_slice_position(ds, normal), inst, path, ds))
    if not records:
        raise ValueError(f"No DICOM slices in {series_dir}")
    records.sort(key=lambda item: (item[0], item[1]))
    slices: list[np.ndarray] = []
    positions: list[float] = []
    for pos, _inst, path, _ds in records:
        ds = pydicom.dcmread(path, force=True)
        arr = ds.pixel_array.astype(np.float32)
        slope = safe_float(getattr(ds, "RescaleSlope", 1.0), 1.0)
        intercept = safe_float(getattr(ds, "RescaleIntercept", 0.0), 0.0)
        slices.append(arr * slope + intercept)
        positions.append(pos)
    volume = np.stack(slices, axis=0).astype(np.float32)
    ds0 = records[0][3]
    pix_spacing = [safe_float(v, 1.0) for v in getattr(ds0, "PixelSpacing", [1.0, 1.0])]
    if len(positions) > 1:
        z_spacing = float(np.median(np.abs(np.diff(sorted(positions)))))
    else:
        z_spacing = safe_float(getattr(ds0, "SliceThickness", 1.0), 1.0)
    meta = {
        "positions": positions,
        "voxel_spacing_mm": [z_spacing, pix_spacing[0], pix_spacing[1]],
        "origin_mm": [float(positions[0]), 0.0, 0.0],
        "series_uid": str(getattr(ds0, "SeriesInstanceUID", "")),
        "study_uid": str(getattr(ds0, "StudyInstanceUID", "")),
        "series_description": str(getattr(ds0, "SeriesDescription", "")),
        "image_orientation_patient": [float(x) for x in getattr(ds0, "ImageOrientationPatient", [])],
    }
    return volume, meta


def _frame_position(frame_group: Any, normal: np.ndarray) -> float | None:
    try:
        ipp = frame_group.PlanePositionSequence[0].ImagePositionPatient
        return float(np.dot(np.asarray(ipp, dtype=np.float32), normal))
    except Exception:
        return None


def load_seg_mask(series_dir: Path, target_shape: tuple[int, int, int], target_positions: list[float], normal: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    files = dicom_files(series_dir)
    if not files:
        raise ValueError(f"No SEG object in {series_dir}")
    ds = pydicom.dcmread(files[0], force=True)
    arr = ds.pixel_array
    if arr.ndim == 2:
        arr = arr[None, ...]
    arr = (arr > 0).astype(np.uint8)
    out = np.zeros(target_shape, dtype=np.uint8)
    frame_positions: list[float | None] = []
    per_frame = getattr(ds, "PerFrameFunctionalGroupsSequence", [])
    for i in range(arr.shape[0]):
        pos = _frame_position(per_frame[i], normal) if i < len(per_frame) else None
        frame_positions.append(pos)
        if pos is None:
            if arr.shape[0] == target_shape[0]:
                z = i
            else:
                z = min(i, target_shape[0] - 1)
        else:
            z = int(np.argmin(np.abs(np.asarray(target_positions, dtype=np.float32) - float(pos))))
        if arr.shape[1:] == target_shape[1:]:
            out[z] = np.maximum(out[z], arr[i])
    meta = {
        "series_uid": str(getattr(ds, "SeriesInstanceUID", "")),
        "series_description": str(getattr(ds, "SeriesDescription", "")),
        "frames": int(arr.shape[0]),
        "frame_positions_found": int(sum(p is not None for p in frame_positions)),
    }
    return out, meta


def _compact_relabel(labels: np.ndarray, min_voxels: int = 8) -> np.ndarray:
    out = np.zeros(labels.shape, dtype=np.int32)
    next_label = 1
    for label in sorted(int(x) for x in np.unique(labels) if int(x) > 0):
        sel = labels == label
        if int(sel.sum()) < int(min_voxels):
            continue
        out[sel] = next_label
        next_label += 1
    if next_label == 1 and int((labels > 0).sum()) > 0:
        out[labels > 0] = 1
    return out


def compact_grid_labels(mask: np.ndarray, block_size: int = 6, min_voxels: int = 8) -> np.ndarray:
    labels = np.zeros(mask.shape, dtype=np.int32)
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return labels
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    label = 1
    for z0 in range(int(mins[0]), int(maxs[0]), int(block_size)):
        for y0 in range(int(mins[1]), int(maxs[1]), int(block_size)):
            for x0 in range(int(mins[2]), int(maxs[2]), int(block_size)):
                sl = (
                    slice(z0, min(z0 + block_size, int(maxs[0]))),
                    slice(y0, min(y0 + block_size, int(maxs[1]))),
                    slice(x0, min(x0 + block_size, int(maxs[2]))),
                )
                sub = mask[sl] > 0
                if int(sub.sum()) < int(min_voxels):
                    continue
                block = labels[sl]
                block[sub] = label
                labels[sl] = block
                label += 1
    if label == 1 and coords.size:
        labels[mask > 0] = 1
    return _compact_relabel(labels, min_voxels=min_voxels)


def slic_labels(
    pe: np.ndarray,
    mask: np.ndarray,
    spacing: list[float],
    *,
    target_voxels: int = 64,
    compactness: float = 0.08,
    sigma: float = 0.0,
    min_voxels: int = 8,
) -> np.ndarray:
    try:
        from skimage.segmentation import slic
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("SLIC requested but scikit-image is unavailable") from exc

    support = mask > 0
    n_support = int(support.sum())
    if n_support <= 0:
        return np.zeros(mask.shape, dtype=np.int32)
    n_segments = max(1, int(round(n_support / max(1, int(target_voxels)))))
    vals = pe.astype(np.float32, copy=False)
    lo, hi = np.nanpercentile(vals[support], [1, 99]) if n_support else (0.0, 1.0)
    norm = np.clip(vals, lo, hi)
    norm = (norm - lo) / max(float(hi - lo), 1e-6)
    labels = slic(
        norm,
        n_segments=n_segments,
        compactness=float(compactness),
        sigma=float(sigma),
        spacing=tuple(float(x) for x in spacing),
        mask=support,
        start_label=1,
        channel_axis=None,
    )
    labels = np.where(support, labels, 0).astype(np.int32)
    return _compact_relabel(labels, min_voxels=min_voxels)


def supervoxel_table(labels: np.ndarray, pe: np.ndarray, ser: np.ndarray, spacing: list[float], origin: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    voxel_ml = float(np.prod(np.asarray(spacing, dtype=np.float32))) / 1000.0
    for label in sorted(int(x) for x in np.unique(labels) if int(x) > 0):
        idx = np.argwhere(labels == label)
        if idx.size == 0:
            continue
        vals_pe = pe[labels == label]
        vals_ser = ser[labels == label]
        centroid_vox = idx.mean(axis=0)
        centroid_mm = np.asarray(origin, dtype=np.float32) + centroid_vox * np.asarray(spacing, dtype=np.float32)
        rows.append(
            {
                "label": int(label),
                "voxel_count": int(idx.shape[0]),
                "volume_ml": float(idx.shape[0] * voxel_ml),
                "centroid_z_mm": float(centroid_mm[0]),
                "centroid_y_mm": float(centroid_mm[1]),
                "centroid_x_mm": float(centroid_mm[2]),
                "pe_mean": float(np.nanmean(vals_pe)) if vals_pe.size else 0.0,
                "pe_std": float(np.nanstd(vals_pe)) if vals_pe.size else 0.0,
                "ser_mean": float(np.nanmean(vals_ser)) if vals_ser.size else 0.0,
                "ser_std": float(np.nanstd(vals_ser)) if vals_ser.size else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _series_dir(raw_root: Path, row: pd.Series) -> Path:
    return raw_root / str(row["series_dir"])


def build_one_visit(
    raw_root: Path,
    rows: pd.DataFrame,
    out_dir: Path,
    *,
    supervoxel_method: str,
    block_size: int,
    target_supervoxel_voxels: int,
    slic_compactness: float,
    slic_sigma: float,
    min_supervoxel_voxels: int,
) -> dict[str, Any]:
    patient_id = str(rows["patient_id"].iloc[0])
    visit = str(rows["visit_dir"].iloc[0])
    pe_row = rows[rows["series_description"].str.endswith(": PE1")].iloc[0]
    ser_row = rows[rows["series_description"].str.endswith(": SER")].iloc[0]
    pe_seg_rows = rows[rows["series_description"].eq("PE Segmentation thresh=70")]
    voi_pe_rows = rows[rows["series_description"].eq("VOI PE Segmentation thresh=70")]
    if pe_seg_rows.empty and voi_pe_rows.empty:
        raise ValueError(f"{patient_id} {visit}: no PE segmentation")

    pe, pe_meta = load_mr_series(_series_dir(raw_root, pe_row))
    ser, ser_meta = load_mr_series(_series_dir(raw_root, ser_row))
    if pe.shape != ser.shape:
        raise ValueError(f"{patient_id} {visit}: PE/SER shape mismatch {pe.shape} vs {ser.shape}")

    orient = np.asarray(pe_meta.get("image_orientation_patient") or [1, 0, 0, 0, 1, 0], dtype=np.float32)
    row_cos, col_cos = orient[:3], orient[3:6]
    normal = np.cross(row_cos, col_cos)
    normal_norm = np.linalg.norm(normal)
    if normal_norm > 0:
        normal = normal / normal_norm
    seg_source = pe_seg_rows.iloc[0] if not pe_seg_rows.empty else voi_pe_rows.iloc[0]
    mask, seg_meta = load_seg_mask(
        _series_dir(raw_root, seg_source),
        target_shape=pe.shape,
        target_positions=list(pe_meta["positions"]),
        normal=normal,
    )
    spacing = [float(x) for x in pe_meta["voxel_spacing_mm"]]
    if supervoxel_method == "slic":
        labels = slic_labels(
            pe,
            mask,
            spacing,
            target_voxels=target_supervoxel_voxels,
            compactness=slic_compactness,
            sigma=slic_sigma,
            min_voxels=min_supervoxel_voxels,
        )
    else:
        labels = compact_grid_labels(mask, block_size=block_size, min_voxels=min_supervoxel_voxels)
    if int(labels.max()) == 0:
        raise ValueError(f"{patient_id} {visit}: empty support after SEG mapping")

    out_dir.mkdir(parents=True, exist_ok=True)
    origin = [float(x) for x in pe_meta["origin_mm"]]
    np.savez_compressed(
        out_dir / "pe_ser.npz",
        pe=pe.astype(np.float32),
        ser=ser.astype(np.float32),
        voxel_spacing=np.asarray(spacing, dtype=np.float32),
        origin=np.asarray(origin, dtype=np.float32),
    )
    np.savez_compressed(out_dir / "supervoxel_labels.npz", labels=labels)
    table = supervoxel_table(labels, pe, ser, spacing, origin)
    table.to_parquet(out_dir / "supervoxels.parquet", index=False)
    meta = {
        "patient_id": patient_id,
        "visit_id": visit,
        "source": "Breast-MRI-NACT-Pilot smoke derived bundle",
        "image_shape_zyx": list(pe.shape),
        "voxel_spacing_mm": spacing,
        "origin_mm": origin,
        "pe_series": pe_meta,
        "ser_series": ser_meta,
        "support_seg": seg_meta,
        "supervoxel_method": supervoxel_method,
        "target_supervoxel_voxels": int(target_supervoxel_voxels),
        "slic_compactness": float(slic_compactness),
        "slic_sigma": float(slic_sigma),
        "support_voxels": int(mask.sum()),
        "n_supervoxels": int(labels.max()),
        "ftv_like_ml": float(mask.sum() * np.prod(np.asarray(spacing, dtype=np.float32)) / 1000.0),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {
        "patient_id": patient_id,
        "visit": visit,
        "status": "ok",
        "shape": "x".join(str(x) for x in pe.shape),
        "support_voxels": int(mask.sum()),
        "n_supervoxels": int(labels.max()),
        "ftv_like_ml": meta["ftv_like_ml"],
        "pe_min": float(np.nanmin(pe)),
        "pe_p99": float(np.nanpercentile(pe, 99)),
        "ser_min": float(np.nanmin(ser)),
        "ser_p99": float(np.nanpercentile(ser, 99)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("datasets/breast_mri_nact_pilot/raw"))
    parser.add_argument(
        "--inspection-csv",
        type=Path,
        default=Path("reports/breast_mri_nact_external/smoke_download_inspection/series_inspection.csv"),
    )
    parser.add_argument("--out-root", type=Path, default=Path("datasets/breast_mri_nact_pilot/derived_v2"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/breast_mri_nact_external/derived_smoke"))
    parser.add_argument("--supervoxel-method", choices=("slic", "grid"), default="slic")
    parser.add_argument("--block-size", type=int, default=6)
    parser.add_argument("--target-supervoxel-voxels", type=int, default=64)
    parser.add_argument("--slic-compactness", type=float, default=0.08)
    parser.add_argument("--slic-sigma", type=float, default=0.0)
    parser.add_argument("--min-supervoxel-voxels", type=int, default=8)
    args = parser.parse_args()

    df = pd.read_csv(args.inspection_csv)
    df = df[df["modality"].isin(["MR", "SEG"])].copy()
    rows: list[dict[str, Any]] = []
    for (patient_id, visit), group in df.groupby(["patient_id", "visit_dir"], sort=True):
        out_dir = args.out_root / str(patient_id) / str(visit)
        try:
            rows.append(
                build_one_visit(
                    args.raw_root,
                    group,
                    out_dir,
                    supervoxel_method=args.supervoxel_method,
                    block_size=args.block_size,
                    target_supervoxel_voxels=args.target_supervoxel_voxels,
                    slic_compactness=args.slic_compactness,
                    slic_sigma=args.slic_sigma,
                    min_supervoxel_voxels=args.min_supervoxel_voxels,
                )
            )
        except Exception as exc:
            rows.append(
                {
                    "patient_id": patient_id,
                    "visit": visit,
                    "status": f"failed:{type(exc).__name__}:{exc}",
                }
            )
    qc = pd.DataFrame(rows)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    qc.to_csv(args.report_dir / "derived_qc.csv", index=False)
    ok = qc["status"].eq("ok").sum() if not qc.empty else 0
    patients_ok = qc.loc[qc["status"].eq("ok"), "patient_id"].nunique() if not qc.empty else 0
    print(f"[nact-derived] visits_ok={ok}/{len(qc)} patients_ok={patients_ok} out={args.out_root}")
    if ok != len(qc):
        print(qc[~qc["status"].eq("ok")].to_string(index=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
