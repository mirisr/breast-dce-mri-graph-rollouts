#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom


def _read_dicom_volume(series_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = sorted(series_dir.glob("*.dcm"))
    frames: list[tuple[float, np.ndarray]] = []
    spacing_yx = (1.0, 1.0)
    dz = 1.0
    for f in files:
        ds = pydicom.dcmread(f)
        px = ds.pixel_array.astype(np.float32)
        ipp = getattr(ds, "ImagePositionPatient", None)
        z = float(ipp[2]) if ipp is not None else float(getattr(ds, "SliceLocation", 0.0))
        frames.append((z, px))
        if getattr(ds, "PixelSpacing", None) is not None:
            spacing_yx = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]))
        if getattr(ds, "SliceThickness", None) is not None:
            dz = float(ds.SliceThickness)
    frames.sort(key=lambda x: x[0])
    if len(frames) > 1:
        dz = abs(frames[1][0] - frames[0][0]) or dz
    vol = np.stack([px for _, px in frames], axis=0)
    spacing = np.asarray([dz, spacing_yx[0], spacing_yx[1]], dtype=np.float32)
    origin = np.asarray([frames[0][0], 0.0, 0.0], dtype=np.float32)
    return vol, spacing, origin


def _resample_nearest(src: np.ndarray, src_spacing: np.ndarray, src_origin: np.ndarray,
                      shape: tuple[int, int, int], spacing: np.ndarray, origin: np.ndarray) -> np.ndarray:
    def idx(r0, rs, rn, s0, ss, sn):
        rc = r0 + np.arange(rn) * rs
        sc = s0 + np.arange(sn) * ss
        return np.abs(rc[:, None] - sc[None, :]).argmin(axis=1)
    iz = idx(origin[0], spacing[0], shape[0], src_origin[0], src_spacing[0], src.shape[0])
    iy = idx(origin[1], spacing[1], shape[1], src_origin[1], src_spacing[1], src.shape[1])
    ix = idx(origin[2], spacing[2], shape[2], src_origin[2], src_spacing[2], src.shape[2])
    return src[np.ix_(iz, iy, ix)].astype(np.float32)


def _stats(v: np.ndarray) -> tuple[float, float, float, float]:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    m = float(v.mean())
    s = float(v.std())
    if s < 1e-8:
        return m, 0.0, 0.0, 0.0
    z = (v - m) / s
    return m, s, float((z**3).mean()), float((z**4).mean() - 3.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--derived-root", type=Path, required=True)
    ap.add_argument("--raw-dwi-root", type=Path, required=True)
    args = ap.parse_args()

    for patient_dir in sorted(args.derived_root.glob("*")):
        if not patient_dir.is_dir():
            continue
        pid = patient_dir.name
        for tp_dir in sorted(patient_dir.glob("T*")):
            pq = tp_dir / "supervoxels.parquet"
            labels_npz = tp_dir / "supervoxel_labels.npz"
            meta_json = tp_dir / "meta.json"
            if not (pq.exists() and labels_npz.exists() and meta_json.exists()):
                continue
            dwi_tp_dir = args.raw_dwi_root / pid / tp_dir.name
            adc_candidates = sorted(dwi_tp_dir.glob("*ADC*/*"))
            if not adc_candidates:
                df = pd.read_parquet(pq)
                for c in ("mean_adc", "adc_std", "adc_skew", "adc_kurtosis"):
                    df[c] = 0.0
                df["adc_missing"] = 1.0
                df.to_parquet(pq, index=False)
                continue
            adc_dir = adc_candidates[0]
            adc, adc_sp, adc_org = _read_dicom_volume(adc_dir)
            meta = json.loads(meta_json.read_text())
            shape = tuple(meta["spatial_shape"])
            spacing = np.asarray(meta["voxel_spacing_mm"], dtype=np.float32)
            origin = np.asarray(meta["origin_mm"], dtype=np.float32)
            adc_ref = _resample_nearest(adc, adc_sp, adc_org, shape, spacing, origin)

            labels = np.load(labels_npz)["labels"].astype(np.int32)
            df = pd.read_parquet(pq)
            means = []
            stds = []
            skews = []
            kurts = []
            miss = []
            for sv in df["supervoxel_id"].to_numpy(dtype=np.int32):
                m, s, sk, ku = _stats(adc_ref[labels == sv])
                means.append(m); stds.append(s); skews.append(sk); kurts.append(ku); miss.append(0.0)
            df["mean_adc"] = means
            df["adc_std"] = stds
            df["adc_skew"] = skews
            df["adc_kurtosis"] = kurts
            df["adc_missing"] = miss
            df.to_parquet(pq, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
