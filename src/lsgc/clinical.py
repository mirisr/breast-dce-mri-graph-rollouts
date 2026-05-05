from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_clinical_table(xlsx_path: Path) -> pd.DataFrame:
    """Load and normalize the I-SPY clinical sheet keyed by patient_id."""
    df = pd.read_excel(xlsx_path, sheet_name=0).copy()
    df["PatientIntID"] = (
        df["Patient_ID"].astype(str).str.extract(r"(\d+)$", expand=False).astype(float)
    )
    df["patient_id"] = df["PatientIntID"].map(lambda v: f"ISPY2-{int(v):06d}" if pd.notna(v) else None)
    out = pd.DataFrame({"patient_id": df["patient_id"]})

    age = pd.to_numeric(df.get("Age_at_Screening"), errors="coerce")
    age_mean = float(age.mean()) if age.notna().any() else 0.0
    age_std = float(age.std()) if age.notna().any() else 1.0
    out["age_z"] = ((age.fillna(age_mean) - age_mean) / max(age_std, 1e-6)).astype(np.float32)
    out["age_missing"] = age.isna().astype(np.float32)

    for src, dst in (("HR", "hr_pos"), ("HER2", "her2_pos"), ("MP", "mp_pos")):
        v = pd.to_numeric(df.get(src), errors="coerce")
        out[dst] = v.fillna(0).astype(np.float32)
        out[f"{dst}_missing"] = v.isna().astype(np.float32)

    arm = df.get("Arm")
    if arm is None:
        arm = pd.Series(["unknown"] * len(df))
    arm_oh = pd.get_dummies(arm.fillna("unknown"), prefix="arm")
    out = pd.concat([out, arm_oh.astype(np.float32)], axis=1)
    out = out.dropna(subset=["patient_id"]).drop_duplicates("patient_id")
    out["clinical_missing"] = np.float32(0.0)
    return out.set_index("patient_id").sort_index()


def clinical_vector_for_patient(clinical: pd.DataFrame, patient_id: str) -> torch.Tensor:
    """Return a deterministic-shape clinical vector.

    The last column of ``clinical`` is always ``clinical_missing`` (0 when the
    patient has clinical data, 1 when zero-filled). Callers can therefore rely
    on a single dimension regardless of cohort coverage.
    """
    if patient_id in clinical.index:
        return torch.tensor(clinical.loc[patient_id].to_numpy(dtype=np.float32))
    vec = torch.zeros(len(clinical.columns), dtype=torch.float32)
    vec[-1] = 1.0
    return vec


def compute_visit_context(payload: dict) -> torch.Tensor:
    """Compute (V, 6) per-visit context tensor from a graph payload."""
    x = payload["x"]
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    feature_names = payload.get("feature_names", [])
    idx = {name: i for i, name in enumerate(feature_names)}

    vol_i = idx.get("volume_ml")
    pe_i = idx.get("mean_pe")
    ser_i = idx.get("mean_ser")
    off = payload["visit_offsets"].tolist()
    visit_ids = payload.get("visit_ids", [f"T{i}" for i in range(len(off) - 1)])
    meta = payload.get("meta", {})
    rows: list[list[float]] = []
    for v, visit_id in enumerate(visit_ids):
        s, e = int(off[v]), int(off[v + 1])
        xv = x[s:e]
        if xv.numel() == 0:
            rows.append([0.0] * 6)
            continue
        total_volume = float(xv[:, vol_i].sum().item()) if vol_i is not None else 0.0
        mean_pe = float(xv[:, pe_i].mean().item()) if pe_i is not None else 0.0
        mean_ser = float(xv[:, ser_i].mean().item()) if ser_i is not None else 0.0
        curve = np.asarray(meta.get(visit_id, {}).get("kinetic_summary", []), dtype=np.float32)
        if curve.size > 1:
            p = int(np.argmax(curve[1:]) + 1)
            time_to_peak_norm = float(p / max(curve.size - 1, 1))
            washout_frac = float(curve[-1] - curve[p])
        else:
            time_to_peak_norm = 0.0
            washout_frac = 0.0
        n_log = float(np.log1p(max(e - s, 0)))
        rows.append([total_volume, mean_pe, mean_ser, time_to_peak_norm, washout_frac, n_log])
    return torch.tensor(rows, dtype=torch.float32)
