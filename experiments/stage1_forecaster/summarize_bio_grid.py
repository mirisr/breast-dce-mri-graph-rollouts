#!/usr/bin/env python3
"""Summarize biological retraining candidates after simulation evaluation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PRIMARY = [
    "swd_mm",
    "chamfer_mm",
    "hausdorff95_mm",
    "displacement_mae_mm",
    "dice",
    "surface_dice",
    "ftv_abs_err_ml",
    "alive_count_abs_err",
]


def _safe_mean(vals: pd.Series) -> float:
    arr = pd.to_numeric(vals, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.mean(arr)) if arr.size else float("nan")


def _safe_se(vals: pd.Series) -> float:
    arr = pd.to_numeric(vals, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else float("nan")


def summarize_eval(eval_dir: Path, tag: str) -> dict[str, Any] | None:
    per_path = eval_dir / tag / "simulation_per_patient.parquet"
    if not per_path.is_file():
        return None
    df = pd.read_parquet(per_path)
    if df.empty:
        return None
    t3 = df[(df["conditioning"] == "rollout_from_T0") & (df["predicted_visit"] == "T3")].copy()
    if t3.empty:
        return None
    t3["ftv_signed_err_ml"] = pd.to_numeric(t3["pred_ftv_ml"], errors="coerce") - pd.to_numeric(
        t3["obs_ftv_ml"], errors="coerce"
    )
    row: dict[str, Any] = {
        "tag": tag,
        "n_patients_t3": int(t3["patient_id"].nunique()),
        "pred_ftv_ml_mean": _safe_mean(t3["pred_ftv_ml"]),
        "obs_ftv_ml_mean": _safe_mean(t3["obs_ftv_ml"]),
        "ftv_signed_err_ml_mean": _safe_mean(t3["ftv_signed_err_ml"]),
        "ftv_signed_err_ml_se": _safe_se(t3["ftv_signed_err_ml"]),
    }
    for col in PRIMARY:
        if col in t3.columns:
            row[f"{col}_mean"] = _safe_mean(t3[col])
            row[f"{col}_se"] = _safe_se(t3[col])
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-root", type=Path, default=Path("reports/consistent_forecaster_v2_bio_eval"))
    ap.add_argument("--out", type=Path, default=Path("reports/consistent_forecaster_v2_bio_eval/bio_grid_summary"))
    ap.add_argument(
        "--tags",
        nargs="*",
        default=[
            "bio_ftv005_alive001",
            "bio_ftv010_alive002",
            "bio_ftv020_alive005",
            "bio_ftv010_alive000",
            "bio_ftv000_alive002",
        ],
    )
    args = ap.parse_args()

    rows = [summarize_eval(args.eval_root, tag) for tag in args.tags]
    rows = [r for r in rows if r is not None]
    if not rows:
        raise SystemExit(f"No complete simulation evaluations found under {args.eval_root}")

    df = pd.DataFrame(rows)
    sort_cols = [c for c in ["swd_mm_mean", "ftv_abs_err_ml_mean", "alive_count_abs_err_mean"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out.with_suffix(".csv"), index=False)
    args.out.with_suffix(".json").write_text(json.dumps(rows, indent=2))
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(args.out.with_suffix(".csv"))
    print(args.out.with_suffix(".json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
