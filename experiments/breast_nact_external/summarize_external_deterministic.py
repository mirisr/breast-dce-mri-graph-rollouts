#!/usr/bin/env python3
"""Summarize deterministic Breast-MRI-NACT-Pilot external evaluation outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MODEL_LABELS = {
    "bio_ftv020_alive005": "Endpoint+Active",
    "hybrid_a50_bio_k8": "Hybrid-Edge k=8",
    "no_edges": "No-edge endpoint",
    "radial_bio_k8": "Radial-biologic k=8",
}


def load_model_rows(model_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for fold_dir in sorted(model_dir.glob("fold*")):
        path = fold_dir / "simulation_per_patient.parquet"
        if not path.is_file():
            continue
        df = pd.read_parquet(path)
        df["source_fold"] = int(fold_dir.name.replace("fold", ""))
        df["model"] = model_dir.name
        df["model_label"] = MODEL_LABELS.get(model_dir.name, model_dir.name)
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No fold simulation outputs under {model_dir}")
    return pd.concat(rows, ignore_index=True)


def _mean_abs(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.mean(np.abs(arr))) if arr.size else float("nan")


def _mean(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.mean(arr)) if arr.size else float("nan")


def summarize_endpoint(df: pd.DataFrame, conditioning: str, predicted_visit: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    endpoint = df[
        df["conditioning"].astype(str).eq(conditioning)
        & df["predicted_visit"].astype(str).eq(predicted_visit)
    ].copy()
    if endpoint.empty:
        raise RuntimeError(f"No rows for {conditioning}->{predicted_visit}")
    endpoint["ftv_err_ml"] = endpoint["pred_ftv_ml"] - endpoint["obs_ftv_ml"]
    endpoint["ftv_abs_err_ml_calc"] = endpoint["ftv_err_ml"].abs()

    row_summary: list[dict[str, Any]] = []
    patient_summary: list[pd.DataFrame] = []
    for (model, label), group in endpoint.groupby(["model", "model_label"], sort=False):
        ensemble = (
            group.groupby("patient_id", as_index=False)
            .agg(
                pred_ftv_ml=("pred_ftv_ml", "mean"),
                obs_ftv_ml=("obs_ftv_ml", "first"),
                pred_ftv_sd_across_folds=("pred_ftv_ml", "std"),
                n_folds=("source_fold", "nunique"),
            )
        )
        ensemble["ftv_err_ml"] = ensemble["pred_ftv_ml"] - ensemble["obs_ftv_ml"]
        ensemble["ftv_abs_err_ml"] = ensemble["ftv_err_ml"].abs()
        ensemble["model"] = model
        ensemble["model_label"] = label
        patient_summary.append(ensemble)

        row = {
            "model": model,
            "model_label": label,
            "conditioning": conditioning,
            "predicted_visit": predicted_visit,
            "n_patients": int(group["patient_id"].nunique()),
            "n_fold_rows": int(len(group)),
            "row_mae_ml": _mean_abs(group["ftv_err_ml"]),
            "row_bias_ml": _mean(group["ftv_err_ml"]),
            "ensemble_mae_ml": _mean(ensemble["ftv_abs_err_ml"]),
            "ensemble_bias_ml": _mean(ensemble["ftv_err_ml"]),
            "mean_obs_ftv_ml": _mean(ensemble["obs_ftv_ml"]),
            "mean_pred_ftv_ml": _mean(ensemble["pred_ftv_ml"]),
            "mean_fold_sd_pred_ftv_ml": _mean(ensemble["pred_ftv_sd_across_folds"]),
        }
        for col in ("swd_mm", "chamfer_mm", "dice", "alive_count_abs_err"):
            if col in group.columns:
                row[f"row_{col}_mean"] = _mean(group[col])
        row_summary.append(row)

    return pd.DataFrame(row_summary), pd.concat(patient_summary, ignore_index=True)


def write_markdown(path: Path, summary: pd.DataFrame, exclusions: pd.DataFrame | None) -> None:
    lines = [
        "# Breast-MRI-NACT-Pilot Four-Visit Deterministic External Summary",
        "",
        "Endpoint: source-trained T0-to-T3 deterministic rollout scored on external NACT four-visit patients.",
        "Fold rows score each source fold separately; ensemble rows average source-fold predictions per external patient.",
        "",
        "## Endpoint Summary",
        "",
        "```",
        summary.to_string(index=False),
        "```",
        "",
    ]
    if exclusions is not None and not exclusions.empty:
        lines.extend(["## Exclusions", "", "```", exclusions.to_string(index=False), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("reports/breast_mri_nact_external/deterministic_4visit"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/breast_mri_nact_external/tables"))
    parser.add_argument("--conditioning", default="rollout_from_T0")
    parser.add_argument("--predicted-visit", default="T3")
    parser.add_argument(
        "--exclusions",
        type=Path,
        default=Path("reports/breast_mri_nact_external/patients_4visit_exclusions.csv"),
    )
    parser.add_argument("--markdown-out", type=Path, default=None)
    args = parser.parse_args()

    frames = [load_model_rows(p) for p in sorted(args.root.iterdir()) if p.is_dir()]
    all_rows = pd.concat(frames, ignore_index=True)
    summary, patient_predictions = summarize_endpoint(all_rows, args.conditioning, args.predicted_visit)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "external_4visit_t0t3_deterministic_summary.csv"
    patient_path = args.out_dir / "external_4visit_t0t3_patient_predictions.csv"
    all_path = args.out_dir / "external_4visit_all_deterministic_rows.parquet"
    summary.to_csv(summary_path, index=False)
    patient_predictions.to_csv(patient_path, index=False)
    all_rows.to_parquet(all_path, index=False)
    exclusions = pd.read_csv(args.exclusions) if args.exclusions.is_file() else None
    markdown_out = args.markdown_out or (args.root.parent / "external_4visit_deterministic_summary.md")
    write_markdown(markdown_out, summary, exclusions)
    print(json.dumps({"summary": str(summary_path), "patients": str(patient_path), "all_rows": str(all_path)}, indent=2))
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
