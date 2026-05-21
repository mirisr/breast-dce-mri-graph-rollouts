#!/usr/bin/env python3
"""Apply source-cohort endpoint residual MC to Breast-MRI-NACT-Pilot outputs.

The external deterministic evaluation scores each NACT patient with all five
source folds. This script averages those fold predictions per patient, then
uses source-cohort MC sample deviations and source conformal nonconformity
scores to form endpoint-only predictive intervals. No NACT residual enters
the MC or conformal calibration pool.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MODEL_CONFIGS = {
    "bio_ftv020_alive005": {
        "label": "Endpoint+Active",
        "source_mc_dir": "reports/conditional_mc_bio_retrained/bio_ftv020_alive005",
    },
    "no_edges": {
        "label": "No-edge endpoint",
        "source_mc_dir": "reports/edge_meaning_breast_mc/no_edges",
    },
    "radial_bio_k8": {
        "label": "Radial-biologic k=8",
        "source_mc_dir": "reports/edge_attr_meaning_breast_mc/radial_bio_k8",
    },
    "hybrid_a50_bio_k8": {
        "label": "Hybrid-Edge k=8",
        "source_mc_dir": "reports/edge_attr_meaning_breast_mc/hybrid_a50_bio_k8",
    },
}


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def conformal_correction(scores: np.ndarray, alpha: float) -> float:
    arr = np.asarray([x for x in scores if np.isfinite(x)], dtype=float)
    if arr.size == 0:
        return 0.0
    arr.sort()
    rank = int(math.ceil((arr.size + 1) * (1.0 - float(alpha))))
    rank = min(max(rank, 1), arr.size)
    return float(arr[rank - 1])


def crps_empirical(samples: np.ndarray, obs: float) -> float:
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0 or not math.isfinite(float(obs)):
        return float("nan")
    term1 = float(np.mean(np.abs(x - float(obs))))
    sx = np.sort(x)
    n = sx.size
    weights = (2 * np.arange(1, n + 1) - n - 1).astype(float)
    pair_abs_mean = float((2.0 / (n * n)) * np.sum(weights * sx))
    return term1 - 0.5 * pair_abs_mean


def load_source_deviation_pools(source_mc_dir: Path) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], float], dict[tuple[int, int], int]]:
    samples_path = source_mc_dir / "conditional_mc_samples.parquet"
    per_path = source_mc_dir / "conditional_mc_per_patient.parquet"
    if not samples_path.is_file():
        raise FileNotFoundError(samples_path)
    if not per_path.is_file():
        raise FileNotFoundError(per_path)

    samples = pd.read_parquet(samples_path)
    per = pd.read_parquet(per_path)
    key_cols = ["patient_id", "fold", "start_idx", "pred_idx"]
    keep_cols = key_cols + ["pred_ftv_det_ml"]
    merged = samples.merge(per[keep_cols], on=key_cols, how="inner", validate="many_to_one")
    merged["ftv_dev_ml"] = pd.to_numeric(merged["ftv_sample_ml"], errors="coerce") - pd.to_numeric(
        merged["pred_ftv_det_ml"], errors="coerce"
    )

    dev_pools: dict[tuple[int, int], np.ndarray] = {}
    q_by_horizon: dict[tuple[int, int], float] = {}
    n_cal_by_horizon: dict[tuple[int, int], int] = {}
    alpha = 0.10
    for (start_idx, pred_idx), grp in merged.groupby(["start_idx", "pred_idx"], dropna=False):
        vals = pd.to_numeric(grp["ftv_dev_ml"], errors="coerce").dropna().to_numpy(float)
        dev_pools[(int(start_idx), int(pred_idx))] = vals
    for (start_idx, pred_idx), grp in per.groupby(["start_idx", "pred_idx"], dropna=False):
        scores = pd.to_numeric(grp["ftv_raw_nonconformity_ml"], errors="coerce").dropna().to_numpy(float)
        q_by_horizon[(int(start_idx), int(pred_idx))] = conformal_correction(scores, alpha=alpha)
        n_cal_by_horizon[(int(start_idx), int(pred_idx))] = int(grp["patient_id"].nunique())
    return dev_pools, q_by_horizon, n_cal_by_horizon


def load_external_centers(external_det_root: Path, model: str) -> pd.DataFrame:
    path = external_det_root / model / "simulation_per_patient.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if df.empty:
        raise RuntimeError(f"No rows in {path}")
    if "source_fold" not in df.columns:
        # The raw deterministic output does not keep source fold explicitly,
        # but row duplicates still correspond to fold ensemble scoring.
        df = df.copy()
        df["source_fold"] = -1
    rows = []
    group_cols = ["patient_id", "conditioning", "start_visit", "predicted_visit", "rollout_depth"]
    for keys, grp in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["start_idx"] = int({"T0": 0, "T1": 1, "T2": 2}.get(str(row["start_visit"]), -1))
        row["pred_idx"] = int({"T1": 1, "T2": 2, "T3": 3}.get(str(row["predicted_visit"]), -1))
        row["pred_ftv_det_ml"] = _safe_float(pd.to_numeric(grp["pred_ftv_ml"], errors="coerce").mean())
        row["pred_ftv_fold_sd_ml"] = _safe_float(pd.to_numeric(grp["pred_ftv_ml"], errors="coerce").std(ddof=0))
        row["obs_ftv_ml"] = _safe_float(pd.to_numeric(grp["obs_ftv_ml"], errors="coerce").iloc[0])
        row["n_source_folds"] = int(grp["source_fold"].nunique()) if "source_fold" in grp.columns else int(len(grp))
        for col in ["swd_mm", "chamfer_mm", "dice", "alive_count_abs_err"]:
            if col in grp.columns:
                row[col] = _safe_float(pd.to_numeric(grp[col], errors="coerce").mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["conditioning", "predicted_visit", "patient_id"]).reset_index(drop=True)


def simulate_external(
    centers: pd.DataFrame,
    dev_pools: dict[tuple[int, int], np.ndarray],
    q_by_horizon: dict[tuple[int, int], float],
    n_cal_by_horizon: dict[tuple[int, int], int],
    *,
    model: str,
    label: str,
    n_mc: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(seed))
    sample_rows: list[dict[str, Any]] = []
    patient_rows: list[dict[str, Any]] = []
    for row in centers.itertuples(index=False):
        start_idx = int(getattr(row, "start_idx"))
        pred_idx = int(getattr(row, "pred_idx"))
        key = (start_idx, pred_idx)
        pool = dev_pools.get(key)
        if pool is None or pool.size == 0:
            continue
        center = float(getattr(row, "pred_ftv_det_ml"))
        obs = float(getattr(row, "obs_ftv_ml"))
        draw_dev = rng.choice(pool, size=int(n_mc), replace=True)
        samples = np.clip(center + draw_dev, 0.0, None)
        lo, hi = np.quantile(samples, [0.05, 0.95])
        q = float(q_by_horizon.get(key, 0.0))
        clo = max(0.0, float(lo) - q)
        chi = float(hi) + q
        for draw, value in enumerate(samples):
            sample_rows.append(
                {
                    "model": model,
                    "model_label": label,
                    "patient_id": getattr(row, "patient_id"),
                    "conditioning": getattr(row, "conditioning"),
                    "start_visit": getattr(row, "start_visit"),
                    "predicted_visit": getattr(row, "predicted_visit"),
                    "start_idx": start_idx,
                    "pred_idx": pred_idx,
                    "draw": int(draw),
                    "ftv_sample_ml": float(value),
                }
            )
        patient_rows.append(
            {
                "model": model,
                "model_label": label,
                "patient_id": getattr(row, "patient_id"),
                "conditioning": getattr(row, "conditioning"),
                "start_visit": getattr(row, "start_visit"),
                "predicted_visit": getattr(row, "predicted_visit"),
                "start_idx": start_idx,
                "pred_idx": pred_idx,
                "rollout_depth": int(getattr(row, "rollout_depth")),
                "n_source_folds": int(getattr(row, "n_source_folds")),
                "n_source_calibration_patients": int(n_cal_by_horizon.get(key, 0)),
                "n_mc": int(n_mc),
                "pred_ftv_det_ml": center,
                "pred_ftv_fold_sd_ml": float(getattr(row, "pred_ftv_fold_sd_ml")),
                "obs_ftv_ml": obs,
                "ftv_mc_mean_ml": float(np.mean(samples)),
                "ftv_mc_std_ml": float(np.std(samples, ddof=1)),
                "ftv_mc_median_ml": float(np.median(samples)),
                "ftv_raw_p05_ml": float(lo),
                "ftv_raw_p95_ml": float(hi),
                "ftv_raw_width90_ml": float(hi - lo),
                "coverage90_ftv_raw": int(float(lo) <= obs <= float(hi)),
                "ftv_conformal_q_ml": q,
                "ftv_conformal_p05_ml": clo,
                "ftv_conformal_p95_ml": chi,
                "ftv_conformal_width90_ml": float(chi - clo),
                "coverage90_ftv_conformal": int(clo <= obs <= chi),
                "crps_ftv": crps_empirical(samples, obs),
                "ftv_abs_err_ml_det": abs(center - obs),
                "ftv_abs_err_ml_mc_mean": abs(float(np.mean(samples)) - obs),
            }
        )
    return pd.DataFrame(patient_rows), pd.DataFrame(sample_rows)


def summarize(per: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = [
        "ftv_abs_err_ml_det",
        "ftv_abs_err_ml_mc_mean",
        "crps_ftv",
        "ftv_raw_width90_ml",
        "coverage90_ftv_raw",
        "ftv_conformal_width90_ml",
        "coverage90_ftv_conformal",
        "pred_ftv_det_ml",
        "obs_ftv_ml",
        "ftv_mc_mean_ml",
    ]
    for keys, grp in per.groupby(["model", "model_label", "conditioning", "predicted_visit"], dropna=False):
        row = dict(zip(["model", "model_label", "conditioning", "predicted_visit"], keys))
        row["n_patients"] = int(grp["patient_id"].nunique())
        for col in metric_cols:
            vals = pd.to_numeric(grp[col], errors="coerce").dropna().to_numpy(float)
            row[f"{col}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
        row["bias_det_ml_mean"] = float(np.mean(pd.to_numeric(grp["pred_ftv_det_ml"], errors="coerce") - pd.to_numeric(grp["obs_ftv_ml"], errors="coerce")))
        row["bias_mc_mean_ml_mean"] = float(np.mean(pd.to_numeric(grp["ftv_mc_mean_ml"], errors="coerce") - pd.to_numeric(grp["obs_ftv_ml"], errors="coerce")))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["conditioning", "predicted_visit", "model_label"]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-det-root", type=Path, default=Path("reports/breast_mri_nact_external/deterministic_4visit"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/breast_mri_nact_external/source_residual_mc_4visit"))
    parser.add_argument("--n-mc", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["bio_ftv020_alive005", "no_edges", "radial_bio_k8", "hybrid_a50_bio_k8"],
        choices=sorted(MODEL_CONFIGS),
        help="Model keys to evaluate from --external-det-root.",
    )
    args = parser.parse_args()

    if args.n_mc <= 0:
        raise ValueError("--n-mc must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_patient = []
    all_samples = []
    meta = {
        "n_mc": int(args.n_mc),
        "seed": int(args.seed),
        "external_det_root": str(args.external_det_root),
        "calibration_policy": "source MC deviations and source conformal scores only; external NACT residuals are used only for final scoring",
    }
    for i, model in enumerate(args.models):
        cfg = MODEL_CONFIGS[model]
        label = str(cfg["label"])
        source_mc_dir = Path(str(cfg["source_mc_dir"]))
        centers = load_external_centers(args.external_det_root, model)
        dev_pools, q_by_horizon, n_cal_by_horizon = load_source_deviation_pools(source_mc_dir)
        per, samples = simulate_external(
            centers,
            dev_pools,
            q_by_horizon,
            n_cal_by_horizon,
            model=model,
            label=label,
            n_mc=int(args.n_mc),
            seed=int(args.seed) + i * 1009,
        )
        all_patient.append(per)
        all_samples.append(samples)
        meta[model] = {
            "label": label,
            "source_mc_dir": str(source_mc_dir),
            "n_external_patients": int(centers["patient_id"].nunique()),
            "n_external_records": int(len(centers)),
        }

    per_df = pd.concat(all_patient, ignore_index=True)
    samples_df = pd.concat(all_samples, ignore_index=True)
    summary_df = summarize(per_df)
    per_path = args.out_dir / "external_source_residual_mc_per_patient.parquet"
    samples_path = args.out_dir / "external_source_residual_mc_samples.parquet"
    summary_path = args.out_dir / "external_source_residual_mc_summary.csv"
    meta_path = args.out_dir / "external_source_residual_mc_metadata.json"
    md_path = args.out_dir / "external_source_residual_mc_summary.md"
    per_df.to_parquet(per_path, index=False)
    samples_df.to_parquet(samples_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2))
    endpoint = summary_df[
        (summary_df["conditioning"] == "rollout_from_T0")
        & (summary_df["predicted_visit"] == "T3")
    ]
    md = [
        "# Breast-MRI-NACT-Pilot Source-Residual Endpoint MC",
        "",
        "Calibration policy: source-cohort MC sample deviations and source conformal scores only; NACT residuals are used only for final scoring.",
        "",
        "## T0-to-T3 Endpoint",
        "",
        "```",
        endpoint.to_string(index=False),
        "```",
        "",
        "## All Horizons",
        "",
        "```",
        summary_df.to_string(index=False),
        "```",
        "",
    ]
    md_path.write_text("\n".join(md))
    print(json.dumps({
        "per_patient": str(per_path),
        "samples": str(samples_path),
        "summary": str(summary_path),
        "markdown": str(md_path),
    }, indent=2))
    print(endpoint.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
