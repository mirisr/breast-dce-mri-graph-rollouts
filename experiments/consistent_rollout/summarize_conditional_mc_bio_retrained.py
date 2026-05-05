#!/usr/bin/env python3
"""Compare retrained conditional Monte Carlo runs against the baseline run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_TAGS = [
    "bio_ftv020_alive005",
    "bio_ftv010_alive000",
    "bio_ftv010_alive002",
]

LABELS = {
    "v2_sched_samp_baseline": "Previous scheduled-sampling baseline",
    "bio_ftv020_alive005": "High FTV + high alive",
    "bio_ftv010_alive000": "FTV-only ablation",
    "bio_ftv010_alive002": "Medium FTV + medium alive",
}

METRIC_COLS = [
    "det_ftv_mae_ml_mean",
    "mc_mean_ftv_mae_ml_mean",
    "mc_median_ftv_mae_ml_mean",
    "coverage90_ftv_raw_mean",
    "coverage90_ftv_conformal_mean",
    "ftv_raw_width90_ml_mean",
    "ftv_conformal_width90_ml_mean",
    "crps_ftv_mean",
    "alive_det_abs_err_mean",
    "alive_count_abs_err_mc_mean_mean",
    "swd_mm_mc_mean_mean",
    "chamfer_mm_mc_mean_mean",
    "dice_mc_mean_mean",
]


def _safe_mean(vals: pd.Series) -> float:
    arr = pd.to_numeric(vals, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.mean(arr)) if arr.size else float("nan")


def _safe_se(vals: pd.Series) -> float:
    arr = pd.to_numeric(vals, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else float("nan")


def _read_run(run_dir: Path, tag: str) -> pd.DataFrame:
    per_path = run_dir / "conditional_mc_per_patient.parquet"
    if not per_path.is_file():
        raise FileNotFoundError(f"Missing per-patient MC output for {tag}: {per_path}")
    df = pd.read_parquet(per_path)
    if df.empty:
        raise ValueError(f"Empty per-patient MC output for {tag}: {per_path}")
    df = df.copy()
    df["model_tag"] = tag
    df["model_label"] = LABELS.get(tag, tag)
    df["det_ftv_bias_ml"] = pd.to_numeric(df["pred_ftv_det_ml"], errors="coerce") - pd.to_numeric(
        df["obs_ftv_ml"], errors="coerce"
    )
    df["det_ftv_mae_ml"] = df["det_ftv_bias_ml"].abs()
    df["mc_mean_ftv_bias_ml"] = pd.to_numeric(df["ftv_mc_mean_ml"], errors="coerce") - pd.to_numeric(
        df["obs_ftv_ml"], errors="coerce"
    )
    df["mc_mean_ftv_mae_ml"] = df["mc_mean_ftv_bias_ml"].abs()
    df["mc_median_ftv_bias_ml"] = pd.to_numeric(df["ftv_mc_median_ml"], errors="coerce") - pd.to_numeric(
        df["obs_ftv_ml"], errors="coerce"
    )
    df["mc_median_ftv_mae_ml"] = df["mc_median_ftv_bias_ml"].abs()
    df["alive_det_abs_err"] = (
        pd.to_numeric(df["alive_mass_det"], errors="coerce") - pd.to_numeric(df["alive_count_obs"], errors="coerce")
    ).abs()
    return df


def _conditioning_context(start_visit: str, predicted_visit: str) -> str:
    visits = ["T0", "T1", "T2", "T3"]
    start_i = visits.index(str(start_visit))
    return f"{','.join(visits[: start_i + 1])}->{predicted_visit}"


def _summarize_group(grp: pd.DataFrame, high_t3_threshold: float | None = None) -> dict[str, Any]:
    start_visit = str(grp["start_visit"].iloc[0])
    predicted_visit = str(grp["predicted_visit"].iloc[0])
    row: dict[str, Any] = {
        "model_tag": str(grp["model_tag"].iloc[0]),
        "model_label": str(grp["model_label"].iloc[0]),
        "bucket": f"{start_visit}->{predicted_visit}",
        "conditioning_context": _conditioning_context(start_visit, predicted_visit),
        "start_visit": start_visit,
        "predicted_visit": predicted_visit,
        "n_rows": int(len(grp)),
        "n_patients": int(grp["patient_id"].nunique()),
        "obs_ftv_ml_mean": _safe_mean(grp["obs_ftv_ml"]),
        "det_ftv_ml_mean": _safe_mean(grp["pred_ftv_det_ml"]),
        "det_ftv_bias_ml_mean": _safe_mean(grp["det_ftv_bias_ml"]),
        "det_ftv_bias_ml_se": _safe_se(grp["det_ftv_bias_ml"]),
        "det_ftv_mae_ml_mean": _safe_mean(grp["det_ftv_mae_ml"]),
        "det_ftv_mae_ml_se": _safe_se(grp["det_ftv_mae_ml"]),
        "mc_mean_ftv_ml_mean": _safe_mean(grp["ftv_mc_mean_ml"]),
        "mc_mean_ftv_bias_ml_mean": _safe_mean(grp["mc_mean_ftv_bias_ml"]),
        "mc_mean_ftv_bias_ml_se": _safe_se(grp["mc_mean_ftv_bias_ml"]),
        "mc_mean_ftv_mae_ml_mean": _safe_mean(grp["mc_mean_ftv_mae_ml"]),
        "mc_mean_ftv_mae_ml_se": _safe_se(grp["mc_mean_ftv_mae_ml"]),
        "mc_median_ftv_ml_mean": _safe_mean(grp["ftv_mc_median_ml"]),
        "mc_median_ftv_bias_ml_mean": _safe_mean(grp["mc_median_ftv_bias_ml"]),
        "mc_median_ftv_bias_ml_se": _safe_se(grp["mc_median_ftv_bias_ml"]),
        "mc_median_ftv_mae_ml_mean": _safe_mean(grp["mc_median_ftv_mae_ml"]),
        "mc_median_ftv_mae_ml_se": _safe_se(grp["mc_median_ftv_mae_ml"]),
        "coverage90_ftv_raw_mean": _safe_mean(grp["coverage90_ftv_raw"]),
        "coverage90_ftv_conformal_mean": _safe_mean(grp["coverage90_ftv_conformal"]),
        "ftv_raw_width90_ml_mean": _safe_mean(grp["ftv_raw_width90_ml"]),
        "ftv_conformal_width90_ml_mean": _safe_mean(grp["ftv_conformal_width90_ml"]),
        "crps_ftv_mean": _safe_mean(grp["crps_ftv"]),
        "near_zero_ftv_prob_mean": _safe_mean(grp["pcr_prob_mc"]),
        "alive_det_abs_err_mean": _safe_mean(grp["alive_det_abs_err"]),
        "alive_count_abs_err_mc_mean_mean": _safe_mean(grp["alive_count_abs_err_mc_mean"]),
        "swd_mm_mc_mean_mean": _safe_mean(grp["swd_mm_mc_mean"]),
        "chamfer_mm_mc_mean_mean": _safe_mean(grp["chamfer_mm_mc_mean"]),
        "dice_mc_mean_mean": _safe_mean(grp["dice_mc_mean"]),
    }
    if predicted_visit == "T3" and high_t3_threshold is not None:
        high = grp[pd.to_numeric(grp["obs_ftv_ml"], errors="coerce") >= high_t3_threshold]
        row.update(
            {
                "high_t3_obs_ftv_p90_threshold_ml": float(high_t3_threshold),
                "high_t3_n_patients": int(high["patient_id"].nunique()),
                "high_t3_obs_ftv_ml_mean": _safe_mean(high["obs_ftv_ml"]),
                "high_t3_det_ftv_mae_ml_mean": _safe_mean(high["det_ftv_mae_ml"]),
                "high_t3_mc_mean_ftv_mae_ml_mean": _safe_mean(high["mc_mean_ftv_mae_ml"]),
                "high_t3_mc_median_ftv_mae_ml_mean": _safe_mean(high["mc_median_ftv_mae_ml"]),
            }
        )
    else:
        row.update(
            {
                "high_t3_obs_ftv_p90_threshold_ml": float("nan"),
                "high_t3_n_patients": 0,
                "high_t3_obs_ftv_ml_mean": float("nan"),
                "high_t3_det_ftv_mae_ml_mean": float("nan"),
                "high_t3_mc_mean_ftv_mae_ml_mean": float("nan"),
                "high_t3_mc_median_ftv_mae_ml_mean": float("nan"),
            }
        )
    return row


def summarize_run(df: pd.DataFrame) -> pd.DataFrame:
    t3_obs = pd.to_numeric(df.loc[df["predicted_visit"] == "T3", "obs_ftv_ml"], errors="coerce").dropna()
    high_t3_threshold = float(t3_obs.quantile(0.90)) if len(t3_obs) else None
    rows = [
        _summarize_group(grp, high_t3_threshold=high_t3_threshold)
        for _, grp in df.groupby(["start_visit", "predicted_visit"], dropna=False, sort=True)
    ]
    return pd.DataFrame(rows)


def add_baseline_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = summary[summary["model_tag"] == "v2_sched_samp_baseline"].set_index("bucket")
    out = summary.copy()
    for col in METRIC_COLS:
        if col not in out.columns:
            continue
        vals = []
        pct_vals = []
        for row in out.itertuples(index=False):
            base_val = baseline.loc[getattr(row, "bucket"), col] if getattr(row, "bucket") in baseline.index else np.nan
            cur = getattr(row, col)
            vals.append(float(cur - base_val) if np.isfinite(cur) and np.isfinite(base_val) else float("nan"))
            if col.endswith("_mae_ml_mean") or col in {
                "ftv_raw_width90_ml_mean",
                "ftv_conformal_width90_ml_mean",
                "crps_ftv_mean",
                "alive_det_abs_err_mean",
                "alive_count_abs_err_mc_mean_mean",
                "swd_mm_mc_mean_mean",
                "chamfer_mm_mc_mean_mean",
            }:
                pct_vals.append(float((base_val - cur) / base_val * 100.0) if base_val else float("nan"))
            else:
                pct_vals.append(float("nan"))
        out[f"{col}_change_vs_baseline"] = vals
        if any(np.isfinite(pct_vals)):
            out[f"{col}_reduction_vs_baseline_pct"] = pct_vals
    return out


def summarize_subgroups(df: pd.DataFrame, subgroup_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in subgroup_cols:
        if col not in df.columns:
            continue
        for keys, grp in df.groupby(["model_tag", "start_visit", "predicted_visit", col], dropna=False, sort=True):
            model_tag, start_visit, predicted_visit, subgroup_value = keys
            base = _summarize_group(grp)
            rows.append(
                {
                    **base,
                    "subgroup_type": col,
                    "subgroup_value": str(subgroup_value),
                    "model_tag": model_tag,
                    "model_label": LABELS.get(str(model_tag), str(model_tag)),
                    "bucket": f"{start_visit}->{predicted_visit}",
                    "conditioning_context": _conditioning_context(str(start_visit), str(predicted_visit)),
                }
            )
    return pd.DataFrame(rows)


def decision_gates(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = summary[summary["model_tag"] == "v2_sched_samp_baseline"].set_index("bucket")
    rows: list[dict[str, Any]] = []
    t0t3 = summary[(summary["model_tag"] != "v2_sched_samp_baseline") & (summary["bucket"] == "T0->T3")]
    for row in t0t3.itertuples(index=False):
        model_tag = getattr(row, "model_tag")
        bucket = getattr(row, "bucket")
        base = baseline.loc[bucket]
        checks = [
            ("det_bias_centered", "det_ftv_bias_ml_mean", getattr(row, "det_ftv_bias_ml_mean"), -4.0, 1.0, True),
            (
                "mc_mean_mae_beats_old_baseline",
                "mc_mean_ftv_mae_ml_mean",
                getattr(row, "mc_mean_ftv_mae_ml_mean"),
                float("-inf"),
                float(base["mc_mean_ftv_mae_ml_mean"]),
                True,
            ),
            (
                "raw_coverage_improves",
                "coverage90_ftv_raw_mean",
                getattr(row, "coverage90_ftv_raw_mean"),
                float(base["coverage90_ftv_raw_mean"]),
                float("inf"),
                True,
            ),
            (
                "conformal_width_shrinks",
                "ftv_conformal_width90_ml_mean",
                getattr(row, "ftv_conformal_width90_ml_mean"),
                float("-inf"),
                float(base["ftv_conformal_width90_ml_mean"]),
                True,
            ),
            (
                "alive_mc_error_below_30",
                "alive_count_abs_err_mc_mean_mean",
                getattr(row, "alive_count_abs_err_mc_mean_mean"),
                float("-inf"),
                30.0,
                True,
            ),
            (
                "swd_not_materially_worse",
                "swd_mm_mc_mean_mean",
                getattr(row, "swd_mm_mc_mean_mean"),
                float("-inf"),
                float(base["swd_mm_mc_mean_mean"]) + 0.25,
                True,
            ),
            (
                "chamfer_not_materially_worse",
                "chamfer_mm_mc_mean_mean",
                getattr(row, "chamfer_mm_mc_mean_mean"),
                float("-inf"),
                float(base["chamfer_mm_mc_mean_mean"]) + 0.50,
                True,
            ),
            (
                "dice_not_materially_worse",
                "dice_mc_mean_mean",
                getattr(row, "dice_mc_mean_mean"),
                float(base["dice_mc_mean_mean"]) - 0.02,
                float("inf"),
                True,
            ),
        ]
        for gate, metric, value, lo, hi, inclusive in checks:
            if inclusive:
                passed = bool(value >= lo and value <= hi)
            else:
                passed = bool(value > lo and value < hi)
            rows.append(
                {
                    "model_tag": model_tag,
                    "model_label": LABELS.get(str(model_tag), str(model_tag)),
                    "bucket": bucket,
                    "gate": gate,
                    "metric": metric,
                    "value": value,
                    "lower_bound": lo,
                    "upper_bound": hi,
                    "pass": passed,
                }
            )
    return pd.DataFrame(rows)


def write_manifest(args: argparse.Namespace, loaded_tags: list[str], out_dir: Path) -> None:
    manifest = {
        "baseline_dir": str(args.baseline_dir),
        "mc_root": str(args.mc_root),
        "tags": loaded_tags,
        "outputs": {
            "summary": str(args.out),
            "subgroup_summary": str(args.subgroup_out),
            "decision_gates": str(args.gates_out),
        },
        "notes": [
            "pcr_prob_mc is treated as near_zero_ftv_prob_mean, not as calibrated pCR probability.",
            "High-volume tail metrics use the 90th percentile of observed T3 FTV within each run.",
            "Decision gates are applied to T0->T3 only.",
        ],
    }
    (out_dir / "mc_bio_comparison_manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-dir", type=Path, default=Path("reports/conditional_mc_consistent_rollout"))
    ap.add_argument("--mc-root", type=Path, default=Path("reports/conditional_mc_bio_retrained"))
    ap.add_argument("--tags", nargs="*", default=DEFAULT_TAGS)
    ap.add_argument("--out", type=Path, default=Path("reports/conditional_mc_bio_retrained/mc_bio_comparison_summary.csv"))
    ap.add_argument(
        "--subgroup-out",
        type=Path,
        default=Path("reports/conditional_mc_bio_retrained/mc_bio_subgroup_summary.csv"),
    )
    ap.add_argument(
        "--gates-out",
        type=Path,
        default=Path("reports/conditional_mc_bio_retrained/mc_bio_decision_gates.csv"),
    )
    args = ap.parse_args()

    run_frames = [_read_run(args.baseline_dir, "v2_sched_samp_baseline")]
    loaded_tags = ["v2_sched_samp_baseline"]
    for tag in args.tags:
        run_dir = args.mc_root / tag
        run_frames.append(_read_run(run_dir, tag))
        loaded_tags.append(tag)

    all_df = pd.concat(run_frames, ignore_index=True)
    summary = pd.concat([summarize_run(df) for _, df in all_df.groupby("model_tag", sort=False)], ignore_index=True)
    summary = add_baseline_deltas(summary)
    subgroups = summarize_subgroups(all_df, ["subtype", "pCR", "collection", "fold"])
    gates = decision_gates(summary)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(summary.to_csv(index=False))
    args.subgroup_out.write_text(subgroups.to_csv(index=False))
    args.gates_out.write_text(gates.to_csv(index=False))
    write_manifest(args, loaded_tags, args.out.parent)

    focus_cols = [
        "model_tag",
        "bucket",
        "det_ftv_bias_ml_mean",
        "det_ftv_mae_ml_mean",
        "mc_mean_ftv_bias_ml_mean",
        "mc_mean_ftv_mae_ml_mean",
        "coverage90_ftv_raw_mean",
        "coverage90_ftv_conformal_mean",
        "ftv_conformal_width90_ml_mean",
        "crps_ftv_mean",
        "alive_count_abs_err_mc_mean_mean",
    ]
    print(summary[summary["predicted_visit"] == "T3"][focus_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(args.out)
    print(args.subgroup_out)
    print(args.gates_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
