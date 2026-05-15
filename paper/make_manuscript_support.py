from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parent
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"

BASELINE_TAG = "v2_sched_samp_baseline"
RETAINED_TAG = "bio_ftv020_alive005"
LATEST_RETAINED_TAG = "hybrid_a50_bio_k8"
FINAL_RETAINED_TAG = LATEST_RETAINED_TAG
ABLATION_TAGS = [
    BASELINE_TAG,
    RETAINED_TAG,
    "bio_ftv010_alive000",
    "bio_ftv010_alive002",
]


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def sync_report_figures() -> None:
    source = REPO / "reports/figures/mc_bio_retraining"
    for name in [
        "bio_retrained_model_design.png",
        "mc_diagnostic_bias_coverage_width.png",
    ]:
        src = source / name
        if src.exists():
            shutil.copy2(src, FIGURES / name)


def mc_report_root(tag: str) -> Path:
    if tag == BASELINE_TAG:
        return REPO / "reports/conditional_mc_consistent_rollout"
    if tag == LATEST_RETAINED_TAG:
        return REPO / f"reports/edge_attr_meaning_breast_mc/{tag}"
    return REPO / f"reports/conditional_mc_bio_retrained/{tag}"


def ci_mean(values: np.ndarray, n_boot: int = 5000, seed: int = 42) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(values.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return mean, float(lo), float(hi)


def binary_auc(y: pd.Series | np.ndarray, score: pd.Series | np.ndarray) -> float:
    tmp = pd.DataFrame({"y": y, "score": score}).dropna()
    if tmp.empty or tmp["y"].nunique() != 2:
        return np.nan
    ranks = tmp["score"].rank(method="average").to_numpy(float)
    yv = tmp["y"].to_numpy(int)
    n_pos = int(yv.sum())
    n_neg = int((1 - yv).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    rank_sum_pos = float(ranks[yv == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def load_mc_per_patient(tag: str) -> pd.DataFrame:
    path = mc_report_root(tag) / "conditional_mc_per_patient.parquet"
    df = pd.read_parquet(path)
    df = df.copy()
    df["bucket"] = df["start_visit"].astype(str) + "->" + df["predicted_visit"].astype(str)
    df["det_ftv_abs_err_ml"] = (df["pred_ftv_det_ml"] - df["obs_ftv_ml"]).abs()
    df["det_ftv_signed_err_ml"] = df["pred_ftv_det_ml"] - df["obs_ftv_ml"]
    df["mc_mean_ftv_abs_err_ml"] = (df["ftv_mc_mean_ml"] - df["obs_ftv_ml"]).abs()
    df["mc_mean_ftv_signed_err_ml"] = df["ftv_mc_mean_ml"] - df["obs_ftv_ml"]
    df["mc_median_ftv_abs_err_ml"] = (df["ftv_mc_median_ml"] - df["obs_ftv_ml"]).abs()
    df["det_alive_abs_err"] = (df["alive_mass_det"] - df["alive_count_obs"]).abs()
    return df


def load_mc_samples(tag: str) -> pd.DataFrame:
    path = mc_report_root(tag) / "conditional_mc_samples.parquet"
    df = pd.read_parquet(path)
    df = df.copy()
    df["bucket"] = df["start_visit"].astype(str) + "->" + df["predicted_visit"].astype(str)
    return df


def paired_delta_table() -> pd.DataFrame:
    base = load_mc_per_patient(BASELINE_TAG)
    retained = load_mc_per_patient(FINAL_RETAINED_TAG)
    rows = []
    metrics = {
        "det_ftv_mae_ml": ("det_ftv_abs_err_ml", "decrease"),
        "mc_mean_ftv_mae_ml": ("mc_mean_ftv_abs_err_ml", "decrease"),
        "mc_median_ftv_mae_ml": ("mc_median_ftv_abs_err_ml", "decrease"),
        "raw_90_coverage": ("coverage90_ftv_raw", "increase"),
        "raw_90_width_ml": ("ftv_raw_width90_ml", "decrease"),
        "conformal_90_width_ml": ("ftv_conformal_width90_ml", "decrease"),
        "crps": ("crps_ftv", "decrease"),
        "mc_alive_abs_err": ("alive_count_abs_err_mc_mean", "decrease"),
        "swd_mm": ("swd_mm_mc_mean", "decrease"),
        "chamfer_mm": ("chamfer_mm_mc_mean", "decrease"),
        "dice": ("dice_mc_mean", "increase"),
    }
    for bucket in ["T0->T3", "T1->T3", "T2->T3"]:
        b = base.loc[base["bucket"] == bucket]
        r = retained.loc[retained["bucket"] == bucket]
        merged = b.merge(
            r,
            on=["patient_id", "start_visit", "predicted_visit", "bucket"],
            suffixes=("_baseline", "_retained"),
        )
        for label, (col, direction) in metrics.items():
            bvals = pd.to_numeric(merged[f"{col}_baseline"], errors="coerce").to_numpy(float)
            rvals = pd.to_numeric(merged[f"{col}_retained"], errors="coerce").to_numpy(float)
            if direction == "decrease":
                delta = bvals - rvals
            else:
                delta = rvals - bvals
            mean, lo, hi = ci_mean(delta, seed=100 + len(rows))
            rows.append(
                {
                    "bucket": bucket,
                    "metric": label,
                    "direction": direction,
                    "n": int(np.isfinite(delta).sum()),
                    "baseline_mean": float(np.nanmean(bvals)),
                    "retained_mean": float(np.nanmean(rvals)),
                    "improvement_mean": mean,
                    "improvement_ci95_low": lo,
                    "improvement_ci95_high": hi,
                }
            )
    return pd.DataFrame(rows)


def model_mean_ci_table() -> pd.DataFrame:
    rows = []
    specs = [
        (BASELINE_TAG, "Baseline"),
        (FINAL_RETAINED_TAG, "Retained"),
    ]
    metrics = {
        "det_ftv_bias_ml": "det_ftv_signed_err_ml",
        "det_ftv_mae_ml": "det_ftv_abs_err_ml",
        "mc_mean_ftv_bias_ml": "mc_mean_ftv_signed_err_ml",
        "mc_mean_ftv_mae_ml": "mc_mean_ftv_abs_err_ml",
        "raw_90_coverage": "coverage90_ftv_raw",
        "raw_90_width_ml": "ftv_raw_width90_ml",
        "conformal_90_width_ml": "ftv_conformal_width90_ml",
        "crps": "crps_ftv",
        "mc_alive_abs_err": "alive_count_abs_err_mc_mean",
    }
    for tag, label in specs:
        df = load_mc_per_patient(tag)
        for bucket in ["T0->T3", "T1->T3", "T2->T3"]:
            grp = df.loc[df["bucket"] == bucket]
            for metric, col in metrics.items():
                mean, lo, hi = ci_mean(pd.to_numeric(grp[col], errors="coerce").to_numpy(float), seed=200 + len(rows))
                rows.append(
                    {
                        "model_tag": tag,
                        "model_label": label,
                        "bucket": bucket,
                        "metric": metric,
                        "n": int(grp["patient_id"].nunique()),
                        "mean": mean,
                        "ci95_low": lo,
                        "ci95_high": hi,
                    }
                )
    return pd.DataFrame(rows)


def ablation_t0t3_table() -> pd.DataFrame:
    path = REPO / "reports/consistent_forecaster_v2_bio_eval/notebook_exports/derived_notebook_tables/mc_t0t3_model_comparison.csv"
    df = pd.read_csv(path)
    df = df.loc[df["model_tag"].isin(ABLATION_TAGS)].copy()
    label_map = {
        BASELINE_TAG: "Baseline",
        RETAINED_TAG: "FTV .020 + alive .005",
        "bio_ftv010_alive000": "FTV only",
        "bio_ftv010_alive002": "FTV .010 + alive .002",
    }
    df["paper_label"] = df["model_tag"].map(label_map)
    cols = [
        "model_tag",
        "paper_label",
        "det_ftv_bias_ml_mean",
        "det_ftv_mae_ml_mean",
        "mc_mean_ftv_mae_ml_mean",
        "coverage90_ftv_raw_mean",
        "ftv_conformal_width90_ml_mean",
        "crps_ftv_mean",
        "alive_count_abs_err_mc_mean_mean",
        "swd_mm_mc_mean_mean",
        "high_t3_mc_mean_ftv_mae_ml_mean",
    ]
    return df[cols]


def subtype_t0t3_table() -> pd.DataFrame:
    base = load_mc_per_patient(BASELINE_TAG).query("bucket == 'T0->T3'")
    retained = load_mc_per_patient(FINAL_RETAINED_TAG).query("bucket == 'T0->T3'")
    rows = []
    for subtype in sorted(retained["subtype"].dropna().unique()):
        b = base.loc[base["subtype"] == subtype]
        r = retained.loc[retained["subtype"] == subtype]
        m = b.merge(r, on=["patient_id"], suffixes=("_baseline", "_retained"))
        delta = m["mc_mean_ftv_abs_err_ml_baseline"].to_numpy(float) - m["mc_mean_ftv_abs_err_ml_retained"].to_numpy(float)
        mean, lo, hi = ci_mean(delta, seed=300 + len(rows))
        rows.append(
            {
                "subtype": subtype,
                "n": int(m.shape[0]),
                "baseline_mc_mean_ftv_mae_ml": float(m["mc_mean_ftv_abs_err_ml_baseline"].mean()),
                "retained_mc_mean_ftv_mae_ml": float(m["mc_mean_ftv_abs_err_ml_retained"].mean()),
                "mae_reduction_ml": mean,
                "mae_reduction_ci95_low": lo,
                "mae_reduction_ci95_high": hi,
                "retained_raw_coverage": float(m["coverage90_ftv_raw_retained"].mean()),
            }
        )
    return pd.DataFrame(rows)


def calibration_subgroup_robustness_table() -> pd.DataFrame:
    buckets = ["T0->T3", "T1->T3", "T2->T3"]
    bucket_df = bucket_calibration_summary()
    delta_df = paired_delta_table()
    rows = []
    for bucket in buckets:
        retained = bucket_df.loc[(bucket_df["bucket"] == bucket) & (bucket_df["paper_label"] == "Retained")].iloc[0]
        delta = delta_df.loc[(delta_df["bucket"] == bucket) & (delta_df["metric"] == "mc_mean_ftv_mae_ml")].iloc[0]
        rows.append(
            {
                "check": "horizon",
                "stratum": bucket,
                "n": int(retained["n_patients"]),
                "retained_mc_mean_ftv_mae_ml": float(retained["mc_mean_ftv_mae_ml_mean"]),
                "mae_reduction_ml": float(delta["improvement_mean"]),
                "mae_reduction_ci95_low": float(delta["improvement_ci95_low"]),
                "mae_reduction_ci95_high": float(delta["improvement_ci95_high"]),
                "retained_raw_coverage": float(retained["coverage90_ftv_raw_mean"]),
                "retained_conformal_coverage": float(retained["coverage90_ftv_conformal_mean"]),
                "retained_conformal_width_ml": float(retained["ftv_conformal_width90_ml_mean"]),
                "retained_crps": float(retained["crps_ftv_mean"]),
                "note": "Final-visit conditioning horizon from the full-cohort MC rollout.",
            }
        )

    for row in subtype_t0t3_table().itertuples(index=False):
        rows.append(
            {
                "check": "subtype_post_hoc",
                "stratum": row.subtype,
                "n": int(row.n),
                "retained_mc_mean_ftv_mae_ml": float(row.retained_mc_mean_ftv_mae_ml),
                "mae_reduction_ml": float(row.mae_reduction_ml),
                "mae_reduction_ci95_low": float(row.mae_reduction_ci95_low),
                "mae_reduction_ci95_high": float(row.mae_reduction_ci95_high),
                "retained_raw_coverage": float(row.retained_raw_coverage),
                "retained_conformal_coverage": np.nan,
                "retained_conformal_width_ml": np.nan,
                "retained_crps": np.nan,
                "note": "Post hoc T0->T3 subtype stratum; residual and conformal calibration were not refit by subgroup.",
            }
        )
    return pd.DataFrame(rows)


def retained_full_rollout_subtype_calibration_table() -> pd.DataFrame:
    df = load_mc_per_patient(FINAL_RETAINED_TAG).copy()
    out = (
        df.groupby(["bucket", "subtype"], observed=True)
        .agg(
            n=("patient_id", "nunique"),
            det_ftv_mae_ml=("det_ftv_abs_err_ml", "mean"),
            mc_mean_ftv_mae_ml=("mc_mean_ftv_abs_err_ml", "mean"),
            raw_90_coverage=("coverage90_ftv_raw", "mean"),
            conformal_90_coverage=("coverage90_ftv_conformal", "mean"),
            raw_width90_ml=("ftv_raw_width90_ml", "mean"),
            conformal_width90_ml=("ftv_conformal_width90_ml", "mean"),
            crps=("crps_ftv", "mean"),
        )
        .reset_index()
        .sort_values(["bucket", "subtype"])
    )
    out["note"] = "Post hoc subtype stratum from the latest retained full-cohort MC rollout; no subgroup-specific residual pool was fit."
    return out


def calibration_by_subtype_full_table() -> pd.DataFrame:
    df = retained_full_rollout_subtype_calibration_table()
    return df.loc[df["bucket"].isin(["T0->T3", "T1->T3", "T2->T3"])].copy()


def make_pred_obs_figure() -> None:
    import matplotlib.pyplot as plt

    pieces = []
    for tag, label in [(BASELINE_TAG, "Original graph rollout"), (FINAL_RETAINED_TAG, "Hybrid-Edge k=8")]:
        df = load_mc_per_patient(tag).query("bucket == 'T0->T3'").copy()
        df["paper_label"] = label
        pieces.append(df[["patient_id", "paper_label", "obs_ftv_ml", "pred_ftv_det_ml"]])
    plot_df = pd.concat(pieces, ignore_index=True)
    xmax = np.nanpercentile(plot_df[["obs_ftv_ml", "pred_ftv_det_ml"]].to_numpy().reshape(-1), 99.5)
    xmax = max(80.0, float(xmax))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), sharex=True, sharey=True)
    for ax, label in zip(axes, ["Original graph rollout", "Hybrid-Edge k=8"]):
        sub = plot_df.loc[plot_df["paper_label"] == label]
        x = sub["obs_ftv_ml"].to_numpy(float)
        y = sub["pred_ftv_det_ml"].to_numpy(float)
        ax.scatter(x, y, s=9, alpha=0.28, linewidths=0, color="#2A6F97")
        ax.plot([0, xmax], [0, xmax], color="#202020", lw=1.2, ls="--")
        bias = np.nanmean(y - x)
        mae = np.nanmean(np.abs(y - x))
        ax.text(
            0.04,
            0.94,
            f"bias {bias:.1f} mL\nMAE {mae:.1f} mL",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D0D0D0", alpha=0.9),
        )
        ax.set_title(label, fontsize=11, weight="bold")
        ax.grid(True, color="#E7E7E7", linewidth=0.7)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, xmax)
    axes[0].set_ylabel("Predicted T3 FTV (mL)")
    for ax in axes:
        ax.set_xlabel("Observed T3 FTV (mL)")
    fig.suptitle("Deterministic T0-to-T3 FTV calibration", fontsize=12, weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "deterministic_t3_pred_obs.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def related_work_positioning_table() -> pd.DataFrame:
    rows = [
        {
            "comparison_family": "I-SPY FTV biomarker studies",
            "representative_work": "Hylton et al. 2012; Hylton et al. 2016; Li et al. 2020",
            "primary_task": "response/prognosis association from serial MRI features",
            "output": "pCR, recurrence risk, or endpoint response association",
            "relationship_to_this_work": "establishes FTV and serial MRI as clinically meaningful response signals",
            "remaining_gap_addressed_here": "does not provide a patient-level forecast distribution for future FTV",
        },
        {
            "comparison_family": "Longitudinal MRI fusion models",
            "representative_work": "Huang et al. 2023",
            "primary_task": "predict pCR or response class from serial MRI",
            "output": "patient-level pCR or response label",
            "relationship_to_this_work": "uses serial MRI to improve response classification",
            "remaining_gap_addressed_here": "does not roll forward a tumor graph state or produce a calibrated future FTV distribution",
        },
        {
            "comparison_family": "Spatiotemporal MRI classifiers",
            "representative_work": "Tang et al. 2025; Huang et al. 2026 BSTNet",
            "primary_task": "predict pCR or response class from temporal/spatial MRI patterns",
            "output": "patient-level pCR or response probability",
            "relationship_to_this_work": "captures temporal and spatial MRI patterns for classification",
            "remaining_gap_addressed_here": "prediction target remains a response label rather than a future tumor-burden trajectory",
        },
        {
            "comparison_family": "Mechanistic breast cancer digital twins",
            "representative_work": "Weis et al. 2013/2015; Wu et al. 2022/2025; Christenson et al. 2025",
            "primary_task": "calibrate patient-specific tumor-growth or treatment-response equations",
            "output": "spatiotemporal response simulation or regimen optimization",
            "relationship_to_this_work": "closest conceptual neighbor for patient-updated forecasting",
            "remaining_gap_addressed_here": "uses learned registration-consistent graph rollouts with empirical FTV uncertainty rather than mechanistic equations",
        },
        {
            "comparison_family": "Current graph rollout baseline",
            "representative_work": "scheduled-sampling graph-convolutional rollout",
            "primary_task": "roll forward registration-consistent tumor supervoxel graphs",
            "output": "future spatial tumor cloud and deterministic FTV center",
            "relationship_to_this_work": "direct internal baseline with the same graph substrate and rollout protocol",
            "remaining_gap_addressed_here": "endpoint FTV/alive calibration fixes the biased burden center before residual MC sampling",
        },
    ]
    return pd.DataFrame(rows)


def bucket_calibration_summary() -> pd.DataFrame:
    rows = []
    label_map = {
        BASELINE_TAG: "Original graph rollout",
        FINAL_RETAINED_TAG: "Hybrid-Edge k=8",
    }
    paper_label = {
        BASELINE_TAG: "Baseline",
        FINAL_RETAINED_TAG: "Retained",
    }
    for tag in [BASELINE_TAG, FINAL_RETAINED_TAG]:
        df = load_mc_per_patient(tag)
        for bucket in ["T0->T3", "T1->T3", "T2->T3"]:
            grp = df.loc[df["bucket"].eq(bucket)].copy()
            cutoff = float(grp["obs_ftv_ml"].quantile(0.90))
            high = grp.loc[grp["obs_ftv_ml"].ge(cutoff)]
            rows.append(
                {
                    "model_tag": tag,
                    "model_label": label_map[tag],
                    "bucket": bucket,
                    "n_patients": int(grp["patient_id"].nunique()),
                    "det_ftv_mae_ml_mean": float(grp["det_ftv_abs_err_ml"].mean()),
                    "det_ftv_bias_ml_mean": float(grp["det_ftv_signed_err_ml"].mean()),
                    "mc_mean_ftv_mae_ml_mean": float(grp["mc_mean_ftv_abs_err_ml"].mean()),
                    "mc_mean_ftv_bias_ml_mean": float(grp["mc_mean_ftv_signed_err_ml"].mean()),
                    "coverage90_ftv_raw_mean": float(grp["coverage90_ftv_raw"].mean()),
                    "coverage90_ftv_conformal_mean": float(grp["coverage90_ftv_conformal"].mean()),
                    "ftv_raw_width90_ml_mean": float(grp["ftv_raw_width90_ml"].mean()),
                    "ftv_conformal_width90_ml_mean": float(grp["ftv_conformal_width90_ml"].mean()),
                    "crps_ftv_mean": float(grp["crps_ftv"].mean()),
                    "alive_count_abs_err_mc_mean_mean": float(grp["alive_count_abs_err_mc_mean"].mean()),
                    "high_t3_mc_mean_ftv_mae_ml_mean": float(high["mc_mean_ftv_abs_err_ml"].mean()),
                    "paper_label": paper_label[tag],
                }
            )
    return pd.DataFrame(rows)


def low_residual_burden_readout(threshold_ml: float = 0.1) -> pd.DataFrame:
    df = load_mc_per_patient(FINAL_RETAINED_TAG).query("bucket == 'T0->T3'").copy()
    df["low_residual_observed"] = df["obs_ftv_ml"] <= threshold_ml
    df["pcr_label"] = pd.to_numeric(df["pCR"], errors="coerce")
    score = pd.to_numeric(df["pcr_prob_mc"], errors="coerce")

    rows = [
        {
            "readout": f"observed_ftv_le_{threshold_ml:g}_ml",
            "n_patients": int(df.shape[0]),
            "event_rate": float(df["low_residual_observed"].mean()),
            "score_mean": float(score.mean()),
            "score_auc": binary_auc(df["low_residual_observed"].astype(int), score),
            "note": "Low residual imaging burden score: fraction of MC draws with near-zero T3 MRI FTV, not a pathology response probability.",
        },
        {
            "readout": "pathology_pCR",
            "n_patients": int(df["pcr_label"].notna().sum()),
            "event_rate": float(df["pcr_label"].mean()),
            "score_mean": float(score.mean()),
            "score_auc": binary_auc(df["pcr_label"], score),
            "note": "Association with pathology pCR where labels are available; no pathology response head was trained.",
        },
    ]

    # Quantile bins are useful even when many scores are near zero; duplicates are dropped.
    valid = df.loc[score.notna()].copy()
    valid["score"] = score.loc[valid.index]
    try:
        valid["score_bin"] = pd.qcut(valid["score"], q=5, duplicates="drop")
        for bin_label, grp in valid.groupby("score_bin", observed=True):
            rows.append(
                {
                    "readout": f"score_bin_{bin_label}",
                    "n_patients": int(grp.shape[0]),
                    "event_rate": float(grp["pcr_label"].mean()),
                    "score_mean": float(grp["score"].mean()),
                    "score_auc": np.nan,
                    "note": "Binned association with pathology pCR by low residual imaging burden score.",
                }
            )
    except ValueError:
        pass
    return pd.DataFrame(rows)


def _visit_ftv_table(source: pd.DataFrame) -> pd.DataFrame:
    meta_cols = ["patient_id", "fold", "collection", "pCR", "subtype"]
    t0 = source[meta_cols + ["ftv_t0_ml"]].drop_duplicates("patient_id").copy()
    t0 = t0.rename(columns={"ftv_t0_ml": "ftv_ml"})
    t0["visit"] = "T0"
    pieces = [t0[meta_cols + ["visit", "ftv_ml"]]]
    for visit in ["T1", "T2", "T3"]:
        rows = source.loc[
            source["start_visit"].eq("T0") & source["predicted_visit"].eq(visit),
            meta_cols + ["obs_ftv_ml"],
        ].copy()
        rows = rows.rename(columns={"obs_ftv_ml": "ftv_ml"})
        rows["visit"] = visit
        pieces.append(rows[meta_cols + ["visit", "ftv_ml"]])
    return pd.concat(pieces, ignore_index=True).drop_duplicates(["patient_id", "visit"])


def _standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x_train, axis=0)
    std = np.nanstd(x_train, axis=0)
    std[~np.isfinite(std) | (std == 0.0)] = 1.0
    x_train_std = (x_train - mean) / std
    x_test_std = (x_test - mean) / std
    x_train_std = np.nan_to_num(x_train_std, nan=0.0, posinf=0.0, neginf=0.0)
    x_test_std = np.nan_to_num(x_test_std, nan=0.0, posinf=0.0, neginf=0.0)
    return x_train_std, x_test_std


def _ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float = 10.0,
) -> np.ndarray:
    x_train_std, x_test_std = _standardize_train_test(x_train, x_test)
    design = np.c_[np.ones(x_train_std.shape[0]), x_train_std]
    design_test = np.c_[np.ones(x_test_std.shape[0]), x_test_std]
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + alpha * penalty, design.T @ y_train)
    return design_test @ beta


def _rbf_kernel_ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float = 10.0,
) -> np.ndarray:
    x_train_std, x_test_std = _standardize_train_test(x_train, x_test)
    train_sq = np.sum((x_train_std[:, None, :] - x_train_std[None, :, :]) ** 2, axis=2)
    nonzero = train_sq[train_sq > 1e-8]
    gamma = 1.0 / float(np.median(nonzero)) if nonzero.size else 1.0
    k_train = np.exp(-gamma * train_sq)
    test_sq = np.sum((x_test_std[:, None, :] - x_train_std[None, :, :]) ** 2, axis=2)
    k_test = np.exp(-gamma * test_sq)
    coef = np.linalg.solve(k_train + alpha * np.eye(k_train.shape[0]), y_train)
    return k_test @ coef


def scalar_temporal_feature_frame() -> tuple[pd.DataFrame, list[str]]:
    source = load_mc_per_patient(FINAL_RETAINED_TAG)
    visits = _visit_ftv_table(source)
    wide = visits.pivot(index="patient_id", columns="visit", values="ftv_ml")
    meta = visits.drop_duplicates("patient_id")[["patient_id", "fold", "collection", "pCR", "subtype"]].set_index("patient_id")
    targets = source.loc[
        source["predicted_visit"].eq("T3") & source["start_visit"].isin(["T0", "T1", "T2"])
    ].copy()

    visit_idx = {"T0": 0, "T1": 1, "T2": 2, "T3": 3}
    rows = []
    for _, row in targets.iterrows():
        pid = row["patient_id"]
        start = row["start_visit"]
        start_i = visit_idx[start]
        record = {
            "patient_id": pid,
            "fold": int(row["fold"]),
            "collection": row["collection"],
            "subtype": row["subtype"],
            "bucket": f"{start}->T3",
            "start_visit": start,
            "start_idx": start_i,
            "obs_ftv_ml": float(row["obs_ftv_ml"]),
        }
        for visit in ["T0", "T1", "T2"]:
            known = visit_idx[visit] <= start_i
            value = float(wide.loc[pid, visit]) if known and np.isfinite(wide.loc[pid, visit]) else 0.0
            record[f"{visit.lower()}_ftv_ml"] = value
            record[f"{visit.lower()}_log_ftv"] = float(np.log1p(max(value, 0.0)))
            record[f"{visit.lower()}_observed"] = float(known)
        record["last_observed_ftv_ml"] = float(wide.loc[pid, start])
        record["last_observed_log_ftv"] = float(np.log1p(max(float(wide.loc[pid, start]), 0.0)))
        rows.append(record)

    df = pd.DataFrame(rows)
    cats = pd.get_dummies(df[["collection", "subtype", "start_visit"]], prefix=["collection", "subtype", "start"])
    df = pd.concat([df, cats.astype(float)], axis=1)
    feature_cols = [
        "start_idx",
        "t0_ftv_ml",
        "t1_ftv_ml",
        "t2_ftv_ml",
        "t0_log_ftv",
        "t1_log_ftv",
        "t2_log_ftv",
        "t0_observed",
        "t1_observed",
        "t2_observed",
        "last_observed_ftv_ml",
        "last_observed_log_ftv",
    ] + list(cats.columns)
    return df, feature_cols


def scalar_baseline_t3_table() -> pd.DataFrame:
    source = load_mc_per_patient(FINAL_RETAINED_TAG)
    visits = _visit_ftv_table(source)
    visit_idx = {"T0": 0, "T1": 1, "T2": 2, "T3": 3}
    wide = visits.pivot(index="patient_id", columns="visit", values="ftv_ml")
    targets = source.loc[
        source["predicted_visit"].eq("T3") & source["start_visit"].isin(["T0", "T1", "T2"])
    ].copy()

    pred_rows = []
    for _, row in targets.iterrows():
        pid = row["patient_id"]
        start = row["start_visit"]
        start_i = visit_idx[start]
        known_visits = [v for v, idx in visit_idx.items() if idx <= start_i and v != "T3"]
        y = wide.loc[pid, known_visits].astype(float).to_numpy()
        x = np.array([visit_idx[v] for v in known_visits], dtype=float)
        last_observed = float(wide.loc[pid, start])
        obs_t3 = float(row["obs_ftv_ml"])

        preds = {"last_observed": last_observed}
        if len(y) >= 2 and np.isfinite(y).all():
            slope, intercept = np.polyfit(x, y, deg=1)
            preds["linear_visit_trend"] = max(float(intercept + slope * 3.0), 0.0)
            log_slope, log_intercept = np.polyfit(x, np.log1p(np.clip(y, 0.0, None)), deg=1)
            preds["log_linear_visit_trend"] = max(float(np.expm1(log_intercept + log_slope * 3.0)), 0.0)
        else:
            preds["linear_visit_trend"] = last_observed
            preds["log_linear_visit_trend"] = last_observed

        for model, pred in preds.items():
            pred_rows.append(
                {
                    "patient_id": pid,
                    "fold": int(row["fold"]),
                    "collection": row["collection"],
                    "subtype": row["subtype"],
                    "bucket": f"{start}->T3",
                    "model": model,
                    "pred_ftv_ml": pred,
                    "obs_ftv_ml": obs_t3,
                    "err_ml": pred - obs_t3,
                    "abs_err_ml": abs(pred - obs_t3),
                }
            )

    preds = pd.DataFrame(pred_rows)
    corrected_rows = []
    for _, row in preds.loc[preds["model"].eq("last_observed")].iterrows():
        train = preds.loc[
            preds["model"].eq("last_observed")
            & preds["bucket"].eq(row["bucket"])
            & preds["fold"].ne(row["fold"])
        ].copy()
        bucket_resid = float((train["obs_ftv_ml"] - train["pred_ftv_ml"]).mean())
        subtype_train = train.loc[train["subtype"].eq(row["subtype"])]
        subtype_resid = (
            float((subtype_train["obs_ftv_ml"] - subtype_train["pred_ftv_ml"]).mean())
            if subtype_train.shape[0] >= 20
            else bucket_resid
        )
        for model, resid in [
            ("fold_calibrated_last_observed", bucket_resid),
            ("fold_subtype_calibrated_last_observed", subtype_resid),
        ]:
            pred = max(float(row["pred_ftv_ml"] + resid), 0.0)
            corrected_rows.append(
                {
                    "patient_id": row["patient_id"],
                    "fold": int(row["fold"]),
                    "collection": row["collection"],
                    "subtype": row["subtype"],
                    "bucket": row["bucket"],
                    "model": model,
                    "pred_ftv_ml": pred,
                    "obs_ftv_ml": float(row["obs_ftv_ml"]),
                    "err_ml": pred - float(row["obs_ftv_ml"]),
                    "abs_err_ml": abs(pred - float(row["obs_ftv_ml"])),
                }
            )
    all_preds = pd.concat([preds, pd.DataFrame(corrected_rows)], ignore_index=True)
    summary = (
        all_preds.groupby(["bucket", "model"], as_index=False)
        .agg(
            n_patients=("patient_id", "nunique"),
            pred_ftv_ml_mean=("pred_ftv_ml", "mean"),
            obs_ftv_ml_mean=("obs_ftv_ml", "mean"),
            bias_ml_mean=("err_ml", "mean"),
            mae_ml_mean=("abs_err_ml", "mean"),
        )
        .sort_values(["bucket", "mae_ml_mean"])
    )
    return summary


def center_prediction_t3_rows() -> pd.DataFrame:
    source = load_mc_per_patient(FINAL_RETAINED_TAG)
    visits = _visit_ftv_table(source)
    wide = visits.pivot(index="patient_id", columns="visit", values="ftv_ml")
    targets = source.loc[
        source["predicted_visit"].eq("T3") & source["start_visit"].isin(["T0", "T1", "T2"])
    ].copy()

    base_rows = []
    for _, row in targets.iterrows():
        pid = row["patient_id"]
        start = row["start_visit"]
        base_rows.append(
            {
                "patient_id": pid,
                "fold": int(row["fold"]),
                "collection": row["collection"],
                "subtype": row["subtype"],
                "bucket": f"{start}->T3",
                "start_visit": start,
                "graph_det_ftv_ml": float(row["pred_ftv_det_ml"]),
                "last_observed_ftv_ml": float(wide.loc[pid, start]),
                "t0_ftv_ml": float(wide.loc[pid, "T0"]),
                "obs_ftv_ml": float(row["obs_ftv_ml"]),
            }
        )
    base = pd.DataFrame(base_rows)

    pred_rows = []
    for _, row in base.iterrows():
        for model, pred in [
            ("last_observed", row["last_observed_ftv_ml"]),
            ("graph_retained_det", row["graph_det_ftv_ml"]),
        ]:
            pred_rows.append(
                {
                    **row.to_dict(),
                    "center_model": model,
                    "pred_center_ml": max(float(pred), 0.0),
                }
            )

    features = ["graph_det_ftv_ml", "last_observed_ftv_ml", "t0_ftv_ml"]
    ridge_alpha = 10.0
    for bucket, bucket_df in base.groupby("bucket"):
        for fold, test in bucket_df.groupby("fold"):
            train = bucket_df.loc[bucket_df["fold"].ne(fold)]
            x_train = train[features].to_numpy(float)
            y_train = train["obs_ftv_ml"].to_numpy(float)
            x_test = test[features].to_numpy(float)

            mean = x_train.mean(axis=0)
            std = x_train.std(axis=0)
            std[std == 0.0] = 1.0
            x_train_std = (x_train - mean) / std
            x_test_std = (x_test - mean) / std
            design = np.c_[np.ones(x_train_std.shape[0]), x_train_std]
            design_test = np.c_[np.ones(x_test_std.shape[0]), x_test_std]
            penalty = np.eye(design.shape[1])
            penalty[0, 0] = 0.0
            beta = np.linalg.solve(design.T @ design + ridge_alpha * penalty, design.T @ y_train)
            preds = design_test @ beta

            for pred, (_, row) in zip(preds, test.iterrows()):
                pred_rows.append(
                    {
                        **row.to_dict(),
                        "center_model": "hybrid_ridge_graph_last_t0",
                        "pred_center_ml": max(float(pred), 0.0),
                    }
                )

    out = pd.DataFrame(pred_rows)
    out["center_err_ml"] = out["pred_center_ml"] - out["obs_ftv_ml"]
    out["center_abs_err_ml"] = out["center_err_ml"].abs()
    return out


def center_prediction_t3_table() -> pd.DataFrame:
    rows = center_prediction_t3_rows()
    summary = (
        rows.groupby(["bucket", "center_model"], as_index=False)
        .agg(
            n_patients=("patient_id", "nunique"),
            pred_ftv_ml_mean=("pred_center_ml", "mean"),
            obs_ftv_ml_mean=("obs_ftv_ml", "mean"),
            bias_ml_mean=("center_err_ml", "mean"),
            mae_ml_mean=("center_abs_err_ml", "mean"),
        )
        .sort_values(["bucket", "mae_ml_mean"])
    )
    return summary


def strong_scalar_center_prediction_t3_rows() -> pd.DataFrame:
    features, feature_cols = scalar_temporal_feature_frame()
    pred_rows = []
    model_specs = [
        ("ridge_temporal_raw", "ridge", "raw"),
        ("ridge_temporal_log", "ridge", "log"),
        ("rbf_kernel_temporal_log", "rbf", "log"),
    ]
    for bucket, bucket_df in features.groupby("bucket"):
        for fold, test in bucket_df.groupby("fold"):
            train = bucket_df.loc[bucket_df["fold"].ne(fold)]
            x_train = train[feature_cols].to_numpy(float)
            x_test = test[feature_cols].to_numpy(float)
            y_raw = train["obs_ftv_ml"].to_numpy(float)
            y_log = np.log1p(np.clip(y_raw, 0.0, None))
            for model, family, target_scale in model_specs:
                if family == "ridge":
                    pred = _ridge_predict(
                        x_train,
                        y_log if target_scale == "log" else y_raw,
                        x_test,
                        alpha=10.0,
                    )
                elif family == "rbf":
                    pred = _rbf_kernel_ridge_predict(
                        x_train,
                        y_log if target_scale == "log" else y_raw,
                        x_test,
                        alpha=10.0,
                    )
                else:
                    raise ValueError(f"unknown scalar family: {family}")
                if target_scale == "log":
                    pred = np.expm1(pred)
                pred = np.clip(pred, 0.0, None)
                for pred_value, (_, row) in zip(pred, test.iterrows()):
                    pred_rows.append(
                        {
                            "patient_id": row["patient_id"],
                            "fold": int(row["fold"]),
                            "collection": row["collection"],
                            "subtype": row["subtype"],
                            "bucket": bucket,
                            "start_visit": row["start_visit"],
                            "graph_det_ftv_ml": np.nan,
                            "last_observed_ftv_ml": float(row["last_observed_ftv_ml"]),
                            "t0_ftv_ml": float(row["t0_ftv_ml"]),
                            "obs_ftv_ml": float(row["obs_ftv_ml"]),
                            "center_model": model,
                            "pred_center_ml": float(pred_value),
                        }
                    )
    out = pd.DataFrame(pred_rows)
    out["center_err_ml"] = out["pred_center_ml"] - out["obs_ftv_ml"]
    out["center_abs_err_ml"] = out["center_err_ml"].abs()
    return out


def strong_scalar_baselines_t3_table() -> pd.DataFrame:
    rows = strong_scalar_center_prediction_t3_rows()
    return (
        rows.groupby(["bucket", "center_model"], as_index=False)
        .agg(
            n_patients=("patient_id", "nunique"),
            pred_ftv_ml_mean=("pred_center_ml", "mean"),
            obs_ftv_ml_mean=("obs_ftv_ml", "mean"),
            bias_ml_mean=("center_err_ml", "mean"),
            mae_ml_mean=("center_abs_err_ml", "mean"),
        )
        .sort_values(["bucket", "mae_ml_mean"])
    )


def residual_mc_t3_per_patient_for_centers(center_models: list[str]) -> pd.DataFrame:
    centers = pd.concat(
        [center_prediction_t3_rows(), strong_scalar_center_prediction_t3_rows()],
        ignore_index=True,
        sort=False,
    )
    centers = centers.loc[centers["center_model"].isin(center_models)].copy()

    per_patient = []
    for (bucket, center_model), model_df in centers.groupby(["bucket", "center_model"]):
        for fold, test in model_df.groupby("fold"):
            train = model_df.loc[model_df["fold"].ne(fold)]
            residuals = (train["obs_ftv_ml"] - train["pred_center_ml"]).to_numpy(float)
            residuals = residuals[np.isfinite(residuals)]
            if residuals.size < 2:
                continue
            pairwise_abs = float(np.abs(residuals[:, None] - residuals[None, :]).mean())
            for _, row in test.iterrows():
                samples = float(row["pred_center_ml"]) + residuals
                raw_p05, raw_p95 = np.percentile(samples, [5, 95])
                mc_mean = float(samples.mean())
                obs = float(row["obs_ftv_ml"])
                crps = float(np.abs(samples - obs).mean() - 0.5 * pairwise_abs)
                per_patient.append(
                    {
                        "patient_id": row["patient_id"],
                        "fold": int(row["fold"]),
                        "bucket": bucket,
                        "center_model": center_model,
                        "pred_center_ml": float(row["pred_center_ml"]),
                        "obs_ftv_ml": obs,
                        "mc_mean_ftv_ml": mc_mean,
                        "center_abs_err_ml": float(row["center_abs_err_ml"]),
                        "mc_mean_abs_err_ml": abs(mc_mean - obs),
                        "coverage90_ftv_raw": float(raw_p05 <= obs <= raw_p95),
                        "ftv_raw_width90_ml": float(raw_p95 - raw_p05),
                        "ftv_raw_p05_ml": float(raw_p05),
                        "ftv_raw_p95_ml": float(raw_p95),
                        "crps_ftv": crps,
                    }
                )

    pp = pd.DataFrame(per_patient)
    if pp.empty:
        return pp
    pp["ftv_raw_nonconformity_ml"] = np.maximum.reduce(
        [
            pp["ftv_raw_p05_ml"].to_numpy(float) - pp["obs_ftv_ml"].to_numpy(float),
            pp["obs_ftv_ml"].to_numpy(float) - pp["ftv_raw_p95_ml"].to_numpy(float),
            np.zeros(pp.shape[0]),
        ]
    )
    conformal_rows = []
    for _, row in pp.iterrows():
        calib = pp.loc[
            pp["bucket"].eq(row["bucket"])
            & pp["center_model"].eq(row["center_model"])
            & pp["fold"].ne(row["fold"]),
            "ftv_raw_nonconformity_ml",
        ].to_numpy(float)
        q = float(np.quantile(calib, 0.90)) if calib.size else 0.0
        lo = float(row["ftv_raw_p05_ml"] - q)
        hi = float(row["ftv_raw_p95_ml"] + q)
        obs = float(row["obs_ftv_ml"])
        conformal_rows.append(
            {
                "ftv_conformal_q_ml": q,
                "ftv_conformal_p05_ml": lo,
                "ftv_conformal_p95_ml": hi,
                "ftv_conformal_width90_ml": hi - lo,
                "coverage90_ftv_conformal": float(lo <= obs <= hi),
            }
        )
    pp = pd.concat([pp.reset_index(drop=True), pd.DataFrame(conformal_rows)], axis=1)
    return pp


def residual_mc_t3_table_for_centers(center_models: list[str]) -> pd.DataFrame:
    pp = residual_mc_t3_per_patient_for_centers(center_models)
    summary = (
        pp.groupby(["bucket", "center_model"], as_index=False)
        .agg(
            n_patients=("patient_id", "nunique"),
            center_ftv_mae_ml=("center_abs_err_ml", "mean"),
            mc_mean_ftv_mae_ml=("mc_mean_abs_err_ml", "mean"),
            coverage90_ftv_raw=("coverage90_ftv_raw", "mean"),
            coverage90_ftv_conformal=("coverage90_ftv_conformal", "mean"),
            ftv_raw_width90_ml=("ftv_raw_width90_ml", "mean"),
            ftv_conformal_width90_ml=("ftv_conformal_width90_ml", "mean"),
            crps_ftv=("crps_ftv", "mean"),
        )
        .sort_values(["bucket", "center_model"])
    )
    return summary


def scalar_residual_mc_t3_table() -> pd.DataFrame:
    return residual_mc_t3_table_for_centers(["last_observed"]).rename(columns={"center_model": "model"})


def hybrid_residual_mc_t3_table() -> pd.DataFrame:
    return residual_mc_t3_table_for_centers(["hybrid_ridge_graph_last_t0"]).rename(columns={"center_model": "model"})


def strong_scalar_baselines_mc_t3_table() -> pd.DataFrame:
    models = ["ridge_temporal_raw", "ridge_temporal_log", "rbf_kernel_temporal_log"]
    return residual_mc_t3_table_for_centers(models).rename(columns={"center_model": "model"})


def scalar_vs_graph_mc_t3_table() -> pd.DataFrame:
    scalar = scalar_residual_mc_t3_table().rename(
        columns={
            "center_ftv_mae_ml": "det_ftv_mae_ml",
            "coverage90_ftv_raw": "raw_90_coverage",
        }
    )
    scalar = scalar.assign(model_family="Scalar residual MC", paper_label="Last-observed scalar MC")
    scalar = scalar[
        [
            "bucket",
            "model_family",
            "paper_label",
            "n_patients",
            "det_ftv_mae_ml",
            "mc_mean_ftv_mae_ml",
            "raw_90_coverage",
            "ftv_raw_width90_ml",
            "crps_ftv",
        ]
    ]

    strong = strong_scalar_baselines_mc_t3_table().rename(
        columns={
            "center_ftv_mae_ml": "det_ftv_mae_ml",
            "coverage90_ftv_raw": "raw_90_coverage",
        }
    )
    label_map = {
        "ridge_temporal_raw": "Ridge temporal scalar MC",
        "ridge_temporal_log": "Log-ridge temporal scalar MC",
        "rbf_kernel_temporal_log": "Kernel temporal scalar MC",
    }
    strong["model_family"] = "Temporal scalar residual MC"
    strong["paper_label"] = strong["model"].map(label_map)
    strong = strong[
        [
            "bucket",
            "model_family",
            "paper_label",
            "n_patients",
            "det_ftv_mae_ml",
            "mc_mean_ftv_mae_ml",
            "raw_90_coverage",
            "ftv_raw_width90_ml",
            "crps_ftv",
        ]
    ]

    hybrid = hybrid_residual_mc_t3_table().rename(
        columns={
            "center_ftv_mae_ml": "det_ftv_mae_ml",
            "coverage90_ftv_raw": "raw_90_coverage",
        }
    )
    hybrid = hybrid.assign(model_family="Hybrid residual MC", paper_label="Hybrid graph+scalar MC")
    hybrid = hybrid[
        [
            "bucket",
            "model_family",
            "paper_label",
            "n_patients",
            "det_ftv_mae_ml",
            "mc_mean_ftv_mae_ml",
            "raw_90_coverage",
            "ftv_raw_width90_ml",
            "crps_ftv",
        ]
    ]

    graph = bucket_calibration_summary().copy()
    graph = graph.rename(
        columns={
            "det_ftv_mae_ml_mean": "det_ftv_mae_ml",
            "mc_mean_ftv_mae_ml_mean": "mc_mean_ftv_mae_ml",
            "coverage90_ftv_raw_mean": "raw_90_coverage",
            "ftv_raw_width90_ml_mean": "ftv_raw_width90_ml",
            "crps_ftv_mean": "crps_ftv",
        }
    )
    graph["model_family"] = "Graph residual MC"
    graph["paper_label"] = graph["paper_label"].map({"Baseline": "Graph baseline", "Retained": "Graph retained"})
    graph = graph[
        [
            "bucket",
            "model_family",
            "paper_label",
            "n_patients",
            "det_ftv_mae_ml",
            "mc_mean_ftv_mae_ml",
            "raw_90_coverage",
            "ftv_raw_width90_ml",
            "crps_ftv",
        ]
    ]
    return pd.concat([graph, scalar, strong, hybrid], ignore_index=True).sort_values(["bucket", "mc_mean_ftv_mae_ml"])


def t3_per_patient_model_metrics() -> pd.DataFrame:
    rows = []
    graph_specs = [
        (BASELINE_TAG, "Graph baseline", "Graph residual MC"),
        (FINAL_RETAINED_TAG, "Graph retained", "Graph residual MC"),
    ]
    for tag, label, family in graph_specs:
        df = load_mc_per_patient(tag).query("bucket == 'T0->T3'").copy()
        rows.append(
            pd.DataFrame(
                {
                    "patient_id": df["patient_id"],
                    "fold": df["fold"],
                    "collection": df["collection"],
                    "subtype": df["subtype"],
                    "model": label,
                    "model_family": family,
                    "obs_ftv_ml": df["obs_ftv_ml"],
                    "det_ftv_mae_ml": df["det_ftv_abs_err_ml"],
                    "mc_mean_ftv_mae_ml": df["mc_mean_ftv_abs_err_ml"],
                    "coverage90_ftv_raw": df["coverage90_ftv_raw"],
                    "coverage90_ftv_conformal": df["coverage90_ftv_conformal"],
                    "ftv_raw_width90_ml": df["ftv_raw_width90_ml"],
                    "ftv_conformal_width90_ml": df["ftv_conformal_width90_ml"],
                    "crps_ftv": df["crps_ftv"],
                }
            )
        )

    center_labels = {
        "last_observed": ("Last-observed scalar MC", "Scalar residual MC"),
        "ridge_temporal_raw": ("Ridge temporal scalar MC", "Temporal scalar residual MC"),
        "ridge_temporal_log": ("Log-ridge temporal scalar MC", "Temporal scalar residual MC"),
        "rbf_kernel_temporal_log": ("Kernel temporal scalar MC", "Temporal scalar residual MC"),
        "hybrid_ridge_graph_last_t0": ("Hybrid graph+scalar MC", "Hybrid residual MC"),
    }
    center_pp = residual_mc_t3_per_patient_for_centers(list(center_labels)).query("bucket == 'T0->T3'")
    for center_model, (label, family) in center_labels.items():
        df = center_pp.loc[center_pp["center_model"].eq(center_model)]
        rows.append(
            pd.DataFrame(
                {
                    "patient_id": df["patient_id"],
                    "fold": df["fold"],
                    "collection": np.nan,
                    "subtype": np.nan,
                    "model": label,
                    "model_family": family,
                    "obs_ftv_ml": df["obs_ftv_ml"],
                    "det_ftv_mae_ml": df["center_abs_err_ml"],
                    "mc_mean_ftv_mae_ml": df["mc_mean_abs_err_ml"],
                    "coverage90_ftv_raw": df["coverage90_ftv_raw"],
                    "coverage90_ftv_conformal": df["coverage90_ftv_conformal"],
                    "ftv_raw_width90_ml": df["ftv_raw_width90_ml"],
                    "ftv_conformal_width90_ml": df["ftv_conformal_width90_ml"],
                    "crps_ftv": df["crps_ftv"],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _metric_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        df.groupby(group_cols, observed=True)
        .agg(
            n=("patient_id", "nunique"),
            obs_ftv_ml_mean=("obs_ftv_ml", "mean"),
            det_ftv_mae_ml=("det_ftv_mae_ml", "mean"),
            mc_mean_ftv_mae_ml=("mc_mean_ftv_mae_ml", "mean"),
            raw_90_coverage=("coverage90_ftv_raw", "mean"),
            conformal_90_coverage=("coverage90_ftv_conformal", "mean"),
            raw_width90_ml=("ftv_raw_width90_ml", "mean"),
            conformal_width90_ml=("ftv_conformal_width90_ml", "mean"),
            crps=("crps_ftv", "mean"),
        )
        .reset_index()
    )


def calibration_by_t3_burden_quartile_table() -> pd.DataFrame:
    df = t3_per_patient_model_metrics()
    reference = df.loc[df["model"].eq("Graph retained"), ["patient_id", "obs_ftv_ml"]].drop_duplicates("patient_id")
    reference["burden_quartile"] = pd.qcut(
        reference["obs_ftv_ml"],
        q=4,
        labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"],
    )
    out = df.merge(reference[["patient_id", "burden_quartile"]], on="patient_id", how="left")
    out = _metric_summary(out, ["burden_quartile", "model_family", "model"])
    return out.sort_values(["burden_quartile", "mc_mean_ftv_mae_ml"])


def calibration_by_t3_burden_tail_table() -> pd.DataFrame:
    df = t3_per_patient_model_metrics()
    reference = df.loc[df["model"].eq("Graph retained"), ["patient_id", "obs_ftv_ml"]].drop_duplicates("patient_id")
    decile = float(reference["obs_ftv_ml"].quantile(0.90))
    top5 = float(reference["obs_ftv_ml"].quantile(0.95))
    rows = []
    for label, mask in [
        (f"top_decile_ge_{decile:.2f}_ml", reference["obs_ftv_ml"].ge(decile)),
        (f"top_5pct_ge_{top5:.2f}_ml", reference["obs_ftv_ml"].ge(top5)),
        (f"below_top_decile_lt_{decile:.2f}_ml", reference["obs_ftv_ml"].lt(decile)),
    ]:
        subset_ids = set(reference.loc[mask, "patient_id"])
        sub = df.loc[df["patient_id"].isin(subset_ids)].copy()
        summary = _metric_summary(sub, ["model_family", "model"])
        summary.insert(0, "burden_tail", label)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True).sort_values(["burden_tail", "mc_mean_ftv_mae_ml"])


def coverage_vs_nominal_table() -> pd.DataFrame:
    levels = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    rows = []
    for tag, label in [(BASELINE_TAG, "Graph baseline"), (FINAL_RETAINED_TAG, "Graph retained")]:
        per_patient = load_mc_per_patient(tag)[["patient_id", "bucket", "obs_ftv_ml"]].copy()
        samples = load_mc_samples(tag).merge(per_patient, on=["patient_id", "bucket"], how="inner")
        for bucket in ["T0->T3", "T1->T3", "T2->T3"]:
            bucket_samples = samples.loc[samples["bucket"].eq(bucket)]
            for level in levels:
                lo_q = (1.0 - level) / 2.0
                hi_q = 1.0 - lo_q
                quant = bucket_samples.groupby("patient_id")["ftv_sample_ml"].quantile([lo_q, hi_q]).unstack()
                quant.columns = ["lo", "hi"]
                obs = per_patient.loc[per_patient["bucket"].eq(bucket)].drop_duplicates("patient_id").set_index("patient_id")["obs_ftv_ml"]
                joined = quant.join(obs, how="inner")
                covered = joined["obs_ftv_ml"].between(joined["lo"], joined["hi"])
                rows.append(
                    {
                        "model": label,
                        "bucket": bucket,
                        "nominal": level,
                        "empirical_coverage": float(covered.mean()),
                        "mean_width_ml": float((joined["hi"] - joined["lo"]).mean()),
                        "n": int(joined.shape[0]),
                    }
                )
    return pd.DataFrame(rows)


def pit_t0t3_table() -> pd.DataFrame:
    rows = []
    for tag, label in [(BASELINE_TAG, "Graph baseline"), (FINAL_RETAINED_TAG, "Graph retained")]:
        per_patient = load_mc_per_patient(tag).query("bucket == 'T0->T3'")[["patient_id", "obs_ftv_ml"]].copy()
        samples = load_mc_samples(tag).query("bucket == 'T0->T3'").merge(per_patient, on="patient_id", how="inner")
        grouped = samples.groupby("patient_id")
        for patient_id, grp in grouped:
            obs = float(grp["obs_ftv_ml"].iloc[0])
            vals = grp["ftv_sample_ml"].to_numpy(float)
            rows.append(
                {
                    "model": label,
                    "bucket": "T0->T3",
                    "patient_id": patient_id,
                    "pit": float(np.mean(vals <= obs)),
                    "obs_ftv_ml": obs,
                }
            )
    return pd.DataFrame(rows)


def imaging_burden_threshold_metrics_t3() -> pd.DataFrame:
    thresholds = [0.1, 1.0, 5.0]
    per_patient = load_mc_per_patient(FINAL_RETAINED_TAG).query("bucket == 'T0->T3'")[["patient_id", "obs_ftv_ml"]].copy()
    samples = load_mc_samples(FINAL_RETAINED_TAG).query("bucket == 'T0->T3'").merge(per_patient, on="patient_id", how="inner")
    score_rows = []
    for threshold in thresholds:
        scores = (
            samples.assign(draw_event=samples["ftv_sample_ml"].le(threshold).astype(float))
            .groupby("patient_id", as_index=False)
            .agg(score_prob=("draw_event", "mean"), obs_ftv_ml=("obs_ftv_ml", "first"))
        )
        y = scores["obs_ftv_ml"].le(threshold).astype(int)
        pred = scores["score_prob"].ge(0.5).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        score_rows.append(
            {
                "threshold_ml": threshold,
                "n": int(scores.shape[0]),
                "event_rate": float(y.mean()),
                "score_auc": binary_auc(y, scores["score_prob"]),
                "probability_cutoff": 0.5,
                "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
                "specificity": tn / (tn + fp) if tn + fp else np.nan,
                "ppv": tp / (tp + fp) if tp + fp else np.nan,
                "npv": tn / (tn + fn) if tn + fn else np.nan,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "note": "MRI low-residual-burden threshold, not pathology response.",
            }
        )
    return pd.DataFrame(score_rows)


def decision_curve_t3_burden_table() -> pd.DataFrame:
    thresholds = [0.1, 1.0, 5.0]
    cutoffs = np.round(np.arange(0.05, 0.96, 0.05), 2)
    per_patient = load_mc_per_patient(FINAL_RETAINED_TAG).query("bucket == 'T0->T3'")[["patient_id", "obs_ftv_ml"]].copy()
    samples = load_mc_samples(FINAL_RETAINED_TAG).query("bucket == 'T0->T3'").merge(per_patient, on="patient_id", how="inner")
    rows = []
    for threshold in thresholds:
        scores = (
            samples.assign(draw_event=samples["ftv_sample_ml"].le(threshold).astype(float))
            .groupby("patient_id", as_index=False)
            .agg(score_prob=("draw_event", "mean"), obs_ftv_ml=("obs_ftv_ml", "first"))
        )
        y = scores["obs_ftv_ml"].le(threshold).astype(int).to_numpy(int)
        n = y.size
        event_rate = float(y.mean())
        for cutoff in cutoffs:
            pred = scores["score_prob"].ge(cutoff).astype(int).to_numpy(int)
            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            weight = cutoff / (1.0 - cutoff)
            rows.append(
                {
                    "threshold_ml": threshold,
                    "probability_cutoff": float(cutoff),
                    "net_benefit_model": (tp / n) - (fp / n) * weight,
                    "net_benefit_treat_all": event_rate - (1.0 - event_rate) * weight,
                    "net_benefit_treat_none": 0.0,
                    "event_rate": event_rate,
                    "n": int(n),
                }
            )
    return pd.DataFrame(rows)


def split_leakage_audit_table() -> pd.DataFrame:
    retained_path = mc_report_root(FINAL_RETAINED_TAG) / "conditional_mc_per_patient.parquet"
    summary_path = mc_report_root(FINAL_RETAINED_TAG) / "conditional_mc_summary.json"
    retained = pd.read_parquet(retained_path)
    metadata = json.loads(summary_path.read_text())["metadata"]

    fold_counts = (
        retained[["patient_id", "fold"]]
        .drop_duplicates()
        .groupby("fold")["patient_id"]
        .nunique()
        .sort_index()
    )
    fold_text = ", ".join(f"fold {int(k)}: {int(v)}" for k, v in fold_counts.items())
    bucket_counts = (
        retained.groupby(["start_visit", "predicted_visit"])
        .agg(
            n_patients=("patient_id", "nunique"),
            min_calibration_patients=("n_calibration_patients", "min"),
            max_calibration_patients=("n_calibration_patients", "max"),
        )
        .reset_index()
    )
    bucket_text = (
        f"{int(bucket_counts['n_patients'].min())}-{int(bucket_counts['n_patients'].max())} patients per bucket; "
        f"{int(bucket_counts['min_calibration_patients'].min())}-"
        f"{int(bucket_counts['max_calibration_patients'].max())} calibration patients per target"
    )

    rows = [
        {
            "audit_item": "Deterministic graph folds",
            "implementation": "For fold k, train on patients with fold != k and validate/score patients with fold == k; normalization statistics are computed on the training patients only.",
            "evidence": f"{metadata['n_patients']} graph-ready held-out patients scored across five folds ({fold_text}).",
        },
        {
            "audit_item": "Residual MC pools",
            "implementation": "Residual buckets are built within each start/predicted-visit pair; target patient residuals are removed before sampling.",
            "evidence": bucket_text,
        },
        {
            "audit_item": "Conformal calibration",
            "implementation": "Raw-interval nonconformity scores are computed within the same bucket, and the target patient is excluded from the peer set before the conformal quantile is applied.",
            "evidence": "Conformal correction uses bucket peers only; the retained run reports 90% interval calibration for all six buckets.",
        },
        {
            "audit_item": "Scalar and hybrid baselines",
            "implementation": "Scalar residual means, residual MC pools, and hybrid ridge centers are fit using fold != target fold; target-fold rows are scored after fitting.",
            "evidence": "Hybrid features are retained graph FTV, last-observed FTV, and T0 FTV; no target T3 feature is included.",
        },
        {
            "audit_item": "Target-visit leakage",
            "implementation": "For a start visit s, scalar trend features use observed FTV visits with index <= s; observed T3 FTV is used only as an outcome for evaluation or as a training label for other folds.",
            "evidence": "T0->T3 trend baselines reduce to carry-forward because only T0 is observed; later buckets use only visits observed before T3.",
        },
    ]
    return pd.DataFrame(rows)


def make_scalar_baseline_figure() -> None:
    import matplotlib.pyplot as plt

    df = scalar_baseline_t3_table()
    buckets = ["T0->T3", "T1->T3", "T2->T3"]
    models = [
        "last_observed",
        "linear_visit_trend",
        "log_linear_visit_trend",
        "fold_calibrated_last_observed",
        "fold_subtype_calibrated_last_observed",
    ]
    labels = {
        "last_observed": "last obs.",
        "linear_visit_trend": "linear",
        "log_linear_visit_trend": "log-linear",
        "fold_calibrated_last_observed": "fold-cal.",
        "fold_subtype_calibrated_last_observed": "subtype-cal.",
    }
    colors = ["#7A869A", "#A66999", "#E76F51", "#2A9D8F", "#1F77B4"]
    x = np.arange(len(buckets))
    width = 0.14
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    for i, model in enumerate(models):
        sub = df.loc[df["model"].eq(model)].set_index("bucket").reindex(buckets)
        ax.bar(x + (i - 2) * width, sub["mae_ml_mean"].to_numpy(float), width=width, label=labels[model], color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_ylabel("T3 FTV MAE (mL)")
    ax.grid(True, axis="y", color="#E6E6E6", linewidth=0.7)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.suptitle("Task-adapted scalar FTV baselines", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "scalar_baseline_t3_mae.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_mc_diagnostic_figure() -> None:
    import matplotlib.pyplot as plt

    rows = []
    specs = [
        (BASELINE_TAG, "Original"),
        (RETAINED_TAG, "Endpoint+Alive"),
        (FINAL_RETAINED_TAG, "Hybrid-Edge"),
    ]
    for tag, label in specs:
        df = load_mc_per_patient(tag).query("bucket == 'T0->T3'").copy()
        rows.append(
            {
                "model": label,
                "det_bias": float(df["det_ftv_signed_err_ml"].mean()),
                "raw_coverage": float(df["coverage90_ftv_raw"].mean()),
                "conformal_width": float(df["ftv_conformal_width90_ml"].mean()),
                "crps": float(df["crps_ftv"].mean()),
            }
        )
    df = pd.DataFrame(rows)
    colors = ["#7A869A", "#2A9D8F", "#1F77B4"]
    metrics = [
        ("det_bias", "Deterministic bias (mL)", None),
        ("raw_coverage", "Raw 90% coverage", 0.90),
        ("conformal_width", "Conformal width (mL)", None),
        ("crps", "CRPS", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2))
    axes = axes.ravel()
    for ax, (metric, title, ref) in zip(axes, metrics):
        vals = df[metric].to_numpy(float)
        ax.bar(df["model"], vals, color=colors, width=0.65)
        if ref is not None:
            ax.axhline(ref, color="#202020", linestyle="--", linewidth=1.0)
        if metric == "det_bias":
            ax.axhline(0.0, color="#202020", linestyle="--", linewidth=1.0)
        ax.set_title(title, fontsize=10, weight="bold")
        ax.grid(True, axis="y", color="#E6E6E6", linewidth=0.7)
        for tick in ax.get_xticklabels():
            tick.set_rotation(12)
            tick.set_ha("right")
    fig.suptitle("Monte Carlo diagnostics across graph model updates", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "mc_diagnostic_bias_coverage_width.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_scalar_vs_graph_mc_figure() -> None:
    import matplotlib.pyplot as plt

    df = scalar_vs_graph_mc_t3_table()
    buckets = ["T0->T3", "T1->T3", "T2->T3"]
    labels = [
        "Last-observed scalar MC",
        "Log-ridge temporal scalar MC",
        "Kernel temporal scalar MC",
        "Hybrid graph+scalar MC",
        "Graph retained",
        "Graph baseline",
    ]
    colors = {
        "Last-observed scalar MC": "#2A9D8F",
        "Log-ridge temporal scalar MC": "#F4A261",
        "Kernel temporal scalar MC": "#E76F51",
        "Hybrid graph+scalar MC": "#9467BD",
        "Graph retained": "#1F77B4",
        "Graph baseline": "#7A869A",
    }
    metrics = [
        ("mc_mean_ftv_mae_ml", "MC mean MAE (mL)", "lower"),
        ("raw_90_coverage", "Raw 90% coverage", "higher"),
        ("ftv_raw_width90_ml", "Raw width (mL)", "lower"),
        ("crps_ftv", "CRPS", "lower"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.4))
    axes = axes.ravel()
    x = np.arange(len(buckets))
    width = 0.12
    for ax, (metric, ylabel, direction) in zip(axes, metrics):
        for i, label in enumerate(labels):
            sub = df.loc[df["paper_label"].eq(label)].set_index("bucket").reindex(buckets)
            ax.bar(x + (i - (len(labels) - 1) / 2) * width, sub[metric].to_numpy(float), width=width, label=label, color=colors[label])
        ax.set_title(ylabel, fontsize=10, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(buckets)
        ax.grid(True, axis="y", color="#E6E6E6", linewidth=0.7)
        if metric == "raw_90_coverage":
            ax.axhline(0.9, color="#202020", linestyle="--", linewidth=1.0)
            ax.set_ylim(0.65, 0.95)
        ax.text(0.02, 0.95, direction, transform=ax.transAxes, va="top", ha="left", fontsize=8, color="#555555")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Graph MC against a task-adapted scalar residual baseline", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "scalar_vs_graph_mc_t3.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_calibration_by_burden_figure() -> None:
    import matplotlib.pyplot as plt

    df = calibration_by_t3_burden_quartile_table()
    keep = ["Graph retained", "Hybrid graph+scalar MC", "Kernel temporal scalar MC", "Last-observed scalar MC"]
    plot = df.loc[df["model"].isin(keep)].copy()
    quartiles = ["Q1 lowest", "Q2", "Q3", "Q4 highest"]
    colors = {
        "Graph retained": "#1F77B4",
        "Hybrid graph+scalar MC": "#9467BD",
        "Kernel temporal scalar MC": "#E76F51",
        "Last-observed scalar MC": "#2A9D8F",
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), sharex=True)
    x = np.arange(len(quartiles))
    width = 0.18
    for i, model in enumerate(keep):
        sub = plot.loc[plot["model"].eq(model)].set_index("burden_quartile").reindex(quartiles)
        offset = (i - (len(keep) - 1) / 2) * width
        axes[0].bar(x + offset, sub["raw_90_coverage"].to_numpy(float), width=width, color=colors[model], label=model)
        axes[1].bar(x + offset, sub["crps"].to_numpy(float), width=width, color=colors[model], label=model)
    axes[0].axhline(0.9, color="#202020", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Raw 90% coverage")
    axes[0].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("CRPS")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(quartiles, rotation=18, ha="right")
        ax.grid(True, axis="y", color="#E6E6E6", linewidth=0.7)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("T3 burden-conditional calibration", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "calibration_by_t3_burden.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "calibration_by_t3_burden.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_reliability_pit_figure() -> None:
    import matplotlib.pyplot as plt

    cov = coverage_vs_nominal_table()
    pit = pit_t0t3_table()
    colors = {"Graph baseline": "#7A869A", "Graph retained": "#1F77B4"}
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))

    for model, sub in cov.loc[cov["bucket"].eq("T0->T3")].groupby("model"):
        axes[0].plot(
            sub["nominal"].to_numpy(float),
            sub["empirical_coverage"].to_numpy(float),
            marker="o",
            linewidth=1.8,
            color=colors[model],
            label=model,
        )
    axes[0].plot([0.45, 0.97], [0.45, 0.97], color="#202020", linestyle="--", linewidth=1.0)
    axes[0].set_xlim(0.48, 0.97)
    axes[0].set_ylim(0.48, 1.0)
    axes[0].set_xlabel("Nominal central interval")
    axes[0].set_ylabel("Empirical coverage")
    axes[0].grid(True, color="#E6E6E6", linewidth=0.7)
    axes[0].legend(frameon=False, fontsize=8)

    bins = np.linspace(0, 1, 11)
    for model, alpha in [("Graph baseline", 0.55), ("Graph retained", 0.55)]:
        sub = pit.loc[pit["model"].eq(model)]
        axes[1].hist(
            sub["pit"].to_numpy(float),
            bins=bins,
            alpha=alpha,
            color=colors[model],
            edgecolor="white",
            linewidth=0.8,
            label=model,
        )
    axes[1].axhline(pit["patient_id"].nunique() / 10, color="#202020", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("PIT value")
    axes[1].set_ylabel("Patients")
    axes[1].grid(True, axis="y", color="#E6E6E6", linewidth=0.7)
    axes[1].legend(frameon=False, fontsize=8)

    fig.suptitle("T0-to-T3 reliability and PIT diagnostics", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "reliability_pit_t0t3.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "reliability_pit_t0t3.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_decision_curve_t3_burden_figure() -> None:
    import matplotlib.pyplot as plt

    df = decision_curve_t3_burden_table()
    colors = {0.1: "#1F77B4", 1.0: "#2A9D8F", 5.0: "#E76F51"}
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for threshold, sub in df.groupby("threshold_ml"):
        ax.plot(
            sub["probability_cutoff"].to_numpy(float),
            sub["net_benefit_model"].to_numpy(float),
            color=colors[float(threshold)],
            linewidth=1.8,
            label=f"FTV < {threshold:g} mL",
        )
    ax.axhline(0.0, color="#202020", linewidth=1.0, linestyle="--", label="treat none")
    ax.set_xlabel("Decision probability cutoff")
    ax.set_ylabel("Net benefit")
    ax.grid(True, color="#E6E6E6", linewidth=0.7)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Low residual MRI burden decision curves", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "decision_curve_t3_burden.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "decision_curve_t3_burden.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_uncertainty_sorted_figure() -> None:
    import matplotlib.pyplot as plt

    baseline = load_mc_per_patient(BASELINE_TAG).query("bucket == 'T0->T3'").copy()
    final = load_mc_per_patient(FINAL_RETAINED_TAG).query("bucket == 'T0->T3'").copy()
    merged = baseline.merge(
        final,
        on="patient_id",
        suffixes=("_baseline", "_final"),
    ).sort_values("obs_ftv_ml_final")
    x = np.arange(merged.shape[0])

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharex=True, sharey=True)
    panel_specs = [
        (
            axes[0],
            "Original graph rollout",
            "baseline",
            "15.26 mL MC MAE; 56.87 mL raw width",
            "#9CA3AF",
        ),
        (
            axes[1],
            "Hybrid-Edge k=8",
            "final",
            "5.56 mL MC MAE; 19.21 mL raw width",
            "#60A5FA",
        ),
    ]
    ymax = np.nanpercentile(
        merged[
            [
                "ftv_raw_p95_ml_baseline",
                "ftv_raw_p95_ml_final",
                "obs_ftv_ml_final",
            ]
        ].to_numpy(float),
        99,
    )
    ylim_top = max(float(ymax) * 1.05, 10.0)

    observed_color = "#009E73"
    mean_color = "#D55E00"
    for ax, title, suffix, metric_note, interval_color in panel_specs:
        lo = merged[f"ftv_raw_p05_ml_{suffix}"].to_numpy(float)
        hi = merged[f"ftv_raw_p95_ml_{suffix}"].to_numpy(float)
        mean = merged[f"ftv_mc_mean_ml_{suffix}"].to_numpy(float)
        obs = merged["obs_ftv_ml_final"].to_numpy(float)
        ax.fill_between(
            x,
            lo,
            hi,
            color=interval_color,
            alpha=0.24,
            linewidth=0,
            label="Raw 90% interval",
        )
        ax.plot(x, lo, color=interval_color, linewidth=0.9, alpha=0.95, label="Interval bounds")
        ax.plot(x, hi, color=interval_color, linewidth=0.9, alpha=0.95)
        ax.plot(
            x,
            mean,
            color=mean_color,
            linewidth=1.45,
            linestyle=(0, (1.5, 2.0)),
            label="MC mean",
        )
        ax.plot(
            x,
            obs,
            color=observed_color,
            linewidth=1.25,
            label="Observed T3",
        )
        ax.text(
            0.012,
            0.90,
            metric_note,
            transform=ax.transAxes,
            fontsize=8.5,
            ha="left",
            va="top",
            bbox=dict(facecolor="white", edgecolor="#BDBDBD", linewidth=0.6, alpha=0.92),
        )
        ax.set_title(title, fontsize=10.5, weight="bold", loc="left")
        ax.set_ylim(0, ylim_top)
        ax.set_xlabel("Patients sorted by observed T3 FTV")
        ax.grid(True, axis="y", color="#E6E6E6", linewidth=0.7)
    axes[0].set_ylabel("T3 FTV (mL)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("Cohort-wide T0-to-T3 MC uncertainty", fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIGURES / "mc_t0t3_patient_uncertainty_sorted.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _patient_t0_trajectory(df: pd.DataFrame, patient_id: str) -> pd.DataFrame:
    sub = df.loc[
        df["patient_id"].eq(patient_id)
        & df["start_visit"].eq("T0")
        & df["predicted_visit"].isin(["T1", "T2", "T3"])
    ].sort_values("pred_idx")
    if sub.empty:
        return pd.DataFrame()
    rows = [
        {
            "visit": "T0",
            "visit_idx": 0,
            "obs_ftv_ml": float(sub["ftv_t0_ml"].iloc[0]),
            "ftv_mc_mean_ml": float(sub["ftv_t0_ml"].iloc[0]),
            "ftv_mc_std_ml": 0.0,
        }
    ]
    for row in sub.itertuples(index=False):
        rows.append(
            {
                "visit": row.predicted_visit,
                "visit_idx": int(row.pred_idx),
                "obs_ftv_ml": float(row.obs_ftv_ml),
                "ftv_mc_mean_ml": float(row.ftv_mc_mean_ml),
                "ftv_mc_std_ml": float(row.ftv_mc_std_ml),
            }
        )
    return pd.DataFrame(rows)


def make_patient_trajectory_figure() -> None:
    import matplotlib.pyplot as plt

    baseline = load_mc_per_patient(BASELINE_TAG)
    final = load_mc_per_patient(FINAL_RETAINED_TAG)
    b_t3 = baseline.query("bucket == 'T0->T3'").copy()
    f_t3 = final.query("bucket == 'T0->T3'").copy()
    merged = b_t3.merge(
        f_t3,
        on="patient_id",
        suffixes=("_baseline", "_final"),
    )
    merged["baseline_abs_err"] = (merged["ftv_mc_mean_ml_baseline"] - merged["obs_ftv_ml_final"]).abs()
    merged["final_abs_err"] = (merged["ftv_mc_mean_ml_final"] - merged["obs_ftv_ml_final"]).abs()
    merged["improvement"] = merged["baseline_abs_err"] - merged["final_abs_err"]

    targets: list[tuple[str, str]] = []
    used: set[str] = set()
    specs = [
        ("low T3 burden", merged["obs_ftv_ml_final"].quantile(0.20), "obs_ftv_ml_final"),
        ("median T3 burden", merged["obs_ftv_ml_final"].quantile(0.50), "obs_ftv_ml_final"),
        ("high T3 burden", merged["obs_ftv_ml_final"].quantile(0.90), "obs_ftv_ml_final"),
    ]
    for label, target, col in specs:
        candidates = merged.loc[~merged["patient_id"].isin(used)].copy()
        idx = (candidates[col] - target).abs().idxmin()
        pid = str(candidates.loc[idx, "patient_id"])
        targets.append((label, pid))
        used.add(pid)
    candidates = merged.loc[~merged["patient_id"].isin(used)].copy()
    pid = str(candidates.sort_values("improvement", ascending=False)["patient_id"].iloc[0])
    targets.append(("large MC-mean improvement", pid))

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2), sharex=True)
    axes = axes.ravel()
    xticks = [0, 1, 2, 3]
    xticklabels = ["T0", "T1", "T2", "T3"]
    for ax, (reason, pid) in zip(axes, targets):
        b = _patient_t0_trajectory(baseline, pid)
        f = _patient_t0_trajectory(final, pid)
        if b.empty or f.empty:
            continue
        x = f["visit_idx"].to_numpy(float)
        obs = f["obs_ftv_ml"].to_numpy(float)
        for traj, color, fill, label in [
            (b, "#6B7280", "#D1D5DB", "Original MC mean +/- SD"),
            (f, "#1D4ED8", "#93C5FD", "Hybrid-Edge MC mean +/- SD"),
        ]:
            mean = traj["ftv_mc_mean_ml"].to_numpy(float)
            std = traj["ftv_mc_std_ml"].to_numpy(float)
            ax.fill_between(
                x,
                np.clip(mean - std, 0.0, None),
                mean + std,
                color=fill,
                alpha=0.30,
                linewidth=0,
            )
            ax.plot(x, mean, color=color, linewidth=1.8, marker="o", markersize=3.2, label=label)
        ax.plot(x, obs, color="#111827", linewidth=1.4, marker="D", markersize=3.4, label="Observed")
        ax.set_title(f"{reason}: {pid}", fontsize=9.5, weight="bold")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_ylabel("FTV (mL)")
        ax.grid(True, color="#E6E6E6", linewidth=0.7)
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("Patient-level T0-to-T3 MC trajectory uncertainty", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "mc_patient_trajectory_mean_std_examples.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_two_patient_rollout_static() -> None:
    import matplotlib.pyplot as plt

    # One I-SPY2 and one ACRIN held-out example with non-trivial T3 burden,
    # raw interval coverage, and strong deterministic agreement. The aggregate
    # tables carry the cohort-level performance claims; this panel is a visual
    # example of successful graph-to-endpoint rollout behavior.
    patient_ids = ["ISPY2-910706", "ACRIN-6698-734997"]
    latest_root = REPO / f"reports/edge_attr_meaning_breast_mc/{LATEST_RETAINED_TAG}"
    per_patient = pd.read_parquet(latest_root / "conditional_mc_per_patient.parquet").copy()
    per_patient["bucket"] = per_patient["start_visit"].astype(str) + "->" + per_patient["predicted_visit"].astype(str)
    samples_path = latest_root / "conditional_mc_samples.parquet"
    samples = pd.read_parquet(samples_path)

    rows = []
    for pid in patient_ids:
        row = per_patient[
            per_patient["patient_id"].eq(pid)
            & per_patient["start_visit"].eq("T0")
            & per_patient["predicted_visit"].eq("T3")
        ]
        if not row.empty:
            rows.append(row.iloc[0])
    if not rows:
        return

    fig, axes = plt.subplots(len(rows), 2, figsize=(8.4, 2.85 * len(rows)))
    if len(rows) == 1:
        axes = np.asarray([axes])
    for row_idx, row in enumerate(rows):
        pid = str(row["patient_id"])
        draws = samples[
            samples["patient_id"].eq(pid)
            & samples["start_visit"].eq("T0")
            & samples["predicted_visit"].eq("T3")
        ]["ftv_sample_ml"].to_numpy(float)
        ax_bar, ax_hist = axes[row_idx]

        labels = ["Observed T0", "Predicted T3", "Observed T3"]
        values = [float(row["ftv_t0_ml"]), float(row["pred_ftv_det_ml"]), float(row["obs_ftv_ml"])]
        ax_bar.bar(labels, values, color=["#7A869A", "#1F77B4", "#2A9D8F"], width=0.6)
        ax_bar.set_ylabel("FTV (mL)")
        ax_bar.set_title(pid, fontsize=10, weight="bold")
        ax_bar.grid(True, axis="y", color="#E6E6E6", linewidth=0.7)
        for tick in ax_bar.get_xticklabels():
            tick.set_rotation(12)
            tick.set_ha("right")

        x_max = max(float(np.nanpercentile(draws, 98)), float(row["obs_ftv_ml"]) * 1.4, 5.0)
        draws_clip = draws[(draws >= 0) & (draws <= x_max)]
        ax_hist.hist(draws_clip, bins=24, color="#bfdbfe", edgecolor="#60a5fa", linewidth=0.6, density=True)
        ax_hist.axvspan(float(row["ftv_raw_p05_ml"]), float(row["ftv_raw_p95_ml"]), color="#93c5fd", alpha=0.25, lw=0)
        ax_hist.axvline(float(row["pred_ftv_det_ml"]), color="#1F77B4", linewidth=1.8, label="det. pred.")
        ax_hist.axvline(float(row["obs_ftv_ml"]), color="#2A9D8F", linewidth=1.8, label="observed")
        ax_hist.set_title(
            f"MC mean {row['ftv_mc_mean_ml']:.1f} mL, raw 90% {row['ftv_raw_p05_ml']:.1f}-{row['ftv_raw_p95_ml']:.1f}",
            fontsize=10,
        )
        ax_hist.set_xlabel("T3 FTV sample (mL)")
        ax_hist.set_yticks([])
        ax_hist.grid(True, axis="x", color="#E6E6E6", linewidth=0.7)
        ax_hist.legend(frameon=False, fontsize=8)
    fig.suptitle("Illustrative Hybrid-Edge k=8 T0-to-T3 FTV forecasts", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "ispy2_two_patient_rollout_static.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "ispy2_two_patient_rollout_static.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    sync_report_figures()
    related_work_positioning_table().to_csv(TABLES / "related_work_positioning.csv", index=False)
    bucket_calibration_summary().to_csv(TABLES / "calibration_by_bucket.csv", index=False)
    calibration_subgroup_robustness_table().to_csv(TABLES / "calibration_subgroup_robustness.csv", index=False)
    retained_full_rollout_subtype_calibration_table().to_csv(TABLES / "retained_full_rollout_subtype_calibration.csv", index=False)
    calibration_by_subtype_full_table().to_csv(TABLES / "calibration_by_subtype_full.csv", index=False)
    calibration_by_t3_burden_quartile_table().to_csv(TABLES / "calibration_by_t3_burden_quartile.csv", index=False)
    calibration_by_t3_burden_tail_table().to_csv(TABLES / "calibration_by_t3_burden_tail.csv", index=False)
    coverage_vs_nominal_table().to_csv(TABLES / "coverage_vs_nominal.csv", index=False)
    pit_t0t3_table().to_csv(TABLES / "pit_t0t3.csv", index=False)
    imaging_burden_threshold_metrics_t3().to_csv(TABLES / "imaging_burden_threshold_metrics_t3.csv", index=False)
    decision_curve_t3_burden_table().to_csv(TABLES / "decision_curve_t3_burden.csv", index=False)
    low_residual_burden_readout().to_csv(TABLES / "low_residual_burden_readout.csv", index=False)
    scalar_baseline_t3_table().to_csv(TABLES / "scalar_baseline_t3.csv", index=False)
    strong_scalar_baselines_t3_table().to_csv(TABLES / "strong_scalar_baselines_t3.csv", index=False)
    center_prediction_t3_table().to_csv(TABLES / "hybrid_center_t3.csv", index=False)
    scalar_residual_mc_t3_table().to_csv(TABLES / "scalar_residual_mc_t3.csv", index=False)
    strong_scalar_baselines_mc_t3_table().to_csv(TABLES / "strong_scalar_baselines_mc_t3.csv", index=False)
    hybrid_residual_mc_t3_table().to_csv(TABLES / "hybrid_residual_mc_t3.csv", index=False)
    scalar_vs_graph_mc_t3_table().to_csv(TABLES / "scalar_vs_graph_mc_t3.csv", index=False)
    split_leakage_audit_table().to_csv(TABLES / "split_leakage_audit.csv", index=False)
    paired_delta_table().to_csv(TABLES / "paired_bootstrap_deltas.csv", index=False)
    model_mean_ci_table().to_csv(TABLES / "model_mean_bootstrap_ci.csv", index=False)
    ablation_t0t3_table().to_csv(TABLES / "ablation_t0t3_mc.csv", index=False)
    subtype_t0t3_table().to_csv(TABLES / "subtype_t0t3_mc.csv", index=False)
    make_pred_obs_figure()
    make_scalar_baseline_figure()
    make_mc_diagnostic_figure()
    make_scalar_vs_graph_mc_figure()
    make_calibration_by_burden_figure()
    make_reliability_pit_figure()
    make_decision_curve_t3_burden_figure()
    make_uncertainty_sorted_figure()
    make_patient_trajectory_figure()
    make_two_patient_rollout_static()


if __name__ == "__main__":
    main()
