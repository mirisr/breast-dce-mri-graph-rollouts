from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = Path(__file__).resolve().parent
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"

BASELINE_TAG = "v2_sched_samp_baseline"
RETAINED_TAG = "bio_ftv020_alive005"
ABLATION_TAGS = [
    BASELINE_TAG,
    RETAINED_TAG,
    "bio_ftv010_alive000",
    "bio_ftv010_alive002",
]


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


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


def load_mc_per_patient(tag: str) -> pd.DataFrame:
    if tag == BASELINE_TAG:
        path = ROOT / "results/conditional_mc_consistent_rollout/conditional_mc_per_patient.parquet"
    else:
        path = ROOT / f"results/conditional_mc_bio_retrained/{tag}/conditional_mc_per_patient.parquet"
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


def paired_delta_table() -> pd.DataFrame:
    base = load_mc_per_patient(BASELINE_TAG)
    retained = load_mc_per_patient(RETAINED_TAG)
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
        (RETAINED_TAG, "Retained"),
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
    path = ROOT / "results/consistent_forecaster_v2_bio_eval/notebook_exports/derived_notebook_tables/mc_t0t3_model_comparison.csv"
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
    retained = load_mc_per_patient(RETAINED_TAG).query("bucket == 'T0->T3'")
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


def make_pred_obs_figure() -> None:
    import matplotlib.pyplot as plt

    path = ROOT / "results/consistent_forecaster_v2_bio_eval/notebook_exports/bio_vs_baseline_t3_per_patient.csv"
    df = pd.read_csv(path)
    keep = {
        BASELINE_TAG: "Baseline",
        RETAINED_TAG: "Retained",
    }
    plot_df = df.loc[df["tag"].isin(keep)].copy()
    xmax = np.nanpercentile(plot_df[["obs_ftv_ml", "pred_ftv_ml"]].to_numpy().reshape(-1), 99.5)
    xmax = max(80.0, float(xmax))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), sharex=True, sharey=True)
    for ax, (tag, label) in zip(axes, keep.items()):
        sub = plot_df.loc[plot_df["tag"] == tag]
        x = sub["obs_ftv_ml"].to_numpy(float)
        y = sub["pred_ftv_ml"].to_numpy(float)
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


def main() -> None:
    ensure_dirs()
    paired_delta_table().to_csv(TABLES / "paired_bootstrap_deltas.csv", index=False)
    model_mean_ci_table().to_csv(TABLES / "model_mean_bootstrap_ci.csv", index=False)
    ablation_t0t3_table().to_csv(TABLES / "ablation_t0t3_mc.csv", index=False)
    subtype_t0t3_table().to_csv(TABLES / "subtype_t0t3_mc.csv", index=False)
    make_pred_obs_figure()


if __name__ == "__main__":
    main()
