#!/usr/bin/env python3
"""Summarize clinical burden-monitoring readouts from retained MC rollouts."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def auc_rank(y_true: pd.Series | np.ndarray, score: pd.Series | np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score).astype(float)
    keep = np.isfinite(s)
    y = y[keep]
    s = s[keep]
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s)
    sorted_scores = s[order]
    ranks = np.empty_like(sorted_scores, dtype=float)
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[i:j] = (i + 1 + j) / 2.0
        i = j

    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    rank_sum_pos = original_ranks[y == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true).astype(bool)
    p = np.asarray(y_pred).astype(bool)
    tp = int((p & y).sum())
    fp = int((p & ~y).sum())
    tn = int((~p & ~y).sum())
    fn = int((~p & y).sum())
    return {
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "ppv": tp / (tp + fp) if tp + fp else float("nan"),
        "npv": tn / (tn + fn) if tn + fn else float("nan"),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def markdown_table(frame: pd.DataFrame) -> str:
    text = frame.copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        else:
            text[col] = text[col].astype(str)
    headers = list(text.columns)
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in text.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(rows)


def observed_visit_table(per_patient: pd.DataFrame) -> pd.DataFrame:
    t3_rows = per_patient[per_patient["predicted_visit"].eq("T3")]
    records: list[dict[str, object]] = []
    for _, row in t3_rows[["patient_id", "ftv_t0_ml"]].drop_duplicates("patient_id").iterrows():
        records.append(
            {
                "patient_id": row["patient_id"],
                "visit": "T0",
                "observed_visit_ftv_ml": float(row["ftv_t0_ml"]),
            }
        )
    for visit in ["T1", "T2", "T3"]:
        tmp = per_patient[
            per_patient["start_visit"].eq("T0")
            & per_patient["predicted_visit"].eq(visit)
        ][["patient_id", "obs_ftv_ml"]]
        for _, row in tmp.iterrows():
            records.append(
                {
                    "patient_id": row["patient_id"],
                    "visit": visit,
                    "observed_visit_ftv_ml": float(row["obs_ftv_ml"]),
                }
            )
    return pd.DataFrame(records).drop_duplicates(["patient_id", "visit"])


def build_analysis(per_patient: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    probs = (
        samples[samples["predicted_visit"].eq("T3")]
        .groupby(["patient_id", "start_visit"])
        .agg(
            prob_t3_ftv_lt5=("ftv_sample_ml", lambda x: float(np.mean(x < 5.0))),
            prob_t3_ftv_gt20=("ftv_sample_ml", lambda x: float(np.mean(x > 20.0))),
            prob_t3_ftv_gt50=("ftv_sample_ml", lambda x: float(np.mean(x > 50.0))),
            n_draws=("ftv_sample_ml", "size"),
        )
        .reset_index()
    )
    obs_visits = observed_visit_table(per_patient)
    t3 = per_patient[per_patient["predicted_visit"].eq("T3")].merge(
        probs, on=["patient_id", "start_visit"], how="left"
    )
    t3 = t3.merge(
        obs_visits.rename(
            columns={"visit": "start_visit", "observed_visit_ftv_ml": "last_observed_ftv_ml"}
        ),
        on=["patient_id", "start_visit"],
        how="left",
    )
    t3["percent_reduction_from_t0"] = (
        t3["ftv_t0_ml"] - t3["last_observed_ftv_ml"]
    ) / t3["ftv_t0_ml"].clip(lower=1e-6)
    t3["event_t3_ftv_lt5"] = t3["obs_ftv_ml"] < 5.0
    t3["event_t3_ftv_gt20"] = t3["obs_ftv_ml"] > 20.0
    t3["event_t3_ftv_gt50"] = t3["obs_ftv_ml"] > 50.0
    return t3


def clinical_threshold_table(t3: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    t0 = t3[t3["start_visit"].eq("T0")].copy()
    definitions = [
        (
            "Low residual MRI burden",
            "T3 FTV < 5 mL",
            t0["event_t3_ftv_lt5"],
            "MC probability P(T3 FTV < 5 mL)",
            t0["prob_t3_ftv_lt5"],
            t0["prob_t3_ftv_lt5"] >= 0.5,
        ),
        (
            "Low residual MRI burden",
            "T3 FTV < 5 mL",
            t0["event_t3_ftv_lt5"],
            "Last observed FTV < 5 mL",
            -t0["last_observed_ftv_ml"],
            t0["last_observed_ftv_ml"] < 5.0,
        ),
        (
            "High residual MRI burden",
            "T3 FTV > 20 mL",
            t0["event_t3_ftv_gt20"],
            "MC probability P(T3 FTV > 20 mL)",
            t0["prob_t3_ftv_gt20"],
            t0["prob_t3_ftv_gt20"] >= 0.5,
        ),
        (
            "High residual MRI burden",
            "T3 FTV > 20 mL",
            t0["event_t3_ftv_gt20"],
            "Last observed FTV > 20 mL",
            t0["last_observed_ftv_ml"],
            t0["last_observed_ftv_ml"] > 20.0,
        ),
    ]
    for readout, endpoint, event, score_label, score, pred in definitions:
        row = {
            "readout": readout,
            "endpoint": endpoint,
            "score": score_label,
            "n": int(len(t0)),
            "event_rate": float(np.mean(event)),
            "auc": auc_rank(event, score),
        }
        row.update(binary_metrics(event, pred))
        rows.append(row)
    return pd.DataFrame(rows)


def monitoring_table(t3: pd.DataFrame) -> pd.DataFrame:
    labels = {"T0": "T0 only", "T1": "T0+T1", "T2": "T0+T1+T2"}
    rows: list[dict[str, object]] = []
    for start_visit, group in t3.groupby("start_visit", sort=True):
        rows.append(
            {
                "conditioning": labels.get(start_visit, start_visit),
                "n": int(len(group)),
                "det_ftv_mae_ml": float(np.mean(np.abs(group["pred_ftv_det_ml"] - group["obs_ftv_ml"]))),
                "mc_mean_ftv_mae_ml": float(np.mean(np.abs(group["ftv_mc_mean_ml"] - group["obs_ftv_ml"]))),
                "mc_median_ftv_mae_ml": float(np.mean(np.abs(group["ftv_mc_median_ml"] - group["obs_ftv_ml"]))),
                "raw_90_coverage": float(group["coverage90_ftv_raw"].mean()),
                "raw_90_width_ml": float(group["ftv_raw_width90_ml"].mean()),
                "crps": float(group["crps_ftv"].mean()),
                "auc_t3_ftv_lt5": auc_rank(group["event_t3_ftv_lt5"], group["prob_t3_ftv_lt5"]),
                "auc_t3_ftv_gt20": auc_rank(group["event_t3_ftv_gt20"], group["prob_t3_ftv_gt20"]),
                "mean_prob_t3_ftv_lt5": float(group["prob_t3_ftv_lt5"].mean()),
                "mean_prob_t3_ftv_gt20": float(group["prob_t3_ftv_gt20"].mean()),
            }
        )
    return pd.DataFrame(rows)


def make_plot(monitoring: pd.DataFrame, out_pdf: Path, out_png: Path) -> None:
    x = np.arange(len(monitoring))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))

    axes[0].plot(x, monitoring["auc_t3_ftv_lt5"], marker="o", color="#1f77b4", label="FTV < 5 mL")
    axes[0].plot(x, monitoring["auc_t3_ftv_gt20"], marker="s", color="#d62728", label="FTV > 20 mL")
    axes[0].set_xticks(x, monitoring["conditioning"])
    axes[0].set_ylim(0.90, 1.00)
    axes[0].set_ylabel("AUC")
    axes[0].set_title("Residual-burden discrimination")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(x - 0.18, monitoring["mc_mean_ftv_mae_ml"], width=0.36, color="#4c78a8", label="MC-mean MAE")
    axes[1].bar(x + 0.18, monitoring["raw_90_width_ml"], width=0.36, color="#f58518", label="90% width")
    axes[1].set_xticks(x, monitoring["conditioning"])
    axes[1].set_ylabel("mL")
    axes[1].set_title("Endpoint error and uncertainty width")
    axes[1].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mc-dir",
        type=Path,
        default=Path("reports/edge_attr_meaning_breast_mc/hybrid_a50_bio_k8"),
    )
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=Path("paper/bio_ftv020_mc_manuscript"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/bio_ftv_clinical_burden_monitoring"),
    )
    args = parser.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.paper_dir / "tables").mkdir(parents=True, exist_ok=True)
    (args.paper_dir / "figures").mkdir(parents=True, exist_ok=True)

    per_patient = pd.read_parquet(args.mc_dir / "conditional_mc_per_patient.parquet")
    samples = pd.read_parquet(args.mc_dir / "conditional_mc_samples.parquet")
    t3 = build_analysis(per_patient, samples)
    threshold = clinical_threshold_table(t3)
    monitoring = monitoring_table(t3)

    for out_dir in [args.report_dir, args.paper_dir / "tables"]:
        threshold.to_csv(out_dir / "clinical_burden_threshold_readouts.csv", index=False)
        monitoring.to_csv(out_dir / "clinical_burden_monitoring_by_visit.csv", index=False)
    t3.to_parquet(args.report_dir / "clinical_burden_patient_scores.parquet", index=False)

    make_plot(
        monitoring,
        args.paper_dir / "figures" / "clinical_burden_monitoring_by_visit.pdf",
        args.paper_dir / "figures" / "clinical_burden_monitoring_by_visit.png",
    )

    lines = [
        "# Clinical Burden Monitoring Summary",
        "",
        "This analysis uses the retained Hybrid-Edge k=8 residual MC rollout and treats pCR only as exploratory.",
        "",
        "## Threshold Readouts",
        "",
        markdown_table(threshold),
        "",
        "## Serial Monitoring",
        "",
        markdown_table(monitoring),
        "",
    ]
    (args.report_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.report_dir}")
    print(threshold.round(3).to_string(index=False))
    print(monitoring.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
