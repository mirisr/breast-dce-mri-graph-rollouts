#!/usr/bin/env python3
"""Generate Stage-3 paper figures from consistent-graph forecaster results.

Produces three PNG figures:
  1. fig_stage3_training_curves.png        -- 5-fold val MAE & fold-0 loss gap
  2. fig_stage3_pred_vs_actual.png         -- 3D scatter: predicted vs actual T3
  3. fig_stage3_displacement_error_dist.png -- error histogram & per-fold MAE bars
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lsgc.forecaster import LSGCForecaster
from experiments.stage1_forecaster.train_consistent_forecaster import (
    load_sample, forward_patient, ConsistentSample,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_FOLDS = 5
RUNS_DIR = "runs/consistent_forecaster_5fold"


# ── Helpers ──────────────────────────────────────────────────────────────

def load_model(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    in_ch = int(ckpt["state_dict"]["embed.weight"].shape[1])
    hidden = int(cfg.get("hidden", ckpt["state_dict"]["embed.weight"].shape[0]))
    n_layers = int(cfg.get("num_layers", 2))

    model = LSGCForecaster(
        in_channels=in_ch, hidden=hidden, num_layers=n_layers,
        feat_out_dim=in_ch, use_delta_t=True,
        use_edge_gating=True, edge_attr_dim=0,
    ).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["mean"].to(DEVICE), ckpt["std"].to(DEVICE), ckpt


def load_fold_val_samples(
    graphs_root: Path, folds_parquet: Path, fold: int,
) -> list[ConsistentSample]:
    df = pd.read_parquet(folds_parquet)
    pids = df.loc[df["fold"] == fold, "patient_id"].tolist()
    samples = []
    for pid in pids:
        s = load_sample(graphs_root / f"{pid}.pt")
        if s is not None:
            samples.append(s)
    return samples


def zero_baseline_mae(samples: list[ConsistentSample]) -> float:
    """MAE (mm) when predicting zero displacement for every supervoxel."""
    maes: list[float] = []
    for s in samples:
        off = s.visit_offsets
        for v in range(len(off) - 2):
            sl_s = slice(int(off[v]), int(off[v + 1]))
            sl_d = slice(int(off[v + 1]), int(off[v + 2]))
            mask = s.alive[sl_d].bool()
            if mask.sum() == 0:
                continue
            delta = (s.pos[sl_d] - s.pos[sl_s])[mask]
            maes.append(delta.norm(dim=-1).mean().item())
    return float(np.mean(maes)) if maes else float("nan")


@torch.no_grad()
def predict_cumulative_t3(
    model: LSGCForecaster,
    s: ConsistentSample,
    mean: torch.Tensor,
    std: torch.Tensor,
):
    """Accumulate predicted deltas from T0 to obtain predicted T3 positions."""
    x = (s.x.to(DEVICE) - mean) / std
    pos = s.pos.to(DEVICE)
    t = s.t.to(DEVICE)
    ei = s.edge_index.to(DEVICE)

    out = model(x, pos, t, ei)
    off = s.visit_offsets

    pos_pred = pos[int(off[0]):int(off[1])].clone()
    for v in range(len(off) - 2):  # 0→1, 1→2, 2→3
        sl = slice(int(off[v]), int(off[v + 1]))
        pos_pred = pos_pred + out["delta_pos"][sl]

    sl_t3 = slice(int(off[-2]), int(off[-1]))
    pos_actual = pos[sl_t3]
    alive_t3 = s.alive[sl_t3].bool()
    return pos_pred.cpu(), pos_actual.cpu(), alive_t3.cpu()


@torch.no_grad()
def per_supervoxel_errors(
    model: LSGCForecaster,
    samples: list[ConsistentSample],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> np.ndarray:
    """Per-supervoxel displacement prediction error norms (mm)."""
    all_err: list[np.ndarray] = []
    for s in samples:
        x = (s.x.to(DEVICE) - mean) / std
        pos = s.pos.to(DEVICE)
        t = s.t.to(DEVICE)
        ei = s.edge_index.to(DEVICE)
        alive = s.alive.to(DEVICE)

        out = model(x, pos, t, ei)
        off = s.visit_offsets
        for v in range(len(off) - 2):
            sl_s = slice(int(off[v]), int(off[v + 1]))
            sl_d = slice(int(off[v + 1]), int(off[v + 2]))
            mask = alive[sl_d].bool()
            if mask.sum() == 0:
                continue
            delta_gt = (pos[sl_d] - pos[sl_s])[mask]
            delta_pred = out["delta_pos"][sl_s][mask]
            all_err.append((delta_pred - delta_gt).norm(dim=-1).cpu().numpy())
    return np.concatenate(all_err) if all_err else np.array([])


# ── Figure 1: Training Curves ───────────────────────────────────────────

def fig_training_curves(
    runs_root: Path,
    graphs_root: Path,
    folds_parquet: Path,
    out_path: Path,
):
    print("Figure 1: training curves …")
    fig, (ax_mae, ax_loss) = plt.subplots(1, 2, figsize=(10, 4))

    styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    cmap = plt.cm.tab10.colors

    for fold in range(N_FOLDS):
        log_path = runs_root / f"fold{fold}" / "train_log.json"
        if not log_path.exists():
            print(f"  ⚠  {log_path} not found — skipping fold {fold}")
            continue

        log = json.loads(log_path.read_text())
        epochs = [e["epoch"] for e in log]
        val_mae = [e["val_mae_mm"] for e in log]
        best_idx = int(np.argmin([e["val_loss"] for e in log]))

        col = cmap[fold]
        ax_mae.plot(
            epochs, val_mae,
            linestyle=styles[fold], color=col,
            label=f"Fold {fold}", linewidth=1.2,
        )
        ax_mae.plot(
            epochs[best_idx], val_mae[best_idx], "o",
            color=col, markersize=6, zorder=5,
        )

        val_samples = load_fold_val_samples(graphs_root, folds_parquet, fold)
        baseline = zero_baseline_mae(val_samples)
        ax_mae.axhline(
            baseline, color=col, linestyle=":", alpha=0.4, linewidth=0.8,
        )

    ax_mae.set_xlabel("Epoch")
    ax_mae.set_ylabel("Validation MAE (mm)")
    ax_mae.legend(fontsize=8, frameon=False)
    ax_mae.set_title("Per-Fold Validation MAE")
    ax_mae.spines[["top", "right"]].set_visible(False)

    # Right panel: fold 0 train vs val loss
    log0_path = runs_root / "fold0" / "train_log.json"
    if log0_path.exists():
        log0 = json.loads(log0_path.read_text())
        ep = [e["epoch"] for e in log0]
        ax_loss.plot(ep, [e["train_loss"] for e in log0],
                     label="Train loss", linewidth=1.2)
        ax_loss.plot(ep, [e["val_loss"] for e in log0],
                     label="Val loss", linewidth=1.2)
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend(fontsize=8, frameon=False)
        ax_loss.set_title("Fold 0: Train vs Val Loss")
        ax_loss.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ── Figure 2: Predicted vs Actual 3-D Scatter ───────────────────────────

def fig_pred_vs_actual(
    runs_root: Path,
    graphs_root: Path,
    folds_parquet: Path,
    out_path: Path,
):
    print("Figure 2: predicted vs actual T3 positions …")

    model, mean, std, _ = load_model(runs_root / "fold0" / "best.pt")
    val_samples = load_fold_val_samples(graphs_root, folds_parquet, fold=0)
    val_samples.sort(key=lambda s: s.n_sv)

    n = len(val_samples)
    n_panels = min(6, n)
    if n >= 6:
        indices = np.linspace(0, n - 1, 6, dtype=int)
    else:
        indices = list(range(n))
    selected = [val_samples[i] for i in indices]

    nrows = 2 if n_panels > 3 else 1
    ncols = min(n_panels, 3)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 5 * nrows),
        subplot_kw={"projection": "3d"},
    )
    if n_panels == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()

    for idx, (ax, s) in enumerate(zip(axes, selected)):
        pred, actual, alive = predict_cumulative_t3(model, s, mean, std)
        mask = alive.numpy()
        p = pred.numpy()[mask]
        a = actual.numpy()[mask]

        ax.scatter(a[:, 0], a[:, 1], a[:, 2],
                   c="#2196F3", s=8, alpha=0.6, label="Actual")
        ax.scatter(p[:, 0], p[:, 1], p[:, 2],
                   c="#FF5722", s=8, alpha=0.6, label="Predicted")

        ax.set_title(s.patient_id, fontsize=9)
        ax.view_init(elev=25, azim=135)
        ax.set_xlabel("X", fontsize=7, labelpad=1)
        ax.set_ylabel("Y", fontsize=7, labelpad=1)
        ax.set_zlabel("Z", fontsize=7, labelpad=1)
        ax.tick_params(labelsize=6)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left", frameon=False)

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    fig.suptitle(
        "Predicted vs Actual Supervoxel Positions at T3",
        fontsize=12, y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ── Figure 3: Error Distribution + Per-Fold MAE Bars ────────────────────

def fig_error_distribution(
    runs_root: Path,
    graphs_root: Path,
    folds_parquet: Path,
    out_path: Path,
):
    print("Figure 3: displacement error distribution …")
    fig, (ax_hist, ax_bar) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: per-supervoxel error histogram (fold 0)
    model, mean, std, _ = load_model(runs_root / "fold0" / "best.pt")
    val_f0 = load_fold_val_samples(graphs_root, folds_parquet, fold=0)
    errors = per_supervoxel_errors(model, val_f0, mean, std)

    ax_hist.hist(
        errors, bins=50, color="#4CAF50", alpha=0.7,
        edgecolor="white", linewidth=0.5,
    )
    mean_err = errors.mean()
    med_err = float(np.median(errors))
    ax_hist.axvline(
        mean_err, color="#D32F2F", linestyle="--", linewidth=1.2,
        label=f"Mean = {mean_err:.2f} mm",
    )
    ax_hist.axvline(
        med_err, color="#1976D2", linestyle="-.", linewidth=1.2,
        label=f"Median = {med_err:.2f} mm",
    )
    ax_hist.set_xlabel("Displacement Error (mm)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Per-Supervoxel Error Distribution (Fold 0)")
    ax_hist.legend(fontsize=8, frameon=False)
    ax_hist.spines[["top", "right"]].set_visible(False)

    # Right: per-fold model MAE vs zero-baseline
    fold_maes: list[float] = []
    baselines: list[float] = []
    for fold in range(N_FOLDS):
        ckpt_f = runs_root / f"fold{fold}" / "best.pt"
        if not ckpt_f.exists():
            fold_maes.append(float("nan"))
            baselines.append(float("nan"))
            continue
        ck = torch.load(ckpt_f, map_location="cpu", weights_only=False)
        fold_maes.append(float(ck.get("val_mae_mm", float("nan"))))

        val_s = load_fold_val_samples(graphs_root, folds_parquet, fold)
        baselines.append(zero_baseline_mae(val_s))

    x_pos = np.arange(N_FOLDS)
    w = 0.35
    ax_bar.bar(x_pos - w / 2, fold_maes, w, color="#2196F3", label="Model MAE")
    ax_bar.bar(x_pos + w / 2, baselines, w, color="#BDBDBD", label="Zero Baseline")

    for i in range(N_FOLDS):
        m, b = fold_maes[i], baselines[i]
        if np.isfinite(m) and np.isfinite(b) and b > 0:
            pct = (1 - m / b) * 100
            ax_bar.text(
                x_pos[i] - w / 2, m + 0.05,
                f"{pct:.0f}%\u2193",
                ha="center", va="bottom", fontsize=7, color="#1565C0",
            )

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([f"Fold {i}" for i in range(N_FOLDS)])
    ax_bar.set_ylabel("MAE (mm)")
    ax_bar.set_title("Per-Fold Model vs Zero-Baseline MAE")
    ax_bar.legend(fontsize=8, frameon=False)
    ax_bar.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate Stage-3 paper figures")
    ap.add_argument("--out-dir", default="paper/figures",
                    help="Directory for output PNGs")
    ap.add_argument("--graphs-root", default="datasets/ispy2/graphs_consistent",
                    help="Root directory of consistent .pt graph files")
    ap.add_argument("--runs-root", default=RUNS_DIR,
                    help="Root of 5-fold run directories")
    ap.add_argument("--folds-parquet", default="datasets/ispy2/folds.parquet",
                    help="Parquet file with patient_id / fold columns")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    graphs_root = Path(args.graphs_root)
    runs_root = Path(args.runs_root)
    folds_pq = Path(args.folds_parquet)

    fig_training_curves(
        runs_root, graphs_root, folds_pq,
        out_dir / "fig_stage3_training_curves.png",
    )
    fig_pred_vs_actual(
        runs_root, graphs_root, folds_pq,
        out_dir / "fig_stage3_pred_vs_actual.png",
    )
    fig_error_distribution(
        runs_root, graphs_root, folds_pq,
        out_dir / "fig_stage3_displacement_error_dist.png",
    )

    print("\nDone — all figures saved to", out_dir)


if __name__ == "__main__":
    main()
