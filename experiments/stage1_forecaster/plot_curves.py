#!/usr/bin/env python
"""Plot cross-fold learning curves for Stage-1 forecaster.

Reads ``*/fold*/train_log.csv`` under ``--runs-root`` and saves a
figure with one panel per validation metric, colored by match method
(mean line + shaded per-fold envelope).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


METRICS = [
    ("val_pos_l1_mm", "position L1 (mm)", False),
    ("val_position_emd_mm", "position EMD (mm)", False),
    ("val_feat_mse", "feature MSE", False),
    ("val_alive_auc", "alive AUC", True),
]

COLORS = {"sinkhorn": "tab:orange", "nn": "tab:blue"}


def _load_group(root: Path):
    groups: dict[str, list[pd.DataFrame]] = {}
    for group_dir in sorted(root.glob("*_L*")):
        method = group_dir.name.split("_")[0]
        dfs = []
        for fold_dir in sorted(group_dir.glob("fold*")):
            csv = fold_dir / "train_log.csv"
            if csv.exists():
                dfs.append(pd.read_csv(csv))
        if dfs:
            groups.setdefault(method, []).extend(dfs)
    return groups


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path,
                    default=Path("experiments/stage1_forecaster/runs"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    groups = _load_group(args.runs_root)
    if not groups:
        raise SystemExit(f"no train_log.csv under {args.runs_root}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, (col, label, higher_better) in zip(axes.ravel(), METRICS):
        for method, dfs in groups.items():
            # Align on epoch; some folds may stop at different epochs.
            n_max = min(len(df) for df in dfs)
            mat = np.stack([df[col].to_numpy()[:n_max] for df in dfs], axis=0)
            x = np.arange(n_max)
            mean = mat.mean(axis=0)
            lo = mat.min(axis=0); hi = mat.max(axis=0)
            c = COLORS.get(method, None)
            ax.plot(x, mean, label=method, color=c, lw=2)
            ax.fill_between(x, lo, hi, color=c, alpha=0.15)
        ax.set_title(label)
        ax.set_xlabel("epoch"); ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Stage-1 LSGC-Forecaster — val metrics (5-fold mean ± min/max)", y=1.01)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
