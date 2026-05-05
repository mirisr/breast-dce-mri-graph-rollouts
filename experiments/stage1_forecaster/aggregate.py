#!/usr/bin/env python
"""Aggregate Stage-1 forecaster metrics across folds and match methods.

Produces a tidy parquet (``summary.parquet``) with one row per
(match_method, fold) plus a human-readable ``summary.md`` that
includes the cross-fold means and a Sinkhorn-vs-NN comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _collect(runs_root: Path) -> pd.DataFrame:
    rows = []
    for fold_dir in sorted(runs_root.glob("*/fold*")):
        mj = fold_dir / "metrics.json"
        if not mj.exists():
            continue
        data = json.loads(mj.read_text())
        bv = data.get("best_val") or data.get("final_val") or {}
        cfg = data.get("config", {})
        row = {
            "group": fold_dir.parent.name,       # e.g. sinkhorn_L2_skips-1-2-3
            "match": cfg.get("config", cfg).get("temporal_skip_hops", None),
            "fold": int(cfg.get("fold", -1)),
            "best_epoch": int(data.get("best_epoch", -1)),
            "num_layers": cfg.get("num_layers"),
            "hidden": cfg.get("hidden"),
            "temporal_skip_hops": "-".join(str(h) for h in cfg.get("temporal_skip_hops", [])),
            "use_delta_t": cfg.get("use_delta_t"),
            "pos_l1_mm": bv.get("pos_l1_mm"),
            "feat_mse": bv.get("feat_mse"),
            "position_emd_mm": bv.get("position_emd_mm"),
            "alive_auc": bv.get("alive_auc"),
            "n_val": bv.get("n_patients"),
            "wall_s": data.get("wall_s"),
        }
        # Derive match method from the group name prefix.
        row["match_method"] = fold_dir.parent.name.split("_")[0]
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt_mean_std(xs: Iterable[float], fmt: str = "{:.3f}") -> str:
    arr = np.array(list(xs), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return "nan"
    if arr.size == 1:
        return fmt.format(arr[0])
    return f"{fmt.format(arr.mean())} ± {fmt.format(arr.std(ddof=1))}"


def _summary_md(df: pd.DataFrame) -> str:
    lines = ["# Stage-1 Forecaster — cross-fold summary", ""]
    grp = df.groupby("match_method")
    lines.append("## Per-method mean ± std across folds")
    lines.append("")
    lines.append("| match_method | pos_l1 (mm) | feat MSE | position EMD (mm) | alive AUC |")
    lines.append("|---|---:|---:|---:|---:|")
    for m, sub in grp:
        lines.append(
            f"| {m} | {_fmt_mean_std(sub['pos_l1_mm'], '{:.2f}')} | "
            f"{_fmt_mean_std(sub['feat_mse'], '{:.1f}')} | "
            f"{_fmt_mean_std(sub['position_emd_mm'], '{:.2f}')} | "
            f"{_fmt_mean_std(sub['alive_auc'], '{:.3f}')} |"
        )
    lines.append("")
    lines.append("## Per-fold detail")
    lines.append("")
    lines.append("| match | fold | best_epoch | pos_l1 | feat_mse | emd | alive_auc |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in df.sort_values(["match_method", "fold"]).iterrows():
        lines.append(
            f"| {r['match_method']} | {r['fold']} | {r['best_epoch']} | "
            f"{r['pos_l1_mm']:.2f} | {r['feat_mse']:.1f} | "
            f"{r['position_emd_mm']:.2f} | {r['alive_auc']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path,
                    default=Path("experiments/stage1_forecaster/runs"))
    ap.add_argument("--out-parquet", type=Path,
                    default=Path("experiments/stage1_forecaster/summary.parquet"))
    ap.add_argument("--out-md", type=Path,
                    default=Path("experiments/stage1_forecaster/summary.md"))
    args = ap.parse_args()

    df = _collect(args.runs_root)
    if df.empty:
        print("no metrics.json found"); return 1
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_parquet, index=False)
    args.out_md.write_text(_summary_md(df))
    print(f"wrote {args.out_parquet} ({len(df)} rows)")
    print(f"wrote {args.out_md}")
    print()
    print(args.out_md.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
