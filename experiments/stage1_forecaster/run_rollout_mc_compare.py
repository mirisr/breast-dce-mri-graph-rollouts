#!/usr/bin/env python3
"""Monte Carlo-style rollout comparison across forecaster variants.

For deterministic forecasters, we use checkpoint ensembles (5-fold best.pt files)
as Monte Carlo draws and compare rollout-from-T0 metrics at a target horizon.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NB_ISPY2_DIR = REPO_ROOT / "notebooks" / "ispy2"
if str(NB_ISPY2_DIR) not in sys.path:
    sys.path.insert(0, str(NB_ISPY2_DIR))

import consistent_twin_lib as lib  # noqa: E402
from lsgc.metrics import compute_all_cloud_metrics, coverage_90, crps_empirical  # noqa: E402


def _load_models(ckpts: list[Path]):
    out = []
    for p in ckpts:
        if not p.is_file():
            continue
        try:
            model, mean, std = lib.load_model(p)
            out.append((p, model, mean, std))
        except Exception:
            continue
    return out


def _experiment_ckpts(repo_root: Path, exp: str) -> list[Path]:
    if exp == "v1_teacher_forced":
        return sorted((repo_root / "runs" / "consistent_forecaster_5fold").glob("fold*/best.pt"))
    return sorted((repo_root / "runs" / "consistent_forecaster_v2" / exp).glob("fold*/best.pt"))


def _safe_float(v: Any) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphs-root", type=Path, default=Path("datasets/ispy2/graphs_consistent"))
    ap.add_argument("--folds", type=Path, default=Path("datasets/ispy2/folds.parquet"))
    ap.add_argument("--cohort", type=Path, default=Path("datasets/ispy2/cohort.parquet"))
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--horizon", type=str, default="T3", choices=("T1", "T2", "T3"))
    ap.add_argument("--eps-pcr", type=float, default=0.1)
    ap.add_argument("--patient-limit", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("reports/rollout_mc_compare"))
    args = ap.parse_args()

    cohort = pd.read_parquet(args.cohort).set_index("patient_id")
    folds = pd.read_parquet(args.folds)
    pids = folds.loc[folds["fold"] == int(args.val_fold), "patient_id"].astype(str).tolist()
    pids = [p for p in pids if (args.graphs_root / f"{p}.pt").is_file()]
    if args.patient_limit > 0:
        pids = pids[: args.patient_limit]

    experiments = [
        "v1_teacher_forced",
        "v2_sched_samp",
        "v2_sched_horizon",
        "v2_sched_horizon_randstart",
    ]
    models_by_exp = {}
    for exp in experiments:
        ckpts = _experiment_ckpts(REPO_ROOT, exp)
        loaded = _load_models(ckpts)
        if loaded:
            models_by_exp[exp] = loaded
        print(f"{exp}: {len(loaded)} checkpoints")

    rows: list[dict[str, Any]] = []
    for pid in pids:
        g = lib.load_graph_dataset(pid, args.graphs_root)
        crow = {}
        if pid in cohort.index:
            c = cohort.loc[pid]
            crow = {
                "collection": str(c["collection"]) if pd.notna(c.get("collection", None)) else None,
                "pCR": int(c["pCR"]) if pd.notna(c.get("pCR", None)) else None,
                "subtype": str(c["subtype"]) if pd.notna(c.get("subtype", None)) else None,
            }

        for exp, model_pack in models_by_exp.items():
            sample_metrics = []
            ftv_samples = []
            ftv_obs = float("nan")
            for _, model, mean, std in model_pack:
                try:
                    steps = lib.rollout_from_visit(model, mean, std, g, start_visit=0)
                    st = next((x for x in steps if x["visit_name"] == args.horizon), None)
                    if st is None:
                        continue
                    pred_ftv = float(np.sum(np.clip(st["x_pred"][:, 1], 0.0, None) * np.asarray(st["alive_prob"])))
                    obs_ftv = float(np.sum(st["x_obs"][:, 1]))
                    ftv_obs = obs_ftv
                    m = compute_all_cloud_metrics(
                        pred_pos=st["pos_pred"],
                        obs_pos=st["pos_obs"],
                        pred_alive=np.asarray(st["alive_prob"]),
                        obs_alive=np.asarray(st["alive_obs"]),
                        pred_ftv=pred_ftv,
                        obs_ftv=obs_ftv,
                        ftv_t0=float(np.sum(g["x"][int(g["visit_offsets"][0]): int(g["visit_offsets"][1]), 1].numpy())),
                        compute_topology=False,
                    )
                    sample_metrics.append(m)
                    ftv_samples.append(pred_ftv)
                except Exception:
                    continue

            if not sample_metrics:
                continue

            agg: dict[str, Any] = {}
            metric_keys = [k for k, v in sample_metrics[0].items() if isinstance(v, (int, float))]
            for k in metric_keys:
                vals = np.asarray([_safe_float(d.get(k, np.nan)) for d in sample_metrics], dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                agg[f"{k}_mc_mean"] = float(np.mean(vals)) if vals.size else float("nan")
                agg[f"{k}_mc_std"] = float(np.std(vals)) if vals.size else float("nan")

            ftv_arr = np.asarray(ftv_samples, dtype=np.float64)
            row = {
                "patient_id": pid,
                "experiment": exp,
                "n_mc": int(len(ftv_arr)),
                "horizon": args.horizon,
                "ftv_obs": ftv_obs,
                "ftv_mc_mean": float(np.mean(ftv_arr)) if ftv_arr.size else float("nan"),
                "ftv_mc_std": float(np.std(ftv_arr)) if ftv_arr.size else float("nan"),
                "pcr_prob_mc": float(np.mean(ftv_arr < args.eps_pcr)) if ftv_arr.size else float("nan"),
                "coverage90_ftv": int(coverage_90(ftv_arr, ftv_obs)) if ftv_arr.size and np.isfinite(ftv_obs) else np.nan,
                "crps_ftv": float(crps_empirical(ftv_arr, ftv_obs)) if ftv_arr.size and np.isfinite(ftv_obs) else np.nan,
                **crow,
                **agg,
            }
            rows.append(row)

    if not rows:
        print("No MC rows generated.")
        return 1

    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_dir / "mc_compare_per_patient.parquet", index=False)

    # Aggregate summary
    key_cols = [c for c in df.columns if c.endswith("_mc_mean") or c in ("pcr_prob_mc", "coverage90_ftv", "crps_ftv")]
    srows = []
    for exp, grp in df.groupby("experiment"):
        s = {"experiment": exp, "n_patients": int(grp["patient_id"].nunique())}
        for c in key_cols:
            vals = pd.to_numeric(grp[c], errors="coerce").dropna().values
            s[f"{c}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
            s[f"{c}_se"] = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else float("nan")
        srows.append(s)
    summary = {"horizon": args.horizon, "rows": srows}
    (args.out_dir / "mc_compare_summary.json").write_text(json.dumps(summary, indent=2))
    print(args.out_dir / "mc_compare_per_patient.parquet")
    print(args.out_dir / "mc_compare_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
