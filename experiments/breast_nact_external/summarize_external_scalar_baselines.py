#!/usr/bin/env python3
"""Summarize scalar carry-forward baselines for external NACT graphs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


VISITS = ("T0", "T1", "T2", "T3")


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def load_visit_ftv(graph_path: Path) -> dict[str, float]:
    g = torch.load(graph_path, map_location="cpu", weights_only=False)
    off = [int(v) for v in g["visit_offsets"].tolist()]
    out = {}
    for i, visit in enumerate(VISITS):
        sl = slice(off[i], off[i + 1])
        out[visit] = _safe_float(g["x"][sl, 1].sum().item())
    out["n_nodes"] = int(off[1] - off[0])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs-root", type=Path, default=Path("datasets/breast_mri_nact_pilot/graphs_consistent_4visit"))
    parser.add_argument("--patient-list", type=Path, default=Path("reports/breast_mri_nact_external/patients_4visit.txt"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/breast_mri_nact_external/tables"))
    args = parser.parse_args()

    patients = [ln.strip() for ln in args.patient_list.read_text().splitlines() if ln.strip()]
    rows = []
    for pid in patients:
        graph_path = args.graphs_root / f"{pid}.pt"
        if not graph_path.is_file():
            continue
        ftv = load_visit_ftv(graph_path)
        rows.append({"patient_id": pid, **ftv})
    patient_df = pd.DataFrame(rows)
    if patient_df.empty:
        raise RuntimeError("No scalar baseline rows generated")

    summary_rows = []
    for start in ("T0", "T1", "T2"):
        err = pd.to_numeric(patient_df[start], errors="coerce") - pd.to_numeric(patient_df["T3"], errors="coerce")
        summary_rows.append(
            {
                "baseline": f"{start}_carry_forward_to_T3",
                "n_patients": int(len(patient_df)),
                "mae_ml": float(np.mean(np.abs(err))),
                "bias_ml": float(np.mean(err)),
                "mean_pred_ftv_ml": float(pd.to_numeric(patient_df[start], errors="coerce").mean()),
                "mean_obs_t3_ftv_ml": float(pd.to_numeric(patient_df["T3"], errors="coerce").mean()),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    patient_path = args.out_dir / "external_4visit_observed_ftv_by_visit.csv"
    summary_path = args.out_dir / "external_4visit_scalar_carryforward_baselines.csv"
    meta_path = args.out_dir / "external_4visit_scalar_carryforward_baselines.json"
    patient_df.to_csv(patient_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    meta_path.write_text(json.dumps({
        "graphs_root": str(args.graphs_root),
        "patient_list": str(args.patient_list),
        "n_patients": int(len(patient_df)),
    }, indent=2))
    print(json.dumps({
        "patient_table": str(patient_path),
        "summary": str(summary_path),
        "metadata": str(meta_path),
    }, indent=2))
    print(patient_df.to_string(index=False))
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
