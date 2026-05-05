#!/usr/bin/env python3
"""Pick 30 'good prediction' patients for the consistent-graph prototype.

For each patient with a v3_full graph, run S1.8 and compute the mean centroid
NN error across all available transitions. Sort by this score (best first),
stratify by (pCR, cohort), and select 30 patients.

Outputs:
  reports/prototype_patients.csv  — selected patients with metrics
  reports/prototype_patients.txt  — newline-separated patient IDs
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

# Repo root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from lsgc.forecaster import LSGCForecaster
from experiments.stage1_forecaster.train_forecaster import _rebuild_edges

REPO = Path("/home/irisseaman01/3DGCNN")


@torch.no_grad()
def score_patient(pid: str, model, mean, std, graphs_root: Path) -> dict:
    """Return per-patient mean centroid NN error across all transitions, plus n_tx."""
    gp = graphs_root / f"{pid}.pt"
    mp = graphs_root / f"{pid}.match-nn.pt"
    if not (gp.exists() and mp.exists()):
        return {}
    try:
        g = torch.load(gp, map_location="cpu", weights_only=False)
        m = torch.load(mp, map_location="cpu", weights_only=False)
    except Exception:
        return {}

    # Edges (mixed+attr bio mode, matching S1.8 training)
    try:
        edge_index, edge_attr, _ = _rebuild_edges(
            g, [1, 2, 3], k_spatial=8, k_temporal=4,
            edge_mode="mixed", add_edge_attr=True, edge_attr_mode="bio",
        )
    except Exception as e:
        return {}

    x = ((g["x"].float() - mean) / std)
    # Center positions per-visit (same convention as training)
    off = g["visit_offsets"].long()
    pos = g["pos"].float()
    pos_c = pos.clone()
    centroids = m.get("visit_centroids", None)
    if centroids is not None:
        for v in range(len(off) - 1):
            pos_c[off[v]: off[v + 1]] -= centroids[v]
    else:
        for v in range(len(off) - 1):
            sl = slice(int(off[v]), int(off[v + 1]))
            pos_c[sl] = pos_c[sl] - pos_c[sl].mean(dim=0, keepdim=True)
    t = g.get("t", torch.zeros(len(x), 1)).float()

    out = model(x, pos_c, t, edge_index, edge_attr=edge_attr, delta_t=1.0)
    pos_pred = (pos_c + out["delta_pos"]).numpy()

    nn_errors = []
    T = int(len(off)) - 1
    for k in range(T - 1):
        src_sl = slice(int(off[k]), int(off[k + 1]))
        dst_sl = slice(int(off[k + 1]), int(off[k + 2]))
        pred = pos_pred[src_sl]
        actual = pos_c[dst_sl].numpy()
        if len(pred) == 0 or len(actual) == 0:
            continue
        try:
            from scipy.spatial import cKDTree
            d, _ = cKDTree(actual).query(pred, k=1)
            nn_errors.append(float(np.mean(d)))
        except Exception:
            continue

    if not nn_errors:
        return {}

    return {
        "patient_id": pid,
        "mean_nn_err_mm": float(np.mean(nn_errors)),
        "n_transitions": len(nn_errors),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path,
                    default=REPO / "experiments/stage1_forecaster/runs/bio_gated_5fold/fold0/best.pt")
    ap.add_argument("--graphs-root", type=Path,
                    default=REPO / "datasets/ispy2/graphs_v3_full")
    ap.add_argument("--cohort", type=Path,
                    default=REPO / "datasets/ispy2/cohort.parquet")
    ap.add_argument("--folds", type=Path,
                    default=REPO / "datasets/ispy2/folds.parquet")
    ap.add_argument("--n-select", type=int, default=30)
    ap.add_argument("--out", type=Path,
                    default=REPO / "reports/prototype_patients.csv")
    args = ap.parse_args()

    cohort = pd.read_parquet(args.cohort)
    folds = pd.read_parquet(args.folds)

    # Eligible: has graph, has 4 visits, has pCR
    elig = cohort[(cohort["has_graph"] == True)
                  & (cohort["visit_count"] >= 4)
                  & (cohort["pCR"].notna())]
    print(f"Eligible patients: {len(elig)}")

    # Load model
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    state_dict = ck["state_dict"]

    # Input feature dim from embed.weight (or fall back to mean shape).
    if "embed.weight" in state_dict:
        in_ch = int(state_dict["embed.weight"].shape[1])
    else:
        in_ch = int(ck["mean"].shape[0])

    # Infer edge_attr_dim from the conv filter_net weight shape:
    #   filter_net.0.weight has shape (hidden, 2*pos_dim + edge_attr_dim + 1)
    # The constant offset depends on conv impl; just compute the diff from a
    # reference build with edge_attr_dim=0 by checking: hidden_in = 27 (no edges)
    # vs 31 (4 bio edge channels). We compare against a no-edge model.
    cur_in = state_dict["convs.0.filter_net.0.weight"].shape[1]
    # Determine edge_attr_dim by trying both 0 and 4 (most common)
    edge_attr_dim = 4 if cur_in == 31 else 0

    model = LSGCForecaster(
        in_channels=in_ch, hidden=cfg["hidden"], num_layers=cfg["num_layers"],
        feat_out_dim=in_ch, use_delta_t=cfg.get("use_delta_t", True),
        use_edge_gating=cfg.get("use_edge_gating", False),
        edge_attr_dim=edge_attr_dim,
    )
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint: in_ch={in_ch}, edge_attr_dim={edge_attr_dim}")

    # Score every eligible patient
    rows = []
    for i, pid in enumerate(elig["patient_id"]):
        if i % 50 == 0:
            print(f"  scoring {i}/{len(elig)} ({pid})", flush=True)
        r = score_patient(pid, model, ck["mean"], ck["std"], args.graphs_root)
        if not r:
            continue
        meta = elig[elig["patient_id"] == pid].iloc[0]
        r.update(dict(
            cohort="ISPY2" if pid.startswith("ISPY2") else "ACRIN",
            pCR=int(meta["pCR"]),
            subtype=meta["subtype"],
        ))
        rows.append(r)

    df = pd.DataFrame(rows)
    print(f"\nScored {len(df)} patients")
    print(df["mean_nn_err_mm"].describe())

    # Stratify: pCR=0/1 × cohort=ISPY2/ACRIN, pick top by best (lowest) NN err
    target = {
        ("ISPY2", 0): 11,   # ~22 ISPY2 (74%) split 11+11
        ("ISPY2", 1): 11,
        ("ACRIN", 0): 4,    # ~8 ACRIN (26%) split 4+4
        ("ACRIN", 1): 4,
    }
    selected = []
    for (coh, pcr), n in target.items():
        sub = df[(df["cohort"] == coh) & (df["pCR"] == pcr)].sort_values("mean_nn_err_mm")
        selected.append(sub.head(n))
    sel_df = pd.concat(selected).sort_values("mean_nn_err_mm").reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sel_df.to_csv(args.out, index=False)
    txt_out = args.out.with_suffix(".txt")
    txt_out.write_text("\n".join(sel_df["patient_id"]))

    print(f"\n=== Selected {len(sel_df)} patients ===")
    print(sel_df.groupby(["cohort", "pCR"]).size())
    print(f"\nWritten to {args.out}")
    print(f"Patient list: {txt_out}")
    print(f"\nPredicted error range: "
          f"{sel_df['mean_nn_err_mm'].min():.2f} - {sel_df['mean_nn_err_mm'].max():.2f} mm "
          f"(median {sel_df['mean_nn_err_mm'].median():.2f})")


if __name__ == "__main__":
    main()
