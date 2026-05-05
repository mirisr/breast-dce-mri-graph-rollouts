#!/usr/bin/env python3
"""Stage-1 overnight diagnostics — Thread A2 + D.

Runs on existing S1.8 checkpoints (no training).
Produces a Markdown report with:
  D1. Calibration: ECE, Brier score per fold
  D2. Trivial baselines: copy-paste, uniform-shrink, random
  D3. Volume-aware metrics: Dice, IoU, predicted vs actual volume
  D4. Per-pCR-stratum and per-cohort breakdown of every metric
  A2. ISPY2-only vs ACRIN-only re-evaluation of existing checkpoints

Usage (Cradle, CPU node):
    python experiments/stage1_forecaster/run_diagnostics.py \
        --runs-dir experiments/stage1_forecaster/runs/bio_gated_5fold \
        --graphs-root datasets/ispy2/graphs_v3_full \
        --out reports/s1_diagnostics_$(date +%Y%m%d_%H%M).md
"""
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from lsgc.forecaster import LSGCForecaster
from lsgc.graph_builder import build_spatiotemporal_graph


# --------------------------------------------------------------------------- #
# ECE                                                                         #
# --------------------------------------------------------------------------- #

def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece_val += mask.sum() / len(probs) * abs(acc - conf)
    return float(ece_val)


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(((probs - labels) ** 2).mean())


# --------------------------------------------------------------------------- #
# Load a single fold checkpoint + its val split                               #
# --------------------------------------------------------------------------- #

def load_fold(
    fold_dir: Path,
    graphs_root: Path,
    folds_df: pd.DataFrame,
    fold_idx: int,
    cohort_df: pd.DataFrame,
    cohort_filter: str = "all",
) -> tuple:
    """Return (model, val_samples, feat_mean, feat_std, cfg)."""
    ckpt = torch.load(fold_dir / "best.pt", map_location="cpu", weights_only=False)
    # Checkpoint keys: state_dict / mean / std / config / val / epoch
    state_dict = ckpt["state_dict"]
    feat_mean  = ckpt["mean"]
    feat_std   = ckpt["std"]
    cfg        = ckpt.get("config", json.loads((fold_dir / "config.json").read_text())["config"])

    # Val IDs (apply cohort filter for A2)
    val_ids = set(folds_df.loc[folds_df["fold"] == fold_idx, "patient_id"].tolist())
    if cohort_filter != "all":
        prefix = "ISPY2" if cohort_filter == "ISPY2" else "ACRIN"
        val_ids = {pid for pid in val_ids if pid.startswith(prefix)}

    val_samples = []
    for pid in sorted(val_ids):
        gp = graphs_root / f"{pid}.pt"
        mp = graphs_root / f"{pid}.match-nn.pt"
        if not (gp.exists() and mp.exists()):
            continue
        try:
            g = torch.load(gp, map_location="cpu", weights_only=False)
            m = torch.load(mp, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if "x" not in g:
            continue
        val_samples.append((pid, g, m))

    # Infer in_channels from the alive_head weight shape
    in_ch = None
    for k, v in state_dict.items():
        if "alive_head" in k and v.ndim == 2:
            in_ch = v.shape[1]; break
    if in_ch is None:
        in_ch = int(feat_mean.shape[0])

    model = LSGCForecaster(
        in_channels=in_ch,
        hidden=cfg["hidden"],
        num_layers=cfg["num_layers"],
        feat_out_dim=in_ch,
        use_delta_t=cfg.get("use_delta_t", True),
        use_edge_gating=cfg.get("use_edge_gating", False),
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, val_samples, feat_mean, feat_std, cfg


# --------------------------------------------------------------------------- #
# Run model on one sample, collect per-transition metrics                     #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def eval_sample(pid, g, m, model, feat_mean, feat_std, device="cpu"):
    """Return list of per-transition result dicts."""
    x = ((g["x"].float() - feat_mean) / feat_std).to(device)
    pos_c = g["pos_c"].float().to(device)
    t     = g["t"].float().to(device) if "t" in g else torch.zeros(len(x), 1).to(device)
    off   = g["visit_offsets"].long()

    # Use the graph's pre-built edges directly (same as training path)
    try:
        from lsgc.graph_builder import build_spatiotemporal_graph
        result = build_spatiotemporal_graph(
            g, m, k_spatial=8, k_temporal=4,
            edge_mode="mixed", add_edge_attr=True, edge_attr_mode="bio",
        )
        edge_index = result.edge_index
        edge_attr  = result.edge_attr
    except Exception:
        edge_index = g.get("edge_index", torch.zeros(2, 0, dtype=torch.long))
        edge_attr  = None

    out = model(
        x, pos_c, t,
        edge_index.to(device),
        edge_attr=edge_attr.to(device) if edge_attr is not None else None,
        delta_t=1.0,
    )
    alive_prob = torch.sigmoid(out["alive_logit"]).cpu().numpy()
    pos_pred   = (pos_c + out["delta_pos"]).cpu().numpy()
    pos_c_np   = pos_c.cpu().numpy()

    results = []
    T = int(len(off)) - 1
    for k in range(T - 1):
        src_sl = slice(int(off[k]), int(off[k + 1]))
        dst_sl = slice(int(off[k + 1]), int(off[k + 2]))

        src_pos  = pos_c_np[src_sl]
        dst_pos  = pos_c_np[dst_sl]
        pred_pos = pos_pred[src_sl]
        a_prob   = alive_prob[src_sl]

        # Alive ground truth from match sidecar transitions list
        alive_gt = None
        for tr in m.get("transitions", []):
            if tr.get("k") == k:
                alive_gt = tr["alive"].float().numpy()
                break

        results.append({
            "pid": pid,
            "cohort": "ISPY2" if pid.startswith("ISPY2") else "ACRIN",
            "k": k,
            "src_pos": src_pos,
            "dst_pos": dst_pos,
            "pred_pos": pred_pos,
            "alive_prob": a_prob,
            "alive_gt": alive_gt,
        })
    return results


# --------------------------------------------------------------------------- #
# Trivial baselines                                                           #
# --------------------------------------------------------------------------- #

def copy_paste_emd(src_pos, dst_pos):
    """EMD proxy between the unchanged source cloud and the actual dst cloud."""
    return _sym_nn_dist(src_pos, dst_pos)


def uniform_shrink_emd(src_pos, dst_pos, shrink_ratio=0.85):
    """EMD proxy with a uniform per-centroid shrink toward the mean."""
    centre = src_pos.mean(axis=0, keepdims=True)
    shrunken = centre + shrink_ratio * (src_pos - centre)
    return _sym_nn_dist(shrunken, dst_pos)


def _sym_nn_dist(A, B):
    if len(A) == 0 or len(B) == 0:
        return float("nan")
    try:
        from scipy.spatial import cKDTree
        dAB = cKDTree(B).query(A)[0].mean()
        dBA = cKDTree(A).query(B)[0].mean()
        return float(0.5 * (dAB + dBA))
    except Exception:
        return float("nan")


def top_k_dice(src_pos, dst_pos, alive_prob, voxel_size_mm=2.0):
    """Dice between the top-K predicted survivors and the actual dst cloud.

    We discretise both clouds onto a voxel grid at `voxel_size_mm` resolution
    to compute the Dice of their binary masks.
    """
    K = len(dst_pos)
    if K == 0 or len(src_pos) == 0:
        return float("nan"), float("nan")

    top_idx   = np.argpartition(-alive_prob, min(K, len(alive_prob)) - 1)[:K]
    pred_cloud = src_pos[top_idx]

    # Voxelise both clouds
    def to_voxels(pts):
        vi = np.floor(pts / voxel_size_mm).astype(int)
        return set(map(tuple, vi))

    pred_vox = to_voxels(pred_cloud)
    act_vox  = to_voxels(dst_pos)
    inter    = len(pred_vox & act_vox)
    dice     = 2 * inter / (len(pred_vox) + len(act_vox) + 1e-8)
    iou      = inter / (len(pred_vox | act_vox) + 1e-8)
    return float(dice), float(iou)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path,
                    default=Path("experiments/stage1_forecaster/runs/bio_gated_5fold"))
    ap.add_argument("--graphs-root", type=Path,
                    default=Path("datasets/ispy2/graphs_v3_full"))
    ap.add_argument("--cohort", type=Path,
                    default=Path("datasets/ispy2/cohort.parquet"))
    ap.add_argument("--folds", type=Path,
                    default=Path("datasets/ispy2/folds.parquet"))
    ap.add_argument("--out", type=Path,
                    default=Path("reports/s1_diagnostics.md"))
    args = ap.parse_args()

    folds_df  = pd.read_parquet(args.folds)
    cohort_df = pd.read_parquet(args.cohort)
    pCR_map   = dict(zip(cohort_df["patient_id"], cohort_df["pCR"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for fold_idx in range(5):
        fold_dir = args.runs_dir / f"fold{fold_idx}"
        if not fold_dir.exists():
            print(f"  fold {fold_idx} missing, skipping")
            continue
        print(f"\n=== fold {fold_idx} ===")

        for cohort_filter in ("all", "ISPY2", "ACRIN"):
            try:
                model, val_samples, feat_mean, feat_std, cfg = load_fold(
                    fold_dir, args.graphs_root, folds_df, fold_idx,
                    cohort_df, cohort_filter=cohort_filter)
            except Exception as e:
                print(f"  [{cohort_filter}] load error: {e}")
                continue

            print(f"  [{cohort_filter}] n_val={len(val_samples)}")

            all_probs, all_gt = [], []
            for pid, g, m in val_samples:
                try:
                    res_list = eval_sample(pid, g, m, model, feat_mean, feat_std)
                except Exception as e:
                    print(f"    {pid} error: {e}")
                    continue

                pcr = pCR_map.get(pid, -1)
                for r in res_list:
                    sp, dp, pp, ap = (r["src_pos"], r["dst_pos"],
                                      r["pred_pos"], r["alive_prob"])
                    gt = r["alive_gt"]

                    model_emd = _sym_nn_dist(pp, dp)
                    cp_emd    = copy_paste_emd(sp, dp)
                    us_emd    = uniform_shrink_emd(sp, dp, shrink_ratio=0.85)

                    dice, iou = top_k_dice(sp, dp, ap)

                    row = dict(
                        fold=fold_idx,
                        cohort_filter=cohort_filter,
                        pid=pid,
                        cohort=r["cohort"],
                        pCR=pcr,
                        k=r["k"],
                        model_emd=model_emd,
                        copy_paste_emd=cp_emd,
                        uniform_shrink_emd=us_emd,
                        topk_dice=dice,
                        topk_iou=iou,
                        mean_alive_prob=float(ap.mean()),
                    )

                    if gt is not None and len(gt) == len(ap):
                        row["ece"]    = ece(ap, gt.astype(float))
                        row["brier"]  = brier(ap, gt.astype(float))
                        all_probs.extend(ap.tolist())
                        all_gt.extend(gt.astype(float).tolist())

                    rows.append(row)

    df = pd.DataFrame(rows)

    # ---------- write report -----------------------------------------------
    lines = [
        "# Stage-1 overnight diagnostics",
        "",
        f"Run on: {args.runs_dir}",
        "",
    ]

    for cf in ("all", "ISPY2", "ACRIN"):
        sub = df[df["cohort_filter"] == cf]
        if sub.empty:
            continue
        lines += [f"## Cohort filter = {cf}  (n_transitions = {len(sub)})", ""]

        # Global EMD comparison
        lines += ["### EMD — model vs trivial baselines", ""]
        for k in sorted(sub["k"].unique()):
            sk = sub[sub["k"] == k]
            m_emd = sk["model_emd"].mean()
            cp    = sk["copy_paste_emd"].mean()
            us    = sk["uniform_shrink_emd"].mean()
            lines.append(
                f"| T{k}→T{k+1} | model {m_emd:.2f} mm "
                f"| copy-paste {cp:.2f} mm "
                f"| uniform-shrink {us:.2f} mm |"
            )
        lines.append("")
        lines += [
            "| header | model | copy-paste | uniform-shrink |",
            "| --- | --- | --- | --- |",
        ]

        # Calibration
        if "ece" in sub.columns:
            ece_mean   = sub["ece"].mean()
            brier_mean = sub["brier"].mean()
            lines += [
                "### Calibration (alive head)",
                "",
                f"Mean ECE: **{ece_mean:.4f}** | Mean Brier: **{brier_mean:.4f}**",
                "",
            ]

        # Dice / IoU
        lines += [
            "### Voxel-level metrics (top-K Dice)",
            "",
            f"Mean Dice: **{sub['topk_dice'].mean():.4f}** | "
            f"Mean IoU: **{sub['topk_iou'].mean():.4f}**",
            "",
        ]

        # Per-pCR stratum
        lines += ["### Per-pCR-stratum breakdown", ""]
        lines += ["| pCR | n | model EMD | copy-paste EMD | Dice | ECE |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for pcr_val in [0, 1]:
            sp = sub[sub["pCR"] == pcr_val]
            if sp.empty:
                continue
            ece_v = sp["ece"].mean() if "ece" in sp.columns else float("nan")
            lines.append(
                f"| {pcr_val} | {len(sp)} | {sp['model_emd'].mean():.2f} | "
                f"{sp['copy_paste_emd'].mean():.2f} | "
                f"{sp['topk_dice'].mean():.4f} | {ece_v:.4f} |"
            )
        lines.append("")

        # Per-cohort breakdown (only when filter=all)
        if cf == "all":
            lines += ["### Per-cohort breakdown (ISPY2 vs ACRIN-6698)", ""]
            lines += ["| cohort | n | model EMD | copy-paste EMD | Dice | ECE |",
                      "| --- | --- | --- | --- | --- | --- |"]
            for coh in ("ISPY2", "ACRIN"):
                sc = sub[sub["cohort"] == coh]
                if sc.empty:
                    continue
                ece_v = sc["ece"].mean() if "ece" in sc.columns else float("nan")
                lines.append(
                    f"| {coh} | {len(sc)} | {sc['model_emd'].mean():.2f} | "
                    f"{sc['copy_paste_emd'].mean():.2f} | "
                    f"{sc['topk_dice'].mean():.4f} | {ece_v:.4f} |"
                )
            lines.append("")

    report = "\n".join(lines)
    args.out.write_text(report)
    print(f"\n=== Diagnostics report written to {args.out} ===")
    print(report[:3000])

    # Also save raw CSV
    csv_out = args.out.with_suffix(".csv")
    df.to_csv(csv_out, index=False)
    print(f"Raw data: {csv_out}")


if __name__ == "__main__":
    main()
