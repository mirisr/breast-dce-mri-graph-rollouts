#!/usr/bin/env python
"""Train the LSGC-Forecaster (Stage 1) on I-SPY 2.

For each patient graph we run a single forward pass of the LSGCNet
backbone with three per-node heads (see :class:`LSGCForecaster`). The
training signal comes from the precomputed Sinkhorn matches
(``<patient>.match.pt``):

* matched pairs at transition ``k -> k+1`` supervise ``delta_pos`` and
  ``delta_feat``;
* the ``alive`` label supervises ``alive_logit`` on every source node at
  every non-terminal visit.

Losses (per the proposal, Section 5.1):

    L = lambda_pos  * SmoothL1(delta_pos_hat,  delta_pos_gt)    # live pairs only
      + lambda_feat * MSE     (delta_feat_hat, delta_feat_gt)   # live pairs only
      + lambda_alive* BCE     (alive_logit,    alive_gt)        # all source nodes

Evaluation metrics (per the proposal, Section 5.1 "Evaluation"):

* position L1 per pair (matched pairs),
* sliced Wasserstein between predicted and observed T_{k+1} point clouds,
* feature MSE per channel (matched pairs),
* existence AUC on all source nodes.

Usage
-----

5-fold CV, best checkpoint config (L=2, skips=(1,2,3))::

    python experiments/stage1_forecaster/train_forecaster.py \
        --out-dir experiments/stage1_forecaster/runs/L2_skips-1-2-3 \
        --fold 0 --num-layers 2 --temporal-skip-hops 1 2 3 \
        --epochs 60 --device cuda

CPU smoke test on 40 patients::

    python experiments/stage1_forecaster/train_forecaster.py \
        --out-dir /tmp/forecaster_smoke --fold 0 \
        --num-layers 2 --temporal-skip-hops 1 2 3 \
        --smoke --device cpu
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lsgc.forecaster import LSGCForecaster  # noqa: E402
from lsgc.graph_builder import build_spatiotemporal_graph  # noqa: E402


# --------------------------------------------------------------------------- #
# Data loading                                                                #
# --------------------------------------------------------------------------- #


def _bio_indices_from_feature_names(feature_names: list[str] | None) -> dict:
    """Extract ADC / DCE indices for the bio edge_attr mode from feature_names.

    Returns a dict with keys ``adc_idx``, ``adc_missing_idx``,
    ``dce_idx_start``, ``dce_n_phases``. All-None means we can't find any
    of the expected biology features and the bio edge_attr will fall back
    to mostly-zeros (still safe, just less informative).
    """
    out = {"adc_idx": None, "adc_missing_idx": None,
           "dce_idx_start": 0, "dce_n_phases": 0}
    if not feature_names:
        return out
    if "mean_adc" in feature_names:
        out["adc_idx"] = feature_names.index("mean_adc")
    if "adc_missing" in feature_names:
        out["adc_missing_idx"] = feature_names.index("adc_missing")
    dce_idxs = [i for i, n in enumerate(feature_names)
                if n.startswith("phase") and n.endswith("_mean_enh")]
    if dce_idxs and dce_idxs == list(range(min(dce_idxs), max(dce_idxs) + 1)):
        out["dce_idx_start"] = min(dce_idxs)
        out["dce_n_phases"] = len(dce_idxs)
    return out


def _rebuild_edges(
    graph: dict, temporal_skip_hops: Sequence[int],
    k_spatial: int, k_temporal: int,
    edge_mode: str = "geometric",
    spatial_alpha: float = 0.7,
    add_edge_attr: bool = False,
    edge_attr_mode: str = "legacy",
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Rebuild edge_index (and optional edge_attr) with requested edge mode."""
    offsets = graph["visit_offsets"].tolist()
    x = graph["x"]
    pos = graph["pos"]
    visit_feats = [x[offsets[v]: offsets[v + 1]] for v in range(len(offsets) - 1)]
    visit_pos = [pos[offsets[v]: offsets[v + 1]] for v in range(len(offsets) - 1)]
    bio = _bio_indices_from_feature_names(graph.get("feature_names"))
    g = build_spatiotemporal_graph(
        visit_feats, visit_pos,
        k_spatial=k_spatial, k_temporal=k_temporal,
        temporal_skip_hops=tuple(temporal_skip_hops),
        edge_mode=edge_mode,
        spatial_alpha=spatial_alpha,
        add_edge_attr=add_edge_attr,
        edge_attr_mode=edge_attr_mode,
        adc_idx=bio["adc_idx"],
        adc_missing_idx=bio["adc_missing_idx"],
        dce_idx_start=bio["dce_idx_start"],
        dce_n_phases=bio["dce_n_phases"],
        habitat_labels=None if "habitat" not in graph else [
            graph["habitat"][offsets[v]: offsets[v + 1]] for v in range(len(offsets) - 1)
        ],
    )
    return (
        g.edge_index.long(),
        None if g.edge_attr is None else g.edge_attr.float(),
        None if g.edge_type is None else g.edge_type.long(),
    )


@dataclass
class ForecasterSample:
    patient_id: str
    x: torch.Tensor
    pos: torch.Tensor          # absolute (mm)
    pos_c: torch.Tensor        # centered per-visit (mm), matches .match.pt frame
    t: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor | None
    edge_type: torch.Tensor | None
    visit_offsets: torch.Tensor
    visit_centroids: torch.Tensor  # (T, 3)
    transitions: list[dict]
    clinical: torch.Tensor | None = None
    visit_context: torch.Tensor | None = None


def _load_sample(
    graph_path: Path, match_path: Path,
    temporal_skip_hops: Sequence[int],
    k_spatial: int, k_temporal: int,
    edge_mode: str,
    spatial_alpha: float,
    add_edge_attr: bool,
    edge_attr_mode: str = "legacy",
) -> ForecasterSample | None:
    try:
        graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        match = torch.load(match_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    edge_index, edge_attr, edge_type = _rebuild_edges(
        graph,
        temporal_skip_hops,
        k_spatial,
        k_temporal,
        edge_mode=edge_mode,
        spatial_alpha=spatial_alpha,
        add_edge_attr=add_edge_attr,
        edge_attr_mode=edge_attr_mode,
    )

    # Centered positions (per visit). Mirror match_visits.compute_patient_matches.
    offsets = graph["visit_offsets"].tolist()
    pos = graph["pos"].float()
    pos_c = pos.clone()
    centroids = match["visit_centroids"]
    for v in range(len(offsets) - 1):
        pos_c[offsets[v]: offsets[v + 1]] -= centroids[v]

    return ForecasterSample(
        patient_id=graph.get("patient_id", graph_path.stem),
        x=graph["x"].float(),
        pos=pos,
        pos_c=pos_c,
        t=graph["t"].long(),
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type,
        visit_offsets=graph["visit_offsets"].long(),
        visit_centroids=centroids.float(),
        transitions=match["transitions"],
        clinical=None if "clinical" not in graph else graph["clinical"].float(),
        visit_context=None if "visit_context" not in graph else graph["visit_context"].float(),
    )


def _normalize_features(
    samples: list[ForecasterSample],
    *,
    min_std: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Global z-score of ``x`` across the training split.

    Columns with training std < ``min_std`` (dead features such as
    phase0_mean_enh which is always 0) are zeroed in the output rather
    than divided by a tiny number to avoid large activations for rare
    non-zero entries. Returns (mean, std_safe) where std_safe >= min_std
    for all active columns and equals 1.0 for dead columns (so dividing
    by it leaves them at ~0).
    """
    allx = torch.cat([s.x for s in samples], dim=0)
    mean = allx.mean(dim=0)
    std = allx.std(dim=0)
    active = std >= min_std
    std_safe = std.where(active, torch.ones_like(std))
    n_dead = int((~active).sum().item())
    if n_dead:
        import sys
        print(f"  [normalize] zeroing {n_dead} dead feature columns", file=sys.stderr, flush=True)
    return mean, std_safe


# --------------------------------------------------------------------------- #
# Loss + metrics                                                              #
# --------------------------------------------------------------------------- #


def _slice_visit(t: torch.Tensor, off: torch.Tensor, v: int) -> torch.Tensor:
    return t[off[v]: off[v + 1]]


def forecaster_losses(
    out: dict[str, torch.Tensor], sample: ForecasterSample,
    *, lambda_pos: float, lambda_feat: float, lambda_alive: float,
    pos_weight: torch.Tensor | None = None,
    feat_std: torch.Tensor | None = None,
    alive_loss_type: str = "bce",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the combined Stage-1 loss + per-term diagnostics.

    ``feat_std`` should be the per-feature training standard deviation
    (shape ``(C,)``). When provided, ``delta_feat_gt`` is divided by
    ``feat_std`` before the MSE so the feature loss is always in
    z-scored space regardless of raw feature scale (critical when
    features span multiple orders of magnitude, e.g. ADC vs one-hots).

    ``alive_loss_type`` controls the alive head loss:
      'bce'   — standard weighted BCE (default)
      'focal' — focal loss with gamma=2; down-weights easy examples so
                the model focuses on hard borderline supervoxels.
    """
    smooth_l1 = nn.SmoothL1Loss(reduction="mean")
    mse = nn.MSELoss(reduction="mean")

    def _alive_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if alive_loss_type == "focal":
            # Focal loss: FL = -α(1-p)^γ log(p)  with γ=2.
            # We use the sigmoid-BCE form for numerical stability.
            gamma = 2.0
            bce_each = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight, reduction="none")
            p_t = torch.exp(-bce_each)   # p_t approximation from BCE value
            return ((1 - p_t) ** gamma * bce_each).mean()
        else:
            return F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight)

    device = out["delta_pos"].device
    total = torch.tensor(0.0, device=device)
    loss_pos = torch.tensor(0.0, device=device)
    loss_feat = torch.tensor(0.0, device=device)
    loss_alive = torch.tensor(0.0, device=device)
    n_live = 0; n_nodes = 0

    feat_scale = feat_std.to(device) if feat_std is not None else None

    off = sample.visit_offsets
    for tr in sample.transitions:
        k = tr["k"]
        src_slice = slice(int(off[k]), int(off[k + 1]))
        alive_gt = tr["alive"].to(device).float()
        src_live_idx = tr["src_live_idx"].to(device)

        # Existence BCE over all source nodes (alive vs dead).
        alive_logit = out["alive_logit"][src_slice]
        lb = _alive_loss(alive_logit, alive_gt)
        loss_alive = loss_alive + lb
        n_nodes += alive_gt.numel()

        # Position + feature losses only on live pairs.
        if src_live_idx.numel() > 0:
            dp_hat = out["delta_pos"][src_slice][src_live_idx]
            dp_gt = tr["delta_pos"].to(device)
            lp = smooth_l1(dp_hat, dp_gt)
            loss_pos = loss_pos + lp

            dh_hat = out["delta_feat"][src_slice][src_live_idx]
            dh_gt = tr["delta_feat"].to(device)
            if feat_scale is not None:
                dh_gt = dh_gt / feat_scale.clamp_min(1e-6)
            lf = mse(dh_hat, dh_gt)
            loss_feat = loss_feat + lf

            n_live += src_live_idx.numel()

    n_tr = max(len(sample.transitions), 1)
    loss_alive = loss_alive / n_tr
    loss_pos = loss_pos / n_tr
    loss_feat = loss_feat / n_tr
    total = lambda_pos * loss_pos + lambda_feat * loss_feat + lambda_alive * loss_alive

    diag = {
        "loss": float(total.detach().item()),
        "loss_pos": float(loss_pos.detach().item()),
        "loss_feat": float(loss_feat.detach().item()),
        "loss_alive": float(loss_alive.detach().item()),
        "n_live": n_live,
        "n_nodes": n_nodes,
    }
    return total, diag


def sliced_wasserstein(
    X: torch.Tensor, Y: torch.Tensor, n_projections: int = 64,
    resample: int = 512, generator: torch.Generator | None = None,
) -> float:
    """Sliced Wasserstein approx between 3-D point clouds.

    Returns a scalar mean of 1-D Wassersteins on ``n_projections``
    random directions. Both clouds are resampled (with replacement) to
    ``resample`` points along each projection, so unequal-size inputs
    compare correctly.
    """
    if X.shape[0] == 0 or Y.shape[0] == 0:
        return float("nan")
    device = X.device
    dirs = torch.randn(3, n_projections, device=device, generator=generator)
    dirs = dirs / dirs.norm(dim=0, keepdim=True).clamp_min(1e-8)

    Xp = (X @ dirs).T  # (n_proj, n_x)
    Yp = (Y @ dirs).T  # (n_proj, n_y)

    # Resample via quantile interpolation so the two sorted sequences are
    # comparable even when n_x != n_y.
    q = torch.linspace(0, 1, resample, device=device)
    Xq = torch.quantile(Xp, q, dim=1).T  # (n_proj, resample)
    Yq = torch.quantile(Yp, q, dim=1).T
    return (Xq - Yq).abs().mean().item()


def evaluate(
    model: LSGCForecaster, samples: list[ForecasterSample],
    mean: torch.Tensor, std: torch.Tensor, *, device: str,
) -> dict[str, float]:
    model.eval()
    pos_err = []; feat_err = []; emd = []
    all_alive_gt = []; all_alive_pred = []
    with torch.no_grad():
        for s in samples:
            x = ((s.x - mean) / std).to(device)
            pos_c = s.pos_c.to(device)
            t = s.t.to(device)
            edge_index = s.edge_index.to(device)
            out = model(
                x,
                pos_c,
                t,
                edge_index,
                edge_attr=s.edge_attr.to(device) if s.edge_attr is not None else None,
                edge_type=s.edge_type.to(device) if s.edge_type is not None else None,
                delta_t=1.0,
            )

            off = s.visit_offsets
            for tr in s.transitions:
                k = tr["k"]
                src_slice = slice(int(off[k]), int(off[k + 1]))
                src_live_idx = tr["src_live_idx"].to(device)
                alive_gt = tr["alive"].to(device)

                if src_live_idx.numel() > 0:
                    dp_hat = out["delta_pos"][src_slice][src_live_idx].cpu()
                    dp_gt = tr["delta_pos"]
                    pos_err.append((dp_hat - dp_gt).norm(dim=1).mean().item())

                    dh_hat = out["delta_feat"][src_slice][src_live_idx].cpu()
                    dh_gt = tr["delta_feat"]
                    feat_err.append(((dh_hat - dh_gt) ** 2).mean().item())

                    # Predicted T_{k+1} cloud: only live sources, moved by predicted delta.
                    src_pos_c = s.pos_c[src_slice][src_live_idx.cpu()]
                    pred_cloud = src_pos_c + dp_hat
                    # Observed T_{k+1} cloud (centered already).
                    obs_cloud = s.pos_c[int(off[k + 1]): int(off[k + 2])]
                    emd.append(sliced_wasserstein(pred_cloud, obs_cloud))

                all_alive_gt.append(alive_gt.cpu())
                all_alive_pred.append(torch.sigmoid(out["alive_logit"][src_slice]).cpu())

    alive_gt = torch.cat(all_alive_gt).numpy().astype(np.int32)
    alive_pred = torch.cat(all_alive_pred).numpy()
    try:
        alive_auc = roc_auc_score(alive_gt, alive_pred)
    except ValueError:
        alive_auc = float("nan")

    return {
        "pos_l1_mm": float(np.mean(pos_err)) if pos_err else float("nan"),
        "feat_mse":  float(np.mean(feat_err)) if feat_err else float("nan"),
        "position_emd_mm": float(np.nanmean(emd)) if emd else float("nan"),
        "alive_auc": float(alive_auc),
        "n_patients": len(samples),
    }


# --------------------------------------------------------------------------- #
# Training loop                                                               #
# --------------------------------------------------------------------------- #


@dataclass
class TrainConfig:
    fold: int
    num_layers: int
    temporal_skip_hops: list[int]
    hidden: int
    use_delta_t: bool
    epochs: int
    batch_accum: int
    lr: float
    weight_decay: float
    lambda_pos: float
    lambda_feat: float
    lambda_alive: float
    seed: int
    k_spatial: int
    k_temporal: int
    conv_type: str = "lsgc"
    use_edge_gating: bool = False
    alive_loss_type: str = "bce"
    alive_pos_weight_scale: float = 1.0


def _set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_and_eval(
    train_samples: list[ForecasterSample],
    val_samples: list[ForecasterSample],
    cfg: TrainConfig, device: str, out_dir: Path,
) -> dict:
    _set_seed(cfg.seed)

    mean, std = _normalize_features(train_samples)

    in_channels = train_samples[0].x.shape[1]
    edge_attr_dim = (
        train_samples[0].edge_attr.shape[1]
        if train_samples[0].edge_attr is not None else 0
    )
    conv_kwargs = {
        "edge_attr_dim": edge_attr_dim,
        "use_edge_gating": cfg.use_edge_gating,
    }
    if cfg.conv_type == "relational":
        conv_kwargs.update({"num_relations": 4, "num_bases": 2})
    model = LSGCForecaster(
        in_channels=in_channels,
        hidden=cfg.hidden,
        num_layers=cfg.num_layers,
        feat_out_dim=in_channels,
        use_delta_t=cfg.use_delta_t,
        conv_type=cfg.conv_type,
        **conv_kwargs,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=cfg.epochs)

    # Compute a pos_weight for the BCE head from the *training* alive rate.
    n_alive = n_total = 0
    for s in train_samples:
        for tr in s.transitions:
            n_alive += int(tr["alive"].sum().item())
            n_total += int(tr["alive"].numel())
    neg = max(n_total - n_alive, 1); pos = max(n_alive, 1)
    pos_weight = torch.tensor([neg / pos * cfg.alive_pos_weight_scale], device=device)
    print(f"  alive pos_weight={float(pos_weight):.3f} "
          f"(base={neg/pos:.3f} × scale={cfg.alive_pos_weight_scale})", flush=True)

    history = []
    best = {"epoch": -1, "val": None}
    t0 = time.time()

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"
    with log_path.open("w") as fh:
        fh.write("epoch,train_loss,val_pos_l1_mm,val_feat_mse,val_position_emd_mm,val_alive_auc\n")

    for epoch in range(cfg.epochs):
        model.train()
        order = list(range(len(train_samples)))
        random.shuffle(order)

        opt.zero_grad()
        running = 0.0; step_count = 0
        for step, idx in enumerate(order):
            s = train_samples[idx]
            x = ((s.x - mean) / std).to(device)
            pos_c = s.pos_c.to(device)
            t = s.t.to(device)
            edge_index = s.edge_index.to(device)
            out = model(
                x,
                pos_c,
                t,
                edge_index,
                edge_attr=s.edge_attr.to(device) if s.edge_attr is not None else None,
                edge_type=s.edge_type.to(device) if s.edge_type is not None else None,
                delta_t=1.0,
            )

            loss, diag = forecaster_losses(
                out, _move_transitions(s, device),
                lambda_pos=cfg.lambda_pos, lambda_feat=cfg.lambda_feat,
                lambda_alive=cfg.lambda_alive, pos_weight=pos_weight,
                feat_std=std, alive_loss_type=cfg.alive_loss_type,
            )
            (loss / cfg.batch_accum).backward()
            running += diag["loss"]

            if (step + 1) % cfg.batch_accum == 0 or step == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step(); opt.zero_grad()
                step_count += 1

        sched.step()
        train_loss = running / max(len(order), 1)

        val_metrics = evaluate(model, val_samples, mean, std, device=device)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        with log_path.open("a") as fh:
            fh.write(
                f"{epoch},{train_loss:.5f},{val_metrics['pos_l1_mm']:.4f},"
                f"{val_metrics['feat_mse']:.5f},{val_metrics['position_emd_mm']:.4f},"
                f"{val_metrics['alive_auc']:.4f}\n"
            )

        # "Best" = highest alive_auc; tie-break lowest pos_l1.
        score = (val_metrics["alive_auc"], -val_metrics["pos_l1_mm"])
        if best["val"] is None or score > (best["val"]["alive_auc"], -best["val"]["pos_l1_mm"]):
            best = {"epoch": epoch, "val": val_metrics,
                    "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

        print(f"[fold {cfg.fold}] epoch {epoch:3d}  train_loss={train_loss:.4f}  "
              f"val alive_auc={val_metrics['alive_auc']:.3f}  pos_l1={val_metrics['pos_l1_mm']:.2f}mm  "
              f"emd={val_metrics['position_emd_mm']:.2f}mm")

    torch.save({
        "state_dict": best["state_dict"],
        "mean": mean, "std": std, "config": asdict(cfg),
        "val": best["val"], "epoch": best["epoch"],
    }, out_dir / "best.pt")

    wall_s = time.time() - t0
    return {
        "best_epoch": best["epoch"],
        "best_val": best["val"],
        "final_val": history[-1],
        "wall_s": wall_s,
        "config": asdict(cfg),
    }


def _move_transitions(s: ForecasterSample, device: str) -> ForecasterSample:
    """Shallow copy that leaves original on CPU (memory-cheap on small data)."""
    return s  # losses move per-tensor to device on access


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", type=Path,
                    default=Path("datasets/ispy2/cohort.parquet"))
    ap.add_argument("--folds", type=Path,
                    default=Path("datasets/ispy2/folds.parquet"))
    ap.add_argument("--graphs-root", type=Path,
                    default=Path("datasets/ispy2/graphs"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--match-method", choices=("sinkhorn", "nn"), default="sinkhorn",
                    help="Which match sidecar suffix to load (.match.pt or .match-nn.pt).")
    # Architecture
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--temporal-skip-hops", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--conv-type", choices=("lsgc", "relational"), default="lsgc",
                    help="Backbone conv: standard LSGC or relational LSGC with 4 bio relations.")
    ap.add_argument("--no-delta-t", action="store_true",
                    help="Ablation: drop the explicit delta_t conditioning token.")
    # Optim
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lambda-pos", type=float, default=1.0)
    ap.add_argument("--lambda-feat", type=float, default=0.5)
    ap.add_argument("--lambda-alive", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-spatial", type=int, default=8)
    ap.add_argument("--k-temporal", type=int, default=4)
    ap.add_argument("--edge-mode", choices=("geometric", "mixed", "mixed+attr"), default="geometric")
    ap.add_argument("--spatial-alpha", type=float, default=0.7)
    ap.add_argument("--edge-attr-mode", choices=("legacy", "bio"), default="legacy",
                    help="Edge-attribute encoding when --edge-mode=mixed+attr. "
                         "'legacy' = original 4 binary channels (same_h, pe<.25, "
                         "vol>.5, intra-flag). 'bio' = 4 continuous biological "
                         "channels: |ΔADC|, cross-habitat, 1-cos(DCE_curve), "
                         "is_inter_visit. The bio mode requires v3 features "
                         "(mean_adc + DCE phases) and habitat labels in the graph.")
    ap.add_argument("--use-edge-gating", action="store_true",
                    help="Enable per-edge attention gate g_ij = σ(MLP([edge_attr, h_i, h_j])) "
                         "in every LSGCConv layer. Lets the model down-weight low-information "
                         "edges (e.g. necrotic-boundary connections). Best paired with "
                         "--edge-attr-mode=bio. Adds ~O(hidden×gate_hidden) parameters per layer.")
    ap.add_argument("--cohort-filter", choices=("all", "ISPY2", "ACRIN"),
                    default="all",
                    help="Restrict both train and val to a single source cohort. "
                         "'ISPY2' drops the 203 ACRIN-6698 patients from every split.")
    ap.add_argument("--alive-loss-type", choices=("bce", "focal"), default="bce",
                    help="Loss for the alive head. 'bce' = weighted BCE (default, existing "
                         "behaviour). 'focal' = focal loss with gamma=2 that down-weights "
                         "easy examples and focuses capacity on the hard margin cases.")
    ap.add_argument("--alive-pos-weight-scale", type=float, default=1.0,
                    help="Scalar multiplier applied to the auto-computed pos_weight for the "
                         "alive BCE head. Default 1.0 = natural class imbalance weighting. "
                         "Use > 1.0 to further penalise missed positives (e.g. 2.0 = double "
                         "the cost of predicting death when the supervoxel actually survived).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true",
                    help="Run on the first 40 patients, 3 epochs (CI / local sanity).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    folds = pd.read_parquet(args.folds)
    cohort = pd.read_parquet(args.cohort)

    # Optional cohort filter: restrict to ISPY2 or ACRIN-6698 only.
    if args.cohort_filter != "all":
        prefix = "ISPY2" if args.cohort_filter == "ISPY2" else "ACRIN"
        keep = set(folds.loc[folds["patient_id"].str.startswith(prefix), "patient_id"])
        folds = folds[folds["patient_id"].isin(keep)]
        print(f"[cohort_filter={args.cohort_filter}] keeping {len(folds)} patients "
              f"(prefix={prefix!r})")

    # Validation = rows where folds["fold"] == args.fold; training = everything else.
    val_ids = set(folds.loc[folds["fold"] == args.fold, "patient_id"].tolist())
    train_ids = set(folds.loc[folds["fold"] != args.fold, "patient_id"].tolist())

    suffix = ".match.pt" if args.match_method == "sinkhorn" else f".match-{args.match_method}.pt"
    add_edge_attr = args.edge_mode == "mixed+attr"
    edge_mode = "mixed" if args.edge_mode in {"mixed", "mixed+attr"} else "geometric"

    def _load_split(ids: set[str]) -> list[ForecasterSample]:
        out = []
        expected_dim: int | None = None
        for pid in sorted(ids):
            gp = args.graphs_root / f"{pid}.pt"
            mp = args.graphs_root / f"{pid}{suffix}"
            if not (gp.exists() and mp.exists()):
                continue
            s = _load_sample(gp, mp, args.temporal_skip_hops,
                             args.k_spatial, args.k_temporal,
                             edge_mode=edge_mode,
                             spatial_alpha=args.spatial_alpha,
                             add_edge_attr=add_edge_attr,
                             edge_attr_mode=args.edge_attr_mode)
            if s is None or len(s.transitions) == 0:
                continue
            if expected_dim is None:
                expected_dim = int(s.x.shape[1])
            if int(s.x.shape[1]) != expected_dim:
                continue
            out.append(s)
        return out

    print(f"loading train split ...")
    train_samples = _load_split(train_ids)
    print(f"loading val split ...")
    val_samples = _load_split(val_ids)
    print(f"train={len(train_samples)}  val={len(val_samples)}")

    if args.smoke:
        train_samples = train_samples[:40]
        val_samples = val_samples[:20]
        args.epochs = 3
        print("SMOKE MODE: train=40, val=20, epochs=3")

    cfg = TrainConfig(
        fold=args.fold, num_layers=args.num_layers,
        temporal_skip_hops=list(args.temporal_skip_hops),
        hidden=args.hidden, use_delta_t=not args.no_delta_t,
        epochs=args.epochs, batch_accum=args.batch_accum,
        lr=args.lr, weight_decay=args.weight_decay,
        lambda_pos=args.lambda_pos, lambda_feat=args.lambda_feat,
        lambda_alive=args.lambda_alive, seed=args.seed,
        k_spatial=args.k_spatial, k_temporal=args.k_temporal,
        conv_type=args.conv_type, use_edge_gating=args.use_edge_gating,
        alive_loss_type=args.alive_loss_type,
        alive_pos_weight_scale=args.alive_pos_weight_scale,
    )
    with (args.out_dir / "config.json").open("w") as fh:
        json.dump({"cli": vars(args), "config": asdict(cfg)}, fh,
                  indent=2, default=str)

    result = train_and_eval(train_samples, val_samples, cfg,
                            device=args.device, out_dir=args.out_dir)
    with (args.out_dir / "metrics.json").open("w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
