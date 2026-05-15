#!/usr/bin/env python3
"""Rollout-aware training for the consistent forecaster (v2).

Adds three training upgrades on top of train_consistent_forecaster.py:
1) scheduled sampling (self-feeding),
2) multi-horizon cloud loss,
3) random-start conditioning (start from T0/T1/T2).
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lsgc.forecaster import LSGCForecaster
from experiments.stage1_forecaster.train_consistent_forecaster import (
    ConsistentSample,
    compute_stats,
    load_sample,
)
from experiments.stage1_forecaster.edge_modes import (
    EDGE_ATTR_MODE_CHOICES,
    EDGE_MODE_CHOICES,
    build_history_graph,
    edge_attr_dim,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VISITS = ("T0", "T1", "T2", "T3")


def _epsilon_schedule(epoch: int, warmup: int, anneal: int, eps_max: float) -> float:
    if epoch < warmup:
        return 0.0
    if anneal <= 0:
        return float(eps_max)
    progress = min(1.0, (epoch - warmup) / max(1, anneal))
    return float(eps_max * progress)


def sliced_wasserstein_tensor(
    X: Tensor, Y: Tensor, n_projections: int = 64, resample: int = 256
) -> Tensor:
    if X.shape[0] == 0 or Y.shape[0] == 0:
        return torch.zeros((), device=X.device, dtype=X.dtype)
    if X.shape[0] > resample:
        idx = torch.randperm(X.shape[0], device=X.device)[:resample]
        X = X[idx]
    if Y.shape[0] > resample:
        idx = torch.randperm(Y.shape[0], device=Y.device)[:resample]
        Y = Y[idx]
    dirs = torch.randn((n_projections, X.shape[1]), device=X.device, dtype=X.dtype)
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-8)
    px = X @ dirs.T
    py = Y @ dirs.T
    q = min(px.shape[0], py.shape[0])
    q_grid = torch.linspace(0, 1, q, device=X.device, dtype=X.dtype)
    qx = torch.quantile(px, q_grid, dim=0)
    qy = torch.quantile(py, q_grid, dim=0)
    return (qx - qy).abs().mean()


def chamfer_feat_loss(A: Tensor, B: Tensor) -> Tensor:
    if A.shape[0] == 0 or B.shape[0] == 0:
        return torch.zeros((), device=A.device, dtype=A.dtype)
    d = torch.cdist(A, B, p=2.0)
    return 0.5 * (d.min(dim=1).values.pow(2).mean() + d.min(dim=0).values.pow(2).mean())


def _raw_features(xn: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    return xn * std + mean


def _bio_rollout_loss(
    alive_logit: Tensor,
    pred_xn: Tensor,
    dst_xn: Tensor,
    dst_alive: Tensor,
    mean: Tensor,
    std: Tensor,
    *,
    volume_idx: int,
    lambda_alive_bce: float,
    lambda_alive_mass: float,
    lambda_ftv: float,
) -> tuple[Tensor, dict[str, float]]:
    """Aggregate alive/FTV penalties for one predicted visit."""
    total = torch.zeros((), device=pred_xn.device, dtype=pred_xn.dtype)
    stats = {
        "bio_alive_bce": float("nan"),
        "bio_alive_mass": float("nan"),
        "bio_log_ftv": float("nan"),
        "bio_pred_ftv_ml": float("nan"),
        "bio_obs_ftv_ml": float("nan"),
        "bio_pred_alive_mass": float("nan"),
        "bio_obs_alive_count": float("nan"),
    }
    alive_target = dst_alive.float().reshape(-1)
    alive_prob = torch.sigmoid(alive_logit.reshape(-1))
    pred_alive_mass = alive_prob.sum()
    obs_alive_count = alive_target.sum()

    if lambda_alive_bce > 0.0:
        alive_bce = F.binary_cross_entropy_with_logits(alive_logit.reshape(-1), alive_target)
        total = total + (lambda_alive_bce * alive_bce)
        stats["bio_alive_bce"] = float(alive_bce.detach().item())

    if lambda_alive_mass > 0.0:
        alive_mass = (pred_alive_mass - obs_alive_count).abs()
        total = total + (lambda_alive_mass * alive_mass)
        stats["bio_alive_mass"] = float(alive_mass.detach().item())

    if lambda_ftv > 0.0:
        pred_raw = _raw_features(pred_xn, mean, std)
        obs_raw = _raw_features(dst_xn, mean, std)
        pred_vol = pred_raw[:, volume_idx].clamp_min(0.0)
        obs_vol = obs_raw[:, volume_idx].clamp_min(0.0)
        pred_ftv = (pred_vol * alive_prob).sum()
        obs_ftv = obs_vol.sum()
        log_ftv = (torch.log1p(pred_ftv) - torch.log1p(obs_ftv)).abs()
        total = total + (lambda_ftv * log_ftv)
        stats["bio_log_ftv"] = float(log_ftv.detach().item())
        stats["bio_pred_ftv_ml"] = float(pred_ftv.detach().item())
        stats["bio_obs_ftv_ml"] = float(obs_ftv.detach().item())

    stats["bio_pred_alive_mass"] = float(pred_alive_mass.detach().item())
    stats["bio_obs_alive_count"] = float(obs_alive_count.detach().item())
    return total, stats


def _transition_tf_loss(
    dp: Tensor,
    dh: Tensor,
    src_pos: Tensor,
    dst_pos: Tensor,
    src_xn: Tensor,
    dst_xn: Tensor,
    dst_alive: Tensor,
    lambda_pos: float,
    lambda_feat: float,
) -> tuple[Tensor, float]:
    mask = dst_alive.bool()
    if mask.sum() == 0:
        z = torch.zeros((), device=dp.device, dtype=dp.dtype)
        return z, float("nan")
    gt_dp = (dst_pos - src_pos)[mask]
    gt_dh = (dst_xn - src_xn)[mask]
    pred_dp = dp[mask]
    pred_dh = dh[mask]
    l_pos = F.mse_loss(pred_dp, gt_dp)
    l_feat = F.mse_loss(pred_dh, gt_dh)
    pos_mae = float((pred_dp - gt_dp).norm(dim=-1).mean().detach().item())
    return lambda_pos * l_pos + lambda_feat * l_feat, pos_mae


def curriculum_forward_patient(
    model: LSGCForecaster,
    s: ConsistentSample,
    mean: Tensor,
    std: Tensor,
    *,
    epsilon: float,
    rng: random.Random,
    k_spatial: int,
    edge_mode: str,
    edge_attr_mode: str,
    lambda_pos: float,
    lambda_feat: float,
    lambda_cloud: float,
    lambda_cloud_feat: float,
    lambda_horizon: float,
    lambda_alive_bce: float,
    lambda_alive_mass: float,
    lambda_ftv: float,
    volume_idx: int,
    random_start: bool,
) -> tuple[Tensor, dict]:
    x_all = ((s.x.to(DEVICE) - mean) / std)
    pos_all = s.pos.to(DEVICE)
    alive_all = s.alive.to(DEVICE)
    off = [int(v) for v in s.visit_offsets.tolist()]
    T = len(off) - 1

    obs_xn = [x_all[off[v]:off[v + 1]] for v in range(T)]
    obs_pos = [pos_all[off[v]:off[v + 1]] for v in range(T)]
    obs_alive = [alive_all[off[v]:off[v + 1]] for v in range(T)]

    start = rng.randint(0, T - 2) if random_start else 0
    history_x = [obs_xn[v] for v in range(start + 1)]
    history_pos = [obs_pos[v] for v in range(start + 1)]
    history_t = [torch.full((obs_xn[v].shape[0],), float(v), device=DEVICE) for v in range(start + 1)]
    selffed_steps = 0
    maes: list[float] = []
    bio_stats: dict[str, list[float]] = {}
    total = torch.zeros((), device=DEVICE)

    for k in range(start, T - 1):
        x_cat = torch.cat(history_x, dim=0)
        p_cat = torch.cat(history_pos, dim=0)
        t_cat = torch.cat(history_t, dim=0)
        edge_index, edge_attr = build_history_graph(
            history_pos,
            history_x,
            k_spatial=k_spatial,
            edge_mode=edge_mode,
            edge_attr_mode=edge_attr_mode,
        )
        out = model(x_cat, p_cat, t_cat, edge_index, edge_attr=edge_attr)
        n_last = history_x[-1].shape[0]
        sl = slice(x_cat.shape[0] - n_last, x_cat.shape[0])
        dp = out["delta_pos"][sl]
        dh = out["delta_feat"][sl]
        alive_logit = out["alive_logit"][sl]

        pred_pos = history_pos[-1] + dp
        pred_xn = history_x[-1] + dh
        dst_pos = obs_pos[k + 1]
        dst_xn = obs_xn[k + 1]
        dst_alive = obs_alive[k + 1]

        use_self_feed = (k > start) and (rng.random() < epsilon)
        if use_self_feed:
            selffed_steps += 1
            l = lambda_cloud * sliced_wasserstein_tensor(pred_pos, dst_pos)
            if lambda_cloud_feat > 0.0:
                l = l + (lambda_cloud_feat * chamfer_feat_loss(pred_xn, dst_xn))
            total = total + l
            maes.append(float(sliced_wasserstein_tensor(pred_pos.detach(), dst_pos.detach()).item()))
            history_x.append(pred_xn)
            history_pos.append(pred_pos)
        else:
            l_tf, pos_mae = _transition_tf_loss(
                dp, dh,
                history_pos[-1], dst_pos,
                history_x[-1], dst_xn,
                dst_alive,
                lambda_pos=lambda_pos,
                lambda_feat=lambda_feat,
            )
            total = total + l_tf
            if np.isfinite(pos_mae):
                maes.append(float(pos_mae))
            history_x.append(dst_xn)
            history_pos.append(dst_pos)

        bio_l, bio_info = _bio_rollout_loss(
            alive_logit,
            pred_xn,
            dst_xn,
            dst_alive,
            mean,
            std,
            volume_idx=volume_idx,
            lambda_alive_bce=lambda_alive_bce,
            lambda_alive_mass=lambda_alive_mass,
            lambda_ftv=lambda_ftv,
        )
        total = total + bio_l
        for key, val in bio_info.items():
            if np.isfinite(val):
                bio_stats.setdefault(key, []).append(float(val))

        history_t.append(torch.full((history_x[-1].shape[0],), float(k + 1), device=DEVICE))

        if lambda_horizon > 0.0 and k == T - 2:
            total = total + (lambda_horizon * sliced_wasserstein_tensor(pred_pos, dst_pos))

    n_steps = max(1, T - 1 - start)
    bio_summary = {
        key: float(np.mean(vals)) if vals else float("nan")
        for key, vals in bio_stats.items()
    }
    return total / n_steps, {
        "start_visit": start,
        "n_self_fed": selffed_steps,
        "mean_step_mae": float(np.mean(maes)) if maes else float("nan"),
        **bio_summary,
    }


@torch.no_grad()
def rollout_eval_metrics(
    model: LSGCForecaster,
    s: ConsistentSample,
    mean: Tensor,
    std: Tensor,
    k_spatial: int,
    edge_mode: str,
    edge_attr_mode: str,
    volume_idx: int,
) -> dict[str, float]:
    x_all = ((s.x.to(DEVICE) - mean) / std)
    pos_all = s.pos.to(DEVICE)
    alive_all = s.alive.to(DEVICE)
    off = [int(v) for v in s.visit_offsets.tolist()]
    T = len(off) - 1

    obs_xn = [x_all[off[v]:off[v + 1]] for v in range(T)]
    obs_pos = [pos_all[off[v]:off[v + 1]] for v in range(T)]
    obs_alive = [alive_all[off[v]:off[v + 1]] for v in range(T)]

    history_x = [obs_xn[0]]
    history_pos = [obs_pos[0]]
    history_t = [torch.zeros((obs_xn[0].shape[0],), device=DEVICE)]
    maes: list[float] = []
    ftv_abs: list[float] = []
    ftv_bias: list[float] = []
    alive_abs: list[float] = []
    alive_bias: list[float] = []
    t3_ftv_abs = float("nan")
    t3_ftv_bias = float("nan")
    t3_alive_abs = float("nan")
    t3_alive_bias = float("nan")

    for k in range(T - 1):
        x_cat = torch.cat(history_x, dim=0)
        p_cat = torch.cat(history_pos, dim=0)
        t_cat = torch.cat(history_t, dim=0)
        edge_index, edge_attr = build_history_graph(
            history_pos,
            history_x,
            k_spatial=k_spatial,
            edge_mode=edge_mode,
            edge_attr_mode=edge_attr_mode,
        )
        out = model(x_cat, p_cat, t_cat, edge_index, edge_attr=edge_attr)
        n_last = history_x[-1].shape[0]
        sl = slice(x_cat.shape[0] - n_last, x_cat.shape[0])
        dp = out["delta_pos"][sl]
        dh = out["delta_feat"][sl]
        alive_prob = torch.sigmoid(out["alive_logit"][sl])

        pred_pos = history_pos[-1] + dp
        pred_xn = history_x[-1] + dh
        dst_pos = obs_pos[k + 1]
        dst_alive = obs_alive[k + 1].bool()
        if dst_alive.sum() > 0:
            diff = (pred_pos - dst_pos)[dst_alive]
            maes.append(float(diff.norm(dim=-1).mean().item()))

        pred_raw = _raw_features(pred_xn, mean, std)
        obs_raw = _raw_features(obs_xn[k + 1], mean, std)
        pred_ftv = (pred_raw[:, volume_idx].clamp_min(0.0) * alive_prob).sum()
        obs_ftv = obs_raw[:, volume_idx].clamp_min(0.0).sum()
        ftv_err = float((pred_ftv - obs_ftv).detach().item())
        alive_err = float((alive_prob.sum() - dst_alive.float().sum()).detach().item())
        ftv_abs.append(abs(ftv_err))
        ftv_bias.append(ftv_err)
        alive_abs.append(abs(alive_err))
        alive_bias.append(alive_err)
        if k == T - 2:
            t3_ftv_abs = abs(ftv_err)
            t3_ftv_bias = ftv_err
            t3_alive_abs = abs(alive_err)
            t3_alive_bias = alive_err
        history_x.append(pred_xn)
        history_pos.append(pred_pos)
        history_t.append(torch.full((pred_xn.shape[0],), float(k + 1), device=DEVICE))

    return {
        "rollout_mae_mm": float(np.mean(maes)) if maes else float("nan"),
        "rollout_ftv_abs_err_ml": float(np.mean(ftv_abs)) if ftv_abs else float("nan"),
        "rollout_ftv_bias_ml": float(np.mean(ftv_bias)) if ftv_bias else float("nan"),
        "rollout_alive_abs_err": float(np.mean(alive_abs)) if alive_abs else float("nan"),
        "rollout_alive_bias": float(np.mean(alive_bias)) if alive_bias else float("nan"),
        "t3_ftv_abs_err_ml": t3_ftv_abs,
        "t3_ftv_bias_ml": t3_ftv_bias,
        "t3_alive_abs_err": t3_alive_abs,
        "t3_alive_bias": t3_alive_bias,
    }


@torch.no_grad()
def rollout_eval_mae(
    model: LSGCForecaster,
    s: ConsistentSample,
    mean: Tensor,
    std: Tensor,
    k_spatial: int,
    edge_mode: str,
    edge_attr_mode: str,
) -> float:
    return rollout_eval_metrics(
        model,
        s,
        mean,
        std,
        k_spatial,
        edge_mode=edge_mode,
        edge_attr_mode=edge_attr_mode,
        volume_idx=1,
    )["rollout_mae_mm"]


@dataclass
class Config:
    graphs_root: str = "datasets/ispy2/graphs_consistent"
    patient_list: str = "reports/all_patients_4visit.txt"
    out_dir: str = "runs/consistent_forecaster_v2"
    pretrained_ckpt: str = "experiments/stage1_forecaster/runs/bio_gated_5fold/fold0/best.pt"
    hidden: int = 64
    num_layers: int = 2
    k_spatial: int = 8
    edge_mode: str = "full"
    edge_attr_mode: str = "none"
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 180
    patience: int = 40
    val_frac: float = 0.2
    seed: int = 42
    freeze_backbone_epochs: int = 30
    fold: int = -1
    folds_parquet: str = "datasets/ispy2/folds.parquet"

    scheduled_sampling: bool = False
    eps_max: float = 0.5
    eps_warmup_epochs: int = 60
    eps_anneal_epochs: int = 60
    random_start: bool = False
    lambda_pos: float = 1.0
    lambda_feat: float = 0.5
    lambda_cloud: float = 1.0
    lambda_cloud_feat: float = 0.0
    lambda_horizon: float = 0.0
    lambda_alive_bce: float = 0.0
    lambda_alive_mass: float = 0.0
    lambda_ftv: float = 0.0
    volume_idx: int = 1
    bio_warmup_epochs: int = 0
    bio_anneal_epochs: int = 0
    selection_ftv_weight: float = 0.0
    selection_alive_weight: float = 0.0
    patient_limit: int = 0


def train(cfg: Config):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pids = [p.strip() for p in Path(cfg.patient_list).read_text().splitlines() if p.strip()]
    if cfg.patient_limit > 0:
        pids = pids[: cfg.patient_limit]
    samples_all: list[ConsistentSample] = []
    for pid in pids:
        s = load_sample(Path(cfg.graphs_root) / f"{pid}.pt")
        if s is not None:
            samples_all.append(s)
    print(f"Loaded {len(samples_all)}/{len(pids)} graphs")
    if not samples_all:
        raise RuntimeError("No samples loaded.")

    if cfg.fold >= 0:
        import pandas as pd

        folds_df = pd.read_parquet(cfg.folds_parquet)
        fold_map = dict(zip(folds_df["patient_id"], folds_df["fold"]))
        val_samples = [s for s in samples_all if fold_map.get(s.patient_id) == cfg.fold]
        train_samples = [s for s in samples_all if fold_map.get(s.patient_id) != cfg.fold]
        print(f"Fold {cfg.fold}/5 Train: {len(train_samples)} Val: {len(val_samples)}")
    else:
        random.shuffle(samples_all)
        n_val = max(1, int(len(samples_all) * cfg.val_frac))
        val_samples = samples_all[:n_val]
        train_samples = samples_all[n_val:]
        print(f"Train: {len(train_samples)} Val: {len(val_samples)}")

    mean, std = compute_stats(train_samples)
    mean, std = mean.to(DEVICE), std.to(DEVICE)
    in_ch = samples_all[0].x.shape[1]

    ckpt_path = Path(cfg.pretrained_ckpt)
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        s1_in_ch = int(ck["state_dict"]["embed.weight"].shape[1])
        print(f"Pretrained checkpoint in_ch={s1_in_ch} (target in_ch={in_ch})")
    else:
        ck = None
        s1_in_ch = -1
        print("No pretrained checkpoint found — training from scratch")

    model = LSGCForecaster(
        in_channels=in_ch,
        hidden=cfg.hidden,
        num_layers=cfg.num_layers,
        feat_out_dim=in_ch,
        use_delta_t=True,
        use_edge_gating=True,
        edge_attr_dim=edge_attr_dim(cfg.edge_attr_mode),
    ).to(DEVICE)

    if ck is not None and s1_in_ch == in_ch:
        sd = model.state_dict()
        loaded = 0
        for k, v in ck["state_dict"].items():
            if k in sd and sd[k].shape == v.shape:
                sd[k] = v
                loaded += 1
        model.load_state_dict(sd)
        print(f"Warm-start loaded {loaded} tensors")
    else:
        print("Skipping warm-start (dim mismatch or missing checkpoint)")

    def _set_backbone_grad(requires_grad: bool):
        for name, p in model.named_parameters():
            if name.startswith("embed") or name.startswith("convs"):
                p.requires_grad_(requires_grad)

    _set_backbone_grad(False)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    best_val = float("inf")
    patience_cnt = 0
    log: list[dict] = []
    rng = random.Random(cfg.seed + 123)

    for epoch in range(1, cfg.epochs + 1):
        if epoch == cfg.freeze_backbone_epochs + 1:
            _set_backbone_grad(True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr * 0.3, weight_decay=cfg.weight_decay)
            print(f"[Epoch {epoch}] Unfreezing backbone, lr={cfg.lr * 0.3:.1e}")

        eps = (
            _epsilon_schedule(epoch, cfg.eps_warmup_epochs, cfg.eps_anneal_epochs, cfg.eps_max)
            if cfg.scheduled_sampling
            else 0.0
        )
        bio_scale = _epsilon_schedule(epoch, cfg.bio_warmup_epochs, cfg.bio_anneal_epochs, 1.0)
        lambda_alive_bce = cfg.lambda_alive_bce * bio_scale
        lambda_alive_mass = cfg.lambda_alive_mass * bio_scale
        lambda_ftv = cfg.lambda_ftv * bio_scale

        model.train()
        random.shuffle(train_samples)
        tr_loss, tr_step_mae, tr_selffed = [], [], []
        tr_bio: dict[str, list[float]] = {}
        for s in train_samples:
            optimizer.zero_grad()
            loss, info = curriculum_forward_patient(
                model,
                s,
                mean,
                std,
                epsilon=eps,
                rng=rng,
                k_spatial=cfg.k_spatial,
                edge_mode=cfg.edge_mode,
                edge_attr_mode=cfg.edge_attr_mode,
                lambda_pos=cfg.lambda_pos,
                lambda_feat=cfg.lambda_feat,
                lambda_cloud=cfg.lambda_cloud,
                lambda_cloud_feat=cfg.lambda_cloud_feat,
                lambda_horizon=cfg.lambda_horizon,
                lambda_alive_bce=lambda_alive_bce,
                lambda_alive_mass=lambda_alive_mass,
                lambda_ftv=lambda_ftv,
                volume_idx=cfg.volume_idx,
                random_start=cfg.random_start,
            )
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            tr_loss.append(loss.item())
            if np.isfinite(info["mean_step_mae"]):
                tr_step_mae.append(info["mean_step_mae"])
            tr_selffed.append(info["n_self_fed"])
            for key, val in info.items():
                if key.startswith("bio_") and np.isfinite(val):
                    tr_bio.setdefault(key, []).append(float(val))

        model.eval()
        val_loss = []
        val_metrics: dict[str, list[float]] = {}
        with torch.no_grad():
            for s in val_samples:
                l, _ = curriculum_forward_patient(
                    model,
                    s,
                    mean,
                    std,
                    epsilon=0.0,
                    rng=rng,
                    k_spatial=cfg.k_spatial,
                    edge_mode=cfg.edge_mode,
                    edge_attr_mode=cfg.edge_attr_mode,
                    lambda_pos=cfg.lambda_pos,
                    lambda_feat=cfg.lambda_feat,
                    lambda_cloud=cfg.lambda_cloud,
                    lambda_cloud_feat=cfg.lambda_cloud_feat,
                    lambda_horizon=cfg.lambda_horizon,
                    lambda_alive_bce=lambda_alive_bce,
                    lambda_alive_mass=lambda_alive_mass,
                    lambda_ftv=lambda_ftv,
                    volume_idx=cfg.volume_idx,
                    random_start=False,
                )
                val_loss.append(l.item())
                metrics = rollout_eval_metrics(
                    model,
                    s,
                    mean,
                    std,
                    cfg.k_spatial,
                    edge_mode=cfg.edge_mode,
                    edge_attr_mode=cfg.edge_attr_mode,
                    volume_idx=cfg.volume_idx,
                )
                for key, val in metrics.items():
                    if np.isfinite(val):
                        val_metrics.setdefault(key, []).append(float(val))

        tr_l = float(np.mean(tr_loss)) if tr_loss else float("nan")
        vl_l = float(np.mean(val_loss)) if val_loss else float("nan")
        tr_m = float(np.mean(tr_step_mae)) if tr_step_mae else float("nan")
        val_means = {
            key: float(np.mean(vals)) if vals else float("nan")
            for key, vals in val_metrics.items()
        }
        train_bio_means = {
            f"train_{key}": float(np.mean(vals)) if vals else float("nan")
            for key, vals in tr_bio.items()
        }
        vl_roll = val_means.get("rollout_mae_mm", float("nan"))
        tr_sf = float(np.mean(tr_selffed)) if tr_selffed else 0.0

        log_row = {
            "epoch": epoch,
            "epsilon": eps,
            "bio_scale": bio_scale,
            "lambda_alive_bce_eff": lambda_alive_bce,
            "lambda_alive_mass_eff": lambda_alive_mass,
            "lambda_ftv_eff": lambda_ftv,
            "train_loss": tr_l,
            "val_loss": vl_l,
            "train_step_mae_mm": tr_m,
            "val_rollout_mae_mm": vl_roll,
            "train_self_fed_steps_mean": tr_sf,
            **{f"val_{key}": val for key, val in val_means.items()},
            **train_bio_means,
        }
        log.append(log_row)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d} eps={eps:.3f} train_loss={tr_l:.4f} val_loss={vl_l:.4f} "
                f"train_step_mae={tr_m:.3f} val_rollout_mae={vl_roll:.3f} "
                f"val_t3_ftv_abs={val_means.get('t3_ftv_abs_err_ml', float('nan')):.3f} "
                f"val_t3_alive_abs={val_means.get('t3_alive_abs_err', float('nan')):.3f} "
                f"selffed={tr_sf:.2f}"
            )

        # Early-stop on a geometry-first composite, with optional biology terms.
        score = vl_roll if np.isfinite(vl_roll) else vl_l
        if np.isfinite(score):
            t3_ftv_abs = val_means.get("t3_ftv_abs_err_ml", float("nan"))
            t3_alive_abs = val_means.get("t3_alive_abs_err", float("nan"))
            if cfg.selection_ftv_weight > 0.0 and np.isfinite(t3_ftv_abs):
                score += cfg.selection_ftv_weight * t3_ftv_abs
            if cfg.selection_alive_weight > 0.0 and np.isfinite(t3_alive_abs):
                score += cfg.selection_alive_weight * t3_alive_abs
        if score < best_val:
            best_val = score
            patience_cnt = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "mean": mean.cpu(),
                    "std": std.cpu(),
                    "config": asdict(cfg),
                    "epoch": epoch,
                    "val_loss": vl_l,
                    "val_rollout_mae_mm": vl_roll,
                    "val_t3_ftv_abs_err_ml": val_means.get("t3_ftv_abs_err_ml", float("nan")),
                    "val_t3_ftv_bias_ml": val_means.get("t3_ftv_bias_ml", float("nan")),
                    "val_t3_alive_abs_err": val_means.get("t3_alive_abs_err", float("nan")),
                    "val_t3_alive_bias": val_means.get("t3_alive_bias", float("nan")),
                    "selection_score": score,
                },
                out_dir / "best.pt",
            )
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.patience:
                print(f"Early stop at epoch {epoch}")
                break

    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean": mean.cpu(),
            "std": std.cpu(),
            "config": asdict(cfg),
        },
        out_dir / "last.pt",
    )
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))
    print(f"Best val rollout MAE: {best_val:.4f} mm")
    print(f"Outputs: {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs-root", default="datasets/ispy2/graphs_consistent")
    ap.add_argument("--patient-list", default="reports/all_patients_4visit.txt")
    ap.add_argument("--out-dir", default="runs/consistent_forecaster_v2")
    ap.add_argument(
        "--pretrained",
        default="experiments/stage1_forecaster/runs/bio_gated_5fold/fold0/best.pt",
    )
    ap.add_argument("--epochs", type=int, default=180)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--freeze-backbone-epochs", type=int, default=30)
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--folds-parquet", default="datasets/ispy2/folds.parquet")

    ap.add_argument("--k-spatial", type=int, default=8)
    ap.add_argument(
        "--edge-mode",
        choices=EDGE_MODE_CHOICES,
        default="full",
        help="Dynamic intra-visit edge mode; all non-none modes keep temporal identity links.",
    )
    ap.add_argument(
        "--edge-attr-mode",
        choices=EDGE_ATTR_MODE_CHOICES,
        default="none",
        help="Optional dynamic edge attributes. radial_bio adds tumor-centered geometry and feature-contrast channels.",
    )
    ap.add_argument("--lambda-pos", type=float, default=1.0)
    ap.add_argument("--lambda-feat", type=float, default=0.5)
    ap.add_argument("--lambda-cloud", type=float, default=1.0)
    ap.add_argument("--lambda-cloud-feat", type=float, default=0.0)
    ap.add_argument("--lambda-horizon", type=float, default=0.0)
    ap.add_argument("--lambda-alive-bce", type=float, default=0.0)
    ap.add_argument("--lambda-alive-mass", type=float, default=0.0)
    ap.add_argument("--lambda-ftv", type=float, default=0.0)
    ap.add_argument("--volume-idx", type=int, default=1)
    ap.add_argument("--bio-warmup-epochs", type=int, default=0)
    ap.add_argument("--bio-anneal-epochs", type=int, default=0)
    ap.add_argument("--selection-ftv-weight", type=float, default=0.0)
    ap.add_argument("--selection-alive-weight", type=float, default=0.0)
    ap.add_argument("--patient-limit", type=int, default=0)

    ap.add_argument("--scheduled-sampling", action="store_true")
    ap.add_argument("--eps-max", type=float, default=0.5)
    ap.add_argument("--eps-warmup-epochs", type=int, default=60)
    ap.add_argument("--eps-anneal-epochs", type=int, default=60)
    ap.add_argument("--random-start", action="store_true")
    args = ap.parse_args()

    cfg = Config(
        graphs_root=args.graphs_root,
        patient_list=args.patient_list,
        out_dir=args.out_dir,
        pretrained_ckpt=args.pretrained,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        num_layers=args.num_layers,
        patience=args.patience,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        fold=args.fold,
        folds_parquet=args.folds_parquet,
        k_spatial=args.k_spatial,
        edge_mode=args.edge_mode,
        edge_attr_mode=args.edge_attr_mode,
        lambda_pos=args.lambda_pos,
        lambda_feat=args.lambda_feat,
        lambda_cloud=args.lambda_cloud,
        lambda_cloud_feat=args.lambda_cloud_feat,
        lambda_horizon=args.lambda_horizon,
        lambda_alive_bce=args.lambda_alive_bce,
        lambda_alive_mass=args.lambda_alive_mass,
        lambda_ftv=args.lambda_ftv,
        volume_idx=args.volume_idx,
        bio_warmup_epochs=args.bio_warmup_epochs,
        bio_anneal_epochs=args.bio_anneal_epochs,
        selection_ftv_weight=args.selection_ftv_weight,
        selection_alive_weight=args.selection_alive_weight,
        patient_limit=args.patient_limit,
        scheduled_sampling=args.scheduled_sampling,
        eps_max=args.eps_max,
        eps_warmup_epochs=args.eps_warmup_epochs,
        eps_anneal_epochs=args.eps_anneal_epochs,
        random_start=args.random_start,
    )
    train(cfg)


if __name__ == "__main__":
    main()
