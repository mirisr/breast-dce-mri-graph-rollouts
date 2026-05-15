"""Dynamic edge construction for consistent graph rollout experiments."""

from __future__ import annotations

import re

import torch
from torch import Tensor


EDGE_MODE_CHOICES = (
    "full",
    "spatial_knn",
    "none",
    "radial_knn",
    "feature_knn",
    "feature_knn_all",
    "feature_knn_pe_ser",
    "feature_knn_volume",
    "hybrid_spatial_feature_a25",
    "hybrid_spatial_feature_a50",
    "hybrid_spatial_feature_a75",
)

EDGE_ATTR_MODE_CHOICES = (
    "none",
    "radial_geometry",
    "radial_bio",
)


def edge_attr_dim(edge_attr_mode: str = "none") -> int:
    """Return the per-edge attribute width for a named encoding."""
    mode = str(edge_attr_mode)
    if mode == "none":
        return 0
    if mode == "radial_geometry":
        return 9
    if mode == "radial_bio":
        return 12
    raise ValueError(f"unknown edge_attr_mode={edge_attr_mode!r}")


def _empty_edges(device: torch.device) -> Tensor:
    return torch.zeros((2, 0), dtype=torch.long, device=device)


def _empty_edge_attr(device: torch.device, dim: int) -> Tensor:
    return torch.zeros((0, int(dim)), dtype=torch.float32, device=device)


def _positive_scale(d: Tensor) -> Tensor:
    vals = d[torch.isfinite(d) & (d > 0)]
    if vals.numel() == 0:
        return torch.ones((), dtype=d.dtype, device=d.device)
    return vals.median().clamp_min(1e-6)


def _knn_from_distance(d: Tensor, k: int) -> Tensor:
    n = int(d.shape[0])
    if n <= 1 or k <= 0:
        return _empty_edges(d.device)
    d = d.clone()
    d.fill_diagonal_(float("inf"))
    k_eff = min(int(k), n - 1)
    _, idx = torch.topk(d, k=k_eff, largest=False, dim=1)
    src = torch.arange(n, device=d.device).unsqueeze(1).expand_as(idx).reshape(-1)
    dst = idx.reshape(-1)
    ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    return ei.unique(dim=1)


def _feature_view(x: Tensor, mode: str) -> Tensor:
    if mode == "feature_knn_pe_ser":
        # Current feature order: voxel_count_log, volume_ml, pe_mean, pe_std, ser_mean, ser_std.
        if x.shape[1] >= 6:
            return x[:, 2:6]
        return x
    if mode == "feature_knn_volume":
        if x.shape[1] >= 2:
            return x[:, 0:2]
        return x
    return x


def _hybrid_alpha(mode: str) -> float:
    m = re.search(r"_a(\d+)$", mode)
    if not m:
        return 0.50
    return float(int(m.group(1))) / 100.0


def _visit_edges(pos: Tensor, x: Tensor | None, k: int, mode: str) -> Tensor:
    if mode in {"full", "spatial_knn"}:
        return _knn_from_distance(torch.cdist(pos, pos), k)

    if mode == "radial_knn":
        radius = pos.norm(dim=1, keepdim=True)
        return _knn_from_distance(torch.cdist(radius, radius), k)

    if mode.startswith("feature_knn"):
        if x is None:
            raise ValueError(f"edge_mode={mode!r} requires history_x")
        return _knn_from_distance(torch.cdist(_feature_view(x, mode), _feature_view(x, mode)), k)

    if mode.startswith("hybrid_spatial_feature"):
        if x is None:
            raise ValueError(f"edge_mode={mode!r} requires history_x")
        alpha = _hybrid_alpha(mode)
        d_pos = torch.cdist(pos, pos)
        d_feat = torch.cdist(x, x)
        score = alpha * (d_pos / _positive_scale(d_pos)) + (1.0 - alpha) * (d_feat / _positive_scale(d_feat))
        return _knn_from_distance(score, k)

    raise ValueError(f"unknown edge_mode={mode!r}")


def build_history_edges(
    history_pos: list[Tensor],
    history_x: list[Tensor] | None = None,
    *,
    k_spatial: int = 8,
    edge_mode: str = "full",
) -> Tensor:
    """Build dynamic rollout edges for a list of visit histories.

    All non-``none`` modes keep deterministic identity temporal edges between
    visits. The ``edge_mode`` only changes intra-visit neighbor definition.
    """
    if not history_pos:
        raise ValueError("history_pos must contain at least one visit")
    mode = "spatial_knn" if edge_mode == "full" else str(edge_mode)
    if mode == "none":
        return _empty_edges(history_pos[0].device)
    if mode not in EDGE_MODE_CHOICES:
        raise ValueError(f"unknown edge_mode={edge_mode!r}")
    if history_x is None:
        history_x = [None] * len(history_pos)  # type: ignore[list-item]
    if len(history_x) != len(history_pos):
        raise ValueError("history_x and history_pos must have the same number of visits")

    all_src: list[Tensor] = []
    all_dst: list[Tensor] = []
    offsets: list[int] = []
    cur = 0
    for pos in history_pos:
        offsets.append(cur)
        cur += int(pos.shape[0])
    offsets.append(cur)

    for v, pos in enumerate(history_pos):
        off = offsets[v]
        n_v = int(pos.shape[0])
        if n_v > 1 and k_spatial > 0:
            ei_v = _visit_edges(pos, history_x[v], int(k_spatial), mode)
            if ei_v.numel():
                all_src.append(ei_v[0] + off)
                all_dst.append(ei_v[1] + off)
        if v < len(history_pos) - 1:
            n_next = int(history_pos[v + 1].shape[0])
            n_link = min(n_v, n_next)
            if n_link > 0:
                src_t = torch.arange(n_link, device=pos.device) + off
                dst_t = torch.arange(n_link, device=pos.device) + offsets[v + 1]
                all_src += [src_t, dst_t]
                all_dst += [dst_t, src_t]

    if not all_src:
        return _empty_edges(history_pos[0].device)
    return torch.stack([torch.cat(all_src), torch.cat(all_dst)], dim=0).long()


def _edge_attr_from_index(
    pos: Tensor,
    x: Tensor | None,
    t: Tensor,
    edge_index: Tensor,
    *,
    edge_attr_mode: str = "none",
) -> Tensor | None:
    mode = str(edge_attr_mode)
    dim = edge_attr_dim(mode)
    if dim == 0:
        return None
    if edge_index.numel() == 0:
        return _empty_edge_attr(pos.device, dim).to(dtype=pos.dtype)

    src = edge_index[0].long()
    dst = edge_index[1].long()
    p_src = pos[src]
    p_dst = pos[dst]
    delta = p_src - p_dst
    dist = delta.norm(dim=-1, keepdim=True)
    r_src = p_src.norm(dim=-1, keepdim=True)
    r_dst = p_dst.norm(dim=-1, keepdim=True)
    radial_delta = r_src - r_dst

    dist_scale = _positive_scale(dist)
    radius_scale = _positive_scale(torch.cat([r_src, r_dst], dim=0))
    radial_delta_scale = _positive_scale(radial_delta.abs())
    cos_radial = (p_src * p_dst).sum(dim=-1, keepdim=True) / (r_src * r_dst + 1e-6)
    same_shell = torch.exp(-radial_delta.abs() / radial_delta_scale)
    dt = (t[src].float().view(-1, 1) - t[dst].float().view(-1, 1)).abs()
    dt_scale = _positive_scale(dt)
    is_temporal = (dt > 0).to(dtype=pos.dtype)

    attrs = [
        (dist / dist_scale).clamp(0.0, 6.0),
        (radial_delta.abs() / radial_delta_scale).clamp(0.0, 6.0),
        (radial_delta / radial_delta_scale).clamp(-6.0, 6.0),
        (r_src / radius_scale).clamp(0.0, 6.0),
        (r_dst / radius_scale).clamp(0.0, 6.0),
        cos_radial.clamp(-1.0, 1.0),
        same_shell.clamp(0.0, 1.0),
        (dt / dt_scale).clamp(0.0, 6.0),
        is_temporal,
    ]

    if mode == "radial_bio":
        if x is None:
            x = torch.zeros((pos.shape[0], 0), dtype=pos.dtype, device=pos.device)
        if x.shape[1] >= 2:
            vol_diff = (x[src, 0:2] - x[dst, 0:2]).norm(dim=-1, keepdim=True)
        else:
            vol_diff = torch.zeros_like(dist)
        if x.shape[1] >= 6:
            pe_ser_diff = (x[src, 2:6] - x[dst, 2:6]).norm(dim=-1, keepdim=True)
        else:
            pe_ser_diff = torch.zeros_like(dist)
        if x.shape[1] > 0:
            feat_diff = (x[src] - x[dst]).norm(dim=-1, keepdim=True)
        else:
            feat_diff = torch.zeros_like(dist)
        attrs += [
            (vol_diff / _positive_scale(vol_diff)).clamp(0.0, 6.0),
            (pe_ser_diff / _positive_scale(pe_ser_diff)).clamp(0.0, 6.0),
            (feat_diff / _positive_scale(feat_diff)).clamp(0.0, 6.0),
        ]

    out = torch.cat(attrs, dim=-1).to(dtype=pos.dtype)
    if out.shape[1] != dim:
        raise RuntimeError(f"edge_attr_mode={mode!r} produced width {out.shape[1]} not {dim}")
    return out


def build_history_graph(
    history_pos: list[Tensor],
    history_x: list[Tensor] | None = None,
    *,
    k_spatial: int = 8,
    edge_mode: str = "full",
    edge_attr_mode: str = "none",
) -> tuple[Tensor, Tensor | None]:
    """Build dynamic rollout edges plus optional geometric/biology edge attributes."""
    edge_index = build_history_edges(
        history_pos,
        history_x,
        k_spatial=k_spatial,
        edge_mode=edge_mode,
    )
    if edge_attr_dim(edge_attr_mode) == 0:
        return edge_index, None
    pos_cat = torch.cat(history_pos, dim=0)
    x_cat = torch.cat(history_x, dim=0) if history_x is not None else None
    t_cat = torch.cat([
        torch.full((p.shape[0],), float(v), dtype=pos_cat.dtype, device=pos_cat.device)
        for v, p in enumerate(history_pos)
    ])
    edge_attr = _edge_attr_from_index(
        pos_cat,
        x_cat,
        t_cat,
        edge_index,
        edge_attr_mode=edge_attr_mode,
    )
    return edge_index, edge_attr
