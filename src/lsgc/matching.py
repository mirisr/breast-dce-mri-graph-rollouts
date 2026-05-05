"""Inter-visit supervoxel matching for the Stage-1 forecaster.

Given two consecutive visits with per-supervoxel positions ``p`` (mm) and
features ``h``, we need a pairing
``M = {(i, j) : i in visit_k, j in visit_{k+1}}`` plus an "alive" label
for every source supervoxel ``i`` (``1`` if successfully matched, ``0`` if
the cost to its best partner exceeds a threshold and we treat it as died).

Two matchers are supported:

* ``sinkhorn``  --  entropic optimal transport on the mixed cost matrix
  ``C_{ij} = alpha * ||p_i - p_j||_2^2 + beta * ||h_i - h_j||_2^2``.
  Hard pairs come from the row-argmax of the transport plan, same as the
  proposal specifies.
* ``nn``        --  plain one-way nearest-neighbor on the same cost.
  Used as an ablation baseline; also falls back to this when the smaller
  visit has <3 supervoxels (Sinkhorn is brittle on tiny problems).

Alive-label rule: an ``i`` is considered "alive at k+1" iff the cost to
its assigned partner is <= ``cost_quantile_alive`` of all matched-pair
costs for that transition (default 0.90, i.e. the top-10 % of costs are
flagged as implausible matches -> dead). The threshold is per-patient
per-transition to stay scale-free.

This module is pure numpy + torch; no POT dependency.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


# --------------------------------------------------------------------------- #
# Core algorithms                                                             #
# --------------------------------------------------------------------------- #


def cost_matrix(
    p_k: Tensor, h_k: Tensor,
    p_kp1: Tensor, h_kp1: Tensor,
    alpha: float = 1.0, beta: float = 1.0,
) -> Tensor:
    """Mixed position + feature squared-Euclidean cost. Shape (n_k, n_{k+1})."""
    dp = torch.cdist(p_k, p_kp1) ** 2
    dh = torch.cdist(h_k, h_kp1) ** 2
    return alpha * dp + beta * dh


def sinkhorn_plan(
    C: Tensor, *, reg: float = 0.1, n_iters: int = 200,
    a: Tensor | None = None, b: Tensor | None = None,
) -> Tensor:
    """Entropic OT via log-stabilized Sinkhorn. Returns a transport plan pi.

    ``a`` and ``b`` default to uniform marginals. ``reg`` is the entropy
    coefficient -- smaller = sharper plan, more numerical pain. We
    autoscale it by the mean cost so the algorithm is robust to the
    absolute scale of the cost matrix.
    """
    n, m = C.shape
    if a is None:
        a = torch.full((n,), 1.0 / n, device=C.device, dtype=C.dtype)
    if b is None:
        b = torch.full((m,), 1.0 / m, device=C.device, dtype=C.dtype)

    reg_eff = reg * C.mean().item()
    reg_eff = max(reg_eff, 1e-3)  # avoid underflow on very easy problems

    log_K = -C / reg_eff  # (n, m)
    log_u = torch.zeros(n, device=C.device, dtype=C.dtype)
    log_v = torch.zeros(m, device=C.device, dtype=C.dtype)
    log_a = torch.log(a + 1e-30)
    log_b = torch.log(b + 1e-30)

    for _ in range(n_iters):
        log_u = log_a - torch.logsumexp(log_K + log_v[None, :], dim=1)
        log_v = log_b - torch.logsumexp(log_K + log_u[:, None], dim=0)
    log_pi = log_u[:, None] + log_K + log_v[None, :]
    return torch.exp(log_pi)


def nearest_neighbor_plan(C: Tensor) -> Tensor:
    """Hard one-way NN: each source row assigns probability 1 to its argmin.

    The returned matrix is a transport "plan" in the same shape as
    Sinkhorn's but rank-deficient (one non-zero per row). Lets the rest
    of the pipeline treat both matchers uniformly.
    """
    n, m = C.shape
    idx = torch.argmin(C, dim=1)
    plan = torch.zeros_like(C)
    plan[torch.arange(n), idx] = 1.0 / n  # mass 1/n per row so it sums to 1
    return plan


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


@dataclass
class VisitMatch:
    """Matching result between visit_k and visit_{k+1}.

    Attributes
    ----------
    target_idx     (n_k,) long, argmax partner index in visit_{k+1};
                   -1 where ``alive==0`` (no usable partner).
    pair_cost      (n_k,) float, cost of the assigned pair.
    alive          (n_k,) bool, True if the match is within the keep-threshold.
    threshold      float, cost threshold above which we call the node dead.
    method         "sinkhorn" | "nn".
    """
    target_idx: Tensor
    pair_cost: Tensor
    alive: Tensor
    threshold: float
    method: str


def match_visits(
    p_k: Tensor, h_k: Tensor,
    p_kp1: Tensor, h_kp1: Tensor,
    *,
    method: str = "sinkhorn",
    alpha: float = 1.0, beta: float = 1.0,
    sinkhorn_reg: float = 0.1, sinkhorn_iters: int = 200,
    cost_quantile_alive: float = 0.90,
) -> VisitMatch:
    """Return hard pairings from visit_k -> visit_{k+1}.

    Matching quality is unfortunately scale-dependent: ``p`` is in
    millimeters (range ~200) and ``h`` is 5 features on roughly unit
    scale. In practice we standardize ``h`` across the union before
    calling this so ``alpha=beta=1`` is sensible.

    Degenerate cases: if either side has 0 supervoxels the result is
    empty (all-dead for source). If the smaller side has <=2 we fall
    back to ``nn`` -- Sinkhorn is unstable on tiny inputs.
    """
    if method not in {"sinkhorn", "nn"}:
        raise ValueError(f"unknown matcher {method!r}")

    n, m = p_k.shape[0], p_kp1.shape[0]
    if n == 0:
        device = p_k.device
        return VisitMatch(
            target_idx=torch.empty(0, dtype=torch.long, device=device),
            pair_cost=torch.empty(0, dtype=p_k.dtype, device=device),
            alive=torch.empty(0, dtype=torch.bool, device=device),
            threshold=0.0, method=method,
        )
    if m == 0:
        return VisitMatch(
            target_idx=torch.full((n,), -1, dtype=torch.long, device=p_k.device),
            pair_cost=torch.full((n,), float("inf"), dtype=p_k.dtype, device=p_k.device),
            alive=torch.zeros(n, dtype=torch.bool, device=p_k.device),
            threshold=0.0, method=method,
        )

    C = cost_matrix(p_k, h_k, p_kp1, h_kp1, alpha=alpha, beta=beta)

    effective_method = method
    if method == "sinkhorn" and min(n, m) <= 2:
        effective_method = "nn"  # fallback

    if effective_method == "sinkhorn":
        pi = sinkhorn_plan(C, reg=sinkhorn_reg, n_iters=sinkhorn_iters)
        target_idx = torch.argmax(pi, dim=1)
    else:
        target_idx = torch.argmin(C, dim=1)

    pair_cost = C[torch.arange(n, device=C.device), target_idx]
    threshold = torch.quantile(pair_cost, cost_quantile_alive).item()
    alive = pair_cost <= threshold
    target_idx = torch.where(alive, target_idx, torch.full_like(target_idx, -1))

    return VisitMatch(
        target_idx=target_idx.long(),
        pair_cost=pair_cost,
        alive=alive,
        threshold=float(threshold),
        method=effective_method,
    )


def standardize_features(h_list: list[Tensor]) -> list[Tensor]:
    """Per-visit-bundle z-score so alpha=beta=1 is sensible in the cost.

    Computes mean/std over the *concatenation* of all visits for one
    patient so inter-visit feature deltas still carry the right
    discriminative structure (we don't want per-visit standardization to
    erase the signal we're trying to learn).
    """
    if not h_list:
        return h_list
    allh = torch.cat(h_list, dim=0)
    mean = allh.mean(dim=0)
    std = allh.std(dim=0).clamp_min(1e-6)
    return [(h - mean) / std for h in h_list]
