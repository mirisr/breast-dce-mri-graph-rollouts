"""Unit tests for the inter-visit matching helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from lsgc.matching import (  # noqa: E402
    cost_matrix,
    match_visits,
    nearest_neighbor_plan,
    sinkhorn_plan,
    standardize_features,
)


def test_cost_matrix_diagonal_is_zero_when_identical():
    p = torch.randn(5, 3)
    h = torch.randn(5, 5)
    C = cost_matrix(p, h, p, h)
    assert C.shape == (5, 5)
    assert torch.allclose(C.diagonal(), torch.zeros(5), atol=1e-6)
    assert (C >= 0).all()


def test_sinkhorn_transports_close_to_identity_when_clusters_disjoint():
    """5 identical source/target supervoxels permuted -> OT should recover inv_perm.

    If ``tgt = src[perm]``, then source ``i`` matches target ``j`` where
    ``perm[j] == i`` (i.e. the inverse permutation).
    """
    torch.manual_seed(0)
    p = torch.randn(5, 3) * 50  # well-separated
    h = torch.randn(5, 5)
    perm = torch.tensor([2, 0, 4, 1, 3])
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(5)
    pi = sinkhorn_plan(cost_matrix(p, h, p[perm], h[perm]),
                       reg=0.05, n_iters=300)
    assert torch.allclose(pi.sum(dim=1), torch.full((5,), 0.2), atol=1e-4)
    assert torch.equal(pi.argmax(dim=1), inv_perm)


def test_nn_plan_each_row_has_single_nonzero():
    torch.manual_seed(1)
    C = torch.rand(4, 6)
    plan = nearest_neighbor_plan(C)
    assert plan.shape == (4, 6)
    assert (plan > 0).sum() == 4
    assert torch.equal(plan.argmax(dim=1), C.argmin(dim=1))


def test_match_visits_sinkhorn_recovers_permutation():
    torch.manual_seed(2)
    p = torch.randn(6, 3) * 50
    h = torch.randn(6, 5)
    perm = torch.tensor([4, 2, 0, 5, 1, 3])
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(6)
    res = match_visits(p, h, p[perm], h[perm], method="sinkhorn")
    assert torch.equal(res.target_idx, inv_perm)
    assert res.alive.all()


def test_match_visits_flags_impossible_targets_as_dead():
    """A source with no close target should be flagged alive=False."""
    p_src = torch.tensor([[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]])
    h_src = torch.zeros(2, 5)
    # Single target far from the second source
    p_tgt = torch.tensor([[0.0, 0.0, 0.0]])
    h_tgt = torch.zeros(1, 5)
    res = match_visits(p_src, h_src, p_tgt, h_tgt, method="nn",
                       cost_quantile_alive=0.5)
    assert res.target_idx.shape == (2,)
    # Lowest-cost source stays alive; the outlier gets flagged dead.
    assert res.alive[0] is True or bool(res.alive[0]) is True  # noqa
    assert bool(res.alive[1]) is False
    assert int(res.target_idx[1]) == -1


def test_match_visits_empty_next_visit():
    p_src = torch.randn(3, 3); h_src = torch.randn(3, 5)
    p_tgt = torch.empty(0, 3); h_tgt = torch.empty(0, 5)
    res = match_visits(p_src, h_src, p_tgt, h_tgt, method="sinkhorn")
    assert res.alive.sum() == 0
    assert (res.target_idx == -1).all()


def test_match_visits_falls_back_to_nn_for_small_inputs():
    p_src = torch.randn(1, 3); h_src = torch.randn(1, 5)
    p_tgt = torch.randn(1, 3); h_tgt = torch.randn(1, 5)
    res = match_visits(p_src, h_src, p_tgt, h_tgt, method="sinkhorn")
    assert res.method == "nn"


def test_standardize_features_matches_concat_statistics():
    torch.manual_seed(3)
    h1 = torch.randn(10, 4)
    h2 = torch.randn(7, 4) + 3.0
    out = standardize_features([h1, h2])
    stacked = torch.cat(out, dim=0)
    assert torch.allclose(stacked.mean(dim=0), torch.zeros(4), atol=1e-5)
    assert torch.allclose(stacked.std(dim=0), torch.ones(4), atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
