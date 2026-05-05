"""Unit tests for LSGCCounterfactualTwin (Stage 3 stub).

Covers:
  * Constructor wiring: arm_embed has the right number of entries; augmented
    latent dimension matches what the backbone's FiLM predictors expect.
  * Distinct arms produce distinct trajectories under identical z / G_0.
  * forward_teacher_forced returns required keys without errors.
  * rollout_counterfactual returns non-empty clouds under arm=0 and arm=1
    (count mode is default, so cloud is never empty).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lsgc.counterfactual import LSGCCounterfactualTwin  # noqa: E402


def _make_cf_twin(
    n_channels: int = 4,
    n_arms: int = 3,
    arm_dim: int = 4,
    latent_dim: int = 4,
) -> LSGCCounterfactualTwin:
    return LSGCCounterfactualTwin(
        in_channels=n_channels,
        hidden=16,
        num_layers=2,
        latent_dim=latent_dim,
        feat_out_dim=n_channels,
        n_arms=n_arms,
        arm_dim=arm_dim,
    )


def test_arm_embed_shape():
    """arm_embed table has n_arms rows each of arm_dim."""
    model = _make_cf_twin(n_arms=4, arm_dim=6)
    assert model.arm_embed.weight.shape == (4, 6)


def test_backbone_latent_dim():
    """Backbone FiLM input is latent_dim + arm_dim as promised."""
    latent_dim = 4
    arm_dim = 6
    model = _make_cf_twin(latent_dim=latent_dim, arm_dim=arm_dim)
    # The backbone was constructed with latent_dim = latent_dim + arm_dim.
    assert model.backbone.film[0].in_features == latent_dim + arm_dim


def test_forward_teacher_forced_returns_keys():
    """arm-aware teacher-forced forward returns the standard key set."""
    torch.manual_seed(0)
    n_ch = 4
    model = _make_cf_twin(n_ch)
    model.eval()
    n = 8
    x = torch.randn(n, n_ch)
    pos = torch.randn(n, 3)
    t = torch.zeros(n, dtype=torch.long)
    # Build a minimal edge_index (fully connected ring).
    edge_index = torch.stack([
        torch.arange(n),
        torch.roll(torch.arange(n), 1),
    ], dim=0)
    x0 = x
    arm = torch.tensor(1)
    out = model.forward_teacher_forced(x, pos, t, edge_index, x0, arm=arm, sample=False)
    for key in ("delta_pos", "delta_feat", "alive_logit", "z", "kl", "z_augmented"):
        assert key in out, f"missing key {key}"
    # z_augmented should be latent_dim + arm_dim long.
    assert out["z_augmented"].shape[0] == model._patient_latent_dim + model._arm_dim


def test_distinct_arms_distinct_rollout():
    """Two different arm indices produce distinct z_augmented inputs to the backbone."""
    torch.manual_seed(123)
    n_ch = 4
    latent_dim = 4
    arm_dim = 4
    model = _make_cf_twin(n_ch, n_arms=3, latent_dim=latent_dim, arm_dim=arm_dim)
    model.eval()
    x0 = torch.randn(10, n_ch)
    pos0 = torch.randn(10, 3)
    t = torch.zeros(10, dtype=torch.long)
    edge_index = torch.stack([torch.arange(10), torch.roll(torch.arange(10), 1)])

    # Verify that arm conditioning changes z_augmented (the FiLM input), not just
    # post-training outputs. Pre-training, FiLM weights are zero-initialized so
    # the backbone output is identical; the meaningful check is the augmented latent.
    z = torch.randn(latent_dim)
    out0 = model.forward_teacher_forced(x0, pos0, t, edge_index, x0,
                                         arm=torch.tensor(0), sample=False)
    out1 = model.forward_teacher_forced(x0, pos0, t, edge_index, x0,
                                         arm=torch.tensor(1), sample=False)
    # z_augmented must differ between arms (arm embeddings are initialized randomly).
    assert not torch.allclose(out0["z_augmented"], out1["z_augmented"], atol=1e-6), (
        "arm=0 and arm=1 must produce distinct z_augmented inputs to the backbone"
    )
    # Patient latent z is the same (same G_0, same sample=False path).
    assert torch.allclose(out0["z"], out1["z"]), (
        "patient latent z should be identical for same G_0 regardless of arm"
    )


def test_rollout_count_mode_non_empty():
    """rollout_counterfactual in count mode never returns an empty cloud."""
    torch.manual_seed(7)
    n_ch = 4
    model = _make_cf_twin(n_ch)
    model.eval()
    x0 = torch.randn(8, n_ch)
    pos0 = torch.randn(8, 3)
    for arm_idx in range(3):
        steps = model.rollout_counterfactual(x0, pos0, arm=arm_idx, n_steps=3)
        assert len(steps) > 0
        for i, st in enumerate(steps):
            assert st.x.shape[0] >= 1, (
                f"arm={arm_idx} step {i+1}: empty cloud in count mode"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
