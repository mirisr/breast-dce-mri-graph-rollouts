"""Unit tests for LSGC-Twin (Stage 2)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from lsgc.twin import (  # noqa: E402
    FiLMLSGCBackbone, GraphLatentEncoder, LSGCTwin,
)
from lsgc.graph_builder import build_spatiotemporal_graph  # noqa: E402


def _dummy_patient():
    torch.manual_seed(0)
    visit_feats = [torch.randn(n, 5) for n in (10, 8, 7)]
    visit_pos = [torch.randn(n, 3) * 10 for n in (10, 8, 7)]
    g = build_spatiotemporal_graph(
        visit_feats, visit_pos, k_spatial=3, k_temporal=2,
        temporal_skip_hops=(1, 2),
    )
    return g, visit_feats, visit_pos


# --------------------------------------------------------------------------- #
# Encoder                                                                     #
# --------------------------------------------------------------------------- #


def test_encoder_output_shapes():
    enc = GraphLatentEncoder(in_channels=5, hidden=16, latent_dim=4)
    x0 = torch.randn(12, 5)
    mu, log_sigma = enc(x0)
    assert mu.shape == (4,) and log_sigma.shape == (4,)
    z = GraphLatentEncoder.reparameterize(mu, log_sigma)
    assert z.shape == (4,)


def test_encoder_kl_at_prior_is_zero():
    # When mu=0 and sigma=1 (log_sigma=0), KL(q || N(0, I)) should be exactly 0.
    mu = torch.zeros(4); log_sigma = torch.zeros(4)
    kl = GraphLatentEncoder.kl_divergence(mu, log_sigma)
    assert torch.allclose(kl, torch.tensor(0.0))


def test_encoder_kl_grows_with_mu_shift():
    mu1 = torch.zeros(4); log_sigma = torch.zeros(4)
    mu2 = torch.ones(4) * 2.0
    kl1 = GraphLatentEncoder.kl_divergence(mu1, log_sigma)
    kl2 = GraphLatentEncoder.kl_divergence(mu2, log_sigma)
    assert kl2 > kl1


# --------------------------------------------------------------------------- #
# FiLM backbone                                                               #
# --------------------------------------------------------------------------- #


def test_film_initialized_to_identity_collapses_to_stage1_like():
    """At init, FiLM is zero so (gamma, beta) = (1, 0). Verify the backbone
    is still well-defined and produces the same output for two different z
    values at initialization."""
    g, _, _ = _dummy_patient()
    net = FiLMLSGCBackbone(in_channels=5, hidden=16, num_layers=2, latent_dim=4)
    z_a = torch.randn(4); z_b = torch.randn(4)
    h_a = net(g.x, g.pos, g.t, g.edge_index, z_a)
    h_b = net(g.x, g.pos, g.t, g.edge_index, z_b)
    # Same output because film modules are initialized to zero.
    assert torch.allclose(h_a, h_b, atol=1e-6)


def test_film_varies_after_step():
    g, _, _ = _dummy_patient()
    net = FiLMLSGCBackbone(in_channels=5, hidden=16, num_layers=2, latent_dim=4)
    # Perturb the FiLM weights so the modulation actually kicks in.
    for lin in net.film:
        with torch.no_grad():
            lin.weight.add_(torch.randn_like(lin.weight) * 0.1)
            lin.bias.add_(torch.randn_like(lin.bias) * 0.1)
    z_a = torch.zeros(4); z_b = torch.ones(4) * 2.0
    h_a = net(g.x, g.pos, g.t, g.edge_index, z_a)
    h_b = net(g.x, g.pos, g.t, g.edge_index, z_b)
    assert not torch.allclose(h_a, h_b, atol=1e-4)


# --------------------------------------------------------------------------- #
# Twin model: teacher-forced forward                                          #
# --------------------------------------------------------------------------- #


def test_twin_teacher_forced_shapes_and_kl_is_positive():
    g, feats, _ = _dummy_patient()
    model = LSGCTwin(in_channels=5, hidden=16, num_layers=2, latent_dim=4)
    x0 = feats[0]  # baseline nodes
    out = model.forward_teacher_forced(g.x, g.pos, g.t, g.edge_index, x0=x0)
    n = g.x.shape[0]
    assert out["delta_pos"].shape == (n, 3)
    assert out["delta_feat"].shape == (n, 5)
    assert out["alive_logit"].shape == (n,)
    assert out["z"].shape == (4,)
    assert out["kl"].ndim == 0 and out["kl"].item() >= 0


def test_twin_sample_false_is_deterministic():
    g, feats, _ = _dummy_patient()
    model = LSGCTwin(in_channels=5, hidden=16, num_layers=2, latent_dim=4).eval()
    x0 = feats[0]
    a = model.forward_teacher_forced(g.x, g.pos, g.t, g.edge_index, x0=x0, sample=False)
    b = model.forward_teacher_forced(g.x, g.pos, g.t, g.edge_index, x0=x0, sample=False)
    assert torch.allclose(a["delta_pos"], b["delta_pos"])
    assert torch.allclose(a["z"], b["z"])


def test_twin_gradients_flow_through_film_and_encoder():
    g, feats, _ = _dummy_patient()
    # Break the FiLM-identity initialization so grads through film are non-zero.
    model = LSGCTwin(in_channels=5, hidden=16, num_layers=2, latent_dim=4)
    for lin in model.backbone.film:
        torch.nn.init.normal_(lin.weight, std=0.1)
        torch.nn.init.normal_(lin.bias, std=0.1)

    x0 = feats[0]
    out = model.forward_teacher_forced(g.x, g.pos, g.t, g.edge_index, x0=x0)
    loss = out["delta_pos"].square().mean() + out["kl"]
    loss.backward()

    # Encoder grads are non-zero.
    any_enc = any(p.grad is not None and p.grad.abs().sum() > 0
                  for p in model.encoder.parameters())
    assert any_enc
    # At least one FiLM module has non-zero grad.
    any_film = any(p.grad is not None and p.grad.abs().sum() > 0
                   for lin in model.backbone.film for p in lin.parameters())
    assert any_film


# --------------------------------------------------------------------------- #
# Rollout                                                                     #
# --------------------------------------------------------------------------- #


def test_rollout_produces_n_steps_or_stops_on_empty():
    torch.manual_seed(0)
    x0 = torch.randn(10, 5); pos0 = torch.randn(10, 3) * 10
    model = LSGCTwin(in_channels=5, hidden=16, num_layers=2, latent_dim=4).eval()
    steps = model.rollout(x0, pos0, n_steps=3, alive_threshold=0.0)
    # With alive_threshold=0, every node survives -> exactly n_steps returned.
    assert len(steps) == 3
    for st in steps:
        assert st.pos.ndim == 2 and st.pos.shape[1] == 3
        assert st.x.shape[0] == st.pos.shape[0] == st.alive_prob.shape[0]


def test_rollout_with_high_threshold_prunes_everything():
    torch.manual_seed(1)
    x0 = torch.randn(8, 5); pos0 = torch.randn(8, 3) * 5
    model = LSGCTwin(in_channels=5, hidden=16, num_layers=2, latent_dim=4).eval()
    steps = model.rollout(x0, pos0, n_steps=3, alive_threshold=1.01)
    # No node can exceed probability 1.0, so the rollout stops immediately.
    assert len(steps) == 1 and steps[0].x.shape[0] == 0


def test_rollout_source_idx_tracks_original_voxels():
    torch.manual_seed(2)
    n0 = 10
    x0 = torch.randn(n0, 5); pos0 = torch.randn(n0, 3) * 10
    model = LSGCTwin(in_channels=5, hidden=16, num_layers=2, latent_dim=4).eval()
    steps = model.rollout(x0, pos0, n_steps=2, alive_threshold=0.0)
    # Every kept voxel must point to a valid baseline index.
    for st in steps:
        if st.source_idx.numel() == 0:
            continue
        assert int(st.source_idx.min()) >= 0
        assert int(st.source_idx.max()) < n0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
