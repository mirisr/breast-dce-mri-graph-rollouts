"""Unit tests for LSGCConv / LSGCNet and the spatio-temporal graph builder."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lsgc import LSGCConv, LSGCNet, build_spatiotemporal_graph  # noqa: E402


torch.manual_seed(0)


def _random_graph(n_per_visit=(6, 5, 5, 4), c=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    feats = [torch.randn(n, c, generator=g) for n in n_per_visit]
    poses = [torch.randn(n, 3, generator=g) * 10.0 for n in n_per_visit]
    return build_spatiotemporal_graph(feats, poses, k_spatial=3, k_temporal=2)


# ---------------------------------------------------------------------------
# LSGCConv
# ---------------------------------------------------------------------------

def test_lsgcconv_output_shape():
    g = _random_graph()
    conv = LSGCConv(in_channels=8, out_channels=16)
    h = conv(g.x, g.pos, g.t, g.edge_index)
    assert h.shape == (g.x.size(0), 16)


def test_lsgcconv_backward_flows():
    g = _random_graph()
    conv = LSGCConv(in_channels=8, out_channels=4)
    h = conv(g.x, g.pos, g.t, g.edge_index)
    loss = h.pow(2).mean()
    loss.backward()
    grads = [p.grad for p in conv.parameters() if p.requires_grad]
    assert all(gr is not None for gr in grads)
    assert all(torch.isfinite(gr).all() for gr in grads)


def test_lsgcconv_invariant_to_node_reordering():
    """Permuting the node index set (and re-indexing edges) must permute the
    output identically. This is a property of any valid MessagePassing layer
    and guards against accidental absolute-index leakage in the filter."""
    g = _random_graph()
    conv = LSGCConv(in_channels=8, out_channels=8).eval()
    with torch.no_grad():
        h1 = conv(g.x, g.pos, g.t, g.edge_index)

    perm = torch.randperm(g.x.size(0))
    inv = torch.argsort(perm)
    x2 = g.x[perm]
    pos2 = g.pos[perm]
    t2 = g.t[perm]
    edge2 = inv[g.edge_index]  # remap old ids -> new positions

    with torch.no_grad():
        h2 = conv(x2, pos2, t2, edge2)

    torch.testing.assert_close(h1, h2[inv], atol=1e-5, rtol=1e-5)


def test_filter_reduces_to_spatial_when_all_same_visit():
    """When every node shares the same visit (all delta_t = 0), LSGC must
    behave as a pure spatial continuous-filter conv (SchNet-like). We test
    this by checking that disabling time_freqs has no effect on the output
    in that regime."""
    g = _random_graph(n_per_visit=(10,), c=8)
    conv = LSGCConv(in_channels=8, out_channels=8).eval()
    with torch.no_grad():
        h_before = conv(g.x, g.pos, g.t, g.edge_index)
        conv.time_freqs.data *= 0.0  # delta_t still 0, so sin=0/cos=1 regardless
        h_after = conv(g.x, g.pos, g.t, g.edge_index)
    torch.testing.assert_close(h_before, h_after, atol=1e-6, rtol=1e-6)


def test_filter_is_sensitive_to_dt_when_visits_differ():
    """Conversely, when nodes span multiple visits, zeroing the time basis
    must materially change the output. Guards against a dead time branch."""
    g = _random_graph()
    conv = LSGCConv(in_channels=8, out_channels=8).eval()
    with torch.no_grad():
        h_before = conv(g.x, g.pos, g.t, g.edge_index)
        conv.time_freqs.data *= 0.0
        h_after = conv(g.x, g.pos, g.t, g.edge_index)
    assert (h_before - h_after).abs().mean() > 1e-4


# ---------------------------------------------------------------------------
# LSGCNet
# ---------------------------------------------------------------------------

def test_lsgcnet_single_graph_forward_backward():
    g = _random_graph()
    net = LSGCNet(in_channels=8, hidden=16, out_channels=1, num_layers=2)
    y = net(g.x, g.pos, g.t, g.edge_index)
    assert y.shape == (1, 1)
    y.sum().backward()
    n_grad = sum(p.grad is not None for p in net.parameters())
    assert n_grad > 0


def test_lsgcnet_batched_readout():
    g1 = _random_graph(seed=1)
    g2 = _random_graph(seed=2)
    x = torch.cat([g1.x, g2.x])
    pos = torch.cat([g1.pos, g2.pos])
    t = torch.cat([g1.t, g2.t])
    edge = torch.cat(
        [g1.edge_index, g2.edge_index + g1.x.size(0)], dim=1
    )
    batch = torch.cat(
        [torch.zeros(g1.x.size(0), dtype=torch.long),
         torch.ones(g2.x.size(0), dtype=torch.long)]
    )
    net = LSGCNet(in_channels=8, hidden=16, out_channels=3, num_layers=2)
    y = net(x, pos, t, edge, batch=batch)
    assert y.shape == (2, 3)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def test_builder_node_and_edge_counts():
    feats = [torch.randn(6, 4), torch.randn(5, 4), torch.randn(5, 4), torch.randn(4, 4)]
    poses = [torch.randn(n, 3) * 10 for n in (6, 5, 5, 4)]
    g = build_spatiotemporal_graph(feats, poses, k_spatial=3, k_temporal=2)
    assert g.x.size(0) == 20
    assert g.pos.size(0) == 20
    assert g.t.size(0) == 20
    assert g.t[:6].eq(0).all() and g.t[6:11].eq(1).all()
    assert g.edge_index.size(0) == 2
    # intra-visit: sum_v N_v * k_spatial; inter-visit: sum_v N_v * k_temporal
    intra = sum(n * 3 for n in (6, 5, 5, 4))
    inter = sum(n * 2 for n in (6, 5, 5))  # T-1 transitions
    assert g.edge_index.size(1) == 2 * (intra + inter)  # symmetric = *2


def test_builder_self_loops_excluded_intra_visit():
    feats = [torch.randn(5, 4)]
    poses = [torch.randn(5, 3)]
    g = build_spatiotemporal_graph(feats, poses, k_spatial=2, k_temporal=2)
    src, dst = g.edge_index
    assert (src != dst).all()


def test_builder_temporal_skip_edges_add_expected_counts():
    """With temporal_skip_hops=(1,2,3) on a 4-visit graph, the inter-visit
    edge count is sum over hops of (T-hop) * k_temporal * N_v-source."""
    sizes = (6, 5, 5, 4)
    feats = [torch.randn(n, 4) for n in sizes]
    poses = [torch.randn(n, 3) * 10 for n in sizes]
    g = build_spatiotemporal_graph(
        feats, poses, k_spatial=3, k_temporal=2, temporal_skip_hops=(1, 2, 3)
    )
    intra = sum(n * 3 for n in sizes)
    inter = 0
    for hop in (1, 2, 3):
        for v in range(len(sizes) - hop):
            inter += sizes[v] * 2
    assert g.edge_index.size(1) == 2 * (intra + inter)


def test_builder_skip_hops_connect_claimed_visits():
    """Edges added under temporal_skip_hops=(2,) must only connect visits v
    and v+2 (plus the symmetric reverse)."""
    sizes = (4, 4, 4, 4)
    feats = [torch.randn(n, 4) for n in sizes]
    poses = [torch.randn(n, 3) * 10 for n in sizes]
    g = build_spatiotemporal_graph(
        feats, poses, k_spatial=2, k_temporal=2, temporal_skip_hops=(2,)
    )
    src, dst = g.edge_index
    dt = (g.t[dst] - g.t[src]).abs()
    # Allowed dt values are 0 (intra-visit kNN) and 2 (the only skip hop).
    assert set(dt.unique().tolist()) <= {0, 2}
    assert (dt == 2).any(), "expected at least one skip-2 edge"


def test_builder_default_matches_hop1_explicit():
    """Default behavior must be identical to temporal_skip_hops=(1,)."""
    sizes = (3, 4, 3)
    g1 = torch.Generator().manual_seed(7)
    feats = [torch.randn(n, 4, generator=g1) for n in sizes]
    poses = [torch.randn(n, 3, generator=g1) * 10 for n in sizes]
    a = build_spatiotemporal_graph(feats, poses, k_spatial=2, k_temporal=2)
    b = build_spatiotemporal_graph(
        feats, poses, k_spatial=2, k_temporal=2, temporal_skip_hops=(1,)
    )
    assert a.edge_index.size(1) == b.edge_index.size(1)
    torch.testing.assert_close(
        a.edge_index.sort(dim=1).values, b.edge_index.sort(dim=1).values
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
