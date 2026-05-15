"""Helpers for the Stage-1 forecaster visualization notebook.

Loads pretrained Stage-1 forecasters, runs inference on a small set of
patients, and builds clean Plotly volumetric visualisations of the real
tumour segmentation versus model predictions.

Designed so that the notebook only needs to call:

    from forecaster_viz_lib import *
    bundle = load_everything(repo_root)
    show_real_trajectory(bundle, 'ISPY2-559021')
    show_predicted_survival(bundle, 'ISPY2-559021')
    show_error_comparison(bundle, 'ISPY2-559021')
"""

from __future__ import annotations

import json as _json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from IPython.display import display

# Use plotly_mimetype + notebook so the figures render reliably both in
# JupyterLab / Cursor (mime-bundle path) AND classic Jupyter (HTML path).
pio.renderers.default = 'plotly_mimetype+notebook_connected'


# =============================================================================
# Configuration
# =============================================================================

VISIT_IDS = ['T0', 'T1', 'T2', 'T3']

HAB_COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']  # blue, green, orange, pink
# Habitat labels reflect the empirical DCE enhancement patterns measured across
# the cohort (mean curve over 8 phases). NOTE: K-means cluster IDs are arbitrary
# — the labels below were verified from per-habitat mean curves, NOT assumed.
HAB_LABELS = [
    'sustained moderate',     # H0 blue:    plateau ~1.4, no washout
    'low / sparse',           # H1 green:   small or empty cluster in this cohort
    'washout / necrotic',     # H2 orange:  rises to ~1.4, drops to ~0 by phase 7
    'high perfusion',         # H3 pink:    PEAK ~2.3, classic malignant wash-in/out
]

# Patient roster — (id, description, pCR label or None)
PATIENTS_DEFAULT = [
    ('ISPY2-559021',      'Large responder (HR+/HER2-)',      1),
    ('ISPY2-622688',      'TNBC responder',                   1),
    ('ISPY2-159284',      'Large non-responder',              0),
    ('ISPY2-311316',      'Large, 3 habitats',                0),
    ('ACRIN-6698-104268', 'Heterogeneous, 3 habitats',     None),
    ('ACRIN-6698-760011', 'Small tumor (36 sv)',           None),
]


# =============================================================================
# Repo / path detection
# =============================================================================

def detect_repo_root(start: Optional[str] = None) -> str:
    """Walk a few likely locations and return the path that contains lsgc/."""
    start = os.path.abspath(start or os.getcwd())
    candidates = [start, os.path.join(start, '..'), os.path.join(start, '..', '..'),
                  '/Users/irisseaman/Research/3DGCNN']
    for c in candidates:
        if os.path.isdir(os.path.join(c, 'lsgc')):
            return os.path.abspath(c)
    raise RuntimeError(f'Could not find repo root (lsgc/) starting from {start}')


# =============================================================================
# Model loading
# =============================================================================

def _bio_indices(feature_names):
    """Resolve ADC + DCE feature indices used by the bio edge_attr mode."""
    out = {'adc_idx': None, 'adc_missing_idx': None,
           'dce_idx_start': 0, 'dce_n_phases': 0}
    if not feature_names:
        return out
    if 'mean_adc' in feature_names:
        out['adc_idx'] = feature_names.index('mean_adc')
    if 'adc_missing' in feature_names:
        out['adc_missing_idx'] = feature_names.index('adc_missing')
    dce = [i for i, n in enumerate(feature_names)
           if n.startswith('phase') and n.endswith('_mean_enh')]
    if dce:
        out['dce_idx_start'] = min(dce)
        out['dce_n_phases']  = len(dce)
    return out


def load_forecaster(ckpt_path, device='cpu'):
    """Load a Stage-1 forecaster from a `best.pt` checkpoint.

    Robust to checkpoints where `edge_attr_dim` was not saved in the config —
    the value is inferred from the saved filter_net weight shape.
    """
    from lsgc.forecaster import LSGCForecaster

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt['config']
    mean = ckpt['mean'].to(device)
    std  = ckpt['std'].to(device)

    in_channels     = int(mean.shape[0])
    hidden          = int(cfg.get('hidden', 64))
    num_layers      = int(cfg.get('num_layers', 2))
    conv_type       = cfg.get('conv_type', 'lsgc')
    use_edge_gating = bool(cfg.get('use_edge_gating', False))

    probe = LSGCForecaster(in_channels=in_channels, hidden=hidden,
                           num_layers=num_layers, conv_type=conv_type,
                           edge_attr_dim=0, use_edge_gating=False)
    base_filter_in = probe.convs[0].filter_net[0].weight.shape[1]
    ckpt_filter_in = ckpt['state_dict']['convs.0.filter_net.0.weight'].shape[1]
    edge_attr_dim  = int(ckpt_filter_in - base_filter_in)

    model = LSGCForecaster(in_channels=in_channels, hidden=hidden,
                           num_layers=num_layers, conv_type=conv_type,
                           edge_attr_dim=edge_attr_dim,
                           use_edge_gating=use_edge_gating).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, mean, std, edge_attr_dim


# =============================================================================
# Inference
# =============================================================================

def _per_visit_centroid(pos):
    return pos.mean(dim=0, keepdim=True)


def _nn_match(pos_src, pos_dst):
    d   = torch.cdist(pos_src.float(), pos_dst.float())
    idx = d.argmin(dim=1)
    return idx, d[torch.arange(len(idx)), idx]


@torch.no_grad()
def run_forecaster(model, mean, std, g, edge_mode='mixed+attr',
                   edge_attr_mode='legacy'):
    """Run a Stage-1 forecaster over every transition for one patient."""
    from lsgc.graph_builder import build_spatiotemporal_graph

    fn  = g.get('feature_names'); bio = _bio_indices(fn)
    off = g['visit_offsets'].tolist()
    x   = g['x']; pos = g['pos']; hab = g.get('habitat')
    T   = len(off) - 1

    visit_feats = [x[int(off[v]):int(off[v+1])]   for v in range(T)]
    visit_pos   = [pos[int(off[v]):int(off[v+1])] for v in range(T)]
    hab_labels  = [hab[int(off[v]):int(off[v+1])] for v in range(T)] \
                  if hab is not None else None

    g_st = build_spatiotemporal_graph(
        visit_feats, visit_pos,
        k_spatial=8, k_temporal=4, temporal_skip_hops=(1, 2, 3),
        edge_mode=edge_mode,
        add_edge_attr=(edge_attr_mode == 'bio'),
        edge_attr_mode=edge_attr_mode,
        adc_idx=bio['adc_idx'],
        adc_missing_idx=bio['adc_missing_idx'],
        dce_idx_start=bio['dce_idx_start'],
        dce_n_phases=bio['dce_n_phases'],
        habitat_labels=hab_labels,
    )

    xn    = (x - mean) / std
    pos_c = pos.clone()
    for v in range(T):
        sl = slice(int(off[v]), int(off[v+1]))
        pos_c[sl] = pos[sl] - _per_visit_centroid(pos[sl])

    t_all = g['t']
    ea    = g_st.edge_attr.float() if g_st.edge_attr is not None else None
    out   = model(xn, pos_c, t_all, g_st.edge_index.long(),
                  edge_attr=ea, delta_t=1.0)

    results = []
    for k in range(T - 1):
        sl_k  = slice(int(off[k]),   int(off[k+1]))
        sl_k1 = slice(int(off[k+1]), int(off[k+2]))
        pos_obs_k  = pos_c[sl_k]; pos_obs_k1 = pos_c[sl_k1]
        dp         = out['delta_pos'][sl_k]
        pos_pred   = pos_obs_k + dp
        alive_pr   = torch.sigmoid(out['alive_logit'][sl_k])
        _, dist    = _nn_match(pos_pred, pos_obs_k1)
        results.append({
            'k': k,
            'pos_obs_src':   pos_obs_k.numpy(),
            'pos_obs_dst':   pos_obs_k1.numpy(),
            'pos_pred':      pos_pred.detach().numpy(),
            'alive_prob':    alive_pr.detach().numpy(),
            'match_dist_mm': dist.detach().numpy(),
        })
    return results


# =============================================================================
# Voxel volume loading
# =============================================================================

def load_voxel_volume(sv_dir: Path, pid: str, visit_id: str,
                     max_points: int = 7000, seed: int = 42):
    """Load the supervoxel label map and return tumour voxels in world-mm."""
    sv_path = Path(sv_dir) / pid / visit_id
    if not (sv_path / 'supervoxel_labels.npz').exists():
        return None, None
    labels  = np.load(sv_path / 'supervoxel_labels.npz')['labels']
    meta    = _json.load(open(sv_path / 'meta.json'))
    spacing = np.array(meta['voxel_spacing_mm'], dtype=np.float32)
    origin  = np.array(meta['origin_mm'],        dtype=np.float32)

    tumor_idx = np.argwhere(labels > 0).astype(np.int32)
    sv_ids    = labels[tumor_idx[:, 0], tumor_idx[:, 1], tumor_idx[:, 2]] - 1
    pos_mm    = origin + tumor_idx * spacing

    if len(pos_mm) > max_points:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(pos_mm), max_points, replace=False)
        pos_mm = pos_mm[sel]; sv_ids = sv_ids[sel]
    return pos_mm.astype(np.float32), sv_ids.astype(np.int32)


# =============================================================================
# Color helpers
# =============================================================================

def _safe_index(arr, idx):
    return arr[np.clip(idx, 0, len(arr) - 1)]

def color_by_habitat(sv_ids, habitat_per_node, alpha=0.9):
    hab = _safe_index(np.asarray(habitat_per_node), sv_ids)
    return [mcolors.to_hex(mcolors.to_rgba(HAB_COLORS[int(h) % 4], alpha))
            for h in hab]

def color_by_alive_prob(sv_ids, alive_prob_per_node):
    cmap  = plt.get_cmap('RdYlBu_r')
    probs = np.clip(_safe_index(np.asarray(alive_prob_per_node), sv_ids), 0, 1)
    return [mcolors.to_hex(cmap(float(p))) for p in probs]

def color_by_error(sv_ids, err_per_node, vmax=8.0):
    cmap = plt.get_cmap('YlOrRd')
    errs = _safe_index(np.asarray(err_per_node), sv_ids)
    return [mcolors.to_hex(cmap(min(float(e), vmax) / vmax)) for e in errs]


# =============================================================================
# Plot primitives
# =============================================================================

def _voxel_scatter(pos_mm, colors, name, size=2, opacity=0.75):
    return go.Scatter3d(
        x=pos_mm[:, 2], y=pos_mm[:, 1], z=pos_mm[:, 0],
        mode='markers',
        marker=dict(size=size, color=colors, opacity=opacity),
        name=name,
    )

def _axis():
    return dict(showgrid=False, showticklabels=False, title='',
                backgroundcolor='#0d1117', gridcolor='#222',
                zerolinecolor='#222')

def _scene_cfg():
    return dict(xaxis=_axis(), yaxis=_axis(), zaxis=_axis(),
                bgcolor='#0d1117', aspectmode='data',
                camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)))

def _col_titles(labels, n_cols, y=-0.06):
    """Column titles placed *below* each subplot (no overlap with main title)."""
    xs = [1 / (2 * n_cols) + i / n_cols for i in range(n_cols)]
    return [dict(text=f'<b>{labels[i]}</b>',
                 x=xs[i], y=y, xref='paper', yref='paper',
                 showarrow=False,
                 font=dict(size=12, color='#333'),
                 xanchor='center', yanchor='top')
            for i in range(min(len(labels), n_cols))]


# =============================================================================
# High-level figure builders
# =============================================================================

def fig_real_trajectory(pid, vox_per_visit, hab_per_visit, visit_ids,
                        pcr=None):
    """Multi-panel actual tumour voxel cloud across treatment visits."""
    n_v = len(visit_ids)
    fig = make_subplots(rows=1, cols=n_v,
                        specs=[[{'type': 'scatter3d'}] * n_v],
                        horizontal_spacing=0.02)
    for vi, vid in enumerate(visit_ids):
        pos_mm, sv_ids = vox_per_visit[vi]
        if pos_mm is None:
            continue
        cols = color_by_habitat(sv_ids, hab_per_visit[vi])
        fig.add_trace(_voxel_scatter(pos_mm, cols, vid, size=2),
                      row=1, col=vi + 1)
        fig.update_scenes(_scene_cfg(), row=1, col=vi + 1)

    pcr_str = f' · pCR = {pcr}' if pcr is not None else ''
    title   = f'<b>{pid}</b>{pcr_str}<br><sup>Real tumour voxels coloured by DCE perfusion habitat</sup>'

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center',
                   y=0.97, yanchor='top',
                   font=dict(size=14, color='#222')),
        height=420,
        showlegend=False,
        margin=dict(t=80, b=60, l=10, r=10),
        annotations=_col_titles(visit_ids, n_v),
        paper_bgcolor='white',
    )
    return fig


def fig_predicted_survival(pid, t0_vox, t1_vox, alive_prob, delta_pos, t0_sv_centroids,
                           model_name, pcr=None, alive_threshold: float = 0.5):
    """**Predicted T_{k+1} cloud** vs actual T_{k+1}.

    The predicted cloud is built by:
      1. taking each T0 voxel,
      2. applying the per-supervoxel predicted displacement Δpos,
      3. keeping only voxels whose supervoxel has p(alive) > threshold.

    This is a fair comparison: if the model is correct, the predicted cloud
    should both (a) have the right size — fewer points for responders — and
    (b) sit at the right place. Side-by-side panels are centred on each
    cloud's own centroid so inter-visit patient repositioning is removed.
    """
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'type': 'scatter3d'}] * 2],
                        horizontal_spacing=0.04)
    t0_pos, t0_sv = t0_vox
    t1_pos, _     = t1_vox

    # Build the predicted T1 voxel cloud
    n_t0_vox = len(t0_pos) if t0_pos is not None else 0
    n_pred   = 0
    pred_pos = pred_alive = None
    if t0_pos is not None and delta_pos is not None:
        sv_clipped  = np.clip(t0_sv, 0, len(delta_pos) - 1)
        voxel_delta = delta_pos[sv_clipped]
        voxel_alive = alive_prob[sv_clipped]
        # center T0 voxels by their own voxel centroid (matches training convention)
        t0_centered = t0_pos - t0_pos.mean(axis=0, keepdims=True)
        # apply per-supervoxel predicted displacement
        predicted_centered = t0_centered + voxel_delta
        survives    = voxel_alive > alive_threshold
        pred_pos    = predicted_centered[survives]
        pred_alive  = voxel_alive[survives]
        n_pred      = int(survives.sum())

    # Center actual T1 by its own centroid for like-for-like comparison
    if t1_pos is not None:
        t1_centered = t1_pos - t1_pos.mean(axis=0, keepdims=True)
    else:
        t1_centered = None

    if pred_pos is not None and len(pred_pos):
        fig.add_trace(_voxel_scatter(
            pred_pos, color_by_alive_prob(np.arange(len(pred_alive)), pred_alive),
            'predicted survivors', size=2),
            row=1, col=1)
    if t1_centered is not None:
        fig.add_trace(_voxel_scatter(
            t1_centered, ['#888'] * len(t1_centered),
            'actual', size=2, opacity=0.6),
            row=1, col=2)

    for c in (1, 2):
        fig.update_scenes(_scene_cfg(), row=1, col=c)

    pcr_str    = f' · pCR = {pcr}' if pcr is not None else ''
    n_t1_vox   = len(t1_pos) if t1_pos is not None else 0
    pct_pred   = 100.0 * n_pred / max(n_t0_vox, 1)
    pct_actual = 100.0 * n_t1_vox / max(n_t0_vox, 1)
    title = (f'<b>{pid}</b>{pcr_str} · {model_name}<br>'
             f'<sup>Predicted survivors: <b>{n_pred:,}/{n_t0_vox:,} ({pct_pred:.0f}% of T0)</b> · '
             f'Actual T+1 size: <b>{n_t1_vox:,}/{n_t0_vox:,} ({pct_actual:.0f}% of T0)</b><br>'
             f'A correct model should have both ratios match.</sup>')

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center',
                   y=0.97, yanchor='top',
                   font=dict(size=13, color='#222')),
        height=480,
        showlegend=False,
        margin=dict(t=110, b=60, l=10, r=10),
        annotations=_col_titles(['Predicted survivors\n(displaced + filtered by p(alive))',
                                 'Actual ground truth'], 2),
        paper_bgcolor='white',
    )
    return fig


def fig_error_comparison(pid, t0_vox, err_per_model, model_names,
                         transition='T0→T1'):
    """Per-supervoxel position error painted onto the T0 voxel cloud."""
    fig = make_subplots(rows=1, cols=3,
                        specs=[[{'type': 'scatter3d'}] * 3],
                        horizontal_spacing=0.02)
    t0_pos, t0_sv = t0_vox
    for ci, (mname, err) in enumerate(zip(model_names, err_per_model), 1):
        if t0_pos is None:
            continue
        cols = color_by_error(t0_sv, err)
        fig.add_trace(_voxel_scatter(t0_pos, cols, mname, size=2),
                      row=1, col=ci)
        fig.update_scenes(_scene_cfg(), row=1, col=ci)

    title = (f'<b>{pid}</b> · per-supervoxel error ({transition})<br>'
             f'<sup>Yellow = accurate · red = large error · vmax = 8 mm</sup>')

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center',
                   y=0.97, yanchor='top',
                   font=dict(size=14, color='#222')),
        height=420,
        showlegend=False,
        margin=dict(t=80, b=60, l=10, r=10),
        annotations=_col_titles(model_names, 3),
        paper_bgcolor='white',
    )
    return fig


# =============================================================================
# Bundle: load everything once
# =============================================================================

@dataclass
class VizBundle:
    """Container for all data + models needed for visualisations."""
    repo_root:    str
    data_dir:     Path
    cohort:       pd.DataFrame
    folds:        pd.DataFrame
    models:       dict          # name -> (model, mean, std, edge_attr_dim)
    model_specs:  dict          # name -> (edge_mode, edge_attr_mode)
    patients:     list          # [(pid, desc, pcr), ...]
    results:      dict = field(default_factory=dict)   # pid -> mname -> [transitions]
    voxels:       dict = field(default_factory=dict)   # pid -> visit -> (pos, sv_ids)
    habitats:     dict = field(default_factory=dict)   # pid -> visit -> habitat array

    def graph_path(self, pid):
        return self.data_dir / 'graphs' / f'{pid}.pt'

    def load_graph(self, pid):
        return torch.load(self.graph_path(pid), map_location='cpu',
                          weights_only=False)


def load_everything(repo_root: Optional[str] = None,
                    patients=None) -> VizBundle:
    """Load all checkpoints, graphs, voxel volumes, and run inference."""
    if repo_root is None:
        repo_root = detect_repo_root()
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    data_dir = Path(repo_root) / 'notebooks' / 'ispy2' / 'viz_data'
    ckpt_dir = data_dir / 'checkpoints'
    sv_dir   = data_dir / 'supervoxels'

    cohort = pd.read_parquet(data_dir / 'cohort.parquet').set_index('patient_id')
    folds  = pd.read_parquet(data_dir / 'folds.parquet')

    models = {}
    for tag, fname in [
        ('Baseline (33-dim)', 's1_baseline_fold0.pt'),
        ('S1.7 Bio attrs',    's1_bio_fold0.pt'),
        ('S1.8 Bio + gated',  's1_bio_gated_fold0.pt'),
    ]:
        models[tag] = load_forecaster(ckpt_dir / fname)

    model_specs = {
        'Baseline (33-dim)': ('mixed+attr', 'legacy'),
        'S1.7 Bio attrs':    ('mixed+attr', 'bio'),
        'S1.8 Bio + gated':  ('mixed+attr', 'bio'),
    }

    patients = patients or PATIENTS_DEFAULT
    bundle = VizBundle(repo_root=repo_root, data_dir=data_dir,
                       cohort=cohort, folds=folds,
                       models=models, model_specs=model_specs,
                       patients=patients)

    print(f'{"Patient":30s}  T0    T1    T2    T3   description')
    print('-' * 80)
    for pid, desc, pcr in patients:
        g = bundle.load_graph(pid)
        bundle.results[pid] = {}
        for mname, (model, mean, std, _) in models.items():
            em, eam = model_specs[mname]
            bundle.results[pid][mname] = run_forecaster(
                model, mean, std, g, edge_mode=em, edge_attr_mode=eam)

        off = g['visit_offsets'].tolist(); T = len(off) - 1
        bundle.habitats[pid] = {}
        for vi, vid in enumerate(VISIT_IDS[:T]):
            sl = slice(int(off[vi]), int(off[vi+1]))
            bundle.habitats[pid][vid] = (
                g['habitat'][sl].numpy() if g.get('habitat') is not None
                else np.zeros(int(off[vi+1]) - int(off[vi]), dtype=np.int32))

        bundle.voxels[pid] = {vid: load_voxel_volume(sv_dir, pid, vid)
                              for vid in VISIT_IDS}
        n = {vid: (len(bundle.voxels[pid][vid][0])
                   if bundle.voxels[pid][vid][0] is not None else 0)
             for vid in VISIT_IDS}
        print(f'{pid:30s}  {n["T0"]:>4d}  {n["T1"]:>4d}  {n["T2"]:>4d}  {n["T3"]:>4d}   {desc}')

    print(f'\nLoaded {len(patients)} patients × {len(models)} models')
    return bundle


# =============================================================================
# Convenience wrappers — these are the only things the notebook calls
# =============================================================================

def show_real_trajectory(bundle: VizBundle, pid: str):
    """Display the actual tumour voxel volume across all available visits."""
    desc, pcr = next((d, p) for q, d, p in bundle.patients if q == pid)
    g     = bundle.load_graph(pid)
    T     = len(g['visit_offsets']) - 1
    vids  = VISIT_IDS[:T]
    vox   = [bundle.voxels[pid][v]   for v in vids]
    hab   = [bundle.habitats[pid][v] for v in vids]
    fig   = fig_real_trajectory(pid, vox, hab, vids, pcr=pcr)
    fig.show()


def show_predicted_survival(bundle: VizBundle, pid: str,
                            model_name='S1.8 Bio + gated', k_show=0,
                            alive_threshold: float = 0.5):
    """Predicted T_{k+1} cloud (displaced + filtered by p(alive))
    side-by-side with actual T_{k+1}.
    """
    desc, pcr = next((d, p) for q, d, p in bundle.patients if q == pid)
    res = bundle.results[pid].get(model_name, [])
    if not res or k_show >= len(res):
        print(f'No prediction available for {pid}, k={k_show}')
        return
    r = res[k_show]
    src_v = VISIT_IDS[k_show]
    dst_v = VISIT_IDS[k_show + 1]

    # delta_pos per supervoxel (in centred coords)
    delta_pos = r['pos_pred'] - r['pos_obs_src']
    fig = fig_predicted_survival(
        pid,
        bundle.voxels[pid][src_v],
        bundle.voxels[pid][dst_v],
        r['alive_prob'], delta_pos, r['pos_obs_src'],
        model_name, pcr=pcr, alive_threshold=alive_threshold,
    )
    fig.show()
    print(f'  {src_v}→{dst_v}   '
          f'mean centroid NN error: {np.mean(r["match_dist_mm"]):.2f} mm   '
          f'p50: {np.percentile(r["match_dist_mm"], 50):.2f} mm   '
          f'mean p(alive): {np.mean(r["alive_prob"]):.3f}')


def show_predicted_survival_all(bundle: VizBundle, pid: str,
                                model_name='S1.8 Bio + gated'):
    """Show predicted survival for every available transition (T0→T1, T1→T2, T2→T3).

    The forecaster outputs predictions for *every* transition in a single
    forward pass. This helper iterates through them so we can see how the
    model's accuracy degrades (or holds up) deeper into treatment.
    """
    desc, pcr = next((d, p) for q, d, p in bundle.patients if q == pid)
    res = bundle.results[pid].get(model_name, [])
    if not res:
        print(f'No predictions for {pid}'); return
    print(f'\n{pid}  ({desc}, pCR={pcr})')
    for k_show, r in enumerate(res):
        show_predicted_survival(bundle, pid, model_name=model_name, k_show=k_show)


# ---------- centroid-level prediction-vs-actual view ----------------------
def show_centroid_matches(bundle: VizBundle, pid: str,
                          model_name='S1.8 Bio + gated', k_show=0,
                          max_match_mm: float = 12.0):
    """**Supervoxel centroid view** of the prediction.

    The forecaster predicts at the supervoxel-centroid level — *not* the voxel
    level. The voxel-cloud overlays in Part 2 amplify the inter-visit
    supervoxel re-segmentation (each visit is independently re-segmented, so a
    voxel-level "match" is over-strict). This view shows what the model
    actually outputs:

      - colored dots  = predicted T_{k+1} supervoxel centroids (T_k + Δpos),
                        only those with p(alive) > 0.5; colored red→blue by
                        p(alive)
      - gray dots     = actual T_{k+1} supervoxel centroids
      - thin lines    = nearest-neighbour matches, drawn only when the match
                        distance is ≤ ``max_match_mm`` (default 12 mm so very
                        unmatched points don't add visual clutter)

    A clean visual = predicted dots clustering on top of gray dots with short
    coloured connectors. This is the visualisation that should match the
    held-out 5-fold metrics (T1 EMD ≈ 2.2 mm for S1.8).
    """
    desc, pcr = next((d, p) for q, d, p in bundle.patients if q == pid)
    res = bundle.results[pid].get(model_name, [])
    if not res or k_show >= len(res):
        print(f'No prediction for {pid} k={k_show}'); return
    r = res[k_show]
    src_v = VISIT_IDS[k_show]; dst_v = VISIT_IDS[k_show + 1]

    pos_pred = r['pos_pred']                 # (N_src, 3) predicted T+1 centroids
    pos_obs  = r['pos_obs_dst']              # (N_dst, 3) actual T+1 centroids
    alive    = r['alive_prob']               # (N_src,)
    survives = alive > 0.5
    pred     = pos_pred[survives]
    pred_a   = alive[survives]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=pos_obs[:, 2], y=pos_obs[:, 1], z=pos_obs[:, 0],
        mode='markers',
        marker=dict(size=6, color='#888', opacity=0.7,
                    line=dict(color='#444', width=1)),
        name=f'actual {dst_v} centroids ({len(pos_obs)})'))

    if len(pred):
        fig.add_trace(go.Scatter3d(
            x=pred[:, 2], y=pred[:, 1], z=pred[:, 0],
            mode='markers',
            marker=dict(size=6,
                        color=color_by_alive_prob(np.arange(len(pred_a)), pred_a),
                        opacity=0.92,
                        line=dict(color='black', width=0.6)),
            name=f'predicted {dst_v} survivors ({len(pred)})'))

        # NN-match connectors
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(pos_obs)
            d, idx = tree.query(pred, k=1)
            xs, ys, zs = [], [], []
            for i, (di, j) in enumerate(zip(d, idx)):
                if di > max_match_mm:
                    continue
                a = pred[i]
                b = pos_obs[j]
                xs += [a[2], b[2], None]
                ys += [a[1], b[1], None]
                zs += [a[0], b[0], None]
            if xs:
                fig.add_trace(go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode='lines',
                    line=dict(color='#444', width=2),
                    opacity=0.45,
                    showlegend=True,
                    name='NN match (≤ {:.0f} mm)'.format(max_match_mm)))
        except Exception:
            pass

    nn = np.array(r['match_dist_mm'])
    p50 = float(np.percentile(nn, 50))
    p90 = float(np.percentile(nn, 90))
    pct_close = 100.0 * float(np.mean(nn <= 5.0))

    title = (f'<b>{pid}</b> · pCR = {pcr} · {model_name} · {src_v}→{dst_v} '
             f'(supervoxel centroid view)<br>'
             f'<sup>Centroid NN error: median <b>{p50:.2f} mm</b>, p90 {p90:.2f} mm, '
             f'<b>{pct_close:.0f}%</b> within 5 mm. '
             f'Predicted survivors: {len(pred)}/{len(pos_pred)}, actual: {len(pos_obs)}.</sup>')

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center', y=0.97, yanchor='top',
                   font=dict(size=12)),
        scene=_scene_cfg(),
        height=560,
        showlegend=True,
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01,
                    bgcolor='rgba(255,255,255,0.7)'),
        margin=dict(t=110, b=40, l=10, r=10),
        paper_bgcolor='white',
    )
    fig.show()


def patient_quality_table(bundle: VizBundle, model_name='S1.8 Bio + gated'):
    """Per-patient supervoxel-centroid match quality for every transition.

    For each patient and every available T_k → T_{k+1} step, we report:
      - n_pred : number of supervoxels with p(alive) > 0.5 (predicted survivors)
      - n_act  : number of actual T_{k+1} supervoxels
      - med    : median NN distance from predicted survivors to actual centroids (mm)
      - <5mm   : fraction of predicted survivors whose NN match is ≤ 5 mm
      - <10mm  : same with 10 mm
      - emd    : symmetric mean NN distance between the two centroid clouds (mm)

    This is the table to read **before** judging a per-patient overlay — the
    voxel rendering can mislead, but these centroid-level numbers tie directly
    to the held-out 5-fold metrics (T1 EMD ≈ 2.2 mm, alive-AUC 0.82).
    """
    try:
        from scipy.spatial import cKDTree
    except Exception:
        cKDTree = None

    rows = []
    for pid, desc, pcr in bundle.patients:
        res = bundle.results[pid].get(model_name, [])
        for k, r in enumerate(res):
            alive = r['alive_prob']
            pred  = r['pos_pred'][alive > 0.5]
            act   = r['pos_obs_dst']
            if len(pred) == 0 or len(act) == 0:
                continue
            if cKDTree is None:
                continue
            tA = cKDTree(act);  d_pa, _ = tA.query(pred, k=1)
            tP = cKDTree(pred); d_ap, _ = tP.query(act,  k=1)
            rows.append(dict(
                patient=pid, pCR=pcr, transition=f'{VISIT_IDS[k]}→{VISIT_IDS[k+1]}',
                n_pred=len(pred), n_act=len(act),
                med_NN_mm=float(np.median(d_pa)),
                pct_under_5=float((d_pa <= 5).mean() * 100.0),
                pct_under_10=float((d_pa <= 10).mean() * 100.0),
                emd_proxy_mm=float(0.5 * (d_pa.mean() + d_ap.mean())),
            ))
    return pd.DataFrame(rows)


def show_calibrated_survivors(bundle: VizBundle, pid: str,
                              model_name='S1.8 Bio + gated', k_show=0):
    """**Top-K calibrated** voxel overlay.

    The voxel overlay in Part 2 thresholds at p(alive) > 0.5, but the alive
    head's mean output is empirically near 0.5, so a 0.5 threshold is
    extremely sensitive to *calibration* rather than *ranking*. This view
    instead keeps the **top-K T_k voxels by p(alive)** where K is set so the
    predicted-survivor count matches the actual T_{k+1} voxel count.

    What this isolates: pure *discrimination* (does the model rank doomed vs
    surviving tissue correctly?), independent of calibration. If discrimination
    works, the predicted cloud should geometrically match the actual cloud
    even if absolute p(alive) is biased.
    """
    desc, pcr = next((d, p) for q, d, p in bundle.patients if q == pid)
    res = bundle.results[pid].get(model_name, [])
    if not res or k_show >= len(res):
        print(f'No prediction for {pid} k={k_show}'); return
    r = res[k_show]
    src_v = VISIT_IDS[k_show]; dst_v = VISIT_IDS[k_show + 1]
    t0_pos, t0_sv = bundle.voxels[pid][src_v]
    t1_pos, _     = bundle.voxels[pid][dst_v]
    if t0_pos is None or t1_pos is None:
        print(f'Missing voxels for {pid}'); return

    delta_pos   = r['pos_pred'] - r['pos_obs_src']
    sv_clipped  = np.clip(t0_sv, 0, len(delta_pos) - 1)
    voxel_delta = delta_pos[sv_clipped]
    voxel_alive = r['alive_prob'][sv_clipped]
    t0_centered = t0_pos - t0_pos.mean(axis=0, keepdims=True)
    pred_voxels = t0_centered + voxel_delta

    # Top-K calibrated: keep the K most-confident-alive voxels, where K =
    # number of actual T_{k+1} voxels. This removes the calibration bias.
    K = min(len(t1_pos), len(pred_voxels))
    if K == 0:
        print('No voxels to show.'); return
    top_idx = np.argpartition(-voxel_alive, K - 1)[:K]
    pred = pred_voxels[top_idx]
    palv = voxel_alive[top_idx]

    t1_centered = t1_pos - t1_pos.mean(axis=0, keepdims=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=t1_centered[:, 2], y=t1_centered[:, 1], z=t1_centered[:, 0],
        mode='markers',
        marker=dict(size=2, color='#888', opacity=0.55),
        name=f'actual {dst_v} ({len(t1_centered)} vox)'))
    fig.add_trace(go.Scatter3d(
        x=pred[:, 2], y=pred[:, 1], z=pred[:, 0],
        mode='markers',
        marker=dict(size=2,
                    color=color_by_alive_prob(np.arange(len(palv)), palv),
                    opacity=0.85),
        name=f'top-{K} predicted survivors'))

    # Cloud-level match metric: mean nearest-neighbour distance
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(t1_centered)
        d, _ = tree.query(pred, k=1)
        mean_d = float(d.mean()); med_d = float(np.median(d))
    except Exception:
        mean_d = med_d = float('nan')

    title = (f'<b>{pid}</b> · pCR = {pcr} · {model_name} · {src_v}→{dst_v} '
             f'(top-K calibrated)<br>'
             f'<sup>Predicted = top {K:,} T0 voxels by p(alive) (matches actual T+1 size). '
             f'Mean voxel NN distance: <b>{mean_d:.2f} mm</b> (median {med_d:.2f} mm).<br>'
             f'This isolates ranking quality from calibration.</sup>')

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center', y=0.97, yanchor='top',
                   font=dict(size=12)),
        scene=_scene_cfg(),
        height=560,
        showlegend=True,
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01,
                    bgcolor='rgba(255,255,255,0.7)'),
        margin=dict(t=110, b=40, l=10, r=10),
        paper_bgcolor='white',
    )
    fig.show()


def show_error_comparison(bundle: VizBundle, pid: str, k_show=0):
    """3-panel comparison: per-supervoxel position error across all 3 models."""
    res_dict = bundle.results[pid]
    model_names = list(bundle.models.keys())
    if not all(res_dict[m] for m in model_names):
        print(f'No predictions for {pid}'); return
    if k_show >= len(res_dict[model_names[0]]):
        return
    err_per_model = [np.array(res_dict[m][k_show]['match_dist_mm'])
                     for m in model_names]
    src_v = VISIT_IDS[k_show]
    transition = f'{VISIT_IDS[k_show]}→{VISIT_IDS[k_show+1]}'
    fig = fig_error_comparison(pid, bundle.voxels[pid][src_v],
                               err_per_model, model_names, transition=transition)
    fig.show()

    print()
    print(f'  {transition}   {"Model":<22s}  {"mean":>7s}  {"p50":>7s}  {"p90":>7s}')
    for mname, err in zip(model_names, err_per_model):
        print(f'             {mname:<22s}  {np.mean(err):>6.2f}m  '
              f'{np.percentile(err, 50):>6.2f}m  '
              f'{np.percentile(err, 90):>6.2f}m')


def show_error_comparison_all(bundle: VizBundle, pid: str):
    """Loop through all transitions and call show_error_comparison."""
    n_trans = max(len(bundle.results[pid][m]) for m in bundle.models)
    for k in range(n_trans):
        show_error_comparison(bundle, pid, k_show=k)


def show_overlay_slider(bundle: VizBundle, pid: str,
                        model_name='S1.8 Bio + gated', k_show=0,
                        alive_threshold: float = 0.5):
    """Overlay the **predicted T_{k+1} voxel cloud** on top of the actual
    T_{k+1} voxel cloud, in the same coordinate system.

    The predicted cloud is constructed properly:
      - Each T_k voxel is **displaced** by its supervoxel's predicted Δpos.
      - The cloud is **filtered** to only voxels whose supervoxel was
        predicted to survive (`p(alive) > alive_threshold`).
      - Both clouds are centred on their per-visit centroid (the same
        convention the model was trained under, removing the 20–80 mm
        inter-visit patient repositioning that the model does not model).

    The slider fades the predicted overlay in/out — at 0.0 only the actual
    cloud is visible, at 1.0 only the predicted cloud. In the middle you
    can spot:
      - **Tight overlap** → model nailed both location and survival fate
      - **Predicted cloud larger than actual** → over-predicting survival
      - **Predicted cloud smaller than actual** → over-predicting death
      - **Predicted cloud offset** → directional bias in the position head
    """
    desc, pcr = next((d, p) for q, d, p in bundle.patients if q == pid)
    res  = bundle.results[pid].get(model_name, [])
    if not res or k_show >= len(res):
        print(f'No prediction for {pid} k={k_show}'); return
    r = res[k_show]
    src_v = VISIT_IDS[k_show]; dst_v = VISIT_IDS[k_show + 1]
    t0_pos, t0_sv = bundle.voxels[pid][src_v]
    t1_pos, _     = bundle.voxels[pid][dst_v]
    if t0_pos is None or t1_pos is None:
        print(f'Missing voxels for {pid}'); return

    # Build predicted T_{k+1} voxel cloud
    delta_pos   = r['pos_pred'] - r['pos_obs_src']         # (N_sv, 3)
    sv_clipped  = np.clip(t0_sv, 0, len(delta_pos) - 1)
    voxel_delta = delta_pos[sv_clipped]
    voxel_alive = r['alive_prob'][sv_clipped]
    t0_centered = t0_pos - t0_pos.mean(axis=0, keepdims=True)
    pred_voxels = t0_centered + voxel_delta
    survives    = voxel_alive > alive_threshold
    pred_voxels = pred_voxels[survives]
    pred_alive  = voxel_alive[survives]

    # Centre actual T_{k+1} by its own centroid
    t1_centered = t1_pos - t1_pos.mean(axis=0, keepdims=True)

    n_pred  = int(survives.sum())
    n_t0    = len(t0_pos)
    n_t1    = len(t1_pos)
    pct_p   = 100.0 * n_pred / max(n_t0, 1)
    pct_a   = 100.0 * n_t1   / max(n_t0, 1)

    pred_colors = color_by_alive_prob(np.arange(len(pred_alive)), pred_alive)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=t1_centered[:, 2], y=t1_centered[:, 1], z=t1_centered[:, 0],
        mode='markers',
        marker=dict(size=2, color='#888', opacity=0.55),
        name=f'actual {dst_v}'))
    fig.add_trace(go.Scatter3d(
        x=pred_voxels[:, 2], y=pred_voxels[:, 1], z=pred_voxels[:, 0],
        mode='markers',
        marker=dict(size=2, color=pred_colors, opacity=0.85),
        name=f'predicted {dst_v} (survivors)'))

    # Slider that fades the predicted layer in/out
    steps = []
    for alpha in np.linspace(0, 1, 11):
        steps.append(dict(
            method='restyle',
            args=[{'marker.opacity': [0.55 * (1 - alpha) + 0.05,
                                       0.05 + 0.85 * alpha]}],
            label=f'{alpha:.1f}',
        ))
    sliders = [dict(active=10, currentvalue={'prefix': 'predicted ↔ actual: '},
                    steps=steps, pad=dict(t=40))]

    title = (f'<b>{pid}</b> · pCR = {pcr} · {model_name} · {src_v}→{dst_v}<br>'
             f'<sup>Predicted survivors: <b>{n_pred:,} ({pct_p:.0f}% of T0)</b> · '
             f'Actual size: <b>{n_t1:,} ({pct_a:.0f}% of T0)</b>. '
             f'Both centred on per-visit centroid.<br>'
             f'Drag slider to fade between '
             f'<span style="color:#888">actual (gray)</span> and '
             f'<span style="color:#d73027">predicted survivors</span>.</sup>')
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center', y=0.97, yanchor='top',
                   font=dict(size=12)),
        scene=_scene_cfg(),
        height=580,
        showlegend=True,
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01,
                    bgcolor='rgba(255,255,255,0.7)'),
        margin=dict(t=120, b=80, l=10, r=10),
        sliders=sliders,
        paper_bgcolor='white',
    )
    fig.show()
    print(f'  {src_v}→{dst_v}   centroid NN error: {np.mean(r["match_dist_mm"]):.2f} mm  '
          f'(measures supervoxel centroid prediction accuracy)')
    print(f'  Predicted/actual size ratio:  '
          f'{pct_p/max(pct_a,1e-6):.2f}× '
          f'(closer to 1.0 = better-calibrated survival)')


def alive_prob_panels(bundle: VizBundle, model_name='S1.8 Bio + gated',
                      k_show=0):
    """Histogram of `p(alive)` per supervoxel for each patient.

    What's plotted? For each patient, the Stage-1 forecaster outputs *one alive
    probability per supervoxel* at the source visit (here T_{k_show}). A patient
    with N supervoxels at T_{k_show} therefore contributes N values to the
    histogram. So if you're wondering where the distribution comes from: it is
    the spread of model confidence *across all supervoxels of one patient at
    one transition*. It is not a sampling distribution; it is the natural
    sub-tumour heterogeneity of the model's confidence.

    A bimodal histogram (mass near 0 and near 1) means the model is making
    decisive per-region calls — some supervoxels confidently die, others
    confidently survive. A unimodal mass at 0.5 means the model is uncertain
    and is essentially defaulting to a coin flip.
    """
    src_v = VISIT_IDS[k_show]; dst_v = VISIT_IDS[k_show + 1]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), tight_layout=True)
    fig.suptitle(f'{model_name} — distribution of p(alive) per supervoxel '
                 f'at {src_v}→{dst_v}',
                 fontsize=12, fontweight='bold', y=1.02)
    for ax, (pid, desc, pcr) in zip(axes.flat, bundle.patients):
        res = bundle.results[pid].get(model_name, [])
        if not res or k_show >= len(res):
            ax.set_visible(False); continue
        probs = res[0]['alive_prob']
        n_alive = (probs > 0.5).sum()
        ax.hist(probs, bins=20, range=(0, 1),
                color='#4CAF50', edgecolor='white', alpha=0.85)
        ax.axvline(0.5, color='#888', ls='--', lw=1, alpha=0.7)
        title_pcr = f'pCR={pcr}' if pcr is not None else 'pCR=?'
        ax.set_title(f'{pid}\n{desc} · {title_pcr}\n'
                     f'N={len(probs)} sv,  {n_alive}/{len(probs)} predicted alive',
                     fontsize=9)
        ax.set_xlabel('p(alive)', fontsize=9)
        ax.set_ylabel('# supervoxels', fontsize=9)
        ax.set_xlim(0, 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    return fig


def comparison_dataframe(bundle: VizBundle) -> pd.DataFrame:
    """Build a per-(patient × model × transition) error DataFrame."""
    rows = []
    for pid, desc, pcr in bundle.patients:
        for mname in bundle.models:
            for r in bundle.results[pid].get(mname, []):
                rows.append({
                    'patient':       pid,
                    'description':   desc,
                    'pCR':           pcr,
                    'model':         mname,
                    'transition':    f'T{r["k"]}→T{r["k"]+1}',
                    'mean_err_mm':   float(np.mean(r['match_dist_mm'])),
                    'p50_err_mm':    float(np.percentile(r['match_dist_mm'], 50)),
                    'p90_err_mm':    float(np.percentile(r['match_dist_mm'], 90)),
                    'mean_alive':    float(np.mean(r['alive_prob'])),
                })
    return pd.DataFrame(rows)


def model_comparison_bar(df: pd.DataFrame, model_names):
    """Mean per-supervoxel position error across all patients × transitions."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    means = df.groupby('model')['mean_err_mm'].mean()
    stds  = df.groupby('model')['mean_err_mm'].std()
    bars  = ax.bar(range(len(model_names)),
                   means[model_names], yerr=stds[model_names],
                   color=['#90A4AE', '#66BB6A', '#EF5350'],
                   edgecolor='white', capsize=5, width=0.55)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels([m.replace(' ', '\n') for m in model_names], fontsize=10)
    ax.set_ylabel('Mean NN positional error (mm)', fontsize=11)
    ax.set_title('Stage-1 forecaster: per-supervoxel position error\n'
                 '(6 patients × all transitions)', fontsize=11)
    for bar, m in zip(bars, means[model_names]):
        ax.text(bar.get_x() + bar.get_width()/2, m + 0.1,
                f'{m:.2f}\u00a0mm', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig
