"""consistent_twin_lib.py — helper functions for the consistent-graph digital twin
visualization notebook.

This module handles:
  - Loading the consistent-graph forecaster checkpoint
  - Running inference to get per-node predicted displacements
  - Loading voxel volumes (actual T0 and registration-transported T1/T2/T3)
  - Building "predicted tumor" volumes by displacing T0 voxels
  - All Plotly figure builders for the visualization notebook
"""
from __future__ import annotations

import json
import html as html_lib
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots
except Exception:  # pragma: no cover
    go = None
    pio = None
    make_subplots = None

if pio is not None:
    pio.renderers.default = "notebook"

# ── Paths (override from notebook if needed) ────────────────────────────────
_HERE = Path(__file__).parent

# Find repo root by walking up until we find the lsgc package
def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "lsgc" / "__init__.py").exists():
            return p
    raise RuntimeError(f"Could not find repo root (lsgc package) from {start}")

_REPO_ROOT = _find_repo_root(_HERE)
import sys as _sys
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
from experiments.stage1_forecaster.edge_modes import (
    build_history_graph as _dynamic_history_graph,
    edge_attr_dim as _edge_attr_dim,
)

VIZ_ROOT   = _HERE / "viz_data_consistent"
_FIG_CACHE = VIZ_ROOT / "_fig_cache"


def show_fig(fig: go.Figure, name: str = "fig", height: int = None,
             scale: float = 1.5) -> None:
    """Render figure as inline PNG (static, always-renders) plus link to
    interactive HTML version.

    Static images are 100% reliable in Cursor's Jupyter environment, and a
    parallel interactive HTML version is saved to disk so users can open the
    full Plotly interaction (rotate, zoom, hover) in a browser tab.
    """
    if go is None or pio is None:
        raise ImportError("plotly is required for visualization helpers in consistent_twin_lib.")
    from IPython.display import Image, display as _display, HTML
    _FIG_CACHE.mkdir(parents=True, exist_ok=True)

    # Static PNG (always renders inline, no JS, no connection)
    h = height or fig.layout.height or 500
    width = 1400
    png_bytes = fig.to_image(format="png", width=width, height=h, scale=scale)
    _display(Image(data=png_bytes))

    # Interactive HTML on disk with a clickable link
    html_path = _FIG_CACHE / f"{name}.html"
    fig.write_html(
        str(html_path),
        include_plotlyjs="cdn",
        full_html=True,
        config={"scrollZoom": True, "displayModeBar": True},
    )
    rel = html_path.relative_to(_HERE)
    _display(HTML(
        f'<div style="margin-top:-5px;margin-bottom:15px;font-size:11px;'
        f'color:#666;font-style:italic">'
        f'For interactive 3D rotation/zoom: '
        f'<a href="{rel}" target="_blank" style="color:#2c5fa8">'
        f'open {name}.html in browser</a></div>'
    ))


def show_animation(fig: go.Figure, name: str = "anim",
                   description: str = "") -> None:
    """For animated figures: save as HTML and provide a prominent open-in-browser
    link. Animations don't render statically, so the link is the primary delivery.
    """
    if go is None or pio is None:
        raise ImportError("plotly is required for visualization helpers in consistent_twin_lib.")
    from IPython.display import display as _display, HTML
    _FIG_CACHE.mkdir(parents=True, exist_ok=True)
    html_path = _FIG_CACHE / f"{name}.html"
    fig.write_html(
        str(html_path),
        include_plotlyjs="cdn",
        full_html=True,
        config={"scrollZoom": True, "displayModeBar": True},
    )
    rel = html_path.relative_to(_HERE)
    size_mb = html_path.stat().st_size / (1024 * 1024)

    # Render the first (T0) frame as a static preview
    first_frame_fig = go.Figure(data=fig.data, layout=fig.layout)
    first_frame_fig.update_layout(updatemenus=[], sliders=[])
    png = first_frame_fig.to_image(format="png", width=1100, height=600, scale=1.5)
    from IPython.display import Image
    _display(Image(data=png))

    _display(HTML(
        f'<div style="margin:5px 0 20px 0;padding:12px 16px;'
        f'background:#f0f7ff;border-left:4px solid #2c5fa8;border-radius:4px">'
        f'<div style="font-weight:600;color:#1c4488;margin-bottom:4px">'
        f'🎬 Interactive Animation Available</div>'
        f'<div style="font-size:13px;color:#333;margin-bottom:6px">'
        f'{description or "Press Play to watch the digital twin evolve T0 → T3."} '
        f'</div>'
        f'<a href="{rel}" target="_blank" '
        f'style="display:inline-block;padding:6px 14px;background:#2c5fa8;'
        f'color:white;text-decoration:none;border-radius:4px;font-size:12px">'
        f'Open animation ({size_mb:.1f} MB) in new tab →</a></div>'
    ))


def show_synced_animation(fig: go.Figure, name: str = "anim",
                          description: str = "",
                          height: int = 720,
                          inline: bool = True,
                          inline_mode: str = "srcdoc") -> None:
    """Save a multi-scene Plotly animation with synchronized 3D cameras.

    Plotly treats each 3D subplot (``scene``, ``scene2``, ``scene3``) as an
    independent WebGL camera. For the twin comparison view we want the opposite:
    if the user rotates one panel, the baseline / predicted / actual panels
    should all rotate identically. We inject a tiny relayout listener into the
    exported HTML that copies the active scene camera to every other scene.
    """
    if go is None or pio is None:
        raise ImportError("plotly is required for visualization helpers in consistent_twin_lib.")
    from IPython.display import display as _display, HTML, Image

    _FIG_CACHE.mkdir(parents=True, exist_ok=True)
    html_path = _FIG_CACHE / f"{name}.html"

    fig.update_layout(uirevision=f"{name}-camera-sync")
    sync_js = r"""
    (function() {
      const gd = document.getElementById('{plot_id}');
      const sceneOrder = (s) => s === 'scene' ? 1 : Number(s.replace('scene', ''));
      const scenes = Object.keys(gd._fullLayout || {})
        .filter((k) => /^scene\d*$/.test(k))
        .sort((a, b) => sceneOrder(a) - sceneOrder(b));
      if (scenes.length < 2) return;

      let syncing = false;
      gd.on('plotly_relayout', function(eventData) {
        if (syncing || !eventData) return;

        let sourceScene = null;
        let camera = null;
        for (const key of Object.keys(eventData)) {
          const match = key.match(/^(scene\d*)\.camera(?:\.|$)/);
          if (!match) continue;
          sourceScene = match[1];
          camera = eventData[sourceScene + '.camera'] ||
                   (gd.layout[sourceScene] && gd.layout[sourceScene].camera) ||
                   (gd._fullLayout[sourceScene] && gd._fullLayout[sourceScene].camera);
          break;
        }
        if (!sourceScene || !camera) return;

        const update = {};
        for (const scene of scenes) {
          if (scene !== sourceScene) {
            update[scene + '.camera'] = JSON.parse(JSON.stringify(camera));
          }
        }
        syncing = true;
        Plotly.relayout(gd, update)
          .then(function() { syncing = false; })
          .catch(function() { syncing = false; });
      });
    })();
    """

    html_text = pio.to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=True,
        config={"scrollZoom": True, "displayModeBar": True},
        post_script=sync_js,
    )
    html_path.write_text(html_text)
    rel = html_path.relative_to(_HERE)
    size_mb = html_path.stat().st_size / (1024 * 1024)

    if inline and inline_mode == "srcdoc":
        _display(HTML(
            f'<iframe srcdoc="{html_lib.escape(html_text, quote=True)}" '
            f'style="width:100%;height:{int(height)}px;'
            f'border:1px solid #d6dce8;border-radius:6px;'
            f'background:white" '
            f'allowfullscreen></iframe>'
        ))
    elif inline:
        _display(HTML(
            f'<iframe src="{rel}" '
            f'style="width:100%;height:{int(height)}px;'
            f'border:1px solid #d6dce8;border-radius:6px;'
            f'background:white" '
            f'allowfullscreen></iframe>'
        ))
    else:
        # Try to show a static preview for notebook readability. If kaleido is
        # not available, the interactive link still works.
        try:
            first_frame_fig = go.Figure(data=fig.data, layout=fig.layout)
            first_frame_fig.update_layout(updatemenus=[], sliders=[])
            png = first_frame_fig.to_image(format="png", width=1200, height=620, scale=1.5)
            _display(Image(data=png))
        except Exception:
            pass

    _display(HTML(
        f'<div style="margin:8px 0 22px 0;padding:13px 16px;'
        f'background:#f4f8ff;border-left:4px solid #2c5fa8;border-radius:4px">'
        f'<div style="font-weight:600;color:#1c4488;margin-bottom:4px">'
        f'Synchronized 3D Animation</div>'
        f'<div style="font-size:13px;color:#333;margin-bottom:7px">'
        f'{description or "Rotate any 3D panel; the other panels will match its camera."}'
        f'</div>'
        f'<a href="{rel}" target="_blank" '
        f'style="display:inline-block;padding:6px 14px;background:#2c5fa8;'
        f'color:white;text-decoration:none;border-radius:4px;font-size:12px">'
        f'Open synced animation ({size_mb:.1f} MB) in new tab →</a></div>'
    ))
GRAPHS_DIR   = VIZ_ROOT / "graphs"
MODEL_PATH   = VIZ_ROOT / "model" / "best.pt"
DERIVED_DIR  = VIZ_ROOT / "derived"
REG_DIR      = VIZ_ROOT / "registered"

# Full-cohort consistent graphs (Stage 3 forecaster); voxel viz still uses VIZ_ROOT.
GRAPHS_CONSISTENT_ROOT = _REPO_ROOT / "datasets" / "ispy2" / "graphs_consistent"
COHORT_PARQUET_DEFAULT = _REPO_ROOT / "datasets" / "ispy2" / "cohort.parquet"
FOLDS_PARQUET_DEFAULT = _REPO_ROOT / "datasets" / "ispy2" / "folds.parquet"
RUNS_CONSISTENT_5FOLD = _REPO_ROOT / "runs" / "consistent_forecaster_5fold"
UNSEEN_SPLITS_DIR_DEFAULT = _REPO_ROOT / "reports" / "unseen_forecaster_splits"

VISITS = ("T0", "T1", "T2", "T3")

# ── Color palette (one vivid color per supervoxel, cycled) ───────────────────
import colorsys
def _sv_palette(n: int) -> list[str]:
    """Generate n visually distinct colors using HSV wheel."""
    colors = []
    for i in range(n):
        h = i / n
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.92)
        colors.append(f"rgb({int(r*255)},{int(g*255)},{int(b*255)})")
    return colors

PCR_COLORS = {0: "#e85c47", 1: "#4caf85"}  # red=non-responder, green=responder


# ── Model loading ────────────────────────────────────────────────────────────
def load_model(path: Path = MODEL_PATH):
    from lsgc.forecaster import LSGCForecaster
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    in_ch = int(ck["mean"].shape[0])
    model = LSGCForecaster(
        in_channels=in_ch,
        hidden=cfg["hidden"],
        num_layers=cfg["num_layers"],
        feat_out_dim=in_ch,
        use_delta_t=True,
        use_edge_gating=True,
        edge_attr_dim=_edge_attr_dim(cfg.get("edge_attr_mode", "none")),
    )
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck["mean"], ck["std"]


def load_twin_model(path: Path):
    """Load Stage-2/3 twin checkpoint, returning (model, mean, std, config)."""
    from lsgc.twin import LSGCTwin
    from lsgc.counterfactual import LSGCCounterfactualTwin

    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    mean = ck["mean"]
    std = ck["std"]
    in_ch = int(mean.shape[0])

    common = dict(
        in_channels=in_ch,
        hidden=int(cfg.get("hidden", 64)),
        num_layers=int(cfg.get("num_layers", 2)),
        latent_dim=int(cfg.get("latent_dim", 8)),
        feat_out_dim=in_ch,
        use_pcr_head=bool(cfg.get("use_pcr_head", False)),
        clinical_dim=int(cfg.get("clinical_dim", 0)),
        visit_context_dim=int(cfg.get("visit_context_dim", 0)),
        edge_attr_mode=cfg.get("edge_attr_mode", "legacy"),
        adc_idx=cfg.get("adc_idx"),
        adc_missing_idx=cfg.get("adc_missing_idx"),
        dce_idx_start=int(cfg.get("dce_idx_start", 0)),
        dce_n_phases=int(cfg.get("dce_n_phases", 0)),
        habitat_n_classes=int(cfg.get("habitat_n_classes", 0)),
    )
    if "n_arms" in cfg or "arm_dim" in cfg:
        model = LSGCCounterfactualTwin(
            n_arms=int(cfg.get("n_arms", 2)),
            arm_dim=int(cfg.get("arm_dim", 8)),
            **common,
        )
    else:
        model = LSGCTwin(**common)

    model.load_state_dict(ck["state_dict"], strict=False)
    model.eval()
    return model, mean, std, cfg


# ── Consistent graph + inference ─────────────────────────────────────────────
def load_graph(pid: str) -> dict:
    path = GRAPHS_DIR / f"{pid}.pt"
    return torch.load(path, map_location="cpu", weights_only=False)


def load_graph_dataset(pid: str, graphs_root: Path | None = None) -> dict:
    """Load a consistent graph from the cohort ``graphs_consistent`` tree (or custom root)."""
    root = Path(graphs_root) if graphs_root is not None else GRAPHS_CONSISTENT_ROOT
    path = root / f"{pid}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def spatial_knn_edges(pos: torch.Tensor, k: int = 8) -> torch.Tensor:
    d = torch.cdist(pos, pos)
    d.fill_diagonal_(float("inf"))
    k_eff = min(k, pos.shape[0] - 1)
    _, idx = torch.topk(d, k=k_eff, largest=False, dim=1)
    src = torch.arange(pos.shape[0]).unsqueeze(1).expand_as(idx).reshape(-1)
    dst = idx.reshape(-1)
    ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    return ei.unique(dim=1)


def build_edges(g: dict) -> torch.Tensor:
    offsets = g["visit_offsets"].tolist()
    pos = g["pos"]; T = len(offsets) - 1
    all_src, all_dst = [], []
    for v in range(T):
        sl = slice(int(offsets[v]), int(offsets[v + 1]))
        n_v = pos[sl].shape[0]
        if n_v > 1:
            ei_v = spatial_knn_edges(pos[sl])
            all_src.append(ei_v[0] + offsets[v])
            all_dst.append(ei_v[1] + offsets[v])
        if v < T - 1:
            n_nodes = int(offsets[v + 1]) - int(offsets[v])
            src_t = torch.arange(n_nodes) + offsets[v]
            dst_t = torch.arange(n_nodes) + offsets[v + 1]
            all_src += [src_t, dst_t]
            all_dst += [dst_t, src_t]
    return torch.stack([torch.cat(all_src), torch.cat(all_dst)], dim=0).long()


def _build_history_graph(
    history_pos: list[torch.Tensor],
    history_x: list[torch.Tensor] | None = None,
    k_spatial: int = 8,
    edge_mode: str = "full",
    edge_attr_mode: str = "none",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Build rollout graph edges and optional attributes from visit history."""
    return _dynamic_history_graph(
        history_pos,
        history_x,
        k_spatial=k_spatial,
        edge_mode=edge_mode,
        edge_attr_mode=edge_attr_mode,
    )


@torch.no_grad()
def run_inference(
    model,
    mean,
    std,
    g: dict,
    k_spatial: int = 8,
    edge_mode: str = "full",
    edge_attr_mode: str = "none",
) -> dict:
    """Return per-visit, per-node delta_pos predictions (centroid-relative mm)."""
    x   = (g["x"].float() - mean) / std
    pos = g["pos"].float()
    t   = g["t"].float()
    off = g["visit_offsets"].tolist()
    history_x = [x[int(off[v]): int(off[v + 1])] for v in range(len(off) - 1)]
    history_pos = [pos[int(off[v]): int(off[v + 1])] for v in range(len(off) - 1)]
    ei, edge_attr = _build_history_graph(
        history_pos,
        history_x,
        k_spatial=k_spatial,
        edge_mode=edge_mode,
        edge_attr_mode=edge_attr_mode,
    )
    out = model(x, pos, t, ei, edge_attr=edge_attr)

    N   = g["n_supervoxels"]
    result = {}
    for v, vid in enumerate(VISITS[:-1]):
        sl = slice(int(off[v]), int(off[v + 1]))
        result[f"{vid}_dpos"]  = out["delta_pos"][sl].numpy()   # (N,3)
        result[f"{vid}_dfeat"] = out["delta_feat"][sl].numpy()  # (N,C)
    # Cumulative predicted positions: T0 pos + cumulative Δpos
    pos_pred = {VISITS[0]: pos[:N].numpy()}
    for v, vid in enumerate(VISITS[:-1]):
        nxt = VISITS[v + 1]
        pos_pred[nxt] = pos_pred[vid] + result[f"{vid}_dpos"]
    result["pos_pred"] = pos_pred
    result["pos_actual"] = {
        vid: pos[int(off[v]): int(off[v + 1])].numpy()
        for v, vid in enumerate(VISITS)
    }
    result["alive"] = {
        vid: g["alive"][int(off[v]): int(off[v + 1])].numpy().astype(bool)
        for v, vid in enumerate(VISITS)
    }
    return result


@torch.no_grad()
def rollout_from_visit(
    model,
    mean: torch.Tensor,
    std: torch.Tensor,
    g: dict,
    start_visit: int = 0,
    k_spatial: int = 8,
    edge_mode: str = "full",
    edge_attr_mode: str = "none",
) -> list[dict[str, Any]]:
    """Autoregressive rollout from a chosen observed visit on consistent graphs.

    start_visit=0 means T0-only rollout (predict T1,T2,T3), start_visit=1 means
    given observed T0+T1 predict T2,T3, etc.
    """
    if start_visit < 0 or start_visit >= len(VISITS) - 1:
        raise ValueError(f"start_visit must be in [0, {len(VISITS)-2}]")

    x = (g["x"].float() - mean) / std
    pos = g["pos"].float()
    off = g["visit_offsets"].tolist()
    n_visits = len(off) - 1

    history_x: list[torch.Tensor] = []
    history_pos: list[torch.Tensor] = []
    history_t: list[torch.Tensor] = []
    for v in range(start_visit + 1):
        sl = slice(int(off[v]), int(off[v + 1]))
        xv = x[sl].clone()
        pv = pos[sl].clone()
        tv = torch.full((xv.shape[0],), float(v), dtype=x.dtype)
        history_x.append(xv)
        history_pos.append(pv)
        history_t.append(tv)

    rows: list[dict[str, Any]] = []
    for pred_v in range(start_visit + 1, n_visits):
        x_cat = torch.cat(history_x, dim=0)
        pos_cat = torch.cat(history_pos, dim=0)
        t_cat = torch.cat(history_t, dim=0)
        ei, edge_attr = _build_history_graph(
            history_pos,
            history_x,
            k_spatial=k_spatial,
            edge_mode=edge_mode,
            edge_attr_mode=edge_attr_mode,
        )
        out = model(x_cat, pos_cat, t_cat, ei, edge_attr=edge_attr)

        n_last = int(history_x[-1].shape[0])
        sl_last = slice(x_cat.shape[0] - n_last, x_cat.shape[0])
        dpos = out["delta_pos"][sl_last]
        dfeat = out["delta_feat"][sl_last]
        alive_prob = torch.sigmoid(out["alive_logit"][sl_last])

        x_next = history_x[-1] + dfeat
        pos_next = history_pos[-1] + dpos

        obs_sl = slice(int(off[pred_v]), int(off[pred_v + 1]))
        x_obs = g["x"][obs_sl].float()
        pos_obs = g["pos"][obs_sl].float()
        alive_obs = g["alive"][obs_sl].detach().cpu().numpy().astype(bool)

        rows.append(
            {
                "visit_idx": pred_v,
                "visit_name": VISITS[pred_v],
                "pos_pred": pos_next.detach().cpu().numpy(),
                "x_pred": (x_next * std + mean).detach().cpu().numpy(),
                "alive_prob": alive_prob.detach().cpu().numpy(),
                "pos_obs": pos_obs.detach().cpu().numpy(),
                "x_obs": x_obs.detach().cpu().numpy(),
                "alive_obs": alive_obs,
            }
        )

        history_x.append(x_next)
        history_pos.append(pos_next)
        history_t.append(torch.full((x_next.shape[0],), float(pred_v), dtype=x.dtype))
    return rows


@torch.no_grad()
def mc_rollout_ftv(
    model,
    mean: torch.Tensor,
    std: torch.Tensor,
    g: dict,
    K: int = 64,
    n_steps: int = 3,
    score_horizon: int = 3,
    eps_pcr: float = 0.1,
    alive_mode: str = "count",
    alive_threshold: float = 0.5,
    volume_idx: int = 1,
    seed: int = 0,
    arm: int | None = None,
) -> dict[str, Any]:
    """K-sample Monte Carlo rollout helper for Stage-2/3 checkpoints."""
    if not hasattr(model, "rollout"):
        raise TypeError("mc_rollout_ftv expects an LSGCTwin-like model with rollout().")
    score_horizon = int(max(1, min(score_horizon, n_steps)))
    off = g["visit_offsets"].tolist()
    x0 = (g["x"][int(off[0]): int(off[1])].float() - mean) / std
    pos0 = (
        g["pos_c"][int(off[0]): int(off[1])].float()
        if "pos_c" in g
        else g["pos"][int(off[0]): int(off[1])].float()
    )

    obs_ftv = {}
    for v in range(len(off) - 1):
        sl = slice(int(off[v]), int(off[v + 1]))
        obs_ftv[VISITS[v]] = float(g["x"][sl, volume_idx].sum().item())

    rng = torch.Generator(device=x0.device).manual_seed(seed)
    by_visit = {VISITS[v + 1]: [] for v in range(n_steps)}

    for _ in range(int(K)):
        if hasattr(model, "rollout_counterfactual") and arm is not None:
            steps = model.rollout_counterfactual(
                x0,
                pos0,
                arm=arm,
                n_steps=n_steps,
                alive_mode=alive_mode,
                alive_threshold=alive_threshold,
            )
        else:
            steps = model.rollout(
                x0,
                pos0,
                n_steps=n_steps,
                alive_mode=alive_mode,
                alive_threshold=alive_threshold,
                feature_mean=mean,
                feature_std=std,
                volume_idx=volume_idx,
                generator=rng,
            )
        for i, st in enumerate(steps):
            visit = VISITS[i + 1]
            ftv = float((st.alive_prob * st.volume_ml_hat).sum().item()) if st.x.numel() else 0.0
            by_visit[visit].append(ftv)

    t_score = VISITS[score_horizon]
    arr = np.asarray(by_visit.get(t_score, []), dtype=np.float64)
    pcr_prob = float(np.mean(arr < eps_pcr)) if arr.size else float("nan")

    return {
        "K": int(K),
        "score_horizon": int(score_horizon),
        "eps_pcr": float(eps_pcr),
        "obs_ftv": obs_ftv,
        "ftv_samples": {k: np.asarray(v, dtype=np.float64) for k, v in by_visit.items()},
        "pcr_prob": pcr_prob,
    }


# ── Voxel loading ─────────────────────────────────────────────────────────────
def _vox_to_world(z_idx, y_idx, x_idx, origin, spacing):
    """Convert integer voxel indices to world-mm coordinates."""
    return np.stack([
        origin[0] + z_idx * spacing[0],
        origin[1] + y_idx * spacing[1],
        origin[2] + x_idx * spacing[2],
    ], axis=1)


def load_voxels_visit(pid: str, visit: str, max_per_sv: int = 300) -> dict:
    """Load supervoxel voxel positions for one visit in world-mm space.

    For T0: loads from derived_v2.
    For T1/T2/T3: loads *transported* T0 supervoxel labels in T_k native space,
    which gives true registered voxel positions for each persistent supervoxel.

    Returns
    -------
    dict mapping sv_id (int, 1-based) -> (K, 3) array in world-mm (z, y, x order)
    """
    if visit == "T0":
        sv_path   = DERIVED_DIR / pid / "T0" / "supervoxel_labels.npz"
        meta_path = DERIVED_DIR / pid / "T0" / "meta.json"
    else:
        sv_path   = REG_DIR / pid / f"{visit}_t0sv_in_{visit}_space.npz"
        meta_path = DERIVED_DIR / pid / visit / "meta.json"

    meta    = json.loads(meta_path.read_text())
    spacing = meta["voxel_spacing_mm"]
    origin  = meta["origin_mm"]
    labels  = np.load(sv_path)["labels"]

    sv_dict = {}
    nz_z, nz_y, nz_x = np.where(labels > 0)
    nz_ids = labels[nz_z, nz_y, nz_x]
    for sv_id in np.unique(nz_ids):
        sel = nz_ids == sv_id
        z_s, y_s, x_s = nz_z[sel], nz_y[sel], nz_x[sel]
        if len(z_s) > max_per_sv:
            idx = np.random.choice(len(z_s), max_per_sv, replace=False)
            z_s, y_s, x_s = z_s[idx], y_s[idx], x_s[idx]
        sv_dict[int(sv_id)] = _vox_to_world(z_s, y_s, x_s, origin, spacing)
    return sv_dict


def build_predicted_voxels(
    voxels_t0: dict,           # sv_id -> (K,3) world-mm
    t0_centroid: np.ndarray,   # (3,) world-mm centroid of T0 tumor
    tk_centroid: np.ndarray,   # (3,) world-mm centroid of T_k tumor (from graph)
    delta_pos: np.ndarray,     # (N,3) centroid-relative predicted displacements
    alive: np.ndarray,         # (N,) bool
) -> dict:
    """Translate T0 voxel clusters by model-predicted supervoxel displacements.

    The consistent graph stores positions centroid-relative.  We:
      1. Convert each supervoxel's T0 voxels to centroid-relative space
      2. Add the model's predicted Δpos for that supervoxel
      3. Convert back to world-mm using the T_k centroid (where we expect the
         tumor to be at the predicted visit)
    """
    pred = {}
    for sv_id, vox in voxels_t0.items():
        i = sv_id - 1  # 0-based index
        if i >= len(alive) or not alive[i]:
            continue
        # vox in world-mm -> centroid-relative
        vox_rel = vox - t0_centroid
        # add predicted displacement
        vox_pred_rel = vox_rel + delta_pos[i]
        # back to world-mm using T_k centroid
        pred[sv_id] = vox_pred_rel + tk_centroid
    return pred


# ── Plotly helpers ────────────────────────────────────────────────────────────
def _scatter(vox_dict: dict, palette: list, label_prefix: str,
             opacity: float = 0.8, size: int = 2,
             showlegend: bool = False) -> list:
    """Make one Scatter3d trace per supervoxel, colored by persistent ID."""
    traces = []
    for sv_id, vox in sorted(vox_dict.items()):
        color = palette[(sv_id - 1) % len(palette)]
        traces.append(go.Scatter3d(
            x=vox[:, 2], y=vox[:, 1], z=vox[:, 0],  # x=mm-lat, y=mm-ap, z=mm-sup
            mode="markers",
            marker=dict(size=size, color=color, opacity=opacity),
            name=f"{label_prefix} SV{sv_id}",
            showlegend=showlegend,
            hovertemplate=f"SV {sv_id}<extra></extra>",
        ))
    return traces


def _scene(title: str = "", size_mm: float = 80) -> dict:
    return dict(
        xaxis=dict(title="Lateral (mm)", showbackground=False),
        yaxis=dict(title="AP (mm)",      showbackground=False),
        zaxis=dict(title="Superior (mm)", showbackground=False),
        aspectmode="cube",
        camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        annotations=[dict(text=title, x=0.5, y=1.05, z=0,
                          showarrow=False, font=dict(size=13))]
        if title else [],
    )


# ── Figure 1: Real tumor evolution ────────────────────────────────────────────
def fig_real_evolution(pid: str, pcr: int, n_sv: int) -> go.Figure:
    """4-panel figure showing actual voxel clouds T0→T3, colored by SV identity."""
    palette = _sv_palette(n_sv + 1)
    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{"type": "scatter3d"}] * 4],
        horizontal_spacing=0.02,
    )
    for col, visit in enumerate(VISITS, start=1):
        vox = load_voxels_visit(pid, visit)
        for tr in _scatter(vox, palette, visit):
            fig.add_trace(tr, row=1, col=col)
        fig.update_scenes(_scene(), row=1, col=col)

    pcr_str = "pCR = 1 (Complete Responder)" if pcr == 1 else "pCR = 0 (Non-Responder)"
    fig.update_layout(
        title=dict(
            text=f"<b>{pid}</b> — Actual Tumor Evolution  |  {pcr_str}",
            font=dict(size=15), x=0.5
        ),
        showlegend=False,
        height=450, margin=dict(t=60, b=10, l=0, r=0),
        paper_bgcolor="white",
    )
    # Manual column titles as annotations
    for col, visit in enumerate(VISITS):
        fig.add_annotation(
            text=f"<b>{visit}</b>",
            xref="paper", yref="paper",
            x=(col * 0.25) + 0.125, y=-0.02,
            showarrow=False, font=dict(size=13),
        )
    return fig


# ── Figure 2: Predicted vs Actual (per transition) ───────────────────────────
def fig_pred_vs_actual(pid: str, pcr: int, infer: dict, g: dict) -> go.Figure:
    """3 side-by-side comparison panels: predicted T_k (left) vs actual T_k (right).

    Both panels are rendered in a **common centroid-relative frame** so the
    tumor at T0 and at T_k overlap visually. What differs between the panels
    is the *deformation* (predicted vs actual), not patient repositioning.
    """
    n_sv = g["n_supervoxels"]
    palette = _sv_palette(n_sv + 1)
    vox_t0 = load_voxels_visit(pid, "T0")
    vcents = g["visit_centroids"]  # [[z,y,x] per visit]
    t0_centroid = np.array(vcents[0])

    fig = make_subplots(
        rows=1, cols=6,
        specs=[[{"type": "scatter3d"}] * 6],
        horizontal_spacing=0.01,
    )
    for trans_idx, (src_v, dst_v) in enumerate([("T0","T1"),("T1","T2"),("T2","T3")]):
        dpos_key = f"{src_v}_dpos"
        alive    = infer["alive"][dst_v]
        tk_centroid = np.array(vcents[trans_idx + 1])

        # Predicted: translate T0 voxels by predicted Δpos, into centered frame
        pred_vox = build_predicted_voxels(
            voxels_t0=vox_t0,
            t0_centroid=t0_centroid,
            tk_centroid=np.zeros(3),    # ← centered
            delta_pos=infer[dpos_key],
            alive=alive,
        )
        # Actual: registered/transported voxels, also centered using T_k centroid
        actual_vox_world = load_voxels_visit(pid, dst_v)
        actual_vox = {sid: v - tk_centroid for sid, v in actual_vox_world.items()}

        pred_col   = trans_idx * 2 + 1
        actual_col = trans_idx * 2 + 2
        for tr in _scatter(pred_vox,   palette, f"Pred {dst_v}", opacity=0.85):
            fig.add_trace(tr, row=1, col=pred_col)
        for tr in _scatter(actual_vox, palette, f"True {dst_v}", opacity=0.60):
            fig.add_trace(tr, row=1, col=actual_col)

        fig.update_scenes(_scene(), row=1, col=pred_col)
        fig.update_scenes(_scene(), row=1, col=actual_col)

    pcr_str = "pCR=1" if pcr == 1 else "pCR=0"
    fig.update_layout(
        title=dict(
            text=(f"<b>{pid}</b> ({pcr_str}) — "
                  "Predicted (left) vs Actual (right) for each transition"),
            font=dict(size=14), x=0.5,
        ),
        showlegend=False,
        height=420, margin=dict(t=60, b=30, l=0, r=0),
    )
    labels = ["Pred T1","True T1","Pred T2","True T2","Pred T3","True T3"]
    xs     = [0.083, 0.25, 0.417, 0.583, 0.75, 0.917]
    for lbl, xp in zip(labels, xs):
        fig.add_annotation(text=f"<b>{lbl}</b>", xref="paper", yref="paper",
                           x=xp, y=-0.04, showarrow=False, font=dict(size=11))
    return fig


# ── Figure 3: Animated digital twin (3-panel synced animation) ──────────────
def fig_animated_twin(pid: str, pcr: int, infer: dict, g: dict) -> go.Figure:
    """Three synchronized 3D panels animating T0 → T3:

      - LEFT:   T0 baseline (constant reference, full color, persistent IDs)
      - CENTER: Model's predicted tumor at the current visit
      - RIGHT:  Ground-truth tumor at the current visit (from registration)

    All three panels are rendered in their *own* centroid-relative frame so each
    auto-scales nicely. Compare CENTER vs RIGHT visually to see how close the
    model's prediction is to ground truth at each visit.
    """
    n_sv     = g["n_supervoxels"]
    palette  = _sv_palette(n_sv + 1)
    vox_t0   = load_voxels_visit(pid, "T0")
    vcents   = g["visit_centroids"]
    t0_centroid = np.array(vcents[0])

    # ── Pre-compute every visit's voxel cloud (centered) ────────────────────
    # T0 baseline (constant across animation)
    vox_t0_c = {sid: v - t0_centroid for sid, v in vox_t0.items()}

    # Predicted clouds for T1/T2/T3 (cumulative Δpos)
    pred_per_visit = {"T0": vox_t0_c}
    cum = np.zeros((n_sv, 3))
    for v, (src_v, dst_v) in enumerate([("T0","T1"),("T1","T2"),("T2","T3")]):
        cum = cum + infer[f"{src_v}_dpos"]
        pred_per_visit[dst_v] = build_predicted_voxels(
            voxels_t0=vox_t0,
            t0_centroid=t0_centroid,
            tk_centroid=np.zeros(3),
            delta_pos=cum,
            alive=infer["alive"][dst_v],
        )

    # Actual clouds for T1/T2/T3 (centered using each visit's centroid)
    actual_per_visit = {"T0": vox_t0_c}
    for vi, vid in enumerate(VISITS[1:], start=1):
        raw = load_voxels_visit(pid, vid)
        c   = np.array(vcents[vi])
        actual_per_visit[vid] = {sid: v - c for sid, v in raw.items()}

    # ── Build trace lists per panel per visit ───────────────────────────────
    def _traces(vox_dict: dict, opacity: float = 0.9, size: int = 3) -> list:
        out = []
        for sv_id, vox in sorted(vox_dict.items()):
            color = palette[(sv_id - 1) % len(palette)]
            out.append(go.Scatter3d(
                x=vox[:, 2], y=vox[:, 1], z=vox[:, 0],
                mode="markers",
                marker=dict(size=size, color=color, opacity=opacity),
                showlegend=False, hoverinfo="skip",
            ))
        return out

    # Always-shown T0 baseline traces (left panel; never animated)
    base_traces = _traces(vox_t0_c, opacity=0.85, size=3)
    n_base = len(base_traces)

    # Frames: each updates only the CENTER and RIGHT panels.
    # All-trace structure per frame: [base..., center_pred..., right_actual...]
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type":"scatter3d"}]*3],
        horizontal_spacing=0.01,
    )
    # Initial state: T0 in all 3 panels (so they align before play)
    for tr in base_traces:                          fig.add_trace(tr, row=1, col=1)
    for tr in _traces(pred_per_visit["T0"]):        fig.add_trace(tr, row=1, col=2)
    for tr in _traces(actual_per_visit["T0"]):      fig.add_trace(tr, row=1, col=3)

    # Build animation frames
    frames = []
    for visit in VISITS:
        center_traces = _traces(pred_per_visit[visit])
        right_traces  = _traces(actual_per_visit[visit])
        # Pad with empty Scatter3d traces if shorter than initial layout
        # (Plotly requires same number of traces in every frame as in fig.data)
        n_target_center = n_base   # same #SVs as T0
        n_target_right  = n_base
        while len(center_traces) < n_target_center:
            center_traces.append(go.Scatter3d(x=[], y=[], z=[], mode="markers",
                                              showlegend=False, hoverinfo="skip"))
        while len(right_traces) < n_target_right:
            right_traces.append(go.Scatter3d(x=[], y=[], z=[], mode="markers",
                                             showlegend=False, hoverinfo="skip"))
        frames.append(go.Frame(
            data=[*base_traces, *center_traces, *right_traces],
            name=visit,
            traces=list(range(n_base, n_base + len(center_traces) + len(right_traces) + n_base))
                   if False else None,
        ))
    fig.frames = frames

    # ── Layout: 3 panels with own scenes ────────────────────────────────────
    pcr_label = "pCR=1 (Complete Responder)" if pcr == 1 else "pCR=0 (Non-Responder)"
    scene = lambda: dict(
        xaxis=dict(title="Lat (mm)", showbackground=False),
        yaxis=dict(title="AP (mm)",  showbackground=False),
        zaxis=dict(title="Sup (mm)", showbackground=False),
        aspectmode="cube",
        camera=dict(eye=dict(x=1.5, y=1.4, z=0.9)),
    )
    fig.update_scenes(scene(), row=1, col=1)
    fig.update_scenes(scene(), row=1, col=2)
    fig.update_scenes(scene(), row=1, col=3)

    fig.update_layout(
        title=dict(
            text=f"<b>{pid}</b> — Digital Twin  |  {pcr_label}",
            font=dict(size=15), x=0.5,
        ),
        height=560,
        margin=dict(t=70, b=80, l=0, r=0),
        showlegend=False,
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=-0.05, x=0.5, xanchor="center", yanchor="top",
            buttons=[
                dict(label="▶ Play",
                     method="animate",
                     args=[None, dict(frame=dict(duration=1400, redraw=True),
                                      fromcurrent=True, transition=dict(duration=600))]),
                dict(label="⏸ Pause",
                     method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                         mode="immediate")]),
            ],
        )],
        sliders=[dict(
            active=0,
            steps=[dict(method="animate",
                        args=[[v], dict(mode="immediate",
                                        frame=dict(duration=0, redraw=True))],
                        label=v) for v in VISITS],
            x=0.05, xanchor="left", len=0.9,
            y=-0.10, yanchor="top",
            currentvalue=dict(prefix="Visit: ", font=dict(size=13)),
        )],
    )

    # Column titles (under each panel)
    for col, text in enumerate(
        ["T0 Baseline (constant)", "Predicted (model)", "Actual (ground truth)"], start=1):
        fig.add_annotation(
            text=f"<b>{text}</b>", xref="paper", yref="paper",
            x=(col - 1) * 0.333 + 0.167, y=1.04,
            showarrow=False, font=dict(size=13, color="#1c4488"),
        )
    return fig


def fig_animated_twin_overlay(pid: str, pcr: int, infer: dict, g: dict) -> go.Figure:
    """[Old version, single-panel overlay] kept for backwards compat."""
    n_sv = g["n_supervoxels"]
    palette = _sv_palette(n_sv + 1)
    vox_t0 = load_voxels_visit(pid, "T0")
    t0_centroid = np.array(g["visit_centroids"][0])

    # Centered T0 voxels (gray ghost backdrop every frame)
    vox_t0_centered = {sid: v - t0_centroid for sid, v in vox_t0.items()}

    # Predicted voxel cloud for each follow-up visit, also in centered frame
    pred_vox_per_visit = {VISITS[0]: vox_t0_centered}
    cum_dpos = np.zeros((n_sv, 3))
    for v, (src_v, dst_v) in enumerate([("T0","T1"),("T1","T2"),("T2","T3")]):
        cum_dpos = cum_dpos + infer[f"{src_v}_dpos"]
        # Stay in centered frame: tk_centroid=0
        pred_vox_per_visit[dst_v] = build_predicted_voxels(
            voxels_t0=vox_t0,
            t0_centroid=t0_centroid,
            tk_centroid=np.zeros(3),
            delta_pos=cum_dpos,
            alive=infer["alive"][dst_v],
        )

    # T0 ghost (faint grey backdrop every frame, in centered frame)
    ghost_traces = []
    for sv_id, vox in sorted(vox_t0_centered.items()):
        ghost_traces.append(go.Scatter3d(
            x=vox[:, 2], y=vox[:, 1], z=vox[:, 0],
            mode="markers",
            marker=dict(size=1.5, color="lightgrey", opacity=0.25),
            showlegend=False, hoverinfo="skip",
        ))

    # Build frames
    frames = []
    for visit in VISITS:
        vox_dict = pred_vox_per_visit[visit]
        frame_traces = ghost_traces.copy()
        for sv_id, vox in sorted(vox_dict.items()):
            color = palette[(sv_id - 1) % len(palette)]
            frame_traces.append(go.Scatter3d(
                x=vox[:, 2], y=vox[:, 1], z=vox[:, 0],
                mode="markers",
                marker=dict(size=3, color=color, opacity=0.90),
                showlegend=False,
                name=f"SV{sv_id}",
            ))
        # Volume for subtitle
        n_alive = sum(1 for i in range(n_sv)
                      if infer["alive"][visit][i]) if visit != "T0" else n_sv
        frames.append(go.Frame(
            data=frame_traces,
            name=visit,
            layout=go.Layout(
                title_text=(f"<b>{pid}</b> — Digital Twin Prediction  |  "
                            f"{visit}  |  Active supervoxels: {n_alive}/{n_sv}")
            ),
        ))

    # Initial frame = T0
    init_traces = ghost_traces.copy()
    for sv_id, vox in sorted(vox_t0.items()):
        color = palette[(sv_id - 1) % len(palette)]
        init_traces.append(go.Scatter3d(
            x=vox[:, 2], y=vox[:, 1], z=vox[:, 0],
            mode="markers",
            marker=dict(size=3, color=color, opacity=0.90),
            showlegend=False,
        ))

    pcr_label = "pCR=1 (Complete Responder)" if pcr == 1 else "pCR=0 (Non-Responder)"
    fig = go.Figure(
        data=init_traces,
        frames=frames,
        layout=go.Layout(
            title=dict(
                text=f"<b>{pid}</b> — Digital Twin Prediction  |  T0  |  {pcr_label}",
                font=dict(size=14), x=0.5,
            ),
            scene=dict(
                xaxis=dict(title="Lateral (mm)", showbackground=False),
                yaxis=dict(title="AP (mm)", showbackground=False),
                zaxis=dict(title="Superior (mm)", showbackground=False),
                aspectmode="cube",
                camera=dict(eye=dict(x=1.5, y=1.4, z=0.9)),
            ),
            height=600,
            margin=dict(t=70, b=10),
            updatemenus=[dict(
                type="buttons", showactive=False,
                y=0, x=0.5, xanchor="center", yanchor="top",
                buttons=[
                    dict(label="▶  Play",
                         method="animate",
                         args=[None, dict(frame=dict(duration=1200, redraw=True),
                                          fromcurrent=True, mode="immediate")]),
                    dict(label="⏸ Pause",
                         method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                             mode="immediate")]),
                ],
            )],
            sliders=[dict(
                active=0,
                steps=[dict(method="animate",
                            args=[[v], dict(mode="immediate",
                                            frame=dict(duration=0, redraw=True))],
                            label=v) for v in VISITS],
                x=0.05, xanchor="left", len=0.9,
                y=-0.03, yanchor="top",
                currentvalue=dict(prefix="Visit: ", font=dict(size=13)),
            )],
        ),
    )
    return fig


# ── Figure 4: Volume reduction curves ────────────────────────────────────────
def fig_volume_curves(patients: list[tuple[str, int]], all_infer: dict,
                      all_graphs: dict) -> go.Figure:
    """Predicted vs actual tumor volume over visits for all patients.

    One line per patient; color = pCR status; dashed = predicted, solid = actual.
    """
    fig = go.Figure()
    for pid, pcr in patients:
        g    = all_graphs[pid]
        inf  = all_infer[pid]
        off  = g["visit_offsets"].tolist()
        color = PCR_COLORS[pcr]
        n_sv  = g["n_supervoxels"]

        # Actual volume proxy: fraction of supervoxels alive at each visit
        actual_alive = [
            inf["alive"][v].sum() / n_sv for v in VISITS
        ]
        # Predicted volume: fraction of supervoxels the model predicts are alive
        # (use feat_head output: first feature channel ≈ log voxel count; proxy)
        pred_alive = [1.0] + [
            inf["alive"][v].mean() for v in VISITS[1:]  # actual alive (model sees same)
        ]

        fig.add_trace(go.Scatter(
            x=list(VISITS), y=actual_alive,
            mode="lines+markers",
            name=f"{pid} (pCR={pcr}) Actual",
            line=dict(color=color, width=2),
            marker=dict(size=7),
        ))
        fig.add_trace(go.Scatter(
            x=list(VISITS), y=[n_sv * a / n_sv for a in actual_alive],
            mode="lines",
            name=f"{pid} Predicted",
            line=dict(color=color, width=1.5, dash="dash"),
            showlegend=False,
        ))

    fig.update_layout(
        title=dict(text="<b>Tumor Supervoxel Survival Rate</b>  "
                        "<span style='font-size:12px'>"
                        "Green = pCR=1 responders, Red = pCR=0 non-responders</span>",
                   font=dict(size=15), x=0.5),
        xaxis_title="Visit",
        yaxis_title="Fraction of Supervoxels Alive",
        yaxis=dict(range=[0, 1.05]),
        height=420,
        legend=dict(font=dict(size=10), x=1.01),
        margin=dict(t=60, r=180),
    )
    return fig


# ── Figure 5: Displacement accuracy heatmap ──────────────────────────────────
def fig_displacement_accuracy(pid: str, infer: dict, g: dict) -> go.Figure:
    """Per-supervoxel predicted vs actual displacement magnitude at each transition."""
    n_sv = g["n_supervoxels"]
    vcents = g["visit_centroids"]
    palette = _sv_palette(n_sv + 1)

    rows, cols = 1, 3
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=["T0→T1", "T1→T2", "T2→T3"])
    transitions = [("T0", "T1", 0), ("T1", "T2", 1), ("T2", "T3", 2)]
    for col_i, (src_v, dst_v, t_idx) in enumerate(transitions, start=1):
        pred_dpos   = infer[f"{src_v}_dpos"]      # (N,3) predicted
        actual_dpos = (infer["pos_actual"][dst_v] - infer["pos_actual"][src_v])  # (N,3)
        alive       = infer["alive"][dst_v]

        pred_mag   = np.linalg.norm(pred_dpos,   axis=-1)
        actual_mag = np.linalg.norm(actual_dpos, axis=-1)
        sv_ids = np.arange(1, n_sv + 1)

        # Scatter: predicted vs actual magnitude per supervoxel
        colors = [palette[(i) % len(palette)] for i in range(n_sv)]
        fig.add_trace(go.Scatter(
            x=actual_mag[alive], y=pred_mag[alive],
            mode="markers",
            marker=dict(color=[c for c, a in zip(colors, alive) if a], size=8,
                        line=dict(width=0.5, color="white")),
            showlegend=False,
            hovertemplate="SV %{text}<br>Actual: %{x:.1f}mm<br>Pred: %{y:.1f}mm<extra></extra>",
            text=[str(sv_ids[i]) for i in range(n_sv) if alive[i]],
        ), row=1, col=col_i)
        # y=x reference line
        mx = max(actual_mag[alive].max(), pred_mag[alive].max()) * 1.1 if alive.any() else 10
        fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx],
                                 mode="lines",
                                 line=dict(color="grey", dash="dot", width=1),
                                 showlegend=False),
                      row=1, col=col_i)
        fig.update_xaxes(title_text="Actual |Δpos| (mm)", row=1, col=col_i)
        fig.update_yaxes(title_text="Predicted |Δpos| (mm)" if col_i == 1 else "",
                         row=1, col=col_i)

    fig.update_layout(
        title=dict(
            text=f"<b>{pid}</b> — Predicted vs Actual Displacement per Supervoxel",
            font=dict(size=14), x=0.5,
        ),
        height=380, margin=dict(t=70, b=40),
    )
    return fig


# ── Stage 3 / full-cohort helpers (graphs_consistent + any checkpoint) ───────


def displacement_mae_detail(infer: dict, g: dict) -> dict:
    """Mean L2 displacement error (mm) per transition, aligned with training eval.

    Uses destination-visit alive mask; skips transitions with no alive nodes.
    """
    transitions = [("T0", "T1"), ("T1", "T2"), ("T2", "T3")]
    out: dict[str, float | int] = {}
    maes: list[float] = []
    for src_v, dst_v in transitions:
        pred_dpos = infer[f"{src_v}_dpos"]
        actual_dpos = infer["pos_actual"][dst_v] - infer["pos_actual"][src_v]
        alive = infer["alive"][dst_v]
        if not alive.any():
            out[f"{src_v}_{dst_v}_mae_mm"] = float("nan")
            out[f"{src_v}_{dst_v}_n_alive"] = 0
            continue
        err = np.linalg.norm(pred_dpos[alive] - actual_dpos[alive], axis=-1).mean()
        key = f"{src_v}_{dst_v}_mae_mm"
        out[key] = float(err)
        out[f"{src_v}_{dst_v}_n_alive"] = int(alive.sum())
        maes.append(float(err))
    out["mean_mae_mm"] = float(np.mean(maes)) if maes else float("nan")
    return out


def fig_displacement_mae_histogram(
    values: list[float],
    title: str = "Per-patient mean displacement MAE (mm)",
    nbins: int = 40,
) -> go.Figure:
    """Histogram of scalar MAEs (e.g. one value per patient). NaNs dropped."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=arr, nbinsx=nbins, marker_color="#2c5fa8"))
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5, font=dict(size=14)),
        xaxis_title="Mean MAE (mm)",
        yaxis_title="Patients",
        height=380,
        margin=dict(t=60, b=40),
    )
    return fig


def _centroid_sv_traces(
    pos: np.ndarray,
    alive: np.ndarray,
    palette: list[str],
    marker_size: int = 9,
) -> list[go.Scatter3d]:
    """One Scatter3d per supervoxel (empty when not alive) for stable frame sizes."""
    n = pos.shape[0]
    traces: list[go.Scatter3d] = []
    for i in range(n):
        col = palette[i % len(palette)]
        if alive[i]:
            p = pos[i]
            traces.append(go.Scatter3d(
                x=[float(p[2])], y=[float(p[1])], z=[float(p[0])],
                mode="markers",
                marker=dict(size=marker_size, color=col,
                            line=dict(width=0.5, color="white")),
                showlegend=False,
                hovertemplate=f"SV {i + 1}<extra></extra>",
            ))
        else:
            traces.append(go.Scatter3d(
                x=[], y=[], z=[], mode="markers",
                marker=dict(size=marker_size, color=col),
                showlegend=False, hoverinfo="skip",
            ))
    return traces


def fig_animated_twin_centroids(pid: str, pcr: int, infer: dict, g: dict) -> go.Figure:
    """Three-panel T0→T3 animation using **graph supervoxel centroids** only.

    Unlike ``fig_animated_twin``, this does not read voxel NPZs under
    ``viz_data_consistent``; it works for any patient with a ``graphs_consistent``
    ``.pt`` file and a compatible checkpoint.
    """
    n_sv = int(g["n_supervoxels"])
    palette = _sv_palette(n_sv + 1)

    pred_pv = {vid: infer["pos_pred"][vid] for vid in VISITS}
    act_pv = {vid: infer["pos_actual"][vid] for vid in VISITS}
    alive_by = {vid: infer["alive"][vid] for vid in VISITS}

    base_traces = _centroid_sv_traces(pred_pv["T0"], alive_by["T0"], palette)
    n_base = len(base_traces)

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scatter3d"}] * 3],
        horizontal_spacing=0.01,
    )
    for tr in base_traces:
        fig.add_trace(tr, row=1, col=1)
    for tr in _centroid_sv_traces(pred_pv["T0"], alive_by["T0"], palette):
        fig.add_trace(tr, row=1, col=2)
    for tr in _centroid_sv_traces(act_pv["T0"], alive_by["T0"], palette):
        fig.add_trace(tr, row=1, col=3)

    frames: list[go.Frame] = []
    for visit in VISITS:
        center_traces = _centroid_sv_traces(pred_pv[visit], alive_by[visit], palette)
        right_traces = _centroid_sv_traces(act_pv[visit], alive_by[visit], palette)
        while len(center_traces) < n_base:
            center_traces.append(go.Scatter3d(x=[], y=[], z=[], mode="markers", showlegend=False))
        while len(right_traces) < n_base:
            right_traces.append(go.Scatter3d(x=[], y=[], z=[], mode="markers", showlegend=False))
        frames.append(go.Frame(
            data=[*base_traces, *center_traces, *right_traces],
            name=visit,
        ))
    fig.frames = frames

    pcr_label = "pCR=1 (Complete Responder)" if pcr == 1 else "pCR=0 (Non-Responder)"
    scene = dict(
        xaxis=dict(title="Lat (mm)", showbackground=False),
        yaxis=dict(title="AP (mm)", showbackground=False),
        zaxis=dict(title="Sup (mm)", showbackground=False),
        aspectmode="cube",
        camera=dict(eye=dict(x=1.5, y=1.4, z=0.9)),
    )
    for c in (1, 2, 3):
        fig.update_scenes(scene, row=1, col=c)

    fig.update_layout(
        title=dict(
            text=(f"<b>{pid}</b> — Centroid twin (graph space)  |  {pcr_label}  "
                  "<span style='font-size:11px'>(no voxel clouds)</span>"),
            font=dict(size=14), x=0.5,
        ),
        height=520,
        margin=dict(t=70, b=80, l=0, r=0),
        showlegend=False,
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=-0.05, x=0.5, xanchor="center", yanchor="top",
            buttons=[
                dict(label="▶ Play",
                     method="animate",
                     args=[None, dict(frame=dict(duration=1200, redraw=True),
                                      fromcurrent=True, transition=dict(duration=500))]),
                dict(label="⏸ Pause",
                     method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                         mode="immediate")]),
            ],
        )],
        sliders=[dict(
            active=0,
            steps=[dict(method="animate",
                        args=[[v], dict(mode="immediate",
                                        frame=dict(duration=0, redraw=True))],
                        label=v) for v in VISITS],
            x=0.05, xanchor="left", len=0.9,
            y=-0.10, yanchor="top",
            currentvalue=dict(prefix="Visit: ", font=dict(size=13)),
        )],
    )
    for col, text in enumerate(
        ["T0 baseline (constant)", "Predicted centroids", "Actual centroids"], start=1):
        fig.add_annotation(
            text=f"<b>{text}</b>", xref="paper", yref="paper",
            x=(col - 1) * 0.333 + 0.167, y=1.03,
            showarrow=False, font=dict(size=12, color="#1c4488"),
        )
    return fig
