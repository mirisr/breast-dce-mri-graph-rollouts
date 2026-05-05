"""LSGC-Twin: Stage-2 trajectory simulator.

Given the observed baseline graph $G_0$ only, roll the Stage-1 forecaster
forward to produce a predicted sequence $(\\hat G_1, \\hat G_2, \\hat G_3)$
plus per-node alive probabilities. Two ingredients on top of Stage 1 (per
``proposals/dt_proposal.tex`` Sec. 5.2):

1. **Stochastic global latent.** A graph-level code
   ``z ~ q(z | G_0) = N(mu(G_0), sigma(G_0))`` is sampled once per
   trajectory and *FiLM*-injected into every LSGC layer:
   ``h' = gamma(z) * h + beta(z)``. At inference we sample ``K`` different
   ``z``'s to draw plausible futures; at training we use the
   reparameterization trick and regularize with
   ``KL(q(z | G_0) || N(0, I))``.
2. **Rollout wrapper.** ``LSGCTwin.rollout(G0)`` iteratively forwards the
   current spatio-temporal graph through the forecaster, applies the
   predicted per-node deltas to the last-observed visit to synthesize
   the next visit, rebuilds edges, and repeats. The wrapper returns the
   predicted visits as plain ``x, pos, alive_prob`` tensors ready for
   downstream FTV / pCR computation.

For training in the teacher-forced regime, we do *not* call ``rollout``;
we only need ``z`` and the FiLM-conditioned backbone. Teacher-forcing
reuses the observed spatio-temporal graph (identical to Stage 1) plus
``z`` injection, which makes the Stage-2 training loop a small delta on
top of ``train_forecaster.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from .graph_builder import build_spatiotemporal_graph
from .lsgc_layer import LSGCConv


# --------------------------------------------------------------------------- #
# Stochastic encoder                                                          #
# --------------------------------------------------------------------------- #


class GraphLatentEncoder(nn.Module):
    """Amortized posterior ``q(z | G_0) = N(mu, sigma^2)``.

    Input is the baseline visit's nodes only (``x0``, ``pos0``). We reduce
    them to a single graph-level vector via a small MLP + mean pool, then
    produce ``mu`` and ``log_sigma``. Keeping it on ``G_0`` alone matches
    the proposal: z is the patient-level "what trajectory is this patient
    on" code, not a per-visit code.
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int = 64,
        latent_dim: int = 8,
        clinical_dim: int = 0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.clinical_dim = clinical_dim
        self.embed = nn.Sequential(
            nn.Linear(in_channels, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        enc_in = hidden + clinical_dim
        self.mu_head = nn.Linear(enc_in, latent_dim)
        self.log_sigma_head = nn.Linear(enc_in, latent_dim)

    def forward(self, x0: Tensor, clinical: Tensor | None = None) -> tuple[Tensor, Tensor]:
        h = self.embed(x0).mean(dim=0, keepdim=True)  # (1, hidden)
        if self.clinical_dim > 0:
            if clinical is None:
                clinical = torch.zeros((1, self.clinical_dim), dtype=h.dtype, device=h.device)
            if clinical.dim() == 1:
                clinical = clinical.unsqueeze(0)
            h = torch.cat([h, clinical], dim=1)
        mu = self.mu_head(h).squeeze(0)               # (latent_dim,)
        log_sigma = self.log_sigma_head(h).squeeze(0).clamp(-6.0, 3.0)
        return mu, log_sigma

    @staticmethod
    def reparameterize(mu: Tensor, log_sigma: Tensor,
                       generator: torch.Generator | None = None) -> Tensor:
        eps = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype,
                          generator=generator)
        return mu + torch.exp(log_sigma) * eps

    @staticmethod
    def kl_divergence(mu: Tensor, log_sigma: Tensor) -> Tensor:
        """KL( N(mu, sigma^2) || N(0, I) ), summed over latent dims."""
        return 0.5 * (mu.pow(2) + torch.exp(2.0 * log_sigma) - 1.0 - 2.0 * log_sigma).sum()

    def load_pretrained_backbone(self, ckpt_path, *, strict: bool = False) -> dict:
        """Initialise ``self.embed[0]`` from a Stage-0 encoder checkpoint.

        The source is an ``LSGCNet`` whose ``embed`` is a single
        ``nn.Linear`` (state-dict keys ``embed.weight``, ``embed.bias``).
        Our ``embed`` is a 2-layer Sequential
        (``Linear → SiLU → Linear → SiLU``), so we load only the FIRST
        linear (keys ``0.weight``, ``0.bias``) when the shapes match.

        Returns a status dict with ``loaded_keys``, ``skipped``,
        ``src_in_channels``, and ``src_hidden``.
        """
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        src = sd["state_dict"] if "state_dict" in sd else sd
        target = self.embed.state_dict()
        rename = {"embed.weight": "0.weight", "embed.bias": "0.bias"}
        loaded: dict[str, Tensor] = {}
        skipped: list = []
        for k, v in src.items():
            new_k = rename.get(k)
            if new_k is None:
                continue
            if new_k in target and target[new_k].shape == v.shape:
                loaded[new_k] = v
            else:
                tgt_shape = tuple(target[new_k].shape) if new_k in target else None
                skipped.append((k, tuple(v.shape), f"target={tgt_shape}"))
        target.update(loaded)
        self.embed.load_state_dict(target, strict=strict)
        return {
            "loaded_keys": list(loaded.keys()),
            "skipped": skipped,
            "src_in_channels": int(sd.get("in_channels", -1)),
            "src_hidden": int(sd.get("hidden", -1)),
        }


# --------------------------------------------------------------------------- #
# FiLM wrapper around LSGC backbone                                           #
# --------------------------------------------------------------------------- #


class FiLMLSGCBackbone(nn.Module):
    """Stack of LSGCConv with per-layer FiLM modulation from latent ``z``.

    Each layer gets its own ``(gamma_l, beta_l)`` produced by a tiny MLP from
    ``z``. When ``z`` is the zero vector, the modulation collapses to
    ``gamma=1, beta=0`` by construction (``gamma = 1 + tanh(MLP(z))``), so
    ablations that zero ``z`` reduce to the Stage-1 behavior.
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int = 64,
        num_layers: int = 2,
        latent_dim: int = 8,
        visit_context_dim: int = 0,
        **conv_kwargs,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden = hidden
        self.visit_context_dim = visit_context_dim
        self.embed = nn.Linear(in_channels + visit_context_dim, hidden)
        self.convs = nn.ModuleList(
            [LSGCConv(hidden, hidden, **conv_kwargs) for _ in range(num_layers)]
        )
        # Per-layer FiLM predictors: z -> (gamma, beta) of shape (hidden,) each.
        self.film = nn.ModuleList(
            [nn.Linear(latent_dim, 2 * hidden) for _ in range(num_layers)]
        )
        self.act = nn.SiLU()
        self.reset_film()

    def reset_film(self) -> None:
        """Initialize FiLM to the identity (gamma=1, beta=0)."""
        for lin in self.film:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(
        self,
        x: Tensor, pos: Tensor, t: Tensor, edge_index: Tensor,
        z: Tensor,
        visit_context: Tensor | None = None,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        if self.visit_context_dim > 0:
            if visit_context is None:
                vc = torch.zeros((x.shape[0], self.visit_context_dim), dtype=x.dtype, device=x.device)
            else:
                visit_context = visit_context.to(device=x.device, dtype=x.dtype)
                vc = visit_context[t.long()]
            x = torch.cat([x, vc], dim=1)
        h = self.embed(x)
        for conv, film in zip(self.convs, self.film):
            gb = film(z)                                # (2*hidden,)
            gamma = 1.0 + torch.tanh(gb[: self.hidden])  # (hidden,)
            beta = gb[self.hidden:]                     # (hidden,)
            h_mod = gamma * h + beta                    # broadcast over nodes
            h = h + self.act(conv(h_mod, pos, t, edge_index, edge_attr=edge_attr))
        return h


# --------------------------------------------------------------------------- #
# Twin model                                                                  #
# --------------------------------------------------------------------------- #


@dataclass
class TwinRolloutStep:
    """Synthesized visit in a rollout."""
    x: Tensor
    pos: Tensor                # in centered (per-visit-centroid) frame
    alive_prob: Tensor         # (N,)
    volume_ml_hat: Tensor      # convenience: predicted per-node volume
    source_idx: Tensor         # which original T0 node this voxel came from


class LSGCTwin(nn.Module):
    """Stage-2 trajectory simulator.

    Same backbone + three per-node heads as ``LSGCForecaster``, wrapped
    with a stochastic latent and a rollout helper. In teacher-forced
    training we only use ``forward_teacher_forced``; ``rollout`` is used
    at inference (and in Stage 2 "free rollout" training if/when we add
    scheduled sampling).
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int = 64,
        num_layers: int = 2,
        latent_dim: int = 8,
        feat_out_dim: Optional[int] = None,
        use_pcr_head: bool = False,
        clinical_dim: int = 0,
        visit_context_dim: int = 0,
        # Bio edge parameters — stored so rollout rebuilds edges consistently
        # with how the training graph was constructed.
        edge_attr_mode: str = "legacy",
        adc_idx: Optional[int] = None,
        adc_missing_idx: Optional[int] = None,
        dce_idx_start: int = 0,
        dce_n_phases: int = 0,
        habitat_n_classes: int = 0,
        **conv_kwargs,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.feat_out_dim = feat_out_dim if feat_out_dim is not None else in_channels
        self.latent_dim = latent_dim
        self.clinical_dim = clinical_dim
        self.visit_context_dim = visit_context_dim
        # Bio edge construction parameters (used in rollout to stay consistent with training).
        self._edge_attr_mode = edge_attr_mode
        self._adc_idx = adc_idx
        self._adc_missing_idx = adc_missing_idx
        self._dce_idx_start = dce_idx_start
        self._dce_n_phases = dce_n_phases
        self._habitat_n_classes = habitat_n_classes

        self.encoder = GraphLatentEncoder(in_channels, hidden, latent_dim, clinical_dim=clinical_dim)
        self.backbone = FiLMLSGCBackbone(
            in_channels=in_channels, hidden=hidden, num_layers=num_layers,
            latent_dim=latent_dim, visit_context_dim=visit_context_dim, **conv_kwargs,
        )
        head_in = hidden

        def _head(out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(head_in, hidden), nn.SiLU(), nn.Linear(hidden, out_dim)
            )

        self.pos_head = _head(3)
        self.feat_head = _head(self.feat_out_dim)
        self.alive_head = _head(1)

        # Optional direct pCR prediction head on the patient latent mu (Path B fix).
        # Operates on mu (latent_dim,) -> scalar logit.  Disabled by default so
        # existing checkpoints load without modification.
        self.pcr_head: Optional[nn.Sequential] = (
            nn.Sequential(
                nn.Linear(latent_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1)
            ) if use_pcr_head else None
        )

    # ------------------------------------------------------------------ #
    # Teacher-forced forward: same signature shape as LSGCForecaster     #
    # ------------------------------------------------------------------ #

    def forward_teacher_forced(
        self,
        x: Tensor, pos: Tensor, t: Tensor, edge_index: Tensor,
        x0: Tensor,
        sample: bool = True,
        generator: torch.Generator | None = None,
        z_override: Tensor | None = None,
        clinical: Tensor | None = None,
        visit_context: Tensor | None = None,
        edge_attr: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Run the full observed graph through the FiLM backbone.

        ``x0`` is the node-feature matrix of visit 0 only; used to produce
        the stochastic latent ``z``. When ``sample=False`` (eval mean
        mode), ``z`` is set to the posterior mean without noise.

        ``z_override`` lets the caller supply a pre-sampled latent --
        required for the scheduled-sampling curriculum, where we make
        multiple sequential forward passes on the same patient and want
        a single shared ``z`` per rollout (the patient's ``z`` shouldn't
        flicker between transitions). When provided we still compute
        ``mu, log_sigma, kl`` so the KL regularizer can still run, but
        skip reparameterization.
        """
        mu, log_sigma = self.encoder(x0, clinical=clinical)
        if z_override is not None:
            z = z_override
        elif sample:
            z = GraphLatentEncoder.reparameterize(mu, log_sigma, generator)
        else:
            z = mu
        h = self.backbone(
            x, pos, t, edge_index, z, visit_context=visit_context, edge_attr=edge_attr
        )
        out = {
            "delta_pos": self.pos_head(h),
            "delta_feat": self.feat_head(h),
            "alive_logit": self.alive_head(h).squeeze(-1),
            "hidden": h,
            "mu": mu,
            "log_sigma": log_sigma,
            "z": z,
            "kl": GraphLatentEncoder.kl_divergence(mu, log_sigma),
        }
        if self.pcr_head is not None:
            out["pcr_logit"] = self.pcr_head(mu.unsqueeze(0)).squeeze()
        return out

    # ------------------------------------------------------------------ #
    # Inference-time rollout from G_0                                    #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def rollout(
        self,
        x0: Tensor, pos0_c: Tensor,
        *,
        n_steps: int = 3,
        volume_idx: int = 1,
        feature_mean: Tensor | None = None,
        feature_std: Tensor | None = None,
        alive_threshold: float = 0.5,
        alive_mode: str = "threshold",
        z: Tensor | None = None,
        generator: torch.Generator | None = None,
        k_spatial: int = 8, k_temporal: int = 4,
        temporal_skip_hops: Sequence[int] = (1,),
    ) -> list[TwinRolloutStep]:
        """Predict future visits starting from observed ``G_0`` only.

        Parameters
        ----------
        x0, pos0_c
            Observed baseline features and *centered* positions. If the
            model was trained on normalized features, ``x0`` should
            already be normalized and ``feature_mean`` / ``feature_std``
            provided so the alive/feature heads interpret volumes
            correctly for FTV reconstruction.
        volume_idx
            Index of the volume-in-mL channel in ``x``; used to compute
            ``volume_ml_hat`` at each predicted visit (for downstream
            FTV / pCR computation without another model pass).
        feature_mean, feature_std
            If the model was trained on z-scored features, pass the
            training-set statistics here so we can de-normalize
            ``volume_ml`` in the output. If ``None`` the rollout returns
            features in the same space the model sees.
        alive_mode
            How to decide which nodes survive at each step.

            ``"threshold"`` (default)
                ``node_alive = sigmoid(logit) * acc_survival > alive_threshold``
                Current behaviour; can produce an empty cloud when the
                alive head is over-confident-negative.

            ``"count"``
                Threshold-free: predict the alive *fraction*
                ``rho_hat = mean(sigmoid(logit_k))`` for the current cloud,
                keep the top-``k`` nodes by logit magnitude, where
                ``k = max(1, round(rho_hat * N_k))``.  This guarantees a
                non-empty cloud as long as any node has a positive logit.
                ``alive_threshold`` is ignored in this mode.
        """
        device = x0.device
        # Track per-supervoxel identity back to T0 so we can compute
        # e.g. FTV contributions from individual source voxels.
        n0 = x0.shape[0]
        current_x = x0.clone()
        current_pos = pos0_c.clone()
        current_alive = torch.ones(n0, device=device)
        source_idx = torch.arange(n0, device=device)

        # Seed the latent if the caller didn't provide one.
        if z is None:
            mu, log_sigma = self.encoder(x0)
            z = GraphLatentEncoder.reparameterize(mu, log_sigma, generator)

        # Keep a list of all synthesized visits (index 0 = baseline, observed).
        history_x = [current_x]; history_pos = [current_pos]
        history_t = [torch.zeros(n0, dtype=torch.long, device=device)]
        steps: list[TwinRolloutStep] = []

        # Bio edge mode: track per-visit habitat labels for graph rebuilding.
        # Habitat one-hot is stored in the last ``habitat_n_classes`` columns of x.
        # For predicted visits we carry the habitat forward (tissue class is
        # quasi-static across the 3-visit horizon) to avoid zeroing the cross-habitat
        # edge attribute channel.
        _bio_mode = self._edge_attr_mode == "bio"
        _hab_n = self._habitat_n_classes
        if _bio_mode and _hab_n > 0:
            # graph_builder expects integer class labels (N,), not one-hot (N, K).
            hab0 = x0[:, -_hab_n:].argmax(dim=1).detach().cpu()   # (N0,) long
            history_hab = [hab0]
        else:
            history_hab = None

        def _denorm_volume(x_norm: Tensor) -> Tensor:
            """Predicted per-supervoxel volume (mL), clamped to >= 0.

            Negative volumes are physically meaningless but easy for the
            unconstrained feature head to predict once the rollout drifts
            out of distribution. Clamping at 0 keeps FTV = sum(alive * vol)
            a sane non-negative quantity without hiding drift in other
            channels (those are still visible via EMD, alive AUC, etc.).
            """
            if feature_mean is None or feature_std is None:
                return x_norm[:, volume_idx].clamp_min(0.0)
            v = x_norm[:, volume_idx] * feature_std[volume_idx] + feature_mean[volume_idx]
            return v.clamp_min(0.0)

        for step in range(n_steps):
            # Build spatio-temporal graph over all accumulated visits.
            x_stack = torch.cat(history_x, dim=0)
            pos_stack = torch.cat(history_pos, dim=0)
            t_stack = torch.cat(history_t, dim=0)

            # Rebuild edges. ``build_spatiotemporal_graph`` uses
            # ``torch.arange`` internally which defaults to CPU, so we
            # run it on CPU tensors and then move the edge index back to
            # the model's device.
            visit_feats = [h.detach().cpu() for h in history_x]
            visit_pos = [p.detach().cpu() for p in history_pos]
            if _bio_mode:
                # Build the graph topology with predicted features (so edge
                # selection in mixed+attr mode uses current state), then OVERRIDE
                # the edge_attr block with zeros. The model is trained with
                # ``edge_attr_dropout`` so it has a calibrated no-bio fallback;
                # passing zeros here is strictly safer than feeding noisy
                # predicted-feature bio attrs (which actively mislead the gate).
                g = build_spatiotemporal_graph(
                    visit_feats, visit_pos, k_spatial=k_spatial,
                    k_temporal=k_temporal, temporal_skip_hops=tuple(temporal_skip_hops),
                    add_edge_attr=True, edge_attr_mode="bio",
                    adc_idx=self._adc_idx,
                    adc_missing_idx=self._adc_missing_idx,
                    dce_idx_start=self._dce_idx_start,
                    dce_n_phases=self._dce_n_phases,
                    habitat_labels=history_hab,
                )
                edge_index = g.edge_index.long().to(device)
                if g.edge_attr is not None:
                    edge_attr_step = torch.zeros_like(g.edge_attr).float().to(device)
                else:
                    edge_attr_step = None
            else:
                g = build_spatiotemporal_graph(
                    visit_feats, visit_pos, k_spatial=k_spatial,
                    k_temporal=k_temporal, temporal_skip_hops=tuple(temporal_skip_hops),
                )
                edge_index = g.edge_index.long().to(device)
                edge_attr_step = None

            # FiLM backbone expects z to be (latent_dim,).
            h = self.backbone(x_stack, pos_stack, t_stack, edge_index, z,
                              edge_attr=edge_attr_step)
            delta_pos = self.pos_head(h)
            delta_feat = self.feat_head(h)
            alive_logit = self.alive_head(h).squeeze(-1)

            # Slice the per-node outputs at the *current last visit* -- those
            # are the nodes whose predicted delta builds the *next* visit.
            cur_off = int(sum(len(hh) for hh in history_x[:-1]))
            cur_n = int(history_x[-1].shape[0])
            dp_cur = delta_pos[cur_off: cur_off + cur_n]
            dh_cur = delta_feat[cur_off: cur_off + cur_n]
            al_cur = torch.sigmoid(alive_logit[cur_off: cur_off + cur_n])

            # Synthesize next visit.
            new_x = current_x + dh_cur
            new_pos = current_pos + dp_cur
            new_alive = current_alive * al_cur       # accumulate survival prob
            new_source = source_idx                  # identity carried through

            # Alive gating: which nodes survive to the next step?
            if alive_mode == "count":
                # Counting operator: keep top-k nodes by accumulated survival,
                # where k = max(1, round(rho_hat * N_k)).
                # rho_hat = mean sigmoid(logit) over current cloud (prior to acc).
                rho_hat = float(al_cur.mean().item())
                k = max(1, round(rho_hat * len(new_alive)))
                # Top-k by accumulated survival probability.
                _, top_idx = torch.topk(new_alive, k=min(k, len(new_alive)))
                keep = torch.zeros(len(new_alive), dtype=torch.bool, device=new_alive.device)
                keep[top_idx] = True
            else:
                # Threshold mode: current behaviour.
                keep = new_alive > alive_threshold

            if keep.sum() == 0:
                # Degenerate case: full tumor response predicted.
                # Emit empty steps for ALL remaining iterations so len(steps) == n_steps
                # always. This prevents _rollout_ftv_sample from returning NaN for
                # ftv_pred_T{k} when the cloud dies before the final horizon.
                empty_step = TwinRolloutStep(
                    x=torch.empty(0, new_x.shape[1], device=device),
                    pos=torch.empty(0, 3, device=device),
                    alive_prob=torch.empty(0, device=device),
                    volume_ml_hat=torch.empty(0, device=device),
                    source_idx=torch.empty(0, dtype=torch.long, device=device),
                )
                for _ in range(n_steps - step):
                    steps.append(empty_step)
                break

            kept_x = new_x[keep]
            kept_pos = new_pos[keep]
            kept_alive = new_alive[keep]
            kept_source = new_source[keep]

            steps.append(TwinRolloutStep(
                x=kept_x, pos=kept_pos, alive_prob=kept_alive,
                volume_ml_hat=_denorm_volume(kept_x),
                source_idx=kept_source,
            ))

            # Advance rollout state.
            current_x = kept_x
            current_pos = kept_pos
            current_alive = kept_alive
            source_idx = kept_source
            history_x.append(kept_x)
            history_pos.append(kept_pos)
            history_t.append(torch.full((kept_x.shape[0],), step + 1,
                                        dtype=torch.long, device=device))
            if history_hab is not None:
                # argmax of one-hot columns -> integer class label (N,)
                kept_hab = kept_x[:, -_hab_n:].argmax(dim=1).detach().cpu()
                history_hab.append(kept_hab)

        return steps
