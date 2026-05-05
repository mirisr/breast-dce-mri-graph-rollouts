"""LSGC-CounterfactualTwin: Stage-3 arm-conditioned trajectory simulator.

Extends ``LSGCTwin`` with a treatment-arm one-hot that is concatenated with
the patient latent ``z`` before FiLM injection.  Everything else (rollout,
heads, counting-operator inference) is inherited unchanged.

Design choices (from ``proposals/dt_proposal.tex`` §5.3):

* **Conditioning point**: the arm embedding is combined with the patient
  latent *after* the graph encoder.  This means the encoder still sees
  only ``G_0`` (no arm leakage at the representation level), and the arm
  signal enters only through FiLM.  This matches the conditional ignorability
  assumption: arm assignment is independent of baseline given ``z``.

* **No separate arm encoder**: arm is categorical (K <= 10 arms), so a
  learned embedding table is sufficient.  We do not need a graph-arm
  interaction at the backbone level for a first Stage-3 pass.

* **Counterfactual inference**: at inference time, given a patient's
  ``z ~ q(z | G_0)``, we call ``rollout`` K times under each arm
  ``a in {a_1, ..., a_K}`` to draw the distribution
  ``p(FTV(T3) | G_0, A = a)``.  The counterfactual treatment effect is the
  difference in the predicted-FTV distributions across arms.

Causal assumptions (must be met for the counterfactual to be valid):
  1. **Positivity**: every arm has positive probability for every patient
     reachable from ``G_0``.  In I-SPY 2 this is approximate (arm
     assignment is not fully randomized within biomarker strata); we
     document this as a limitation.
  2. **Conditional ignorability**: arm assignment is independent of
     potential outcomes given the encoder's latent ``z(G_0)``; i.e.
     ``G_0`` captures the confounders.  This is the key model assumption.
  3. **SUTVA / no-interference**: each patient's outcome under arm ``a``
     does not depend on other patients' arm assignments.  Trivially met
     in a clinical trial.

TODO (Stage 3 sweeps, not yet implemented):
  * Train with arm supervision: add cross-entropy arm-prediction auxiliary
    head so the encoder cannot collapse arm information into ``z``.
  * Tune ``n_arms`` from the treatment-arm audit (see
    ``notebooks/ispy2/session3/s3_00_treatment_arm_audit.ipynb``).
  * Evaluate with ITE metrics (PEHE, policy value) once arm labels are
    attached to each fold.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .twin import (
    LSGCTwin,
    GraphLatentEncoder,
    FiLMLSGCBackbone,
    TwinRolloutStep,
)


class LSGCCounterfactualTwin(LSGCTwin):
    """Stage-3 arm-conditioned LSGC-Twin.

    Adds a small ``arm_embed`` table that maps an integer arm index to a
    vector of size ``arm_dim`` (default 8), which is concatenated with the
    patient latent ``z`` before FiLM modulation.

    Parameters
    ----------
    n_arms : int
        Number of distinct treatment arms (including control if present).
    arm_dim : int
        Dimension of the arm embedding.  Concatenated with ``latent_dim``
        so the effective FiLM input is ``latent_dim + arm_dim``.
    All other parameters are forwarded to ``LSGCTwin``.
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int = 64,
        num_layers: int = 2,
        latent_dim: int = 8,
        feat_out_dim: Optional[int] = None,
        n_arms: int = 2,
        arm_dim: int = 8,
        **conv_kwargs,
    ) -> None:
        # The backbone FiLM predictors expect latent_dim + arm_dim as input.
        super().__init__(
            in_channels=in_channels,
            hidden=hidden,
            num_layers=num_layers,
            latent_dim=latent_dim + arm_dim,   # widened for FiLM
            feat_out_dim=feat_out_dim,
            **conv_kwargs,
        )
        # Overwrite the encoder to keep its output at the original latent_dim
        # (the arm embedding is added on top, not via the encoder).
        self.encoder = GraphLatentEncoder(in_channels, hidden, latent_dim)
        self._patient_latent_dim = latent_dim
        self._arm_dim = arm_dim

        self.arm_embed = nn.Embedding(n_arms, arm_dim)
        nn.init.normal_(self.arm_embed.weight, std=0.01)

    # ------------------------------------------------------------------ #
    # Override the latent sampling helpers to carry the arm embedding     #
    # ------------------------------------------------------------------ #

    def _augment_z(self, z: Tensor, arm: Tensor) -> Tensor:
        """Concatenate arm embedding to patient latent z.

        Parameters
        ----------
        z : (latent_dim,) or (B, latent_dim)
        arm : scalar int tensor (0-indexed arm index)
        """
        arm_vec = self.arm_embed(arm.long().to(z.device))   # (arm_dim,)
        if z.dim() == 1:
            return torch.cat([z, arm_vec.squeeze(0)], dim=0)
        return torch.cat([z, arm_vec.expand(z.shape[0], -1)], dim=-1)

    # ------------------------------------------------------------------ #
    # arm-aware teacher-forced forward                                    #
    # ------------------------------------------------------------------ #

    def forward_teacher_forced(
        self,
        x: Tensor, pos: Tensor, t: Tensor, edge_index: Tensor,
        x0: Tensor,
        arm: Tensor | None = None,
        sample: bool = True,
        generator: torch.Generator | None = None,
        z_override: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Same as ``LSGCTwin.forward_teacher_forced`` + optional arm conditioning.

        Parameters
        ----------
        arm : scalar int tensor, or None (zeros out the arm embedding).
        """
        mu, log_sigma = self.encoder(x0)
        if z_override is not None:
            z_pat = z_override[: self._patient_latent_dim]
        elif sample:
            z_pat = GraphLatentEncoder.reparameterize(mu, log_sigma, generator)
        else:
            z_pat = mu

        if arm is None:
            arm = torch.zeros(1, dtype=torch.long, device=x0.device)
        z = self._augment_z(z_pat, arm)

        h = self.backbone(x, pos, t, edge_index, z)
        return {
            "delta_pos": self.pos_head(h),
            "delta_feat": self.feat_head(h),
            "alive_logit": self.alive_head(h).squeeze(-1),
            "hidden": h,
            "mu": mu,
            "log_sigma": log_sigma,
            "z": z_pat,
            "z_augmented": z,
            "kl": GraphLatentEncoder.kl_divergence(mu, log_sigma),
        }

    @torch.no_grad()
    def rollout_counterfactual(
        self,
        x0: Tensor, pos0_c: Tensor,
        arm: int | Tensor,
        *,
        n_steps: int = 3,
        alive_mode: str = "count",
        alive_threshold: float = 0.5,
        z: Tensor | None = None,
        **rollout_kwargs,
    ) -> list[TwinRolloutStep]:
        """Rollout under a specific treatment arm.

        Draws the arm-conditioned distribution ``p(trajectory | G_0, A=arm)``.
        By default uses ``alive_mode="count"`` for calibration-robust inference.
        """
        if isinstance(arm, int):
            arm = torch.tensor(arm, dtype=torch.long, device=x0.device)

        if z is None:
            mu, log_sigma = self.encoder(x0)
            z_pat = GraphLatentEncoder.reparameterize(mu, log_sigma)
        else:
            z_pat = z

        z_aug = self._augment_z(z_pat, arm)
        return super().rollout(
            x0, pos0_c,
            n_steps=n_steps,
            alive_mode=alive_mode,
            alive_threshold=alive_threshold,
            z=z_aug,
            **rollout_kwargs,
        )
