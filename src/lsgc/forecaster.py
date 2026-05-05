"""LSGC-Forecaster: one-step supervoxel-level digital twin.

Same LSGC backbone as :class:`~lsgc.lsgc_layer.LSGCNet`, but the graph-level
mean-pool + classification head is replaced by three *per-node* MLP heads
predicting the supervoxel's state at the next visit:

* ``delta_pos`` (N, 3)   -- predicted centroid drift in millimeters,
                            in the centroid-subtracted frame used by the
                            Stage-1 matching.
* ``delta_feat`` (N, C)  -- predicted feature change (same dims as ``x``).
* ``alive_logit`` (N,)   -- BCE logit for the probability the supervoxel
                            survives to the next visit.

An explicit ``delta_t`` conditioning token is concatenated to the final
hidden state before the heads, per the proposal
(Sec.~5 "Architecture delta"). Setting ``use_delta_t=False`` reproduces
the no-conditioning ablation.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .lsgc_layer import LSGCConv
from .rgcn_layer import RelationalLSGCConv


class LSGCForecaster(nn.Module):
    """LSGC backbone with three per-node forecasting heads.

    Parameters
    ----------
    in_channels, hidden, num_layers, conv_kwargs
        Same semantics as :class:`LSGCNet`. The winning encoder-validation
        configuration was ``num_layers=2`` with temporal skip edges
        ``(1, 2, 3)`` in the graph; that setup is carried forward as the
        default Stage-1 config.
    feat_out_dim
        Dimensionality of ``delta_feat``. Defaults to ``in_channels`` so
        the feature head predicts changes on the same channels the model
        sees at input.
    use_delta_t
        If True, concatenate a learned embedding of the scalar ``delta_t``
        (next visit offset in visit-index units) to every node's final
        hidden state before the heads.
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int = 64,
        num_layers: int = 2,
        feat_out_dim: Optional[int] = None,
        use_delta_t: bool = True,
        conv_type: str = "lsgc",
        **conv_kwargs,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.feat_out_dim = feat_out_dim if feat_out_dim is not None else in_channels
        self.use_delta_t = use_delta_t
        if conv_type not in {"lsgc", "relational"}:
            raise ValueError("conv_type must be 'lsgc' or 'relational'")
        self.conv_type = conv_type

        self.embed = nn.Linear(in_channels, hidden)
        conv_cls = RelationalLSGCConv if conv_type == "relational" else LSGCConv
        self.convs = nn.ModuleList(
            [conv_cls(hidden, hidden, **conv_kwargs) for _ in range(num_layers)]
        )
        self.act = nn.SiLU()

        # delta_t is a scalar in {1, 2, ...} of visit-index units; we embed
        # it through a tiny MLP and concatenate to every node's hidden.
        self.dt_dim = 8 if use_delta_t else 0
        if use_delta_t:
            self.dt_mlp = nn.Sequential(
                nn.Linear(1, self.dt_dim), nn.SiLU(), nn.Linear(self.dt_dim, self.dt_dim)
            )

        head_in = hidden + self.dt_dim

        def _head(out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(head_in, hidden), nn.SiLU(), nn.Linear(hidden, out_dim)
            )

        self.pos_head = _head(3)
        self.feat_head = _head(self.feat_out_dim)
        self.alive_head = _head(1)

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        t: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        edge_type: Tensor | None = None,
        delta_t: float | Tensor = 1.0,
    ) -> dict[str, Tensor]:
        """Run the LSGC backbone and return per-node head outputs.

        ``delta_t`` is broadcast to every node. Scalars and tensors of
        shape ``(N,)`` are both accepted.
        """
        h = self.embed(x)
        for conv in self.convs:
            if self.conv_type == "relational":
                h = h + self.act(
                    conv(h, pos, t, edge_index, edge_attr=edge_attr, edge_type=edge_type)
                )
            else:
                h = h + self.act(conv(h, pos, t, edge_index, edge_attr=edge_attr))

        if self.use_delta_t:
            if not torch.is_tensor(delta_t):
                dt = torch.full((x.size(0), 1), float(delta_t), device=x.device, dtype=x.dtype)
            else:
                dt = delta_t.view(-1, 1).to(x.dtype).to(x.device)
                if dt.shape[0] == 1:
                    dt = dt.expand(x.size(0), 1)
            dt_emb = self.dt_mlp(dt)
            hdt = torch.cat([h, dt_emb], dim=-1)
        else:
            hdt = h

        return {
            "delta_pos": self.pos_head(hdt),
            "delta_feat": self.feat_head(hdt),
            "alive_logit": self.alive_head(hdt).squeeze(-1),
            "hidden": h,
        }
