#!/usr/bin/env python
"""Append a one-hot tumor-habitat label to every graph in a directory.

Schema-deterministic and idempotent:

* Every graph in ``--graphs-root`` is updated so that its ``x`` tensor has a
  trailing block of ``k`` columns named ``habitat_0..habitat_{k-1}`` and a
  per-node ``habitat`` index tensor.
* If the target columns are already present (re-run), the habitat block and
  ``habitat`` indices are recomputed in place rather than appended again.
* Patients lacking DCE phase columns receive an all-zero one-hot block and a
  zero index tensor (we log how many such patients we hit instead of silently
  skipping them).

This combines with ``inject_clinical_context.py`` to give every graph the same
33-column ``x`` schema regardless of upstream data availability.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def _phase_indices(feature_names: list[str]) -> list[int]:
    targets = [f"phase{i}_mean_enh" for i in range(8)]
    return [feature_names.index(c) for c in targets if c in feature_names]


def _assign_habitat(curves: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    if curves.shape[1] < centroids.shape[1]:
        curves = np.concatenate(
            [
                curves,
                np.zeros(
                    (curves.shape[0], centroids.shape[1] - curves.shape[1]),
                    dtype=np.float32,
                ),
            ],
            axis=1,
        )
    elif curves.shape[1] > centroids.shape[1]:
        curves = curves[:, : centroids.shape[1]]
    d = ((curves[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1).astype(np.int64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphs-root", type=Path, required=True)
    ap.add_argument("--codebook", type=Path, required=True)
    args = ap.parse_args()

    centroids = np.load(args.codebook)["centroids"].astype(np.float32)
    k = centroids.shape[0]
    habitat_cols = [f"habitat_{j}" for j in range(k)]

    paths = sorted(
        p for p in args.graphs_root.glob("*.pt")
        if ".match" not in p.name
    )
    n_total = len(paths)
    n_no_phase = 0
    for i, p in enumerate(paths, 1):
        payload = torch.load(p, map_location="cpu", weights_only=False)
        names = list(payload.get("feature_names", []))
        x = payload["x"]
        already = names[-k:] == habitat_cols if len(names) >= k else False
        if already:
            base_x = x[:, : x.shape[1] - k]
            base_names = names[: -k]
        else:
            base_x = x
            base_names = names

        idx = _phase_indices(base_names)
        n_nodes = base_x.shape[0]
        if not idx:
            n_no_phase += 1
            habitat = np.zeros((n_nodes,), dtype=np.int64)
        else:
            curves = base_x[:, idx].float().numpy()
            habitat = _assign_habitat(curves, centroids)

        onehot = np.eye(k, dtype=np.float32)[habitat]
        payload["x"] = torch.cat(
            [base_x, torch.from_numpy(onehot)],
            dim=1,
        )
        payload["feature_names"] = list(base_names) + habitat_cols
        payload["habitat"] = torch.from_numpy(habitat)
        torch.save(payload, p)
        if i % 100 == 0 or i == n_total:
            print(f"[{i}/{n_total}] {p.name}", flush=True)
    print(
        f"[done] graphs={n_total} no_phase_columns={n_no_phase} habitats={k}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
