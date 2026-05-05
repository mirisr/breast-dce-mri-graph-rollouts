#!/usr/bin/env python
"""Build graphs_v3_lite by dropping known-noisy / dead feature columns.

Reads every <pid>.pt from --src-root, drops the columns listed in
DROP_FEATURES from x and feature_names, and writes the result to
--dst-root with the SAME filename. All other keys (clinical, habitat,
visit_offsets, pos, t, edge_index, etc.) are copied verbatim.

Usage::

    python scripts/build_graphs_v3_lite.py \
        --src-root datasets/ispy2/graphs_v3_full \
        --dst-root datasets/ispy2/graphs_v3_lite

The destination is created if missing. Existing files are overwritten.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

# Drop: phase0_mean_enh (always 0 — pre-contrast baseline), and the ADC
# block (10% coverage produces near-degenerate columns + noisy 90% imputed
# zeros). Keep DCE phases 1-3, texture, heterogeneity, shape, habitat.
DROP_FEATURES = [
    "phase0_mean_enh",
    "mean_adc",
    "adc_std",
    "adc_skew",
    "adc_kurtosis",
    "adc_missing",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", type=Path, required=True)
    ap.add_argument("--dst-root", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, process only the first N graphs (debug).")
    args = ap.parse_args()

    args.dst_root.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        p for p in args.src_root.glob("*.pt")
        if not p.name.endswith(".match.pt")
        and not p.name.startswith(".")
        and ".match-" not in p.name
    )
    if args.limit:
        paths = paths[: args.limit]

    n_done = n_skipped = 0
    drop_set = set(DROP_FEATURES)
    for i, p in enumerate(paths, 1):
        payload = torch.load(p, map_location="cpu", weights_only=False)
        names = list(payload["feature_names"])
        keep_idx = [j for j, n in enumerate(names) if n not in drop_set]
        if len(keep_idx) == len(names):
            n_skipped += 1
        x = payload["x"][:, keep_idx]
        new_names = [names[j] for j in keep_idx]

        out = dict(payload)
        out["x"] = x.contiguous()
        out["feature_names"] = new_names
        torch.save(out, args.dst_root / p.name)
        n_done += 1
        if i % 100 == 0:
            print(f"  [{i}/{len(paths)}] kept={len(keep_idx)} dropped="
                  f"{len(names)-len(keep_idx)}", flush=True)

    print(f"done. wrote {n_done} graphs (skipped-untouched={n_skipped}) "
          f"to {args.dst_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
