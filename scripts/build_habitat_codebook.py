#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--derived-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260427)
    args = ap.parse_args()

    phase_cols = [f"phase{i}_mean_enh" for i in range(8)]
    curves: list[np.ndarray] = []
    for pq in sorted(args.derived_root.glob("*/T*/supervoxels.parquet")):
        df = pd.read_parquet(pq)
        have = [c for c in phase_cols if c in df.columns]
        if len(have) < 2:
            continue
        x = df[have].to_numpy(dtype=np.float32)
        if len(have) < len(phase_cols):
            pad = np.zeros((x.shape[0], len(phase_cols) - len(have)), dtype=np.float32)
            x = np.concatenate([x, pad], axis=1)
        curves.append(x)
    X = np.concatenate(curves, axis=0)
    km = KMeans(n_clusters=args.k, random_state=args.seed, n_init=10)
    km.fit(X)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, centroids=km.cluster_centers_.astype(np.float32))
    print({"n_samples": int(X.shape[0]), "k": args.k, "out": str(args.out)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
