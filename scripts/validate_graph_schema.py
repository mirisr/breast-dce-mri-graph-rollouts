#!/usr/bin/env python
"""Validate that every graph in a directory shares an identical schema.

Run this as a precondition gate before submitting training jobs. It exits
non-zero if any inconsistency is found, so it can be wired into ``sbatch``
scripts via ``set -e``.

Checks performed:

* All ``*.pt`` (excluding match sidecars) load and contain ``x``, ``pos``,
  ``edge_index``, ``feature_names``, ``visit_offsets``.
* All have identical ``x.shape[1]`` and identical ``feature_names`` list.
* All have ``clinical`` (1D) of identical length and ``visit_context`` (2D)
  with identical second dimension.
* If ``--require-habitat`` is set, ``habitat`` per-node indices must exist.
* If ``--expected-dim`` is provided, the common ``x.shape[1]`` must equal it.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch


def _load(p: Path) -> dict:
    return torch.load(p, map_location="cpu", weights_only=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphs-root", type=Path, required=True)
    ap.add_argument("--require-habitat", action="store_true")
    ap.add_argument("--expected-dim", type=int, default=None)
    args = ap.parse_args()

    paths = sorted(
        p for p in args.graphs_root.glob("*.pt")
        if ".match" not in p.name
    )
    if not paths:
        print(f"no graphs under {args.graphs_root}", file=sys.stderr)
        return 1

    issues: list[str] = []
    x_dims: Counter[int] = Counter()
    fname_signatures: Counter[tuple[str, ...]] = Counter()
    clin_dims: Counter[int] = Counter()
    vctx_dims: Counter[int] = Counter()
    missing_keys: Counter[str] = Counter()

    for p in paths:
        try:
            d = _load(p)
        except Exception as exc:  # pragma: no cover - diagnostic
            issues.append(f"{p.name}: load failed ({exc})")
            continue
        for key in ("x", "pos", "edge_index", "feature_names", "visit_offsets"):
            if key not in d:
                missing_keys[key] += 1
                issues.append(f"{p.name}: missing {key}")
        if "x" not in d:
            continue
        x = d["x"]
        x_dims[int(x.shape[1])] += 1
        fname_signatures[tuple(d.get("feature_names", []))] += 1
        clin = d.get("clinical")
        if clin is None:
            missing_keys["clinical"] += 1
            issues.append(f"{p.name}: missing clinical")
        else:
            clin_dims[int(clin.numel())] += 1
        vctx = d.get("visit_context")
        if vctx is None:
            missing_keys["visit_context"] += 1
            issues.append(f"{p.name}: missing visit_context")
        elif vctx.ndim != 2:
            issues.append(f"{p.name}: visit_context wrong rank {vctx.shape}")
        else:
            vctx_dims[int(vctx.shape[1])] += 1
        if args.require_habitat and "habitat" not in d:
            missing_keys["habitat"] += 1
            issues.append(f"{p.name}: missing habitat")

    print(f"graphs={len(paths)}", flush=True)
    print(f"x_dim_counts={dict(x_dims)}", flush=True)
    print(f"clinical_dim_counts={dict(clin_dims)}", flush=True)
    print(f"visit_context_dim_counts={dict(vctx_dims)}", flush=True)
    print(f"feature_name_variants={len(fname_signatures)}", flush=True)
    if missing_keys:
        print(f"missing_key_counts={dict(missing_keys)}", flush=True)

    fail = False
    if len(x_dims) != 1:
        print("FAIL: heterogeneous x feature dimensions", file=sys.stderr)
        fail = True
    if args.expected_dim is not None and list(x_dims.keys()) != [args.expected_dim]:
        print(
            f"FAIL: x.shape[1] != expected {args.expected_dim}",
            file=sys.stderr,
        )
        fail = True
    if len(fname_signatures) != 1:
        print("FAIL: feature_names disagree across graphs", file=sys.stderr)
        fail = True
    if len(clin_dims) > 1:
        print("FAIL: heterogeneous clinical dimensions", file=sys.stderr)
        fail = True
    if len(vctx_dims) > 1:
        print("FAIL: heterogeneous visit_context dimensions", file=sys.stderr)
        fail = True
    if missing_keys:
        print("FAIL: required keys missing", file=sys.stderr)
        fail = True
    if issues and fail:
        print(f"first 10 issues:", file=sys.stderr)
        for s in issues[:10]:
            print(f"  - {s}", file=sys.stderr)
    if fail:
        return 1
    print("OK: schema is consistent", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
