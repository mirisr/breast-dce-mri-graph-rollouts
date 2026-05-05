#!/usr/bin/env python
"""Inject clinical and per-visit context tensors into every graph payload.

Schema-deterministic: every graph in ``--graphs-root`` is updated to carry a
``clinical`` tensor of shape ``(D,)`` (last column is ``clinical_missing``) and
a ``visit_context`` tensor of shape ``(V, ctx_dim)``. The script never skips a
graph; missing upstream data is recorded via the missing flag rather than via
absence of a key. Re-running this script is safe and idempotent.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lsgc.clinical import clinical_vector_for_patient, compute_visit_context, load_clinical_table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphs-root", type=Path, required=True)
    ap.add_argument("--clinical-xlsx", type=Path, required=True)
    args = ap.parse_args()

    clinical = load_clinical_table(args.clinical_xlsx)
    paths = sorted(
        p for p in args.graphs_root.glob("*.pt")
        if ".match" not in p.name
    )
    n_total = len(paths)
    n_missing = 0
    for i, p in enumerate(paths, 1):
        payload = torch.load(p, map_location="cpu", weights_only=False)
        pid = str(payload.get("patient_id", p.stem))
        vec = clinical_vector_for_patient(clinical, pid)
        if float(vec[-1].item()) > 0.5:
            n_missing += 1
        payload["clinical"] = vec
        payload["visit_context"] = compute_visit_context(payload)
        torch.save(payload, p)
        if i % 100 == 0 or i == n_total:
            print(f"[{i}/{n_total}] {p.name}", flush=True)
    print(
        f"[done] graphs={n_total} clinical_present={n_total - n_missing} "
        f"clinical_missing={n_missing} clinical_dim={len(clinical.columns)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
