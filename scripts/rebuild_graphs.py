#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _build_one(task: tuple[str, str, str, str, str, str, str]) -> int:
    pid, script, derived_root, out_root, feature_set, k_spatial, k_temporal = task
    cmd = [
        "python",
        script,
        "--derived-root",
        derived_root,
        "--patient-id",
        pid,
        "--out-root",
        out_root,
        "--k-spatial",
        k_spatial,
        "--k-temporal",
        k_temporal,
        "--feature-set",
        feature_set,
    ]
    return subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--derived-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--feature-set", choices=("v1", "v2", "v3"), default="v3")
    ap.add_argument("--k-spatial", type=int, default=8)
    ap.add_argument("--k-temporal", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    script = Path("datasets/ispy2/scripts/build_patient_graph.py")
    pids = sorted([p.name for p in args.derived_root.iterdir() if p.is_dir()])
    ok = 0
    fail = 0
    tasks = [
        (
            pid,
            str(script),
            str(args.derived_root),
            str(args.out_root),
            args.feature_set,
            str(args.k_spatial),
            str(args.k_temporal),
        )
        for pid in pids
    ]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(_build_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            rc = fut.result()
            if rc == 0:
                ok += 1
            else:
                fail += 1
            if i % 100 == 0 or i == len(tasks):
                print(f"[{i}/{len(tasks)}] ok={ok} fail={fail}", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
