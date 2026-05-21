#!/usr/bin/env python3
"""Create T0-T3 aliases and smoke transports for four-visit NACT patients."""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np


VISIT_MAP = {"V1": "T0", "V2": "T1", "V3": "T2", "V4": "T3"}


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        rel = os.path.relpath(src, dst.parent)
        dst.symlink_to(rel, target_is_directory=True)
    except OSError:
        shutil.copytree(src, dst)


def make_patient_aliases(pid: str, source_root: Path, alias_root: Path, registered_root: Path) -> bool:
    src_patient = source_root / pid
    if not src_patient.exists():
        return False
    for src_visit, dst_visit in VISIT_MAP.items():
        src = src_patient / src_visit
        if not src.exists():
            return False
        link_or_copy(src, alias_root / pid / dst_visit)

    t0_labels = np.load(alias_root / pid / "T0" / "supervoxel_labels.npz")["labels"]
    registered_root.joinpath(pid).mkdir(parents=True, exist_ok=True)
    for dst_visit in ("T1", "T2", "T3"):
        target_labels = np.load(alias_root / pid / dst_visit / "supervoxel_labels.npz")["labels"]
        if target_labels.shape != t0_labels.shape:
            raise ValueError(f"{pid} {dst_visit}: shape mismatch {t0_labels.shape} vs {target_labels.shape}")
        transported = np.where(target_labels > 0, t0_labels, 0).astype(np.int32)
        np.savez_compressed(registered_root / pid / f"{dst_visit}_t0sv_in_{dst_visit}_space.npz", labels=transported)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("datasets/breast_mri_nact_pilot/derived_v2"))
    parser.add_argument("--alias-root", type=Path, default=Path("datasets/breast_mri_nact_pilot/derived_4visit_smoke"))
    parser.add_argument("--registered-root", type=Path, default=Path("datasets/breast_mri_nact_pilot/registered_identity_4visit_smoke"))
    parser.add_argument("--patient-list", type=Path, default=Path("reports/breast_mri_nact_external/audit/smoke_patients.txt"))
    parser.add_argument("--out-patient-list", type=Path, default=Path("reports/breast_mri_nact_external/patients_4visit_smoke.txt"))
    args = parser.parse_args()

    patients = [ln.strip() for ln in args.patient_list.read_text().splitlines() if ln.strip()]
    ok: list[str] = []
    failed: list[str] = []
    for pid in patients:
        try:
            if make_patient_aliases(pid, args.source_root, args.alias_root, args.registered_root):
                ok.append(pid)
            else:
                failed.append(pid)
        except Exception as exc:
            failed.append(f"{pid}: {type(exc).__name__}: {exc}")
    args.out_patient_list.parent.mkdir(parents=True, exist_ok=True)
    args.out_patient_list.write_text("\n".join(ok) + ("\n" if ok else ""), encoding="utf-8")
    print(f"[nact-alias] ok={len(ok)} failed={len(failed)} alias_root={args.alias_root}")
    if failed:
        print("\n".join(failed))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
